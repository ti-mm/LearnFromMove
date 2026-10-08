"""Episode-weighted gain compensation analysis for egocentric Drag."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from itertools import pairwise
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def correlation(x, y):
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return statistics.correlation(x, y)


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if not math.isfinite(row['alpha']) or not math.isfinite(row['gain']) or row['gain'] <= 0:
            raise ValueError('alpha must be finite and gain must be finite and positive')
        grouped[row['pair_id']].append(row)
    episodes = []
    for pair_id, moves in sorted(grouped.items()):
        if len({r['gain'] for r in moves}) != 1:
            raise ValueError(f'{pair_id}: gain changes within episode')
        episodes.append({'pair_id': pair_id, 'gain': moves[0]['gain'],
                         'alpha': statistics.median(r['alpha'] for r in moves)})
    result = {
        'episodes': len(episodes), 'movements': len(rows),
        'pearson_alpha_gain': correlation([r['alpha'] for r in episodes], [r['gain'] for r in episodes]),
        'pearson_alpha_inverse_gain': correlation(
            [r['alpha'] for r in episodes], [1 / r['gain'] for r in episodes]),
        'episode_rows': episodes,
    }
    for band in ('low', 'high'):
        subset = [r for r in episodes if (r['gain'] < 1 if band == 'low' else r['gain'] > 1)]
        result[band] = {
            'episodes': len(subset),
            'median_alpha': statistics.median(r['alpha'] for r in subset) if subset else None,
            'median_ideal_alpha': statistics.median(1 / r['gain'] for r in subset) if subset else None,
        }
    return result


def extract_movements(record):
    """Read a locally generated episode before its temporary evaluation files close."""
    if record['variant'] not in ('drag_first_person', 'drag_egocentric'):
        raise ValueError('gain analysis requires egocentric Drag')
    if record.get('infra_error'):
        raise ValueError(f"{record['pair_id']}: evaluation infrastructure error")
    trace = read_jsonl(record['trace_path'])
    audit = {r['frame_index']: r['evaluator_only'] for r in read_jsonl(record['audit_path'])}
    rows = []
    for previous, current in pairwise(trace):
        action = current['action']
        if not action or action['kind'] != 'move_to':
            continue
        before, after = audit[previous['frame_index']], audit[current['frame_index']]
        state, dynamics = before['state'], before['hidden_dynamics']
        gain, zoom = dynamics['sensitivity'], dynamics['view_zoom']
        if dynamics['direction_xy'] != [1, 1] or not math.isfinite(gain) or gain <= 0:
            raise ValueError('gain analysis requires positive isotropic control')
        movement = [(b - a) * zoom for a, b in zip(state['piece_world_xy'], after['state']['piece_world_xy'])]
        if math.hypot(*movement) <= 1e-9:
            continue
        error = [(b - a) * zoom for a, b in zip(state['piece_world_xy'], state['target_world_xy'])]
        error_sq = sum(v * v for v in error)
        if error_sq <= 1e-12:
            continue
        # The paper uses a 1280 x 720 viewport and center-relative 0..999 commands.
        command = [(action['x'] - 500) * 1.28, (action['y'] - 500) * .72]
        target = state['target_screen_xy']
        ideal = [500 + e / gain / scale for e, scale in zip(error, (1.28, .72))]
        rows.append({
            'pair_id': record['pair_id'], 'frame': current['frame_index'], 'gain': gain,
            'alpha': sum(u * e for u, e in zip(command, error)) / error_sq,
            'error_px': math.sqrt(error_sq),
            'visible': 0 <= target[0] < 1280 and 0 <= target[1] < 720,
            'ideal_action_in_range': all(0 <= v <= 999 for v in ideal),
            'unclamped': math.hypot(*(m - gain * u for m, u in zip(movement, command))) < 1e-4,
        })
    return rows


def analyze(payload):
    episodes, rows = payload['episodes'], payload['movements']
    ids = [r['pair_id'] for r in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError('episode IDs must be unique within a policy')
    if any(r['pair_id'] not in set(ids) for r in rows):
        raise ValueError('movement references an unknown episode')
    frames = [(r['pair_id'], r['frame']) for r in rows]
    if len(frames) != len(set(frames)):
        raise ValueError('movement frames must be unique within an episode')
    filtered = [r for r in rows if r['visible'] and r['error_px'] >= 20
                and r['ideal_action_in_range'] and r['unclamped']]
    observed = {r['pair_id'] for r in rows}
    return {
        'total_episodes': len(episodes),
        'success_count': sum(bool(r['success']) for r in episodes),
        'episodes_without_computable_movement': sorted(set(ids) - observed),
        'all_observed_held_movements': summarize(rows), 'filtered': summarize(filtered),
    }


def compare(sft, rl):
    result = {'sft': analyze(sft), 'rl': analyze(rl)}
    sft_ids, rl_ids = ({r['pair_id'] for r in p['episodes']} for p in (sft, rl))
    if sft_ids != rl_ids:
        raise ValueError('SFT and RL must evaluate the same episode IDs')
    for cohort in ('all_observed_held_movements', 'filtered'):
        shared = ({r['pair_id'] for r in result['sft'][cohort]['episode_rows']}
                  & {r['pair_id'] for r in result['rl'][cohort]['episode_rows']})
        result.setdefault('shared_episode_ids', {})[cohort] = {}
        for name, payload in (('sft', sft), ('rl', rl)):
            rows = [r for r in payload['movements'] if r['pair_id'] in shared]
            if cohort == 'filtered':
                rows = [r for r in rows if r['visible'] and r['error_px'] >= 20
                        and r['ideal_action_in_range'] and r['unclamped']]
            result['shared_episode_ids'][cohort][name] = summarize(rows)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft', required=True, type=Path, help='SFT gain_movements.json')
    parser.add_argument('--rl', required=True, type=Path, help='RL gain_movements.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    result = compare(json.loads(args.sft.read_text()), json.loads(args.rl.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name in ('sft', 'rl'):
        for cohort in ('all_observed_held_movements', 'filtered'):
            row = result[name][cohort]
            print(name, cohort, json.dumps({k: row[k] for k in
                  ('episodes', 'movements', 'pearson_alpha_gain', 'pearson_alpha_inverse_gain')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
