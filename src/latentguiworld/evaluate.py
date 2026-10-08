"""Evaluate a local checkpoint or a vLLM endpoint on the paper's benchmark."""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

from gui_agent_captcha.benchmarks.exploration_depth.contracts import PAPER_VARIANTS

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / 'data/formal_benchmarks/learn_from_move/manifest.json'
VARIANTS = ('ten_choice_third_person', 'ten_choice_first_person', 'rotation_inner',
            'rotation_outer', 'drag_third_person', 'drag_first_person')


def evaluate(backend, *, manifest=SUITE, variants=VARIANTS, output=Path('results/evaluation'), limit=None, gain_analysis=False):
    from gui_agent_captcha.benchmarks.exploration_depth.contracts import episodes_for_variant
    from latentguiworld.suite import load_evaluation_manifest
    from gui_agent_captcha.benchmarks.exploration_depth.service import RotationReplayServer
    from gui_agent_captcha.eval.exploration_depth_local import run_episode
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    suite = load_evaluation_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    gain_payload = {"episodes": [], "movements": []}
    if gain_analysis:
        from gui_agent_captcha.benchmarks.exploration_depth.contracts import runtime_variant
        if not any(runtime_variant(v) == "drag_first_person" for v in variants):
            raise ValueError("--gain-analysis requires egocentric Drag")
    for variant in variants:
        episodes = episodes_for_variant(suite, variant)
        if len(episodes) != 150:
            raise ValueError(f'{variant}: expected 150 episodes')
        episodes = episodes[:limit] if limit else episodes
        context = RotationReplayServer(public_root=manifest.parent) if variant.startswith('rotation_') else nullcontext(None)
        records = []
        with tempfile.TemporaryDirectory(prefix='latentguiworld-eval-') as temporary, context as replay:
            for episode in episodes:
                if hasattr(backend, 'route_key'):
                    backend.route_key = str(episode['episode_id'])
                record = run_episode(backend=backend, episode=episode, manifest_path=manifest,
                    output_root=Path(temporary), checkpoint_step=0,
                    rotation_base_url=replay.base_url if replay else None,
                    max_turns=12, thinking_mode='with-think', strict_first_release=True)
                records.append(record)
                if gain_analysis and record['variant'] == 'drag_first_person':
                    from latentguiworld.pearson import extract_movements
                    gain_payload['movements'].extend(extract_movements(record))
                    gain_payload['episodes'].append({'pair_id': record['pair_id'],
                                                     'success': bool(record['success'])})
                print(f"{variant} {len(records)}/{len(episodes)} {record['terminal_reason']}", flush=True)
        success = sum(bool(r['success']) for r in records)
        errors = sum(r.get('infra_error') is not None for r in records)
        summaries[variant] = {'episodes': len(records), 'successes': success,
            'success_rate': success / len(records), 'infrastructure_errors': errors,
            'terminal_reasons': dict(Counter(r['terminal_reason'] for r in records))}
        (output / f'{variant}.json').write_text(json.dumps(summaries[variant], indent=2) + '\n')
    total = sum(s['episodes'] for s in summaries.values())
    result = {'suite_id': suite['suite_id'], 'variants': summaries,
        'episodes': total, 'full_suite': total == 900,
        'success_rate': sum(s['successes'] for s in summaries.values()) / total,
        'infrastructure_errors': sum(s['infrastructure_errors'] for s in summaries.values()),
        'protocol': {'max_turns': 12, 'context_length': 16384, 'image_history': 3,
                     'max_new_tokens': backend.max_new_tokens, 'greedy': True, 'strict_first_release': True}}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    if gain_analysis:
        (output / 'gain_movements.json').write_text(json.dumps(gain_payload, indent=2, allow_nan=False) + '\n')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--processor', type=Path)
    p.add_argument('--server-url', help='vLLM endpoint, e.g. http://localhost:8000/v1')
    p.add_argument('--served-model', default='latentlearner')
    p.add_argument('--variant', choices=['all', *PAPER_VARIANTS, *VARIANTS], default='all')
    p.add_argument('--manifest', type=Path, default=SUITE)
    p.add_argument('--output', type=Path, default=Path('results/evaluation'))
    p.add_argument('--limit', type=int, help='Evaluate this many episodes per variant')
    p.add_argument('--gain-analysis', action='store_true', help='Save numeric egocentric Drag measurements for Pearson analysis')
    a = p.parse_args(argv)
    if a.limit is not None and not 1 <= a.limit <= 150:
        p.error('--limit must be between 1 and 150')
    options = dict(checkpoint_path=a.model, processor_checkpoint_path=a.processor or a.model,
        max_new_tokens=128, max_length=16384, image_max_pixels=921600,
        image_history_max=3, enable_thinking=True)
    if a.server_url:
        from gui_agent_captcha.models.exploration_depth_vllm import Qwen35ExplorationDepthWithThinkVLLMBackend as Backend
        options.update(server_base_url=a.server_url, served_model_name=a.served_model)
    else:
        from gui_agent_captcha.models.exploration_depth_with_think import Qwen35ExplorationDepthWithThinkLocalBackend as Backend
    result = evaluate(Backend(**options), manifest=a.manifest,
        variants=VARIANTS if a.variant == 'all' else (a.variant,), output=a.output, limit=a.limit, gain_analysis=a.gain_analysis)
    print(json.dumps(result, indent=2))
    return 1 if result['infrastructure_errors'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
