"""Check episode weighting, cohort filters, and displacement measurements."""
import json
import statistics

import pytest

from latentguiworld.pearson import analyze, compare, extract_movements, main, summarize


def move(pair_id, frame, gain, alpha, **kwargs):
    return dict(pair_id=pair_id, frame=frame, gain=gain, alpha=alpha,
                error_px=30, visible=True, ideal_action_in_range=True,
                unclamped=kwargs.pop('unclamped', True), **kwargs)


def payload(rows, ids=('a', 'b', 'c')):
    return {'episodes': [{'pair_id': p, 'success': p == 'a'} for p in ids], 'movements': rows}


def test_equal_episode_weight_and_failed_episodes():
    rows = [move('a', 1, .5, 1), move('a', 2, .5, 3), move('a', 3, .5, 2),
            move('b', 1, 2, .5), move('c', 1, 4, .25)]
    result = analyze(payload(rows, ('a', 'b', 'c', 'empty')))
    assert result['total_episodes'] == 4
    assert result['success_count'] == 1
    assert result['episodes_without_computable_movement'] == ['empty']
    cohort = result['all_observed_held_movements']
    assert cohort['episodes'] == 3 and cohort['movements'] == 5
    assert cohort['pearson_alpha_inverse_gain'] == pytest.approx(1)
    assert cohort['pearson_alpha_gain'] == pytest.approx(statistics.correlation([2, .5, .25], [.5, 2, 4]))


def test_filtered_medians_are_recomputed_and_shared_ids_match():
    rows = [move('a', 1, .5, 2), move('a', 2, .5, 100, unclamped=False),
            move('b', 1, 2, .5), move('c', 1, 4, .25)]
    sft = payload(rows)
    rl = payload([move('a', 1, .5, 2), move('b', 1, 2, .5)])
    result = compare(sft, rl)
    assert result['sft']['filtered']['episode_rows'][0]['alpha'] == 2
    assert result['sft']['all_observed_held_movements']['episode_rows'][0]['alpha'] == 51
    shared = result['shared_episode_ids']['all_observed_held_movements']
    assert shared['sft']['episodes'] == shared['rl']['episodes'] == 2
    assert shared['sft']['movements'] == 3
    assert shared['rl']['movements'] == 2
    assert shared['rl']['pearson_alpha_inverse_gain'] == pytest.approx(1)
    with pytest.raises(ValueError, match='same episode IDs'):
        compare(sft, payload([], ('a',)))


def test_undefined_correlations_and_invalid_input():
    assert summarize([])['pearson_alpha_gain'] is None
    assert summarize([move('a', 1, 1, 2)])['pearson_alpha_inverse_gain'] is None
    assert summarize([move('a', 1, 1, 2), move('b', 1, 1, 3)])['pearson_alpha_gain'] is None
    with pytest.raises(ValueError, match='positive'):
        summarize([move('a', 1, 0, 2)])
    with pytest.raises(ValueError, match='changes within episode'):
        summarize([move('a', 1, 1, 2), move('a', 2, 2, 2)])


def test_extract_screen_pixel_displacements(tmp_path):
    def state(piece):
        return {'state': {'piece_world_xy': piece, 'target_world_xy': [200, 100],
                          'target_screen_xy': [700, 360]},
                'hidden_dynamics': {'direction_xy': [1, 1], 'sensitivity': .5, 'view_zoom': 2}}
    trace = [{'frame_index': 0, 'action': None},
             {'frame_index': 1, 'action': {'kind': 'move_to', 'x': 812.5, 'y': 500}},
             {'frame_index': 2, 'action': {'kind': 'move_to', 'x': 600, 'y': 500}}]
    audits = [{'frame_index': i, 'evaluator_only': state(piece)}
              for i, piece in enumerate(([100, 100], [200, 100], [200, 100]))]
    for name, content in (('trace.jsonl', trace), ('audit.jsonl', audits)):
        (tmp_path / name).write_text(''.join(json.dumps(r) + '\n' for r in content))
    record = {'variant': 'drag_first_person', 'pair_id': 'a',
              'trace_path': str(tmp_path / 'trace.jsonl'), 'audit_path': str(tmp_path / 'audit.jsonl')}
    rows = extract_movements(record)
    assert len(rows) == 1
    assert rows[0]['alpha'] == 2
    assert rows[0]['error_px'] == 200
    assert rows[0]['visible'] and rows[0]['unclamped'] and rows[0]['ideal_action_in_range']


def test_cli_writes_finite_json(tmp_path):
    p = tmp_path / 'input.json'
    p.write_text(json.dumps(payload([move('a', 1, .5, 2), move('b', 1, 2, .5)])))
    output = tmp_path / 'pearson.json'
    assert main(['--sft', str(p), '--rl', str(p), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['sft']['filtered']['pearson_alpha_inverse_gain'] == pytest.approx(1)


@pytest.mark.browser
def test_gain_measurements_survive_temporary_trace_cleanup(tmp_path):
    from gui_agent_captcha.actions import PrimitiveAction
    from latentguiworld.evaluate import evaluate
    class Backend:
        checkpoint_path = tmp_path / 'probe'
        _call_index = 0
        max_new_tokens = 128
        call_log_dir = None
        last_raw_prediction = None
        last_think_text = None
        def predict_action(self, obs, history, **kwargs):
            self._call_index += 1
            if not history:
                return PrimitiveAction(kind='move_to', x=999, y=500)
            if len(history) == 1:
                return PrimitiveAction(kind='mouse_down')
            return PrimitiveAction(kind='move_to', x=550, y=500)
    result = evaluate(Backend(), variants=('drag_egocentric',), output=tmp_path, limit=1, gain_analysis=True)
    assert result['infrastructure_errors'] == 0
    measurements = json.loads((tmp_path / 'gain_movements.json').read_text())
    assert len(measurements['episodes']) == 1
    assert measurements['movements']
    assert analyze(measurements)['total_episodes'] == 1
    assert not list(tmp_path.rglob('trace.jsonl'))
    assert not list(tmp_path.rglob('audit.jsonl'))
