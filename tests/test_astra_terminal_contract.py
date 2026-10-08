"""Exercise API evaluation through terminal submissions in all six environments."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pytest
from PIL import Image

from gui_agent_captcha.core import Observation, StepResult
from gui_agent_captcha.eval import exploration_depth_astra as runner
from gui_agent_captcha.models.openai_code_execution_computer import (
    OpenAICodeExecutionComputerBackend,
)

VARIANTS = runner.ROTATION_VARIANTS + runner.FIRST_PERSON_VARIANTS + runner.THIRD_PERSON_VARIANTS


class TerminalEnvironment:
    def __init__(self, path: Path, variant: str, success: bool):
        Image.new('RGB', (1280, 720), 'white').save(path)
        self.observation = Observation('Complete the task.', str(path), (1280, 720), (640, 360), {})
        self.variant = variant
        self.success = success
        self.actions = []
        self.held = False
        self.closed = False
        self.audit = {'rotation_state': {'attempt': None}} if variant.startswith('rotation_') else {'terminal_success': None}

    def reset(self, **kwargs):
        return self.observation

    def refresh_observation(self):
        return self.observation

    def get_evaluator_audit(self):
        return self.audit

    def close(self):
        self.closed = True

    def step(self, action):
        self.actions.append(action.kind)
        terminal = action.kind == 'mouse_up' and self.held
        if action.kind == 'mouse_down':
            self.held = True
        if terminal:
            self.held = False
            self.audit = ({'rotation_state': {'attempt': {'success': self.success}}}
                          if self.variant.startswith('rotation_') else {'terminal_success': self.success})
        return StepResult(self.observation, float(self.success) if terminal else 0., terminal, {}, action)


def run_case(tmp_path, monkeypatch, variant, success, profile, codes, budget=12):
    environment = TerminalEnvironment(tmp_path / 'screen.png', variant, success)
    requests = []

    def requester(url, headers, body, timeout_s):
        requests.append(body)
        number = len(requests)
        code = codes[min(number - 1, len(codes) - 1)]
        return {'id': f'resp-{number}', 'status': 'completed', 'model': profile.model,
                'output': [{'type': 'function_call', 'name': 'exec_py', 'call_id': f'call-{number}',
                            'arguments': json.dumps({'code': code})}]}

    monkeypatch.setattr(runner, 'build_benchmark_variant', lambda *args, **kwargs: environment)
    monkeypatch.setattr(runner, 'OpenAICodeExecutionComputerBackend',
                        partial(OpenAICodeExecutionComputerBackend, requester=requester))
    task = runner.EvaluationTask(episode={
        'instruction': 'Complete the task.', 'suite_id': 'test-suite', 'pair_id': 'pair',
        'family': 'rotation' if variant.startswith('rotation_') else 'drag' if variant.startswith('drag_') else 'ten_choice',
        'variant': variant, 'exploration_level': 1, 'episode_id': 'episode', 'case_seed': 1, 'split': 'heldout',
    })
    variants = next(group for group in (runner.ROTATION_VARIANTS, runner.FIRST_PERSON_VARIANTS,
                                        runner.THIRD_PERSON_VARIANTS) if variant in group)
    config = runner.resolve_config({'OPENAI_API_KEY': 'test-key'}, profile=profile)
    record = runner.run_task(task, manifest_path=tmp_path / 'manifest.json', rotation_base_url=None,
                            output_root=tmp_path / 'output', config=config, max_responses=budget,
                            limit_per_variant=150, variants=variants, timeout_s=1, retries=1, profile=profile)
    return environment, requests, record, config, variants


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('success', [False, True])
@pytest.mark.parametrize('profile', [runner.ASTRA_PROFILE, runner.GPT56_PROFILE])
def test_success_failure_and_trailing_actions(tmp_path, monkeypatch, variant, success, profile):
    environment, requests, record, config, variants = run_case(
        tmp_path, monkeypatch, variant, success, profile,
        [('pyautogui.mouseUp()\n' if variant.startswith('rotation_') else '') + 'display(pyautogui.screenshot())',
         'pyautogui.mouseDown()\npyautogui.moveTo(640, 360)\npyautogui.mouseUp()\npyautogui.moveTo(999, 999)'])
    assert record['success'] is success
    assert record['terminal_reason'] == ('first_release_success' if success else 'first_release_failure')
    assert record['infra_error'] is None and record['protocol_error'] is None
    assert record['api_calls'] == len(requests) == 2
    assert environment.actions == (['mouse_up'] if variant.startswith('rotation_') else []) + ['mouse_down', 'move_to', 'mouse_up']
    assert environment.closed
    assert runner._record_matches(record, config, variants=variants, limit_per_variant=150, profile=profile)
    previous = {k: v for k, v in record.items() if k != 'terminal_detection'}
    # Old exocentric records must be reevaluated instead of resuming their faulty score.
    assert runner._record_matches(previous, config, variants=variants, limit_per_variant=150,
                                  profile=profile) is (variant not in runner.THIRD_PERSON_VARIANTS)


@pytest.mark.parametrize('variant', VARIANTS)
def test_no_submission_at_response_limit_is_failure(tmp_path, monkeypatch, variant):
    _, requests, record, _, _ = run_case(tmp_path, monkeypatch, variant, True, runner.ASTRA_PROFILE,
                                       ['pyautogui.moveTo(640, 360)'], budget=2)
    assert not record['success'] and not record['attempt_seen']
    assert record['terminal_reason'] == 'max_responses_without_release'
    assert len(requests) == record['api_calls'] == 2


@pytest.mark.browser
@pytest.mark.parametrize('variant', runner.THIRD_PERSON_VARIANTS)
@pytest.mark.parametrize('success', [False, True])
def test_real_exocentric_environment_scores_submission(tmp_path, monkeypatch, variant, success):
    from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
        episodes_for_variant,
    )
    from latentguiworld.evaluate import SUITE
    from latentguiworld.suite import load_evaluation_manifest

    episode = episodes_for_variant(load_evaluation_manifest(SUITE), variant)[0]
    shared = episode['shared_scene_config']
    if variant == 'drag_third_person':
        x, y = shared['piece_start_screen_xy']
        tx, ty = shared['slot_center_screen_xy'] if success else (10, 10)
        code = f'pyautogui.moveTo({x}, {y})\npyautogui.mouseDown()\npyautogui.moveTo({tx}, {ty})\npyautogui.mouseUp()'
        expected_actions = ['move_to', 'mouse_down', 'move_to', 'mouse_up']
    else:
        target = shared['target_object']['target_index']
        x, y = shared['icon_centers_xy'][target if success else (target + 1) % 10]
        code = f'pyautogui.click({x}, {y})'
        expected_actions = ['move_to', 'mouse_down', 'mouse_up']
    code += '\npyautogui.moveTo(100, 100)'

    def requester(url, headers, body, timeout_s):
        return {'id': 'response', 'status': 'completed', 'model': runner.ASTRA_PROFILE.model,
                'output': [{'type': 'function_call', 'name': 'exec_py', 'call_id': 'call',
                            'arguments': json.dumps({'code': code})}]}

    monkeypatch.setattr(runner, 'OpenAICodeExecutionComputerBackend',
                        partial(OpenAICodeExecutionComputerBackend, requester=requester))
    record = runner.run_task(runner.EvaluationTask(episode), manifest_path=SUITE,
                            rotation_base_url=None, output_root=tmp_path,
                            config=runner.resolve_config({'OPENAI_API_KEY': 'test-key'}),
                            max_responses=12, variants=runner.THIRD_PERSON_VARIANTS,
                            limit_per_variant=150, timeout_s=1, retries=1)
    assert record['infra_error'] is None and record['protocol_error'] is None
    assert record['success'] is success
    assert record['api_calls'] == 1
    traces = [json.loads(line) for line in Path(record['trace_path']).read_text().splitlines()]
    assert [r['action']['kind'] for r in traces if r['row_type'] == 'environment_action'] == expected_actions
