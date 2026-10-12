"""Run with PYTHONPATH pointing to the prepared, task-specific snapshot."""

import asyncio
from types import SimpleNamespace

import pytest

from gui_agent_captcha.benchmarks.exploration_depth.first_person_online_agent_loop import (
    FirstPersonWithThinkProtocolState,
    ResponseFormatError,
    first_person_with_think_trajectory_format_valid,
)
from gui_agent_captcha.core import Observation, StepResult
from gui_agent_captcha.integrations.browser_runtime import BrowserSlotPool
from gui_agent_captcha.integrations.browser_trajectory import (
    BrowserTrajectoryRunnerV4,
    ContextBudgetErrorV4,
    DynamicGenerationBudgetV4,
    GeneratedTurnV4,
)



MOVE = '<think>Inspect.</think>{"action":{"kind":"move_to","x":600,"y":500}}'
CLICK = '<think>Select.</think>{"action":{"kind":"left_click"}}'


def test_budget_overrides_both_sampling_aliases_and_honors_remaining_context():
    budget = DynamicGenerationBudgetV4(max_generation_tokens=384)
    assert budget.sampling_params(
        {"max_tokens": 8192, "max_new_tokens": 8192},
        prompt_token_count=1000,
        accumulated_response_token_count=0,
    ) == {"max_tokens": 384}
    assert budget.remaining(prompt_token_count=8192, accumulated_response_token_count=8100) == 60


def make_runner(tmp_path, monkeypatch, turns):
    calls = []
    actions = []
    resets = []

    class Env:
        def reset(self, **kwargs):
            resets.append(kwargs)
            return self.observation()

        def observation(self):
            return Observation(
                instruction="test",
                screenshot_path=str(tmp_path / f"{len(actions)}.png"),
                size_px=(1280, 720),
                cursor_xy=None,
            )

        def step(self, action):
            actions.append(action)
            done = action.kind == "left_click"
            return StepResult(
                observation=self.observation(),
                reward=99,
                done=done,
                info={"success": done},
                action=action,
            )

        def close(self):
            pass

    sequence = iter(turns)

    async def generate(request, params):
        calls.append(request)
        value = next(sequence)
        if isinstance(value, Exception):
            raise value
        text, count = value
        return GeneratedTurnV4(
            text=text,
            token_ids=[1] * count,
            logprobs=[-0.1] * count,
            extra_fields={"generation_budget": 384},
        )

    monkeypatch.setattr(
        BrowserTrajectoryRunnerV4, "_fingerprint", staticmethod(lambda path: str(path))
    )
    runner = BrowserTrajectoryRunnerV4(
        env_factory=lambda *_: Env(),
        turn_generator=generate,
        artifact_root=tmp_path / "artifacts",
        slot_pool=BrowserSlotPool(tmp_path / "slots", capacity=1),
        task_type="ten_choice_first_person",
        prompt_builder=lambda **_: SimpleNamespace(),
        protocol_state_factory=lambda n: FirstPersonWithThinkProtocolState(max_steps=n),
        trajectory_format_validator=first_person_with_think_trajectory_format_valid,
        response_format_errors=(ResponseFormatError,),
        success_action_kinds=("left_click",),
        max_steps=12,
        max_infra_retries=2,
        non_retryable_errors=(ContextBudgetErrorV4,),
    )
    runner.terminate_on_generation_limit = True
    return runner, calls, actions, resets


def test_length_terminates_whole_trajectory_before_executing_valid_action(tmp_path, monkeypatch):
    runner, calls, actions, resets = make_runner(tmp_path, monkeypatch, [(MOVE, 384), (CLICK, 10)])
    result = asyncio.run(runner.run(seed=1, sampling_params={}))
    assert result.reward_breakdown.reward == 0
    assert result.terminal_source == "generation_length"
    assert len(calls) == len(resets) == len(result.turns) == 1
    assert actions == [] and result.infra_retry_count == 0
    assert result.diagnostics["generation_budgets"] == [384]
    assert result.diagnostics["response_token_counts"] == [384]


def test_context_exhaustion_preserves_generated_turns_without_retry(tmp_path, monkeypatch):
    runner, calls, actions, resets = make_runner(
        tmp_path, monkeypatch, [(MOVE, 10), ContextBudgetErrorV4("full")]
    )
    result = asyncio.run(runner.run(seed=1, sampling_params={}))
    assert result.reward_breakdown.reward == 0
    assert result.terminal_source == "context_budget"
    assert len(result.turns) == len(actions) == len(resets) == 1
    assert len(calls) == 2 and result.infra_retry_count == 0


def test_other_independent_rollouts_still_complete_with_original_success_reward(
    tmp_path, monkeypatch
):
    results = []
    for index in range(5):
        turns = [(MOVE, 384)] if index == 0 else [(MOVE, 10), (CLICK, 10)]
        runner, _, _, _ = make_runner(tmp_path / str(index), monkeypatch, turns)
        results.append(asyncio.run(runner.run(seed=1, sampling_params={})))
    assert [r.reward_breakdown.reward for r in results] == [0, 1.1, 1.1, 1.1, 1.1]


@pytest.mark.parametrize(
    "outcome,expected",
    [("browser_failure", 0.1), ("loop", 0.1), ("max_steps", 0.1), ("parse_error", 0)],
)
def test_existing_reward_categories_unchanged(outcome, expected):
    from gui_agent_captcha.integrations.online_rl import (
        V4TrajectoryOutcome,
        reward_breakdown_for_outcome,
    )

    assert reward_breakdown_for_outcome(V4TrajectoryOutcome(outcome)).reward == expected
