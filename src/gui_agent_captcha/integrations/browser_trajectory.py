from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from gui_agent_captcha.actions import Action, AtomicAction, PrimitiveAction
from gui_agent_captcha.core import Observation, StepResult
from gui_agent_captcha.integrations.browser_runtime import (
    BrowserSlotPool,
    OnlineInfrastructureError,
)
from gui_agent_captcha.integrations.online_rl import (
    ActionBudgetViolationV4,
    LoopViolationV4,
    NoProgressViolationV4,
    OnlineRewardBreakdownV4,
    PolicyActionViolationV4,
    ProtocolViolationV4,
    V4TrajectoryOutcome,
    reward_breakdown_for_outcome,
)

ONLINE_V4_VIEWPORT = (1280, 720)
ONLINE_V4_IMAGE_MAX_PIXELS = 1280 * 720
ONLINE_V4_CONTEXT_WINDOW = 16384
ONLINE_V4_IMAGE_HISTORY_MAX = 3


class ContextBudgetErrorV4(ValueError):
    """The exact prompt/response sequence cannot fit without truncation."""


class OnlineInfrastructureErrorV4(OnlineInfrastructureError):
    """A browser or generation failure invalidates the V4 sample."""


@dataclass(frozen=True)
class DynamicGenerationBudgetV4:
    context_window: int = ONLINE_V4_CONTEXT_WINDOW
    response_capacity: int = 8192
    safety_reserve_tokens: int = 32
    max_generation_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_generation_tokens is not None and self.max_generation_tokens <= 0:
            raise ValueError("max_generation_tokens must be positive")
        if self.context_window <= 0:
            raise ValueError("context_window must be positive")
        if self.safety_reserve_tokens <= 0:
            raise ValueError("safety_reserve_tokens must be positive")
        if self.safety_reserve_tokens >= self.context_window:
            raise ValueError("safety reserve must be smaller than the context window")
        if self.response_capacity <= 0 or self.response_capacity >= self.context_window:
            raise ValueError("response_capacity must be within the context window")

    def remaining(
        self,
        *,
        prompt_token_count: int,
        accumulated_response_token_count: int,
    ) -> int:
        if prompt_token_count < 0 or accumulated_response_token_count < 0:
            raise ContextBudgetErrorV4("token counts must not be negative")
        remaining = (
            self.context_window
            - prompt_token_count
            - accumulated_response_token_count
            - self.safety_reserve_tokens
        )
        if remaining <= 0:
            raise ContextBudgetErrorV4(
                "context budget exhausted without truncation: "
                f"window={self.context_window}, prompt={prompt_token_count}, "
                f"accumulated_response={accumulated_response_token_count}, "
                f"reserve={self.safety_reserve_tokens}"
            )
        response_remaining = self.response_capacity - accumulated_response_token_count
        if response_remaining <= 0:
            raise ContextBudgetErrorV4(
                "PPO response capacity exhausted without truncation: "
                f"capacity={self.response_capacity}, "
                f"accumulated_response={accumulated_response_token_count}"
            )
        return min(remaining, response_remaining, self.max_generation_tokens or response_remaining)

    def sampling_params(
        self,
        base: Mapping[str, Any],
        *,
        prompt_token_count: int,
        accumulated_response_token_count: int,
    ) -> dict[str, Any]:
        params = dict(base)
        params.pop("max_tokens", None)
        params.pop("max_new_tokens", None)
        params["max_tokens"] = self.remaining(
            prompt_token_count=prompt_token_count,
            accumulated_response_token_count=accumulated_response_token_count,
        )
        return params


@dataclass(frozen=True)
class PublicBrowserObservationV4:
    screenshot_path: str
    size_px: tuple[int, int]
    step_index: int
    screenshot_fingerprint: str
    is_terminal: bool
    cursor_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class GeneratedTurnV4:
    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


def restore_prompt_owned_think_opening_v4(response: str) -> str:
    """Restore the chat-template-owned opening tag without rewriting model text."""

    if not isinstance(response, str):
        raise TypeError("model response must be text")
    if response.lstrip().startswith("<think>"):
        return response
    return "<think>\n" + response


@dataclass(frozen=True)
class RuntimeGenerationRequestV4:
    request_id: str
    prompt: Any
    observations: tuple[PublicBrowserObservationV4, ...]
    action_history: tuple[dict[str, Any], ...]
    response_history: tuple[str, ...]
    accumulated_response_token_count: int


@dataclass(frozen=True)
class BrowserStepAuditV4:
    turn_index: int
    response: str
    action: dict[str, Any] | None
    screenshot_before: str
    screenshot_after: str | None
    screenshot_fingerprint_before: str
    screenshot_fingerprint_after: str | None
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    state_changed: bool
    screenshot_changed: bool | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "response": self.response,
            "action": self.action,
            "screenshot_before": self.screenshot_before,
            "screenshot_after": self.screenshot_after,
            "screenshot_fingerprint_before": self.screenshot_fingerprint_before,
            "screenshot_fingerprint_after": self.screenshot_fingerprint_after,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "state_changed": self.state_changed,
            "screenshot_changed": self.screenshot_changed,
            "error": self.error,
        }


@dataclass(frozen=True)
class BrowserTrajectoryResultV4:
    trajectory_id: str
    request_id: str
    seed: int
    artifact_dir: Path
    outcome: V4TrajectoryOutcome
    reward_breakdown: OnlineRewardBreakdownV4
    terminal_source: str
    observations: tuple[PublicBrowserObservationV4, ...]
    turns: tuple[GeneratedTurnV4, ...]
    actions: tuple[Action, ...]
    steps: tuple[BrowserStepAuditV4, ...]
    terminal_info: dict[str, Any]
    infra_retry_count: int = 0
    cleanup_ok: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


EnvFactoryV4 = Callable[[int, Path], Any]
TurnGeneratorV4 = Callable[
    [RuntimeGenerationRequestV4, dict[str, Any]],
    Awaitable[GeneratedTurnV4],
]
PromptBuilderV4 = Callable[..., Any]
ProtocolStateFactoryV4 = Callable[[int], Any]
TrajectoryFormatValidatorV4 = Callable[[Sequence[str]], bool]


def _runtime_action(action: Any) -> Action:
    if action.kind == "move_to":
        return PrimitiveAction(kind="move_to", x=float(action.x), y=float(action.y))
    if action.kind in {"click", "drag"}:
        return AtomicAction(
            kind=action.kind,
            points=[(float(x), float(y)) for x, y in action.points],
        )
    return PrimitiveAction(kind=action.kind)


class BrowserTrajectoryRunnerV4:
    """Run one browser trajectory with injected domain policy and prompt rules."""

    def __init__(
        self,
        *,
        env_factory: EnvFactoryV4,
        turn_generator: TurnGeneratorV4,
        artifact_root: Path,
        slot_pool: BrowserSlotPool,
        task_type: str,
        prompt_builder: PromptBuilderV4,
        protocol_state_factory: ProtocolStateFactoryV4,
        trajectory_format_validator: TrajectoryFormatValidatorV4,
        response_format_errors: tuple[type[Exception], ...],
        success_action_kinds: Sequence[str] = ("mouse_up",),
        max_steps: int = 8,
        max_infra_retries: int = 2,
        id_factory: Callable[[], str] | None = None,
        viewport: tuple[int, int] = ONLINE_V4_VIEWPORT,
        non_retryable_errors: tuple[type[Exception], ...] = (
            ContextBudgetErrorV4,
        ),
        thread_name_prefix: str = "browser-trajectory-v4",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_infra_retries < 0:
            raise ValueError("max_infra_retries cannot be negative")
        if len(viewport) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in viewport
        ):
            raise ValueError("viewport must contain two positive integers")
        if not response_format_errors:
            raise ValueError("response_format_errors must not be empty")
        if not success_action_kinds:
            raise ValueError("success_action_kinds must not be empty")
        if not isinstance(thread_name_prefix, str) or not thread_name_prefix:
            raise ValueError("thread_name_prefix must be non-empty text")
        self.env_factory = env_factory
        self.turn_generator = turn_generator
        self.artifact_root = Path(artifact_root)
        self.slot_pool = slot_pool
        self.task_type = task_type
        self.prompt_builder = prompt_builder
        self.protocol_state_factory = protocol_state_factory
        self.trajectory_format_validator = trajectory_format_validator
        self.response_format_errors = response_format_errors
        self.success_action_kinds = frozenset(str(kind) for kind in success_action_kinds)
        self.max_steps = max_steps
        self.max_infra_retries = max_infra_retries
        self.id_factory = id_factory or (lambda: uuid4().hex)
        self.viewport = viewport
        self.non_retryable_errors = non_retryable_errors
        self.thread_name_prefix = thread_name_prefix

    @staticmethod
    async def _on_browser_thread(
        executor: ThreadPoolExecutor,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            partial(function, *args, **kwargs),
        )

    @staticmethod
    def _fingerprint(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"browser screenshot is missing: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _public_observation(
        self,
        observation: Observation,
        *,
        step_index: int,
        is_terminal: bool,
    ) -> PublicBrowserObservationV4:
        if observation.size_px != self.viewport:
            raise RuntimeError(
                f"online V4 requires a raw {self.viewport[0]}x{self.viewport[1]} "
                f"browser observation, got {observation.size_px!r}"
            )
        path = Path(observation.screenshot_path)
        return PublicBrowserObservationV4(
            screenshot_path=str(path),
            size_px=self.viewport,
            step_index=step_index,
            screenshot_fingerprint=self._fingerprint(path),
            is_terminal=is_terminal,
            cursor_xy=observation.cursor_xy,
        )

    @staticmethod
    def _observed_progress_signature(
        observation: PublicBrowserObservationV4,
        state: Any,
    ) -> tuple[Any, ...]:
        return (
            observation.screenshot_fingerprint,
            observation.cursor_xy,
            state.mouse_down,
            state.held_move,
            state.released,
            observation.is_terminal,
        )

    def _result(
        self,
        *,
        trajectory_id: str,
        request_id: str,
        seed: int,
        artifact_dir: Path,
        outcome: V4TrajectoryOutcome,
        terminal_source: str,
        observations: list[PublicBrowserObservationV4],
        turns: list[GeneratedTurnV4],
        actions: list[Action],
        steps: list[BrowserStepAuditV4],
        terminal_info: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> BrowserTrajectoryResultV4:
        diagnostics: dict[str, Any] = {
            "outcome": outcome.value,
            "num_model_turns": len(turns),
            "num_actions": len(actions),
            "num_observations": len(observations),
            "generation_budgets": [turn.extra_fields.get("generation_budget") for turn in turns],
            "response_token_counts": [len(turn.token_ids) for turn in turns],
        }
        if error is not None:
            diagnostics["policy_error"] = f"{type(error).__name__}: {error}"
        return BrowserTrajectoryResultV4(
            trajectory_id=trajectory_id,
            request_id=request_id,
            seed=seed,
            artifact_dir=artifact_dir,
            outcome=outcome,
            reward_breakdown=reward_breakdown_for_outcome(
                outcome,
                terminal_source=terminal_source,
                format_valid=(
                    False if terminal_source in {"generation_length", "context_budget"}
                    else self.trajectory_format_validator(tuple(turn.text for turn in turns))
                ),
            ),
            terminal_source=terminal_source,
            observations=tuple(observations),
            turns=tuple(turns),
            actions=tuple(actions),
            steps=tuple(steps),
            terminal_info=dict(terminal_info or {}),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _rejected_step(
        *,
        turn_index: int,
        response: str,
        action: dict[str, Any] | None,
        observation: PublicBrowserObservationV4,
        state_before: dict[str, Any],
        state_after: dict[str, Any],
        error: Exception,
    ) -> BrowserStepAuditV4:
        return BrowserStepAuditV4(
            turn_index=turn_index,
            response=response,
            action=action,
            screenshot_before=observation.screenshot_path,
            screenshot_after=None,
            screenshot_fingerprint_before=observation.screenshot_fingerprint,
            screenshot_fingerprint_after=None,
            state_before=state_before,
            state_after=state_after,
            state_changed=state_before != state_after,
            screenshot_changed=None,
            error=f"{type(error).__name__}: {error}",
        )

    async def _run_attempt(
        self,
        *,
        env: Any,
        executor: ThreadPoolExecutor,
        trajectory_id: str,
        request_id: str,
        seed: int,
        artifact_dir: Path,
        sampling_params: dict[str, Any],
    ) -> BrowserTrajectoryResultV4:
        task_id = (
            env.resolve_task_instance_id(self.task_type)
            if hasattr(env, "resolve_task_instance_id")
            else f"{self.task_type}:seed={seed}"
        )
        initial: Observation = await self._on_browser_thread(
            executor,
            env.reset,
            task_type=self.task_type,
            task_id=task_id,
        )
        observations = [
            self._public_observation(initial, step_index=0, is_terminal=False)
        ]
        turns: list[GeneratedTurnV4] = []
        actions: list[Action] = []
        steps: list[BrowserStepAuditV4] = []
        state = self.protocol_state_factory(self.max_steps)
        state.record_screenshot(observations[0].screenshot_fingerprint)
        observed_signature = self._observed_progress_signature(observations[0], state)
        accumulated_response_tokens = 0

        while True:
            prompt = self.prompt_builder(
                observations=observations,
                state=state,
                max_steps=self.max_steps,
            )
            request = RuntimeGenerationRequestV4(
                request_id=request_id,
                prompt=prompt,
                observations=tuple(observations),
                action_history=tuple(
                    action.to_dict() for action in state.action_history
                ),
                response_history=state.response_history,
                accumulated_response_token_count=accumulated_response_tokens,
            )
            try:
                generated = await self.turn_generator(request, dict(sampling_params))
            except ContextBudgetErrorV4 as exc:
                if not getattr(self, "terminate_on_generation_limit", False) or not turns:
                    raise
                return self._result(
                    trajectory_id=trajectory_id, request_id=request_id, seed=seed,
                    artifact_dir=artifact_dir, outcome=V4TrajectoryOutcome.CONTEXT_BUDGET,
                    terminal_source="context_budget", observations=observations,
                    turns=turns, actions=actions, steps=steps, error=exc,
                )
            if not isinstance(generated, GeneratedTurnV4):
                raise TypeError("turn_generator must return GeneratedTurnV4")
            if generated.logprobs is not None and len(generated.logprobs) != len(
                generated.token_ids
            ):
                raise RuntimeError("generated token/logprob lengths are inconsistent")
            if generated.extra_fields.get("prompt_owned_think_opening") is True:
                generated = replace(
                    generated,
                    text=restore_prompt_owned_think_opening_v4(generated.text),
                )
            turns.append(generated)
            accumulated_response_tokens += len(generated.token_ids)
            if getattr(self, "terminate_on_generation_limit", False):
                budget = generated.extra_fields.get("generation_budget")
                exhausted = generated.extra_fields.get("finish_reason") == "length"
                exhausted = exhausted or (isinstance(budget, int) and len(generated.token_ids) >= budget)
                if exhausted:
                    return self._result(
                        trajectory_id=trajectory_id, request_id=request_id, seed=seed,
                        artifact_dir=artifact_dir, outcome=V4TrajectoryOutcome.GENERATION_LENGTH,
                        terminal_source="generation_length", observations=observations,
                        turns=turns, actions=actions, steps=steps,
                    )
            state_before = state.snapshot()
            parsed_action: dict[str, Any] | None = None
            try:
                parsed = state.accept_response(generated.text)
                parsed_action = parsed.action.to_dict()
            except self.response_format_errors as exc:
                steps.append(
                    self._rejected_step(
                        turn_index=len(turns) - 1,
                        response=generated.text,
                        action=None,
                        observation=observations[-1],
                        state_before=state_before,
                        state_after=state.snapshot(),
                        error=exc,
                    )
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.PARSE_ERROR,
                    terminal_source="response_parser",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=exc,
                )
            except PolicyActionViolationV4 as exc:
                steps.append(
                    self._rejected_step(
                        turn_index=len(turns) - 1,
                        response=generated.text,
                        action=None,
                        observation=observations[-1],
                        state_before=state_before,
                        state_after=state.snapshot(),
                        error=exc,
                    )
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.POLICY_ACTION_ERROR,
                    terminal_source="task_policy",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=exc,
                )
            except (ProtocolViolationV4, ActionBudgetViolationV4) as exc:
                steps.append(
                    self._rejected_step(
                        turn_index=len(turns) - 1,
                        response=generated.text,
                        action=parsed_action,
                        observation=observations[-1],
                        state_before=state_before,
                        state_after=state.snapshot(),
                        error=exc,
                    )
                )
                outcome = (
                    V4TrajectoryOutcome.MAX_STEPS
                    if isinstance(exc, ActionBudgetViolationV4)
                    else V4TrajectoryOutcome.PROTOCOL_ERROR
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=outcome,
                    terminal_source=(
                        "action_budget"
                        if outcome is V4TrajectoryOutcome.MAX_STEPS
                        else "protocol_state"
                    ),
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=exc,
                )
            except LoopViolationV4 as exc:
                steps.append(
                    self._rejected_step(
                        turn_index=len(turns) - 1,
                        response=generated.text,
                        action=parsed_action,
                        observation=observations[-1],
                        state_before=state_before,
                        state_after=state.snapshot(),
                        error=exc,
                    )
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.LOOP,
                    terminal_source="loop_guard",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=exc,
                )

            action = _runtime_action(parsed.action)
            before = observations[-1]
            step_result: StepResult = await self._on_browser_thread(
                executor,
                env.step,
                action,
            )
            actions.append(action)
            after = self._public_observation(
                step_result.observation,
                step_index=state.step_count,
                is_terminal=bool(step_result.done),
            )
            observations.append(after)
            state_after = state.snapshot()
            try:
                state.record_screenshot(after.screenshot_fingerprint)
            except NoProgressViolationV4 as exc:
                steps.append(
                    BrowserStepAuditV4(
                        turn_index=len(turns) - 1,
                        response=generated.text,
                        action=parsed.action.to_dict(),
                        screenshot_before=before.screenshot_path,
                        screenshot_after=after.screenshot_path,
                        screenshot_fingerprint_before=before.screenshot_fingerprint,
                        screenshot_fingerprint_after=after.screenshot_fingerprint,
                        state_before=state_before,
                        state_after=state_after,
                        state_changed=state_before != state_after,
                        screenshot_changed=(
                            before.screenshot_fingerprint
                            != after.screenshot_fingerprint
                        ),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.NO_PROGRESS,
                    terminal_source="no_progress_guard",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=exc,
                )
            steps.append(
                BrowserStepAuditV4(
                    turn_index=len(turns) - 1,
                    response=generated.text,
                    action=parsed.action.to_dict(),
                    screenshot_before=before.screenshot_path,
                    screenshot_after=after.screenshot_path,
                    screenshot_fingerprint_before=before.screenshot_fingerprint,
                    screenshot_fingerprint_after=after.screenshot_fingerprint,
                    state_before=state_before,
                    state_after=state_after,
                    state_changed=state_before != state_after,
                    screenshot_changed=(
                        before.screenshot_fingerprint
                        != after.screenshot_fingerprint
                    ),
                )
            )

            next_observed_signature = self._observed_progress_signature(after, state)
            if next_observed_signature == observed_signature:
                error = NoProgressViolationV4(
                    "browser screenshot and observable interaction state did not change"
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.NO_PROGRESS,
                    terminal_source="no_progress_guard",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    error=error,
                )
            observed_signature = next_observed_signature

            success_value = step_result.info.get("success")
            if success_value is True and action.kind not in self.success_action_kinds:
                raise RuntimeError(
                    "browser reported success after a forbidden terminal action: "
                    f"{action.kind}"
                )
            if success_value is True:
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.SUCCESS,
                    terminal_source='step_result.info["success"]',
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    terminal_info={"success": True},
                )
            if step_result.done or state.released:
                terminal_info = (
                    {"success": success_value}
                    if isinstance(success_value, bool)
                    else {}
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.BROWSER_FAILURE,
                    terminal_source="step_result.done_without_success",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                    terminal_info=terminal_info,
                )
            if state.reached_action_budget:
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=V4TrajectoryOutcome.MAX_STEPS,
                    terminal_source="action_budget",
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    steps=steps,
                )

    @staticmethod
    def _write_audit(result: BrowserTrajectoryResultV4) -> None:
        payload = {
            "trajectory_id": result.trajectory_id,
            "request_id": result.request_id,
            "seed": result.seed,
            "outcome": result.outcome.value,
            "terminal_source": result.terminal_source,
            "reward_breakdown": result.reward_breakdown.to_dict(),
            "infra_retry_count": result.infra_retry_count,
            "cleanup_ok": result.cleanup_ok,
            "terminal_info": result.terminal_info,
            "responses": [turn.text for turn in result.turns],
            "actions": [action.to_dict() for action in result.actions],
            "screenshots": [
                {
                    "path": observation.screenshot_path,
                    "fingerprint": observation.screenshot_fingerprint,
                    "size_px": list(observation.size_px),
                    "step_index": observation.step_index,
                    "is_terminal": observation.is_terminal,
                    "cursor_xy": (
                        list(observation.cursor_xy)
                        if observation.cursor_xy is not None
                        else None
                    ),
                }
                for observation in result.observations
            ],
            "steps": [step.to_dict() for step in result.steps],
            "diagnostics": dict(result.diagnostics),
        }
        (result.artifact_dir / "trajectory_result_v4.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def run(
        self,
        *,
        seed: int,
        sampling_params: dict[str, Any],
    ) -> BrowserTrajectoryResultV4:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("trajectory seed must be an integer")
        trajectory_id = str(self.id_factory())
        request_id = str(self.id_factory())
        root = self.artifact_root / f"seed-{seed}" / trajectory_id
        root.mkdir(parents=True, exist_ok=False)
        started = time.time()
        lease = await asyncio.to_thread(self.slot_pool.acquire)
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"{self.thread_name_prefix}-{trajectory_id[:12]}",
        )
        errors: list[str] = []
        try:
            total_attempts = self.max_infra_retries + 1
            for attempt_index in range(total_attempts):
                attempt_dir = root / f"attempt-{attempt_index:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                env: Any = None
                result: BrowserTrajectoryResultV4 | None = None
                error: Exception | None = None
                try:
                    env = await self._on_browser_thread(
                        executor,
                        self.env_factory,
                        seed,
                        attempt_dir,
                    )
                    result = await self._run_attempt(
                        env=env,
                        executor=executor,
                        trajectory_id=trajectory_id,
                        request_id=request_id,
                        seed=seed,
                        artifact_dir=attempt_dir,
                        sampling_params=sampling_params,
                    )
                except self.non_retryable_errors:
                    raise
                except Exception as exc:
                    error = exc
                finally:
                    if env is not None:
                        try:
                            await self._on_browser_thread(executor, env.close)
                        except Exception as close_exc:
                            error = (
                                close_exc
                                if error is None
                                else RuntimeError(
                                    f"{error}; cleanup failed: {close_exc}"
                                )
                            )
                if error is None:
                    if result is None:
                        raise AssertionError("trajectory attempt ended without a result")
                    diagnostics = dict(result.diagnostics)
                    diagnostics.update(
                        {
                            "trajectory_started_at_unix": started,
                            "trajectory_finished_at_unix": time.time(),
                            "infra_errors_before_success": list(errors),
                        }
                    )
                    completed = replace(
                        result,
                        infra_retry_count=attempt_index,
                        cleanup_ok=True,
                        diagnostics=diagnostics,
                    )
                    self._write_audit(completed)
                    return completed
                errors.append(
                    f"attempt-{attempt_index:02d}: {type(error).__name__}: {error}"
                )

            invalid = {
                "trajectory_id": trajectory_id,
                "request_id": request_id,
                "seed": seed,
                "outcome": V4TrajectoryOutcome.INFRA_ERROR.value,
                "reward_breakdown": reward_breakdown_for_outcome(
                    V4TrajectoryOutcome.INFRA_ERROR,
                    terminal_source="infrastructure",
                ).to_dict(),
                "errors": errors,
            }
            (root / "infrastructure_invalid_v4.json").write_text(
                json.dumps(invalid, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            raise OnlineInfrastructureErrorV4(
                "online V4 trajectory exhausted infrastructure retries: " + errors[-1],
                attempts=total_attempts,
                errors=tuple(errors),
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            lease.release()
