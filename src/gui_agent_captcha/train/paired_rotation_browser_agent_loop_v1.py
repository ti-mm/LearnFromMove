from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from gui_agent_captcha.actions import Action, AtomicAction, PrimitiveAction
from gui_agent_captcha.benchmarks.exploration_depth.training_no_think import (
    IMAGE_HISTORY_MAX,
    PROMPT_CONTRACT as NO_THINK_PROMPT_CONTRACT,
    SIX_ACTION_KINDS,
    _no_think_action_context,
    _no_think_action_prompt,
)
from gui_agent_captcha.benchmarks.exploration_depth.training_with_think import (
    PROMPT_CONTRACT as WITH_THINK_PROMPT_CONTRACT,
    _think_action_prompt,
)
from gui_agent_captcha.domains.rotation.action_contract import ResponseFormatError
from gui_agent_captcha.domains.rotation.agent_loop import default_browser_slot_pool
from gui_agent_captcha.domains.rotation.paired_online_dataset import (
    EXECUTABLE_ACTION_KINDS,
    FORMAT_ACTION_KINDS,
    HELDOUT_POOL_NAME,
    MAX_STEPS,
    POOL_NAME,
    SUITE_ID,
    VIEWPORT,
)
from gui_agent_captcha.domains.rotation.outcome_protocol import (
    OnlineProtocolStateV4,
    trajectory_format_valid_v4,
)
from gui_agent_captcha.integrations.browser_trajectory import (
    BrowserTrajectoryRunnerV4,
    ContextBudgetErrorV4,
    EnvFactoryV4,
    PublicBrowserObservationV4,
    TurnGeneratorV4,
)
from gui_agent_captcha.integrations.online_rl import (
    ActionBudgetViolationV4,
    NoProgressViolationV4,
    ProtocolViolationV4,
)
from gui_agent_captcha.integrations.verl_browser import (
    VERL_AVAILABLE_V4,
    VerlBrowserAgentLoopV4,
)

from .qwen3_vl_sft import (
    SftPromptBuildResult,
    build_action_context_from_history,
)
from .interaction_browser_agent_loop_v4 import (
    BrowserTrajectoryRunnerV4 as RotationBrowserTrajectoryRunnerV4,
    InteractionBrowserAgentLoopV4,
)


NO_THINK_AGENT_NAME = "paired_rotation_no_think_browser_online_v1"
WITH_THINK_AGENT_NAME = "paired_rotation_with_think_browser_online_v1"
NO_THINK_RESPONSE_CONTRACT = "empty_think_tag_then_exact_action_v1"
NO_THINK_TASK_REQUIREMENT = "Drag the slider to complete verification."


@dataclass(frozen=True)
class NoThinkRotationResponse:
    action: Action
    normalized_text: str


def _json_object(value: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ResponseFormatError(f"duplicate JSON key: {key!r}")
            result[key] = item
        return result

    try:
        payload = json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ResponseFormatError(f"non-finite JSON number: {item}")
            ),
        )
    except ResponseFormatError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResponseFormatError("response must be exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ResponseFormatError("response JSON value must be an object")
    return payload


def _coordinate(value: Any, *, field_name: str, round_to_int: bool) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResponseFormatError(f"{field_name} must be numeric")
    coordinate = float(value)
    if not math.isfinite(coordinate) or not 0 <= coordinate <= 1000:
        raise ResponseFormatError(f"{field_name} must be finite and within 0..1000")
    return int(round(coordinate)) if round_to_int else coordinate


def parse_no_think_rotation_response(response: str) -> NoThinkRotationResponse:
    if not isinstance(response, str) or not response.strip():
        raise ResponseFormatError("no-think response must be non-empty JSON text")
    if "<think>" in response or "</think>" in response:
        raise ResponseFormatError("empty Think is owned by the chat-template prompt")
    payload = _json_object(response.strip())
    if set(payload) != {"action"} or not isinstance(payload["action"], dict):
        raise ResponseFormatError("top-level JSON must contain exactly one action object")
    raw = payload["action"]
    kind = raw.get("kind")
    if kind == "move_to":
        if set(raw) != {"kind", "x", "y"}:
            raise ResponseFormatError("move_to requires exactly kind, x, and y")
        action: Action = PrimitiveAction(
            kind="move_to",
            x=_coordinate(raw["x"], field_name="move_to.x", round_to_int=True),
            y=_coordinate(raw["y"], field_name="move_to.y", round_to_int=True),
        )
    elif kind in {"mouse_down", "mouse_up", "left_click"}:
        if set(raw) != {"kind"}:
            raise ResponseFormatError(f"{kind} accepts only the kind field")
        action = PrimitiveAction(kind=kind)
    elif kind in {"click", "drag"}:
        if set(raw) != {"kind", "points"}:
            raise ResponseFormatError(f"{kind} requires exactly kind and points")
        points = raw["points"]
        expected = 1 if kind == "click" else 2
        if not isinstance(points, list) or len(points) != expected:
            raise ResponseFormatError(f"{kind}.points must contain {expected} point(s)")
        normalized_points: list[tuple[float, float]] = []
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ResponseFormatError(f"{kind}.points[{index}] must be [x, y]")
            normalized_points.append(
                (
                    float(
                        _coordinate(
                            point[0],
                            field_name=f"{kind}.points[{index}].x",
                            round_to_int=False,
                        )
                    ),
                    float(
                        _coordinate(
                            point[1],
                            field_name=f"{kind}.points[{index}].y",
                            round_to_int=False,
                        )
                    ),
                )
            )
        action = AtomicAction(kind=kind, points=normalized_points)
    else:
        raise ResponseFormatError(f"unsupported no-think action kind: {kind!r}")
    normalized = json.dumps(
        {"action": action.to_dict()},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return NoThinkRotationResponse(action=action, normalized_text=normalized)


def no_think_trajectory_format_valid(responses: Sequence[str]) -> bool:
    if not responses:
        return False
    try:
        for response in responses:
            parse_no_think_rotation_response(response)
    except ValueError:
        return False
    return True


@dataclass
class NoThinkRotationProtocolState:
    max_steps: int = MAX_STEPS
    cursor: tuple[int, int] | None = field(default=None, init=False)
    mouse_down: bool = field(default=False, init=False)
    held_move: bool = field(default=False, init=False)
    released: bool = field(default=False, init=False)
    step_count: int = field(default=0, init=False)
    _actions: list[Action] = field(default_factory=list, init=False)
    _responses: list[str] = field(default_factory=list, init=False)
    _screenshot_fingerprints: list[str] = field(default_factory=list, init=False)
    _screenshot_state_snapshots: list[tuple[Any, ...]] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @property
    def action_history(self) -> tuple[Action, ...]:
        return tuple(self._actions)

    @property
    def response_history(self) -> tuple[str, ...]:
        return tuple(self._responses)

    @property
    def reached_action_budget(self) -> bool:
        return self.step_count >= self.max_steps and not self.released

    def snapshot(self) -> dict[str, Any]:
        return {
            "cursor": list(self.cursor) if self.cursor is not None else None,
            "mouse_down": self.mouse_down,
            "held_move": self.held_move,
            "released": self.released,
            "step_count": self.step_count,
            "action_history": [action.to_dict() for action in self._actions],
            "response_history": list(self._responses),
            "screenshot_fingerprints": list(self._screenshot_fingerprints),
        }

    def _progress_snapshot(self) -> tuple[Any, ...]:
        return (
            self.cursor,
            self.mouse_down,
            self.held_move,
            self.released,
            tuple(json.dumps(action.to_dict(), sort_keys=True) for action in self._actions),
        )

    def record_screenshot(self, fingerprint: str) -> None:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("screenshot fingerprint must be non-empty text")
        snapshot = self._progress_snapshot()
        if (
            self._screenshot_fingerprints
            and self._screenshot_fingerprints[-1] == fingerprint
            and self._screenshot_state_snapshots[-1] == snapshot
        ):
            raise NoProgressViolationV4(
                "screenshot fingerprint and public protocol state are unchanged"
            )
        self._screenshot_fingerprints.append(fingerprint)
        self._screenshot_state_snapshots.append(snapshot)

    def accept_response(self, response: str) -> NoThinkRotationResponse:
        if self.released:
            raise ProtocolViolationV4("actions after a terminal release are forbidden")
        if self.reached_action_budget:
            raise ActionBudgetViolationV4("maximum action budget has been reached")
        parsed = parse_no_think_rotation_response(response)
        action = parsed.action
        if isinstance(action, PrimitiveAction) and action.kind == "move_to":
            assert action.x is not None and action.y is not None
            self.cursor = (int(action.x), int(action.y))
            if self.mouse_down:
                self.held_move = True
        elif isinstance(action, PrimitiveAction) and action.kind == "mouse_down":
            self.mouse_down = True
            self.held_move = False
        elif isinstance(action, PrimitiveAction) and action.kind == "mouse_up":
            self.mouse_down = False
            self.released = True
        elif isinstance(action, PrimitiveAction) and action.kind == "left_click":
            self.mouse_down = False
            self.released = True
        elif isinstance(action, AtomicAction):
            self.mouse_down = False
            self.released = True
        self._actions.append(action)
        self._responses.append(response.strip())
        self.step_count += 1
        return parsed


def build_no_think_runtime_prompt(
    *,
    observations: Sequence[PublicBrowserObservationV4],
    state: NoThinkRotationProtocolState,
    max_steps: int,
) -> SftPromptBuildResult:
    if not observations or len(observations) != len(state.action_history) + 1:
        raise ValueError("no-think prompt requires one observation per action plus current")
    if observations[-1].size_px != (1280, 720):
        raise ValueError("no-think rotation prompt requires a 1280x720 observation")
    retained = tuple(observations[-IMAGE_HISTORY_MAX:])
    first_retained_index = len(observations) - len(retained)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _no_think_action_prompt(NO_THINK_TASK_REQUIREMENT)}
    ]

    def append_previous_response(action_index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {action_index + 1} assistant response "
                    f"(context only):\n{state.response_history[action_index]}\n"
                ),
            }
        )

    for action_index in range(first_retained_index):
        append_previous_response(action_index)
    for offset, observation in enumerate(retained):
        content.append({"type": "image", "image": observation.screenshot_path})
        action_index = first_retained_index + offset
        if action_index < len(state.action_history):
            append_previous_response(action_index)
    content.append(
        {
            "type": "text",
            "text": _no_think_action_context(len(state.action_history)),
        }
    )
    context = build_action_context_from_history(
        action_history=tuple(action.to_dict() for action in state.action_history),
        total_actions=max_steps,
        cursor_xy=state.cursor,
        button_state="down" if state.mouse_down else "up",
        task_type="rotation_captcha",
        allowed_kinds=SIX_ACTION_KINDS,
        budget_remaining=max(0, max_steps - state.step_count),
    )
    return SftPromptBuildResult(
        messages=[{"role": "user", "content": content}],
        image_paths=tuple(Path(item.screenshot_path) for item in retained),
        context=context,
        prompt_contract=NO_THINK_PROMPT_CONTRACT,
    )


def build_with_think_runtime_prompt(
    *,
    observations: Sequence[PublicBrowserObservationV4],
    state: OnlineProtocolStateV4,
    max_steps: int,
) -> SftPromptBuildResult:
    """Rebuild the current six-environment visible-Think SFT prompt."""

    if not observations or len(observations) != len(state.action_history) + 1:
        raise ValueError("with-think prompt requires one observation per action plus current")
    if observations[-1].size_px != VIEWPORT:
        raise ValueError("with-think rotation prompt requires a 1280x720 observation")
    retained = tuple(observations[-IMAGE_HISTORY_MAX:])
    first_retained_index = len(observations) - len(retained)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _think_action_prompt(NO_THINK_TASK_REQUIREMENT)}
    ]

    def append_previous_response(action_index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {action_index + 1} assistant response "
                    f"(context only):\n{state.response_history[action_index]}\n"
                ),
            }
        )

    for action_index in range(first_retained_index):
        append_previous_response(action_index)
    for offset, observation in enumerate(retained):
        content.append({"type": "image", "image": observation.screenshot_path})
        action_index = first_retained_index + offset
        if action_index < len(state.action_history):
            append_previous_response(action_index)
    action_index = len(state.action_history)
    history_text = (
        f"{action_index} previous assistant response(s) are included in "
        "chronological order."
        if action_index
        else "No previous actions are present in this trajectory."
    )
    content.append(
        {
            "type": "text",
            "text": "\n".join(
                (
                    history_text,
                    "The last attached image is the current observation.",
                    f"Return the Think block and JSON action for step {action_index + 1}.",
                )
            ),
        }
    )
    context = build_action_context_from_history(
        action_history=tuple(action.to_dict() for action in state.action_history),
        total_actions=max_steps,
        cursor_xy=state.cursor,
        button_state="down" if state.mouse_down else "up",
        task_type="rotation",
        allowed_kinds=SIX_ACTION_KINDS,
        budget_remaining=max(0, max_steps - state.step_count),
    )
    return SftPromptBuildResult(
        messages=[{"role": "user", "content": content}],
        image_paths=tuple(Path(item.screenshot_path) for item in retained),
        context=context,
        prompt_contract=WITH_THINK_PROMPT_CONTRACT,
    )


def normalize_paired_rotation_task_config(
    task_type: Any,
    task_config: Any,
) -> dict[str, Any]:
    if task_type not in {None, "rotation_captcha"}:
        raise ValueError("paired rotation AgentLoop only supports rotation_captcha")
    if hasattr(task_config, "item") and not isinstance(task_config, Mapping):
        task_config = task_config.item()
    if not isinstance(task_config, Mapping):
        raise ValueError("paired rotation row is missing task_config")
    normalized = dict(task_config)
    for key in ("prompt_contract", "response_contract"):
        if normalized.get(key) is None:
            normalized.pop(key, None)
    required = {
        "background_pool",
        "suite_id",
        "manifest_path",
        "benchmark_variant",
        "episode_id",
        "viewport",
        "max_steps",
        "coordinate_format",
        "format_action_kinds",
        "executable_action_kinds",
    }
    if set(normalized) != required:
        raise ValueError(
            f"paired rotation task_config keys must be exactly {sorted(required)!r}"
        )
    if not isinstance(normalized["suite_id"], str) or not normalized["suite_id"].strip():
        raise ValueError("paired rotation suite_id must be nonempty")
    background_pool = normalized["background_pool"]
    if (
        not isinstance(background_pool, str)
        or not background_pool.strip()
        or "/" in background_pool
        or ".." in background_pool
    ):
        raise ValueError("paired rotation row selects an invalid background pool")
    variant = str(normalized["benchmark_variant"])
    if variant not in {"rotation_inner", "rotation_outer"}:
        raise ValueError("paired rotation variant must be rotation_inner or rotation_outer")
    expected_suffix = f"--{variant.replace('_', '-')}"
    if not str(normalized["episode_id"]).endswith(expected_suffix):
        raise ValueError("paired rotation episode_id does not match its variant")
    manifest_path = Path(str(normalized["manifest_path"]))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("paired rotation manifest_path must be an existing absolute file")
    try:
        episode = _paired_episode_index(str(manifest_path.resolve()))[
            (variant, str(normalized["episode_id"]))
        ]
    except KeyError as exc:
        raise ValueError("paired rotation row selects an unknown episode") from exc
    if normalized["suite_id"] != episode.get("suite_id"):
        raise ValueError("paired rotation suite ID differs from its manifest episode")
    provenance = episode.get("shared_scene_config", {}).get(
        "background_provenance", {}
    )
    expected_pool = provenance.get("pool_name") if isinstance(provenance, Mapping) else None
    if expected_pool and background_pool != expected_pool:
        raise ValueError("paired rotation background pool differs from its manifest episode")
    if not expected_pool and background_pool not in {POOL_NAME, HELDOUT_POOL_NAME}:
        raise ValueError("paired rotation legacy episode selects an unexpected background pool")
    if list(normalized["viewport"]) != list(VIEWPORT):
        raise ValueError("paired rotation viewport must be exactly 1280x720")
    if normalized["max_steps"] != MAX_STEPS:
        raise ValueError(f"paired rotation max_steps must be exactly {MAX_STEPS}")
    if normalized["coordinate_format"] != "qwen_relative_0_1000":
        raise ValueError("paired rotation coordinate format must use Qwen relative bins")
    if tuple(normalized["format_action_kinds"]) != FORMAT_ACTION_KINDS:
        raise ValueError("paired rotation format action grammar is invalid")
    if tuple(normalized["executable_action_kinds"]) != EXECUTABLE_ACTION_KINDS:
        raise ValueError("paired rotation executable action contract is invalid")
    return normalized


def paired_rotation_env_factory(
    *,
    variant: str,
    manifest_path: Path,
    episode_id: str,
) -> EnvFactoryV4:
    episode = _paired_episode_index(str(manifest_path.resolve()))[(variant, episode_id)]

    def create(seed: int, artifact_dir: Path) -> Any:
        from gui_agent_captcha.domains.rotation.paired import (
            PairedInnerRotationEnv,
            PairedOuterRotationEnv,
        )

        env_class = (
            PairedOuterRotationEnv
            if variant == "rotation_outer"
            else PairedInnerRotationEnv
        )
        env = env_class(
            base_url=os.environ.get(
                "INTERACTION_CAPTCHA_URL",
                "http://127.0.0.1:4321/",
            ),
            artifact_dir=artifact_dir,
            challenge_seed=seed,
            enable_playwright=True,
            headless=os.environ.get("INTERACTION_BROWSER_HEADLESS", "1") != "0",
            viewport_width=VIEWPORT[0],
            viewport_height=VIEWPORT[1],
            navigate_on_launch=False,
            navigation_timeout_ms=int(
                os.environ.get("INTERACTION_NAVIGATION_TIMEOUT_MS", "60000")
            ),
            manifest_path=manifest_path,
        )
        env._paired_manifest_cache = {
            "schema": "gui_captcha_exploration_depth_manifest_v1",
            "episodes": [episode],
        }
        return env

    return create


@lru_cache(maxsize=4)
def _paired_episode_index(
    manifest_path: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("schema") != "gui_captcha_exploration_depth_manifest_v1":
        raise ValueError(f"unsupported paired rotation manifest: {manifest_path}")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_episode in payload.get("episodes", []):
        if not isinstance(raw_episode, dict):
            continue
        variant = str(raw_episode.get("variant") or "")
        if variant not in {"rotation_inner", "rotation_outer"}:
            continue
        episode = dict(raw_episode)
        episode_id = str(episode["episode_id"])
        pair_id = str(episode["pair_id"])
        index[(variant, episode_id)] = episode
        index[(variant, pair_id)] = episode
    return index


if not VERL_AVAILABLE_V4:

    class PairedRotationBrowserAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "PairedRotationBrowserAgentLoopV1 requires the project VERL environment"
            )

    class PairedRotationNoThinkBrowserAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "PairedRotationNoThinkBrowserAgentLoopV1 requires the project VERL "
                "environment"
            )

    class PairedRotationWithThinkBrowserAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "PairedRotationWithThinkBrowserAgentLoopV1 requires the project VERL "
                "environment"
            )

else:
    from verl.experimental.agent_loop.agent_loop import register

    @register("paired_rotation_browser_online_v1")
    class PairedRotationBrowserAgentLoopV1(InteractionBrowserAgentLoopV4):
        """Final-1404 self-history loop over paired inner/outer replay tasks."""

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_paired_rotation_task_config(task_type, task_config)

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            variant = str(task_config["benchmark_variant"])
            manifest_path = Path(str(task_config["manifest_path"]))
            episode_id = str(task_config["episode_id"])
            return RotationBrowserTrajectoryRunnerV4(
                env_factory=paired_rotation_env_factory(
                    variant=variant,
                    manifest_path=manifest_path,
                    episode_id=episode_id,
                ),
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                task_type=episode_id,
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
            )

    @register(NO_THINK_AGENT_NAME)
    class PairedRotationNoThinkBrowserAgentLoopV1(VerlBrowserAgentLoopV4):
        """No-think six-action policy loop over paired rotation replay tasks."""

        def __init__(
            self,
            *args: Any,
            artifact_root: str | Path | None = None,
            slot_pool: Any | None = None,
            id_factory: Any | None = None,
            max_steps: int = MAX_STEPS,
            max_infra_retries: int = 2,
            context_window: int = 16384,
            generation_safety_reserve_tokens: int = 32,
            **kwargs: Any,
        ) -> None:
            def unused_env_factory(seed: int, path: Path) -> Any:
                del seed, path
                raise RuntimeError("paired no-think env factory requires task_config")

            super().__init__(
                *args,
                env_factory=unused_env_factory,
                artifact_root=(
                    artifact_root
                    or os.environ.get(
                        "INTERACTION_ONLINE_NOTHINK_ARTIFACT_ROOT",
                        "artifacts/runs/paired_rotation_no_think_grpo",
                    )
                ),
                slot_pool=slot_pool or default_browser_slot_pool(),
                prompt_contract=NO_THINK_PROMPT_CONTRACT,
                prompt_owned_think_opening=False,
                id_factory=id_factory,
                max_steps=max_steps,
                max_infra_retries=max_infra_retries,
                context_window=context_window,
                generation_safety_reserve_tokens=generation_safety_reserve_tokens,
                **kwargs,
            )

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_paired_rotation_task_config(task_type, task_config)

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            variant = str(task_config["benchmark_variant"])
            manifest_path = Path(str(task_config["manifest_path"]))
            episode_id = str(task_config["episode_id"])
            return BrowserTrajectoryRunnerV4(
                env_factory=paired_rotation_env_factory(
                    variant=variant,
                    manifest_path=manifest_path,
                    episode_id=episode_id,
                ),
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                task_type=episode_id,
                prompt_builder=build_no_think_runtime_prompt,
                protocol_state_factory=(
                    lambda max_steps: NoThinkRotationProtocolState(
                        max_steps=max_steps
                    )
                ),
                trajectory_format_validator=no_think_trajectory_format_valid,
                response_format_errors=(ResponseFormatError,),
                success_action_kinds=("mouse_up", "drag"),
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
                viewport=(1280, 720),
                non_retryable_errors=(ContextBudgetErrorV4,),
                thread_name_prefix="paired-rotation-no-think",
            )

    @register(WITH_THINK_AGENT_NAME)
    class PairedRotationWithThinkBrowserAgentLoopV1(VerlBrowserAgentLoopV4):
        """Current visible-Think six-action loop over paired rotation replay tasks."""

        def __init__(
            self,
            *args: Any,
            artifact_root: str | Path | None = None,
            slot_pool: Any | None = None,
            id_factory: Any | None = None,
            max_steps: int = MAX_STEPS,
            max_infra_retries: int = 2,
            context_window: int = 16384,
            generation_safety_reserve_tokens: int = 32,
            **kwargs: Any,
        ) -> None:
            def unused_env_factory(seed: int, path: Path) -> Any:
                del seed, path
                raise RuntimeError("paired with-think env factory requires task_config")

            super().__init__(
                *args,
                env_factory=unused_env_factory,
                artifact_root=(
                    artifact_root
                    or os.environ.get(
                        "INTERACTION_ONLINE_WITH_THINK_ARTIFACT_ROOT",
                        "artifacts/runs/paired_rotation_with_think_grpo",
                    )
                ),
                slot_pool=slot_pool or default_browser_slot_pool(),
                prompt_contract=WITH_THINK_PROMPT_CONTRACT,
                prompt_owned_think_opening=True,
                id_factory=id_factory,
                max_steps=max_steps,
                max_infra_retries=max_infra_retries,
                context_window=context_window,
                generation_safety_reserve_tokens=generation_safety_reserve_tokens,
                **kwargs,
            )

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_paired_rotation_task_config(task_type, task_config)

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            variant = str(task_config["benchmark_variant"])
            manifest_path = Path(str(task_config["manifest_path"]))
            episode_id = str(task_config["episode_id"])
            return BrowserTrajectoryRunnerV4(
                env_factory=paired_rotation_env_factory(
                    variant=variant,
                    manifest_path=manifest_path,
                    episode_id=episode_id,
                ),
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                task_type=episode_id,
                prompt_builder=build_with_think_runtime_prompt,
                protocol_state_factory=(
                    lambda max_steps: OnlineProtocolStateV4(max_steps=max_steps)
                ),
                trajectory_format_validator=trajectory_format_valid_v4,
                response_format_errors=(ResponseFormatError,),
                success_action_kinds=("mouse_up",),
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
                viewport=VIEWPORT,
                non_retryable_errors=(ContextBudgetErrorV4,),
                thread_name_prefix="paired-rotation-with-think",
            )
