"""Live first-person AgentLoop using the inner-rotation GRPO reward contract."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ...integrations.browser_runtime import BrowserSlotPool
from ...integrations.browser_trajectory import (
    BrowserTrajectoryRunnerV4,
    ContextBudgetErrorV4,
    EnvFactoryV4,
    PublicBrowserObservationV4,
    TurnGeneratorV4,
)
from ...integrations.verl_browser import VERL_AVAILABLE_V4, VerlBrowserAgentLoopV4
from ...protocol_tracks import split_think_json_response
from ...domains.rotation.action_contract import (
    normalize_prefilled_think_response,
    thought_has_excessive_repetition,
)
from ...integrations.online_rl import LoopViolationV4
from ...train.paired_rotation_browser_agent_loop_v1 import (
    NoThinkRotationProtocolState,
    ResponseFormatError,
    no_think_trajectory_format_valid,
    parse_no_think_rotation_response,
)
from ...train.qwen3_vl_sft import SftPromptBuildResult, build_action_context_from_history
from .first_person_online_dataset import (
    AGENT_NAME,
    EXECUTABLE_ACTION_KINDS,
    FORMAT_ACTION_KINDS,
    INTERACTION_MARKER,
    MAX_STEPS,
    PROMPT_CONTRACT,
    RESPONSE_CONTRACT,
    TASKS,
    VIEWPORT,
)
from .training_no_think import (
    IMAGE_HISTORY_MAX,
    SIX_ACTION_KINDS,
    _no_think_action_context,
    _no_think_action_prompt,
)
from .training_with_think import (
    PROMPT_CONTRACT as WITH_THINK_PROMPT_CONTRACT,
    RESPONSE_CONTRACT as WITH_THINK_RESPONSE_CONTRACT,
    _think_action_prompt,
)
from .third_person_online_dataset import TASKS as THIRD_PERSON_TASKS


WITH_THINK_AGENT_NAME = "exploration_depth_first_person_with_think_online_v1"
WITH_THINK_DATA_SOURCE = "exploration_depth_first_person_online_grpo_withthink_v1"
RUNTIME_TASKS = (*TASKS, *THIRD_PERSON_TASKS)


@dataclass(frozen=True)
class FirstPersonWithThinkResponse:
    thought: str
    action: Any
    normalized_text: str
    restored_prefilled_think: bool
    json_text: str


def _strict_json_object(value: str) -> dict[str, Any]:
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
        raise ResponseFormatError("response must contain exactly one JSON object") from exc
    if not isinstance(payload, dict):
        raise ResponseFormatError("response JSON value must be an object")
    return payload


def _strict_integer_coordinate(value: Any, *, field_name: str) -> None:
    if type(value) is not int or not 0 <= value <= 1000:
        raise ResponseFormatError(f"{field_name} must be an integer within 0..1000")


def _validate_six_action_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != {"action"} or not isinstance(payload["action"], Mapping):
        raise ResponseFormatError("top-level JSON must contain exactly one action object")
    action = payload["action"]
    kind = action.get("kind")
    if kind == "move_to":
        if set(action) != {"kind", "x", "y"}:
            raise ResponseFormatError("move_to requires exactly kind, x, and y")
        _strict_integer_coordinate(action["x"], field_name="move_to.x")
        _strict_integer_coordinate(action["y"], field_name="move_to.y")
        return
    if kind in {"mouse_down", "mouse_up", "left_click"}:
        if set(action) != {"kind"}:
            raise ResponseFormatError(f"{kind} accepts only the kind field")
        return
    if kind in {"click", "drag"}:
        if set(action) != {"kind", "points"}:
            raise ResponseFormatError(f"{kind} requires exactly kind and points")
        points = action["points"]
        expected = 1 if kind == "click" else 2
        if not isinstance(points, list) or len(points) != expected:
            raise ResponseFormatError(f"{kind}.points must contain {expected} point(s)")
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ResponseFormatError(f"{kind}.points[{index}] must be [x, y]")
            _strict_integer_coordinate(
                point[0], field_name=f"{kind}.points[{index}].x"
            )
            _strict_integer_coordinate(
                point[1], field_name=f"{kind}.points[{index}].y"
            )
        return
    raise ResponseFormatError(f"unsupported first-person action kind: {kind!r}")


def parse_first_person_with_think_response(
    response: str,
) -> FirstPersonWithThinkResponse:
    normalized, restored = normalize_prefilled_think_response(response)
    if normalized.count("<think>") != 1 or normalized.count("</think>") != 1:
        raise ResponseFormatError("response must contain exactly one Think block")
    thought, json_text = split_think_json_response(normalized)
    if thought is None or not thought.strip():
        raise ResponseFormatError("reasoning inside <think> must be non-empty")
    if "<think>" in thought or "</think>" in thought:
        raise ResponseFormatError("nested Think tags are forbidden")
    payload = _strict_json_object(json_text)
    _validate_six_action_payload(payload)
    parsed_action = parse_no_think_rotation_response(json_text)
    return FirstPersonWithThinkResponse(
        thought=thought,
        action=parsed_action.action,
        normalized_text=normalized,
        restored_prefilled_think=restored,
        json_text=json_text,
    )


def first_person_with_think_trajectory_format_valid(
    responses: Sequence[str],
) -> bool:
    if not responses:
        return False
    try:
        for response in responses:
            parse_first_person_with_think_response(response)
    except ValueError:
        return False
    return True


@dataclass
class FirstPersonWithThinkProtocolState(NoThinkRotationProtocolState):
    _thoughts: list[str] = field(default_factory=list, init=False)

    @property
    def thought_history(self) -> tuple[str, ...]:
        return tuple(self._thoughts)

    def accept_response(self, response: str) -> FirstPersonWithThinkResponse:
        parsed = parse_first_person_with_think_response(response)
        normalized_thought = " ".join(parsed.thought.split()).casefold()
        if normalized_thought in {
            " ".join(thought.split()).casefold() for thought in self._thoughts
        }:
            raise LoopViolationV4("repeated think is forbidden")
        if thought_has_excessive_repetition(parsed.thought):
            raise LoopViolationV4("repetitive think is forbidden")
        super().accept_response(parsed.json_text)
        self._responses[-1] = parsed.normalized_text
        self._thoughts.append(parsed.thought)
        return parsed


def build_first_person_no_think_runtime_prompt(
    *,
    observations: Sequence[PublicBrowserObservationV4],
    state: NoThinkRotationProtocolState,
    max_steps: int,
    instruction: str,
    task: str,
) -> SftPromptBuildResult:
    """Rebuild the no-Think SFT prompt from the live trajectory history."""

    if task not in RUNTIME_TASKS:
        raise ValueError(f"unsupported non-rotation prompt task: {task!r}")
    if not observations or len(observations) != len(state.action_history) + 1:
        raise ValueError("first-person prompt requires one observation per action plus current")
    if observations[-1].size_px != VIEWPORT:
        raise ValueError("first-person online RL requires a 1280x720 observation")
    if not instruction.strip():
        raise ValueError("first-person online RL requires the episode instruction")

    retained = tuple(observations[-IMAGE_HISTORY_MAX:])
    first_retained_index = len(observations) - len(retained)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _no_think_action_prompt(instruction.strip())}
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
        task_type=task,
        allowed_kinds=SIX_ACTION_KINDS,
        budget_remaining=max(0, max_steps - state.step_count),
    )
    build = SftPromptBuildResult(
        messages=[{"role": "user", "content": content}],
        image_paths=tuple(Path(item.screenshot_path) for item in retained),
        context=context,
        prompt_contract=PROMPT_CONTRACT,
    )
    if any(message.get("role") == "system" for message in build.messages):
        raise RuntimeError("first-person no-Think prompt must not contain a system message")
    return replace(build, prompt_contract=PROMPT_CONTRACT)


def build_first_person_with_think_runtime_prompt(
    *,
    observations: Sequence[PublicBrowserObservationV4],
    state: FirstPersonWithThinkProtocolState,
    max_steps: int,
    instruction: str,
    task: str,
) -> SftPromptBuildResult:
    """Rebuild the visible-Think SFT prompt from the live trajectory history."""

    if task not in RUNTIME_TASKS:
        raise ValueError(f"unsupported non-rotation prompt task: {task!r}")
    if not observations or len(observations) != len(state.action_history) + 1:
        raise ValueError("first-person prompt requires one observation per action plus current")
    if observations[-1].size_px != VIEWPORT:
        raise ValueError("first-person online RL requires a 1280x720 observation")
    if not instruction.strip():
        raise ValueError("first-person online RL requires the episode instruction")

    retained = tuple(observations[-IMAGE_HISTORY_MAX:])
    first_retained_index = len(observations) - len(retained)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _think_action_prompt(instruction.strip())}
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
        task_type=task,
        allowed_kinds=SIX_ACTION_KINDS,
        budget_remaining=max(0, max_steps - state.step_count),
    )
    return SftPromptBuildResult(
        messages=[{"role": "user", "content": content}],
        image_paths=tuple(Path(item.screenshot_path) for item in retained),
        context=context,
        prompt_contract=WITH_THINK_PROMPT_CONTRACT,
    )


@lru_cache(maxsize=8)
def _episode_index(manifest_path: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gui_captcha_exploration_depth_manifest_v1":
        raise ValueError(f"unsupported first-person manifest: {path}")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_episode in payload.get("episodes", []):
        if not isinstance(raw_episode, dict):
            continue
        task = str(raw_episode.get("variant") or "")
        if task not in RUNTIME_TASKS:
            continue
        episode = dict(raw_episode)
        episode_id = str(episode["episode_id"])
        pair_id = str(episode["pair_id"])
        index[(task, episode_id)] = episode
        index[(task, pair_id)] = episode
    return index


def normalize_first_person_task_config(
    task_type: Any,
    task_config: Any,
) -> dict[str, Any]:
    if hasattr(task_config, "item") and not isinstance(task_config, Mapping):
        task_config = task_config.item()
    if not isinstance(task_config, Mapping):
        raise ValueError("first-person online row is missing task_config")
    normalized = dict(task_config)
    if normalized.get("background_pool") is None:
        normalized.pop("background_pool", None)
    required = {
        "suite_id",
        "manifest_path",
        "benchmark_variant",
        "episode_id",
        "viewport",
        "max_steps",
        "coordinate_format",
        "format_action_kinds",
        "executable_action_kinds",
        "prompt_contract",
        "response_contract",
    }
    if set(normalized) != required:
        raise ValueError(
            f"first-person task_config keys must be exactly {sorted(required)!r}"
        )
    task = str(normalized["benchmark_variant"])
    if task not in RUNTIME_TASKS or task_type not in {None, task}:
        raise ValueError("non-rotation task_type must match its benchmark variant")
    episode_id = str(normalized["episode_id"])
    if not episode_id.endswith(f"--{task.replace('_', '-')}"):
        raise ValueError("first-person episode ID does not match its variant")
    manifest_path = Path(str(normalized["manifest_path"]))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("first-person manifest_path must be an existing absolute file")
    try:
        episode = _episode_index(str(manifest_path.resolve()))[(task, episode_id)]
    except KeyError as exc:
        raise ValueError("first-person task row selects an unknown episode") from exc
    if normalized["suite_id"] != episode.get("suite_id"):
        raise ValueError("first-person suite ID differs from its manifest episode")
    environment = episode.get("environment_config")
    if not isinstance(environment, Mapping):
        raise ValueError("first-person manifest episode has no environment config")
    if environment.get("interaction_marker") != INTERACTION_MARKER:
        raise ValueError(
            "first-person RL requires the SFT-matched mouse_icon marker; "
            f"got {environment.get('interaction_marker')!r}"
        )
    if list(normalized["viewport"]) != list(VIEWPORT):
        raise ValueError("first-person viewport must be exactly 1280x720")
    if normalized["max_steps"] != MAX_STEPS:
        raise ValueError(f"first-person max_steps must be exactly {MAX_STEPS}")
    if normalized["coordinate_format"] != "qwen_relative_0_1000":
        raise ValueError("first-person coordinate format must use Qwen relative bins")
    if tuple(normalized["format_action_kinds"]) != FORMAT_ACTION_KINDS:
        raise ValueError("first-person format action grammar is invalid")
    if tuple(normalized["executable_action_kinds"]) != EXECUTABLE_ACTION_KINDS:
        raise ValueError("first-person executable action grammar is invalid")
    if normalized["prompt_contract"] != PROMPT_CONTRACT:
        raise ValueError("first-person prompt contract is invalid")
    if normalized["response_contract"] != RESPONSE_CONTRACT:
        raise ValueError("first-person response contract is invalid")
    return normalized


def normalize_first_person_with_think_task_config(
    task_type: Any,
    task_config: Any,
) -> dict[str, Any]:
    if not isinstance(task_config, Mapping):
        if hasattr(task_config, "item"):
            task_config = task_config.item()
    if not isinstance(task_config, Mapping):
        raise ValueError("first-person online row is missing task_config")
    normalized = dict(task_config)
    if normalized.get("prompt_contract") != WITH_THINK_PROMPT_CONTRACT:
        raise ValueError("first-person with-Think prompt contract is invalid")
    if normalized.get("response_contract") != WITH_THINK_RESPONSE_CONTRACT:
        raise ValueError("first-person with-Think response contract is invalid")
    compatibility = dict(normalized)
    compatibility["prompt_contract"] = PROMPT_CONTRACT
    compatibility["response_contract"] = RESPONSE_CONTRACT
    validated = normalize_first_person_task_config(task_type, compatibility)
    validated["prompt_contract"] = WITH_THINK_PROMPT_CONTRACT
    validated["response_contract"] = WITH_THINK_RESPONSE_CONTRACT
    return validated


def first_person_env_factory(
    *,
    task: str,
    manifest_path: Path,
    episode_id: str,
) -> EnvFactoryV4:
    episode = _episode_index(str(manifest_path.resolve()))[(task, episode_id)]
    environment = episode.get("environment_config")
    if not isinstance(environment, Mapping) or environment.get(
        "interaction_marker"
    ) != INTERACTION_MARKER:
        raise ValueError(
            "non-rotation RL environment requires the SFT-matched mouse_icon marker"
        )

    def create(_seed: int, artifact_dir: Path) -> Any:
        if task == "ten_choice_first_person":
            from ...domains.ten_choice.first_person import FirstPersonTenChoiceEnv

            env = FirstPersonTenChoiceEnv(
                manifest_path=manifest_path,
                artifact_dir=artifact_dir,
                enable_playwright=True,
                headless=os.environ.get("EXPLORATION_DEPTH_BROWSER_HEADLESS", "1")
                != "0",
                viewport_width=VIEWPORT[0],
                viewport_height=VIEWPORT[1],
                navigation_timeout_ms=int(
                    os.environ.get("EXPLORATION_DEPTH_NAVIGATION_TIMEOUT_MS", "60000")
                ),
            )
        elif task == "drag_first_person":
            from .drag import PairedFirstPersonDragEnv

            env = PairedFirstPersonDragEnv(
                manifest_path=manifest_path,
                artifact_dir=artifact_dir,
                viewport_width=VIEWPORT[0],
                viewport_height=VIEWPORT[1],
            )
        elif task == "ten_choice_third_person":
            from ...domains.ten_choice.first_person import PairedHoverRevealEnv

            env = PairedHoverRevealEnv(
                manifest_path=manifest_path,
                artifact_dir=artifact_dir,
                enable_playwright=True,
                viewport_width=VIEWPORT[0],
                viewport_height=VIEWPORT[1],
                navigation_timeout_ms=int(
                    os.environ.get("EXPLORATION_DEPTH_NAVIGATION_TIMEOUT_MS", "60000")
                ),
            )
        elif task == "drag_third_person":
            from .drag import PairedThirdPersonDragEnv

            env = PairedThirdPersonDragEnv(
                manifest_path=manifest_path,
                artifact_dir=artifact_dir,
                viewport_width=VIEWPORT[0],
                viewport_height=VIEWPORT[1],
            )
        else:  # pragma: no cover - guarded before the factory is built
            raise ValueError(f"unsupported non-rotation task: {task!r}")
        env._paired_manifest_cache = {
            "schema": "gui_captcha_exploration_depth_manifest_v1",
            "episodes": [episode],
        }
        return env

    return create


def preflight_first_person_task_config(
    task_type: str,
    task_config: Mapping[str, Any],
    *,
    artifact_dir: Path,
) -> Path:
    normalized = normalize_first_person_task_config(task_type, task_config)
    env = first_person_env_factory(
        task=str(normalized["benchmark_variant"]),
        manifest_path=Path(str(normalized["manifest_path"])),
        episode_id=str(normalized["episode_id"]),
    )(0, Path(artifact_dir))
    try:
        observation = env.reset(
            task_type=str(normalized["episode_id"]),
            task_id=str(normalized["episode_id"]),
        )
        screenshot = Path(observation.screenshot_path)
        if observation.size_px != VIEWPORT or not screenshot.is_file():
            raise RuntimeError("first-person preflight did not produce a 1280x720 frame")
        _assert_mouse_marker_frame(str(normalized["benchmark_variant"]), screenshot)
        if normalized["benchmark_variant"] in {
            "ten_choice_first_person",
            "ten_choice_third_person",
        }:
            if env.page is None:
                raise RuntimeError("ten-choice preflight did not start Playwright")
            page_audit = env.page.evaluate(
                """() => ({
                    bodyText: document.body.innerText || '',
                    iconCount: document.querySelectorAll('.icon-button').length,
                    loadedImageCount: [...document.images].filter(
                        image => image.complete && image.naturalWidth > 0
                    ).length,
                    interactionMarker: document.body.dataset.interactionMarker || '',
                })"""
            )
            if not isinstance(page_audit, Mapping):
                raise RuntimeError("ten-choice preflight returned no page audit")
            if (
                len(str(page_audit.get("bodyText") or "").strip()) < 20
                or page_audit.get("iconCount") != 10
                or page_audit.get("loadedImageCount") != 10
                or page_audit.get("interactionMarker") != INTERACTION_MARKER
            ):
                raise RuntimeError(
                    f"ten-choice page content/marker preflight failed: {page_audit}"
                )
            _assert_ten_choice_header_rendered(screenshot)
        return screenshot
    finally:
        env.close()


def preflight_first_person_with_think_task_config(
    task_type: str,
    task_config: Mapping[str, Any],
    *,
    artifact_dir: Path,
) -> Path:
    normalized = normalize_first_person_with_think_task_config(task_type, task_config)
    compatibility = dict(normalized)
    compatibility["prompt_contract"] = PROMPT_CONTRACT
    compatibility["response_contract"] = RESPONSE_CONTRACT
    return preflight_first_person_task_config(
        task_type,
        compatibility,
        artifact_dir=artifact_dir,
    )


def _assert_mouse_marker_frame(task: str, screenshot: Path) -> None:
    """Fail closed if the center marker is absent or is the legacy red dot."""

    from PIL import Image

    expected_pixels = {
        "ten_choice_first_person": {
            (640, 360): (0, 0, 0),
            (641, 361): (150, 150, 150),
            (642, 364): (150, 150, 150),
        },
        "ten_choice_third_person": {
            (640, 360): (0, 0, 0),
            (641, 361): (150, 150, 150),
            (642, 364): (150, 150, 150),
        },
        "drag_first_person": {
            (640, 360): (15, 23, 42),
            (642, 364): (248, 250, 252),
            (655, 385): (15, 23, 42),
        },
        "drag_third_person": {
            (640, 360): (15, 23, 42),
            (642, 364): (248, 250, 252),
            (655, 385): (15, 23, 42),
        },
    }
    try:
        expected = expected_pixels[task]
    except KeyError as exc:  # pragma: no cover - normalized before this helper
        raise ValueError(f"unsupported non-rotation task: {task!r}") from exc
    with Image.open(screenshot) as source:
        image = source.convert("RGB")
        if image.size != VIEWPORT:
            raise RuntimeError(f"non-rotation frame has unexpected size: {image.size}")
        actual = {xy: image.getpixel(xy) for xy in expected}
        if actual != expected:
            raise RuntimeError(
                "non-rotation frame does not contain the required center mouse icon; "
                f"expected {expected}, got {actual}"
            )


def _assert_ten_choice_header_rendered(screenshot: Path) -> None:
    from PIL import Image

    with Image.open(screenshot) as source:
        header = source.convert("RGB").crop((0, 0, VIEWPORT[0], 100))
        dark_pixels = sum(
            min(header.getpixel((x, y))) < 220
            for y in range(header.height)
            for x in range(header.width)
        )
    if dark_pixels < 500:
        raise RuntimeError(
            "ten-choice frame header appears blank; font/content rendering failed"
        )


def _artifact_root(value: str | Path | None) -> Path:
    raw = value or os.environ.get("EXPLORATION_DEPTH_ONLINE_RL_ARTIFACT_ROOT")
    if not raw:
        raise ValueError("EXPLORATION_DEPTH_ONLINE_RL_ARTIFACT_ROOT is required")
    return Path(raw)


def _success_action_kinds(task: str) -> tuple[str, ...]:
    if task.startswith("ten_choice_"):
        return ("mouse_up", "left_click", "click")
    if task.startswith("drag_"):
        return ("mouse_up", "drag")
    raise ValueError(f"unsupported non-rotation task: {task!r}")


def _prepend_env_path_once(name: str, value: str | Path) -> None:
    """Put one path first without growing a process-global environment value."""

    prefix = str(value)
    current = [
        part
        for part in os.environ.get(name, "").split(os.pathsep)
        if part and part != prefix
    ]
    os.environ[name] = os.pathsep.join((prefix, *current))


def _runtime_path(
    explicit: str | Path | None,
    env_name: str,
    fallback: Path,
) -> Path:
    return Path(explicit or os.environ.get(env_name) or fallback)


if not VERL_AVAILABLE_V4:

    class ExplorationDepthFirstPersonNoThinkAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "ExplorationDepthFirstPersonNoThinkAgentLoopV1 requires VERL"
            )

    class ExplorationDepthFirstPersonWithThinkAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "ExplorationDepthFirstPersonWithThinkAgentLoopV1 requires VERL"
            )
else:
    from verl.experimental.agent_loop.agent_loop import register

    @register(AGENT_NAME)
    class ExplorationDepthFirstPersonNoThinkAgentLoopV1(VerlBrowserAgentLoopV4):
        """No-Think six-action loop for first-person ten-choice and drag."""

        def __init__(
            self,
            *args: Any,
            artifact_root: str | Path | None = None,
            storage_root: str | Path | None = None,
            browser_slot_dir: str | Path | None = None,
            browser_concurrency: int = 4,
            playwright_browsers_path: str | Path | None = None,
            playwright_library_path: str | Path | None = None,
            fontconfig_file: str | Path | None = None,
            slot_pool: BrowserSlotPool | None = None,
            max_steps: int = MAX_STEPS,
            max_infra_retries: int = 2,
            context_window: int = 16384,
            generation_safety_reserve_tokens: int = 32,
            **kwargs: Any,
        ) -> None:
            resolved_artifact_root = _artifact_root(artifact_root)
            repo_root = Path(__file__).resolve().parents[4]
            resolved_storage_root = _runtime_path(
                storage_root,
                "GUI_CAPTCHA_STORAGE_ROOT",
                repo_root,
            )
            resolved_playwright_browsers = _runtime_path(
                playwright_browsers_path,
                "PLAYWRIGHT_BROWSERS_PATH",
                Path.home() / ".cache/ms-playwright",
            )
            resolved_playwright_libraries = _runtime_path(
                playwright_library_path,
                "EXPLORATION_DEPTH_PLAYWRIGHT_LIBRARY_PATH",
                repo_root / "artifacts/playwright-libs",
            )
            resolved_fontconfig = _runtime_path(
                fontconfig_file,
                "FONTCONFIG_FILE",
                Path("/etc/fonts/fonts.conf"),
            )
            os.environ["GUI_CAPTCHA_STORAGE_ROOT"] = str(resolved_storage_root)
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(
                resolved_playwright_browsers
            )
            _prepend_env_path_once(
                "LD_LIBRARY_PATH",
                resolved_playwright_libraries,
            )
            os.environ["FONTCONFIG_FILE"] = str(resolved_fontconfig)
            os.environ["IMAGE_MAX_PIXELS"] = "921600"

            def task_owned_factory(_seed: int, _path: Path) -> Any:
                raise RuntimeError("first-person env factory requires task_config")

            super().__init__(
                *args,
                env_factory=task_owned_factory,
                artifact_root=resolved_artifact_root,
                slot_pool=slot_pool
                or BrowserSlotPool(
                    root=Path(
                        browser_slot_dir
                        or os.environ.get("EXPLORATION_DEPTH_BROWSER_SLOT_DIR")
                        or Path("/tmp")
                        / f"{resolved_artifact_root.parent.name}-browser-slots"
                    ),
                    capacity=int(browser_concurrency),
                    acquire_timeout_s=float(
                        os.environ.get("EXPLORATION_DEPTH_BROWSER_SLOT_TIMEOUT_S", "300")
                    ),
                ),
                prompt_contract=PROMPT_CONTRACT,
                prompt_owned_think_opening=False,
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
            return normalize_first_person_task_config(task_type, task_config)

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            task = str(task_config["benchmark_variant"])
            manifest_path = Path(str(task_config["manifest_path"]))
            episode_id = str(task_config["episode_id"])
            episode = _episode_index(str(manifest_path.resolve()))[(task, episode_id)]

            def prompt_builder(**kwargs: Any) -> SftPromptBuildResult:
                return build_first_person_no_think_runtime_prompt(
                    **kwargs,
                    instruction=str(episode["instruction"]),
                    task=task,
                )

            success_action_kinds = _success_action_kinds(task)
            return BrowserTrajectoryRunnerV4(
                env_factory=first_person_env_factory(
                    task=task,
                    manifest_path=manifest_path,
                    episode_id=episode_id,
                ),
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                task_type=episode_id,
                prompt_builder=prompt_builder,
                protocol_state_factory=lambda max_steps: NoThinkRotationProtocolState(
                    max_steps=max_steps
                ),
                trajectory_format_validator=no_think_trajectory_format_valid,
                response_format_errors=(ResponseFormatError,),
                success_action_kinds=success_action_kinds,
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
                viewport=VIEWPORT,
                non_retryable_errors=(ContextBudgetErrorV4,),
                thread_name_prefix=f"{task}-no-think-online",
            )

    @register(WITH_THINK_AGENT_NAME)
    class ExplorationDepthFirstPersonWithThinkAgentLoopV1(
        ExplorationDepthFirstPersonNoThinkAgentLoopV1
    ):
        """Visible-Think six-action loop for first-person ten-choice and drag."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.prompt_contract = WITH_THINK_PROMPT_CONTRACT
            self.prompt_owned_think_opening = True

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_first_person_with_think_task_config(
                task_type, task_config
            )

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            task = str(task_config["benchmark_variant"])
            manifest_path = Path(str(task_config["manifest_path"]))
            episode_id = str(task_config["episode_id"])
            episode = _episode_index(str(manifest_path.resolve()))[(task, episode_id)]

            def prompt_builder(**kwargs: Any) -> SftPromptBuildResult:
                return build_first_person_with_think_runtime_prompt(
                    **kwargs,
                    instruction=str(episode["instruction"]),
                    task=task,
                )

            success_action_kinds = _success_action_kinds(task)
            return BrowserTrajectoryRunnerV4(
                env_factory=first_person_env_factory(
                    task=task,
                    manifest_path=manifest_path,
                    episode_id=episode_id,
                ),
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                task_type=episode_id,
                prompt_builder=prompt_builder,
                protocol_state_factory=lambda max_steps: FirstPersonWithThinkProtocolState(
                    max_steps=max_steps
                ),
                trajectory_format_validator=(
                    first_person_with_think_trajectory_format_valid
                ),
                response_format_errors=(ResponseFormatError,),
                success_action_kinds=success_action_kinds,
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
                viewport=VIEWPORT,
                non_retryable_errors=(ContextBudgetErrorV4,),
                thread_name_prefix=f"{task}-with-think-online",
            )
