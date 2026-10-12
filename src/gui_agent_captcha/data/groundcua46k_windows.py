"""Normalize GroundCUA 46k trajectories into complete multi-turn SFT windows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from gui_agent_captcha.prompts.screenspot_pro_groundcua import (
    QWEN25_DIRECT_PROFILE,
    QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE,
    format_groundcua_think_tool_call,
    get_groundcua_profile,
    groundcua_system_prompt,
    parse_groundcua_think_tool_call,
    require_official_qwen25_direct_click,
)
from gui_agent_captcha.prompts.qwen3_gui_agent_multiturn import (
    QWEN3_GUI_AGENT_MULTITURN_PROFILE,
    QWEN3_GUI_AGENT_MULTITURN_SYSTEM_PROMPT,
    qwen3_gui_agent_multiturn_user_text,
)
from gui_agent_captcha.train.qwen3_vl_sft import resized_size_for_pixel_range


_MODEL_FAMILY_ALIASES = {
    "qwen3": "qwen3_vl",
    "qwen3_vl": "qwen3_vl",
    "qwen3-vl": "qwen3_vl",
    "qwen2.5": "qwen2_5_vl",
    "qwen2_5": "qwen2_5_vl",
    "qwen2_5_vl": "qwen2_5_vl",
    "qwen2.5-vl": "qwen2_5_vl",
}
_PROFILE_BY_MODEL_FAMILY = {
    "qwen3_vl": QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE,
}


@dataclass(frozen=True)
class NormalizedAction:
    action_index: int
    action: str
    coordinate: tuple[int, int] | None
    thought: str
    assistant_response: str
    image_path: Path
    source_action_indices: tuple[int, ...]
    source_kinds: tuple[str, ...]
    model_image_size: tuple[int, int]


@dataclass(frozen=True)
class NormalizedTrajectory:
    source_record_id: str
    instruction: str
    model_family: str
    prompt_profile: str
    actions: tuple[NormalizedAction, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WindowExample:
    messages: list[dict[str, Any]]
    image_paths: tuple[Path, ...]
    target_action_indices: tuple[int, ...]
    source_record_id: str
    window_index: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _SourceAction:
    position: int
    kind: str
    x: int | None
    y: int | None
    thought: str
    image_path: Path


def _canonical_model_family(model_family: str) -> str:
    if not isinstance(model_family, str):
        raise ValueError("model_family must be qwen3_vl or qwen2_5_vl")
    try:
        return _MODEL_FAMILY_ALIASES[model_family.strip().lower()]
    except KeyError as exc:
        raise ValueError("model_family must be qwen3_vl or qwen2_5_vl") from exc


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _source_image_size(record: Mapping[str, Any], image_path: Path) -> tuple[int, int]:
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping):
        width = metadata.get("image_width")
        height = metadata.get("image_height")
        if (
            isinstance(width, int)
            and not isinstance(width, bool)
            and width > 0
            and isinstance(height, int)
            and not isinstance(height, bool)
            and height > 0
        ):
            return width, height
    if not image_path.is_file():
        raise FileNotFoundError(f"GroundCUA image not found: {image_path}")
    with Image.open(image_path) as image:
        return int(image.width), int(image.height)


def _resolve_source_image_path(raw_path: str) -> Path:
    """Resolve source images after the governed GroundCUA storage migration.

    Older JSONL rows contain absolute paths below ``datasets/groundcua`` while
    the read-only 46k assets now live below ``datasets/groundcua/46k``.  The
    original path remains the first choice; the fallback is purely a path
    lookup and never copies or mutates source data.
    """

    original = Path(raw_path)
    if original.is_file():
        return original
    marker = "/artifacts/datasets/groundcua/"
    text = str(original)
    if marker in text:
        prefix, suffix = text.split(marker, 1)
        if suffix.startswith("46k/"):
            return original
        relocated = Path(prefix + marker + "46k/" + suffix)
        if relocated.is_file():
            return relocated
    return original


def _canonical_action(
    step: Mapping[str, Any], *, position: int
) -> tuple[str, tuple[int, int] | None]:
    forbidden = {"kind", "x", "y"} & set(step)
    if forbidden:
        raise ValueError(
            f"legacy action fields are forbidden at source action {position}: {sorted(forbidden)}"
        )
    tool_call = step.get("tool_call")
    if not isinstance(tool_call, Mapping) or set(tool_call) != {"name", "arguments"}:
        raise ValueError(f"source action {position} requires exactly one canonical tool_call")
    if tool_call.get("name") != "computer_use":
        raise ValueError(f"source action {position} tool_call name must be computer_use")
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"source action {position} tool_call arguments must be an object")
    action = arguments.get("action")
    if action in {"move_to", "mouse_move"}:
        coordinate = arguments.get("coordinate")
        if (
            set(arguments) != {"action", "coordinate"}
            or not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
            or any(value < 0 for value in coordinate)
            or (action == "move_to" and any(value > 1000 for value in coordinate))
        ):
            raise ValueError(f"source action {position} has invalid {action} coordinate")
        return str(action), (coordinate[0], coordinate[1])
    if action == "left_click":
        if set(arguments) != {"action"}:
            raise ValueError(f"source action {position} left_click must not have coordinate")
        return "left_click", None
    raise ValueError(
        f"source action {position} has unsupported computer_use action {action!r}"
    )


def _canonical_action_response(
    step: Mapping[str, Any], *, position: int, profile_name: str
) -> tuple[str, str, tuple[int, int] | None]:
    response = step.get("assistant_response")
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"source action {position} is missing assistant_response")
    try:
        return parse_groundcua_think_tool_call(profile_name, response)
    except ValueError as exc:
        raise ValueError(
            f"source action {position} has malformed canonical teacher response"
        ) from exc


def _read_source_actions(
    record: Mapping[str, Any], *, profile_name: str, action_image_timing: str
) -> tuple[_SourceAction, ...]:
    if action_image_timing not in {"action_before", "action_after"}:
        raise ValueError("action_image_timing must be action_before or action_after")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("GroundCUA trajectory requires nonempty steps")
    if len(steps) % 2 == 0:
        raise ValueError("GroundCUA steps must alternate observation/action and end in observation")
    source_actions: list[_SourceAction] = []
    for index, step in enumerate(steps):
        expected_type = "observation" if index % 2 == 0 else "action"
        if not isinstance(step, Mapping) or step.get("type") != expected_type:
            raise ValueError(
                f"GroundCUA step {index} must be an alternating {expected_type} step"
            )
        if expected_type == "observation":
            image_path = step.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise ValueError(f"GroundCUA observation {index} is missing image_path")
            continue

        position = len(source_actions) + 1
        kind, coordinate = _canonical_action(step, position=position)
        thought, parsed_kind, parsed_coordinate = _canonical_action_response(
            step,
            position=position,
            profile_name=profile_name,
        )
        if (parsed_kind, parsed_coordinate) != (kind, coordinate):
            raise ValueError(
                f"source action {position} disagrees with its canonical teacher response"
            )
        x, y = coordinate if coordinate is not None else (None, None)
        action_observation = (
            steps[index + 1]
            if action_image_timing == "action_after"
            else steps[index - 1]
        )
        if action_image_timing == "action_after":
            if action_observation.get("observation_timing") != "action_after":
                raise ValueError(
                    f"source action {position} requires an action-after observation"
                )
            if action_observation.get("cursor_renderer_version") != "solid_red_pointer_v1":
                raise ValueError(
                    f"source action {position} requires the solid red cursor renderer"
                )
        source_actions.append(
            _SourceAction(
                position=position,
                kind=str(kind),
                x=x,
                y=y,
                thought=thought,
                image_path=_resolve_source_image_path(str(action_observation["image_path"])),
            )
        )
    return tuple(source_actions)


def _half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _source_id_split(record_id: str) -> str:
    """Return an ID's legacy source bucket for provenance only.

    The bucket is never used to select rows: every source trajectory is emitted
    as ``dataset_split=train`` for the requested SFT run.
    """

    marker = "groundcua_"
    if marker in record_id:
        suffix = record_id.split(marker, 1)[1]
        bucket = suffix.split("_", 1)[0]
        if bucket in {"train", "val", "test"}:
            return bucket
    return "unknown"


def _visual_tokens_lower_bound(
    model_size: tuple[int, int],
    *,
    image_factor: int,
) -> int:
    """Estimate visual tokens from resized dimensions, excluding text tokens.

    Qwen vision encoders may add implementation-specific merges, so this is a
    conservative geometry lower bound rather than a processor-token count.
    """

    width, height = model_size
    return (width // image_factor) * (height // image_factor)


def _qwen25_coordinate(
    x: int,
    y: int,
    *,
    model_size: tuple[int, int],
) -> tuple[int, int]:
    # GroundCUA source coordinates are already Qwen3's relative 0..1000
    # values.  Map them directly into the factor-28 resized physical image.
    width, height = model_size
    return (
        _half_up(Decimal(x) * Decimal(width) / Decimal(1000)),
        _half_up(Decimal(y) * Decimal(height) / Decimal(1000)),
    )


def normalize_groundcua46k_trajectory(
    record: Mapping[str, Any],
    *,
    model_family: str,
    action_image_timing: str = "action_before",
    prompt_profile: str | None = None,
    include_hash_metadata: bool = True,
) -> NormalizedTrajectory:
    """Normalize one read-only source trajectory for exactly one model family."""

    if not isinstance(record, Mapping):
        raise ValueError("GroundCUA trajectory must be an object")
    family = _canonical_model_family(model_family)
    if family == "qwen2_5_vl":
        require_official_qwen25_direct_click(
            prompt_profile or QWEN25_DIRECT_PROFILE,
            action_contract="mouse_move_then_coordinate_free_left_click",
            multi_turn=True,
            has_think=True,
            preserves_history=True,
        )
    record_id = record.get("id")
    instruction = record.get("instruction")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("GroundCUA trajectory id is required")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("GroundCUA instruction must be nonempty")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("GroundCUA trajectory metadata is required")
    if metadata.get("coordinate_format") != "qwen3_relative_0_1000":
        raise ValueError("source trajectory must use qwen3_relative_0_1000 coordinates")

    profile_name = _PROFILE_BY_MODEL_FAMILY[family] if prompt_profile is None else prompt_profile
    if not isinstance(profile_name, str) or not profile_name:
        raise ValueError("prompt_profile must be a nonempty GroundCUA profile name")
    profile = get_groundcua_profile(profile_name)
    if profile.model_family != family:
        raise ValueError("prompt_profile model family does not match model_family")
    source_actions = _read_source_actions(
        record,
        profile_name=profile_name,
        action_image_timing=action_image_timing,
    )
    normalized: list[NormalizedAction] = []
    source_index = 0
    while source_index < len(source_actions):
        source = source_actions[source_index]
        source_size = _source_image_size(record, source.image_path)
        model_size = resized_size_for_pixel_range(
            source_size,
            image_min_pixels=profile.image_min_pixels,
            image_max_pixels=profile.image_max_pixels,
            size_factor=profile.image_factor,
        )
        if source.kind in {"move_to", "mouse_move"}:
            assert source.x is not None and source.y is not None
            if family == "qwen3_vl":
                if source.kind != "move_to":
                    raise ValueError("Qwen3 source action must use move_to")
                action = "move_to"
                coordinate = (source.x, source.y)
            else:
                if source.kind != "mouse_move":
                    raise ValueError("Qwen2.5 source action must use mouse_move")
                action = "mouse_move"
                coordinate = _qwen25_coordinate(
                    source.x,
                    source.y,
                    model_size=model_size,
                )
            response = format_groundcua_think_tool_call(
                profile_name,
                source.thought,
                action,
                coordinate,
            )
            source_indices = (source.position,)
            source_kinds = (source.kind,)
            source_index += 1
        elif source.kind == "left_click":
            action = "left_click"
            coordinate = None
            response = format_groundcua_think_tool_call(
                profile_name,
                source.thought,
                action,
                None,
            )
            source_indices = (source.position,)
            source_kinds = (source.kind,)
            source_index += 1
        else:
            raise ValueError(f"unsupported canonical source action {source.kind!r}")

        parsed_thought, parsed_action, parsed_coordinate = parse_groundcua_think_tool_call(
            profile_name,
            response,
        )
        if (parsed_thought, parsed_action, parsed_coordinate) != (
            source.thought,
            action,
            coordinate,
        ):
            raise RuntimeError("normalized GroundCUA action failed round-trip validation")
        normalized.append(
            NormalizedAction(
                action_index=len(normalized) + 1,
                action=action,
                coordinate=coordinate,
                thought=source.thought,
                assistant_response=response,
                image_path=source.image_path,
                source_action_indices=source_indices,
                source_kinds=source_kinds,
                model_image_size=model_size,
            )
        )

    if not normalized or normalized[-1].action != "left_click":
        raise ValueError("normalized GroundCUA trajectory must end in left_click")
    if any(action.action == "left_click" for action in normalized[:-1]):
        raise ValueError("normalized GroundCUA trajectory may contain only one terminal click")
    normalized_metadata = {
        "source_record_id": record_id,
        "source_id_split": _source_id_split(record_id),
        "dataset_split": "train",
        "source_action_count": len(source_actions),
        "normalized_action_count": len(normalized),
        "model_family": family,
        "prompt_profile": profile_name,
        "action_image_timing": action_image_timing,
        "cursor_renderer_version": (
            metadata.get("cursor_renderer_version")
            if action_image_timing == "action_after"
            else None
        ),
        "coordinate_format": profile.coordinate_format,
        "image_factor": profile.image_factor,
        "image_min_pixels": profile.image_min_pixels,
        "image_max_pixels": profile.image_max_pixels,
        "visual_token_lower_bound_is_estimate": True,
        "visual_token_lower_bound_per_action": [
            _visual_tokens_lower_bound(
                action.model_image_size,
                image_factor=profile.image_factor,
            )
            for action in normalized
        ],
    }
    if include_hash_metadata:
        normalized_metadata["terminal_click_thought_sha256"] = hashlib.sha256(
            normalized[-1].thought.encode("utf-8")
        ).hexdigest()
    return NormalizedTrajectory(
        source_record_id=record_id,
        instruction=instruction.strip(),
        model_family=family,
        prompt_profile=profile_name,
        actions=tuple(normalized),
        metadata=normalized_metadata,
    )


def _user_content(
    instruction: str,
    image_path: Path | None,
    *,
    prompt_variant: str | None = None,
    round_index: int | None = None,
) -> list[dict[str, str]]:
    content: list[dict[str, str]] = []
    if image_path is not None:
        content.append({"type": "image", "image": str(image_path)})
    if prompt_variant == QWEN3_GUI_AGENT_MULTITURN_PROFILE:
        if round_index is None:
            raise ValueError("multi-turn prompt rows require round_index")
        text = qwen3_gui_agent_multiturn_user_text(
            instruction,
            round_index=round_index,
        )
    elif prompt_variant is not None:
        raise ValueError(f"unsupported GroundCUA prompt variant: {prompt_variant!r}")
    else:
        text = instruction
    content.append({"type": "text", "text": text})
    return content


def build_groundcua46k_windows(
    trajectory: NormalizedTrajectory,
    *,
    model_family: str,
    images_to_keep: int = 3,
    prompt_variant: str | None = None,
) -> list[WindowExample]:
    """Build complete conversations with all actions and the latest 3 images."""

    family = _canonical_model_family(model_family)
    if not isinstance(trajectory, NormalizedTrajectory):
        raise ValueError("trajectory must be a NormalizedTrajectory")
    if family != trajectory.model_family:
        raise ValueError("model_family does not match normalized trajectory")
    if family == "qwen2_5_vl":
        require_official_qwen25_direct_click(
            trajectory.prompt_profile,
            action_contract="mouse_move_then_coordinate_free_left_click",
            multi_turn=True,
            has_think=True,
            preserves_history=True,
        )
    if images_to_keep != 3:
        raise ValueError("GroundCUA 46k windows require images_to_keep=3")
    if prompt_variant not in {None, QWEN3_GUI_AGENT_MULTITURN_PROFILE}:
        raise ValueError(f"unsupported GroundCUA prompt variant: {prompt_variant!r}")
    if prompt_variant is not None and family != "qwen3_vl":
        raise ValueError("qwen3 GUI-agent prompt variant requires model_family=qwen3_vl")
    actions = trajectory.actions
    if not actions:
        raise ValueError("normalized trajectory has no actions")

    if prompt_variant == QWEN3_GUI_AGENT_MULTITURN_PROFILE:
        # The prompt exposes left_click and terminate only.  Keep every
        # move_to response as causal history, but supervise the terminal click
        # that is valid under the prompt's tool contract.
        endpoints = [len(actions)]
    else:
        # Up to three normalized actions fit in the first row.  Each later
        # action adds one sliding-window row while retaining every earlier
        # action as text.
        endpoints = [min(images_to_keep, len(actions))]
        endpoints.extend(range(images_to_keep + 1, len(actions) + 1))
    profile = get_groundcua_profile(trajectory.prompt_profile)
    windows: list[WindowExample] = []
    for window_index, endpoint in enumerate(endpoints):
        retained_start = max(1, endpoint - images_to_keep + 1)
        current = actions[endpoint - 1]
        screen_width, screen_height = current.model_image_size
        if prompt_variant == QWEN3_GUI_AGENT_MULTITURN_PROFILE:
            system = QWEN3_GUI_AGENT_MULTITURN_SYSTEM_PROMPT
            target_indices = (endpoint,)
            target_policy = "terminal_left_click_only"
            row_schema_version = 3
        else:
            system = groundcua_system_prompt(
                profile.name,
                screen_width=screen_width if family == "qwen2_5_vl" else None,
                screen_height=screen_height if family == "qwen2_5_vl" else None,
            )
            target_indices = (
                tuple(range(1, endpoint + 1))
                if window_index == 0
                else (endpoint,)
            )
            target_policy = "first_window_all_then_new_action_only"
            row_schema_version = 2
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        retained_images: list[Path] = []
        for action in actions[:endpoint]:
            image_path = action.image_path if action.action_index >= retained_start else None
            if image_path is not None:
                retained_images.append(image_path)
            messages.append(
                {
                    "role": "user",
                    "content": _user_content(
                        trajectory.instruction,
                        image_path,
                        prompt_variant=prompt_variant,
                        round_index=action.action_index,
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": action.assistant_response,
                    "trainable": action.action_index in target_indices,
                }
            )
        metadata = {
            **trajectory.metadata,
            "window_index": window_index,
            "window_action_endpoint": endpoint,
            "image_history_max": images_to_keep,
            "retained_image_action_indices": list(range(retained_start, endpoint + 1)),
            "visual_token_lower_bound": sum(
                _visual_tokens_lower_bound(
                    actions[index - 1].model_image_size,
                    image_factor=profile.image_factor,
                )
                for index in range(retained_start, endpoint + 1)
            ),
            "target_action_indices": list(target_indices),
            "target_policy": target_policy,
            "row_schema_version": row_schema_version,
            "prompt_variant": prompt_variant,
        }
        windows.append(
            WindowExample(
                messages=messages,
                image_paths=tuple(retained_images),
                target_action_indices=target_indices,
                source_record_id=trajectory.source_record_id,
                window_index=window_index,
                metadata=metadata,
            )
        )
    return windows


def iter_groundcua46k_windows(
    records: Iterable[Mapping[str, Any]],
    *,
    model_family: str,
    images_to_keep: int = 3,
    prompt_variant: str | None = None,
):
    """Stream normalized windows without copying source images."""

    for record in records:
        trajectory = normalize_groundcua46k_trajectory(
            record,
            model_family=model_family,
        )
        yield from build_groundcua46k_windows(
            trajectory,
            model_family=model_family,
            images_to_keep=images_to_keep,
            prompt_variant=prompt_variant,
        )
