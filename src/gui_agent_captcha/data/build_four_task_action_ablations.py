"""Build matched four-task action-space ablation datasets.

The two output variants are:

``atomic_click_drag``
    One macro action per trajectory: ``click`` for ten-choice and ``drag`` for
    rotation plus both drag tasks.

``single_move_per_phase``
    Keep the existing primitive action contract, but retain exactly one
    ``move_to`` per semantic phase. Ten-choice becomes ``move_to ->
    left_click``. Drag tasks become ``move_to -> mouse_down -> move_to ->
    mouse_up``.

History is rebuilt from each transformed trajectory through the repository's
existing SFT prompt builder: all previous assistant responses remain causal,
while image history is capped at the latest three observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..integrations.storage import storage_path
from ..protocol_tracks import (
    PROTOCOL_VERSION,
    build_canonical_assistant_response,
    extract_assistant_response,
    extract_ten_choice_target_label,
    split_think_json_response,
)
from ..train.qwen3_vl_sft import load_sft_examples

ATOMIC_VARIANT = "atomic_click_drag"
SINGLE_MOVE_VARIANT = "single_move_per_phase"
VARIANTS = (ATOMIC_VARIANT, SINGLE_MOVE_VARIANT)
TASKS = (
    "ten_choice",
    "rotation",
    "first_person_drag",
    "third_person_drag",
)
SCHEMA = "four_task_action_ablations_matched_v1"
HISTORY_CONTRACT = "latest_3_images_all_previous_assistant_responses"
MATCHING_POLICY = "intersection_of_source_records_valid_for_both_variants"
SHUFFLE_SEED = 20260810
HD720_IMAGE_MAX_PIXELS = 1280 * 720
IMAGE_HISTORY_MAX = 3
PROMPT_FAMILY = "qwen3_vl_sft_closed_loop_context_v5_interleaved_history"
TASK_REQUIREMENT_POLICY = "single_task_requirement_v1"
FROZEN_PROMPT_RENDERER = "baseline_202607_two_message_v1"
FROZEN_BASELINE_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
)
FROZEN_BASELINE_PROMPT_SUFFIX_LENGTH = 3323
FROZEN_BASELINE_PROMPT_SUFFIX_SHA256 = (
    "0dcdb46025e9b4e4aea2d2b6462714411855ad9cb6fdbff1dcd2c624cfb2ee7a"
)
COORDINATE_MIN = 0.0
COORDINATE_MAX = 1000.0
COORDINATE_CENTER = 500.0

SOURCE_RELATIVE_PATHS = {
    "ten_choice": Path(
        "artifacts/datasets/"
        "ten_choice_terminal_leftclick_teacher_qwen35_397b_full4000_"
        "20260717/train.jsonl"
    ),
    "rotation": Path(
        "artifacts/datasets/"
        "rotation_teacher_think_qwen35_397b_a17b_full_hd720_20260714/train.jsonl"
    ),
    "first_person_drag": Path(
        "artifacts/datasets/"
        "slot_drag_teacher_think_qwen35_397b_a17b_v2_4k_720p_20260716/train.jsonl"
    ),
    "third_person_drag": Path(
        "artifacts/datasets/"
        "third_person_drag_teacher_think_qwen35_9b_deployed_v2_4k_720p_"
        "20260730/train.jsonl"
    ),
}


@dataclass(frozen=True)
class IndexedAction:
    action_index: int
    step_index: int
    step: dict[str, Any]
    before_observation: dict[str, Any]
    after_observation: dict[str, Any] | None


@dataclass(frozen=True)
class ConversionResult:
    record: dict[str, Any] | None
    rejection: dict[str, Any] | None


class RecordContractError(ValueError):
    """Raised when a source row is not one supported successful trajectory."""


def default_output_root() -> Path:
    return storage_path(
        "data",
        "action_ablations",
        "four_task_matched_v1_20260810",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RecordContractError(f"{path}:{line_number}: expected JSON object")
            records.append(value)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_ids_sha256(record_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for record_id in record_ids:
        digest.update(str(record_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _normalized_observation(step: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(step)
    raw_path = normalized.get("image_path") or normalized.get("observation_image")
    if not isinstance(raw_path, str) or not raw_path:
        raise RecordContractError("observation is missing image_path")
    normalized["image_path"] = str(Path(raw_path).resolve())
    normalized.pop("observation_image", None)
    return normalized


def _indexed_actions(record: dict[str, Any]) -> list[IndexedAction]:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RecordContractError(f"{record.get('id')}: missing steps")
    if not isinstance(steps[0], dict) or steps[0].get("type") != "observation":
        raise RecordContractError(f"{record.get('id')}: first step must be an observation")

    actions: list[IndexedAction] = []
    current_observation = _normalized_observation(steps[0])
    for step_index, raw_step in enumerate(steps[1:], start=1):
        if not isinstance(raw_step, dict):
            raise RecordContractError(f"{record.get('id')}: step {step_index} is not an object")
        step_type = raw_step.get("type")
        if step_type == "observation":
            current_observation = _normalized_observation(raw_step)
            continue
        if step_type != "action":
            raise RecordContractError(
                f"{record.get('id')}: unsupported step type at {step_index}: {step_type!r}"
            )
        after_observation = None
        if step_index + 1 < len(steps):
            candidate = steps[step_index + 1]
            if isinstance(candidate, dict) and candidate.get("type") == "observation":
                after_observation = _normalized_observation(candidate)
        actions.append(
            IndexedAction(
                action_index=len(actions),
                step_index=step_index,
                step=dict(raw_step),
                before_observation=current_observation,
                after_observation=after_observation,
            )
        )
    if not actions:
        raise RecordContractError(f"{record.get('id')}: trajectory has no actions")
    return actions


def _clean_action(step: dict[str, Any]) -> dict[str, Any]:
    kind = step.get("kind")
    if not isinstance(kind, str) or not kind:
        raise RecordContractError("action step is missing kind")
    action: dict[str, Any] = {"kind": kind}
    if kind == "move_to":
        x_value = step.get("x")
        y_value = step.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            raise RecordContractError("move_to requires numeric x and y")
        action.update({"x": x_value, "y": y_value})
    return action


def _action_from_response(response: str) -> dict[str, Any]:
    thought, remainder = split_think_json_response(response)
    if not isinstance(thought, str) or not thought.strip():
        raise RecordContractError("assistant_response requires a nonempty think block")
    try:
        payload = json.loads(remainder)
    except json.JSONDecodeError as exc:
        raise RecordContractError(f"assistant_response action JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("action"), dict):
        raise RecordContractError("assistant_response requires an action object")
    return dict(payload["action"])


def _validate_action_response(action_step: dict[str, Any]) -> None:
    response = extract_assistant_response(action_step)
    if not isinstance(response, str):
        raise RecordContractError("action step is missing assistant_response")
    if _action_from_response(response) != _clean_action(action_step):
        raise RecordContractError("assistant_response action does not match action step")


def _round_coordinate(value: float) -> float | int:
    rounded = round(float(value), 3)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _in_coordinate_range(*values: float | int) -> bool:
    return all(COORDINATE_MIN <= float(value) <= COORDINATE_MAX for value in values)


def _center_locked_aggregate(moves: list[IndexedAction]) -> tuple[float | int, float | int]:
    x_value = COORDINATE_CENTER + sum(
        float(_clean_action(item.step)["x"]) - COORDINATE_CENTER for item in moves
    )
    y_value = COORDINATE_CENTER + sum(
        float(_clean_action(item.step)["y"]) - COORDINATE_CENTER for item in moves
    )
    return _round_coordinate(x_value), _round_coordinate(y_value)


def _drag_phases(actions: list[IndexedAction], *, record_id: str) -> tuple[
    list[IndexedAction], IndexedAction, list[IndexedAction], IndexedAction
]:
    down_positions = [index for index, item in enumerate(actions) if item.step.get("kind") == "mouse_down"]
    up_positions = [index for index, item in enumerate(actions) if item.step.get("kind") == "mouse_up"]
    if len(down_positions) != 1 or len(up_positions) != 1:
        raise RecordContractError(f"{record_id}: expected exactly one mouse_down and mouse_up")
    down_position = down_positions[0]
    up_position = up_positions[0]
    if down_position >= up_position:
        raise RecordContractError(f"{record_id}: mouse_down must precede mouse_up")
    free_moves = [item for item in actions[:down_position] if item.step.get("kind") == "move_to"]
    held_moves = [item for item in actions[down_position + 1 : up_position] if item.step.get("kind") == "move_to"]
    if len(free_moves) != down_position or len(held_moves) != up_position - down_position - 1:
        raise RecordContractError(f"{record_id}: drag phases contain unsupported actions")
    if not free_moves or not held_moves:
        raise RecordContractError(f"{record_id}: drag phases require free and held move_to actions")
    if up_position != len(actions) - 1:
        raise RecordContractError(f"{record_id}: actions after terminal mouse_up are unsupported")
    return free_moves, actions[down_position], held_moves, actions[up_position]


def _direct_thought(task: str, phase: str) -> str:
    thoughts = {
        ("ten_choice", "atomic"): (
            "The task asks for one specific icon among the visible candidates. Under the one-step macro contract, the correct operation is an immediate click rather than a hover-and-confirm sequence. I therefore place the click point at the center of the target icon and complete the selection in this single action."
        ),
        ("ten_choice", "move"): (
            "The task requires selecting the icon with the requested label, and this ablation permits only one positioning move before the click. I therefore skip the multi-icon scan and move directly to the target icon's center. The resulting hover observation will expose its tooltip so the selection can be confirmed before clicking."
        ),
        ("ten_choice", "click"): (
            "The current observation shows the cursor on the candidate whose revealed tooltip matches the requested label. The pointer is already at the correct location, so another move would be unnecessary. I use left_click now because it selects the currently hovered icon without adding coordinates."
        ),
        ("rotation", "atomic"): (
            "The screenshot shows the rotation control and its draggable handle, while the central region still needs to be aligned with the background. The one-step contract requires a single continuous gesture, so I start on the handle and drag directly to the final aligned position. This macro action includes the press, movement, and release."
        ),
        ("rotation", "held_move"): (
            "The slider is already held, so the remaining operation is to place it at the final alignment position. This ablation removes the intermediate coarse-to-fine moves while retaining the same final state. I therefore move directly to the final coordinate that aligns the circular region with the background."
        ),
        ("first_person_drag", "atomic"): (
            "This first-person task uses a fixed center reticle, so cursor coordinates represent view displacement around the center rather than ordinary absolute pointing. I use the aggregated source motion to acquire the solid shape and carry it to the matching outline. The complete press-drag-release interaction is expressed as one macro drag."
        ),
        ("first_person_drag", "free_move"): (
            "The reticle remains fixed at the screen center while the view moves around it. I combine the original approach motions into their exact net displacement and move directly until the solid shape is centered under the reticle. This reaches the same pre-grab state without intermediate move_to steps."
        ),
        ("first_person_drag", "held_move"): (
            "The shape is already held at the fixed center reticle. I combine the original drag adjustments into their exact net displacement and move directly until the matching outline reaches the reticle. This preserves the final placement while removing intermediate held move_to steps."
        ),
        ("third_person_drag", "atomic"): (
            "The solid shape and its matching gray outline are both visible in ordinary screen coordinates. A single continuous drag is sufficient: I start at the center of the solid shape, keep it held while moving, and release at the center of the matching outline. This completes the placement in one macro action."
        ),
        ("third_person_drag", "free_move"): (
            "I move directly to the center of the solid shape to prepare for dragging it."
        ),
        ("third_person_drag", "held_move"): (
            "While holding the shape, I move directly to the center of its matching outline."
        ),
    }
    try:
        return thoughts[(task, phase)]
    except KeyError as exc:
        raise RecordContractError(f"missing direct-thought template for {task}/{phase}") from exc


def _new_action_step(
    source: IndexedAction,
    action: dict[str, Any],
    *,
    thought: str | None,
    source_action_indices: list[int],
) -> dict[str, Any]:
    if thought is None and action == _clean_action(source.step):
        step = dict(source.step)
        _validate_action_response(step)
        source_metadata = step.get("metadata")
        metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    else:
        step = {"type": "action", **action}
        step["assistant_response"] = build_canonical_assistant_response(
            thought=thought or "I take the next correct action.",
            action=action,
        )
        metadata = {}
    metadata["ablation"] = {
        "source_action_indices": source_action_indices,
        "response_policy": "preserve_exact" if thought is None else "action_consistent_template_v1",
    }
    step["metadata"] = metadata
    _validate_action_response(step)
    return step


def _updated_metadata(
    record: dict[str, Any],
    *,
    task: str,
    variant: str,
    source_action_count: int,
    output_action_count: int,
    selected_source_action_indices: list[int],
    semantic_noop: bool,
) -> dict[str, Any]:
    source_metadata = record.get("metadata")
    metadata = dict(source_metadata) if isinstance(source_metadata, dict) else {}
    for stale_key in (
        "aim_step_count",
        "success_step_count",
        "mouse_down_action_position",
        "trajectory_pattern",
        "max_steps",
    ):
        metadata.pop(stale_key, None)
    metadata.update(
        {
            "action_step_count": output_action_count,
            "source_action_step_count": source_action_count,
            "max_steps": output_action_count,
            "trajectory_pattern": (
                "atomic_macro_direct" if variant == ATOMIC_VARIANT else "single_move_per_phase"
            ),
            "protocol_version": PROTOCOL_VERSION,
            "ablation": {
                "schema": SCHEMA,
                "task": task,
                "variant": variant,
                "source_record_id": str(record.get("id")),
                "source_action_count": source_action_count,
                "output_action_count": output_action_count,
                "selected_source_action_indices": selected_source_action_indices,
                "history_contract": HISTORY_CONTRACT,
                "semantic_noop": semantic_noop,
            },
        }
    )
    if variant == ATOMIC_VARIANT:
        action_kind = "click" if task == "ten_choice" else "drag"
        metadata.update(
            {
                "protocol_track": "captcha_old_legacy",
                "action_space_type": "legacy_macro",
                "action_paradigm": "macro_only",
                "task_action_kinds": [action_kind],
            }
        )
    else:
        metadata.update(
            {
                "protocol_track": "captcha_mixed_left_click",
                "action_space_type": "mixed_primitive_macro",
                "action_paradigm": "mixed",
                # This ablation changes only trajectory length. Keep the exact
                # five-action baseline contract as a controlled variable.
                "task_action_kinds": list(FROZEN_BASELINE_ACTION_KINDS),
            }
        )
    return metadata


def _rejection(
    record: dict[str, Any],
    *,
    task: str,
    variant: str,
    reason: str,
    details: dict[str, Any],
) -> ConversionResult:
    return ConversionResult(
        record=None,
        rejection={
            "id": str(record.get("id")),
            "task": task,
            "variant": variant,
            "reason": reason,
            "details": details,
        },
    )


def _ten_choice_conversion(
    record: dict[str, Any],
    *,
    variant: str,
    actions: list[IndexedAction],
) -> ConversionResult:
    if actions[-1].step.get("kind") != "left_click":
        raise RecordContractError(f"{record.get('id')}: ten-choice must end with left_click")
    moves = [item for item in actions[:-1] if item.step.get("kind") == "move_to"]
    if len(moves) != len(actions) - 1 or not moves:
        raise RecordContractError(f"{record.get('id')}: invalid ten-choice action sequence")
    target_move = moves[-1]
    target_action = _clean_action(target_move.step)
    target_label = extract_ten_choice_target_label(str(record.get("instruction", "")))
    instruction = (
        f'Click the icon that displays "{target_label}".'
        if isinstance(target_label, str) and target_label.strip()
        else str(record.get("instruction", ""))
    )

    if variant == ATOMIC_VARIANT:
        macro_action = {
            "kind": "click",
            "points": [[target_action["x"], target_action["y"]]],
        }
        metadata = _updated_metadata(
            record,
            task="ten_choice",
            variant=variant,
            source_action_count=len(actions),
            output_action_count=1,
            selected_source_action_indices=list(range(len(actions))),
            semantic_noop=False,
        )
        output = {
            "id": str(record["id"]),
            "source": record.get("source", ""),
            "instruction": instruction,
            "metadata": metadata,
            "image_path": actions[0].before_observation["image_path"],
            "action": "click",
            "x": target_action["x"],
            "y": target_action["y"],
            "assistant_response": build_canonical_assistant_response(
                thought=_direct_thought("ten_choice", "atomic"),
                action=macro_action,
            ),
        }
        return ConversionResult(record=output, rejection=None)

    selected_indices = [target_move.action_index, actions[-1].action_index]
    direct_move = _new_action_step(
        target_move,
        target_action,
        thought=(None if len(moves) == 1 else _direct_thought("ten_choice", "move")),
        source_action_indices=[item.action_index for item in moves],
    )
    terminal_click = _new_action_step(
        actions[-1],
        _clean_action(actions[-1].step),
        thought=_direct_thought("ten_choice", "click"),
        source_action_indices=[actions[-1].action_index],
    )
    output_steps: list[dict[str, Any]] = [actions[0].before_observation, direct_move]
    if target_move.after_observation is None:
        raise RecordContractError(f"{record.get('id')}: target hover observation is missing")
    output_steps.extend([target_move.after_observation, terminal_click])
    if actions[-1].after_observation is not None:
        output_steps.append(actions[-1].after_observation)
    output = {
        "id": str(record["id"]),
        "source": record.get("source", ""),
        "instruction": instruction,
        "metadata": _updated_metadata(
            record,
            task="ten_choice",
            variant=variant,
            source_action_count=len(actions),
            output_action_count=2,
            selected_source_action_indices=selected_indices,
            semantic_noop=len(actions) == 2,
        ),
        "steps": output_steps,
    }
    return ConversionResult(record=output, rejection=None)


def _drag_coordinates(
    *,
    task: str,
    free_moves: list[IndexedAction],
    held_moves: list[IndexedAction],
) -> tuple[tuple[float | int, float | int], tuple[float | int, float | int]]:
    if task == "first_person_drag":
        return _center_locked_aggregate(free_moves), _center_locked_aggregate(held_moves)
    free_action = _clean_action(free_moves[-1].step)
    held_action = _clean_action(held_moves[-1].step)
    return (
        (free_action["x"], free_action["y"]),
        (held_action["x"], held_action["y"]),
    )


def _drag_conversion(
    record: dict[str, Any],
    *,
    task: str,
    variant: str,
    actions: list[IndexedAction],
) -> ConversionResult:
    free_moves, down, held_moves, up = _drag_phases(actions, record_id=str(record.get("id")))
    free_xy, held_xy = _drag_coordinates(task=task, free_moves=free_moves, held_moves=held_moves)

    if variant == ATOMIC_VARIANT:
        if task == "first_person_drag":
            end_xy = (
                _round_coordinate(float(free_xy[0]) + float(held_xy[0]) - COORDINATE_CENTER),
                _round_coordinate(float(free_xy[1]) + float(held_xy[1]) - COORDINATE_CENTER),
            )
        else:
            end_xy = held_xy
        if not _in_coordinate_range(*free_xy, *end_xy):
            return _rejection(
                record,
                task=task,
                variant=variant,
                reason="atomic_drag_coordinate_out_of_range",
                details={"start": list(free_xy), "end": list(end_xy)},
            )
        macro_action = {
            "kind": "drag",
            "points": [list(free_xy), list(end_xy)],
        }
        output = {
            "id": str(record["id"]),
            "source": record.get("source", ""),
            "instruction": str(record.get("instruction", "")),
            "metadata": _updated_metadata(
                record,
                task=task,
                variant=variant,
                source_action_count=len(actions),
                output_action_count=1,
                selected_source_action_indices=list(range(len(actions))),
                semantic_noop=False,
            ),
            "image_path": actions[0].before_observation["image_path"],
            "action": "drag",
            "start_x": free_xy[0],
            "start_y": free_xy[1],
            "end_x": end_xy[0],
            "end_y": end_xy[1],
            "assistant_response": build_canonical_assistant_response(
                thought=_direct_thought(task, "atomic"),
                action=macro_action,
            ),
        }
        return ConversionResult(record=output, rejection=None)

    if not _in_coordinate_range(*free_xy, *held_xy):
        return _rejection(
            record,
            task=task,
            variant=variant,
            reason="single_move_coordinate_out_of_range",
            details={"free_move": list(free_xy), "held_move": list(held_xy)},
        )

    free_source = free_moves[-1]
    held_source = held_moves[-1]
    free_action = {"kind": "move_to", "x": free_xy[0], "y": free_xy[1]}
    held_action = {"kind": "move_to", "x": held_xy[0], "y": held_xy[1]}
    free_changed = len(free_moves) > 1 or free_action != _clean_action(free_source.step)
    held_changed = len(held_moves) > 1 or held_action != _clean_action(held_source.step)
    selected_indices = [
        free_source.action_index,
        down.action_index,
        held_source.action_index,
        up.action_index,
    ]
    selected = [
        _new_action_step(
            free_source,
            free_action,
            thought=_direct_thought(task, "free_move") if free_changed else None,
            source_action_indices=[item.action_index for item in free_moves],
        ),
        _new_action_step(
            down,
            _clean_action(down.step),
            thought=None,
            source_action_indices=[down.action_index],
        ),
        _new_action_step(
            held_source,
            held_action,
            thought=_direct_thought(task, "held_move") if held_changed else None,
            source_action_indices=[item.action_index for item in held_moves],
        ),
        _new_action_step(
            up,
            _clean_action(up.step),
            thought=None,
            source_action_indices=[up.action_index],
        ),
    ]
    sources = [free_source, down, held_source, up]
    output_steps: list[dict[str, Any]] = [actions[0].before_observation]
    for source, action_step in zip(sources, selected, strict=True):
        output_steps.append(action_step)
        if source.after_observation is None:
            raise RecordContractError(
                f"{record.get('id')}: selected action {source.action_index} lacks its observation"
            )
        output_steps.append(source.after_observation)
    semantic_noop = (
        task == "third_person_drag"
        and len(free_moves) == 1
        and len(held_moves) == 1
        and len(actions) == 4
    )
    output = {
        "id": str(record["id"]),
        "source": record.get("source", ""),
        "instruction": str(record.get("instruction", "")),
        "metadata": _updated_metadata(
            record,
            task=task,
            variant=variant,
            source_action_count=len(actions),
            output_action_count=4,
            selected_source_action_indices=selected_indices,
            semantic_noop=semantic_noop,
        ),
        "steps": output_steps,
    }
    return ConversionResult(record=output, rejection=None)


def convert_record(
    record: dict[str, Any],
    *,
    task: str,
    variant: str,
) -> ConversionResult:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    actions = _indexed_actions(record)
    for action in actions:
        _validate_action_response(action.step)
    if task == "ten_choice":
        return _ten_choice_conversion(record, variant=variant, actions=actions)
    return _drag_conversion(record, task=task, variant=variant, actions=actions)


def _content_to_verl(
    content: Any,
    *,
    image_max_pixels: int,
) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []
    text_parts: list[str] = []
    images: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            text_parts.append(str(item))
            continue
        if item.get("type") == "image":
            raw_path = item.get("image") or item.get("image_url")
            if raw_path is None:
                continue
            image_path = Path(str(raw_path)).resolve()
            text_parts.append("<image>")
            images.append({"image": str(image_path), "max_pixels": image_max_pixels})
        elif item.get("type") == "text":
            text_parts.append(str(item.get("text", "")))
        else:
            text_parts.append(str(item))
    return "".join(text_parts), images


_FROZEN_ACTION_PROMPT = (
    "You are controlling a GUI agent. Given the screenshot and task, "
    "return exactly one response with a long explicit <think>...</think> block first, "
    'followed by exactly one compact JSON object with this schema: {"action":{...}}.'
)
_FROZEN_LONG_THINKING_REQUIREMENTS = (
    "The thinking must be long, explicit, and step-by-step.",
    "The thinking must be placed inside <think>...</think>, not inside a JSON thought field.",
    "Observe the key visible signals in the screenshot first.",
    "Then analyze the current state and constraints.",
    "Then judge the candidate actions.",
    "Finally decide the single next action.",
)
_FROZEN_REASONING_BODY = (
    "First restate and anchor yourself to the user's command, the action history, and the current UI screenshot. "
    "Then infer what has already been completed, what still needs to be done, and which visible UI element or input field is most relevant for the next step. "
    "Your reasoning should be grounded in the command({text}) and history({history}), but you may choose the specific analysis order freely. "
    "Do not over-constrain the reasoning into too many rigid substeps; keep it concise, task-oriented, and visually grounded. "
)
_FROZEN_REASONING_SUFFIX = (
    "The thinking process should explain why the chosen action is the best next atomic GUI operation, "
    "without mentioning any hidden label, target answer, or ground-truth action."
)
_FROZEN_SCHEMA_LINES = {
    "click": '- click: one immediate left click at a screenshot-relative point; schema {"kind":"click","points":[[<int 0-1000>,<int 0-1000>]]}',
    "drag": '- drag: one continuous press-drag-release gesture between two screenshot-relative points; schema {"kind":"drag","points":[[<int 0-1000>,<int 0-1000>],[<int 0-1000>,<int 0-1000>]]}',
    "move_to": '- move_to: move the cursor without pressing; schema {"kind":"move_to","x":<int 0-1000>,"y":<int 0-1000>}',
    "mouse_down": '- mouse_down: press and hold at the current cursor position; schema {"kind":"mouse_down"}',
    "mouse_up": '- mouse_up: release at the current cursor position; schema {"kind":"mouse_up"}',
    "left_click": '- left_click: one immediate left click at the current cursor or reticle position; no coordinates; schema {"kind":"left_click"}',
}


def _frozen_reasoning_contract(action_kinds: tuple[str, ...]) -> str:
    clauses: list[str] = []
    kinds = set(action_kinds)
    if "move_to" in kinds:
        clauses.append(
            "When move_to is needed, identify the target element or location and use its approximate center point."
        )
    if "mouse_down" in kinds:
        clauses.append(
            "When mouse_down is needed, use it only after the cursor is already positioned on the intended target."
        )
    if "mouse_up" in kinds:
        clauses.append(
            "When mouse_up is needed, use it only to release at the current cursor position when releasing is intended."
        )
    if "left_click" in kinds:
        clauses.append(
            "When left_click is needed, use it only to click or fire at the current cursor or reticle position; it takes no coordinates."
        )
    if "click" in kinds:
        clauses.append(
            "When click is needed, identify the target element and use the approximate center point of that element."
        )
    if "drag" in kinds:
        clauses.append(
            "When drag is needed, choose visible start and end points grounded in the screenshot."
        )
    action_list = "[" + ", ".join(f"'{kind}'" for kind in action_kinds) + "]"
    guidance = " ".join(clauses)
    return (
        f"Before deciding the next action enumerate from {action_list}. "
        f"{_FROZEN_REASONING_BODY}{guidance} {_FROZEN_REASONING_SUFFIX}"
    )


@lru_cache(maxsize=None)
def _frozen_action_prompt(instruction: str, action_kinds: tuple[str, ...]) -> str:
    if not action_kinds or any(kind not in _FROZEN_SCHEMA_LINES for kind in action_kinds):
        raise RecordContractError(f"unsupported frozen prompt action kinds: {action_kinds}")
    display = ", ".join(action_kinds)
    coordinate_actions = [kind for kind in ("click", "drag", "move_to") if kind in action_kinds]
    non_coordinate_actions = [
        kind for kind in ("left_click", "mouse_down", "mouse_up") if kind in action_kinds
    ]
    lines = [
        f"Task requirement: {instruction}",
        "Action space for this task:",
        f"Your action space contains these mouse actions: {display}.",
        "Use the repository JSON kind names below when you answer:",
        "Allowed action kinds for this task: " + display,
        "Action schemas and meanings:",
        *(_FROZEN_SCHEMA_LINES[kind] for kind in action_kinds),
        _FROZEN_ACTION_PROMPT,
        "A <think>...</think> block is required.",
        "The <think> block must appear before the JSON action object.",
        "Do not use a JSON thought field.",
        *_FROZEN_LONG_THINKING_REQUIREMENTS,
        _frozen_reasoning_contract(action_kinds),
        'The JSON object must carry only the action body: {"action":{...}}.',
        "Output exactly one action for this prompt.",
    ]
    if coordinate_actions:
        lines.extend(
            [
                f"Coordinate contract: {', '.join(coordinate_actions)} coordinates must use 0-1000 screenshot-relative integer coordinates.",
                "Coordinate meaning: x=0 is the left edge, x=1000 is the right edge; y=0 is the top edge, y=1000 is the bottom edge.",
                "Do not output pixel coordinates or 0-1 decimal coordinates.",
            ]
        )
    if non_coordinate_actions:
        lines.append(f"{', '.join(non_coordinate_actions)} must not include x or y.")
    prompt = "\n".join(lines)
    if action_kinds == FROZEN_BASELINE_ACTION_KINDS:
        suffix = prompt.split("\n", 1)[1]
        suffix_sha256 = hashlib.sha256(suffix.encode("utf-8")).hexdigest()
        if (
            len(suffix) != FROZEN_BASELINE_PROMPT_SUFFIX_LENGTH
            or suffix_sha256 != FROZEN_BASELINE_PROMPT_SUFFIX_SHA256
        ):
            raise RecordContractError(
                "frozen baseline prompt suffix does not match the July 2026 parquet contract"
            )
    return prompt


def _frozen_action_kinds(*, task: str, variant: str) -> tuple[str, ...]:
    if variant == SINGLE_MOVE_VARIANT:
        return FROZEN_BASELINE_ACTION_KINDS
    return ("click",) if task == "ten_choice" else ("drag",)


def _frozen_action_context(action_position: int) -> str:
    if action_position == 1:
        history_intro = "Current prediction context: 1 previous assistant response is included"
    else:
        history_intro = (
            f"Current prediction context: {action_position} previous assistant responses are included"
        )
    lines = [
        f"{history_intro} in this prompt in chronological order; older screenshots may be omitted, but all previous "
        "<think>...</think> blocks and JSON actions are retained. The last attached image is the current observation.",
        (
            "Use the previous <think>...</think> and JSON actions as context only."
            if action_position
            else "No previous actions in this trajectory."
        ),
        f"Return <think>...</think> followed by the JSON action for step {action_position + 1}.",
    ]
    return "\n".join(lines)


def _frozen_messages(
    example: Any,
    *,
    task: str,
    variant: str,
    images_to_keep: int,
) -> list[dict[str, Any]]:
    action_position = int(example.context.action_position)
    image_paths = tuple(example.image_paths) or (example.image_path,)
    response_history = tuple(example.assistant_response_history)
    if len(image_paths) != action_position + 1:
        raise RecordContractError(
            f"action {action_position}: expected {action_position + 1} image paths, got {len(image_paths)}"
        )
    if len(response_history) != action_position:
        raise RecordContractError(
            f"action {action_position}: expected {action_position} exact prior responses, got {len(response_history)}"
        )
    if not isinstance(example.assistant_response, str) or not example.assistant_response.strip():
        raise RecordContractError(f"action {action_position}: missing exact assistant response")

    retained_paths = image_paths[-images_to_keep:]
    first_retained_index = len(image_paths) - len(retained_paths)
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": _frozen_action_prompt(
                example.instruction,
                _frozen_action_kinds(task=task, variant=variant),
            ),
        }
    ]

    def append_previous_response(action_index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {action_index + 1} assistant response (context only):\n"
                    f"{response_history[action_index]}\n"
                ),
            }
        )

    for action_index in range(first_retained_index):
        append_previous_response(action_index)
    for retained_offset, path in enumerate(retained_paths):
        content.append({"type": "image", "image": str(path)})
        action_index = first_retained_index + retained_offset
        if action_index < action_position:
            append_previous_response(action_index)
    content.append({"type": "text", "text": _frozen_action_context(action_position)})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": example.assistant_response.strip()},
    ]


def _messages_to_verl(
    messages: list[dict[str, Any]],
    *,
    image_max_pixels: int,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    output_messages: list[dict[str, str]] = []
    images: list[dict[str, Any]] = []
    for message in messages:
        content, message_images = _content_to_verl(
            message.get("content", ""),
            image_max_pixels=image_max_pixels,
        )
        output_messages.append({"role": str(message.get("role", "user")), "content": content})
        images.extend(message_images)
    return output_messages, images


def _manifest_provenance(path: Path) -> list[tuple[str, int]]:
    provenance: list[tuple[str, int]] = []
    for record in _read_jsonl(path):
        if isinstance(record.get("action"), str):
            action_count = 1
        else:
            action_count = sum(
                1
                for step in record.get("steps", [])
                if isinstance(step, dict) and step.get("type") == "action"
            )
        provenance.extend((str(record.get("id")), index) for index in range(action_count))
    return provenance


def build_verl_rows(
    manifest_path: Path,
    *,
    task: str,
    variant: str,
    image_history_max: int,
    image_max_pixels: int,
) -> list[dict[str, Any]]:
    examples = load_sft_examples([manifest_path])
    provenance = _manifest_provenance(manifest_path)
    if len(examples) != len(provenance):
        raise RecordContractError(
            f"{manifest_path}: example/provenance mismatch {len(examples)} != {len(provenance)}"
        )
    rows: list[dict[str, Any]] = []
    for source_index, (example, (record_id, action_index)) in enumerate(
        zip(examples, provenance, strict=True)
    ):
        messages, images = _messages_to_verl(
            _frozen_messages(
                example,
                task=task,
                variant=variant,
                images_to_keep=image_history_max,
            ),
            image_max_pixels=image_max_pixels,
        )
        expected_images = 1 if variant == ATOMIC_VARIANT else min(action_index + 1, image_history_max)
        if len(images) != expected_images:
            raise RecordContractError(
                f"{record_id} action {action_index}: expected {expected_images} images, got {len(images)}"
            )
        if [message["role"] for message in messages] != ["user", "assistant"]:
            raise RecordContractError(f"{record_id} action {action_index}: invalid message roles")
        expected_history = 0 if variant == ATOMIC_VARIANT else action_index
        history_count = messages[0]["content"].count("Previous step ")
        if history_count != expected_history:
            raise RecordContractError(
                f"{record_id} action {action_index}: expected {expected_history} prior responses, got {history_count}"
            )
        rows.append(
            {
                "messages": messages,
                "images": images,
                "tools": [],
                "metadata": {
                    "source_index": source_index,
                    "source_record_id": record_id,
                    "action_index": action_index,
                    "task": task,
                    "variant": variant,
                    "image_history_max": image_history_max,
                    "image_max_pixels": image_max_pixels,
                    "history_contract": HISTORY_CONTRACT,
                    "prompt_family": PROMPT_FAMILY,
                    "task_requirement_policy": TASK_REQUIREMENT_POLICY,
                    "prompt_renderer": FROZEN_PROMPT_RENDERER,
                },
            }
        )
    return rows


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _action_counts(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if isinstance(record.get("action"), str):
            counts[str(record["action"])] += 1
            continue
        for step in record.get("steps", []):
            if isinstance(step, dict) and step.get("type") == "action":
                counts[str(step.get("kind"))] += 1
    return counts


def _task_records(
    source_records: list[dict[str, Any]],
    *,
    task: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    converted: dict[str, dict[str, ConversionResult]] = {variant: {} for variant in VARIANTS}
    source_order: list[str] = []
    for record in source_records:
        record_id = str(record.get("id"))
        if not record_id or record_id == "None":
            raise RecordContractError(f"{task}: record without id")
        if record_id in converted[ATOMIC_VARIANT]:
            raise RecordContractError(f"{task}: duplicate id {record_id}")
        source_order.append(record_id)
        for variant in VARIANTS:
            converted[variant][record_id] = convert_record(record, task=task, variant=variant)

    matched_ids = [
        record_id
        for record_id in source_order
        if all(converted[variant][record_id].record is not None for variant in VARIANTS)
    ]
    matched_id_set = set(matched_ids)
    outputs: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    rejections: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    intrinsic_counts: dict[str, Counter[str]] = {variant: Counter() for variant in VARIANTS}
    paired_exclusions: dict[str, int] = {variant: 0 for variant in VARIANTS}
    for variant in VARIANTS:
        for record_id in source_order:
            result = converted[variant][record_id]
            if record_id in matched_id_set:
                assert result.record is not None
                outputs[variant].append(result.record)
                continue
            if result.rejection is not None:
                rejection = dict(result.rejection)
                rejection["exclusion_type"] = "intrinsic_invalid"
                intrinsic_counts[variant][str(rejection["reason"])] += 1
            else:
                rejection = {
                    "id": record_id,
                    "task": task,
                    "variant": variant,
                    "reason": "paired_variant_invalid",
                    "details": {},
                    "exclusion_type": "matched_pair_exclusion",
                }
                paired_exclusions[variant] += 1
            rejections[variant].append(rejection)

    stats = {
        "source_records": len(source_records),
        "intrinsic_valid_records": {
            variant: sum(
                result.record is not None for result in converted[variant].values()
            )
            for variant in VARIANTS
        },
        "matched_records": len(matched_ids),
        "matched_source_ids_sha256": _ordered_ids_sha256(matched_ids),
        "excluded_records": len(source_records) - len(matched_ids),
        "intrinsic_rejection_counts": {
            variant: dict(sorted(counts.items())) for variant, counts in intrinsic_counts.items()
        },
        "paired_exclusion_counts": paired_exclusions,
        "semantic_noop": task == "third_person_drag",
    }
    return outputs, rejections, stats


def build_dataset(
    *,
    project_root: Path,
    output_root: Path,
    source_paths: dict[str, Path] | None = None,
    image_history_max: int = IMAGE_HISTORY_MAX,
    image_max_pixels: int = HD720_IMAGE_MAX_PIXELS,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_sources = source_paths or {
        task: (project_root / relative_path).resolve()
        for task, relative_path in SOURCE_RELATIVE_PATHS.items()
    }
    if set(resolved_sources) != set(TASKS):
        raise ValueError("source_paths must specify exactly the four supported tasks")

    generated_at = datetime.now(timezone.utc).isoformat()
    source_summaries: dict[str, Any] = {}
    task_stats: dict[str, Any] = {}
    variant_task_summaries: dict[str, dict[str, Any]] = {variant: {} for variant in VARIANTS}
    task_parquets: dict[str, list[Path]] = {variant: [] for variant in VARIANTS}

    for task in TASKS:
        source_path = resolved_sources[task]
        if not source_path.is_file():
            raise FileNotFoundError(f"missing source manifest for {task}: {source_path}")
        source_records = _read_jsonl(source_path)
        outputs, rejections, stats = _task_records(source_records, task=task)
        task_stats[task] = stats
        source_summaries[task] = {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "records": len(source_records),
            "actions": sum(_action_counts(source_records).values()),
            "action_counts": dict(sorted(_action_counts(source_records).items())),
        }

        for variant in VARIANTS:
            task_dir = output_root / variant / task
            manifest_path = task_dir / "train.jsonl"
            rejection_path = task_dir / "rejected.jsonl"
            write_jsonl(manifest_path, outputs[variant])
            write_jsonl(rejection_path, rejections[variant])
            rows = build_verl_rows(
                manifest_path,
                task=task,
                variant=variant,
                image_history_max=image_history_max,
                image_max_pixels=image_max_pixels,
            )
            parquet_path = task_dir / "train.parquet"
            _write_parquet(rows, parquet_path)
            image_counts = Counter(len(row["images"]) for row in rows)
            task_summary = {
                "schema": SCHEMA,
                "status": "passed",
                "validation_scope": "dataset_structure_and_contract_only",
                "generated_at_utc": generated_at,
                "task": task,
                "variant": variant,
                "matching_policy": MATCHING_POLICY,
                "source": source_summaries[task],
                "records": len(outputs[variant]),
                "rows": len(rows),
                "action_counts": dict(sorted(_action_counts(outputs[variant]).items())),
                "image_count_distribution": {
                    str(count): value for count, value in sorted(image_counts.items())
                },
                "image_history_max": image_history_max,
                "image_max_pixels": image_max_pixels,
                "history_contract": HISTORY_CONTRACT,
                "prompt_family": PROMPT_FAMILY,
                "prompt_renderer": FROZEN_PROMPT_RENDERER,
                "task_requirement_policy": TASK_REQUIREMENT_POLICY,
                "semantic_noop": task == "third_person_drag" and variant == SINGLE_MOVE_VARIANT,
                "rejected_records": len(rejections[variant]),
                "rejection_counts": dict(
                    sorted(Counter(str(item["reason"]) for item in rejections[variant]).items())
                ),
                "outputs": {
                    "manifest": str(manifest_path),
                    "manifest_sha256": _sha256(manifest_path),
                    "parquet": str(parquet_path),
                    "parquet_sha256": _sha256(parquet_path),
                    "rejections": str(rejection_path),
                    "rejections_sha256": _sha256(rejection_path),
                },
            }
            _write_json(task_dir / "summary.json", task_summary)
            variant_task_summaries[variant][task] = task_summary
            task_parquets[variant].append(parquet_path)

    variant_summaries: dict[str, Any] = {}
    for variant in VARIANTS:
        frames = [pd.read_parquet(path) for path in task_parquets[variant]]
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=SHUFFLE_SEED).reset_index(drop=True)
        combined_path = output_root / variant / "train.parquet"
        combined.to_parquet(combined_path, index=False)
        task_row_counts = Counter(str(metadata["task"]) for metadata in combined["metadata"])
        variant_summary = {
            "schema": SCHEMA,
            "status": "passed",
            "validation_scope": "dataset_structure_and_contract_only",
            "generated_at_utc": generated_at,
            "variant": variant,
            "matching_policy": MATCHING_POLICY,
            "shuffle_seed": SHUFFLE_SEED,
            "records": sum(
                int(variant_task_summaries[variant][task]["records"]) for task in TASKS
            ),
            "rows": len(combined),
            "task_record_counts": {
                task: int(variant_task_summaries[variant][task]["records"]) for task in TASKS
            },
            "task_row_counts": dict(sorted(task_row_counts.items())),
            "image_history_max": image_history_max,
            "image_max_pixels": image_max_pixels,
            "history_contract": HISTORY_CONTRACT,
            "prompt_family": PROMPT_FAMILY,
            "prompt_renderer": FROZEN_PROMPT_RENDERER,
            "third_person_single_move_is_noop": variant == SINGLE_MOVE_VARIANT,
            "train_parquet": str(combined_path),
            "train_parquet_sha256": _sha256(combined_path),
        }
        _write_json(output_root / variant / "summary.json", variant_summary)
        variant_summaries[variant] = variant_summary

    summary = {
        "schema": SCHEMA,
        "status": "passed",
        "validation_scope": "dataset_structure_and_contract_only",
        "training_status": "not_started",
        "environment_evaluation_status": "not_started",
        "generated_at_utc": generated_at,
        "project_root": str(project_root.resolve()),
        "output_root": str(output_root.resolve()),
        "matching_policy": MATCHING_POLICY,
        "history_contract": HISTORY_CONTRACT,
        "prompt_contract": {
            "prompt_family": PROMPT_FAMILY,
            "renderer": FROZEN_PROMPT_RENDERER,
            "roles": ["user", "assistant"],
            "single_move_action_kinds": list(FROZEN_BASELINE_ACTION_KINDS),
            "atomic_ten_choice_action_kinds": ["click"],
            "atomic_drag_task_action_kinds": ["drag"],
            "baseline_suffix_length": FROZEN_BASELINE_PROMPT_SUFFIX_LENGTH,
            "baseline_suffix_sha256": FROZEN_BASELINE_PROMPT_SUFFIX_SHA256,
        },
        "image_history_max": image_history_max,
        "image_max_pixels": image_max_pixels,
        "source_datasets": source_summaries,
        "task_matching": task_stats,
        "variants": variant_summaries,
        "single_to_atomic_row_ratio": round(
            variant_summaries[SINGLE_MOVE_VARIANT]["rows"]
            / variant_summaries[ATOMIC_VARIANT]["rows"],
            6,
        ),
        "construction": [
            "Both variants use the matched source-record intersection that supports a legal atomic drag under the egocentric center-locked controls.",
            "Exocentric Drag uses one free move and one held move; the single-move variant preserves those actions.",
            "Changed actions use deterministic action-consistent English think templates; unchanged retained actions preserve their source teacher responses.",
            "Ten-Choice atomic click selects a candidate directly, with hover feedback removed.",
            "The variants contain different action-level row counts; training budgets are specified in optimizer steps or tokens.",
        ],
    }
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    ready = {
        "schema": SCHEMA,
        "status": "passed",
        "readiness_scope": "ready_for_training_input_only",
        "training_status": "not_started",
        "environment_evaluation_status": "not_started",
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "variant_parquets": {
            variant: {
                "path": variant_summaries[variant]["train_parquet"],
                "sha256": variant_summaries[variant]["train_parquet_sha256"],
            }
            for variant in VARIANTS
        },
    }
    _write_json(output_root / "READY.json", ready)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = default_output_root()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_dataset(
        project_root=args.project_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
