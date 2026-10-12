"""Build matched no-Think action ablations for all six exploration variants.

The exporter creates two datasets for every environment variant:

``atomic_click_drag``
    Ten-choice uses one coordinate-bearing ``click``. Rotation and both drag
    variants use one coordinate-bearing ``drag``.

``single_move_per_phase``
    Ten-choice uses ``move_to -> left_click``. Rotation and drag use
    ``move_to -> mouse_down -> move_to -> mouse_up``. Each semantic phase has
    exactly one move, with no corrective move sequence.

Both arms use the exact same accepted source-record intersection inside each
environment. First-person records that cannot be expressed with legal 0-1000
coordinates are rejected from both arms instead of receiving clipped labels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from ...data.verl_sft_contract import messages_to_verl_row
from ...domains.rotation.environment import SLIDER_THUMB_DIAMETER_PX
from .training_no_think import (
    _INTEGER_ACTION_SCHEMA_LINES,
    IMAGE_HISTORY_MAX,
    IMAGE_MAX_PIXELS,
    MODEL_COORDINATE_CONTRACT,
    PROMPT_CONTRACT,
    RESPONSE_CONTRACT,
    SHUFFLE_SEED,
    SIX_ACTION_KINDS,
    PreparedTrace,
    _atomic_json,
    _model_coordinate,
    _ParquetSink,
    _prepare_trace,
    _read_jsonl,
    empty_think_response,
)

ATOMIC_VARIANT = "atomic_click_drag"
SINGLE_MOVE_VARIANT = "single_move_per_phase"
ABLATION_VARIANTS = (ATOMIC_VARIANT, SINGLE_MOVE_VARIANT)
ENVIRONMENT_VARIANTS = (
    "ten_choice_third_person",
    "ten_choice_first_person",
    "rotation_inner",
    "rotation_outer",
    "drag_third_person",
    "drag_first_person",
)
EXPECTED_SOURCE_TRACES = 4_000
EXPECTED_SOURCE_ACTIONS = 154_713
SOURCE_SUITE_ID = "exploration_depth_train_6x4000_v3"
SOURCE_RAW_SCHEMA = "exploration_depth_safe_probe_raw_ready_v1"
SOURCE_RAW_DIRECTORY = "raw_traces_safe_probe_v3"
SCHEMA = "exploration_depth_six_environment_action_ablations_no_think_matched_v2"
HISTORY_CONTRACT = "latest_3_images_all_previous_assistant_responses"
MATCHING_POLICY = "per_environment_intersection_valid_for_both_ablation_arms"
FIRST_PERSON_MARKER_CONTRACT = "mouse_icon"
ROTATION_GESTURE_CONTRACT = "exactly_one_mouse_down_and_one_mouse_up"

SOURCE_METADATA_ROOT = Path(
    str(Path(__file__).resolve().parents[4] / f'data/training/{SOURCE_SUITE_ID}')
)


class AblationDatasetError(ValueError):
    """A source trace, episode, or transformed trajectory violates the contract."""


@dataclass(frozen=True)
class TransformedTrace:
    record_id: str
    instruction: str
    images: tuple[Path, ...]
    actions: tuple[dict[str, object], ...]
    exploration_level: str


@dataclass(frozen=True)
class MatchedConversion:
    atomic: TransformedTrace | None
    single: TransformedTrace | None
    rejection: dict[str, object] | None

    def accepted(self) -> bool:
        return self.atomic is not None and self.single is not None and self.rejection is None


def default_input_root() -> Path:
    return SOURCE_METADATA_ROOT / SOURCE_RAW_DIRECTORY


def default_manifest_path() -> Path:
    return SOURCE_METADATA_ROOT / "manifest.json"


def default_output_root() -> Path:
    return SOURCE_METADATA_ROOT / "action_ablations_no_think_matched_v2"


def _action(kind: str, **payload: object) -> dict[str, object]:
    return {"kind": kind, **payload}


def _model_xy_from_pixel(
    pixel_xy: tuple[float, float], viewport: tuple[int, int]
) -> tuple[int, int] | None:
    width, height = viewport
    epsilon = 1e-6
    if not (
        -epsilon <= pixel_xy[0] <= width - 1 + epsilon
        and -epsilon <= pixel_xy[1] <= height - 1 + epsilon
    ):
        return None
    pixel_x = min(max(float(pixel_xy[0]), 0.0), float(width - 1))
    pixel_y = min(max(float(pixel_xy[1]), 0.0), float(height - 1))
    return (
        _model_coordinate(pixel_x / width * 1000.0, field="pixel_x"),
        _model_coordinate(pixel_y / height * 1000.0, field="pixel_y"),
    )


def _point_action(kind: str, points: Iterable[tuple[int, int]]) -> dict[str, object]:
    return _action(
        kind,
        points=[[x, y] for x, y in points],
    )


def _trace(
    source: PreparedTrace,
    *,
    images: Iterable[Path],
    actions: Iterable[dict[str, object]],
) -> TransformedTrace:
    image_tuple = tuple(images)
    action_tuple = tuple(dict(action) for action in actions)
    if len(image_tuple) != len(action_tuple) + 1:
        raise AblationDatasetError(
            f"{source.record_id}: transformed images/actions are not causal"
        )
    return TransformedTrace(
        record_id=source.record_id,
        instruction=source.instruction,
        images=image_tuple,
        actions=action_tuple,
        exploration_level=source.exploration_level,
    )


def _reject(source: PreparedTrace, variant: str, reason: str, **details: object) -> MatchedConversion:
    return MatchedConversion(
        atomic=None,
        single=None,
        rejection={
            "source_record_id": source.record_id,
            "environment_variant": variant,
            "reason": reason,
            **details,
        },
    )


def _episode_viewport(episode: Mapping[str, object]) -> tuple[int, int]:
    raw = episode.get("viewport")
    if not isinstance(raw, list) or len(raw) != 2:
        raise AblationDatasetError("episode viewport must contain width and height")
    return (int(raw[0]), int(raw[1]))


def _shared(episode: Mapping[str, object]) -> Mapping[str, object]:
    value = episode.get("shared_scene_config")
    if not isinstance(value, dict):
        raise AblationDatasetError("episode has no shared_scene_config")
    return value


def _pair(value: object, *, field: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise AblationDatasetError(f"{field} must contain x and y")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise AblationDatasetError(f"{field} must be numeric")
    return (float(value[0]), float(value[1]))


def _ten_choice_conversion(
    source: PreparedTrace,
    *,
    episode: Mapping[str, object],
    variant: str,
) -> MatchedConversion:
    if not source.actions or source.actions[-1].get("kind") != "left_click":
        raise AblationDatasetError(f"{source.record_id}: ten-choice must end in left_click")
    move_indices = [
        index for index, action in enumerate(source.actions) if action.get("kind") == "move_to"
    ]
    if not move_indices or move_indices[-1] != len(source.actions) - 2:
        raise AblationDatasetError(f"{source.record_id}: ten-choice terminal move is missing")

    shared = _shared(episode)
    target = shared.get("target_object")
    centers = shared.get("icon_centers_xy")
    if not isinstance(target, dict) or not isinstance(centers, list):
        raise AblationDatasetError(f"{source.record_id}: malformed ten-choice scene")
    target_index = int(target["target_index"])
    target_pixel = _pair(centers[target_index], field="target icon center")
    viewport = _episode_viewport(episode)

    if variant == "ten_choice_third_person":
        model_xy = _model_xy_from_pixel(target_pixel, viewport)
    else:
        dynamics = shared.get("hidden_dynamics")
        if not isinstance(dynamics, dict):
            raise AblationDatasetError(f"{source.record_id}: hidden dynamics are missing")
        sensitivity = float(dynamics["sensitivity"])
        direction = _pair(dynamics.get("direction_xy"), field="direction_xy")
        center = (viewport[0] / 2.0, viewport[1] / 2.0)
        desired_scene_delta = (center[0] - target_pixel[0], center[1] - target_pixel[1])
        pointer_delta = (
            desired_scene_delta[0] / sensitivity / direction[0],
            desired_scene_delta[1] / sensitivity / direction[1],
        )
        model_xy = _model_xy_from_pixel(
            (center[0] + pointer_delta[0], center[1] + pointer_delta[1]), viewport
        )
    if model_xy is None:
        return _reject(
            source,
            variant,
            "one_step_target_coordinate_out_of_range",
            target_pixel=list(target_pixel),
        )

    last_move = move_indices[-1]
    single = _trace(
        source,
        images=(source.images[0], source.images[last_move + 1], source.images[-1]),
        actions=(
            _action("move_to", x=model_xy[0], y=model_xy[1]),
            _action("left_click"),
        ),
    )
    atomic = _trace(
        source,
        images=(source.images[0], source.images[-1]),
        actions=(_point_action("click", (model_xy,)),),
    )
    return MatchedConversion(atomic=atomic, single=single, rejection=None)


def _rotation_conversion(
    source: PreparedTrace,
    *,
    episode: Mapping[str, object],
    variant: str,
) -> MatchedConversion:
    kinds = [str(action.get("kind")) for action in source.actions]
    if not kinds or kinds[0] != "move_to" or kinds[-1] != "mouse_up":
        raise AblationDatasetError(f"{source.record_id}: malformed rotation action sequence")
    if kinds.count("mouse_down") != 1 or kinds.count("mouse_up") != 1:
        raise AblationDatasetError(
            f"{source.record_id}: rotation source must contain exactly one mouse_down "
            "and one mouse_up; retry trajectories are not allowed"
        )
    final_up = len(kinds) - 1
    final_down = kinds.index("mouse_down")
    final_move = max(
        index for index in range(final_down + 1, final_up) if kinds[index] == "move_to"
    )
    shared = _shared(episode)
    geometry = shared.get("slider_geometry")
    if not isinstance(geometry, dict):
        raise AblationDatasetError(f"{source.record_id}: slider geometry is missing")
    slider_x = float(geometry["x"])
    slider_y = float(geometry["y"]) + float(geometry["height"]) / 2.0
    slider_width = float(geometry["width"])
    slider_min = float(shared["slider_min_value"])
    slider_max = float(shared["slider_max_value"])
    slider_range = slider_max - slider_min
    if slider_width <= 0.0 or slider_range <= 0.0:
        raise AblationDatasetError(f"{source.record_id}: invalid slider geometry or range")
    start_fraction = (float(shared["start_slider_value"]) - slider_min) / slider_range
    target_fraction = (float(shared["target_slider_value"]) - slider_min) / slider_range
    if not 0.0 <= start_fraction <= 1.0 or not 0.0 <= target_fraction <= 1.0:
        raise AblationDatasetError(f"{source.record_id}: slider value is outside its range")
    thumb_diameter = min(SLIDER_THUMB_DIAMETER_PX, slider_width)
    thumb_track_width = slider_width - thumb_diameter
    start_pixel = (
        slider_x + start_fraction * thumb_track_width + thumb_diameter / 2.0,
        slider_y,
    )
    atomic_end_pixel = (
        slider_x + target_fraction * thumb_track_width + thumb_diameter / 2.0,
        slider_y,
    )
    single_end_pixel = (slider_x + target_fraction * slider_width, slider_y)
    viewport = _episode_viewport(episode)
    start_xy = _model_xy_from_pixel(start_pixel, viewport)
    atomic_end_xy = _model_xy_from_pixel(atomic_end_pixel, viewport)
    single_end_xy = _model_xy_from_pixel(single_end_pixel, viewport)
    if start_xy is None or atomic_end_xy is None or single_end_xy is None:
        raise AblationDatasetError(f"{source.record_id}: slider coordinate is outside viewport")

    single = _trace(
        source,
        images=(
            source.images[0],
            source.images[1],
            source.images[final_down + 1],
            source.images[final_move + 1],
            source.images[final_up + 1],
        ),
        actions=(
            _action("move_to", x=start_xy[0], y=start_xy[1]),
            _action("mouse_down"),
            _action("move_to", x=single_end_xy[0], y=single_end_xy[1]),
            _action("mouse_up"),
        ),
    )
    atomic = _trace(
        source,
        images=(source.images[0], source.images[final_up + 1]),
        actions=(_point_action("drag", (start_xy, atomic_end_xy)),),
    )
    return MatchedConversion(atomic=atomic, single=single, rejection=None)


def _clamp_view_center(
    center_xy: tuple[float, float],
    *,
    world_size: tuple[float, float],
    viewport: tuple[int, int],
    zoom: float,
) -> tuple[float, float]:
    half_w = viewport[0] / 2.0 / zoom
    half_h = viewport[1] / 2.0 / zoom
    min_x = min(half_w, world_size[0] - half_w)
    max_x = max(half_w, world_size[0] - half_w)
    min_y = min(half_h, world_size[1] - half_h)
    max_y = max(half_h, world_size[1] - half_h)
    return (
        min(max(center_xy[0], min_x), max_x),
        min(max(center_xy[1], min_y), max_y),
    )


def _clamp_world(
    point: tuple[float, float], world_size: tuple[float, float]
) -> tuple[float, float]:
    return (
        min(max(point[0], 0.0), world_size[0]),
        min(max(point[1], 0.0), world_size[1]),
    )


def _drag_indices(source: PreparedTrace) -> tuple[int, int, int, int]:
    kinds = [str(action.get("kind")) for action in source.actions]
    down = kinds.index("mouse_down")
    up = len(kinds) - 1 - kinds[::-1].index("mouse_up")
    free_move = max(index for index in range(down) if kinds[index] == "move_to")
    held_move = max(index for index in range(down + 1, up) if kinds[index] == "move_to")
    return free_move, down, held_move, up


def _drag_third_person_conversion(
    source: PreparedTrace,
    *,
    episode: Mapping[str, object],
    variant: str,
) -> MatchedConversion:
    shared = _shared(episode)
    viewport = _episode_viewport(episode)
    start_pixel = _pair(shared.get("piece_start_screen_xy"), field="piece_start_screen_xy")
    end_pixel = _pair(shared.get("slot_center_screen_xy"), field="slot_center_screen_xy")
    start_xy = _model_xy_from_pixel(start_pixel, viewport)
    end_xy = _model_xy_from_pixel(end_pixel, viewport)
    if start_xy is None or end_xy is None:
        return _reject(source, variant, "third_person_drag_coordinate_out_of_range")
    free_move, down, held_move, up = _drag_indices(source)
    single = _trace(
        source,
        images=(
            source.images[0],
            source.images[free_move + 1],
            source.images[down + 1],
            source.images[held_move + 1],
            source.images[up + 1],
        ),
        actions=(
            _action("move_to", x=start_xy[0], y=start_xy[1]),
            _action("mouse_down"),
            _action("move_to", x=end_xy[0], y=end_xy[1]),
            _action("mouse_up"),
        ),
    )
    atomic = _trace(
        source,
        images=(source.images[0], source.images[up + 1]),
        actions=(_point_action("drag", (start_xy, end_xy)),),
    )
    return MatchedConversion(atomic=atomic, single=single, rejection=None)


def _drag_first_person_conversion(
    source: PreparedTrace,
    *,
    episode: Mapping[str, object],
    variant: str,
) -> MatchedConversion:
    shared = _shared(episode)
    dynamics = shared.get("hidden_dynamics")
    if not isinstance(dynamics, dict):
        raise AblationDatasetError(f"{source.record_id}: hidden dynamics are missing")
    environment = episode.get("environment_config")
    if not isinstance(environment, dict):
        raise AblationDatasetError(f"{source.record_id}: environment config is missing")
    viewport = _episode_viewport(episode)
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    initial_view = _pair(
        environment.get(
            "initial_view_center_xy",
            shared.get("initial_view_center_xy"),
        ),
        field="initial_view_center_xy",
    )
    piece = _pair(shared.get("piece_start_world_xy"), field="piece_start_world_xy")
    target = _pair(shared.get("slot_center_world_xy"), field="slot_center_world_xy")
    world_size = _pair(shared.get("world_size"), field="world_size")
    sensitivity = float(dynamics["sensitivity"])
    zoom = float(dynamics.get("view_zoom", 1.0))

    aim_view = _clamp_view_center(
        piece, world_size=world_size, viewport=viewport, zoom=zoom
    )
    grab_offset = (piece[0] - aim_view[0], piece[1] - aim_view[1])
    desired_final_view = (target[0] - grab_offset[0], target[1] - grab_offset[1])
    final_view = _clamp_view_center(
        desired_final_view, world_size=world_size, viewport=viewport, zoom=zoom
    )
    final_piece = _clamp_world(
        (final_view[0] + grab_offset[0], final_view[1] + grab_offset[1]),
        world_size,
    )
    final_distance_px = math.dist(final_piece, target) * zoom
    evaluator = episode.get("success_evaluator")
    if not isinstance(evaluator, dict):
        raise AblationDatasetError(f"{source.record_id}: success evaluator is missing")
    tolerance_px = float(evaluator["tolerance_px"])
    if final_distance_px > tolerance_px + 1e-6:
        return _reject(
            source,
            variant,
            "one_step_geometry_cannot_reach_target",
            final_distance_px=round(final_distance_px, 6),
            tolerance_px=tolerance_px,
        )

    aim_pointer = (
        (aim_view[0] - initial_view[0]) * zoom / sensitivity,
        (aim_view[1] - initial_view[1]) * zoom / sensitivity,
    )
    held_pointer = (
        (final_view[0] - aim_view[0]) * zoom / sensitivity,
        (final_view[1] - aim_view[1]) * zoom / sensitivity,
    )
    start_pixel = (center[0] + aim_pointer[0], center[1] + aim_pointer[1])
    held_pixel = (center[0] + held_pointer[0], center[1] + held_pointer[1])
    end_pixel = (start_pixel[0] + held_pointer[0], start_pixel[1] + held_pointer[1])
    start_xy = _model_xy_from_pixel(start_pixel, viewport)
    held_xy = _model_xy_from_pixel(held_pixel, viewport)
    end_xy = _model_xy_from_pixel(end_pixel, viewport)
    if start_xy is None or held_xy is None or end_xy is None:
        return _reject(
            source,
            variant,
            "one_step_drag_coordinate_out_of_range",
            start_pixel=[round(value, 6) for value in start_pixel],
            held_pixel=[round(value, 6) for value in held_pixel],
            end_pixel=[round(value, 6) for value in end_pixel],
        )

    free_move, down, held_move, up = _drag_indices(source)
    single = _trace(
        source,
        images=(
            source.images[0],
            source.images[free_move + 1],
            source.images[down + 1],
            source.images[held_move + 1],
            source.images[up + 1],
        ),
        actions=(
            _action("move_to", x=start_xy[0], y=start_xy[1]),
            _action("mouse_down"),
            _action("move_to", x=held_xy[0], y=held_xy[1]),
            _action("mouse_up"),
        ),
    )
    atomic = _trace(
        source,
        images=(source.images[0], source.images[up + 1]),
        actions=(_point_action("drag", (start_xy, end_xy)),),
    )
    return MatchedConversion(atomic=atomic, single=single, rejection=None)


def convert_trace_pair(
    source: PreparedTrace,
    *,
    episode: Mapping[str, object],
    variant: str,
) -> MatchedConversion:
    if str(episode.get("episode_id")) != source.record_id:
        raise AblationDatasetError(f"{source.record_id}: episode id differs")
    if str(episode.get("variant")) != variant:
        raise AblationDatasetError(f"{source.record_id}: episode variant differs")
    if variant in {"ten_choice_third_person", "ten_choice_first_person"}:
        return _ten_choice_conversion(source, episode=episode, variant=variant)
    if variant in {"rotation_inner", "rotation_outer"}:
        return _rotation_conversion(source, episode=episode, variant=variant)
    if variant == "drag_third_person":
        return _drag_third_person_conversion(source, episode=episode, variant=variant)
    if variant == "drag_first_person":
        return _drag_first_person_conversion(source, episode=episode, variant=variant)
    raise KeyError(variant)


def _allowed_action_kinds(ablation_variant: str, environment_variant: str) -> tuple[str, ...]:
    if ablation_variant == SINGLE_MOVE_VARIANT:
        return SIX_ACTION_KINDS
    if environment_variant.startswith("ten_choice_"):
        return ("click",)
    return ("drag",)


def _action_prompt(instruction: str, action_kinds: tuple[str, ...]) -> str:
    display = ", ".join(action_kinds)
    return "\n".join(
        (
            f"Task requirement: {instruction}",
            "Action space for this task:",
            f"Allowed action kinds: {display}",
            "Action schemas:",
            *(_INTEGER_ACTION_SCHEMA_LINES[kind] for kind in action_kinds),
            "Choose exactly one next action from the current screenshot and causal history.",
            "Coordinates are integers in the 0-1000 screenshot-relative system: x increases left to right and y top to bottom.",
            'Return exactly <think></think> followed by one compact JSON object: {"action":{...}}.',
            "Do not place any text inside the think tags and do not add Markdown or other text.",
        )
    )


def _messages(
    trace: TransformedTrace,
    *,
    action_index: int,
    action_kinds: tuple[str, ...],
) -> list[dict[str, object]]:
    prior_responses = [empty_think_response(action) for action in trace.actions[:action_index]]
    causal_images = trace.images[: action_index + 1]
    retained_images = causal_images[-IMAGE_HISTORY_MAX:]
    first_retained_index = len(causal_images) - len(retained_images)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _action_prompt(trace.instruction, action_kinds)}
    ]

    def append_prior(index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {index + 1} assistant response (context only):\n"
                    f"{prior_responses[index]}\n"
                ),
            }
        )

    for index in range(first_retained_index):
        append_prior(index)
    for retained_offset, image_path in enumerate(retained_images):
        content.append({"type": "image", "image": str(image_path)})
        prior_index = first_retained_index + retained_offset
        if prior_index < action_index:
            append_prior(prior_index)
    history = (
        f"{action_index} previous assistant response(s) are included in chronological order; "
        "older screenshots may be omitted, but all previous responses are retained."
        if action_index
        else "No previous actions are present in this trajectory."
    )
    content.append(
        {
            "type": "text",
            "text": "\n".join(
                (
                    history,
                    "The last attached image is the current observation.",
                    f"Return <think></think> followed by the JSON action for step {action_index + 1}.",
                )
            ),
        }
    )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": empty_think_response(trace.actions[action_index])},
    ]


def _write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _record_payload(
    trace: TransformedTrace,
    *,
    environment_variant: str,
    ablation_variant: str,
) -> dict[str, object]:
    return {
        "id": trace.record_id,
        "instruction": trace.instruction,
        "environment_variant": environment_variant,
        "ablation_variant": ablation_variant,
        "images": [str(path) for path in trace.images],
        "actions": [dict(action) for action in trace.actions],
        "metadata": {
            "exploration_level": trace.exploration_level,
            "history_contract": HISTORY_CONTRACT,
            "prompt_contract": PROMPT_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
            "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
            "matched_across_arms": True,
        },
    }


def _build_dataset(
    traces: list[TransformedTrace],
    *,
    environment_variant: str,
    ablation_variant: str,
    output_root: Path,
    shuffle_seed: int,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=False)
    records_path = output_root / "records.jsonl"
    train_path = output_root / "train.parquet"
    summary_path = output_root / "summary.json"
    ready_path = output_root / "READY.json"
    _write_jsonl(
        records_path,
        (
            _record_payload(
                trace,
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
            )
            for trace in traces
        ),
    )

    references = [
        (trace_index, action_index)
        for trace_index, trace in enumerate(traces)
        for action_index in range(len(trace.actions))
    ]
    random.Random(shuffle_seed).shuffle(references)
    action_kinds = _allowed_action_kinds(ablation_variant, environment_variant)
    action_counts: Counter[str] = Counter()
    temporary_train = train_path.with_name(f".{train_path.name}.tmp-{os.getpid()}")
    sink = _ParquetSink(temporary_train)
    try:
        for row_index, (trace_index, action_index) in enumerate(references):
            trace = traces[trace_index]
            action = trace.actions[action_index]
            action_counts[str(action["kind"])] += 1
            row = messages_to_verl_row(
                _messages(trace, action_index=action_index, action_kinds=action_kinds),
                image_max_pixels=IMAGE_MAX_PIXELS,
                validate_image_files=False,
                metadata={
                    "source_index": trace_index,
                    "source_record_id": trace.record_id,
                    "action_index": action_index,
                    "row_index": row_index,
                    "task": environment_variant,
                    "variant": environment_variant,
                    "environment_variant": environment_variant,
                    "ablation_variant": ablation_variant,
                    "exploration_level": trace.exploration_level,
                    "image_history_max": IMAGE_HISTORY_MAX,
                    "image_max_pixels": IMAGE_MAX_PIXELS,
                    "history_contract": HISTORY_CONTRACT,
                    "prompt_contract": PROMPT_CONTRACT,
                    "response_contract": RESPONSE_CONTRACT,
                    "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
                    "allowed_action_kinds": list(action_kinds),
                    "matched_across_arms": True,
                    "no_think": True,
                },
            )
            sink.append(row)
        sink.close()
        os.replace(temporary_train, train_path)
    finally:
        if sink.writer is not None:
            sink.writer.close()
        temporary_train.unlink(missing_ok=True)

    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(train_path)
    if parquet.metadata.num_rows != len(references):
        raise AblationDatasetError(f"{train_path}: parquet row count differs")
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_variant": environment_variant,
        "ablation_variant": ablation_variant,
        "trace_count": len(traces),
        "train_rows": len(references),
        "action_counts": dict(sorted(action_counts.items())),
        "allowed_action_kinds": list(action_kinds),
        "matching_policy": MATCHING_POLICY,
        "history_contract": HISTORY_CONTRACT,
        "prompt_contract": PROMPT_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "image_history_max": IMAGE_HISTORY_MAX,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "shuffle_seed": shuffle_seed,
        "train_parquet": str(train_path.resolve()),
        "records_jsonl": str(records_path.resolve()),
    }
    _atomic_json(summary_path, summary)
    ready: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ready",
        "environment_variant": environment_variant,
        "ablation_variant": ablation_variant,
        "trace_count": len(traces),
        "train_rows": len(references),
        "action_counts": dict(sorted(action_counts.items())),
        "validation": "full_source_contract_coordinate_and_parquet_row_validation",
        "files": {
            "train.parquet": {"size_bytes": train_path.stat().st_size},
            "records.jsonl": {"size_bytes": records_path.stat().st_size},
            "summary.json": {"size_bytes": summary_path.stat().st_size},
        },
    }
    _atomic_json(ready_path, ready)
    return summary


def build_action_ablation_datasets(
    *,
    input_root: Path,
    manifest_path: Path,
    output_root: Path,
    shuffle_seed: int = SHUFFLE_SEED,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_id") != SOURCE_SUITE_ID:
        raise AblationDatasetError(
            f"manifest suite must be {SOURCE_SUITE_ID}, got {manifest.get('suite_id')!r}"
        )
    source_ready_path = input_root / "READY.json"
    source_ready = json.loads(source_ready_path.read_text(encoding="utf-8"))
    if (
        source_ready.get("schema") != SOURCE_RAW_SCHEMA
        or source_ready.get("status") != "ready"
        or int(source_ready.get("trace_count", -1))
        != EXPECTED_SOURCE_TRACES * len(ENVIRONMENT_VARIANTS)
        or int(source_ready.get("action_count", -1)) != EXPECTED_SOURCE_ACTIONS
    ):
        raise AblationDatasetError(f"current v3 raw READY contract failed: {source_ready_path}")
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, list):
        raise AblationDatasetError("manifest has no episodes")
    episodes = {
        (str(episode["variant"]), str(episode["episode_id"])): episode
        for episode in raw_episodes
        if isinstance(episode, dict)
    }
    for episode in episodes.values():
        variant = str(episode.get("variant"))
        environment = episode.get("environment_config")
        if not isinstance(environment, dict):
            raise AblationDatasetError(f"{episode.get('episode_id')}: environment config is missing")
        if (
            variant in {"ten_choice_first_person", "drag_first_person"}
            and environment.get("interaction_marker") != FIRST_PERSON_MARKER_CONTRACT
        ):
            raise AblationDatasetError(
                f"{episode.get('episode_id')}: first-person interaction marker is not mouse_icon"
            )
        if (
            variant in {"rotation_inner", "rotation_outer"}
            and environment.get("gesture_contract") != ROTATION_GESTURE_CONTRACT
        ):
            raise AblationDatasetError(
                f"{episode.get('episode_id')}: rotation gesture contract permits retries"
            )

    accepted_by_variant: dict[str, dict[str, list[TransformedTrace]]] = {}
    rejections_by_variant: dict[str, list[dict[str, object]]] = {}
    rejection_counts: dict[str, dict[str, int]] = {}
    for environment_variant in ENVIRONMENT_VARIANTS:
        source_path = input_root / environment_variant / "train.jsonl"
        sources = [
            _prepare_trace(record, source_path=source_path)
            for record in _read_jsonl(source_path)
        ]
        if len(sources) != EXPECTED_SOURCE_TRACES:
            raise AblationDatasetError(
                f"{environment_variant}: expected {EXPECTED_SOURCE_TRACES} traces, got {len(sources)}"
            )
        atomic: list[TransformedTrace] = []
        single: list[TransformedTrace] = []
        rejected: list[dict[str, object]] = []
        for source in sources:
            episode = episodes.get((environment_variant, source.record_id))
            if episode is None:
                raise AblationDatasetError(f"{source.record_id}: manifest episode is missing")
            converted = convert_trace_pair(
                source,
                episode=episode,
                variant=environment_variant,
            )
            if converted.accepted():
                assert converted.atomic is not None and converted.single is not None
                atomic.append(converted.atomic)
                single.append(converted.single)
            else:
                assert converted.rejection is not None
                rejected.append(converted.rejection)
        if [trace.record_id for trace in atomic] != [trace.record_id for trace in single]:
            raise AblationDatasetError(f"{environment_variant}: arm record ids differ")
        accepted_by_variant[environment_variant] = {
            ATOMIC_VARIANT: atomic,
            SINGLE_MOVE_VARIANT: single,
        }
        rejections_by_variant[environment_variant] = rejected
        rejection_counts[environment_variant] = dict(
            sorted(Counter(str(item["reason"]) for item in rejected).items())
        )

    output_root.mkdir(parents=True, exist_ok=False)
    rejection_root = output_root / "rejections"
    for environment_variant, rejected in rejections_by_variant.items():
        _write_jsonl(rejection_root / f"{environment_variant}.jsonl", rejected)

    datasets: dict[str, dict[str, object]] = {}
    for ablation_index, ablation_variant in enumerate(ABLATION_VARIANTS):
        datasets[ablation_variant] = {}
        for environment_index, environment_variant in enumerate(ENVIRONMENT_VARIANTS):
            dataset_root = output_root / ablation_variant / environment_variant
            summary = _build_dataset(
                accepted_by_variant[environment_variant][ablation_variant],
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
                output_root=dataset_root,
                shuffle_seed=shuffle_seed + ablation_index * 100 + environment_index,
            )
            datasets[ablation_variant][environment_variant] = summary

    matched_counts = {
        environment_variant: len(
            accepted_by_variant[environment_variant][ATOMIC_VARIANT]
        )
        for environment_variant in ENVIRONMENT_VARIANTS
    }
    top_summary: dict[str, object] = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(input_root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "source_suite_id": SOURCE_SUITE_ID,
        "source_raw_ready": str(source_ready_path.resolve()),
        "output_root": str(output_root.resolve()),
        "ablation_variants": list(ABLATION_VARIANTS),
        "environment_variants": list(ENVIRONMENT_VARIANTS),
        "dataset_count": len(ABLATION_VARIANTS) * len(ENVIRONMENT_VARIANTS),
        "source_trace_count_per_environment": EXPECTED_SOURCE_TRACES,
        "matched_trace_count_per_environment": matched_counts,
        "rejected_trace_count_per_environment": {
            variant: EXPECTED_SOURCE_TRACES - matched_counts[variant]
            for variant in ENVIRONMENT_VARIANTS
        },
        "rejection_counts_per_environment": rejection_counts,
        "matching_policy": MATCHING_POLICY,
        "history_contract": HISTORY_CONTRACT,
        "prompt_contract": PROMPT_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "first_person_marker_contract": FIRST_PERSON_MARKER_CONTRACT,
        "rotation_gesture_contract": ROTATION_GESTURE_CONTRACT,
        "datasets": datasets,
    }
    _atomic_json(output_root / "summary.json", top_summary)
    top_ready: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ready",
        "dataset_count": 12,
        "matched_trace_count_per_environment": matched_counts,
        "rejected_trace_count_per_environment": {
            variant: EXPECTED_SOURCE_TRACES - matched_counts[variant]
            for variant in ENVIRONMENT_VARIANTS
        },
        "matching_policy": MATCHING_POLICY,
        "source_suite_id": SOURCE_SUITE_ID,
        "first_person_marker_contract": FIRST_PERSON_MARKER_CONTRACT,
        "rotation_gesture_contract": ROTATION_GESTURE_CONTRACT,
        "validation": "all_12_parquets_materialized_and_row_counts_checked",
        "summary": str((output_root / "summary.json").resolve()),
    }
    _atomic_json(output_root / "READY.json", top_ready)
    return {"summary": top_summary, "ready": top_ready}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_action_ablation_datasets(
        input_root=args.input_root,
        manifest_path=args.manifest,
        output_root=args.output_root,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(result["ready"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
