from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from ...actions import PrimitiveAction
from ...protocol_tracks import (
    CANONICAL_ASSISTANT_FIELD,
    CANONICAL_THINK_FIELD,
    PROTOCOL_VERSION,
    render_action_assistant_response,
)
from .paths import episodes_root, legacy_sft_root

DEFAULT_TEST_COUNT = 150
DEFAULT_VIEWPORT = (1280, 800)
DEFAULT_WORLD_SIZE = (2200, 1600)
DEFAULT_SLOT_RADIUS_SCALE = 1.35
DEFAULT_TOLERANCE_RADIUS_SCALE = 1.15
DEFAULT_SLOT_COLOR_NAME = "gray"
DEFAULT_SLOT_COLOR = "#94a3b8"
DEFAULT_SENSITIVITY_VALUES = (0.8, 0.9, 1.0, 1.1, 1.25)
TASK_DESCRIPTION = "Place the solid colored shape into the matching gray outline using the center reticle."
SFT_SOURCE = "slot_drag_game_synthetic_closed_loop"
DEFAULT_SFT_COUNT = 100
DEFAULT_SFT_MAX_STEPS = 12

SHAPES = ("square", "triangle", "diamond", "hexagon")
SLOT_SHAPES = ("square", "triangle", "diamond", "hexagon", "notched_rectangle", "wide_rectangle")
COLORS = (
    ("emerald", "#16a34a"),
    ("blue", "#2563eb"),
    ("amber", "#d97706"),
    ("lime", "#65a30d"),
    ("indigo", "#4f46e5"),
)
TRAJECTORY_PATTERNS = (
    "monotonic_coarse_to_fine",
    "undershoot_correction",
    "overshoot_correction",
    "left_right_probe",
    "far_then_fine",
)
MOUSE_DOWN_POSITION_WEIGHTS = (
    (3, 1),
    (4, 2),
    (5, 4),
    (6, 2),
    (7, 1),
)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _object_description(*, color_name: str, shape: str) -> str:
    return f"solid {color_name} {shape}"


def _slot_description(*, shape: str) -> str:
    return f"large hollow gray {shape.replace('_', ' ')} slot"


def _shape_mismatches(target_shape: str) -> list[str]:
    return [shape for shape in SLOT_SHAPES if shape != target_shape]


def _move_to_for_view_delta(
    view_delta: tuple[float, float],
    *,
    viewport: tuple[int, int],
    sensitivity: float,
    view_zoom: float,
) -> dict[str, float | str]:
    center_x = viewport[0] / 2.0
    center_y = viewport[1] / 2.0
    pixel_x = center_x + view_delta[0] * view_zoom / sensitivity
    pixel_y = center_y + view_delta[1] * view_zoom / sensitivity
    if not (0.0 <= pixel_x <= viewport[0] and 0.0 <= pixel_y <= viewport[1]):
        raise ValueError(f"oracle move_to is outside the viewport: {(pixel_x, pixel_y)}")
    return {
        "kind": "move_to",
        "x": round(pixel_x / viewport[0] * 1000.0, 3),
        "y": round(pixel_y / viewport[1] * 1000.0, 3),
    }


def _clean_trainable_action(action: dict[str, Any]) -> dict[str, Any]:
    kind = str(action["kind"])
    cleaned: dict[str, Any] = {"kind": kind}
    if kind == "move_to":
        cleaned["x"] = action["x"]
        cleaned["y"] = action["y"]
    return cleaned


def _primitive_from_action(action: dict[str, Any]) -> PrimitiveAction:
    if action["kind"] == "move_to":
        return PrimitiveAction(kind="move_to", x=float(action["x"]), y=float(action["y"]))
    return PrimitiveAction(kind=action["kind"])


def _observation_step(observation: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "observation",
        "image_path": str(Path(observation.screenshot_path).resolve()),
    }
    if observation.cursor_xy is not None and observation.size_px is not None:
        width, height = observation.size_px
        step["cursor_xy"] = [
            round(float(observation.cursor_xy[0]) / width * 1000.0, 3),
            round(float(observation.cursor_xy[1]) / height * 1000.0, 3),
        ]
    return step


def _action_step(action: dict[str, Any], *, thought: str) -> dict[str, Any]:
    cleaned = _clean_trainable_action(action)
    step: dict[str, Any] = {
        "type": "action",
        "kind": cleaned["kind"],
        CANONICAL_THINK_FIELD: thought.strip(),
        CANONICAL_ASSISTANT_FIELD: render_action_assistant_response(thought, cleaned),
    }
    if cleaned["kind"] == "move_to":
        step["x"] = cleaned["x"]
        step["y"] = cleaned["y"]
    return step


def _coerce_xy(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected xy pair, got {value!r}")
    return (float(value[0]), float(value[1]))


def _clamp_center(
    center_xy: tuple[float, float],
    *,
    world_size: tuple[int, int],
    viewport: tuple[int, int],
    view_zoom: float,
) -> tuple[float, float]:
    zoom = max(float(view_zoom), 1e-6)
    half_w = viewport[0] / 2.0 / zoom
    half_h = viewport[1] / 2.0 / zoom
    return (
        min(max(center_xy[0], half_w), world_size[0] - half_w),
        min(max(center_xy[1], half_h), world_size[1] - half_h),
    )


def _monotonic_fractions(step_count: int, *, start: float) -> list[float]:
    if step_count <= 1:
        return [1.0]
    return [
        start + (1.0 - start) * index / (step_count - 1)
        for index in range(step_count)
    ]


def _trajectory_offsets(pattern: str, *, step_count: int) -> list[tuple[float, float]]:
    if step_count < 1:
        raise ValueError("step_count must be positive")
    if pattern == "overshoot_correction":
        if step_count == 1:
            fractions = [1.0]
        elif step_count == 2:
            fractions = [1.06, 1.0]
        elif step_count == 3:
            fractions = [0.62, 1.08, 1.0]
        else:
            base = [0.48, 0.76, 1.08, 0.92, 1.04, 0.97, 1.02]
            fractions = [
                base[min(index, len(base) - 1)]
                if index < step_count - 1
                else 1.0
                for index in range(step_count)
            ]
        return [(fraction, 0.0) for fraction in fractions]
    if pattern == "left_right_probe":
        fractions = _monotonic_fractions(step_count, start=0.42)
        if step_count >= 3:
            fractions[-2] = min(1.06, max(fractions[-2], 1.04))
        lateral_cycle = (0.10, -0.12, 0.08, -0.05, 0.04, -0.03)
        return [
            (fraction, 0.0 if index == step_count - 1 else lateral_cycle[index % len(lateral_cycle)])
            for index, fraction in enumerate(fractions)
        ]
    if pattern == "far_then_fine":
        return [(fraction, 0.0) for fraction in _monotonic_fractions(step_count, start=0.35)]
    if pattern == "undershoot_correction":
        return [(fraction, 0.0) for fraction in _monotonic_fractions(step_count, start=0.50)]
    return [(fraction, 0.0) for fraction in _monotonic_fractions(step_count, start=0.45)]


def _trajectory_centers(
    *,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
    pattern: str,
    step_count: int,
    world_size: tuple[int, int],
    viewport: tuple[int, int],
    view_zoom: float,
) -> list[tuple[float, float]]:
    dx = target_xy[0] - start_xy[0]
    dy = target_xy[1] - start_xy[1]
    norm = max(math.hypot(dx, dy), 1e-6)
    perp = (-dy / norm, dx / norm)
    centers: list[tuple[float, float]] = []
    for fraction, perp_fraction in _trajectory_offsets(pattern, step_count=step_count):
        lateral = min(norm * abs(perp_fraction), 42.0) * (1.0 if perp_fraction >= 0 else -1.0)
        raw = (
            start_xy[0] + dx * fraction + perp[0] * lateral,
            start_xy[1] + dy * fraction + perp[1] * lateral,
        )
        clamped = _clamp_center(raw, world_size=world_size, viewport=viewport, view_zoom=view_zoom)
        if not centers or math.dist(centers[-1], clamped) > 1e-3:
            centers.append(clamped)
    final = _clamp_center(target_xy, world_size=world_size, viewport=viewport, view_zoom=view_zoom)
    if not centers or math.dist(centers[-1], final) > 1e-3:
        centers.append(final)
    return centers


def _weighted_mouse_down_positions(*, count: int, seed: int) -> list[int]:
    total_weight = sum(weight for _, weight in MOUSE_DOWN_POSITION_WEIGHTS)
    counts: dict[int, int] = {}
    remainders: list[tuple[float, int]] = []
    for position, weight in MOUSE_DOWN_POSITION_WEIGHTS:
        exact = count * weight / total_weight
        base = math.floor(exact)
        counts[position] = base
        remainders.append((exact - base, position))
    remaining = count - sum(counts.values())
    remainders.sort(key=lambda item: (-item[0], abs(item[1] - 5), item[1]))
    for _, position in remainders[:remaining]:
        counts[position] += 1

    positions: list[int] = []
    for position, _ in MOUSE_DOWN_POSITION_WEIGHTS:
        positions.extend([position] * counts[position])
    rng = random.Random(seed + 104729)
    rng.shuffle(positions)
    return positions


def _step_pair_for_mouse_down_position(position: int) -> tuple[int, int]:
    if position not in {3, 4, 5, 6, 7}:
        raise ValueError(f"unsupported mouse_down position: {position}")
    aim_step_count = position - 1
    placement_step_count = DEFAULT_SFT_MAX_STEPS - aim_step_count - 2
    return aim_step_count, placement_step_count


def _has_cross_target_retry(
    centers: list[tuple[float, float]],
    *,
    start_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> bool:
    if len(centers) < 2:
        return False
    dx = target_xy[0] - start_xy[0]
    dy = target_xy[1] - start_xy[1]
    norm_sq = dx * dx + dy * dy
    if norm_sq <= 1e-6:
        return False
    projections = [
        ((center[0] - start_xy[0]) * dx + (center[1] - start_xy[1]) * dy) / norm_sq
        for center in centers
    ]
    return any(value > 1.02 for value in projections[:-1]) and abs(projections[-1] - 1.0) <= 0.02


def _thought_for_action(
    *,
    phase: str,
    action: dict[str, Any],
    action_index: int,
    trajectory_pattern: str,
    is_final_move: bool = False,
) -> str:
    kind = action["kind"]
    if kind == "mouse_down":
        return (
            "The solid piece is now under the fixed center reticle. I should press and hold so later "
            "view movement carries the piece with the reticle."
        )
    if kind == "mouse_up":
        return (
            "The matching gray outline is under the center reticle and the held piece has been corrected "
            "onto it, so I release to submit the placement."
        )
    if phase == "aim":
        if is_final_move:
            return (
                "The previous aiming moves have narrowed the offset to the solid piece. I now make the "
                "final small correction so the center reticle sits on the piece before pressing."
            )
        if "overshoot" in trajectory_pattern or "probe" in trajectory_pattern:
            return (
                "I am still estimating how the center-locked view responds. This move intentionally probes "
                "around the piece so the next screenshot can show the remaining correction."
            )
        return (
            "The center reticle is not yet on the movable piece. I move the view partway toward it and will "
            "use the next screenshot to correct the remaining offset."
        )
    if is_final_move:
        return (
            "The held piece is close to the matching slot after the earlier adjustment. I make the final "
            "correction so the piece lands inside the gray outline."
        )
    if "overshoot" in trajectory_pattern or "probe" in trajectory_pattern:
        return (
            "While holding the piece, I deliberately move across the target side to expose the current "
            "sensitivity, then the following step can correct back toward the slot."
        )
    return (
        "The piece is being dragged but the slot is still offset from the reticle. I move closer in a "
        "controlled step and keep the button held for the next correction."
    )


def _build_closed_loop_actions(
    meta: dict[str, Any],
    *,
    pattern: str,
    aim_step_count: int,
    placement_step_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    viewport = tuple(int(value) for value in meta.get("viewport", DEFAULT_VIEWPORT))
    world_size = tuple(int(value) for value in meta.get("world_size", DEFAULT_WORLD_SIZE))
    sensitivity = float(meta.get("sensitivity", 1.0))
    view_zoom = float(meta.get("view_zoom", 1.0))
    initial_center = _coerce_xy(meta["initial_view_center_xy"])
    piece_xy = _coerce_xy(meta["piece_start_world_xy"])
    slot_xy = _coerce_xy(meta["slot_center_world_xy"])

    aim_centers = _trajectory_centers(
        start_xy=initial_center,
        target_xy=piece_xy,
        pattern=pattern,
        step_count=aim_step_count,
        world_size=world_size,
        viewport=viewport,
        view_zoom=view_zoom,
    )
    placement_centers = _trajectory_centers(
        start_xy=piece_xy,
        target_xy=slot_xy,
        pattern=pattern,
        step_count=placement_step_count,
        world_size=world_size,
        viewport=viewport,
        view_zoom=view_zoom,
    )

    actions: list[dict[str, Any]] = []
    current_center = initial_center
    for center in aim_centers:
        actions.append(
            _move_to_for_view_delta(
                (center[0] - current_center[0], center[1] - current_center[1]),
                viewport=viewport,
                sensitivity=sensitivity,
                view_zoom=view_zoom,
            )
        )
        current_center = center
    actions.append({"kind": "mouse_down"})
    current_center = piece_xy
    for center in placement_centers:
        actions.append(
            _move_to_for_view_delta(
                (center[0] - current_center[0], center[1] - current_center[1]),
                viewport=viewport,
                sensitivity=sensitivity,
                view_zoom=view_zoom,
            )
        )
        current_center = center
    actions.append({"kind": "mouse_up"})
    metadata = {
        "aim_step_count": len(aim_centers),
        "success_step_count": len(placement_centers),
        "mouse_down_action_position": len(aim_centers) + 1,
        "action_step_count": len(actions),
        "max_steps": DEFAULT_SFT_MAX_STEPS,
        "trajectory_pattern": pattern,
        "aim_has_cross_target_retry": _has_cross_target_retry(
            aim_centers,
            start_xy=initial_center,
            target_xy=piece_xy,
        ),
        "placement_has_cross_target_retry": _has_cross_target_retry(
            placement_centers,
            start_xy=piece_xy,
            target_xy=slot_xy,
        ),
        "aim_view_centers": [[round(x, 3), round(y, 3)] for x, y in aim_centers],
        "placement_view_centers": [[round(x, 3), round(y, 3)] for x, y in placement_centers],
    }
    if len(actions) > DEFAULT_SFT_MAX_STEPS:
        raise ValueError(
            f"closed-loop trace has {len(actions)} actions, exceeding max_steps={DEFAULT_SFT_MAX_STEPS}"
        )
    return actions, metadata


def _trace_record(
    *,
    index: int,
    record_id: str,
    before_observation: Any,
    action: dict[str, Any],
    result: Any,
    thought: str,
    trajectory_pattern: str,
) -> dict[str, Any]:
    info = result.info
    return {
        "index": index,
        "benchmark": "SlotDragGame",
        "task_type": "slot_drag_game",
        "task_id": record_id,
        "episode_id": record_id,
        "action_kind": action["kind"],
        "action": _clean_trainable_action(action),
        "assistant_response": render_action_assistant_response(thought, _clean_trainable_action(action)),
        "success": info.get("success", False),
        "done": bool(result.done),
        "cursor_error_px": info.get("piece_slot_distance_px"),
        "was_corrective": action["kind"] == "move_to" and index > 0,
        "trajectory_pattern": trajectory_pattern,
        "before_screenshot_path": before_observation.screenshot_path,
        "initial_screenshot_path": before_observation.screenshot_path,
        "initial_metadata": before_observation.metadata,
        "obs_instruction": result.observation.instruction,
        "obs_screenshot_path": result.observation.screenshot_path,
        "obs_metadata": result.observation.metadata,
        "env_info": {
            key: value
            for key, value in info.items()
            if key not in {"action"}
        },
    }


def _clean_sft_output_dir(output_dir: Path) -> None:
    for name in (
        "episodes",
        "screenshots",
        "traces",
        "records.jsonl",
        "train.jsonl",
        "audit.jsonl",
        "excluded.jsonl",
        "summary.json",
        "validation_report.json",
        "README.md",
    ):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _sample_offset(
    rng: random.Random,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_norm: float,
) -> tuple[float, float]:
    for _ in range(1000):
        dx = rng.uniform(*x_range) * rng.choice([-1, 1])
        dy = rng.uniform(*y_range) * rng.choice([-1, 1])
        if (dx * dx + dy * dy) ** 0.5 >= min_norm:
            return dx, dy
    return x_range[1], y_range[0]


def _slot_options(
    rng: random.Random,
    *,
    target_xy: tuple[float, float],
    target_shape: str,
    radius: float,
) -> list[dict[str, Any]]:
    mismatch_shapes = _shape_mismatches(target_shape)
    options = [
        {
            "id": "slot_target",
            "role": "target",
            "center_world_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
            "shape": target_shape,
            "color_name": DEFAULT_SLOT_COLOR_NAME,
            "color": DEFAULT_SLOT_COLOR,
        }
    ]
    distractor_specs = [
        ("slot_decoy_shape_a", mismatch_shapes[0]),
        ("slot_decoy_shape_b", mismatch_shapes[1]),
        ("slot_decoy_shape_c", mismatch_shapes[2]),
        ("slot_decoy_shape_d", mismatch_shapes[3]),
        ("slot_decoy_shape_e", mismatch_shapes[4]),
    ]
    used = [target_xy]
    for index, (slot_id, shape) in enumerate(distractor_specs):
        for _ in range(1000):
            angle = rng.uniform(0, 6.28318)
            distance = rng.uniform(radius * 4.0, radius * 8.5)
            candidate = (
                target_xy[0] + math.cos(angle) * distance,
                target_xy[1] + math.sin(angle) * distance,
            )
            if all(((candidate[0] - xy[0]) ** 2 + (candidate[1] - xy[1]) ** 2) ** 0.5 >= radius * 3.0 for xy in used):
                used.append(candidate)
                options.append(
                    {
                        "id": slot_id,
                        "role": "distractor",
                        "center_world_xy": [round(candidate[0], 3), round(candidate[1], 3)],
                        "shape": shape,
                        "color_name": DEFAULT_SLOT_COLOR_NAME,
                        "color": DEFAULT_SLOT_COLOR,
                    }
                )
                break
        else:
            fallback = (target_xy[0] + (index + 1) * radius * 4.0, target_xy[1] + (index + 1) * radius * 2.5)
            options.append(
                {
                    "id": slot_id,
                    "role": "distractor",
                    "center_world_xy": [round(fallback[0], 3), round(fallback[1], 3)],
                    "shape": shape,
                    "color_name": DEFAULT_SLOT_COLOR_NAME,
                    "color": DEFAULT_SLOT_COLOR,
                }
            )
    rng.shuffle(options)
    return options


def generate_dataset(
    *,
    output_dir: Path | None = None,
    test_count: int = DEFAULT_TEST_COUNT,
    seed: int = 20260626,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
    clean: bool = True,
) -> dict[str, Any]:
    output_dir = episodes_root() if output_dir is None else Path(output_dir)
    rng = random.Random(seed)
    viewport = (int(viewport[0]), int(viewport[1]))
    if viewport[0] < 1 or viewport[1] < 1:
        raise ValueError(f"viewport dimensions must be positive, got {viewport}")
    world_size = DEFAULT_WORLD_SIZE
    test_root = output_dir / "test"
    test_root.mkdir(parents=True, exist_ok=True)
    if clean:
        for child in test_root.glob("sdg_*"):
            if child.is_dir():
                shutil.rmtree(child)
    episode_ids: list[str] = []
    min_center_x = viewport[0] / 2.0 / 1.0
    max_center_x = world_size[0] - min_center_x
    min_center_y = viewport[1] / 2.0 / 1.0
    max_center_y = world_size[1] - min_center_y
    for index in range(test_count):
        episode_id = f"sdg_{index + 1:04d}"
        shape = SHAPES[index % len(SHAPES)]
        color_name, color = COLORS[index % len(COLORS)]
        radius = rng.choice([30, 32, 34, 36])
        tolerance = radius * DEFAULT_TOLERANCE_RADIUS_SCALE
        sensitivity = rng.choice(DEFAULT_SENSITIVITY_VALUES)
        view_zoom = 1.0
        for _ in range(1000):
            initial_center = (
                rng.uniform(900.0, world_size[0] - 900.0),
                rng.uniform(620.0, world_size[1] - 620.0),
            )
            piece_delta = _sample_offset(
                rng,
                x_range=(150.0, 390.0),
                y_range=(90.0, 245.0),
                min_norm=210.0,
            )
            slot_delta_from_piece = _sample_offset(
                rng,
                x_range=(230.0, 470.0),
                y_range=(80.0, 260.0),
                min_norm=280.0,
            )
            piece_xy = (initial_center[0] + piece_delta[0], initial_center[1] + piece_delta[1])
            slot_xy = (piece_xy[0] + slot_delta_from_piece[0], piece_xy[1] + slot_delta_from_piece[1])
            if (
                min_center_x <= piece_xy[0] <= max_center_x
                and min_center_y <= piece_xy[1] <= max_center_y
                and min_center_x <= slot_xy[0] <= max_center_x
                and min_center_y <= slot_xy[1] <= max_center_y
            ):
                break
        else:
            raise RuntimeError("failed to sample a reachable SlotDragGame episode")
        oracle = [
            _move_to_for_view_delta(
                (piece_xy[0] - initial_center[0], piece_xy[1] - initial_center[1]),
                viewport=viewport,
                sensitivity=sensitivity,
                view_zoom=view_zoom,
            ),
            {"kind": "mouse_down"},
            _move_to_for_view_delta(
                (slot_xy[0] - piece_xy[0], slot_xy[1] - piece_xy[1]),
                viewport=viewport,
                sensitivity=sensitivity,
                view_zoom=view_zoom,
            ),
            {"kind": "mouse_up"},
        ]
        meta = {
            "episode_id": episode_id,
            "version": 3,
            "benchmark": "SlotDragGame",
            "task_type": "slot_drag_game",
            "perspective": "first_person",
            "viewport": list(viewport),
            "world_size": list(world_size),
            "initial_view_center_xy": [round(initial_center[0], 3), round(initial_center[1], 3)],
            "sensitivity": sensitivity,
            "view_zoom": view_zoom,
            "piece_start_world_xy": [round(piece_xy[0], 3), round(piece_xy[1], 3)],
            "slot_center_world_xy": [round(slot_xy[0], 3), round(slot_xy[1], 3)],
            "slot_radius_scale": DEFAULT_SLOT_RADIUS_SCALE,
            "slot_frame_radius_px": round(radius * DEFAULT_SLOT_RADIUS_SCALE, 3),
            "tolerance_radius_scale": DEFAULT_TOLERANCE_RADIUS_SCALE,
            "tolerance_px": round(tolerance, 3),
            "movable_object_description": _object_description(color_name=color_name, shape=shape),
            "target_slot_description": _slot_description(shape=shape),
            "piece": {
                "shape": shape,
                "color_name": color_name,
                "color": color,
                "outline": "#111827",
                "radius_px": radius,
            },
            "slot_options": _slot_options(
                rng,
                target_xy=slot_xy,
                target_shape=shape,
                radius=radius,
            ),
            "instruction": TASK_DESCRIPTION,
            "oracle_primitive_sequence": oracle,
        }
        episode_dir = test_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        episode_ids.append(episode_id)
    manifest = {
        "benchmark": "SlotDragGame",
        "output_dir": str(output_dir),
        "test_count": test_count,
        "seed": seed,
        "viewport": list(viewport),
        "splits": {"test": episode_ids},
        "summary": {
            "shape_count": len(set(SHAPES[: min(test_count, len(SHAPES))])),
            "color_count": len(set(name for name, _ in COLORS[: min(test_count, len(COLORS))])),
            "slot_candidate_count_per_case": 6,
            "slot_distractor_count_per_case": 5,
            "shape_mismatch_slot_count_per_case": 5,
            "slot_color_name": DEFAULT_SLOT_COLOR_NAME,
            "slot_color": DEFAULT_SLOT_COLOR,
            "slot_colors_are_uniform": True,
            "slot_only_shapes": ["notched_rectangle", "wide_rectangle"],
            "piece_shapes_exclude_circle": True,
            "slot_radius_scale": DEFAULT_SLOT_RADIUS_SCALE,
            "tolerance_radius_scale": DEFAULT_TOLERANCE_RADIUS_SCALE,
            "cursor_lock": "center",
            "perspective": "first_person",
            "coordinate_contract": "qwen3_relative_0_1000_center_locked",
            "sensitivity_values": list(DEFAULT_SENSITIVITY_VALUES),
            "sensitivity_sampling": "task_wise_random_discrete",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def generate_sft_dataset(
    *,
    output_dir: Path | None = None,
    count: int = DEFAULT_SFT_COUNT,
    seed: int = 20260706,
    dataset_profile: str = "legacy",
    episode_namespace: str = "train",
    clean: bool = True,
) -> dict[str, Any]:
    """Generate success-only SlotDragGame closed-loop SFT traces.

    The exported trainable manifests intentionally mirror the rotation CAPTCHA
    data shape: each record is an observation/action alternating trace, every
    action has a canonical assistant response, and intermediate screenshots
    include off-target states before the final successful placement.
    """

    output_dir = legacy_sft_root() if output_dir is None else Path(output_dir)
    if count < 1:
        raise ValueError("count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_sft_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_root = output_dir / "episodes"
    if dataset_profile == "legacy":
        episode_manifest = generate_dataset(
            output_dir=episode_root,
            test_count=count,
            seed=seed,
            clean=True,
        )
        source_name = SFT_SOURCE
    elif dataset_profile == "paper_v2_shape_ood":
        from .dataset_v2 import generate_training_episode_dataset

        episode_manifest = generate_training_episode_dataset(
            output_dir=episode_root,
            count=count,
            seed=seed,
            episode_namespace=episode_namespace,
            clean=True,
        )
        episode_validation = episode_manifest.get("validation", {})
        if episode_validation.get("status") != "passed":
            raise RuntimeError(
                "SlotDragGame v2 episode validation failed before SFT rendering: "
                f"{episode_validation.get('issues', [])!r}"
            )
        source_name = "slot_drag_game_v2_synthetic_closed_loop"
    else:
        raise ValueError(f"unsupported SlotDragGame SFT dataset profile: {dataset_profile!r}")
    dataset_root = episode_root / "test"

    from .environment import SlotDragGameEnv

    env = SlotDragGameEnv(dataset_root=dataset_root, artifact_dir=output_dir / "screenshots")
    records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    sensitivity_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    step_counts: Counter[str] = Counter()
    aim_step_counts: Counter[str] = Counter()
    mouse_down_position_counts: Counter[str] = Counter()
    action_step_counts: Counter[str] = Counter()
    cross_target_retry_counts: Counter[str] = Counter()
    sensitivity_band_counts: Counter[str] = Counter()
    piece_shape_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    candidate_count_counts: Counter[str] = Counter()
    mouse_down_positions = _weighted_mouse_down_positions(count=count, seed=seed)

    try:
        for index, episode_id in enumerate(episode_manifest["splits"]["test"]):
            pattern = TRAJECTORY_PATTERNS[index % len(TRAJECTORY_PATTERNS)]
            mouse_down_position = mouse_down_positions[index]
            aim_step_count, placement_step_count = _step_pair_for_mouse_down_position(mouse_down_position)
            meta_path = dataset_root / episode_id / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            try:
                action_plan, trace_metadata = _build_closed_loop_actions(
                    meta,
                    pattern=pattern,
                    aim_step_count=aim_step_count,
                    placement_step_count=placement_step_count,
                )
                observation = env.reset(task_type=episode_id, task_id=episode_id)
                steps: list[dict[str, Any]] = [_observation_step(observation)]
                trace_rows: list[dict[str, Any]] = []
                for action_index, action in enumerate(action_plan):
                    phase = (
                        "aim"
                        if action_index < trace_metadata["aim_step_count"]
                        else "place"
                        if action["kind"] == "move_to"
                        else "button"
                    )
                    is_final_move = (
                        action["kind"] == "move_to"
                        and (
                            action_index == trace_metadata["aim_step_count"] - 1
                            or action_index == len(action_plan) - 2
                        )
                    )
                    thought = _thought_for_action(
                        phase=phase,
                        action=action,
                        action_index=action_index,
                        trajectory_pattern=pattern,
                        is_final_move=is_final_move,
                    )
                    steps.append(_action_step(action, thought=thought))
                    before_observation = observation
                    result = env.step(_primitive_from_action(action))
                    action_counts[action["kind"]] += 1
                    trace_rows.append(
                        _trace_record(
                            index=action_index,
                            record_id=episode_id,
                            before_observation=before_observation,
                            action=action,
                            result=result,
                            thought=thought,
                            trajectory_pattern=pattern,
                        )
                    )
                    observation = result.observation
                    steps.append(_observation_step(observation))
                    if result.done and action_index != len(action_plan) - 1:
                        raise RuntimeError(f"{episode_id}: episode ended before the planned final release")

                final_info = trace_rows[-1]["env_info"] if trace_rows else {}
                if not trace_rows or trace_rows[-1].get("success") is not True:
                    raise RuntimeError(f"{episode_id}: generated closed-loop trace did not succeed")

                trainable_metadata = {
                    "task_type": "slot_drag_game",
                    "benchmark": "SlotDragGame",
                    "dataset_profile": dataset_profile,
                    "perspective": "first_person",
                    "source_mode": "synthetic_oracle",
                    "success": True,
                    "aim_step_count": trace_metadata["aim_step_count"],
                    "success_step_count": trace_metadata["success_step_count"],
                    "mouse_down_action_position": trace_metadata["mouse_down_action_position"],
                    "action_step_count": trace_metadata["action_step_count"],
                    "max_steps": DEFAULT_SFT_MAX_STEPS,
                    "trajectory_pattern": pattern,
                    "viewport": meta["viewport"],
                    "coordinate_format": "qwen3_relative_0_1000",
                    "cursor_lock": "center",
                    "action_paradigm": "primitive_closed_loop",
                    "protocol_version": PROTOCOL_VERSION,
                }
                record = {
                    "id": episode_id,
                    "source": source_name,
                    "instruction": TASK_DESCRIPTION,
                    "metadata": trainable_metadata,
                    "steps": steps,
                }
                records.append(record)

                audit_record = {
                    "id": episode_id,
                    "source": source_name,
                    "metadata": {
                        **trainable_metadata,
                        "sensitivity": meta["sensitivity"],
                        "view_zoom": meta["view_zoom"],
                        "piece_start_world_xy": meta["piece_start_world_xy"],
                        "slot_center_world_xy": meta["slot_center_world_xy"],
                        "initial_view_center_xy": meta["initial_view_center_xy"],
                        "piece_slot_distance_px": final_info.get("piece_slot_distance_px"),
                        "tolerance_px": final_info.get("tolerance_px"),
                        "oracle_action_plan": action_plan,
                        **trace_metadata,
                    },
                    "episode_meta_path": str(meta_path),
                    "trace_path": str(output_dir / "traces" / episode_id / "trace.jsonl"),
                }
                audit_records.append(audit_record)
                sensitivity_counts[str(meta["sensitivity"])] += 1
                sensitivity_band_counts[str(meta.get("sensitivity_band", "legacy"))] += 1
                piece_shape_counts[str(meta.get("piece", {}).get("shape", "unknown"))] += 1
                layout_counts[str(meta.get("layout_family", "legacy_target_centered"))] += 1
                candidate_count_counts[str(len(meta.get("slot_options", [])))] += 1
                pattern_counts[pattern] += 1
                step_counts[str(trace_metadata["success_step_count"])] += 1
                aim_step_counts[str(trace_metadata["aim_step_count"])] += 1
                mouse_down_position_counts[str(trace_metadata["mouse_down_action_position"])] += 1
                action_step_counts[str(trace_metadata["action_step_count"])] += 1
                if trace_metadata["aim_has_cross_target_retry"]:
                    cross_target_retry_counts["aim"] += 1
                if trace_metadata["placement_has_cross_target_retry"]:
                    cross_target_retry_counts["placement"] += 1
                if (
                    trace_metadata["aim_has_cross_target_retry"]
                    or trace_metadata["placement_has_cross_target_retry"]
                ):
                    cross_target_retry_counts["any"] += 1

                trace_path = output_dir / "traces" / episode_id / "trace.jsonl"
                _write_jsonl(trace_path, trace_rows)
            except Exception as exc:
                excluded.append(
                    {
                        "id": episode_id,
                        "reason": "generation_error",
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
    finally:
        env.close()

    _write_jsonl(output_dir / "records.jsonl", records)
    _write_jsonl(output_dir / "train.jsonl", records)
    _write_jsonl(output_dir / "audit.jsonl", audit_records)
    _write_jsonl(output_dir / "excluded.jsonl", excluded)

    action_step_count = sum(action_counts.values())
    observation_step_count = sum(
        1
        for record in records
        for step in record["steps"]
        if step.get("type") == "observation"
    )
    validation = {
        "status": "passed" if len(records) == count and not excluded else "failed",
        "requested_records": count,
        "records": len(records),
        "excluded": len(excluded),
        "success_records": sum(1 for record in records if record["metadata"].get("success") is True),
        "action_step_count": action_step_count,
        "observation_step_count": observation_step_count,
        "action_kind_counts": dict(sorted(action_counts.items())),
        "trainable_oracle_leak_count": sum(
            1
            for record in records
            for key in ("sensitivity", "piece_start_world_xy", "slot_center_world_xy", "oracle_action_plan")
            if key in record.get("metadata", {})
        ),
        "all_records_have_final_success": all(record["metadata"].get("success") is True for record in records),
        "all_records_have_intermediate_corrections": all(
            int(record["metadata"].get("success_step_count", 0)) >= 4
            and int(record["metadata"].get("aim_step_count", 0)) >= 2
            for record in records
        ),
        "all_records_within_max_steps": all(
            int(record["metadata"].get("action_step_count", 0)) <= DEFAULT_SFT_MAX_STEPS
            for record in records
        ),
        "mouse_down_position_distribution": dict(sorted(mouse_down_position_counts.items())),
        "max_observed_action_step_count": max((int(key) for key in action_step_counts), default=0),
        "cross_target_retry_counts": dict(sorted(cross_target_retry_counts.items())),
    }
    if (
        validation["trainable_oracle_leak_count"]
        or not validation["all_records_have_final_success"]
        or not validation["all_records_within_max_steps"]
    ):
        validation["status"] = "failed"

    summary = {
        "benchmark": "SlotDragGame",
        "source": source_name,
        "dataset_profile": dataset_profile,
        "output_dir": str(output_dir),
        "episode_root": str(episode_root),
        "records": len(records),
        "train_records": len(records),
        "excluded": len(excluded),
        "sft_examples": action_step_count,
        "observation_steps": observation_step_count,
        "task_description": TASK_DESCRIPTION,
        "split_policy": "train_only",
        "coordinate_format": "qwen3_relative_0_1000",
        "cursor_lock": "center",
        "perspective": "first_person",
        "max_steps": DEFAULT_SFT_MAX_STEPS,
        "max_observed_action_step_count": validation["max_observed_action_step_count"],
        "mouse_down_position_weight_policy": {
            str(position): weight
            for position, weight in MOUSE_DOWN_POSITION_WEIGHTS
        },
        "action_kind_counts": dict(sorted(action_counts.items())),
        "sensitivity_distribution": dict(sorted(sensitivity_counts.items())),
        "sensitivity_band_distribution": dict(sorted(sensitivity_band_counts.items())),
        "piece_shape_distribution": dict(sorted(piece_shape_counts.items())),
        "layout_distribution": dict(sorted(layout_counts.items())),
        "candidate_count_distribution": dict(sorted(candidate_count_counts.items())),
        "trajectory_pattern_distribution": dict(sorted(pattern_counts.items())),
        "aim_step_count_distribution": dict(sorted(aim_step_counts.items())),
        "success_step_count_distribution": dict(sorted(step_counts.items())),
        "mouse_down_action_position_distribution": dict(sorted(mouse_down_position_counts.items())),
        "action_step_count_distribution": dict(sorted(action_step_counts.items())),
        "cross_target_retry_counts": dict(sorted(cross_target_retry_counts.items())),
        "validation": validation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# SlotDragGame SFT Closed-Loop Dataset",
                "",
                f"- Records: {len(records)}",
                f"- SFT action-step examples: {action_step_count}",
                f"- Dataset profile: {dataset_profile}",
                "- Split policy: train-only",
                "- Trace semantics: success-only primitive closed-loop traces with intermediate correction states.",
                "- Trainable metadata excludes sensitivity and world-coordinate oracle fields; audit.jsonl keeps them for debugging.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate center-locked SlotDragGame CAPTCHA-style episodes.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--viewport-width", type=int, default=DEFAULT_VIEWPORT[0])
    parser.add_argument("--viewport-height", type=int, default=DEFAULT_VIEWPORT[1])
    parser.add_argument("--sft-output-dir", type=Path, default=None)
    parser.add_argument("--sft-count", type=int, default=DEFAULT_SFT_COUNT)
    parser.add_argument("--sft-seed", type=int, default=20260706)
    parser.add_argument("--episode-namespace", default="train")
    parser.add_argument(
        "--dataset-profile",
        choices=("legacy", "paper_v2_shape_ood"),
        default="legacy",
    )
    args = parser.parse_args(argv)
    if args.sft_output_dir is not None:
        manifest = generate_sft_dataset(
            output_dir=args.sft_output_dir,
            count=args.sft_count,
            seed=args.sft_seed,
            dataset_profile=args.dataset_profile,
            episode_namespace=args.episode_namespace,
        )
    else:
        manifest = generate_dataset(
            output_dir=args.output_dir,
            test_count=args.test_count,
            seed=args.seed,
            viewport=(args.viewport_width, args.viewport_height),
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
