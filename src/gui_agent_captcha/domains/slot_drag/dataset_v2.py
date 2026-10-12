from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from ...actions import PrimitiveAction
from .dataset import (
    COLORS,
    DEFAULT_SLOT_COLOR,
    DEFAULT_SLOT_COLOR_NAME,
    DEFAULT_SLOT_RADIUS_SCALE,
    DEFAULT_TOLERANCE_RADIUS_SCALE,
    DEFAULT_WORLD_SIZE,
    TASK_DESCRIPTION,
    _coerce_xy,
    _move_to_for_view_delta,
    _object_description,
    _sample_offset,
    _slot_description,
)
from .paths import paper_v2_episodes_root, paper_v2_sft_root

PAPER_V2_VERSION = 4
PAPER_V2_PROFILE = "paper_v2_shape_ood"
PAPER_V2_VIEWPORT = (1280, 720)
PAPER_V2_TRAIN_SHAPES = (
    "circle",
    "square",
    "triangle",
    "hexagon",
    "wide_rectangle",
    "cross",
)
PAPER_V2_OOD_SHAPES = (
    "pentagon",
    "trapezoid",
    "star",
    "l_shape",
)
PAPER_V2_LAYOUTS = (
    "scattered",
    "jittered_grid",
    "ring_arc",
    "two_cluster",
)
PAPER_V2_CANDIDATE_COUNTS = (5, 6, 7, 8)
PAPER_V2_SENSITIVITY_BANDS = {
    "low": (500, 800),
    "high": (1200, 1500),
}
PAPER_V2_GEOMETRY_MARGIN_PX = 12.0
PAPER_V2_TEST_SPLITS = (
    ("iid_a", "iid", 100, 2026071611),
    ("iid_b", "iid", 100, 2026071612),
    ("ood_a", "ood", 50, 2026071621),
    ("ood_b", "ood", 50, 2026071622),
)
PAPER_V2_TRAIN_SEED = 2026071601
def _factor_seed(seed: int, factor: str) -> int:
    digest = hashlib.sha256(f"{seed}:{factor}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _factor_rng(seed: int, factor: str) -> random.Random:
    return random.Random(_factor_seed(seed, factor))


def _balanced_schedule(
    values: Sequence[Any],
    *,
    count: int,
    rng: random.Random,
) -> list[Any]:
    if not values:
        raise ValueError("balanced schedule requires at least one value")
    if count < 0:
        raise ValueError("balanced schedule count must be non-negative")
    quotient, remainder = divmod(count, len(values))
    schedule = [value for value in values for _ in range(quotient)]
    schedule.extend(values[:remainder])
    rng.shuffle(schedule)
    return schedule


def _balanced_target_position_schedule(
    candidate_counts: Sequence[int],
    *,
    seed: int,
) -> list[int]:
    positions = [0] * len(candidate_counts)
    for candidate_count in sorted(set(candidate_counts)):
        episode_indices = [
            index
            for index, value in enumerate(candidate_counts)
            if value == candidate_count
        ]
        schedule = _balanced_schedule(
            tuple(range(candidate_count)),
            count=len(episode_indices),
            rng=_factor_rng(seed, f"target_position:{candidate_count}"),
        )
        for episode_index, target_position in zip(episode_indices, schedule, strict=True):
            positions[episode_index] = int(target_position)
    return positions


def _sensitivity_schedule(count: int, *, seed: int) -> list[tuple[str, int]]:
    band_schedule = _balanced_schedule(
        ("low", "high"),
        count=count,
        rng=_factor_rng(seed, "sensitivity_band"),
    )
    value_rng = _factor_rng(seed, "sensitivity_value")
    return [
        (
            band,
            value_rng.randint(
                PAPER_V2_SENSITIVITY_BANDS[band][0],
                PAPER_V2_SENSITIVITY_BANDS[band][1],
            ),
        )
        for band in band_schedule
    ]


def _minimum_pair_distance(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return math.inf
    return min(
        math.dist(points[left], points[right])
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _center_offsets(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    values = list(points)
    mean_x = sum(point[0] for point in values) / len(values)
    mean_y = sum(point[1] for point in values) / len(values)
    return [(point[0] - mean_x, point[1] - mean_y) for point in values]


def _slot_collision_half_extents(radius: float) -> tuple[float, float]:
    """Conservative AABB for any rendered v2 slot, including its outer frame."""

    outer_radius = radius * DEFAULT_SLOT_RADIUS_SCALE + 10.0
    return (outer_radius * 1.38, outer_radius)


def _piece_collision_half_extents(radius: float) -> tuple[float, float]:
    """Conservative AABB for any rendered v2 movable piece."""

    return (radius * 1.38, radius)


def _aabbs_overlap(
    left: tuple[float, float],
    right: tuple[float, float],
    *,
    left_half_extents: tuple[float, float],
    right_half_extents: tuple[float, float],
    margin: float = PAPER_V2_GEOMETRY_MARGIN_PX,
) -> bool:
    return (
        abs(left[0] - right[0])
        < left_half_extents[0] + right_half_extents[0] + margin
        and abs(left[1] - right[1])
        < left_half_extents[1] + right_half_extents[1] + margin
    )


def _slot_positions_are_separated(
    points: Sequence[tuple[float, float]],
    *,
    radius: float,
) -> bool:
    half_extents = _slot_collision_half_extents(radius)
    return all(
        not _aabbs_overlap(
            points[left],
            points[right],
            left_half_extents=half_extents,
            right_half_extents=half_extents,
        )
        for left in range(len(points))
        for right in range(left + 1, len(points))
    )


def _scattered_offsets(
    rng: random.Random,
    *,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    min_distance = radius * 3.15
    half_width = max(330.0, radius * 8.0)
    half_height = max(250.0, radius * 6.0)
    half_extents = _slot_collision_half_extents(radius)
    for _ in range(200):
        points: list[tuple[float, float]] = []
        for _point_index in range(count):
            for _ in range(500):
                candidate = (
                    rng.uniform(-half_width, half_width),
                    rng.uniform(-half_height, half_height),
                )
                if all(
                    math.dist(candidate, point) >= min_distance
                    and not _aabbs_overlap(
                        candidate,
                        point,
                        left_half_extents=half_extents,
                        right_half_extents=half_extents,
                    )
                    for point in points
                ):
                    points.append(candidate)
                    break
            else:
                break
        if len(points) == count:
            return _center_offsets(points)
    raise RuntimeError("failed to sample scattered slot layout")


def _jittered_grid_offsets(
    rng: random.Random,
    *,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    spacing_x = max(170.0, radius * 5.1)
    spacing_y = max(128.0, radius * 4.1)
    cells = [
        (
            (column - 1.5) * spacing_x,
            (row - 0.5) * spacing_y,
        )
        for row in range(2)
        for column in range(4)
    ]
    rng.shuffle(cells)
    jitter = radius * 0.10
    points = [
        (
            x + rng.uniform(-jitter, jitter),
            y + rng.uniform(-jitter, jitter),
        )
        for x, y in cells[:count]
    ]
    return _center_offsets(points)


def _ring_arc_offsets(
    rng: random.Random,
    *,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    ring_x = max(270.0, radius * 7.0)
    ring_y = max(210.0, radius * 5.5)
    phase = rng.uniform(0.0, math.tau)
    points = []
    for index in range(count):
        angle = phase + math.tau * index / count + rng.uniform(-0.025, 0.025)
        points.append(
            (
                math.cos(angle) * ring_x,
                math.sin(angle) * ring_y,
            )
        )
    return _center_offsets(points)


def _two_cluster_offsets(
    rng: random.Random,
    *,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    left_count = count // 2
    right_count = count - left_count
    cluster_x = max(235.0, radius * 6.0)
    spacing_y = max(138.0, radius * 3.8)
    jitter = radius * 0.06
    points: list[tuple[float, float]] = []
    for center_x, cluster_count in ((-cluster_x, left_count), (cluster_x, right_count)):
        for index in range(cluster_count):
            points.append(
                (
                    center_x + rng.uniform(-jitter, jitter),
                    (index - (cluster_count - 1) / 2.0) * spacing_y
                    + rng.uniform(-jitter, jitter),
                )
            )
    return _center_offsets(points)


def _layout_offsets(
    rng: random.Random,
    *,
    layout_family: str,
    count: int,
    radius: float,
) -> list[tuple[float, float]]:
    if layout_family not in PAPER_V2_LAYOUTS:
        raise ValueError(f"unsupported SlotDragGame v2 layout: {layout_family!r}")

    # Random jitter is allowed to propose an invalid layout, especially for the
    # eight-candidate ring at the largest radii.  Reject that proposal locally
    # instead of aborting an otherwise valid deterministic dataset build.
    for _ in range(200):
        if layout_family == "scattered":
            points = _scattered_offsets(rng, count=count, radius=radius)
        elif layout_family == "jittered_grid":
            points = _jittered_grid_offsets(rng, count=count, radius=radius)
        elif layout_family == "ring_arc":
            points = _ring_arc_offsets(rng, count=count, radius=radius)
        else:
            points = _two_cluster_offsets(rng, count=count, radius=radius)
        if (
            len(points) == count
            and _minimum_pair_distance(points) >= radius * 2.65
            and _slot_positions_are_separated(points, radius=radius)
        ):
            return points
    raise RuntimeError(
        f"failed to sample non-overlapping {layout_family!r} slot positions"
    )


def _distractor_shapes(
    rng: random.Random,
    *,
    distribution: str,
    target_shape: str,
    count: int,
) -> list[str]:
    if distribution == "iid":
        pool = [shape for shape in PAPER_V2_TRAIN_SHAPES if shape != target_shape]
        values: list[str] = []
        while len(values) < count:
            cycle = list(pool)
            rng.shuffle(cycle)
            values.extend(cycle)
        return values[:count]
    if distribution != "ood":
        raise ValueError(f"unsupported SlotDragGame v2 distribution: {distribution!r}")

    held_out_pool = [shape for shape in PAPER_V2_OOD_SHAPES if shape != target_shape]
    seen_pool = list(PAPER_V2_TRAIN_SHAPES)
    held_out_count = min(len(held_out_pool), max(1, math.ceil(count / 2)))
    values = rng.sample(held_out_pool, k=held_out_count)
    remaining_pool = [shape for shape in seen_pool if shape not in values]
    values.extend(rng.sample(remaining_pool, k=count - len(values)))
    rng.shuffle(values)
    return values


def _safe_move_sequence(
    view_delta: tuple[float, float],
    *,
    viewport: tuple[int, int],
    sensitivity: float,
    view_zoom: float,
) -> list[dict[str, float | str]]:
    max_delta_x = viewport[0] * 0.45 * sensitivity / view_zoom
    max_delta_y = viewport[1] * 0.45 * sensitivity / view_zoom
    step_count = max(
        1,
        math.ceil(abs(view_delta[0]) / max(max_delta_x, 1e-6)),
        math.ceil(abs(view_delta[1]) / max(max_delta_y, 1e-6)),
    )
    step_delta = (view_delta[0] / step_count, view_delta[1] / step_count)
    return [
        _move_to_for_view_delta(
            step_delta,
            viewport=viewport,
            sensitivity=sensitivity,
            view_zoom=view_zoom,
        )
        for _ in range(step_count)
    ]


def _episode_geometry(
    rng: random.Random,
    *,
    viewport: tuple[int, int],
    world_size: tuple[int, int],
    radius: float,
    layout_family: str,
    candidate_count: int,
    target_position: int,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    list[tuple[float, float]],
]:
    min_center_x = viewport[0] / 2.0
    max_center_x = world_size[0] - min_center_x
    min_center_y = viewport[1] / 2.0
    max_center_y = world_size[1] - min_center_y
    for _ in range(2000):
        initial_center = (
            rng.uniform(min_center_x + 190.0, max_center_x - 190.0),
            rng.uniform(min_center_y + 150.0, max_center_y - 150.0),
        )
        piece_delta = _sample_offset(
            rng,
            x_range=(150.0, 300.0),
            y_range=(70.0, 160.0),
            min_norm=180.0,
        )
        slot_delta = _sample_offset(
            rng,
            x_range=(220.0, 420.0),
            y_range=(80.0, 230.0),
            min_norm=255.0,
        )
        piece_xy = (
            initial_center[0] + piece_delta[0],
            initial_center[1] + piece_delta[1],
        )
        target_xy = (
            piece_xy[0] + slot_delta[0],
            piece_xy[1] + slot_delta[1],
        )
        offsets = _layout_offsets(
            rng,
            layout_family=layout_family,
            count=candidate_count,
            radius=radius,
        )
        target_offset = offsets[target_position]
        slot_positions = [
            (
                target_xy[0] + offset[0] - target_offset[0],
                target_xy[1] + offset[1] - target_offset[1],
            )
            for offset in offsets
        ]
        piece_half_extents = _piece_collision_half_extents(radius)
        slot_half_extents = _slot_collision_half_extents(radius)
        if any(
            _aabbs_overlap(
                piece_xy,
                slot_xy,
                left_half_extents=piece_half_extents,
                right_half_extents=slot_half_extents,
            )
            for slot_xy in slot_positions
        ):
            continue
        all_centers = [initial_center, piece_xy, target_xy, *slot_positions]
        if all(
            min_center_x <= x <= max_center_x and min_center_y <= y <= max_center_y
            for x, y in all_centers
        ):
            return initial_center, piece_xy, target_xy, slot_positions
    raise RuntimeError("failed to sample reachable SlotDragGame v2 geometry")


def _build_episode_meta(
    *,
    episode_id: str,
    split_name: str,
    distribution: str,
    shape: str,
    color_name: str,
    color: str,
    radius: int,
    sensitivity_band: str,
    sensitivity_milli: int,
    layout_family: str,
    candidate_count: int,
    target_position: int,
    seed: int,
) -> dict[str, Any]:
    rng = _factor_rng(seed, f"episode:{episode_id}")
    viewport = PAPER_V2_VIEWPORT
    world_size = DEFAULT_WORLD_SIZE
    sensitivity = round(sensitivity_milli / 1000.0, 3)
    view_zoom = 1.0
    initial_center, piece_xy, target_xy, slot_positions = _episode_geometry(
        rng,
        viewport=viewport,
        world_size=world_size,
        radius=radius,
        layout_family=layout_family,
        candidate_count=candidate_count,
        target_position=target_position,
    )
    distractor_shapes = _distractor_shapes(
        rng,
        distribution=distribution,
        target_shape=shape,
        count=candidate_count - 1,
    )
    distractor_index = 0
    slot_options = []
    for position_index, center_xy in enumerate(slot_positions):
        role = "target" if position_index == target_position else "distractor"
        option_shape = shape if role == "target" else distractor_shapes[distractor_index]
        if role == "distractor":
            distractor_index += 1
        slot_options.append(
            {
                "id": f"slot_{position_index:02d}",
                "role": role,
                "layout_position_index": position_index,
                "center_world_xy": [round(center_xy[0], 3), round(center_xy[1], 3)],
                "shape": option_shape,
                "color_name": DEFAULT_SLOT_COLOR_NAME,
                "color": DEFAULT_SLOT_COLOR,
            }
        )
    option_order_rng = _factor_rng(seed, f"option_order:{episode_id}")
    option_order_rng.shuffle(slot_options)

    aim_moves = _safe_move_sequence(
        (piece_xy[0] - initial_center[0], piece_xy[1] - initial_center[1]),
        viewport=viewport,
        sensitivity=sensitivity,
        view_zoom=view_zoom,
    )
    placement_moves = _safe_move_sequence(
        (target_xy[0] - piece_xy[0], target_xy[1] - piece_xy[1]),
        viewport=viewport,
        sensitivity=sensitivity,
        view_zoom=view_zoom,
    )
    oracle = [*aim_moves, {"kind": "mouse_down"}, *placement_moves, {"kind": "mouse_up"}]
    tolerance = radius * DEFAULT_TOLERANCE_RADIUS_SCALE
    return {
        "episode_id": episode_id,
        "version": PAPER_V2_VERSION,
        "dataset_profile": PAPER_V2_PROFILE,
        "split": split_name,
        "distribution": distribution,
        "benchmark": "SlotDragGame",
        "task_type": "slot_drag_game",
        "perspective": "first_person",
        "viewport": list(viewport),
        "world_size": list(world_size),
        "initial_view_center_xy": [round(initial_center[0], 3), round(initial_center[1], 3)],
        "sensitivity": sensitivity,
        "sensitivity_milli": sensitivity_milli,
        "sensitivity_band": sensitivity_band,
        "view_zoom": view_zoom,
        "piece_start_world_xy": [round(piece_xy[0], 3), round(piece_xy[1], 3)],
        "slot_center_world_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
        "slot_radius_scale": DEFAULT_SLOT_RADIUS_SCALE,
        "slot_frame_radius_px": round(radius * DEFAULT_SLOT_RADIUS_SCALE, 3),
        "tolerance_radius_scale": DEFAULT_TOLERANCE_RADIUS_SCALE,
        "tolerance_px": round(tolerance, 3),
        "layout_family": layout_family,
        "slot_candidate_count": candidate_count,
        "target_layout_position_index": target_position,
        "movable_object_description": _object_description(
            color_name=color_name,
            shape=shape,
        ),
        "target_slot_description": _slot_description(shape=shape),
        "piece": {
            "shape": shape,
            "color_name": color_name,
            "color": color,
            "outline": "#111827",
            "radius_px": radius,
        },
        "slot_options": slot_options,
        "instruction": TASK_DESCRIPTION,
        "oracle_primitive_sequence": oracle,
        "oracle_aim_move_count": len(aim_moves),
        "oracle_placement_move_count": len(placement_moves),
    }


def _generate_split(
    *,
    test_root: Path,
    split_name: str,
    distribution: str,
    count: int,
    seed: int,
) -> list[str]:
    shape_pool = PAPER_V2_TRAIN_SHAPES if distribution == "iid" else PAPER_V2_OOD_SHAPES
    def split_balanced_values(values: Sequence[Any]) -> tuple[Any, ...]:
        ordered = tuple(values)
        if split_name.endswith("_b") and len(ordered) > 1:
            offset = len(ordered) // 2
            return (*ordered[offset:], *ordered[:offset])
        return ordered

    shapes = _balanced_schedule(
        split_balanced_values(shape_pool),
        count=count,
        rng=_factor_rng(seed, "piece_shape"),
    )
    colors = _balanced_schedule(
        split_balanced_values(COLORS),
        count=count,
        rng=_factor_rng(seed, "piece_color"),
    )
    layouts = _balanced_schedule(
        split_balanced_values(PAPER_V2_LAYOUTS),
        count=count,
        rng=_factor_rng(seed, "layout_family"),
    )
    candidate_counts = [
        int(value)
        for value in _balanced_schedule(
            split_balanced_values(PAPER_V2_CANDIDATE_COUNTS),
            count=count,
            rng=_factor_rng(seed, "candidate_count"),
        )
    ]
    target_positions = _balanced_target_position_schedule(candidate_counts, seed=seed)
    sensitivities = _sensitivity_schedule(count, seed=seed)
    radius_rng = _factor_rng(seed, "piece_radius")

    episode_ids = []
    for index in range(count):
        episode_id = f"sdg_{split_name}_{index + 1:04d}"
        sensitivity_band, sensitivity_milli = sensitivities[index]
        color_name, color = colors[index]
        meta = _build_episode_meta(
            episode_id=episode_id,
            split_name=split_name,
            distribution=distribution,
            shape=str(shapes[index]),
            color_name=str(color_name),
            color=str(color),
            radius=radius_rng.randint(26, 44),
            sensitivity_band=sensitivity_band,
            sensitivity_milli=sensitivity_milli,
            layout_family=str(layouts[index]),
            candidate_count=candidate_counts[index],
            target_position=target_positions[index],
            seed=seed,
        )
        episode_dir = test_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        episode_ids.append(episode_id)
    return episode_ids


def _load_meta_records(test_root: Path, episode_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [
        json.loads((test_root / episode_id / "meta.json").read_text(encoding="utf-8"))
        for episode_id in episode_ids
    ]


def _static_validation(
    *,
    test_root: Path,
    split_ids: dict[str, list[str]],
) -> dict[str, Any]:
    issues: list[str] = []
    all_ids = [episode_id for ids in split_ids.values() for episode_id in ids]
    if len(all_ids) != len(set(all_ids)):
        issues.append("episode ids are not unique across splits")

    split_summaries: dict[str, Any] = {}
    total_slot_overlap_episodes = 0
    total_piece_slot_overlap_episodes = 0
    for split_name, episode_ids in split_ids.items():
        records = _load_meta_records(test_root, episode_ids)
        distributions = {str(record.get("distribution")) for record in records}
        expected_distribution = "ood" if split_name.startswith("ood") else "iid"
        shapes = Counter(str(record["piece"]["shape"]) for record in records)
        sensitivity_bands = Counter(str(record["sensitivity_band"]) for record in records)
        layouts = Counter(str(record["layout_family"]) for record in records)
        candidate_counts = Counter(str(record["slot_candidate_count"]) for record in records)
        target_positions = Counter(
            f"{record['slot_candidate_count']}:{record['target_layout_position_index']}"
            for record in records
        )
        slot_overlap_episodes = 0
        piece_slot_overlap_episodes = 0
        for record in records:
            target_slots = [
                option
                for option in record["slot_options"]
                if option.get("role") == "target"
            ]
            if len(target_slots) != 1:
                issues.append(f"{record['episode_id']}: expected exactly one target slot")
                continue
            if target_slots[0].get("shape") != record["piece"].get("shape"):
                issues.append(f"{record['episode_id']}: target shape does not match piece")
            if any(
                option.get("role") == "distractor"
                and option.get("shape") == record["piece"].get("shape")
                for option in record["slot_options"]
            ):
                issues.append(f"{record['episode_id']}: duplicate matching distractor slot")
            milli = int(record["sensitivity_milli"])
            band = str(record["sensitivity_band"])
            lower, upper = PAPER_V2_SENSITIVITY_BANDS[band]
            if not lower <= milli <= upper or record["sensitivity"] != round(milli / 1000.0, 3):
                issues.append(f"{record['episode_id']}: invalid sensitivity encoding")
            if tuple(record["viewport"]) != PAPER_V2_VIEWPORT:
                issues.append(f"{record['episode_id']}: viewport is not 1280x720")
            if len(record["oracle_primitive_sequence"]) > 12:
                issues.append(f"{record['episode_id']}: oracle exceeds max_steps=12")
            radius = float(record["piece"]["radius_px"])
            slot_positions = [
                _coerce_xy(option["center_world_xy"])
                for option in record["slot_options"]
            ]
            if not _slot_positions_are_separated(slot_positions, radius=radius):
                slot_overlap_episodes += 1
            piece_xy = _coerce_xy(record["piece_start_world_xy"])
            piece_half_extents = _piece_collision_half_extents(radius)
            slot_half_extents = _slot_collision_half_extents(radius)
            if any(
                _aabbs_overlap(
                    piece_xy,
                    slot_xy,
                    left_half_extents=piece_half_extents,
                    right_half_extents=slot_half_extents,
                )
                for slot_xy in slot_positions
            ):
                piece_slot_overlap_episodes += 1
        if distributions != {expected_distribution}:
            issues.append(
                f"{split_name}: distributions={sorted(distributions)!r}, "
                f"expected={[expected_distribution]!r}"
            )
        expected_shapes = (
            set(PAPER_V2_OOD_SHAPES)
            if expected_distribution == "ood"
            else set(PAPER_V2_TRAIN_SHAPES)
        )
        if set(shapes) != expected_shapes:
            issues.append(
                f"{split_name}: shapes={sorted(shapes)!r}, expected={sorted(expected_shapes)!r}"
            )
        if abs(sensitivity_bands["low"] - sensitivity_bands["high"]) > 1:
            issues.append(f"{split_name}: sensitivity bands are not balanced")
        if slot_overlap_episodes:
            issues.append(
                f"{split_name}: {slot_overlap_episodes} episodes have overlapping slot frames"
            )
        if piece_slot_overlap_episodes:
            issues.append(
                f"{split_name}: {piece_slot_overlap_episodes} episodes have initial piece-slot overlap"
            )
        total_slot_overlap_episodes += slot_overlap_episodes
        total_piece_slot_overlap_episodes += piece_slot_overlap_episodes
        split_summaries[split_name] = {
            "records": len(records),
            "distribution": expected_distribution,
            "shape_distribution": dict(sorted(shapes.items())),
            "sensitivity_band_distribution": dict(sorted(sensitivity_bands.items())),
            "layout_distribution": dict(sorted(layouts.items())),
            "candidate_count_distribution": dict(sorted(candidate_counts.items())),
            "target_position_distribution": dict(sorted(target_positions.items())),
            "geometry_overlap_episode_counts": {
                "slot_slot": slot_overlap_episodes,
                "piece_slot": piece_slot_overlap_episodes,
            },
        }
    return {
        "status": "passed" if not issues else "failed",
        "issues": issues,
        "geometry_overlap_episode_counts": {
            "slot_slot": total_slot_overlap_episodes,
            "piece_slot": total_piece_slot_overlap_episodes,
        },
        "splits": split_summaries,
    }


def generate_test_dataset(
    *,
    output_dir: Path | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    output_dir = (
        paper_v2_episodes_root()
        if output_dir is None
        else Path(output_dir)
    )
    test_root = output_dir / "test"
    if clean and test_root.exists():
        for child in test_root.glob("sdg_*"):
            if child.is_dir():
                shutil.rmtree(child)
    test_root.mkdir(parents=True, exist_ok=True)

    split_ids: dict[str, list[str]] = {}
    split_seeds: dict[str, int] = {}
    for split_name, distribution, count, seed in PAPER_V2_TEST_SPLITS:
        split_ids[split_name] = _generate_split(
            test_root=test_root,
            split_name=split_name,
            distribution=distribution,
            count=count,
            seed=seed,
        )
        split_seeds[split_name] = seed
    iid_ids = [*split_ids["iid_a"], *split_ids["iid_b"]]
    ood_ids = [*split_ids["ood_a"], *split_ids["ood_b"]]
    all_ids = [*iid_ids, *ood_ids]
    validation = _static_validation(test_root=test_root, split_ids=split_ids)
    manifest = {
        "benchmark": "SlotDragGame",
        "version": PAPER_V2_VERSION,
        "dataset_profile": PAPER_V2_PROFILE,
        "output_dir": str(output_dir),
        "viewport": list(PAPER_V2_VIEWPORT),
        "test_count": len(all_ids),
        "iid_count": len(iid_ids),
        "ood_count": len(ood_ids),
        "split_seeds": split_seeds,
        "splits": {
            **split_ids,
            "iid": iid_ids,
            "ood": ood_ids,
            "test": all_ids,
        },
        "shape_policy": {
            "train_iid_shapes": list(PAPER_V2_TRAIN_SHAPES),
            "ood_shapes": list(PAPER_V2_OOD_SHAPES),
            "sets_are_disjoint": set(PAPER_V2_TRAIN_SHAPES).isdisjoint(PAPER_V2_OOD_SHAPES),
        },
        "slot_policy": {
            "candidate_counts": list(PAPER_V2_CANDIDATE_COUNTS),
            "layout_families": list(PAPER_V2_LAYOUTS),
            "target_position_assignment": "balanced_after_unlabeled_layout_generation",
            "duplicate_matching_distractor_allowed": False,
        },
        "sensitivity_policy": {
            "sampling": "balanced_band_then_uniform_integer_milli",
            "bands_milli": {
                key: list(value)
                for key, value in PAPER_V2_SENSITIVITY_BANDS.items()
            },
            "precision_decimals": 3,
        },
        "validation": validation,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def generate_training_episode_dataset(
    *,
    output_dir: Path,
    count: int,
    seed: int = PAPER_V2_TRAIN_SEED,
    episode_namespace: str = "train",
    clean: bool = True,
) -> dict[str, Any]:
    if not episode_namespace or any(
        not (character.islower() or character.isdigit() or character == "_")
        for character in episode_namespace
    ):
        raise ValueError(
            "episode_namespace must contain only lowercase letters, digits, and underscores"
        )
    test_root = output_dir / "test"
    if clean and test_root.exists():
        for child in test_root.glob("sdg_*"):
            if child.is_dir():
                shutil.rmtree(child)
    test_root.mkdir(parents=True, exist_ok=True)
    episode_ids = _generate_split(
        test_root=test_root,
        split_name=episode_namespace,
        distribution="iid",
        count=count,
        seed=seed,
    )
    validation = _static_validation(
        test_root=test_root,
        split_ids={episode_namespace: episode_ids},
    )
    manifest = {
        "benchmark": "SlotDragGame",
        "version": PAPER_V2_VERSION,
        "dataset_profile": PAPER_V2_PROFILE,
        "output_dir": str(output_dir),
        "viewport": list(PAPER_V2_VIEWPORT),
        "test_count": count,
        "train_count": count,
        "seed": seed,
        "episode_namespace": episode_namespace,
        "splits": {"train": episode_ids, "test": episode_ids},
        "shape_policy": {
            "train_iid_shapes": list(PAPER_V2_TRAIN_SHAPES),
            "ood_shapes": list(PAPER_V2_OOD_SHAPES),
            "ood_shapes_present": False,
        },
        "slot_policy": {
            "candidate_counts": list(PAPER_V2_CANDIDATE_COUNTS),
            "layout_families": list(PAPER_V2_LAYOUTS),
        },
        "sensitivity_policy": {
            "sampling": "balanced_band_then_uniform_integer_milli",
            "bands_milli": {
                key: list(value)
                for key, value in PAPER_V2_SENSITIVITY_BANDS.items()
            },
            "precision_decimals": 3,
        },
        "validation": validation,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def validate_oracle_sequences(
    *,
    dataset_root: Path,
    episode_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    from .environment import SlotDragGameEnv

    if episode_ids is None:
        episode_ids = sorted(
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir() and (path / "meta.json").is_file()
        )
    failures = []
    action_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="slot-drag-v2-oracle-") as temporary_dir:
        env = SlotDragGameEnv(
            dataset_root=dataset_root,
            artifact_dir=Path(temporary_dir),
        )
        try:
            for episode_id in episode_ids:
                meta = json.loads(
                    (dataset_root / episode_id / "meta.json").read_text(encoding="utf-8")
                )
                env.reset(task_type=episode_id, task_id=episode_id)
                result = None
                for action_dict in meta["oracle_primitive_sequence"]:
                    action = PrimitiveAction(
                        kind=str(action_dict["kind"]),
                        x=action_dict.get("x"),
                        y=action_dict.get("y"),
                    )
                    action_counts[action.kind] += 1
                    result = env.step(action)
                    if result.done:
                        break
                if result is None or result.info.get("success") is not True:
                    failures.append(
                        {
                            "episode_id": episode_id,
                            "final_info": None if result is None else result.info,
                        }
                    )
        finally:
            env.close()
    return {
        "status": "passed" if not failures else "failed",
        "episodes": len(episode_ids),
        "successes": len(episode_ids) - len(failures),
        "failures": failures,
        "action_counts": dict(sorted(action_counts.items())),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SlotDragGame paper-v2 train or IID/OOD test data."
    )
    parser.add_argument("--mode", choices=("test", "train"), required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--train-seed", type=int, default=PAPER_V2_TRAIN_SEED)
    parser.add_argument("--episode-namespace", default="train")
    parser.add_argument("--validate-oracles", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "test":
        output_dir = args.output_dir or paper_v2_episodes_root()
        result = generate_test_dataset(output_dir=output_dir)
        if args.validate_oracles:
            oracle_validation = validate_oracle_sequences(dataset_root=output_dir / "test")
            (output_dir / "oracle_validation.json").write_text(
                json.dumps(oracle_validation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            result = {**result, "oracle_validation": oracle_validation}
    else:
        from .dataset import generate_sft_dataset

        output_dir = args.output_dir or paper_v2_sft_root()
        result = generate_sft_dataset(
            output_dir=output_dir,
            count=args.train_count,
            seed=args.train_seed,
            dataset_profile=PAPER_V2_PROFILE,
            episode_namespace=args.episode_namespace,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("validation", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
