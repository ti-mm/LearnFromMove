from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ...actions import PrimitiveAction
from ..slot_drag.dataset_v2 import (
    PAPER_V2_CANDIDATE_COUNTS as SLOT_DRAG_V2_CANDIDATE_COUNTS,
)
from ..slot_drag.dataset_v2 import (
    PAPER_V2_LAYOUTS as SLOT_DRAG_V2_LAYOUTS,
)
from ..slot_drag.dataset_v2 import (
    PAPER_V2_OOD_SHAPES as SLOT_DRAG_V2_OOD_SHAPES,
)
from ..slot_drag.dataset_v2 import (
    PAPER_V2_TRAIN_SHAPES as SLOT_DRAG_V2_TRAIN_SHAPES,
)
from ..slot_drag.dataset_v2 import (
    _aabbs_overlap,
    _balanced_schedule,
    _balanced_target_position_schedule,
    _distractor_shapes,
    _factor_rng,
    _layout_offsets,
    _piece_collision_half_extents,
    _slot_collision_half_extents,
    _slot_positions_are_separated,
)
from .dataset import (
    COLORS,
    DEFAULT_SLOT_RADIUS_SCALE,
    DEFAULT_TOLERANCE_RADIUS_SCALE,
    TASK_DESCRIPTION,
    _object_description,
    _screen_to_model_xy,
    _slot_description,
)
from .paths import paper_v2_episodes_root, paper_v3_sft_root

PAPER_V2_VERSION = 2
PAPER_V2_PROFILE = "third_person_paper_v2_shape_ood"
PAPER_V2_VIEWPORT = (1280, 720)
PAPER_V2_TRAIN_SHAPES = tuple(SLOT_DRAG_V2_TRAIN_SHAPES)
PAPER_V2_OOD_SHAPES = tuple(SLOT_DRAG_V2_OOD_SHAPES)
PAPER_V2_LAYOUTS = tuple(SLOT_DRAG_V2_LAYOUTS)
PAPER_V2_CANDIDATE_COUNTS = tuple(SLOT_DRAG_V2_CANDIDATE_COUNTS)
PAPER_V2_DRAG_DISTANCE_BANDS = {
    "near": (220.0, 380.0),
    "far": (480.0, 700.0),
}
PAPER_V2_GEOMETRY_MARGIN_PX = 12.0
PAPER_V2_TEST_SPLITS = (
    ("iid_a", "iid", 100, 2026071911),
    ("iid_b", "iid", 100, 2026071912),
    ("ood_a", "ood", 50, 2026071921),
    ("ood_b", "ood", 50, 2026071922),
)
PAPER_V2_TRAIN_SEED = 2026073001
PAPER_V3_TRAIN_VERSION = 3
PAPER_V3_TRAIN_PROFILE = "third_person_paper_v3_move_to_2_or_3"
PAPER_V3_TRAIN_TRAJECTORY_PROFILE = "move_to_down_held_move_to_2_or_3_up"
PAPER_V3_TRAIN_HELD_MOVE_COUNTS = (2, 3)
PAPER_V3_TRAIN_HELD_MOVE_ASSIGNMENT = (
    "balanced_within_each_drag_distance_band"
)


def _coerce_xy(value: Any) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"expected an xy sequence, got {value!r}")
    if len(value) < 2:
        raise ValueError(f"expected two xy values, got {value!r}")
    return (float(value[0]), float(value[1]))


def _split_balanced_values(values: Sequence[Any], *, split_name: str) -> tuple[Any, ...]:
    ordered = tuple(values)
    if split_name.endswith("_b") and len(ordered) > 1:
        offset = len(ordered) // 2
        return (*ordered[offset:], *ordered[:offset])
    return ordered


def _drag_distance_schedule(
    count: int,
    *,
    seed: int,
) -> list[tuple[str, float]]:
    bands = _balanced_schedule(
        tuple(PAPER_V2_DRAG_DISTANCE_BANDS),
        count=count,
        rng=_factor_rng(seed, "drag_distance_band"),
    )
    value_rng = _factor_rng(seed, "drag_distance_value")
    return [
        (
            str(band),
            value_rng.uniform(
                PAPER_V2_DRAG_DISTANCE_BANDS[str(band)][0],
                PAPER_V2_DRAG_DISTANCE_BANDS[str(band)][1],
            ),
        )
        for band in bands
    ]


def _held_move_count_schedule(
    drag_distances: Sequence[tuple[str, float]],
    *,
    seed: int,
) -> list[int]:
    held_move_counts = [0] * len(drag_distances)
    for drag_distance_band in PAPER_V2_DRAG_DISTANCE_BANDS:
        episode_indices = [
            index
            for index, (band, _) in enumerate(drag_distances)
            if band == drag_distance_band
        ]
        schedule = _balanced_schedule(
            PAPER_V3_TRAIN_HELD_MOVE_COUNTS,
            count=len(episode_indices),
            rng=_factor_rng(seed, f"held_move_count:{drag_distance_band}"),
        )
        for episode_index, held_move_count in zip(
            episode_indices,
            schedule,
            strict=True,
        ):
            held_move_counts[episode_index] = int(held_move_count)
    return held_move_counts


def _drag_oracle_sequence(
    piece_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    held_move_count: int,
) -> list[dict[str, Any]]:
    if held_move_count < 1:
        raise ValueError("held_move_count must be positive")
    held_moves: list[dict[str, Any]] = []
    for waypoint_index in range(1, held_move_count + 1):
        if waypoint_index == held_move_count:
            waypoint_xy = target_xy
        else:
            fraction = waypoint_index / held_move_count
            waypoint_xy = (
                piece_xy[0] + (target_xy[0] - piece_xy[0]) * fraction,
                piece_xy[1] + (target_xy[1] - piece_xy[1]) * fraction,
            )
        held_moves.append(
            _screen_to_model_xy(waypoint_xy, viewport=PAPER_V2_VIEWPORT)
        )
    return [
        _screen_to_model_xy(piece_xy, viewport=PAPER_V2_VIEWPORT),
        {"kind": "mouse_down"},
        *held_moves,
        {"kind": "mouse_up"},
    ]


def _screen_geometry(
    *,
    seed: int,
    episode_id: str,
    radius: float,
    layout_family: str,
    candidate_count: int,
    target_position: int,
    drag_distance_band: str,
    requested_drag_distance: float,
) -> tuple[tuple[float, float], tuple[float, float], list[tuple[float, float]]]:
    rng = _factor_rng(seed, f"geometry:{episode_id}")
    width, height = PAPER_V2_VIEWPORT
    slot_half_extents = _slot_collision_half_extents(radius)
    piece_half_extents = _piece_collision_half_extents(radius)
    margin = PAPER_V2_GEOMETRY_MARGIN_PX
    lower_distance, upper_distance = PAPER_V2_DRAG_DISTANCE_BANDS[drag_distance_band]
    requested_distance = min(max(requested_drag_distance, lower_distance), upper_distance)

    for _ in range(2000):
        try:
            offsets = _layout_offsets(
                rng,
                layout_family=layout_family,
                count=candidate_count,
                radius=radius,
            )
        except RuntimeError:
            continue
        min_offset_x = min(point[0] for point in offsets)
        max_offset_x = max(point[0] for point in offsets)
        min_offset_y = min(point[1] for point in offsets)
        max_offset_y = max(point[1] for point in offsets)
        min_anchor_x = slot_half_extents[0] + margin - min_offset_x
        max_anchor_x = width - slot_half_extents[0] - margin - max_offset_x
        min_anchor_y = slot_half_extents[1] + margin - min_offset_y
        max_anchor_y = height - slot_half_extents[1] - margin - max_offset_y
        if min_anchor_x > max_anchor_x or min_anchor_y > max_anchor_y:
            continue
        anchor = (
            rng.uniform(min_anchor_x, max_anchor_x),
            rng.uniform(min_anchor_y, max_anchor_y),
        )
        slot_positions = [
            (anchor[0] + offset[0], anchor[1] + offset[1])
            for offset in offsets
        ]
        # Meta coordinates are persisted at millipixel precision. Keep a small
        # separation margin so rounding cannot turn a valid proposal into an
        # overlapping pair during the serialized-data audit.
        if not _slot_positions_are_separated(
            slot_positions,
            radius=radius + 0.01,
        ):
            continue
        target_xy = slot_positions[target_position]

        for _ in range(500):
            distance = min(
                upper_distance,
                max(lower_distance, requested_distance + rng.uniform(-24.0, 24.0)),
            )
            angle = rng.uniform(0.0, math.tau)
            piece_xy = (
                target_xy[0] + math.cos(angle) * distance,
                target_xy[1] + math.sin(angle) * distance,
            )
            if not (
                piece_half_extents[0] + margin
                <= piece_xy[0]
                <= width - piece_half_extents[0] - margin
                and piece_half_extents[1] + margin
                <= piece_xy[1]
                <= height - piece_half_extents[1] - margin
            ):
                continue
            if any(
                _aabbs_overlap(
                    piece_xy,
                    slot_xy,
                    left_half_extents=piece_half_extents,
                    right_half_extents=slot_half_extents,
                    margin=margin,
                )
                for slot_xy in slot_positions
            ):
                continue
            return piece_xy, target_xy, slot_positions
    raise RuntimeError(
        f"failed to sample visible third-person v2 geometry for {episode_id}"
    )


def _build_episode_meta(
    *,
    episode_id: str,
    split_name: str,
    distribution: str,
    shape: str,
    color_name: str,
    color: str,
    radius: int,
    layout_family: str,
    candidate_count: int,
    target_position: int,
    drag_distance_band: str,
    requested_drag_distance: float,
    seed: int,
    version: int = PAPER_V2_VERSION,
    dataset_profile: str = PAPER_V2_PROFILE,
    held_move_count: int = 1,
    trajectory_profile: str | None = None,
) -> dict[str, Any]:
    piece_xy, target_xy, slot_positions = _screen_geometry(
        seed=seed,
        episode_id=episode_id,
        radius=radius,
        layout_family=layout_family,
        candidate_count=candidate_count,
        target_position=target_position,
        drag_distance_band=drag_distance_band,
        requested_drag_distance=requested_drag_distance,
    )
    distractor_shapes = _distractor_shapes(
        _factor_rng(seed, f"distractor_shapes:{episode_id}"),
        distribution=distribution,
        target_shape=shape,
        count=candidate_count - 1,
    )
    distractor_index = 0
    slot_options: list[dict[str, Any]] = []
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
                "center_screen_xy": [round(center_xy[0], 3), round(center_xy[1], 3)],
                "shape": option_shape,
                "color_name": "gray",
                "color": "#94a3b8",
            }
        )
    _factor_rng(seed, f"option_order:{episode_id}").shuffle(slot_options)

    drag_distance = math.dist(piece_xy, target_xy)
    tolerance = radius * DEFAULT_TOLERANCE_RADIUS_SCALE
    oracle = _drag_oracle_sequence(
        piece_xy,
        target_xy,
        held_move_count=held_move_count,
    )
    meta = {
        "episode_id": episode_id,
        "version": version,
        "dataset_profile": dataset_profile,
        "split": split_name,
        "distribution": distribution,
        "benchmark": "ThirdPersonDragCaptcha",
        "task_type": "third_person_drag_captcha",
        "perspective": "third_person",
        "viewport": list(PAPER_V2_VIEWPORT),
        "cursor_lock": "none",
        "camera_motion": False,
        "coordinate_contract": "qwen3_relative_0_1000_absolute_pointer",
        "movement_mapping": "absolute_screen_position_direct",
        "drag_screen_displacement_scale": 1.0,
        "initial_cursor_screen_xy": [64.0, 64.0],
        "piece_start_screen_xy": [round(piece_xy[0], 3), round(piece_xy[1], 3)],
        "slot_center_screen_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
        "slot_radius_scale": DEFAULT_SLOT_RADIUS_SCALE,
        "slot_frame_radius_px": round(radius * DEFAULT_SLOT_RADIUS_SCALE, 3),
        "tolerance_radius_scale": DEFAULT_TOLERANCE_RADIUS_SCALE,
        "tolerance_px": round(tolerance, 3),
        "layout_family": layout_family,
        "slot_candidate_count": candidate_count,
        "target_layout_position_index": target_position,
        "drag_distance_band": drag_distance_band,
        "drag_distance_px": round(drag_distance, 3),
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
    }
    if trajectory_profile is not None:
        meta.update(
            {
                "geometry_profile": PAPER_V2_PROFILE,
                "trajectory_profile": trajectory_profile,
                "held_move_step_count": held_move_count,
            }
        )
    return meta


def _generate_split(
    *,
    test_root: Path,
    split_name: str,
    distribution: str,
    count: int,
    seed: int,
    training_trajectory: bool = False,
) -> list[str]:
    shape_pool = PAPER_V2_TRAIN_SHAPES if distribution == "iid" else PAPER_V2_OOD_SHAPES
    shapes = _balanced_schedule(
        _split_balanced_values(shape_pool, split_name=split_name),
        count=count,
        rng=_factor_rng(seed, "piece_shape"),
    )
    colors = _balanced_schedule(
        _split_balanced_values(COLORS, split_name=split_name),
        count=count,
        rng=_factor_rng(seed, "piece_color"),
    )
    layouts = _balanced_schedule(
        _split_balanced_values(PAPER_V2_LAYOUTS, split_name=split_name),
        count=count,
        rng=_factor_rng(seed, "layout_family"),
    )
    candidate_counts = [
        int(value)
        for value in _balanced_schedule(
            _split_balanced_values(PAPER_V2_CANDIDATE_COUNTS, split_name=split_name),
            count=count,
            rng=_factor_rng(seed, "candidate_count"),
        )
    ]
    target_positions = _balanced_target_position_schedule(candidate_counts, seed=seed)
    drag_distances = _drag_distance_schedule(count, seed=seed)
    held_move_counts = (
        _held_move_count_schedule(drag_distances, seed=seed)
        if training_trajectory
        else [1] * count
    )
    radius_rng = _factor_rng(seed, "piece_radius")

    episode_ids: list[str] = []
    for index in range(count):
        episode_id = f"tpd_{split_name}_{index + 1:04d}"
        color_name, color = colors[index]
        drag_distance_band, requested_drag_distance = drag_distances[index]
        meta = _build_episode_meta(
            episode_id=episode_id,
            split_name=split_name,
            distribution=distribution,
            shape=str(shapes[index]),
            color_name=str(color_name),
            color=str(color),
            radius=radius_rng.randint(26, 44),
            layout_family=str(layouts[index]),
            candidate_count=candidate_counts[index],
            target_position=target_positions[index],
            drag_distance_band=drag_distance_band,
            requested_drag_distance=requested_drag_distance,
            seed=seed,
            version=(
                PAPER_V3_TRAIN_VERSION if training_trajectory else PAPER_V2_VERSION
            ),
            dataset_profile=(
                PAPER_V3_TRAIN_PROFILE if training_trajectory else PAPER_V2_PROFILE
            ),
            held_move_count=held_move_counts[index],
            trajectory_profile=(
                PAPER_V3_TRAIN_TRAJECTORY_PROFILE
                if training_trajectory
                else None
            ),
        )
        episode_dir = test_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        episode_ids.append(episode_id)
    return episode_ids


def _load_meta_records(
    test_root: Path,
    episode_ids: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        json.loads((test_root / episode_id / "meta.json").read_text(encoding="utf-8"))
        for episode_id in episode_ids
    ]


def _static_validation(
    *,
    test_root: Path,
    split_ids: dict[str, list[str]],
    allowed_held_move_counts: Sequence[int] = (1,),
    include_trajectory_summary: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    allowed_held_move_count_set = {int(value) for value in allowed_held_move_counts}
    all_ids = [episode_id for episode_ids in split_ids.values() for episode_id in episode_ids]
    if len(all_ids) != len(set(all_ids)):
        issues.append("episode ids are not unique across splits")

    split_summaries: dict[str, Any] = {}
    total_slot_overlap_episodes = 0
    total_piece_slot_overlap_episodes = 0
    for split_name, episode_ids in split_ids.items():
        records = _load_meta_records(test_root, episode_ids)
        expected_distribution = "ood" if split_name.startswith("ood") else "iid"
        distributions = {str(record.get("distribution")) for record in records}
        shapes = Counter(str(record["piece"]["shape"]) for record in records)
        layouts = Counter(str(record["layout_family"]) for record in records)
        candidate_counts = Counter(str(record["slot_candidate_count"]) for record in records)
        distance_bands = Counter(str(record["drag_distance_band"]) for record in records)
        held_move_counts: Counter[str] = Counter()
        distance_held_pairs: Counter[str] = Counter()
        target_positions = Counter(
            f"{record['slot_candidate_count']}:{record['target_layout_position_index']}"
            for record in records
        )
        slot_overlap_episodes = 0
        piece_slot_overlap_episodes = 0

        for record in records:
            episode_id = str(record["episode_id"])
            target_slots = [
                option
                for option in record["slot_options"]
                if option.get("role") == "target"
            ]
            if len(target_slots) != 1:
                issues.append(f"{episode_id}: expected exactly one target slot")
                continue
            if target_slots[0].get("shape") != record["piece"].get("shape"):
                issues.append(f"{episode_id}: target shape does not match piece")
            if any(
                option.get("role") == "distractor"
                and option.get("shape") == record["piece"].get("shape")
                for option in record["slot_options"]
            ):
                issues.append(f"{episode_id}: duplicate matching distractor slot")
            if tuple(record["viewport"]) != PAPER_V2_VIEWPORT:
                issues.append(f"{episode_id}: viewport is not 1280x720")
            if "sensitivity" in record or "view_zoom" in record:
                issues.append(f"{episode_id}: third-person metadata contains view conversion")
            oracle = record["oracle_primitive_sequence"]
            oracle_kinds = [action.get("kind") for action in oracle]
            held_move_count = len(oracle_kinds) - 3
            if not (
                oracle_kinds[:2] == ["move_to", "mouse_down"]
                and oracle_kinds[-1:] == ["mouse_up"]
                and set(oracle_kinds[2:-1]) == {"move_to"}
                and held_move_count in allowed_held_move_count_set
            ):
                issues.append(f"{episode_id}: invalid drag oracle sequence")
            if int(record.get("held_move_step_count", 1)) != held_move_count:
                issues.append(f"{episode_id}: held move count metadata mismatch")
            held_move_counts[str(held_move_count)] += 1
            distance_held_pairs[
                f"{record['drag_distance_band']}:{held_move_count}"
            ] += 1
            lower, upper = PAPER_V2_DRAG_DISTANCE_BANDS[str(record["drag_distance_band"])]
            if not lower <= float(record["drag_distance_px"]) <= upper:
                issues.append(f"{episode_id}: drag distance is outside its band")

            radius = float(record["piece"]["radius_px"])
            slot_positions = [
                _coerce_xy(option["center_screen_xy"])
                for option in record["slot_options"]
            ]
            if not _slot_positions_are_separated(slot_positions, radius=radius):
                slot_overlap_episodes += 1
            piece_xy = _coerce_xy(record["piece_start_screen_xy"])
            piece_half_extents = _piece_collision_half_extents(radius)
            slot_half_extents = _slot_collision_half_extents(radius)
            if any(
                _aabbs_overlap(
                    piece_xy,
                    slot_xy,
                    left_half_extents=piece_half_extents,
                    right_half_extents=slot_half_extents,
                    margin=PAPER_V2_GEOMETRY_MARGIN_PX,
                )
                for slot_xy in slot_positions
            ):
                piece_slot_overlap_episodes += 1

            option_shapes = {str(option["shape"]) for option in record["slot_options"]}
            if expected_distribution == "iid":
                if not option_shapes <= set(PAPER_V2_TRAIN_SHAPES):
                    issues.append(f"{episode_id}: IID episode contains held-out shape")
            else:
                distractor_shapes = {
                    str(option["shape"])
                    for option in record["slot_options"]
                    if option.get("role") == "distractor"
                }
                if not distractor_shapes & set(PAPER_V2_TRAIN_SHAPES):
                    issues.append(f"{episode_id}: OOD distractors contain no seen shape")
                if not distractor_shapes & set(PAPER_V2_OOD_SHAPES):
                    issues.append(f"{episode_id}: OOD distractors contain no held-out shape")

        expected_shapes = (
            set(PAPER_V2_OOD_SHAPES)
            if expected_distribution == "ood"
            else set(PAPER_V2_TRAIN_SHAPES)
        )
        if distributions != {expected_distribution}:
            issues.append(
                f"{split_name}: distributions={sorted(distributions)!r}, "
                f"expected={[expected_distribution]!r}"
            )
        if set(shapes) != expected_shapes:
            issues.append(
                f"{split_name}: shapes={sorted(shapes)!r}, "
                f"expected={sorted(expected_shapes)!r}"
            )
        if max(distance_bands.values(), default=0) - min(distance_bands.values(), default=0) > 1:
            issues.append(f"{split_name}: drag distance bands are not balanced")
        if include_trajectory_summary:
            for drag_distance_band in PAPER_V2_DRAG_DISTANCE_BANDS:
                pair_counts = [
                    distance_held_pairs[
                        f"{drag_distance_band}:{held_move_count}"
                    ]
                    for held_move_count in allowed_held_move_counts
                ]
                if max(pair_counts, default=0) - min(pair_counts, default=0) > 1:
                    issues.append(
                        f"{split_name}: held move counts are not balanced within "
                        f"{drag_distance_band} drag distance"
                    )
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
        split_summary = {
            "records": len(records),
            "distribution": expected_distribution,
            "shape_distribution": dict(sorted(shapes.items())),
            "layout_distribution": dict(sorted(layouts.items())),
            "candidate_count_distribution": dict(sorted(candidate_counts.items())),
            "drag_distance_band_distribution": dict(sorted(distance_bands.items())),
            "target_position_distribution": dict(sorted(target_positions.items())),
            "geometry_overlap_episode_counts": {
                "slot_slot": slot_overlap_episodes,
                "piece_slot": piece_slot_overlap_episodes,
            },
        }
        if include_trajectory_summary:
            split_summary.update(
                {
                    "held_move_step_count_distribution": dict(
                        sorted(held_move_counts.items())
                    ),
                    "drag_distance_held_move_distribution": dict(
                        sorted(distance_held_pairs.items())
                    ),
                }
            )
        split_summaries[split_name] = split_summary
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
    if output_dir is None:
        output_dir = paper_v2_episodes_root()
    test_root = output_dir / "test"
    if clean and test_root.exists():
        for child in test_root.glob("tpd_*"):
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
        "benchmark": "ThirdPersonDragCaptcha",
        "version": PAPER_V2_VERSION,
        "dataset_profile": PAPER_V2_PROFILE,
        "output_dir": str(output_dir),
        "viewport": list(PAPER_V2_VIEWPORT),
        "image_max_pixels": PAPER_V2_VIEWPORT[0] * PAPER_V2_VIEWPORT[1],
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
            "iid_shapes": list(PAPER_V2_TRAIN_SHAPES),
            "ood_shapes": list(PAPER_V2_OOD_SHAPES),
            "sets_are_disjoint": set(PAPER_V2_TRAIN_SHAPES).isdisjoint(PAPER_V2_OOD_SHAPES),
            "iid_meaning": "in-distribution relative to the task-specific v2 training set",
        },
        "slot_policy": {
            "candidate_counts": list(PAPER_V2_CANDIDATE_COUNTS),
            "layout_families": list(PAPER_V2_LAYOUTS),
            "target_position_assignment": "balanced_after_unlabeled_layout_generation",
            "duplicate_matching_distractor_allowed": False,
        },
        "direct_drag_policy": {
            "distance_bands_px": {
                key: list(value)
                for key, value in PAPER_V2_DRAG_DISTANCE_BANDS.items()
            },
            "cursor_lock": "none",
            "camera_motion": False,
            "drag_screen_displacement_scale": 1.0,
            "sensitivity_conversion": False,
            "view_zoom_conversion": False,
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
    clean: bool = True,
) -> dict[str, Any]:
    """Generate an IID-only episode pool for task-specific SFT rendering."""

    if count < 1:
        raise ValueError("count must be positive")
    train_root = output_dir / "train"
    if clean and train_root.exists():
        for child in train_root.glob("tpd_train_*"):
            if child.is_dir():
                shutil.rmtree(child)
    train_root.mkdir(parents=True, exist_ok=True)
    episode_ids = _generate_split(
        test_root=train_root,
        split_name="train",
        distribution="iid",
        count=count,
        seed=seed,
        training_trajectory=True,
    )
    validation = _static_validation(
        test_root=train_root,
        split_ids={"train": episode_ids},
        allowed_held_move_counts=PAPER_V3_TRAIN_HELD_MOVE_COUNTS,
        include_trajectory_summary=True,
    )
    manifest = {
        "benchmark": "ThirdPersonDragCaptcha",
        "version": PAPER_V3_TRAIN_VERSION,
        "dataset_profile": PAPER_V3_TRAIN_PROFILE,
        "geometry_profile": PAPER_V2_PROFILE,
        "output_dir": str(output_dir),
        "viewport": list(PAPER_V2_VIEWPORT),
        "image_max_pixels": PAPER_V2_VIEWPORT[0] * PAPER_V2_VIEWPORT[1],
        "train_count": count,
        "seed": seed,
        "splits": {"train": episode_ids},
        "shape_policy": {
            "train_iid_shapes": list(PAPER_V2_TRAIN_SHAPES),
            "ood_shapes": list(PAPER_V2_OOD_SHAPES),
            "ood_shapes_present": False,
        },
        "slot_policy": {
            "candidate_counts": list(PAPER_V2_CANDIDATE_COUNTS),
            "layout_families": list(PAPER_V2_LAYOUTS),
            "target_position_assignment": "balanced_after_unlabeled_layout_generation",
        },
        "direct_drag_policy": {
            "distance_bands_px": {
                key: list(value)
                for key, value in PAPER_V2_DRAG_DISTANCE_BANDS.items()
            },
            "cursor_lock": "none",
            "camera_motion": False,
            "drag_screen_displacement_scale": 1.0,
            "sensitivity_conversion": False,
            "view_zoom_conversion": False,
        },
        "trajectory_policy": {
            "trajectory_profile": PAPER_V3_TRAIN_TRAJECTORY_PROFILE,
            "action_pattern": (
                "move_to -> mouse_down -> move_to * {2,3} -> mouse_up"
            ),
            "held_move_step_counts": list(PAPER_V3_TRAIN_HELD_MOVE_COUNTS),
            "assignment": PAPER_V3_TRAIN_HELD_MOVE_ASSIGNMENT,
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
    from .environment import ThirdPersonDragCaptchaEnv

    if episode_ids is None:
        episode_ids = sorted(
            path.name
            for path in dataset_root.iterdir()
            if path.is_dir() and (path / "meta.json").is_file()
        )
    failures: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="third-person-drag-v2-oracle-") as temporary_dir:
        env = ThirdPersonDragCaptchaEnv(
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
        description=(
            "Generate ThirdPersonDragCaptcha paper-v3 train or paper-v2 IID/OOD test data."
        )
    )
    parser.add_argument("--mode", choices=("test", "train"), default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--train-seed", type=int, default=PAPER_V2_TRAIN_SEED)
    parser.add_argument("--validate-oracles", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = (
            paper_v3_sft_root()
            if args.mode == "train"
            else paper_v2_episodes_root()
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "train":
        from .sft import generate_sft_dataset

        output_dir = args.output_dir or paper_v3_sft_root()
        result = generate_sft_dataset(
            output_dir=output_dir,
            count=args.train_count,
            seed=args.train_seed,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("validation", {}).get("status") == "passed" else 1

    output_dir = args.output_dir or paper_v2_episodes_root()
    result = generate_test_dataset(output_dir=output_dir)
    oracle_validation = None
    if args.validate_oracles:
        oracle_validation = validate_oracle_sequences(dataset_root=output_dir / "test")
        (output_dir / "oracle_validation.json").write_text(
            json.dumps(oracle_validation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result = {**result, "oracle_validation": oracle_validation}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    valid = result.get("validation", {}).get("status") == "passed"
    if oracle_validation is not None:
        valid = valid and oracle_validation.get("status") == "passed"
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
