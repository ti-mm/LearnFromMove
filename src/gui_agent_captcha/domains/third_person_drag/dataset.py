from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

from .paths import legacy_episodes_root

DEFAULT_TRAIN_COUNT = 0
DEFAULT_VAL_COUNT = 0
DEFAULT_TEST_COUNT = 150
DEFAULT_VIEWPORT = (1280, 720)
DEFAULT_SLOT_RADIUS_SCALE = 1.35
DEFAULT_TOLERANCE_RADIUS_SCALE = 1.15
TASK_DESCRIPTION = "Drag the solid colored shape into the matching gray outline."

SHAPES = ("square", "triangle", "diamond", "hexagon")
SLOT_SHAPES = (
    "square",
    "triangle",
    "diamond",
    "hexagon",
    "notched_rectangle",
    "wide_rectangle",
)
COLORS = (
    ("emerald", "#16a34a"),
    ("blue", "#2563eb"),
    ("amber", "#d97706"),
    ("lime", "#65a30d"),
    ("indigo", "#4f46e5"),
)
SPLIT_SEED_OFFSETS = {
    "train": 0,
    "val": 1_000_003,
    "test": 2_000_003,
}


def _screen_to_model_xy(
    screen_xy: tuple[float, float],
    *,
    viewport: tuple[int, int],
) -> dict[str, float | str]:
    return {
        "kind": "move_to",
        "x": round(screen_xy[0] / viewport[0] * 1000.0, 3),
        "y": round(screen_xy[1] / viewport[1] * 1000.0, 3),
    }


def _object_description(*, color_name: str, shape: str) -> str:
    return f"solid {color_name} {shape}"


def _slot_description(*, shape: str) -> str:
    return f"large hollow gray {shape.replace('_', ' ')} slot"


def _shape_mismatches(target_shape: str) -> list[str]:
    return [shape for shape in SLOT_SHAPES if shape != target_shape]


def _grid_slot_centers(viewport: tuple[int, int]) -> list[tuple[float, float]]:
    width, height = viewport
    xs = (width * 0.57, width * 0.70, width * 0.83, width * 0.93)
    ys = (height * 0.25, height * 0.50, height * 0.75)
    return [(x, y) for y in ys for x in xs]


def _sample_piece_xy(
    rng: random.Random,
    *,
    viewport: tuple[int, int],
    radius: float,
) -> tuple[float, float]:
    width, height = viewport
    margin = radius + 72.0
    return (
        rng.uniform(margin, width * 0.37),
        rng.uniform(margin, height - margin),
    )


def _sample_slot_options(
    rng: random.Random,
    *,
    viewport: tuple[int, int],
    piece_xy: tuple[float, float],
    target_shape: str,
    radius: float,
) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    candidates = _grid_slot_centers(viewport)
    rng.shuffle(candidates)
    min_piece_distance = max(320.0, radius * 8.0)
    eligible = [xy for xy in candidates if math.dist(piece_xy, xy) >= min_piece_distance]
    if len(eligible) < 6:
        raise RuntimeError("not enough visible slot positions for third-person drag episode")
    selected = eligible[:6]
    target_xy = selected[0]
    mismatch_shapes = _shape_mismatches(target_shape)
    options: list[dict[str, Any]] = [
        {
            "id": "slot_target",
            "role": "target",
            "center_screen_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
            "shape": target_shape,
            "color_name": "gray",
            "color": "#94a3b8",
        }
    ]
    for index, (center_xy, shape) in enumerate(zip(selected[1:], mismatch_shapes, strict=True)):
        options.append(
            {
                "id": f"slot_decoy_shape_{index + 1}",
                "role": "distractor",
                "center_screen_xy": [round(center_xy[0], 3), round(center_xy[1], 3)],
                "shape": shape,
                "color_name": "gray",
                "color": "#94a3b8",
            }
        )
    rng.shuffle(options)
    return target_xy, options


def _episode_id(split: str, index: int) -> str:
    if split == "test":
        return f"tpd_{index + 1:04d}"
    return f"tpd_{split}_{index + 1:04d}"


def _generate_split(
    *,
    output_dir: Path,
    split: str,
    count: int,
    seed: int,
    viewport: tuple[int, int],
    clean: bool,
) -> list[str]:
    split_root = output_dir / split
    split_root.mkdir(parents=True, exist_ok=True)
    prefix = "tpd_" if split == "test" else f"tpd_{split}_"
    if clean:
        for child in split_root.glob(f"{prefix}*"):
            if child.is_dir():
                shutil.rmtree(child)

    rng = random.Random(seed + SPLIT_SEED_OFFSETS[split])
    episode_ids: list[str] = []
    for index in range(count):
        episode_id = _episode_id(split, index)
        shape = SHAPES[index % len(SHAPES)]
        color_name, color = COLORS[index % len(COLORS)]
        radius = float(rng.choice([30, 32, 34, 36]))
        tolerance = radius * DEFAULT_TOLERANCE_RADIUS_SCALE
        piece_xy = _sample_piece_xy(rng, viewport=viewport, radius=radius)
        target_xy, slot_options = _sample_slot_options(
            rng,
            viewport=viewport,
            piece_xy=piece_xy,
            target_shape=shape,
            radius=radius,
        )
        oracle = [
            _screen_to_model_xy(piece_xy, viewport=viewport),
            {"kind": "mouse_down"},
            _screen_to_model_xy(target_xy, viewport=viewport),
            {"kind": "mouse_up"},
        ]
        meta = {
            "episode_id": episode_id,
            "version": 1,
            "benchmark": "ThirdPersonDragCaptcha",
            "task_type": "third_person_drag_captcha",
            "perspective": "third_person",
            "viewport": list(viewport),
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
        episode_dir = split_root / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        episode_ids.append(episode_id)
    return episode_ids


def generate_dataset(
    *,
    output_dir: Path | None = None,
    train_count: int = DEFAULT_TRAIN_COUNT,
    val_count: int = DEFAULT_VAL_COUNT,
    test_count: int = DEFAULT_TEST_COUNT,
    seed: int = 20260713,
    viewport: tuple[int, int] = DEFAULT_VIEWPORT,
    clean: bool = True,
) -> dict[str, Any]:
    if output_dir is None:
        output_dir = legacy_episodes_root()
    counts = {"train": train_count, "val": val_count, "test": test_count}
    if any(count < 0 for count in counts.values()):
        raise ValueError("split counts must be non-negative")
    if viewport != DEFAULT_VIEWPORT:
        raise ValueError(
            f"paper-bound third-person drag data must use {DEFAULT_VIEWPORT}, got {viewport}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        split: _generate_split(
            output_dir=output_dir,
            split=split,
            count=count,
            seed=seed,
            viewport=viewport,
            clean=clean,
        )
        for split, count in counts.items()
    }
    manifest = {
        "benchmark": "ThirdPersonDragCaptcha",
        "task_type": "third_person_drag_captcha",
        "perspective": "third_person",
        "output_dir": str(output_dir),
        "seed": seed,
        "viewport": list(viewport),
        "image_max_pixels": viewport[0] * viewport[1],
        "split_counts": counts,
        "splits": splits,
        "summary": {
            "slot_candidate_count_per_case": 6,
            "slot_distractor_count_per_case": 5,
            "cursor_lock": "none",
            "camera_motion": False,
            "coordinate_contract": "qwen3_relative_0_1000_absolute_pointer",
            "movement_mapping": "absolute_screen_position_direct",
            "drag_screen_displacement_scale": 1.0,
            "sensitivity_conversion": False,
            "view_zoom_conversion": False,
            "all_objects_visible_in_fixed_view": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate fixed-camera third-person drag CAPTCHA episodes.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--val-count", type=int, default=DEFAULT_VAL_COUNT)
    parser.add_argument("--test-count", type=int, default=DEFAULT_TEST_COUNT)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    manifest = generate_dataset(
        output_dir=args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
