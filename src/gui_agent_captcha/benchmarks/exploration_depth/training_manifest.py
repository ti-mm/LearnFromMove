from __future__ import annotations

import argparse
import copy
import filecmp
import html
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from ...integrations.storage import storage_path
from .contracts import EXPLORATION_BENCHMARK_VARIANTS
from .manifest import (
    FPS_DIRECTION_XY,
    _offscreen_first_person_view_center,
    _screen_to_world,
    _ten_choice_html,
)

TRAINING_SUITE_ID = "exploration_depth_train_6x4000_v2"
DEFAULT_PAIR_COUNT = 4_000
DEFAULT_VIEWPORT = (1280, 720)
SENSITIVITY_BANDS = {
    "low": (0.5, 0.8),
    "high": (1.2, 1.5),
}

ROTATION_MET_POOL = "rotation_met_rl_novel_20260629"
ROTATION_DONOR_TRANSFORM = {
    "exif_transpose": True,
    "color_mode": "RGB",
    "fit": "center",
    "resampling": "LANCZOS",
    "format": "JPEG",
    "quality": 95,
    "subsampling": 0,
    "optimize": False,
    "progressive": False,
}
# One-based background ordinals.  Each ordinal backs four canonical dynamics
# cases, and both paired variants consume the same transformed asset.
ROTATION_DONOR_BY_BACKGROUND_ORDINAL = {
    22: "met-436039",
    23: "met-436554",
    27: "met-437216",
    63: "met-435621",
    80: "met-467639",
    87: "met-438670",
    163: "met-53002",
    172: "met-670541",
    176: "met-699761",
    179: "met-438015",
    197: "met-839045",
    203: "met-437292",
    222: "met-453193",
    240: "met-816766",
    259: "met-13638",
    287: "met-435922",
    309: "met-471911",
    338: "met-283219",
    357: "met-451731",
    366: "met-439344",
    373: "met-53449",
    389: "met-36131",
    405: "met-446561",
    410: "met-437489",
    429: "met-757056",
    445: "met-325564",
    562: "met-49179",
    580: "met-452658",
    605: "met-437518",
    618: "met-451726",
    625: "met-438023",
    649: "met-437310",
    704: "met-438545",
    719: "met-343144",
    734: "met-436528",
    791: "met-72326",
    840: "met-436785",
    849: "met-437261",
    864: "met-10771",
    883: "met-192770",
    900: "met-436535",
    974: "met-438624",
}


def default_rotation_donor_dir() -> Path:
    return storage_path(
        "artifacts",
        "background_pools",
        ROTATION_MET_POOL,
        "donors_web_large",
    )


def default_rotation_donor_metadata() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "artifacts/background_pools"
        / ROTATION_MET_POOL
        / "metadata.json"
    )


@dataclass(frozen=True)
class TrainingSourceRoots:
    ten_choice: Path
    rotation_records: Path
    rotation_background_metadata: Path
    rotation_donor_dir: Path
    rotation_donor_metadata: Path
    third_person_drag: Path

    @classmethod
    def defaults(cls) -> "TrainingSourceRoots":
        return cls(
            ten_choice=storage_path(
                "artifacts",
                "datasets",
                "ten_choice_random_scan_source_4k_20260624",
                "train",
            ),
            rotation_records=storage_path(
                "artifacts",
                "datasets",
                "lujiahao_rotation_sft_4k_train_only_openimages1k_bgfix_english_ui_"
                "20260622",
                "audit.jsonl",
            ),
            rotation_background_metadata=(
                Path(__file__).resolve().parents[4]
                / "artifacts/background_pools/rotation_openimages_20260604/"
                "train_1000_metadata.json"
            ),
            rotation_donor_dir=default_rotation_donor_dir(),
            rotation_donor_metadata=default_rotation_donor_metadata(),
            third_person_drag=storage_path(
                "artifacts",
                "datasets",
                "third_person_drag_captcha_sft_v2_4k_720p_20260730",
                "episodes",
                "train",
            ),
        )


def default_training_root() -> Path:
    return storage_path("data", "training", TRAINING_SUITE_ID)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"expected JSON objects in {path}")
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _met_donor_records(
    metadata_path: Path, *, required_ids: set[str] | None = None
) -> dict[str, dict[str, Any]]:
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"MET donor metadata must be a list: {metadata_path}")
    by_id = {
        str(record["id"]): dict(record)
        for record in records
        if isinstance(record, dict) and record.get("id")
    }
    required = (
        set(ROTATION_DONOR_BY_BACKGROUND_ORDINAL.values())
        if required_ids is None
        else required_ids
    )
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"MET donor metadata is missing ids: {missing[:3]}")
    return by_id


def _web_large_url(record: dict[str, Any]) -> str:
    source = str(record["imageUrl"])
    if "/original/" not in source:
        raise ValueError(f"MET imageUrl has no /original/ path segment: {source}")
    return source.replace("/original/", "/web-large/", 1)


def _rotation_donor_provenance(
    *, donor_id: str, donor_record: dict[str, Any]
) -> dict[str, Any]:
    return {
        "source_dataset": ROTATION_MET_POOL,
        "source_id": donor_id,
        "source_background_image_url": _web_large_url(donor_record),
        "source_file": f"{donor_id}.jpg",
        "license": str(donor_record.get("license") or "CC0"),
        "transform": copy.deepcopy(ROTATION_DONOR_TRANSFORM),
    }


def _fit_rotation_donor(
    *, donor_path: Path, reference_path: Path, destination_path: Path
) -> None:
    if not donor_path.is_file():
        raise FileNotFoundError(donor_path)
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    with Image.open(reference_path) as reference:
        target_size = reference.size
    with Image.open(donor_path) as raw:
        fitted = ImageOps.fit(
            ImageOps.exif_transpose(raw).convert("RGB"),
            target_size,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        fitted.save(
            destination_path,
            format="JPEG",
            quality=95,
            subsampling=0,
            optimize=False,
            progressive=False,
        )


def _assert_direct_byte_unique(paths: list[Path]) -> None:
    groups_by_size: dict[int, list[Path]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        peers = groups_by_size.setdefault(path.stat().st_size, [])
        duplicate = next(
            (peer for peer in peers if filecmp.cmp(path, peer, shallow=False)), None
        )
        if duplicate is not None:
            raise ValueError(
                f"rotation backgrounds are direct-byte duplicates: {duplicate}, {path}"
            )
        peers.append(path)


def _rotation_background_ordinal(path_value: str) -> int:
    match = re.search(r"train-bg-(\d{4})\.[^.]+$", path_value)
    if match is None:
        raise ValueError(f"unrecognized rotation training asset: {path_value}")
    return int(match.group(1))


def _pair_id(family: str, index: int) -> str:
    prefix = {"ten_choice": "ten", "rotation": "rot", "drag": "drag"}[family]
    return f"ed-train-{prefix}-{index + 1:04d}"


def _episode_id(pair_id: str, variant: str) -> str:
    return f"{pair_id}--{variant.replace('_', '-')}"


def _sensitivity(index: int) -> tuple[str, float]:
    band = "low" if index % 2 == 0 else "high"
    lower, upper = SENSITIVITY_BANDS[band]
    milli_lower = int(lower * 1000)
    milli_upper = int(upper * 1000)
    milli = milli_lower + ((index * 73 + 29) % (milli_upper - milli_lower + 1))
    return band, milli / 1000.0


def _common_episode(
    *,
    pair_id: str,
    family: str,
    variant: str,
    exploration_level: str,
    index: int,
    instruction: str,
    viewport: tuple[int, int],
    shared: dict[str, Any],
    environment: dict[str, Any],
    evaluator: dict[str, Any],
    canonical_case_path: str,
) -> dict[str, Any]:
    return {
        "suite_id": TRAINING_SUITE_ID,
        "episode_id": _episode_id(pair_id, variant),
        "pair_id": pair_id,
        "family": family,
        "variant": variant,
        "exploration_level": exploration_level,
        "case_seed": 202608250000
        + {"ten_choice": 0, "rotation": 10_000, "drag": 20_000}[family]
        + index,
        "split": "train",
        "viewport": list(viewport),
        "instruction": instruction,
        "shared_scene_config": copy.deepcopy(shared),
        "environment_config": copy.deepcopy(environment),
        "success_evaluator": copy.deepcopy(evaluator),
        "canonical_case_path": canonical_case_path,
        "policy_observation_allowlist": ["suite_id", "pair_id", "family", "viewport"],
    }


def _move_icons_off_reticle(meta: dict[str, Any]) -> dict[str, Any]:
    """Fit the 800px source layout into 720p and keep reset labels hidden."""

    value = copy.deepcopy(meta)
    centers = [[float(x), float(y)] for x, y in value["icon_centers_xy"]]
    positions = [[float(x), float(y)] for x, y in value["icon_positions_xy"]]
    center = (DEFAULT_VIEWPORT[0] / 2.0, DEFAULT_VIEWPORT[1] / 2.0)
    canvas = dict(value["canvas_rect"])
    canvas["height"] = min(float(canvas.get("height", 580.0)), 580.0)
    min_x = float(canvas["left"]) + 48.0
    max_x = float(canvas["left"]) + float(canvas["width"]) - 48.0
    # The tooltip is rendered above the 92px icon inside an overflow-hidden
    # canvas.  Keeping 104px above each center makes every revealed label
    # visible after the native 1280x800 -> 1280x720 scene reflow.
    min_y = float(canvas["top"]) + 104.0
    max_y = float(canvas["top"]) + float(canvas["height"]) - 52.0
    occupied = [tuple(point) for point in centers]
    for index, point in enumerate(centers):
        inside_canvas = (
            min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
        )
        outside_reset_reticle = (
            abs(point[0] - center[0]) > 54.0
            or abs(point[1] - center[1]) > 54.0
        )
        if inside_canvas and outside_reset_reticle:
            continue
        candidates = sorted(
            (
                (x, y)
                for x in range(int(min_x), int(max_x) + 1, 24)
                for y in range(int(min_y), int(max_y) + 1, 24)
            ),
            key=lambda candidate: (
                (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2,
                candidate[1],
                candidate[0],
            ),
        )
        replacement: tuple[float, float] | None = None
        for candidate in candidates:
            if abs(candidate[0] - center[0]) <= 54.0 and abs(candidate[1] - center[1]) <= 54.0:
                continue
            if all(
                other_index == index
                or abs(candidate[0] - other[0]) >= 96.0
                or abs(candidate[1] - other[1]) >= 96.0
                for other_index, other in enumerate(occupied)
            ):
                replacement = candidate
                break
        if replacement is None:
            raise ValueError(f"could not move reset-hovering icon in {value['episode_id']}")
        delta = (replacement[0] - point[0], replacement[1] - point[1])
        centers[index] = [replacement[0], replacement[1]]
        positions[index] = [positions[index][0] + delta[0], positions[index][1] + delta[1]]
        occupied[index] = replacement
    value["icon_centers_xy"] = centers
    value["icon_positions_xy"] = positions
    value["canvas_rect"] = canvas
    value["viewport"] = list(DEFAULT_VIEWPORT)
    return value


def _ten_choice_episodes(
    *, output_root: Path, source_root: Path, count: int
) -> list[dict[str, Any]]:
    source_dirs = sorted(
        path for path in source_root.iterdir() if (path / "meta.json").is_file()
    )
    if len(source_dirs) < count:
        raise ValueError(f"ten-choice training source has {len(source_dirs)}, need {count}")
    episodes: list[dict[str, Any]] = []
    for index, source_dir in enumerate(source_dirs[:count]):
        source = _move_icons_off_reticle(_read_json(source_dir / "meta.json"))
        pair_id = _pair_id("ten_choice", index)
        case_dir = output_root / "cases/ten_choice" / pair_id
        target_label = str(source["target_label"])
        instruction_html = (
            f'Click the icon that displays "<strong>{html.escape(target_label)}</strong>".'
        )
        _copy_file(
            source_dir / str(source.get("icon_asset", "assets/icon.svg")),
            case_dir / "assets/icon.svg",
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "index.html").write_text(
            _ten_choice_html(source, instruction=instruction_html), encoding="utf-8"
        )
        band, sensitivity = _sensitivity(index)
        shared = {
            "scene_schema": "paired_ten_choice_train_v1",
            "source_case_id": source["episode_id"],
            "background": "ten_choice_gradient_canvas_v1",
            "icon_asset": f"cases/ten_choice/{pair_id}/assets/icon.svg",
            "labels": list(source["labels"]),
            "icon_centers_xy": copy.deepcopy(source["icon_centers_xy"]),
            "icon_positions_xy": copy.deepcopy(source["icon_positions_xy"]),
            "canvas_rect": copy.deepcopy(source["canvas_rect"]),
            "initial_cursor_or_reticle_xy": [640.0, 360.0],
            "target_object": {
                "target_index": int(source["target_idx"]),
                "target_label": target_label,
            },
            "hidden_dynamics": {
                "sensitivity": sensitivity,
                "sensitivity_band": band,
                "direction_xy": list(FPS_DIRECTION_XY),
            },
            "html_path": f"cases/ten_choice/{pair_id}/index.html",
        }
        _write_json(case_dir / "meta.json", shared)
        common = {
            "hidden_dynamics": copy.deepcopy(shared["hidden_dynamics"]),
            "minimum_steps_gate": None,
        }
        for variant, level, environment, evaluator in (
            (
                "ten_choice_third_person",
                "L1",
                {
                    **common,
                    "responsive_coordinate_system": "screen",
                    "interaction_marker": "mouse_icon",
                    "reset_frame_contract": "shared_scene_with_exocentric_mouse_icon",
                },
                {"type": "clicked_target_icon", "terminal_event": "left_click"},
            ),
            (
                "ten_choice_first_person",
                "L2",
                {
                    **common,
                    "responsive_coordinate_system": "scene_offset_from_center",
                    "interaction_marker": "red_dot",
                    "reset_frame_contract": "shared_scene_with_egocentric_red_dot",
                },
                {
                    "type": "target_icon_inside_center_interaction_zone",
                    "terminal_event": "fixed_center_left_click",
                },
            ),
        ):
            episodes.append(
                _common_episode(
                    pair_id=pair_id,
                    family="ten_choice",
                    variant=variant,
                    exploration_level=level,
                    index=index,
                    instruction=f'Click the icon that displays "{target_label}".',
                    viewport=DEFAULT_VIEWPORT,
                    shared=shared,
                    environment=environment,
                    evaluator=evaluator,
                    canonical_case_path=f"cases/ten_choice/{pair_id}/meta.json",
                )
            )
    return episodes


def _rotation_background_map(metadata_path: Path) -> dict[str, Path]:
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    mapping = {
        str(record["localPath"]): Path(str(record["sourceAbsolutePath"]))
        for record in records
    }
    missing = [str(path) for path in mapping.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"rotation training backgrounds are missing: {missing[:3]}")
    return mapping


def _scale_box(box: dict[str, Any], source: tuple[int, int]) -> dict[str, float]:
    sx = DEFAULT_VIEWPORT[0] / source[0]
    sy = DEFAULT_VIEWPORT[1] / source[1]
    return {
        "x": float(box["x"]) * sx,
        "y": float(box["y"]) * sy,
        "width": float(box["width"]) * sx,
        "height": float(box["height"]) * sy,
    }


def _rotation_episodes(
    *,
    output_root: Path,
    records_path: Path,
    metadata_path: Path,
    donor_dir: Path,
    donor_metadata_path: Path,
    count: int,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(records_path)
    if len(rows) < count:
        raise ValueError(f"rotation training source has {len(rows)}, need {count}")
    backgrounds = _rotation_background_map(metadata_path)
    donor_records = _met_donor_records(donor_metadata_path)
    copied_assets: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    for index, row in enumerate(rows[:count]):
        metadata = dict(row["metadata"])
        raw = copy.deepcopy(metadata["raw_challenge"])
        source_url = str(metadata["sampler_diagnostics"]["background_image_url"])
        source_asset = backgrounds[source_url]
        if source_url not in copied_assets:
            ordinal = len(copied_assets) + 1
            name = f"train-bg-{ordinal:04d}{source_asset.suffix.lower()}"
            destination = output_root / "assets/rotation" / name
            donor_id = ROTATION_DONOR_BY_BACKGROUND_ORDINAL.get(ordinal)
            provenance = None
            effective_source_url = source_url
            if donor_id is None:
                _copy_file(source_asset, destination)
            else:
                donor_record = donor_records[donor_id]
                _fit_rotation_donor(
                    donor_path=donor_dir / f"{donor_id}.jpg",
                    reference_path=source_asset,
                    destination_path=destination,
                )
                provenance = _rotation_donor_provenance(
                    donor_id=donor_id, donor_record=donor_record
                )
                effective_source_url = provenance["source_background_image_url"]
            copied_assets[source_url] = {
                "asset": f"assets/rotation/{name}",
                "effective_source_url": effective_source_url,
                "provenance": provenance,
            }
        copied = copied_assets[source_url]
        challenge = {key: value for key, value in raw.items() if key != "rotationRegion"}
        challenge["pairedRelativeRotation"] = True
        challenge["canonicalInitialRelativeDeg"] = (
            float(challenge["targetRotationDeg"])
            + (float(challenge["startSliderValue"]) - float(challenge["targetSliderValue"]))
            * float(challenge["degreesPerSliderUnit"])
            * float(challenge["sensitivityScale"])
            * float(challenge["rotationDirection"])
        ) % 360.0
        challenge["rotationToleranceDeg"] = min(
            5.0, float(metadata.get("rotationToleranceDeg", 5.0))
        )
        source_viewport = tuple(int(value) for value in metadata.get("viewport", [1920, 1080]))
        slider_box = _scale_box(dict(metadata["slider_box"]), source_viewport)
        pair_id = _pair_id("rotation", index)
        shared = {
            "scene_schema": "paired_relative_rotation_train_v1",
            "source_case_id": row["id"],
            "source_background_image_url": copied["effective_source_url"],
            "background_asset": copied["asset"],
            "circle_center": copy.deepcopy(challenge["circleCenter"]),
            "circle_radius": challenge["circleRadius"],
            "initial_relative_angle_deg": challenge["canonicalInitialRelativeDeg"],
            "target_relative_angle_deg": challenge["targetRotationDeg"],
            "slider_geometry": slider_box,
            "slider_min_value": challenge["sliderMinValue"],
            "slider_max_value": challenge["sliderMaxValue"],
            "start_slider_value": challenge["startSliderValue"],
            "target_slider_value": challenge["targetSliderValue"],
            "degrees_per_slider_unit": challenge["degreesPerSliderUnit"],
            "sensitivity": challenge["sensitivityScale"],
            "rotation_direction": challenge["rotationDirection"],
            "tolerance_deg": challenge["rotationToleranceDeg"],
            "initial_cursor_or_reticle_xy": [0.0, 0.0],
            "replay_challenge": challenge,
        }
        if copied["provenance"] is not None:
            shared["background_provenance"] = copy.deepcopy(copied["provenance"])
        _write_json(output_root / "cases/rotation" / pair_id / "meta.json", shared)
        hidden = {
            "degrees_per_slider_unit": challenge["degreesPerSliderUnit"],
            "sensitivity": challenge["sensitivityScale"],
            "rotation_direction": challenge["rotationDirection"],
            "target_slider_value": challenge["targetSliderValue"],
        }
        for variant, region in (("rotation_inner", "center"), ("rotation_outer", "outer")):
            episodes.append(
                _common_episode(
                    pair_id=pair_id,
                    family="rotation",
                    variant=variant,
                    exploration_level="L2",
                    index=index,
                    instruction="Drag the slider to complete verification.",
                    viewport=DEFAULT_VIEWPORT,
                    shared=shared,
                    environment={
                        "interaction_marker": "mouse_icon",
                        "responsive_region": region,
                        "hidden_mapping": hidden,
                        "minimum_steps_gate": None,
                        "reset_frame_contract": "identical_free_mouse_scene",
                    },
                    evaluator={
                        "type": "relative_angular_error_on_release",
                        "responsive_region": region,
                        "tolerance_deg": challenge["rotationToleranceDeg"],
                    },
                    canonical_case_path=f"cases/rotation/{pair_id}/meta.json",
                )
            )
    _assert_direct_byte_unique(sorted((output_root / "assets/rotation").iterdir()))
    return episodes


def _move_to_quarantine(*, source: Path, source_root: Path, quarantine_root: Path) -> bool:
    if not source.exists():
        return False
    relative = source.relative_to(source_root)
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.move(str(source), str(destination))
    return True


def repair_rotation_donor_assets(
    *,
    suite_root: Path,
    donor_dir: Path,
    donor_metadata_path: Path,
    quarantine_root: Path,
    donor_map: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Replace known low-signal rotation assets and quarantine stale traces."""

    suite_root = Path(suite_root)
    donor_dir = Path(donor_dir)
    quarantine_root = Path(quarantine_root)
    manifest_path = suite_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if quarantine_root.exists() and any(quarantine_root.iterdir()):
        raise FileExistsError(f"quarantine directory is not empty: {quarantine_root}")
    mapping = dict(ROTATION_DONOR_BY_BACKGROUND_ORDINAL if donor_map is None else donor_map)
    if not mapping or len(set(mapping.values())) != len(mapping):
        raise ValueError("rotation donor ids must be non-empty and unique")
    donor_records = _met_donor_records(
        donor_metadata_path, required_ids=set(mapping.values())
    )
    manifest = _read_json(manifest_path)
    target_episodes: dict[int, list[dict[str, Any]]] = {ordinal: [] for ordinal in mapping}
    for episode in manifest.get("episodes", []):
        if episode.get("variant") not in {"rotation_inner", "rotation_outer"}:
            continue
        ordinal = _rotation_background_ordinal(
            str(episode["shared_scene_config"]["background_asset"])
        )
        if ordinal in target_episodes:
            target_episodes[ordinal].append(episode)
    invalid_counts = {
        ordinal: len(episodes)
        for ordinal, episodes in target_episodes.items()
        if len(episodes) != 8
    }
    if invalid_counts:
        raise ValueError(f"donor ordinals do not each select eight episodes: {invalid_counts}")

    assets_root = suite_root / "assets/rotation"
    asset_paths = sorted(path for path in assets_root.iterdir() if path.is_file())
    target_asset_paths = {
        ordinal: assets_root / f"train-bg-{ordinal:04d}.jpg" for ordinal in mapping
    }
    prepared_paths: dict[int, Path] = {}
    with tempfile.TemporaryDirectory(
        prefix="rotation-donor-repair-", dir=suite_root.parent
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        for ordinal, donor_id in sorted(mapping.items()):
            prepared = temporary_root / target_asset_paths[ordinal].name
            _fit_rotation_donor(
                donor_path=donor_dir / f"{donor_id}.jpg",
                reference_path=target_asset_paths[ordinal],
                destination_path=prepared,
            )
            prepared_paths[ordinal] = prepared
        target_set = set(target_asset_paths.values())
        _assert_direct_byte_unique(
            [path for path in asset_paths if path not in target_set]
            + list(prepared_paths.values())
        )

        quarantine_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_path, quarantine_root / "original_manifest.json")
        case_paths = sorted(
            {
                suite_root / str(episode["canonical_case_path"])
                for episodes in target_episodes.values()
                for episode in episodes
            }
        )
        for case_path in case_paths:
            destination = quarantine_root / "original_case_meta" / case_path.relative_to(
                suite_root
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(case_path, destination)

        for ordinal in sorted(mapping):
            original = target_asset_paths[ordinal]
            backup = quarantine_root / "original_assets" / original.relative_to(suite_root)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(original), str(backup))
            shutil.move(str(prepared_paths[ordinal]), str(original))

    moved = Counter()
    raw_root = suite_root / "raw_traces"
    for episodes in target_episodes.values():
        for episode in episodes:
            variant = str(episode["variant"])
            episode_id = str(episode["episode_id"])
            pair_id = str(episode["pair_id"])
            for bucket in ("records", "audits", "failures"):
                path = raw_root / bucket / variant / f"{episode_id}.json"
                if _move_to_quarantine(
                    source=path, source_root=suite_root, quarantine_root=quarantine_root
                ):
                    moved[bucket] += 1
            screenshots = (
                raw_root
                / "screenshots"
                / variant
                / "exploration_depth"
                / "rotation"
                / pair_id
            )
            if _move_to_quarantine(
                source=screenshots,
                source_root=suite_root,
                quarantine_root=quarantine_root,
            ):
                moved["screenshot_episode_dirs"] += 1
    for variant in ("rotation_inner", "rotation_outer"):
        for name in ("train.jsonl", "audit.jsonl", "summary.json"):
            path = raw_root / variant / name
            if _move_to_quarantine(
                source=path, source_root=suite_root, quarantine_root=quarantine_root
            ):
                moved["aggregate_files"] += 1

    provenance_by_ordinal = {}
    for ordinal, donor_id in sorted(mapping.items()):
        provenance = _rotation_donor_provenance(
            donor_id=donor_id, donor_record=donor_records[donor_id]
        )
        provenance_by_ordinal[ordinal] = provenance
        for episode in target_episodes[ordinal]:
            shared = episode["shared_scene_config"]
            shared["source_background_image_url"] = provenance[
                "source_background_image_url"
            ]
            shared["background_provenance"] = copy.deepcopy(provenance)
        for case_path in {
            suite_root / str(episode["canonical_case_path"])
            for episode in target_episodes[ordinal]
        }:
            shared = _read_json(case_path)
            shared["source_background_image_url"] = provenance[
                "source_background_image_url"
            ]
            shared["background_provenance"] = copy.deepcopy(provenance)
            _write_json_atomic(case_path, shared)

    _assert_direct_byte_unique(
        sorted(path for path in assets_root.iterdir() if path.is_file())
    )
    repaired_at = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "exploration_depth_rotation_donor_repair_v1",
        "status": "assets_repaired_traces_quarantined_pending_smoke",
        "suite_root": str(suite_root),
        "quarantine_root": str(quarantine_root),
        "repaired_at_utc": repaired_at,
        "donor_count": len(mapping),
        "target_episode_count": sum(len(value) for value in target_episodes.values()),
        "rotation_asset_count": len(asset_paths),
        "rotation_asset_direct_byte_unique_count": len(asset_paths),
        "comparison_method": "direct file bytes via filecmp.cmp(..., shallow=False); no file hash",
        "moved_artifacts": dict(sorted(moved.items())),
        "donors": {
            f"{ordinal:04d}": provenance_by_ordinal[ordinal]
            for ordinal in sorted(mapping)
        },
    }
    previous_repair = manifest.get("rotation_donor_repair")
    combined_mapping = (
        dict(previous_repair.get("mapping", {}))
        if isinstance(previous_repair, dict)
        else {}
    )
    combined_mapping.update(
        {
            f"{ordinal:04d}": donor_id
            for ordinal, donor_id in sorted(mapping.items())
        }
    )
    manifest["rotation_donor_repair"] = {
        "schema": report["schema"],
        "status": report["status"],
        "repaired_at_utc": repaired_at,
        "donor_count": len(combined_mapping),
        "mapping": dict(sorted(combined_mapping.items())),
        "quarantine_root": str(quarantine_root),
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(suite_root / "rotation_donor_repair.json", report)
    _write_json_atomic(quarantine_root / "repair_report.json", report)
    return report


def _drag_episodes(
    *, output_root: Path, source_root: Path, count: int
) -> list[dict[str, Any]]:
    source_paths = sorted(source_root.glob("*/meta.json"))
    if len(source_paths) < count:
        raise ValueError(f"drag training source has {len(source_paths)}, need {count}")
    sources = [_read_json(path) for path in source_paths[:count]]
    offscreen_target = count // 3
    offscreen_centers: dict[int, list[float]] = {}
    for index, source in enumerate(sources):
        if len(offscreen_centers) >= offscreen_target:
            break
        viewport = tuple(int(value) for value in source.get("viewport", DEFAULT_VIEWPORT))
        # Leave enough camera travel for every source piece to reach the fixed
        # center reticle.  The old 2200x1600 world made edge pieces visible but
        # geometrically ungrabbable after the view hit its clamp.
        base_view = (1300.0, 900.0)
        world_size = (2600, 1800)
        piece_world = _screen_to_world(
            source["piece_start_screen_xy"],
            view_center=base_view,
            viewport=viewport,
            zoom=1.0,
        )
        target_world = _screen_to_world(
            source["slot_center_screen_xy"],
            view_center=base_view,
            viewport=viewport,
            zoom=1.0,
        )
        center = _offscreen_first_person_view_center(
            piece_world_xy=piece_world,
            target_world_xy=target_world,
            viewport=viewport,
            world_size=world_size,
            zoom=1.0,
            piece_radius=float(source["piece"]["radius_px"]),
            target_shape=str(source["piece"]["shape"]),
            slot_radius_scale=float(source["slot_radius_scale"]),
        )
        if center is not None:
            offscreen_centers[index] = center
    if len(offscreen_centers) != offscreen_target:
        raise ValueError(
            f"could only construct {len(offscreen_centers)} offscreen drag cases, "
            f"need {offscreen_target}"
        )

    episodes: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        pair_id = _pair_id("drag", index)
        viewport = tuple(int(value) for value in source.get("viewport", DEFAULT_VIEWPORT))
        base_view = (1300.0, 900.0)
        world_size = (2600, 1800)
        band, sensitivity = _sensitivity(index)
        piece_screen = [float(value) for value in source["piece_start_screen_xy"]]
        target_screen = [float(value) for value in source["slot_center_screen_xy"]]
        piece_world = _screen_to_world(
            piece_screen, view_center=base_view, viewport=viewport, zoom=1.0
        )
        target_world = _screen_to_world(
            target_screen, view_center=base_view, viewport=viewport, zoom=1.0
        )
        slot_options: list[dict[str, Any]] = []
        for option in source["slot_options"]:
            item = copy.deepcopy(option)
            screen_xy = [float(value) for value in option["center_screen_xy"]]
            item["center_screen_xy"] = screen_xy
            item["center_world_xy"] = _screen_to_world(
                screen_xy, view_center=base_view, viewport=viewport, zoom=1.0
            )
            slot_options.append(item)
        offscreen = index in offscreen_centers
        reset_group = (
            "egocentric_target_offscreen"
            if offscreen
            else "shared_layout_different_interaction_marker"
        )
        instruction = "Place the solid colored shape into the matching gray outline."
        canonical = {
            "episode_id": pair_id,
            "paired_scene_schema": "paired_drag_scene_train_v1",
            "source_case_id": source["episode_id"],
            "viewport": list(viewport),
            "world_size": list(world_size),
            "initial_cursor_screen_xy": [viewport[0] / 2.0, viewport[1] / 2.0],
            "initial_view_center_xy": list(base_view),
            "piece_start_screen_xy": piece_screen,
            "piece_start_world_xy": piece_world,
            "slot_center_screen_xy": target_screen,
            "slot_center_world_xy": target_world,
            "slot_options": slot_options,
            "piece": copy.deepcopy(source["piece"]),
            "slot_radius_scale": source["slot_radius_scale"],
            "tolerance_px": source["tolerance_px"],
            "instruction": instruction,
            "layout_family": source.get("layout_family"),
            "distribution": "iid",
            "source_split": "train",
            "sensitivity": sensitivity,
            "sensitivity_band": band,
            "view_zoom": 1.0,
            "movement_direction_xy": list(FPS_DIRECTION_XY),
            "reset_pairing_group": reset_group,
        }
        _write_json(output_root / "cases/drag" / pair_id / "meta.json", canonical)
        shared = {
            "scene_schema": canonical["paired_scene_schema"],
            "background": "paired_drag_grid_v1",
            "piece": copy.deepcopy(canonical["piece"]),
            "piece_start_screen_xy": piece_screen,
            "piece_start_world_xy": piece_world,
            "slot_center_screen_xy": target_screen,
            "slot_center_world_xy": target_world,
            "slot_options": copy.deepcopy(slot_options),
            "initial_cursor_or_reticle_xy": canonical["initial_cursor_screen_xy"],
            "initial_view_center_xy": canonical["initial_view_center_xy"],
            "world_size": canonical["world_size"],
            "slot_radius_scale": canonical["slot_radius_scale"],
            "reset_pairing_group": reset_group,
            "hidden_dynamics": {
                "sensitivity": sensitivity,
                "sensitivity_band": band,
                "view_zoom": 1.0,
                "direction_xy": list(FPS_DIRECTION_XY),
            },
        }
        common = {"hidden_dynamics": shared["hidden_dynamics"], "minimum_steps_gate": None}
        first_environment: dict[str, Any] = {
            **common,
            "responsive_coordinate_system": "world_through_hidden_view_mapping",
            "interaction_marker": "red_dot",
            "reset_frame_contract": reset_group,
        }
        if offscreen:
            first_environment["initial_view_center_xy"] = offscreen_centers[index]
        for variant, level, environment, evaluator in (
            (
                "drag_third_person",
                "L0",
                {
                    **common,
                    "responsive_coordinate_system": "screen",
                    "interaction_marker": "mouse_icon",
                    "reset_frame_contract": "third_person_mouse_icon_visible_reference",
                },
                {"type": "final_screen_geometry_match", "tolerance_px": source["tolerance_px"]},
            ),
            (
                "drag_first_person",
                "L2",
                first_environment,
                {"type": "final_world_geometry_match", "tolerance_px": source["tolerance_px"]},
            ),
        ):
            episodes.append(
                _common_episode(
                    pair_id=pair_id,
                    family="drag",
                    variant=variant,
                    exploration_level=level,
                    index=index,
                    instruction=instruction,
                    viewport=viewport,
                    shared=shared,
                    environment=environment,
                    evaluator=evaluator,
                    canonical_case_path=f"cases/drag/{pair_id}/meta.json",
                )
            )
    return episodes


def validate_training_manifest(manifest: dict[str, Any], *, pair_count: int) -> dict[str, Any]:
    episodes = list(manifest["episodes"])
    counts = Counter(str(episode["variant"]) for episode in episodes)
    expected = {contract.key for contract in EXPLORATION_BENCHMARK_VARIANTS}
    if set(counts) != expected or any(value != pair_count for value in counts.values()):
        raise ValueError(f"training variant counts are invalid: {dict(counts)}")
    for family, variants in {
        "ten_choice": ("ten_choice_third_person", "ten_choice_first_person"),
        "rotation": ("rotation_inner", "rotation_outer"),
        "drag": ("drag_third_person", "drag_first_person"),
    }.items():
        pair_sets = [
            {episode["pair_id"] for episode in episodes if episode["variant"] == variant}
            for variant in variants
        ]
        if pair_sets[0] != pair_sets[1] or len(pair_sets[0]) != pair_count:
            raise ValueError(f"{family} training variants are not paired")
    policy_leaks = []
    forbidden = ("sensitivity", "target_index", "target_xy", "variant", "oracle")
    for episode in episodes:
        allowlist = episode["policy_observation_allowlist"]
        if any(fragment in str(key).lower() for key in allowlist for fragment in forbidden):
            policy_leaks.append(episode["episode_id"])
    if policy_leaks:
        raise ValueError(f"policy allowlist leaks hidden fields: {policy_leaks[:3]}")
    drag_offscreen = sum(
        episode["variant"] == "drag_first_person"
        and episode["environment_config"]["reset_frame_contract"]
        == "egocentric_target_offscreen"
        for episode in episodes
    )
    sensitivity_counts = {
        variant: Counter(
            episode["shared_scene_config"]["hidden_dynamics"]["sensitivity_band"]
            for episode in episodes
            if episode["variant"] == variant
        )
        for variant in ("ten_choice_first_person", "drag_first_person")
    }
    return {
        "status": "passed",
        "pair_count_per_family": pair_count,
        "total_episode_count": len(episodes),
        "variant_counts": dict(sorted(counts.items())),
        "drag_first_person_offscreen_count": drag_offscreen,
        "drag_first_person_shared_reset_count": pair_count - drag_offscreen,
        "first_person_sensitivity_bands": {
            key: dict(sorted(value.items())) for key, value in sensitivity_counts.items()
        },
        "policy_allowlist_leak_count": len(policy_leaks),
    }


def build_training_suite(
    *,
    output_root: Path,
    pair_count: int = DEFAULT_PAIR_COUNT,
    sources: TrainingSourceRoots | None = None,
) -> dict[str, Any]:
    if pair_count < 1 or pair_count > DEFAULT_PAIR_COUNT:
        raise ValueError(f"pair_count must be in [1, {DEFAULT_PAIR_COUNT}]")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty training directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    roots = sources or TrainingSourceRoots.defaults()
    episodes = [
        *_ten_choice_episodes(
            output_root=output_root, source_root=roots.ten_choice, count=pair_count
        ),
        *_rotation_episodes(
            output_root=output_root,
            records_path=roots.rotation_records,
            metadata_path=roots.rotation_background_metadata,
            donor_dir=roots.rotation_donor_dir,
            donor_metadata_path=roots.rotation_donor_metadata,
            count=pair_count,
        ),
        *_drag_episodes(
            output_root=output_root,
            source_root=roots.third_person_drag,
            count=pair_count,
        ),
    ]
    manifest = {
        "schema": "gui_captcha_exploration_depth_manifest_v1",
        "suite_id": TRAINING_SUITE_ID,
        "dataset_role": "train_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": False,
        "pair_count_per_family": pair_count,
        "case_count_per_variant": pair_count,
        "total_episode_count": len(episodes),
        "trajectory_policy": "causal_closed_loop_rules_v1",
        "teacher_policy": "action_conditioned_latest3_images_latest2_responses",
        "episodes": sorted(
            episodes, key=lambda episode: (episode["variant"], episode["pair_id"])
        ),
    }
    manifest["validation"] = validate_training_manifest(manifest, pair_count=pair_count)
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paired six-variant exploration-depth training manifest."
    )
    parser.add_argument("--output-root", type=Path, default=default_training_root())
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--repair-existing-root", type=Path, default=None)
    parser.add_argument("--quarantine-root", type=Path, default=None)
    parser.add_argument(
        "--rotation-donor-dir", type=Path, default=default_rotation_donor_dir()
    )
    parser.add_argument(
        "--rotation-donor-metadata",
        type=Path,
        default=default_rotation_donor_metadata(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repair_existing_root is not None:
        if args.quarantine_root is None:
            raise ValueError("--repair-existing-root requires --quarantine-root")
        report = repair_rotation_donor_assets(
            suite_root=args.repair_existing_root,
            donor_dir=args.rotation_donor_dir,
            donor_metadata_path=args.rotation_donor_metadata,
            quarantine_root=args.quarantine_root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    manifest = build_training_suite(
        output_root=args.output_root,
        pair_count=args.pair_count,
    )
    print(json.dumps(manifest["validation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
