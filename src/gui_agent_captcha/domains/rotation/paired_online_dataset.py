from __future__ import annotations

import copy
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps

from .online_dataset import (
    MINIMAL_DATASET_PROMPT,
    ONLINE_REWARD_PLACEHOLDER,
    find_forbidden_fields,
)
from ...integrations.storage import DEFAULT_STORAGE_ROOT

REPO_ROOT = Path(__file__).resolve().parents[4]

SUITE_ID = "rotation_paired_rl_160step_2560_nothink_20260830"
POOL_NAME = "rotation-train-2560-openimages-met-20260830"
HELDOUT_POOL_NAME = "openimages-rotation-test-150-20260514"
DATA_SOURCE = "rotation_paired_browser_online_grpo_160step_nothink_v1"
AGENT_NAME = "paired_rotation_no_think_browser_online_v1"
PROMPT_CONTRACT = (
    "action_ablation_six_action_latest3_images_all_exact_assistant_responses_v1"
)
RESPONSE_CONTRACT = "empty_think_tag_then_exact_action_v1"

TRAINING_STEPS = 160
PROMPT_GROUPS_PER_STEP = 16
ROLLOUTS_PER_PROMPT = 5
TRAIN_PAIR_COUNT = TRAINING_STEPS * PROMPT_GROUPS_PER_STEP
HELDOUT_PAIR_COUNT = 150
VIEWPORT = (1280, 720)
MAX_STEPS = 12
FORMAT_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
    "click",
)
EXECUTABLE_ACTION_KINDS = FORMAT_ACTION_KINDS

DEFAULT_POOL_ROOT = (
    DEFAULT_STORAGE_ROOT
    / "artifacts/background_pools/rotation_rl_160step_2560_20260830"
)
DEFAULT_FORMAL_SUITE_ROOT = (
    DEFAULT_STORAGE_ROOT
    / "data/formal_benchmarks/learn_from_move"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "data/training" / SUITE_ID

_TRAIN_SEED_BASES = {
    "inner": 40_000_000,
    "outer": 50_000_000,
    "mixed": 60_000_000,
}
_VAL_SEED_BASES = {
    "inner": 70_000_000,
    "outer": 80_000_000,
    "mixed": 90_000_000,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("writing paired rotation RL data requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)


def _ensure_directory_symlink(path: Path, target: Path) -> None:
    if not target.is_dir():
        raise FileNotFoundError(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target.resolve(), target_is_directory=True)


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _canvas_geometry(image_path: Path) -> tuple[int, int, int]:
    with Image.open(image_path) as raw:
        width, height = ImageOps.exif_transpose(raw).size
    if width <= 0 or height <= 0:
        raise ValueError(f"background has invalid dimensions: {image_path}")
    scale = min(1.0, 760.0 / width, 508.0 / height)
    canvas_width = max(1, _js_round(width * scale))
    canvas_height = max(1, _js_round(height * scale))
    desired_radius = _js_round(min(canvas_width, canvas_height) * 0.18)
    max_radius = min(canvas_width // 2 - 18, canvas_height // 2 - 18)
    if max_radius < 24:
        raise ValueError(
            f"background aspect ratio leaves no valid rotation circle: {image_path}"
        )
    return canvas_width, canvas_height, min(max(desired_radius, 24), max_radius)


def _seeded_unit_values(seed: int, count: int) -> list[float]:
    state = seed & 0xFFFFFFFF
    values: list[float] = []
    for _ in range(count):
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        values.append(state / 4_294_967_296)
    return values


def _rotation_challenge(*, image_path: Path, seed: int) -> dict[str, Any]:
    canvas_width, canvas_height, radius = _canvas_geometry(image_path)
    target_unit, start_unit, span_unit, sensitivity_unit, direction_unit = (
        _seeded_unit_values(seed, 5)
    )
    target_slider = _js_round(18.0 + 64.0 * target_unit)
    start_slider = 0 if start_unit < 0.5 else 100
    rotation_span = (0.5 + 0.45 * span_unit) * 360.0
    sensitivity = 0.65 + 0.7 * sensitivity_unit
    direction = -1 if direction_unit < 0.5 else 1
    degrees_per_unit = rotation_span / 100.0
    start_rotation = (
        (start_slider - target_slider)
        * degrees_per_unit
        * sensitivity
        * direction
    ) % 360.0
    center = {"x": canvas_width // 2, "y": canvas_height // 2}
    return {
        "circleCenter": center,
        "circleRadius": radius,
        "safeBounds": {
            "minX": 18 + radius,
            "maxX": canvas_width - 18 - radius,
            "minY": 18 + radius,
            "maxY": canvas_height - 18 - radius,
        },
        "sliderMinValue": 0,
        "sliderMaxValue": 100,
        "startSliderValue": start_slider,
        "targetSliderValue": target_slider,
        "rotationSpanDeg": rotation_span,
        "sensitivityScale": sensitivity,
        "rotationDirection": direction,
        "degreesPerSliderUnit": degrees_per_unit,
        "startRotationDeg": start_rotation,
        "targetRotationDeg": 0,
        "rotationToleranceDeg": 5.0,
        "pairedRelativeRotation": True,
        "canonicalInitialRelativeDeg": start_rotation,
    }


def _episode(
    *,
    pair_id: str,
    variant: str,
    case_seed: int,
    shared: Mapping[str, Any],
    split: str,
    canonical_case_path: str,
) -> dict[str, Any]:
    region = "outer" if variant == "rotation_outer" else "center"
    return {
        "suite_id": SUITE_ID,
        "episode_id": f"{pair_id}--{variant.replace('_', '-')}",
        "pair_id": pair_id,
        "family": "rotation",
        "variant": variant,
        "exploration_level": "L2",
        "case_seed": case_seed,
        "split": split,
        "viewport": list(VIEWPORT),
        "instruction": "Drag the slider to complete verification.",
        "shared_scene_config": copy.deepcopy(dict(shared)),
        "environment_config": {
            "action_response": f"rotate only the {region} visual region",
            "interaction_marker": "mouse_icon",
            "responsive_region": region,
            "hidden_mapping": {
                "degrees_per_slider_unit": shared["degrees_per_slider_unit"],
                "sensitivity": shared["sensitivity"],
                "rotation_direction": shared["rotation_direction"],
                "target_slider_value": shared["target_slider_value"],
            },
            "minimum_steps_gate": None,
            "reset_frame_contract": "identical_free_mouse_scene",
            "static_replay_renderer": "scripts/interaction/serve_captcha_static_replay.py",
        },
        "success_evaluator": {
            "type": "relative_angular_error_on_release",
            "responsive_region": region,
            "tolerance_deg": shared["tolerance_deg"],
        },
        "canonical_case_path": canonical_case_path,
        "policy_observation_allowlist": [
            "suite_id",
            "pair_id",
            "family",
            "viewport",
        ],
    }


def _shared_scene(
    *,
    record: Mapping[str, Any],
    image_path: Path,
    background_asset: str,
    case_seed: int,
) -> dict[str, Any]:
    challenge = _rotation_challenge(image_path=image_path, seed=case_seed)
    return {
        "scene_schema": "paired_relative_rotation_train_v1",
        "source_case_id": str(record["id"]),
        "source_background_image_url": str(record["localPath"]),
        "background_asset": background_asset,
        "circle_center": copy.deepcopy(challenge["circleCenter"]),
        "circle_radius": challenge["circleRadius"],
        "initial_relative_angle_deg": challenge["canonicalInitialRelativeDeg"],
        "target_relative_angle_deg": challenge["targetRotationDeg"],
        "slider_geometry": {
            "x": 995.328125,
            "y": 498.65625,
            "width": 218.0,
            "height": 20.0,
        },
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
        "background_provenance": {
            "pool_name": POOL_NAME,
            "source": str(record.get("source") or ""),
            "source_id": str(record["id"]),
            "license": str(record.get("license") or ""),
        },
    }


def _load_train_records(pool_root: Path) -> list[dict[str, Any]]:
    summary = _read_json(pool_root / "summary.json")
    if summary.get("status") != "passed" or summary.get("records") != TRAIN_PAIR_COUNT:
        raise ValueError("the shared background pool has not passed its 2560-image audit")
    overlaps = summary.get("identity_overlap_with_heldout")
    if not isinstance(overlaps, dict) or any(int(value) != 0 for value in overlaps.values()):
        raise ValueError("the shared background pool overlaps the held-out pool")
    records = _read_json(pool_root / "metadata.json")
    if not isinstance(records, list) or len(records) != TRAIN_PAIR_COUNT:
        raise ValueError(f"expected {TRAIN_PAIR_COUNT} background metadata rows")
    ids: set[str] = set()
    names: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise ValueError("background metadata rows must be objects")
        record = dict(raw_record)
        record_id = str(record.get("id") or "")
        served_path = Path(str(record.get("servedPath") or ""))
        if not record_id or not served_path.is_file():
            raise FileNotFoundError(f"invalid background record {record_id!r}: {served_path}")
        if record_id in ids or served_path.name in names:
            raise ValueError(f"duplicate background identity or filename: {record_id}")
        ids.add(record_id)
        names.add(served_path.name)
        normalized.append(record)
    return normalized


def _build_train_episodes(
    *,
    root: Path,
    pool_root: Path,
) -> list[dict[str, Any]]:
    records = _load_train_records(pool_root)
    _ensure_directory_symlink(root / "assets/rotation/train", pool_root / "images")
    episodes: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        pair_id = f"rotation-rl160-{index:04d}"
        case_seed = 202608300000 + index
        image_path = Path(str(record["servedPath"]))
        background_asset = f"assets/rotation/train/{image_path.name}"
        shared = _shared_scene(
            record=record,
            image_path=image_path,
            background_asset=background_asset,
            case_seed=case_seed,
        )
        case_path = f"cases/rotation/train/{pair_id}/meta.json"
        _write_json(root / case_path, shared)
        for variant in ("rotation_inner", "rotation_outer"):
            episodes.append(
                _episode(
                    pair_id=pair_id,
                    variant=variant,
                    case_seed=case_seed,
                    shared=shared,
                    split="train",
                    canonical_case_path=case_path,
                )
            )
    return episodes


def _build_heldout_episodes(
    *,
    root: Path,
    formal_suite_root: Path,
) -> list[dict[str, Any]]:
    manifest = _read_json(formal_suite_root / "manifest.json")
    source_episodes = [
        dict(episode)
        for episode in manifest.get("episodes", [])
        if episode.get("variant") in {"rotation_inner", "rotation_outer"}
    ]
    counts = Counter(str(episode["variant"]) for episode in source_episodes)
    if counts != {"rotation_inner": HELDOUT_PAIR_COUNT, "rotation_outer": HELDOUT_PAIR_COUNT}:
        raise ValueError(f"formal held-out rotation counts are invalid: {dict(counts)}")
    _ensure_directory_symlink(
        root / "assets/rotation/heldout",
        formal_suite_root / "assets/rotation",
    )
    episodes: list[dict[str, Any]] = []
    for source in source_episodes:
        episode = copy.deepcopy(source)
        episode["suite_id"] = SUITE_ID
        episode["split"] = "heldout"
        shared = episode["shared_scene_config"]
        source_asset = Path(str(shared["background_asset"]))
        shared["background_asset"] = f"assets/rotation/heldout/{source_asset.name}"
        pair_id = str(episode["pair_id"])
        case_path = f"cases/rotation/heldout/{pair_id}/meta.json"
        episode["canonical_case_path"] = case_path
        _write_json(root / case_path, shared)
        episodes.append(episode)
    return episodes


def _task_config(
    *,
    manifest_path: Path,
    variant: str,
    episode_id: str,
    background_pool: str,
    suite_id: str = SUITE_ID,
) -> dict[str, Any]:
    return {
        "background_pool": background_pool,
        "suite_id": suite_id,
        "manifest_path": str(manifest_path.resolve()),
        "benchmark_variant": variant,
        "episode_id": episode_id,
        "viewport": list(VIEWPORT),
        "max_steps": MAX_STEPS,
        "coordinate_format": "qwen_relative_0_1000",
        "format_action_kinds": list(FORMAT_ACTION_KINDS),
        "executable_action_kinds": list(EXECUTABLE_ACTION_KINDS),
    }


def _row(
    *,
    episode: Mapping[str, Any],
    seed: int,
    split: str,
    manifest_path: Path,
    background_pool: str,
) -> dict[str, Any]:
    row = {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": MINIMAL_DATASET_PROMPT}],
        "ability": "rotation_captcha",
        "reward_model": {
            "style": "online_browser",
            "ground_truth": ONLINE_REWARD_PLACEHOLDER,
        },
        "agent_name": AGENT_NAME,
        "task_type": "rotation_captcha",
        "seed": seed,
        "task_config": _task_config(
            manifest_path=manifest_path,
            variant=str(episode["variant"]),
            episode_id=str(episode["episode_id"]),
            background_pool=background_pool,
            suite_id=str(episode.get("suite_id") or SUITE_ID),
        ),
        "extra_info": {
            "index": seed,
            "split": split,
            "pair_id": str(episode["pair_id"]),
        },
    }
    assert_paired_task_only_row(row)
    return row


def assert_paired_task_only_row(
    row: Mapping[str, Any],
    *,
    data_source: str = DATA_SOURCE,
    agent_name: str = AGENT_NAME,
) -> None:
    forbidden = find_forbidden_fields(row)
    if forbidden:
        raise ValueError("paired online row contains forbidden fields: " + ", ".join(forbidden))
    expected_keys = {
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "agent_name",
        "task_type",
        "seed",
        "task_config",
        "extra_info",
    }
    if set(row) != expected_keys:
        raise ValueError("paired online row has an unexpected top-level schema")
    if row["data_source"] != data_source or row["agent_name"] != agent_name:
        raise ValueError("paired online row selects the wrong data source or AgentLoop")
    if row["task_type"] != "rotation_captcha" or row["ability"] != "rotation_captcha":
        raise ValueError("paired online rows only support rotation_captcha")
    if row["prompt"] != [{"role": "user", "content": MINIMAL_DATASET_PROMPT}]:
        raise ValueError("paired online prompt must remain task-only")
    if row["reward_model"] != {
        "style": "online_browser",
        "ground_truth": ONLINE_REWARD_PLACEHOLDER,
    }:
        raise ValueError("paired online reward must come from the live AgentLoop")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("paired online seed must be an integer")
    extra_info = row["extra_info"]
    if not isinstance(extra_info, Mapping) or set(extra_info) != {
        "index",
        "split",
        "pair_id",
    }:
        raise ValueError("paired online extra_info schema is invalid")
    if extra_info["index"] != row["seed"] or extra_info["split"] not in {
        "train",
        "heldout",
    }:
        raise ValueError("paired online row index or split is invalid")
    config = row["task_config"]
    required_config = {
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
    if not isinstance(config, Mapping) or set(config) != required_config:
        raise ValueError("paired online task_config schema is invalid")
    if not isinstance(config["suite_id"], str) or not config["suite_id"].strip():
        raise ValueError("paired online task_config requires a nonempty suite_id")
    if (
        not isinstance(config["background_pool"], str)
        or not config["background_pool"].strip()
        or "/" in config["background_pool"]
        or ".." in config["background_pool"]
    ):
        raise ValueError("paired online task_config background_pool is invalid")
    if config["benchmark_variant"] not in {"rotation_inner", "rotation_outer"}:
        raise ValueError("paired online task_config selects an unknown variant")
    if list(config["viewport"]) != list(VIEWPORT) or config["max_steps"] != MAX_STEPS:
        raise ValueError("paired online viewport or action budget is invalid")
    if config["coordinate_format"] != "qwen_relative_0_1000":
        raise ValueError("paired online coordinate format is invalid")
    if tuple(config["format_action_kinds"]) != FORMAT_ACTION_KINDS:
        raise ValueError("paired online format grammar is invalid")
    if tuple(config["executable_action_kinds"]) != EXECUTABLE_ACTION_KINDS:
        raise ValueError("paired online executable action grammar is invalid")


def _rows_for_profile(
    *,
    episodes: Sequence[Mapping[str, Any]],
    profile: str,
    split: str,
    manifest_path: Path,
    background_pool: str | None = None,
) -> list[dict[str, Any]]:
    if profile not in {"inner", "outer", "mixed"}:
        raise ValueError(f"unknown paired rotation profile: {profile}")
    by_pair: dict[str, dict[str, Mapping[str, Any]]] = {}
    for episode in episodes:
        pair_id = str(episode["pair_id"])
        by_pair.setdefault(pair_id, {})[str(episode["variant"])] = episode
    selected: list[Mapping[str, Any]] = []
    for index, pair_id in enumerate(sorted(by_pair)):
        pair = by_pair[pair_id]
        if set(pair) != {"rotation_inner", "rotation_outer"}:
            raise ValueError(f"rotation pair is incomplete: {pair_id}")
        if profile == "inner":
            selected.append(pair["rotation_inner"])
        elif profile == "outer":
            selected.append(pair["rotation_outer"])
        else:
            selected.append(pair["rotation_inner" if index % 2 == 0 else "rotation_outer"])
    seed_base = (
        _TRAIN_SEED_BASES[profile] if split == "train" else _VAL_SEED_BASES[profile]
    )
    if background_pool is None:
        pool_names = {
            str(
                episode.get("shared_scene_config", {})
                .get("background_provenance", {})
                .get("pool_name")
                or ""
            )
            for episode in selected
        }
        pool_names.discard("")
        if len(pool_names) > 1:
            raise ValueError(f"paired rotation rows span multiple background pools: {pool_names}")
        background_pool = (
            next(iter(pool_names))
            if pool_names
            else (POOL_NAME if split == "train" else HELDOUT_POOL_NAME)
        )
    return [
        _row(
            episode=episode,
            seed=seed_base + index,
            split=split,
            manifest_path=manifest_path,
            background_pool=background_pool,
        )
        for index, episode in enumerate(selected)
    ]


def validate_paired_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    episodes = [dict(episode) for episode in manifest.get("episodes", [])]
    counts = Counter((str(e["split"]), str(e["variant"])) for e in episodes)
    expected = {
        ("train", "rotation_inner"): TRAIN_PAIR_COUNT,
        ("train", "rotation_outer"): TRAIN_PAIR_COUNT,
        ("heldout", "rotation_inner"): HELDOUT_PAIR_COUNT,
        ("heldout", "rotation_outer"): HELDOUT_PAIR_COUNT,
    }
    if counts != expected:
        raise ValueError(f"paired manifest split/variant counts are invalid: {dict(counts)}")
    by_key = {(str(e["pair_id"]), str(e["variant"])): e for e in episodes}
    pair_ids = sorted({str(e["pair_id"]) for e in episodes})
    for pair_id in pair_ids:
        inner = by_key.get((pair_id, "rotation_inner"))
        outer = by_key.get((pair_id, "rotation_outer"))
        if inner is None or outer is None:
            raise ValueError(f"paired manifest is missing one variant: {pair_id}")
        if inner["shared_scene_config"] != outer["shared_scene_config"]:
            raise ValueError(f"paired manifest scenes differ within pair: {pair_id}")
        if inner["case_seed"] != outer["case_seed"]:
            raise ValueError(f"paired manifest seeds differ within pair: {pair_id}")
    return {
        "status": "passed",
        "pair_count": len(pair_ids),
        "train_pair_count": TRAIN_PAIR_COUNT,
        "heldout_pair_count": HELDOUT_PAIR_COUNT,
        "episode_count": len(episodes),
        "split_variant_counts": {
            f"{split}:{variant}": count
            for (split, variant), count in sorted(counts.items())
        },
        "shared_scene_mismatch_count": 0,
        "case_seed_mismatch_count": 0,
    }


def build_paired_rotation_rl_suite(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pool_root: Path = DEFAULT_POOL_ROOT,
    formal_suite_root: Path = DEFAULT_FORMAL_SUITE_ROOT,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing paired suite: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.building-", dir=output_root.parent
    ) as temporary_value:
        temporary_root = Path(temporary_value)
        train_episodes = _build_train_episodes(
            root=temporary_root,
            pool_root=Path(pool_root).resolve(),
        )
        heldout_episodes = _build_heldout_episodes(
            root=temporary_root,
            formal_suite_root=Path(formal_suite_root).resolve(),
        )
        episodes = sorted(
            [*train_episodes, *heldout_episodes],
            key=lambda episode: (str(episode["split"]), str(episode["variant"]), str(episode["pair_id"])),
        )
        manifest: dict[str, Any] = {
            "schema": "gui_captcha_exploration_depth_manifest_v1",
            "suite_id": SUITE_ID,
            "dataset_role": "train_with_external_heldout",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen": False,
            "training_steps": TRAINING_STEPS,
            "prompt_groups_per_step": PROMPT_GROUPS_PER_STEP,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "train_pair_count": TRAIN_PAIR_COUNT,
            "heldout_pair_count": HELDOUT_PAIR_COUNT,
            "total_episode_count": len(episodes),
            "trajectory_policy": "causal_closed_loop_rules_v1",
            "episodes": episodes,
        }
        manifest["validation"] = validate_paired_manifest(manifest)
        _write_json(temporary_root / "manifest.json", manifest)

        datasets_root = temporary_root / "datasets"
        outputs: dict[str, str] = {}
        dataset_counts: dict[str, int] = {}
        train_source = [episode for episode in episodes if episode["split"] == "train"]
        heldout_source = [episode for episode in episodes if episode["split"] != "train"]
        for profile in ("inner", "outer", "mixed"):
            train_rows = _rows_for_profile(
                episodes=train_source,
                profile=profile,
                split="train",
                manifest_path=manifest_path,
            )
            val_rows = _rows_for_profile(
                episodes=heldout_source,
                profile=profile,
                split="heldout",
                manifest_path=manifest_path,
            )
            for split_name, rows in (("train", train_rows), ("val", val_rows)):
                name = f"{split_name}_{profile}.parquet"
                _write_parquet(datasets_root / name, rows)
                outputs[f"{split_name}_{profile}"] = str(output_root / "datasets" / name)
                dataset_counts[f"{split_name}_{profile}"] = len(rows)
        summary = {
            "status": "passed",
            "suite_id": SUITE_ID,
            "data_source": DATA_SOURCE,
            "agent_name": AGENT_NAME,
            "manifest_path": str(manifest_path),
            "output_root": str(output_root),
            "outputs": outputs,
            "counts": dataset_counts,
            "training_steps": TRAINING_STEPS,
            "prompt_groups_per_step": PROMPT_GROUPS_PER_STEP,
            "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
            "task_group_capacity_per_profile": TRAIN_PAIR_COUNT,
            "trajectory_capacity_per_profile": TRAIN_PAIR_COUNT * ROLLOUTS_PER_PROMPT,
            "shared_train_pair_count": TRAIN_PAIR_COUNT,
            "heldout_pair_count": HELDOUT_PAIR_COUNT,
            "train_asset_policy": "one shared symlinked pool; no image copies",
            "outer_specific_filter_required": False,
            "prompt_contract": PROMPT_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
            "enable_thinking": False,
            "prompt_family": "qwen3_vl_sft_closed_loop_context_v5_interleaved_history",
            "task_type": "rotation_captcha",
            "image_history_max": 3,
            "image_max_pixels": 921600,
            "system_prompt_present": False,
            "coordinate_format": "qwen3_relative_0_1000",
            "viewport": list(VIEWPORT),
            "max_steps": MAX_STEPS,
            "format_action_kinds": list(FORMAT_ACTION_KINDS),
            "executable_action_kinds": list(EXECUTABLE_ACTION_KINDS),
            "manifest_validation": manifest["validation"],
        }
        _write_json(datasets_root / "summary.json", summary)
        temporary_root.rename(output_root)
    return summary
