"""Fresh four-task RL scenes that are disjoint from SFT and formal evaluation."""

from __future__ import annotations

import copy
import json
import random
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...domains.rotation import paired_online_dataset as rotation_online
from ...domains.ten_choice.dataset import collect_icon_assets, write_episode
from ...domains.ten_choice.icon_assets import DEFAULT_ICON_DIR
from ...domains.third_person_drag.dataset_v2 import _generate_split
from ...integrations.storage import DEFAULT_STORAGE_ROOT
from . import training_manifest
from .first_person_online_dataset import (
    AGENT_NAME as FIRST_PERSON_AGENT_NAME,
    DATA_SOURCE as FIRST_PERSON_DATA_SOURCE,
    FORMAT_ACTION_KINDS as FIRST_PERSON_ACTION_KINDS,
    INTERACTION_MARKER,
    MAX_STEPS,
    MINIMAL_DATASET_PROMPT as FIRST_PERSON_DATASET_PROMPT,
    PROMPT_CONTRACT as FIRST_PERSON_PROMPT_CONTRACT,
    RESPONSE_CONTRACT as FIRST_PERSON_RESPONSE_CONTRACT,
    assert_first_person_task_only_row,
)

RELEASE_ID = "exploration_depth_rl_4env_sft_disjoint_v1"
IMPLEMENTATION_TAG = "6env-v2-rl-sft-disjoint-v1"
TASKS = (
    "rotation_inner",
    "rotation_outer",
    "ten_choice_first_person",
    "drag_first_person",
)
TRAIN_COUNT_PER_TASK = 2_560
VALIDATION_COUNT_PER_TASK = 150
VIEWPORT = (1280, 720)

HISTORICAL_STORAGE_ROOT = Path(
    str(Path(__file__).resolve().parents[4] / '')
)
DEFAULT_SFT_MANIFEST = (
    HISTORICAL_STORAGE_ROOT
    / "data/training/exploration_depth_train_6x4000_v3/manifest.json"
)
DEFAULT_TEST_MANIFEST = (
    HISTORICAL_STORAGE_ROOT
    / "data/formal_benchmarks/learn_from_move/manifest.json"
)
DEFAULT_ROTATION_POOL_ROOT = (
    DEFAULT_STORAGE_ROOT / "data/background_pools/rotation_rl_4env_sft_disjoint_1800_v1"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "data/training" / RELEASE_ID

_SOURCE_SEEDS = {
    ("train", "ten_choice"): 2026090501,
    ("validation", "ten_choice"): 2026090502,
    ("train", "drag"): 2026090503,
    ("validation", "drag"): 2026090504,
}
_CASE_SEED_BASES = {
    ("train", "ten_choice"): 202609051000,
    ("validation", "ten_choice"): 202609052000,
    ("train", "drag"): 202609053000,
    ("validation", "drag"): 202609054000,
    ("train", "rotation"): 202609055000,
    ("validation", "rotation"): 202609056000,
}
_ROW_SEED_BASES = {
    ("train", "rotation_inner"): 310_000_000,
    ("train", "rotation_outer"): 320_000_000,
    ("train", "ten_choice_first_person"): 330_000_000,
    ("train", "drag_first_person"): 340_000_000,
    ("validation", "rotation_inner"): 410_000_000,
    ("validation", "rotation_outer"): 420_000_000,
    ("validation", "ten_choice_first_person"): 430_000_000,
    ("validation", "drag_first_person"): 440_000_000,
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
        raise RuntimeError("writing four-task RL data requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)


def _replace_text(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_text(item, old, new) for item in value]
    if isinstance(value, dict):
        return {key: _replace_text(item, old, new) for key, item in value.items()}
    return value


def _retag_first_person_episode(
    *,
    episode: Mapping[str, Any],
    split_root: Path,
    split: str,
    family: str,
    index: int,
) -> dict[str, Any]:
    old_pair_id = str(episode["pair_id"])
    family_token = "ten" if family == "ten_choice" else "drag"
    new_pair_id = f"ed-rl4-{split}-{family_token}-{index + 1:04d}"
    old_case_path = split_root / str(episode["canonical_case_path"])
    new_case_dir = old_case_path.parent.with_name(new_pair_id)
    if new_case_dir.exists():
        raise FileExistsError(new_case_dir)
    old_case_path.parent.rename(new_case_dir)
    rewritten = _replace_text(copy.deepcopy(dict(episode)), old_pair_id, new_pair_id)
    rewritten["suite_id"] = RELEASE_ID
    rewritten["implementation_tag"] = IMPLEMENTATION_TAG
    rewritten["split"] = split
    rewritten["case_seed"] = _CASE_SEED_BASES[(split, family)] + index
    environment = rewritten["environment_config"]
    shared = rewritten["shared_scene_config"]
    environment["interaction_marker"] = INTERACTION_MARKER
    if family == "ten_choice":
        environment["reset_frame_contract"] = "shared_scene_with_first_person_mouse_icon"
    elif environment.get("reset_frame_contract") == "shared_layout_different_interaction_marker":
        environment["reset_frame_contract"] = "shared_layout_same_mouse_marker"
        shared["reset_pairing_group"] = "shared_layout_same_mouse_marker"

    case_path = split_root / str(rewritten["canonical_case_path"])
    case_meta = _replace_text(_read_json(case_path), old_pair_id, new_pair_id)
    if family == "drag" and isinstance(case_meta, dict):
        case_meta["episode_id"] = new_pair_id
        if case_meta.get("reset_pairing_group") == "shared_layout_different_interaction_marker":
            case_meta["reset_pairing_group"] = "shared_layout_same_mouse_marker"
    _write_json(case_path, case_meta)
    return rewritten


def _build_first_person_episodes(
    *,
    split_root: Path,
    split: str,
    count: int,
    icon_root: Path = DEFAULT_ICON_DIR,
) -> list[dict[str, Any]]:
    sources_root = split_root / "sources"
    ten_choice_source = sources_root / "ten_choice"
    drag_source = sources_root / "drag"
    icon_assets = collect_icon_assets(icon_root)
    ten_choice_rng = random.Random(_SOURCE_SEEDS[(split, "ten_choice")])
    ten_choice_source.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        write_episode(
            episode_id=f"rl4_{split}_hr_{index + 1:04d}",
            split=split,
            rng=ten_choice_rng,
            icon_assets=icon_assets,
            split_dir=ten_choice_source,
            viewport_width=VIEWPORT[0],
            viewport_height=VIEWPORT[1],
        )
    _generate_split(
        test_root=drag_source,
        split_name=f"rl_{split}",
        distribution="iid",
        count=count,
        seed=_SOURCE_SEEDS[(split, "drag")],
        training_trajectory=True,
    )

    ten_choice_pairs = training_manifest._ten_choice_episodes(
        output_root=split_root,
        source_root=ten_choice_source,
        count=count,
    )
    drag_pairs = training_manifest._drag_episodes(
        output_root=split_root,
        source_root=drag_source,
        count=count,
    )
    episodes: list[dict[str, Any]] = []
    for family, source in (
        ("ten_choice", ten_choice_pairs),
        ("drag", drag_pairs),
    ):
        variant = f"{family}_first_person" if family == "ten_choice" else "drag_first_person"
        selected = [episode for episode in source if episode["variant"] == variant]
        for index, episode in enumerate(selected):
            episodes.append(
                _retag_first_person_episode(
                    episode=episode,
                    split_root=split_root,
                    split=split,
                    family=family,
                    index=index,
                )
            )
    return episodes


def _rotation_records(pool_root: Path, *, split: str) -> list[dict[str, Any]]:
    summary = _read_json(pool_root / "summary.json")
    if summary.get("status") != "passed":
        raise ValueError("rotation SFT-disjoint background pool has not passed")
    for key in (
        "identity_overlap_with_sft",
        "identity_overlap_with_test",
    ):
        if any(int(value) != 0 for value in summary.get(key, {}).values()):
            raise ValueError(f"rotation background pool failed {key}")
    for key in (
        "direct_byte_overlap_with_sft_count",
        "direct_byte_overlap_with_test_count",
        "direct_internal_duplicate_count",
    ):
        if int(summary.get(key, -1)) != 0:
            raise ValueError(f"rotation background pool failed {key}")
    records = [
        dict(record)
        for record in _read_json(pool_root / "metadata.json")
        if record.get("split") == split
    ]
    if not records:
        raise ValueError(f"rotation pool contains no {split} backgrounds")
    return records


def _build_rotation_episodes(
    *,
    split_root: Path,
    pool_root: Path,
    split: str,
    count: int,
) -> list[dict[str, Any]]:
    records = _rotation_records(pool_root, split=split)
    pool_summary = _read_json(pool_root / "summary.json")
    pool_name = str(pool_summary["pool_name"])
    asset_link = split_root / pool_name
    asset_link.parent.mkdir(parents=True, exist_ok=True)
    asset_link.symlink_to((pool_root / "images").resolve(), target_is_directory=True)
    episodes: list[dict[str, Any]] = []
    for index in range(count):
        record = records[index % len(records)]
        pair_id = f"ed-rl4-{split}-rot-{index + 1:04d}"
        case_seed = _CASE_SEED_BASES[(split, "rotation")] + index
        image_path = Path(str(record["servedPath"]))
        shared = rotation_online._shared_scene(
            record=record,
            image_path=image_path,
            background_asset=f"{pool_name}/{image_path.name}",
            case_seed=case_seed,
        )
        shared["scene_schema"] = "paired_relative_rotation_rl_sft_disjoint_v1"
        shared["background_provenance"]["pool_name"] = pool_name
        case_path = f"cases/rotation/{pair_id}/meta.json"
        _write_json(split_root / case_path, shared)
        for variant in ("rotation_inner", "rotation_outer"):
            episode = rotation_online._episode(
                pair_id=pair_id,
                variant=variant,
                case_seed=case_seed,
                shared=shared,
                split=split,
                canonical_case_path=case_path,
            )
            episode["suite_id"] = RELEASE_ID
            episode["implementation_tag"] = IMPLEMENTATION_TAG
            episode["environment_config"]["gesture_contract"] = (
                "exactly_one_mouse_down_and_one_mouse_up"
            )
            episodes.append(episode)
    return episodes


def _manifest(
    *,
    split: str,
    episodes: Sequence[Mapping[str, Any]],
    expected_count: int,
) -> dict[str, Any]:
    counts = Counter(str(episode["variant"]) for episode in episodes)
    if counts != {task: expected_count for task in TASKS}:
        raise ValueError(f"four-task {split} counts are invalid: {dict(counts)}")
    return {
        "schema": "gui_captcha_exploration_depth_manifest_v1",
        "suite_id": RELEASE_ID,
        "implementation_tag": IMPLEMENTATION_TAG,
        "dataset_role": "rl_train_only" if split == "train" else "rl_validation_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": False,
        "case_count_per_variant": expected_count,
        "total_episode_count": len(episodes),
        "trajectory_policy": "online_rl_task_only_no_oracle_fields_v1",
        "episodes": sorted(
            (copy.deepcopy(dict(episode)) for episode in episodes),
            key=lambda episode: (str(episode["variant"]), str(episode["pair_id"])),
        ),
    }


def _semantic_identity(episode: Mapping[str, Any]) -> str:
    variant = str(episode["variant"])
    shared = episode["shared_scene_config"]
    if variant == "ten_choice_first_person":
        value = {
            key: shared[key]
            for key in (
                "labels",
                "icon_positions_xy",
                "target_object",
                "hidden_dynamics",
            )
        }
    elif variant == "drag_first_person":
        value = {
            key: shared[key]
            for key in (
                "piece",
                "piece_start_world_xy",
                "slot_center_world_xy",
                "slot_options",
                "hidden_dynamics",
                "initial_view_center_xy",
                "reset_pairing_group",
            )
        }
    else:
        value = {
            "source": shared.get("background_provenance", {}).get("source_id")
            or shared.get("source_background_image_url"),
            "challenge": shared["replay_challenge"],
        }
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _disjoint_report(
    *,
    train_episodes: Sequence[Mapping[str, Any]],
    validation_episodes: Sequence[Mapping[str, Any]],
    sft_manifest: Path,
    test_manifest: Path,
) -> dict[str, Any]:
    sft_episodes = _read_json(sft_manifest)["episodes"]
    formal_test_episodes = _read_json(test_manifest)["episodes"]
    report: dict[str, Any] = {}
    for task in TASKS:
        train = [episode for episode in train_episodes if episode["variant"] == task]
        validation = [
            episode for episode in validation_episodes if episode["variant"] == task
        ]

        def compare(
            candidates: Sequence[Mapping[str, Any]],
            references: Sequence[Mapping[str, Any]],
        ) -> dict[str, int]:
            selected = [episode for episode in references if episode.get("variant") == task]
            return {
                "episode_id_overlap": len(
                    {str(episode["episode_id"]) for episode in candidates}
                    & {str(episode["episode_id"]) for episode in selected}
                ),
                "pair_id_overlap": len(
                    {str(episode["pair_id"]) for episode in candidates}
                    & {str(episode["pair_id"]) for episode in selected}
                ),
                "semantic_scene_overlap": len(
                    {_semantic_identity(episode) for episode in candidates}
                    & {_semantic_identity(episode) for episode in selected}
                ),
            }

        report[task] = {
            "train_vs_sft": compare(train, sft_episodes),
            "train_vs_formal_test": compare(train, formal_test_episodes),
            "train_vs_validation": compare(train, validation),
            "validation_vs_sft": compare(validation, sft_episodes),
            "validation_vs_formal_test": compare(validation, formal_test_episodes),
        }
    if any(
        count
        for task_report in report.values()
        for reference_report in task_report.values()
        for count in reference_report.values()
    ):
        raise ValueError(f"four-task RL scene disjointness failed: {report}")
    return report


def _first_person_row(
    *,
    episode: Mapping[str, Any],
    manifest_path: Path,
    seed: int,
    split: str,
) -> dict[str, Any]:
    task = str(episode["variant"])
    return {
        "data_source": FIRST_PERSON_DATA_SOURCE,
        "prompt": [{"role": "user", "content": FIRST_PERSON_DATASET_PROMPT}],
        "ability": task,
        "reward_model": {
            "style": "online_environment",
            "ground_truth": rotation_online.ONLINE_REWARD_PLACEHOLDER,
        },
        "agent_name": FIRST_PERSON_AGENT_NAME,
        "task_type": task,
        "seed": seed,
        "task_config": {
            "suite_id": RELEASE_ID,
            "manifest_path": str(manifest_path.resolve()),
            "benchmark_variant": task,
            "episode_id": str(episode["episode_id"]),
            "viewport": list(VIEWPORT),
            "max_steps": MAX_STEPS,
            "coordinate_format": "qwen_relative_0_1000",
            "format_action_kinds": list(FIRST_PERSON_ACTION_KINDS),
            "executable_action_kinds": list(FIRST_PERSON_ACTION_KINDS),
            "prompt_contract": FIRST_PERSON_PROMPT_CONTRACT,
            "response_contract": FIRST_PERSON_RESPONSE_CONTRACT,
        },
        "extra_info": {
            "index": seed,
            "split": "train" if split == "train" else "heldout",
            "pair_id": str(episode["pair_id"]),
        },
    }


def _rows_by_task(
    *,
    episodes: Sequence[Mapping[str, Any]],
    manifest_path: Path,
    split: str,
    rotation_pool_name: str,
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    rotation_episodes = [
        episode for episode in episodes if str(episode["variant"]).startswith("rotation_")
    ]
    rows["rotation_inner"] = rotation_online._rows_for_profile(
        episodes=rotation_episodes,
        profile="inner",
        split="train" if split == "train" else "heldout",
        manifest_path=manifest_path,
        background_pool=rotation_pool_name,
    )
    rows["rotation_outer"] = rotation_online._rows_for_profile(
        episodes=rotation_episodes,
        profile="outer",
        split="train" if split == "train" else "heldout",
        manifest_path=manifest_path,
        background_pool=rotation_pool_name,
    )
    for task in ("ten_choice_first_person", "drag_first_person"):
        selected = sorted(
            (episode for episode in episodes if episode["variant"] == task),
            key=lambda episode: str(episode["pair_id"]),
        )
        rows[task] = [
            _first_person_row(
                episode=episode,
                manifest_path=manifest_path,
                seed=_ROW_SEED_BASES[(split, task)] + index,
                split=split,
            )
            for index, episode in enumerate(selected)
        ]
    return rows


def _mixed_rows(
    rows_by_task: Mapping[str, Sequence[dict[str, Any]]],
    *,
    count_per_task: int,
) -> list[dict[str, Any]]:
    selected = {
        task: list(rows)[::4][:count_per_task]
        for task, rows in rows_by_task.items()
    }
    return [
        copy.deepcopy(selected[task][index])
        for index in range(count_per_task)
        for task in TASKS
    ]


def build_four_task_rl_data(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    rotation_pool_root: Path = DEFAULT_ROTATION_POOL_ROOT,
    sft_manifest: Path = DEFAULT_SFT_MANIFEST,
    test_manifest: Path = DEFAULT_TEST_MANIFEST,
    train_count_per_task: int = TRAIN_COUNT_PER_TASK,
    validation_count_per_task: int = VALIDATION_COUNT_PER_TASK,
    icon_root: Path = DEFAULT_ICON_DIR,
) -> dict[str, Any]:
    if train_count_per_task <= 0 or train_count_per_task % 4:
        raise ValueError("train_count_per_task must be a positive multiple of four")
    if validation_count_per_task <= 0:
        raise ValueError("validation_count_per_task must be positive")
    mixed_count_per_task = train_count_per_task // 4
    output_root = output_root.resolve()
    rotation_pool_root = rotation_pool_root.resolve()
    sft_manifest = sft_manifest.resolve()
    test_manifest = test_manifest.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace four-task RL data: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    pool_summary = _read_json(rotation_pool_root / "summary.json")
    pool_name = str(pool_summary["pool_name"])

    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    try:
        split_episodes: dict[str, list[dict[str, Any]]] = {}
        for split, count in (
            ("train", train_count_per_task),
            ("validation", validation_count_per_task),
        ):
            split_root = temporary_root / split
            first_person = _build_first_person_episodes(
                split_root=split_root,
                split=split,
                count=count,
                icon_root=icon_root,
            )
            rotation = _build_rotation_episodes(
                split_root=split_root,
                pool_root=rotation_pool_root,
                split=split,
                count=count,
            )
            split_episodes[split] = [*rotation, *first_person]
            _write_json(
                split_root / "manifest.json",
                _manifest(
                    split=split,
                    episodes=split_episodes[split],
                    expected_count=count,
                ),
            )

        disjointness = _disjoint_report(
            train_episodes=split_episodes["train"],
            validation_episodes=split_episodes["validation"],
            sft_manifest=sft_manifest,
            test_manifest=test_manifest,
        )
        final_train_manifest = output_root / "train/manifest.json"
        final_validation_manifest = output_root / "validation/manifest.json"
        train_rows = _rows_by_task(
            episodes=split_episodes["train"],
            manifest_path=final_train_manifest,
            split="train",
            rotation_pool_name=pool_name,
        )
        validation_rows = _rows_by_task(
            episodes=split_episodes["validation"],
            manifest_path=final_validation_manifest,
            split="validation",
            rotation_pool_name=pool_name,
        )
        for task in TASKS:
            _write_parquet(temporary_root / "datasets" / task / "train.parquet", train_rows[task])
            _write_parquet(
                temporary_root / "datasets" / task / "val.parquet",
                validation_rows[task],
            )
        mixed_train = _mixed_rows(
            train_rows,
            count_per_task=mixed_count_per_task,
        )
        mixed_validation = [
            copy.deepcopy(validation_rows[task][index])
            for index in range(validation_count_per_task)
            for task in TASKS
        ]
        _write_parquet(temporary_root / "datasets/mixed/train.parquet", mixed_train)
        _write_parquet(temporary_root / "datasets/mixed/val.parquet", mixed_validation)

        formal_test = _read_json(test_manifest)
        formal_test_counts = Counter(
            str(episode["variant"])
            for episode in formal_test["episodes"]
            if episode.get("variant") in TASKS
        )
        if formal_test_counts != {task: 150 for task in TASKS}:
            raise ValueError(f"formal test task counts are invalid: {dict(formal_test_counts)}")
        summary = {
            "schema": "exploration_depth_four_task_rl_release_v1",
            "status": "passed",
            "release_id": RELEASE_ID,
            "implementation_tag": IMPLEMENTATION_TAG,
            "tasks": list(TASKS),
            "train_count_per_task": train_count_per_task,
            "validation_count_per_task": validation_count_per_task,
            "mixed_train_count": len(mixed_train),
            "mixed_train_count_per_task": mixed_count_per_task,
            "mixed_validation_count": len(mixed_validation),
            "sft_manifest": str(sft_manifest),
            "formal_test_manifest": str(test_manifest),
            "formal_test_role": "final_report_only_not_checkpoint_selection",
            "formal_test_counts": dict(sorted(formal_test_counts.items())),
            "rotation_background_pool": str(rotation_pool_root),
            "rotation_background_pool_name": pool_name,
            "rotation_variants_share_scene_and_challenge": True,
            "first_person_interaction_marker": INTERACTION_MARKER,
            "disjointness": disjointness,
            "comparison_method": (
                "episode IDs, pair IDs, canonical semantic scene fields, and rotation "
                "background identity/direct bytes; no file hashes"
            ),
            "validation_policy": (
                "independently regenerated RL validation scenes may be used for checkpoint "
                "selection; formal v7 test scenes are not included in val.parquet"
            ),
        }
        _write_json(temporary_root / "summary.json", summary)
        _write_json(
            temporary_root / "READY.json",
            {
                "status": "ready",
                "release_id": RELEASE_ID,
                "summary": str(output_root / "summary.json"),
            },
        )
        temporary_root.rename(output_root)

        for task in ("ten_choice_first_person", "drag_first_person"):
            for row in (*train_rows[task], *validation_rows[task]):
                assert_first_person_task_only_row(row)
        for task in ("rotation_inner", "rotation_outer"):
            for row in (*train_rows[task], *validation_rows[task]):
                rotation_online.assert_paired_task_only_row(row)
        return summary
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
