"""Expand the four-environment RL release with paired third-person tasks."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...domains.rotation import paired_online_dataset as rotation_online
from ...integrations.storage import DEFAULT_STORAGE_ROOT
from .first_person_online_dataset import (
    AGENT_NAME as FIRST_PERSON_AGENT_NAME,
    DATA_SOURCE as FIRST_PERSON_DATA_SOURCE,
    FORMAT_ACTION_KINDS,
    MAX_STEPS,
    MINIMAL_DATASET_PROMPT as FIRST_PERSON_DATASET_PROMPT,
    PROMPT_CONTRACT,
    RESPONSE_CONTRACT,
    assert_first_person_task_only_row,
)
from .fresh_rl_dataset import (
    DEFAULT_SFT_MANIFEST,
    DEFAULT_TEST_MANIFEST,
    RELEASE_ID as SOURCE_RELEASE_ID,
    TASKS as SOURCE_TASKS,
)
from .third_person_online_dataset import (
    TASKS as THIRD_PERSON_TASKS,
    assert_third_person_task_only_row,
    task_only_row as third_person_task_only_row,
)

RELEASE_ID = "exploration_depth_rl_6env_sft_disjoint_v1"
IMPLEMENTATION_TAG = "6env-v2-rl-sft-disjoint-v2"
TASKS = (
    "ten_choice_third_person",
    "ten_choice_first_person",
    "rotation_inner",
    "rotation_outer",
    "drag_third_person",
    "drag_first_person",
)
TRAIN_COUNT_PER_TASK = 2_560
VALIDATION_COUNT_PER_TASK = 150
VIEWPORT = (1280, 720)

DEFAULT_SOURCE_ROOT = DEFAULT_STORAGE_ROOT / "data/training" / SOURCE_RELEASE_ID
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "data/training" / RELEASE_ID

_ROW_SEED_BASES = {
    ("train", "rotation_inner"): 310_000_000,
    ("train", "rotation_outer"): 320_000_000,
    ("train", "ten_choice_first_person"): 330_000_000,
    ("train", "drag_first_person"): 340_000_000,
    ("train", "ten_choice_third_person"): 350_000_000,
    ("train", "drag_third_person"): 360_000_000,
    ("validation", "rotation_inner"): 410_000_000,
    ("validation", "rotation_outer"): 420_000_000,
    ("validation", "ten_choice_first_person"): 430_000_000,
    ("validation", "drag_first_person"): 440_000_000,
    ("validation", "ten_choice_third_person"): 450_000_000,
    ("validation", "drag_third_person"): 460_000_000,
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
        raise RuntimeError("writing six-environment RL data requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)


def _third_person_companion(episode: Mapping[str, Any]) -> dict[str, Any]:
    first_task = str(episode["variant"])
    pair_id = str(episode["pair_id"])
    shared = copy.deepcopy(dict(episode["shared_scene_config"]))
    source_environment = dict(episode["environment_config"])
    common = {
        key: copy.deepcopy(source_environment[key])
        for key in ("hidden_dynamics", "minimum_steps_gate")
        if key in source_environment
    }
    companion = copy.deepcopy(dict(episode))
    companion["suite_id"] = RELEASE_ID
    companion["implementation_tag"] = IMPLEMENTATION_TAG
    companion["shared_scene_config"] = shared
    if first_task == "ten_choice_first_person":
        task = "ten_choice_third_person"
        companion["exploration_level"] = "L1"
        companion["environment_config"] = {
            **common,
            "responsive_coordinate_system": "screen",
            "interaction_marker": "mouse_icon",
            "reset_frame_contract": "shared_scene_with_third_person_mouse_icon",
        }
        companion["success_evaluator"] = {
            "type": "clicked_target_icon",
            "terminal_event": "left_click",
        }
    elif first_task == "drag_first_person":
        task = "drag_third_person"
        companion["exploration_level"] = "L0"
        companion["environment_config"] = {
            **common,
            "responsive_coordinate_system": "screen",
            "interaction_marker": "mouse_icon",
            "reset_frame_contract": "third_person_mouse_icon_visible_reference",
        }
        companion["success_evaluator"] = {
            "type": "final_screen_geometry_match",
            "tolerance_px": episode["success_evaluator"]["tolerance_px"],
        }
    else:
        raise ValueError(f"cannot construct a third-person companion for {first_task!r}")
    companion["variant"] = task
    companion["episode_id"] = f"{pair_id}--{task.replace('_', '-')}"
    return companion


def _expand_manifest(
    source_manifest: Mapping[str, Any],
    *,
    split: str,
    expected_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_episodes = [
        copy.deepcopy(dict(episode))
        for episode in source_manifest.get("episodes", [])
        if isinstance(episode, Mapping)
    ]
    source_counts = Counter(str(episode.get("variant")) for episode in source_episodes)
    if source_counts != {task: expected_count for task in SOURCE_TASKS}:
        raise ValueError(
            f"source four-environment {split} counts are invalid: {dict(source_counts)}"
        )
    episodes: list[dict[str, Any]] = []
    for episode in source_episodes:
        episode["suite_id"] = RELEASE_ID
        episode["implementation_tag"] = IMPLEMENTATION_TAG
        episodes.append(episode)
    for task in ("ten_choice_first_person", "drag_first_person"):
        episodes.extend(
            _third_person_companion(episode)
            for episode in source_episodes
            if episode["variant"] == task
        )
    counts = Counter(str(episode["variant"]) for episode in episodes)
    if counts != {task: expected_count for task in TASKS}:
        raise ValueError(f"six-environment {split} counts are invalid: {dict(counts)}")

    for first_task, third_task in (
        ("ten_choice_first_person", "ten_choice_third_person"),
        ("drag_first_person", "drag_third_person"),
    ):
        first_by_pair = {
            str(episode["pair_id"]): episode
            for episode in episodes
            if episode["variant"] == first_task
        }
        third_by_pair = {
            str(episode["pair_id"]): episode
            for episode in episodes
            if episode["variant"] == third_task
        }
        if set(first_by_pair) != set(third_by_pair):
            raise ValueError(f"{first_task}/{third_task} pair IDs are not aligned")
        if any(
            first_by_pair[pair_id]["shared_scene_config"]
            != third_by_pair[pair_id]["shared_scene_config"]
            for pair_id in first_by_pair
        ):
            raise ValueError(f"{first_task}/{third_task} shared scenes differ")

    manifest = copy.deepcopy(dict(source_manifest))
    manifest.update(
        {
            "suite_id": RELEASE_ID,
            "implementation_tag": IMPLEMENTATION_TAG,
            "source_release_id": SOURCE_RELEASE_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frozen": False,
            "case_count_per_variant": expected_count,
            "total_episode_count": len(episodes),
            "trajectory_policy": "online_rl_task_only_no_oracle_fields_v1",
            "episodes": sorted(
                episodes,
                key=lambda item: (str(item["variant"]), str(item["pair_id"])),
            ),
        }
    )
    return manifest, episodes


def _semantic_identity(episode: Mapping[str, Any]) -> str:
    variant = str(episode["variant"])
    shared = episode["shared_scene_config"]
    if variant.startswith("ten_choice_"):
        value = {
            key: shared[key]
            for key in ("labels", "icon_positions_xy", "target_object", "hidden_dynamics")
        }
    elif variant.startswith("drag_"):
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
    references = {
        "sft": _read_json(sft_manifest)["episodes"],
        "formal_test": _read_json(test_manifest)["episodes"],
    }
    report: dict[str, Any] = {}
    for task in TASKS:
        train = [episode for episode in train_episodes if episode["variant"] == task]
        validation = [
            episode for episode in validation_episodes if episode["variant"] == task
        ]

        def compare(
            candidates: Sequence[Mapping[str, Any]],
            target: Sequence[Mapping[str, Any]],
        ) -> dict[str, int]:
            selected = [episode for episode in target if episode.get("variant") == task]
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
            "train_vs_sft": compare(train, references["sft"]),
            "train_vs_formal_test": compare(train, references["formal_test"]),
            "train_vs_validation": compare(train, validation),
            "validation_vs_sft": compare(validation, references["sft"]),
            "validation_vs_formal_test": compare(
                validation,
                references["formal_test"],
            ),
        }
    if any(
        count
        for task_report in report.values()
        for comparison in task_report.values()
        for count in comparison.values()
    ):
        raise ValueError(f"six-environment RL scene disjointness failed: {report}")
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
            "format_action_kinds": list(FORMAT_ACTION_KINDS),
            "executable_action_kinds": list(FORMAT_ACTION_KINDS),
            "prompt_contract": PROMPT_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
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
    for task, profile in (("rotation_inner", "inner"), ("rotation_outer", "outer")):
        rows[task] = rotation_online._rows_for_profile(
            episodes=rotation_episodes,
            profile=profile,
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
    for task in THIRD_PERSON_TASKS:
        selected = sorted(
            (episode for episode in episodes if episode["variant"] == task),
            key=lambda episode: str(episode["pair_id"]),
        )
        rows[task] = [
            third_person_task_only_row(
                episode=episode,
                manifest_path=manifest_path,
                seed=_ROW_SEED_BASES[(split, task)] + index,
                split="train" if split == "train" else "heldout",
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
        task: list(rows_by_task[task])[::4][:count_per_task]
        for task in TASKS
    }
    return [
        copy.deepcopy(selected[task][index])
        for index in range(count_per_task)
        for task in TASKS
    ]


def build_six_task_rl_data(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    sft_manifest: Path = DEFAULT_SFT_MANIFEST,
    test_manifest: Path = DEFAULT_TEST_MANIFEST,
    train_count_per_task: int = TRAIN_COUNT_PER_TASK,
    validation_count_per_task: int = VALIDATION_COUNT_PER_TASK,
) -> dict[str, Any]:
    if train_count_per_task <= 0 or train_count_per_task % 4:
        raise ValueError("train_count_per_task must be a positive multiple of four")
    if validation_count_per_task <= 0:
        raise ValueError("validation_count_per_task must be positive")
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    sft_manifest = Path(sft_manifest).resolve()
    test_manifest = Path(test_manifest).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace six-environment RL data: {output_root}")
    source_ready = _read_json(source_root / "READY.json")
    source_summary = _read_json(source_root / "summary.json")
    if (
        source_ready.get("status") != "ready"
        or source_ready.get("release_id") != SOURCE_RELEASE_ID
        or source_summary.get("status") != "passed"
        or source_summary.get("release_id") != SOURCE_RELEASE_ID
    ):
        raise ValueError("source four-environment RL release is not ready")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=output_root.parent)
    )
    published = False
    try:
        shutil.copytree(source_root, temporary_root, dirs_exist_ok=True, symlinks=True)
        split_episodes: dict[str, list[dict[str, Any]]] = {}
        for split, expected_count in (
            ("train", train_count_per_task),
            ("validation", validation_count_per_task),
        ):
            source_manifest = _read_json(source_root / split / "manifest.json")
            manifest, episodes = _expand_manifest(
                source_manifest,
                split=split,
                expected_count=expected_count,
            )
            split_episodes[split] = episodes
            _write_json(temporary_root / split / "manifest.json", manifest)

        disjointness = _disjoint_report(
            train_episodes=split_episodes["train"],
            validation_episodes=split_episodes["validation"],
            sft_manifest=sft_manifest,
            test_manifest=test_manifest,
        )
        rotation_pool_name = str(source_summary["rotation_background_pool_name"])
        final_manifests = {
            "train": output_root / "train/manifest.json",
            "validation": output_root / "validation/manifest.json",
        }
        rows = {
            split: _rows_by_task(
                episodes=split_episodes[split],
                manifest_path=final_manifests[split],
                split=split,
                rotation_pool_name=rotation_pool_name,
            )
            for split in ("train", "validation")
        }
        for task in TASKS:
            _write_parquet(
                temporary_root / "datasets" / task / "train.parquet",
                rows["train"][task],
            )
            _write_parquet(
                temporary_root / "datasets" / task / "val.parquet",
                rows["validation"][task],
            )
        mixed_count_per_task = train_count_per_task // 4
        mixed_train = _mixed_rows(rows["train"], count_per_task=mixed_count_per_task)
        mixed_validation = [
            copy.deepcopy(rows["validation"][task][index])
            for index in range(validation_count_per_task)
            for task in TASKS
        ]
        _write_parquet(temporary_root / "datasets/mixed/train.parquet", mixed_train)
        _write_parquet(temporary_root / "datasets/mixed/val.parquet", mixed_validation)

        formal_counts = Counter(
            str(episode["variant"])
            for episode in _read_json(test_manifest)["episodes"]
            if episode.get("variant") in TASKS
        )
        if formal_counts != {task: 150 for task in TASKS}:
            raise ValueError(f"formal test task counts are invalid: {dict(formal_counts)}")
        summary = {
            "schema": "exploration_depth_six_task_rl_release_v1",
            "status": "passed",
            "release_id": RELEASE_ID,
            "implementation_tag": IMPLEMENTATION_TAG,
            "source_release_id": SOURCE_RELEASE_ID,
            "source_release_root": str(source_root),
            "tasks": list(TASKS),
            "train_count_per_task": train_count_per_task,
            "validation_count_per_task": validation_count_per_task,
            "mixed_train_count": len(mixed_train),
            "mixed_train_count_per_task": mixed_count_per_task,
            "mixed_validation_count": len(mixed_validation),
            "sft_manifest": str(sft_manifest),
            "formal_test_manifest": str(test_manifest),
            "formal_test_role": "final_report_only_not_checkpoint_selection",
            "formal_test_counts": dict(sorted(formal_counts.items())),
            "rotation_background_pool": source_summary["rotation_background_pool"],
            "rotation_background_pool_name": rotation_pool_name,
            "paired_scene_contracts": {
                "rotation_inner_rotation_outer": "same_scene_and_challenge",
                "ten_choice_first_third_person": "same_scene_and_target",
                "drag_first_third_person": "same_scene_piece_and_slot",
            },
            "interaction_marker": "mouse_icon",
            "disjointness": disjointness,
            "comparison_method": (
                "episode IDs, pair IDs, and canonical semantic scene fields; rotation "
                "background byte-disjointness is inherited from the passed source release"
            ),
            "validation_policy": (
                "independently generated RL validation scenes may be used for checkpoint "
                "selection; formal v7 test scenes are excluded from val.parquet"
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
        published = True

        for task in ("ten_choice_first_person", "drag_first_person"):
            for row in (*rows["train"][task], *rows["validation"][task]):
                assert_first_person_task_only_row(row)
        for task in THIRD_PERSON_TASKS:
            for row in (*rows["train"][task], *rows["validation"][task]):
                assert_third_person_task_only_row(row)
        for task in ("rotation_inner", "rotation_outer"):
            for row in (*rows["train"][task], *rows["validation"][task]):
                rotation_online.assert_paired_task_only_row(row)
        return summary
    except Exception:
        if published and output_root.exists():
            shutil.rmtree(output_root)
        else:
            shutil.rmtree(temporary_root, ignore_errors=True)
        raise
