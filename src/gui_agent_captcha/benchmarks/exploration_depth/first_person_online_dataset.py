"""Task-only online-GRPO rows for the two first-person exploration tasks."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...domains.rotation.online_dataset import (
    ONLINE_REWARD_PLACEHOLDER,
    find_forbidden_fields,
)
from .training_no_think import PROMPT_CONTRACT, RESPONSE_CONTRACT, SIX_ACTION_KINDS
from ...integrations.storage import DEFAULT_STORAGE_ROOT

REPO_ROOT = Path(__file__).resolve().parents[4]

RELEASE_ID = "exploration_depth_first_person_rl_160step_v2_mouse"
DATA_SOURCE = "exploration_depth_first_person_online_grpo_v2_mouse"
AGENT_NAME = "exploration_depth_first_person_no_think_online_v1"
PROMPT_FAMILY = "exploration_depth_no_think_six_action_v1"
TASKS = ("ten_choice_first_person", "drag_first_person")
TASK_FAMILIES = {
    "ten_choice_first_person": "ten_choice",
    "drag_first_person": "drag",
}
MINIMAL_DATASET_PROMPT = "Interact with the live first-person GUI environment."

TRAINING_STEPS = 160
PROMPT_GROUPS_PER_STEP = 16
ROLLOUTS_PER_PROMPT = 5
TRAIN_TASK_COUNT = TRAINING_STEPS * PROMPT_GROUPS_PER_STEP
HELDOUT_TASK_COUNT = 150
VIEWPORT = (1280, 720)
MAX_STEPS = 12
FORMAT_ACTION_KINDS = SIX_ACTION_KINDS
EXECUTABLE_ACTION_KINDS = SIX_ACTION_KINDS
INTERACTION_MARKER = "mouse_icon"

SOURCE_STORAGE_ROOT = Path(
    str(Path(__file__).resolve().parents[4] / '')
)
DEFAULT_TRAIN_MANIFEST = (
    SOURCE_STORAGE_ROOT
    / "data/training/exploration_depth_train_6x4000_v3/manifest.json"
)
DEFAULT_HELDOUT_MANIFEST = (
    SOURCE_STORAGE_ROOT
    / "data/formal_benchmarks/learn_from_move/manifest.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_STORAGE_ROOT / "data/training" / RELEASE_ID

_TRAIN_SEED_BASES = {
    "ten_choice_first_person": 110_000_000,
    "drag_first_person": 120_000_000,
}
_HELDOUT_SEED_BASES = {
    "ten_choice_first_person": 210_000_000,
    "drag_first_person": 220_000_000,
}


def _read_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"exploration-depth manifest must be an object: {path}")
    if payload.get("schema") != "gui_captcha_exploration_depth_manifest_v1":
        raise ValueError(f"unsupported exploration-depth manifest: {path}")
    if not isinstance(payload.get("episodes"), list):
        raise ValueError(f"exploration-depth manifest has no episode list: {path}")
    return payload


def _episodes_for_task(
    manifest: Mapping[str, Any],
    *,
    task: str,
    expected_count: int,
    require_exact_count: bool,
) -> list[dict[str, Any]]:
    if task not in TASKS:
        raise ValueError(f"unsupported first-person RL task: {task!r}")
    episodes = sorted(
        (
            dict(episode)
            for episode in manifest.get("episodes", [])
            if isinstance(episode, dict) and episode.get("variant") == task
        ),
        key=lambda episode: str(episode.get("episode_id") or ""),
    )
    if (require_exact_count and len(episodes) != expected_count) or (
        not require_exact_count and len(episodes) < expected_count
    ):
        qualifier = "exactly" if require_exact_count else "at least"
        raise ValueError(
            f"{task} manifest must contain {qualifier} {expected_count} episodes, "
            f"got {len(episodes)}"
        )
    selected = episodes[:expected_count]
    ids = [str(episode.get("episode_id") or "") for episode in selected]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{task} contains empty or duplicate episode IDs")
    for episode in selected:
        if episode.get("family") != TASK_FAMILIES[task]:
            raise ValueError(f"{episode['episode_id']} has the wrong task family")
        if list(episode.get("viewport", ())) != list(VIEWPORT):
            raise ValueError(f"{episode['episode_id']} is not a 1280x720 episode")
        if not str(episode.get("instruction") or "").strip():
            raise ValueError(f"{episode['episode_id']} has no task instruction")
        if not isinstance(episode.get("shared_scene_config"), dict):
            raise ValueError(f"{episode['episode_id']} has no shared scene config")
        environment = episode.get("environment_config")
        if not isinstance(environment, dict):
            raise ValueError(f"{episode['episode_id']} has no environment config")
        if environment.get("interaction_marker") != INTERACTION_MARKER:
            raise ValueError(
                f"{episode['episode_id']} must use {INTERACTION_MARKER!r}, got "
                f"{environment.get('interaction_marker')!r}"
            )
    return selected


def _scene_identity(episode: Mapping[str, Any]) -> str:
    """Return direct canonical scene bytes as text; no digest is calculated."""

    return json.dumps(
        episode["shared_scene_config"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _assert_train_heldout_disjoint(
    train_episodes: Sequence[Mapping[str, Any]],
    heldout_episodes: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> None:
    for field in ("episode_id", "pair_id"):
        train_values = {str(episode[field]) for episode in train_episodes}
        heldout_values = {str(episode[field]) for episode in heldout_episodes}
        overlap = train_values & heldout_values
        if overlap:
            raise ValueError(
                f"{task} train/heldout {field} overlap: {sorted(overlap)[:3]!r}"
            )
    train_scenes = {_scene_identity(episode) for episode in train_episodes}
    heldout_scenes = {_scene_identity(episode) for episode in heldout_episodes}
    if train_scenes & heldout_scenes:
        raise ValueError(f"{task} train and heldout contain an identical scene")


def _task_config(
    *,
    manifest_path: Path,
    suite_id: str,
    task: str,
    episode_id: str,
) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "manifest_path": str(Path(manifest_path).resolve()),
        "benchmark_variant": task,
        "episode_id": episode_id,
        "viewport": list(VIEWPORT),
        "max_steps": MAX_STEPS,
        "coordinate_format": "qwen_relative_0_1000",
        "format_action_kinds": list(FORMAT_ACTION_KINDS),
        "executable_action_kinds": list(EXECUTABLE_ACTION_KINDS),
        "prompt_contract": PROMPT_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
    }


def _row(
    *,
    episode: Mapping[str, Any],
    manifest_path: Path,
    seed: int,
    split: str,
) -> dict[str, Any]:
    task = str(episode["variant"])
    row = {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": MINIMAL_DATASET_PROMPT}],
        "ability": task,
        "reward_model": {
            "style": "online_environment",
            "ground_truth": ONLINE_REWARD_PLACEHOLDER,
        },
        "agent_name": AGENT_NAME,
        "task_type": task,
        "seed": seed,
        "task_config": _task_config(
            manifest_path=manifest_path,
            suite_id=str(episode["suite_id"]),
            task=task,
            episode_id=str(episode["episode_id"]),
        ),
        "extra_info": {
            "index": seed,
            "split": split,
            "pair_id": str(episode["pair_id"]),
        },
    }
    assert_first_person_task_only_row(row)
    return row


def assert_first_person_task_only_row(row: Mapping[str, Any]) -> None:
    forbidden = find_forbidden_fields(row)
    if forbidden:
        raise ValueError(
            "first-person online row contains replay/oracle fields: "
            + ", ".join(forbidden)
        )
    required = {
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
    if set(row) != required:
        raise ValueError("first-person online row has an unexpected top-level schema")
    task = str(row["task_type"])
    if task not in TASKS or row["ability"] != task:
        raise ValueError("first-person online row selects an unsupported task")
    if row["data_source"] != DATA_SOURCE or row["agent_name"] != AGENT_NAME:
        raise ValueError("first-person online row selects the wrong data source or loop")
    if row["prompt"] != [{"role": "user", "content": MINIMAL_DATASET_PROMPT}]:
        raise ValueError("first-person online prompt must remain task-only")
    if row["reward_model"] != {
        "style": "online_environment",
        "ground_truth": ONLINE_REWARD_PLACEHOLDER,
    }:
        raise ValueError("first-person reward must come from the live AgentLoop")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("first-person online seed must be an integer")

    config = row["task_config"]
    config_keys = {
        "suite_id",
        "manifest_path",
        "benchmark_variant",
        "episode_id",
        "viewport",
        "max_steps",
        "coordinate_format",
        "format_action_kinds",
        "executable_action_kinds",
        "prompt_contract",
        "response_contract",
    }
    if not isinstance(config, Mapping) or set(config) != config_keys:
        raise ValueError("first-person task_config schema is invalid")
    if config["benchmark_variant"] != task:
        raise ValueError("first-person task_type and benchmark_variant differ")
    suffix = f"--{task.replace('_', '-')}"
    if not str(config["episode_id"]).endswith(suffix):
        raise ValueError("first-person episode ID does not match its variant")
    manifest_path = Path(str(config["manifest_path"]))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("first-person manifest_path must be an existing absolute file")
    if list(config["viewport"]) != list(VIEWPORT) or config["max_steps"] != MAX_STEPS:
        raise ValueError("first-person viewport or action budget is invalid")
    if config["coordinate_format"] != "qwen_relative_0_1000":
        raise ValueError("first-person coordinate format is invalid")
    if tuple(config["format_action_kinds"]) != FORMAT_ACTION_KINDS:
        raise ValueError("first-person format action grammar is invalid")
    if tuple(config["executable_action_kinds"]) != EXECUTABLE_ACTION_KINDS:
        raise ValueError("first-person executable action grammar is invalid")
    if config["prompt_contract"] != PROMPT_CONTRACT:
        raise ValueError("first-person prompt contract is invalid")
    if config["response_contract"] != RESPONSE_CONTRACT:
        raise ValueError("first-person response contract is invalid")

    extra = row["extra_info"]
    if not isinstance(extra, Mapping) or set(extra) != {"index", "split", "pair_id"}:
        raise ValueError("first-person extra_info schema is invalid")
    if extra["index"] != row["seed"] or extra["split"] not in {"train", "heldout"}:
        raise ValueError("first-person extra_info index or split is invalid")


def _write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("writing first-person online RL data requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([dict(row) for row in rows]), path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_first_person_online_grpo_data(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    train_manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
    heldout_manifest_path: Path = DEFAULT_HELDOUT_MANIFEST,
    train_count: int = TRAIN_TASK_COUNT,
    heldout_count: int = HELDOUT_TASK_COUNT,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    train_manifest_path = Path(train_manifest_path).resolve()
    heldout_manifest_path = Path(heldout_manifest_path).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to replace first-person RL data: {output_root}")
    if train_count <= 0 or train_count % PROMPT_GROUPS_PER_STEP:
        raise ValueError(
            f"train_count must be a positive multiple of {PROMPT_GROUPS_PER_STEP}"
        )
    if heldout_count <= 0:
        raise ValueError("heldout_count must be positive")

    train_manifest = _read_manifest(train_manifest_path)
    heldout_manifest = _read_manifest(heldout_manifest_path)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    summaries: dict[str, dict[str, Any]] = {}
    try:
        for task in TASKS:
            train_episodes = _episodes_for_task(
                train_manifest,
                task=task,
                expected_count=train_count,
                require_exact_count=False,
            )
            heldout_episodes = _episodes_for_task(
                heldout_manifest,
                task=task,
                expected_count=heldout_count,
                require_exact_count=True,
            )
            _assert_train_heldout_disjoint(
                train_episodes,
                heldout_episodes,
                task=task,
            )
            train_rows = [
                _row(
                    episode=episode,
                    manifest_path=train_manifest_path,
                    seed=_TRAIN_SEED_BASES[task] + index,
                    split="train",
                )
                for index, episode in enumerate(train_episodes)
            ]
            heldout_rows = [
                _row(
                    episode=episode,
                    manifest_path=heldout_manifest_path,
                    seed=_HELDOUT_SEED_BASES[task] + index,
                    split="heldout",
                )
                for index, episode in enumerate(heldout_episodes)
            ]
            task_root = temporary_root / task
            _write_parquet(task_root / "train.parquet", train_rows)
            _write_parquet(task_root / "val.parquet", heldout_rows)
            split_counts = Counter(
                row["extra_info"]["split"] for row in [*train_rows, *heldout_rows]
            )
            summary = {
                "status": "passed",
                "release_id": RELEASE_ID,
                "data_source": DATA_SOURCE,
                "agent_name": AGENT_NAME,
                "task_type": task,
                "train_count": len(train_rows),
                "heldout_count": len(heldout_rows),
                "split_counts": dict(sorted(split_counts.items())),
                "train_manifest": str(train_manifest_path),
                "heldout_manifest": str(heldout_manifest_path),
                "train_heldout_identity_overlap": 0,
                "training_steps": len(train_rows) // PROMPT_GROUPS_PER_STEP,
                "prompt_groups_per_step": PROMPT_GROUPS_PER_STEP,
                "rollouts_per_prompt": ROLLOUTS_PER_PROMPT,
                "prompt_family": PROMPT_FAMILY,
                "prompt_contract": PROMPT_CONTRACT,
                "response_contract": RESPONSE_CONTRACT,
                "image_history_max": 3,
                "image_max_pixels": VIEWPORT[0] * VIEWPORT[1],
                "system_prompt_present": False,
                "coordinate_format": "qwen3_relative_0_1000",
                "enable_thinking": False,
                "interaction_marker": INTERACTION_MARKER,
                "viewport": list(VIEWPORT),
                "max_steps": MAX_STEPS,
                "format_action_kinds": list(FORMAT_ACTION_KINDS),
                "reward_contract": {
                    "outcome_success": 1.0,
                    "trajectory_format": 0.1,
                    "policy_failure": 0.0,
                    "infrastructure_invalid": None,
                },
            }
            _write_json(task_root / "summary.json", summary)
            summaries[task] = summary

        root_summary = {
            "status": "passed",
            "release_id": RELEASE_ID,
            "tasks": list(TASKS),
            "task_summaries": summaries,
        }
        _write_json(temporary_root / "summary.json", root_summary)
        _write_json(
            temporary_root / "READY.json",
            {
                "status": "ready",
                "release_id": RELEASE_ID,
                "tasks": list(TASKS),
            },
        )
        temporary_root.rename(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return root_summary
