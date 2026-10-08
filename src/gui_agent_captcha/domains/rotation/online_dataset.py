from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .online_contract import INTERACTION_ONLINE_VIEWPORT

DATA_SOURCE = "interaction_rotation_browser_online_grpo"
AGENT_NAME = "interaction_browser_online"
ONLINE_REWARD_PLACEHOLDER = "__agent_loop_online_browser_reward__"
MINIMAL_DATASET_PROMPT = "Interact with the live rotation CAPTCHA."
TRAIN_BACKGROUND_POOL = "openimages-rotation-train-1k-20260604"
HELDOUT_BACKGROUND_POOL = "openimages-rotation-test-150-20260514"


@dataclass(frozen=True)
class SeedSplitSpec:
    start: int
    count: int
    background_pool: str


DEFAULT_SEED_SPLITS: dict[str, SeedSplitSpec] = {
    "train": SeedSplitSpec(
        start=10_000_000,
        count=1024,
        background_pool=TRAIN_BACKGROUND_POOL,
    ),
    "baseline": SeedSplitSpec(
        start=20_000_000,
        count=64,
        background_pool=TRAIN_BACKGROUND_POOL,
    ),
    "heldout": SeedSplitSpec(
        start=30_000_000,
        count=150,
        background_pool=HELDOUT_BACKGROUND_POOL,
    ),
}


FORBIDDEN_TASK_ONLY_FIELDS = frozenset(
    {
        "action",
        "actions",
        "bytes",
        "dom_answer",
        "dom_oracle",
        "dom_verifier",
        "expected_action",
        "expected_actions",
        "expert_action",
        "expert_trajectory",
        "frame",
        "frames",
        "image",
        "image_path",
        "images",
        "next_frame",
        "next_image",
        "next_observation",
        "next_screenshot",
        "observation",
        "observation_image",
        "oracle",
        "screenshot",
        "screenshots",
        "target",
        "target_action",
        "target_angle",
        "target_trajectory",
        "trajectory",
    }
)


def split_seed_values(split: str, *, count: int | None = None) -> list[int]:
    try:
        spec = DEFAULT_SEED_SPLITS[split]
    except KeyError as exc:
        raise ValueError(f"unknown online GRPO split: {split!r}") from exc
    selected_count = spec.count if count is None else count
    if not isinstance(selected_count, int) or isinstance(selected_count, bool):
        raise TypeError("split count must be an integer")
    if selected_count < 0 or selected_count > spec.count:
        raise ValueError(
            f"split {split!r} count must be between 0 and {spec.count}, "
            f"got {selected_count}"
        )
    return list(range(spec.start, spec.start + selected_count))


def _task_config(*, background_pool: str) -> dict[str, Any]:
    return {
        "background_pool": background_pool,
        "viewport": list(INTERACTION_ONLINE_VIEWPORT),
        "max_steps": 6,
        "coordinate_format": "qwen_relative_0_1000",
        "action_kinds": ["move_to", "mouse_down", "mouse_up"],
    }


def build_split_rows(split: str, *, count: int | None = None) -> list[dict[str, Any]]:
    try:
        spec = DEFAULT_SEED_SPLITS[split]
    except KeyError as exc:
        raise ValueError(f"unknown online GRPO split: {split!r}") from exc

    rows: list[dict[str, Any]] = []
    for seed in split_seed_values(split, count=count):
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
            "task_config": _task_config(background_pool=spec.background_pool),
            "extra_info": {
                "index": seed,
                "split": split,
            },
        }
        assert_task_only_row(row)
        rows.append(row)
    return rows


def find_forbidden_fields(value: Any, *, path: str = "row") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}"
            if key.lower() in FORBIDDEN_TASK_ONLY_FIELDS:
                found.append(item_path)
            if key.lower() == "type" and isinstance(item, str) and item.lower() in {
                "image",
                "video",
            }:
                found.append(f"{item_path}={item.lower()}")
            found.extend(find_forbidden_fields(item, path=item_path))
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            found.extend(find_forbidden_fields(item, path=f"{path}[{index}]"))
    return found


def assert_task_only_row(row: Mapping[str, Any]) -> None:
    forbidden = find_forbidden_fields(row)
    if forbidden:
        raise ValueError(
            "online GRPO row contains forbidden offline/replay field(s): "
            + ", ".join(forbidden)
        )

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
        raise ValueError(
            f"online GRPO row keys must be exactly {sorted(expected_keys)!r}; "
            f"got {sorted(row)!r}"
        )
    if row["data_source"] != DATA_SOURCE:
        raise ValueError("unexpected data_source")
    if row["agent_name"] != AGENT_NAME:
        raise ValueError("unexpected agent_name")
    if row["task_type"] != "rotation_captcha" or row["ability"] != "rotation_captcha":
        raise ValueError("only rotation_captcha rows are supported")
    if row["prompt"] != [{"role": "user", "content": MINIMAL_DATASET_PROMPT}]:
        raise ValueError("dataset prompt must remain the fixed task-only placeholder")
    if row["reward_model"] != {
        "style": "online_browser",
        "ground_truth": ONLINE_REWARD_PLACEHOLDER,
    }:
        raise ValueError("reward_model must be the online AgentLoop placeholder")

    seed = row["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    extra_info = row["extra_info"]
    if not isinstance(extra_info, Mapping) or set(extra_info) != {"index", "split"}:
        raise ValueError("extra_info must contain exactly index and split")
    if extra_info["index"] != seed:
        raise ValueError("extra_info.index must equal seed so each row is one GRPO group")

    split = extra_info["split"]
    if split not in DEFAULT_SEED_SPLITS:
        raise ValueError(f"unknown online GRPO split: {split!r}")
    spec = DEFAULT_SEED_SPLITS[split]
    if not spec.start <= seed < spec.start + spec.count:
        raise ValueError(f"seed {seed} is outside the reserved {split!r} range")
    if row["task_config"] != _task_config(background_pool=spec.background_pool):
        raise ValueError("task_config must contain only the frozen public task settings")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Writing parquet requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def export_dataset(
    *,
    output_dir: Path,
    output_format: str = "parquet",
    split_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    if output_format not in {"jsonl", "parquet"}:
        raise ValueError("output_format must be 'jsonl' or 'parquet'")

    counts = {
        split: spec.count
        for split, spec in DEFAULT_SEED_SPLITS.items()
    }
    if split_counts is not None:
        if set(split_counts) != set(DEFAULT_SEED_SPLITS):
            raise ValueError(
                "split_counts must contain exactly train, baseline, and heldout"
            )
        counts = {split: int(split_counts[split]) for split in DEFAULT_SEED_SPLITS}

    split_rows = {
        split: build_split_rows(split, count=counts[split])
        for split in DEFAULT_SEED_SPLITS
    }
    all_seeds = [row["seed"] for rows in split_rows.values() for row in rows]
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("online GRPO split seed ranges overlap")

    outputs: dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        path = output_dir / f"{split}.{output_format}"
        if output_format == "jsonl":
            _write_jsonl(path, rows)
        else:
            _write_parquet(path, rows)
        outputs[split] = str(path)

    summary = {
        "data_source": DATA_SOURCE,
        "agent_name": AGENT_NAME,
        "format": output_format,
        "output_dir": str(output_dir),
        "outputs": outputs,
        "counts": {split: len(rows) for split, rows in split_rows.items()},
        "seed_ranges": {
            split: {
                "start": spec.start,
                "stop_exclusive": spec.start + counts[split],
                "background_pool": spec.background_pool,
            }
            for split, spec in DEFAULT_SEED_SPLITS.items()
        },
        "one_row_per_group": True,
        "contains_recorded_observations": False,
        "reward_source": "InteractionBrowserAgentLoop.reward_score",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
