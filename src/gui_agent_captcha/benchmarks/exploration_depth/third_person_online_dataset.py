"""Task-only online-GRPO rows for third-person ten-choice and drag."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...domains.rotation.online_dataset import (
    ONLINE_REWARD_PLACEHOLDER,
    find_forbidden_fields,
)
from .training_no_think import PROMPT_CONTRACT, RESPONSE_CONTRACT, SIX_ACTION_KINDS

DATA_SOURCE = "exploration_depth_third_person_online_grpo_v1"
AGENT_NAME = "exploration_depth_third_person_no_think_online_v1"
TASKS = ("ten_choice_third_person", "drag_third_person")
TASK_FAMILIES = {
    "ten_choice_third_person": "ten_choice",
    "drag_third_person": "drag",
}
MINIMAL_DATASET_PROMPT = "Interact with the live third-person GUI environment."
VIEWPORT = (1280, 720)
MAX_STEPS = 12
FORMAT_ACTION_KINDS = SIX_ACTION_KINDS
EXECUTABLE_ACTION_KINDS = SIX_ACTION_KINDS
INTERACTION_MARKER = "mouse_icon"


def task_only_row(
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
        "task_config": {
            "suite_id": str(episode["suite_id"]),
            "manifest_path": str(Path(manifest_path).resolve()),
            "benchmark_variant": task,
            "episode_id": str(episode["episode_id"]),
            "viewport": list(VIEWPORT),
            "max_steps": MAX_STEPS,
            "coordinate_format": "qwen_relative_0_1000",
            "format_action_kinds": list(FORMAT_ACTION_KINDS),
            "executable_action_kinds": list(EXECUTABLE_ACTION_KINDS),
            "prompt_contract": PROMPT_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
        },
        "extra_info": {
            "index": seed,
            "split": split,
            "pair_id": str(episode["pair_id"]),
        },
    }
    return row


def assert_third_person_task_only_row(
    row: Mapping[str, Any],
    *,
    data_source: str = DATA_SOURCE,
    agent_name: str = AGENT_NAME,
    prompt_contract: str = PROMPT_CONTRACT,
    response_contract: str = RESPONSE_CONTRACT,
) -> None:
    forbidden = find_forbidden_fields(row)
    if forbidden:
        raise ValueError(
            "third-person online row contains replay/oracle fields: "
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
        raise ValueError("third-person online row has an unexpected top-level schema")
    task = str(row["task_type"])
    if task not in TASKS or row["ability"] != task:
        raise ValueError("third-person online row selects an unsupported task")
    if row["data_source"] != data_source or row["agent_name"] != agent_name:
        raise ValueError("third-person online row selects the wrong data source or loop")
    if row["prompt"] != [{"role": "user", "content": MINIMAL_DATASET_PROMPT}]:
        raise ValueError("third-person online prompt must remain task-only")
    if row["reward_model"] != {
        "style": "online_environment",
        "ground_truth": ONLINE_REWARD_PLACEHOLDER,
    }:
        raise ValueError("third-person reward must come from the live AgentLoop")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("third-person online seed must be an integer")

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
        raise ValueError("third-person task_config schema is invalid")
    if config["benchmark_variant"] != task:
        raise ValueError("third-person task_type and benchmark_variant differ")
    suffix = f"--{task.replace('_', '-')}"
    if not str(config["episode_id"]).endswith(suffix):
        raise ValueError("third-person episode ID does not match its variant")
    manifest_path = Path(str(config["manifest_path"]))
    if not manifest_path.is_absolute() or not manifest_path.is_file():
        raise ValueError("third-person manifest_path must be an existing absolute file")
    if list(config["viewport"]) != list(VIEWPORT) or config["max_steps"] != MAX_STEPS:
        raise ValueError("third-person viewport or action budget is invalid")
    if config["coordinate_format"] != "qwen_relative_0_1000":
        raise ValueError("third-person coordinate format is invalid")
    if tuple(config["format_action_kinds"]) != FORMAT_ACTION_KINDS:
        raise ValueError("third-person format action grammar is invalid")
    if tuple(config["executable_action_kinds"]) != EXECUTABLE_ACTION_KINDS:
        raise ValueError("third-person executable action grammar is invalid")
    if config["prompt_contract"] != prompt_contract:
        raise ValueError("third-person prompt contract is invalid")
    if config["response_contract"] != response_contract:
        raise ValueError("third-person response contract is invalid")

    extra = row["extra_info"]
    if not isinstance(extra, Mapping) or set(extra) != {"index", "split", "pair_id"}:
        raise ValueError("third-person extra_info schema is invalid")
    if extra["index"] != row["seed"] or extra["split"] not in {"train", "heldout"}:
        raise ValueError("third-person extra_info index or split is invalid")
