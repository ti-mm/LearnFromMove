"""Repository-external paths owned by the slot-drag domain."""

from __future__ import annotations

from pathlib import Path

from ...integrations.storage import storage_path


def episodes_root() -> Path:
    return storage_path("data", "slot_drag", "episodes")


def paper_v2_episodes_root() -> Path:
    return storage_path("data", "slot_drag", "episodes_v2")


def formal_benchmark_root() -> Path:
    return storage_path("data", "slot_drag", "benchmark_150_v1")


def legacy_sft_root() -> Path:
    return storage_path("data", "slot_drag", "sft", "legacy")


def paper_v2_sft_root() -> Path:
    return storage_path("data", "slot_drag", "sft", "paper_v2")


def online_rl_root() -> Path:
    return storage_path("data", "slot_drag", "online_rl", "iid")


def environment_run_root() -> Path:
    return storage_path("runs", "slot_drag", "environment")


def evaluation_run_root() -> Path:
    return storage_path("runs", "slot_drag", "evaluation")


def online_run_root() -> Path:
    return storage_path("runs", "slot_drag", "online_rl")
