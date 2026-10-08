"""Repository-external paths owned by the ten-choice domain."""

from __future__ import annotations

from pathlib import Path

from ...integrations.storage import storage_path

_DEFAULT_CHECKPOINT_NAME = (
    "benchmark-sft"
)


def source_episodes_root() -> Path:
    return storage_path(
        "data",
        "ten_choice",
        "dataset",
        "source_episodes",
    )


def real_environment_root() -> Path:
    return storage_path("data", "ten_choice", "dataset")


def real_environment_hd720_root() -> Path:
    return storage_path(
        "data",
        "ten_choice",
        "dataset",
        "test_1280x720",
    )


def sft_root() -> Path:
    return storage_path("data", "ten_choice", "sft")


def trace_root() -> Path:
    return storage_path("runs", "ten_choice", "traces")


def environment_run_root() -> Path:
    return storage_path("runs", "ten_choice", "environment")


def evaluation_run_root() -> Path:
    return storage_path("runs", "ten_choice", "evaluation")


def server_run_root() -> Path:
    return storage_path("runs", "ten_choice", "server")


def default_checkpoint() -> Path:
    return storage_path("models", "ten_choice", _DEFAULT_CHECKPOINT_NAME)
