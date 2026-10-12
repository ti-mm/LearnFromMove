"""Repository-external paths owned by the third-person-drag domain."""

from __future__ import annotations

from pathlib import Path

from ...integrations.storage import storage_path


def legacy_episodes_root() -> Path:
    return storage_path("data", "third_person_drag", "episodes")


def paper_v2_episodes_root() -> Path:
    return storage_path("data", "third_person_drag", "episodes_v2")


def formal_benchmark_root() -> Path:
    return storage_path("data", "third_person_drag", "benchmark_150_v1")


def paper_v2_sft_root() -> Path:
    return storage_path("data", "third_person_drag", "sft", "paper_v2")


def paper_v3_sft_root() -> Path:
    return storage_path("data", "third_person_drag", "sft", "paper_v3")


def teacher_output_root() -> Path:
    return storage_path("data", "third_person_drag", "teacher_thinks")


def teacher_report_root() -> Path:
    return storage_path("artifacts", "third_person_drag", "teacher_thinks")


def environment_run_root() -> Path:
    return storage_path("runs", "third_person_drag", "environment")


def evaluation_run_root() -> Path:
    return storage_path("runs", "third_person_drag", "evaluation")


def server_run_root() -> Path:
    return storage_path("runs", "third_person_drag", "server")
