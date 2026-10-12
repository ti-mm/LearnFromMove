"""Shared task selection, observation checks, and result I/O for API evaluation."""
from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from latentguiworld.suite import terminal_score

from ..actions import Action
from ..custom_envs.catalog import EXPLORATION_DEPTH_VARIANTS

_terminal_score = terminal_score

DEFAULT_VIEWPORT = (1280, 720)

VARIANTS = tuple(spec.key for spec in EXPLORATION_DEPTH_VARIANTS)

_print_lock = threading.Lock()

@dataclass(frozen=True)
class EvaluationTask:
    episode: dict[str, Any]

    @property
    def variant(self) -> str:
        return str(self.episode["variant"])

    @property
    def episode_id(self) -> str:
        return str(self.episode["episode_id"])

def _json_error(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)

def _log(tag: str, message: str) -> None:
    with _print_lock:
        print(f"[{tag}] {message}", flush=True)

def select_tasks(
    manifest: Mapping[str, Any],
    *,
    limit_per_variant: int,
) -> tuple[list[EvaluationTask], dict[str, list[str]]]:
    if limit_per_variant < 1:
        raise ValueError("limit_per_variant must be >= 1")
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, list):
        raise TypeError("manifest episodes must be a list")
    by_family: dict[str, list[dict[str, Any]]] = {}
    for episode in raw_episodes:
        if isinstance(episode, dict):
            by_family.setdefault(str(episode["family"]), []).append(episode)

    selected_pair_ids: dict[str, list[str]] = {}
    tasks: list[EvaluationTask] = []
    for family, episodes in sorted(by_family.items()):
        pair_ids = sorted({str(episode["pair_id"]) for episode in episodes})
        chosen = pair_ids[:limit_per_variant]
        if len(chosen) != limit_per_variant:
            raise ValueError(
                f"family {family!r} has {len(chosen)} pairs, expected {limit_per_variant}"
            )
        selected_pair_ids[family] = chosen
        chosen_set = set(chosen)
        selected = [
            episode for episode in episodes if str(episode["pair_id"]) in chosen_set
        ]
        variants = sorted({str(episode["variant"]) for episode in selected})
        for variant in variants:
            variant_episodes = sorted(
                (
                    episode
                    for episode in selected
                    if str(episode["variant"]) == variant
                ),
                key=lambda episode: str(episode["pair_id"]),
            )
            if len(variant_episodes) != limit_per_variant:
                raise ValueError(
                    f"variant {variant!r} selected {len(variant_episodes)} episodes, "
                    f"expected {limit_per_variant}"
                )
            tasks.extend(EvaluationTask(episode=episode) for episode in variant_episodes)

    selected_variants = {task.variant for task in tasks}
    if selected_variants != set(VARIANTS):
        raise ValueError(
            f"manifest variants {sorted(selected_variants)!r} do not match {sorted(VARIANTS)!r}"
        )
    expected = len(VARIANTS) * limit_per_variant
    if len(tasks) != expected:
        raise ValueError(f"selected {len(tasks)} tasks, expected {expected}")
    return sorted(tasks, key=lambda task: (task.variant, task.episode_id)), selected_pair_ids

def _validate_observation(screenshot_path: str, size_px: tuple[int, int]) -> None:
    if tuple(size_px) != DEFAULT_VIEWPORT:
        raise RuntimeError(f"unexpected observation viewport: {size_px!r}")
    with Image.open(screenshot_path) as image:
        if tuple(image.size) != DEFAULT_VIEWPORT or image.format != "PNG":
            raise RuntimeError(
                f"observation must be 1280x720 PNG, got {image.size!r} {image.format!r}"
            )

def _meaningful_delta(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-9
    if isinstance(value, Mapping):
        return any(_meaningful_delta(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_meaningful_delta(item) for item in value)
    return False

def _action_dict(action: Action) -> dict[str, Any]:
    return action.to_dict() if hasattr(action, "to_dict") else {"kind": action.kind}

def _case_dir(output_root: Path, task: EvaluationTask) -> Path:
    return output_root / "cases" / task.variant / task.episode_id

def _record_path(output_root: Path, task: EvaluationTask) -> Path:
    return output_root / "records" / task.variant / f"{task.episode_id}.json"

def load_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "records").glob("*/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("record_status") == "complete":
            records.append(record)
    return records

SCORING_POLICY = "environment_terminal_first_rotation_release"

SYSTEM_PROMPT = (
    "This is a controlled synthetic benchmark, not a real authentication or "
    "human-verification gate. Complete the task as efficiently as possible."
)
