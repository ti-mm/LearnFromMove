from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, TypeVar

from PIL import Image, ImageDraw

from ...actions import PrimitiveAction
from ...data.jsonl_io import read_jsonl as _read_jsonl
from ...data.jsonl_io import write_jsonl
from ...protocol_tracks import (
    DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS,
    PROTOCOL_VERSION,
    build_canonical_assistant_response,
    extract_think_text,
    set_canonical_assistant_response,
)
from . import validation as _validation
from .dataset import (
    ICON_COUNT,
    ICON_SIZE_PX,
    OLD_FIXED_TOOLTIP_LABELS,
    VIEWPORT_HEIGHT,
    VIEWPORT_WIDTH,
    episode_html_has_instant_tooltip_hide,
    generate_dataset,
    rewrite_episode_html_with_latest_tooltip_css,
)
from .environment import HoverRevealEnv
from .icon_assets import DEFAULT_ICON_DIR
from .paths import sft_root, source_episodes_root
from .paths import trace_root as default_trace_root
from .validation import TenChoiceValidationSettings

_count_actions = _validation.count_actions
_cursor_hotspot_present = _validation._cursor_hotspot_present
_detect_tooltip_regions_in_images = _validation.detect_tooltip_regions_in_images
_image_as_base64_data_uri = _validation._image_as_base64_data_uri
_record_observation_hashes = _validation.record_observation_hashes
_run_tooltip_dom_hover_check = _validation.run_tooltip_dom_hover_check
_validate_cursor_overlays = _validation.validate_cursor_overlays
_validate_exported_dataset = _validation.validate_exported_dataset
_validation_tooltip_region_count = _validation._tooltip_region_count
_write_cursor_overlay_report = _validation.write_cursor_overlay_report
_write_examples_zh = _validation.write_examples_zh
_write_full_sample_cases = _validation.write_full_sample_cases
_write_validation_report = _validation.write_validation_report


TASK_TYPE = "ten_choice_captcha"
BASE_TASK_TYPE = "hover_reveal"
SUPPORTED_ACTION_KINDS = set(DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS)
FORMAL_TARGET_MIN_RECORDS = 2_000
FORMAL_TARGET_MAX_RECORDS = 4_000
ICON_HITBOX_TOLERANCE_PX = 2
TRAINABLE_METADATA_KEYS = {
    "task_type",
    "base_task_type",
    "source_mode",
    "candidate_count",
    "hover_reveal_episode_id",
    "hover_reveal_split",
    "protocol_track",
    "action_space_type",
    "action_paradigm",
    "task_action_kinds",
    "protocol_version",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
ORACLE_FIELD_KEYS = {
    "target_index",
    "target_label",
    "icon_centers_xy",
    "scan_order",
    "candidate_index",
    "candidate_labels",
    "click_events",
    "success_rank",
    "source_episode_dir",
    "source_trace_path",
}
TOOLTIP_RGB = (15, 23, 42)
TOOLTIP_RGB_TOLERANCE = 24
TOOLTIP_MIN_REGION_PIXELS = 2_000
CURSOR_TIP_RGB = (0, 0, 0)
CURSOR_OVERLAY_VERSION = "white_arrow_black_outline_v1"

_VALIDATION_SETTINGS = TenChoiceValidationSettings(
    task_type=TASK_TYPE,
    icon_count=ICON_COUNT,
    supported_action_kinds=frozenset(SUPPORTED_ACTION_KINDS),
    oracle_field_keys=frozenset(ORACLE_FIELD_KEYS),
    old_fixed_tooltip_labels=frozenset(OLD_FIXED_TOOLTIP_LABELS),
    tooltip_rgb=TOOLTIP_RGB,
    tooltip_rgb_tolerance=TOOLTIP_RGB_TOLERANCE,
    tooltip_min_region_pixels=TOOLTIP_MIN_REGION_PIXELS,
    cursor_overlay_version=CURSOR_OVERLAY_VERSION,
    viewport_width=VIEWPORT_WIDTH,
    viewport_height=VIEWPORT_HEIGHT,
)

THOUGHT_POOLS: dict[str, list[str]] = {
    "correct_icon": [
        "当前悬停标签与任务目标一致，可以提交这个候选。",
        "已经看到正确标签，下一步点击该图标完成选择。",
        "这个候选显示为目标标签，准备在当前位置提交。",
        "可见标签确认无误，将选择当前图标。",
    ],
    "wrong_icon": [
        "当前候选显示的标签不符合目标，继续检查其他候选。",
        "这个图标揭示的是错误标签，不能作为最终答案。",
        "已确认当前位置不是目标候选，需要转向下一个候选。",
        "当前标签与任务不匹配，继续按扫描顺序查找目标。",
    ],
    "miss": [
        "当前位置没有命中候选图标，需要移动到候选中心继续检查。",
        "这次点击落在图标外，下一步回到有效候选区域。",
        "当前点没有触发候选反馈，需要调整到图标中心。",
        "未命中任何候选，继续按扫描顺序定位图标。",
    ],
    "move_hover": [
        "移动到下一个候选图标中心，观察悬停后出现的标签。",
        "先把光标放到候选上，通过悬停读取该图标标签。",
        "按照扫描顺序定位候选区域，等待标签显现。",
        "移动到候选中心以获得可见标签反馈。",
    ],
    "mouse_down": [
        "按下鼠标开始这次点击。",
        "在当前位置执行鼠标按下动作。",
        "保持光标位置不变并按下鼠标。",
        "用鼠标按下确认当前交互点。",
    ],
    "mouse_up": [
        "释放鼠标完成这次点击。",
        "抬起鼠标以提交当前点击。",
        "完成按下后的释放动作。",
        "释放按钮，让页面处理这次点击。",
    ],
}

RenderMode = Literal["browser", "placeholder"]
ScanPolicy = Literal["mixed", "random", "left_to_right"]
SplitPolicy = Literal["train_val_test", "train_val_only", "train_only"]
T = TypeVar("T")


@dataclass(frozen=True)
class EpisodeRef:
    split: str
    episode_id: str
    episode_dir: Path
    meta: dict[str, Any]

    @property
    def target_index(self) -> int:
        return int(self.meta["target_idx"])


@dataclass(frozen=True)
class TraceRef:
    trace_path: Path
    episode_id: str
    target_index: int
    lines: tuple[dict[str, Any], ...]
    source_split: str | None = None
    source_episode_dir: Path | None = None


@dataclass(frozen=True)
class TraceDiscoveryResult:
    trace_refs: list[TraceRef]
    excluded: list[dict[str, Any]]


@dataclass(frozen=True)
class SplitResult:
    split: dict[str, list[dict[str, Any]]]
    excluded: list[dict[str, Any]]


@dataclass(frozen=True)
class SelectedItem:
    source_mode: Literal["episode_oracle", "trace"]
    ref: EpisodeRef | TraceRef

    @property
    def target_index(self) -> int:
        return self.ref.target_index

    @property
    def episode_id(self) -> str:
        return self.ref.episode_id


class PlaceholderRenderer:
    """Small offline image writer used by unit tests.

    Production exports should use the browser renderer, which captures the
    real HoverReveal HTML and its hover labels.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def reset(self, item_id: str, episode: EpisodeRef) -> str:
        return str(self._write(item_id, "reset", None, episode).resolve())

    def step(
        self,
        item_id: str,
        action: dict[str, Any],
        episode: EpisodeRef,
        *,
        step_index: int,
        cursor_xy: tuple[float, float] | None,
        capture: bool = True,
    ) -> str:
        if not capture:
            return ""
        return str(
            self._write(
                item_id,
                f"step-{step_index:04d}",
                cursor_xy,
                episode,
            ).resolve()
        )

    def close(self) -> None:
        return

    def _write(
        self,
        item_id: str,
        phase: str,
        cursor_xy: tuple[float, float] | None,
        episode: EpisodeRef,
    ) -> Path:
        path = self.output_dir / "images" / "placeholder" / item_id / f"{phase}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", tuple(episode.meta.get("viewport", [VIEWPORT_WIDTH, VIEWPORT_HEIGHT])), "white")
        draw = ImageDraw.Draw(image)
        for index, (x, y) in enumerate(episode.meta["icon_centers_xy"]):
            color = "#166534" if index == episode.target_index else "#334155"
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
        if cursor_xy is not None:
            x, y = cursor_xy
            draw.rectangle((x - 4, y - 4, x + 4, y + 4), outline="#dc2626", width=2)
        image.save(path)
        return path


class BrowserRenderer:
    def __init__(
        self,
        output_dir: Path,
        *,
        headless: bool = True,
        settle_timeout_ms: int = 30,
    ) -> None:
        self.output_dir = output_dir
        self.headless = headless
        self.settle_timeout_ms = settle_timeout_ms
        self._current_dataset_root: Path | None = None
        self._current_env: HoverRevealEnv | None = None

    def reset(self, item_id: str, episode: EpisodeRef) -> str:
        env = self._env_for_episode(episode)
        observation = env.reset(task_type=episode.episode_id, task_id=item_id)
        return str(Path(observation.screenshot_path).resolve())

    def step(
        self,
        item_id: str,
        action: dict[str, Any],
        episode: EpisodeRef,
        *,
        step_index: int,
        cursor_xy: tuple[float, float] | None,
        capture: bool = True,
    ) -> str:
        del item_id, step_index, cursor_xy
        env = self._env_for_episode(episode)
        primitive = _primitive_from_action(action)
        if env.page is None:
            raise RuntimeError("Browser page is not configured")
        if primitive.kind == "move_to":
            assert primitive.x is not None and primitive.y is not None
            env.page.mouse.move(primitive.x, primitive.y)
            env.cursor_xy = (primitive.x, primitive.y)
        elif primitive.kind == "mouse_down":
            env.page.mouse.down()
        elif primitive.kind == "mouse_up":
            env.page.mouse.up()
        else:
            raise ValueError(f"Unsupported ten-choice action: {primitive.kind}")
        if not capture:
            return ""
        env.page.wait_for_timeout(self.settle_timeout_ms)
        screenshot_path = env._capture_screenshot(phase="step")
        if not screenshot_path:
            raise RuntimeError("Failed to capture HoverReveal screenshot")
        return str(Path(screenshot_path).resolve())

    def close(self) -> None:
        if self._current_env is not None:
            self._current_env.close()
        self._current_env = None
        self._current_dataset_root = None

    def _env_for_episode(self, episode: EpisodeRef) -> HoverRevealEnv:
        dataset_root = episode.episode_dir.parent
        if self._current_dataset_root != dataset_root:
            if self._current_env is not None:
                self._current_env.close()
            self._current_env = HoverRevealEnv(
                dataset_root=dataset_root,
                artifact_dir=self.output_dir,
                enable_playwright=True,
                headless=self.headless,
                screenshot_format="jpeg",
                screenshot_quality=82,
            )
            self._current_dataset_root = dataset_root
        assert self._current_env is not None
        return self._current_env


def _primitive_from_action(action: dict[str, Any]) -> PrimitiveAction:
    kind = action["kind"]
    if kind == "move_to":
        return PrimitiveAction(kind="move_to", x=float(action["x"]), y=float(action["y"]))
    return PrimitiveAction(kind=kind)


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    write_jsonl(path, records)


def _load_meta(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sanitize_text(text: str) -> str:
    return EMAIL_RE.sub("<EMAIL>", text)


def _dedupe_text_pool(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = _sanitize_text(value.strip())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def build_thought_pool_config(*, seed: int, gpt_model: str | None = None) -> dict[str, Any]:
    pools = {name: _dedupe_text_pool(values) for name, values in THOUGHT_POOLS.items()}
    return {
        "seed": seed,
        "source": "deterministic_local_template_pool",
        "gpt": {
            "used": False,
            "model": gpt_model,
            "reason": "local deterministic fallback used for reproducibility and to avoid credential exposure",
        },
        "dedupe_strategy": "strip, email-like PII replacement, exact-text de-duplication",
        "length_control": "all local templates are concise one-sentence thoughts",
        "semantic_policy": (
            "thoughts may mention the currently visible hover/click feedback but must not expose hidden target "
            "oracle fields in trainable metadata"
        ),
        "pools": pools,
        "pool_sizes": {name: len(values) for name, values in pools.items()},
    }


def _thought(
    pool_name: str,
    *,
    seed: int,
    record_id: str,
    step_index: int,
    thought_pools: dict[str, list[str]] | None = None,
) -> str:
    pools = thought_pools or THOUGHT_POOLS
    values = pools[pool_name]
    rng = random.Random(f"{seed}:thought:{record_id}:{step_index}:{pool_name}")
    return values[rng.randrange(len(values))]


def _is_ten_choice_hover_reveal_meta(meta: dict[str, Any]) -> bool:
    centers = meta.get("icon_centers_xy")
    labels = meta.get("labels")
    target_idx = meta.get("target_idx")
    return (
        isinstance(centers, list)
        and len(centers) == ICON_COUNT
        and isinstance(labels, list)
        and len(labels) == ICON_COUNT
        and isinstance(target_idx, int)
        and 0 <= target_idx < ICON_COUNT
    )


def list_episode_refs(dataset_root: Path) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    if not dataset_root.exists():
        return refs

    immediate = sorted(dataset_root.glob("*/meta.json"))
    if immediate:
        split = dataset_root.name
        for meta_path in immediate:
            meta = _load_meta(meta_path)
            if _is_ten_choice_hover_reveal_meta(meta):
                refs.append(
                    EpisodeRef(
                        split=split,
                        episode_id=meta_path.parent.name,
                        episode_dir=meta_path.parent,
                        meta=meta,
                    )
                )
        return refs

    for split_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for meta_path in sorted(split_dir.glob("*/meta.json")):
            meta = _load_meta(meta_path)
            if _is_ten_choice_hover_reveal_meta(meta):
                refs.append(
                    EpisodeRef(
                        split=split_dir.name,
                        episode_id=meta_path.parent.name,
                        episode_dir=meta_path.parent,
                        meta=meta,
                    )
                )
    return refs


def ensure_hover_reveal_episodes(
    *,
    dataset_root: Path,
    icon_dir: Path,
    min_records: int,
    generate_if_missing: bool,
    seed: int,
) -> list[EpisodeRef]:
    refs = list_episode_refs(dataset_root)
    target_indices = {ref.target_index for ref in refs}
    required_records = max(min_records, ICON_COUNT * 8)
    needs_generation = (
        not refs
        or len(refs) < required_records
        or target_indices != set(range(ICON_COUNT))
    )
    if not needs_generation or not generate_if_missing:
        return refs
    generate_dataset(
        output_dir=dataset_root,
        icon_root=icon_dir,
        train_count=required_records,
        test_count=0,
        seed=seed,
    )
    return list_episode_refs(dataset_root)


def ensure_latest_tooltip_html(episode_refs: Iterable[EpisodeRef]) -> dict[str, Any]:
    rewritten: list[str] = []
    already_latest = 0
    stale_after_rewrite: list[str] = []
    for episode in episode_refs:
        changed = rewrite_episode_html_with_latest_tooltip_css(episode.episode_dir)
        if changed:
            rewritten.append(str(episode.episode_dir))
        else:
            already_latest += 1
        html_path = episode.episode_dir / "index.html"
        if not episode_html_has_instant_tooltip_hide(html_path.read_text(encoding="utf-8")):
            stale_after_rewrite.append(str(html_path))
    return {
        "checked_episode_count": len(rewritten) + already_latest,
        "rewritten_episode_count": len(rewritten),
        "already_latest_episode_count": already_latest,
        "stale_after_rewrite_count": len(stale_after_rewrite),
        "stale_after_rewrite_examples": stale_after_rewrite[:10],
        "policy": "non-hover tooltip uses visibility:hidden, opacity:0, transition:none; hover uses visibility:visible with opacity fade-in only",
    }


def _episode_refs_by_id(episode_refs: Iterable[EpisodeRef]) -> dict[str, EpisodeRef]:
    return {ref.episode_id: ref for ref in episode_refs}


def _trace_exclusion(
    trace_path: Path,
    reason: str,
    *,
    episode_id: str | None = None,
    detail: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "reason": reason,
        "source_trace_path": str(trace_path),
    }
    if episode_id is not None:
        item["hover_reveal_episode_id"] = episode_id
    if detail is not None:
        item["detail"] = detail
    return item


def discover_trace_refs(trace_root: Path) -> list[TraceRef]:
    return discover_trace_refs_with_exclusions(trace_root).trace_refs


def discover_trace_refs_with_exclusions(
    trace_root: Path,
    *,
    episode_refs: list[EpisodeRef] | None = None,
) -> TraceDiscoveryResult:
    if not trace_root.exists():
        return TraceDiscoveryResult(trace_refs=[], excluded=[])
    refs: list[TraceRef] = []
    excluded: list[dict[str, Any]] = []
    episode_lookup = _episode_refs_by_id(episode_refs or [])
    for trace_path in sorted(trace_root.glob("hover_reveal*/**/trace.jsonl")):
        try:
            lines = tuple(_read_jsonl(trace_path))
        except Exception as exc:
            excluded.append(
                _trace_exclusion(trace_path, "invalid_trace_jsonl", detail=str(exc))
            )
            continue
        if not lines:
            excluded.append(_trace_exclusion(trace_path, "empty_trace"))
            continue

        metadata = lines[0].get("initial_metadata")
        episode_id = None
        if isinstance(metadata, dict):
            maybe_episode_id = metadata.get("episode_id") or lines[0].get("task_type")
            if isinstance(maybe_episode_id, str):
                episode_id = maybe_episode_id

        if lines[-1].get("success") is not True:
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    "unsuccessful_trace",
                    episode_id=episode_id,
                )
            )
            continue
        action_kinds = [line.get("action", {}).get("kind") for line in lines]
        if not action_kinds or any(kind not in SUPPORTED_ACTION_KINDS for kind in action_kinds):
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    "unsupported_action_kind",
                    episode_id=episode_id,
                )
            )
            continue
        if not isinstance(metadata, dict) or len(metadata.get("icon_centers_xy", [])) != ICON_COUNT:
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    "missing_initial_metadata",
                    episode_id=episode_id,
                )
            )
            continue
        centers = metadata["icon_centers_xy"]
        target_idx = metadata.get("target_idx")
        if not isinstance(target_idx, int) or not isinstance(episode_id, str):
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    "missing_episode_or_target",
                    episode_id=episode_id,
                )
            )
            continue
        if not 0 <= target_idx < ICON_COUNT:
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    "target_index_out_of_range",
                    episode_id=episode_id,
                )
            )
            continue
        sequence_error = _trace_clean_success_sequence_error(
            lines,
            centers=centers,
            target_index=target_idx,
        )
        if sequence_error is not None:
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    sequence_error,
                    episode_id=episode_id,
                )
            )
            continue
        observation_error = _trace_observation_sequence_error(lines, initial_metadata=metadata)
        if observation_error is not None:
            excluded.append(
                _trace_exclusion(
                    trace_path,
                    observation_error,
                    episode_id=episode_id,
                )
            )
            continue
        source_episode = episode_lookup.get(episode_id)
        refs.append(
            TraceRef(
                trace_path=trace_path,
                episode_id=episode_id,
                target_index=target_idx,
                lines=lines,
                source_split=source_episode.split if source_episode is not None else None,
                source_episode_dir=source_episode.episode_dir if source_episode is not None else None,
            )
        )
    return TraceDiscoveryResult(trace_refs=refs, excluded=excluded)


def _stable_shuffle(items: Iterable[T], *, seed: int, salt: str) -> list[T]:
    shuffled = list(items)
    random.Random(f"{seed}:{salt}").shuffle(shuffled)
    return shuffled


def _dedupe_trace_refs_by_episode(trace_refs: list[TraceRef], *, seed: int) -> list[TraceRef]:
    deduped: dict[str, TraceRef] = {}
    for trace in _stable_shuffle(trace_refs, seed=seed, salt="trace-attempt-dedupe"):
        deduped.setdefault(trace.episode_id, trace)
    return sorted(deduped.values(), key=lambda trace: (trace.episode_id, str(trace.trace_path)))


def select_items(
    episode_refs: list[EpisodeRef],
    trace_refs: list[TraceRef],
    *,
    max_records: int | None,
    seed: int,
) -> list[SelectedItem]:
    if not episode_refs and not trace_refs:
        return []
    limit = max_records if max_records is not None else len(episode_refs) + len(trace_refs)
    if limit <= 0:
        return []

    trace_refs = _dedupe_trace_refs_by_episode(trace_refs, seed=seed)
    traces_by_target: dict[int, list[TraceRef]] = {
        index: [] for index in range(ICON_COUNT)
    }
    episodes_by_target: dict[int, list[EpisodeRef]] = {
        index: [] for index in range(ICON_COUNT)
    }
    for trace in _stable_shuffle(trace_refs, seed=seed, salt="trace-coverage"):
        traces_by_target[trace.target_index].append(trace)
    for episode in _stable_shuffle(episode_refs, seed=seed, salt="episode-coverage"):
        episodes_by_target[episode.target_index].append(episode)

    selected: list[SelectedItem] = []
    used_episode_ids: set[str] = set()

    if limit >= ICON_COUNT:
        missing_targets = [
            index
            for index in range(ICON_COUNT)
            if not traces_by_target[index] and not episodes_by_target[index]
        ]
        if missing_targets:
            raise ValueError(
                "Cannot cover all ten target indices; missing targets: "
                + ", ".join(str(index) for index in missing_targets)
            )
        for target_index in range(ICON_COUNT):
            if traces_by_target[target_index]:
                trace = next(
                    (
                        candidate
                        for candidate in traces_by_target[target_index]
                        if candidate.episode_id not in used_episode_ids
                    ),
                    traces_by_target[target_index][0],
                )
                selected.append(SelectedItem(source_mode="trace", ref=trace))
                used_episode_ids.add(trace.episode_id)
            else:
                episode = next(
                    (
                        candidate
                        for candidate in episodes_by_target[target_index]
                        if candidate.episode_id not in used_episode_ids
                    ),
                    episodes_by_target[target_index][0],
                )
                selected.append(SelectedItem(source_mode="episode_oracle", ref=episode))
                used_episode_ids.add(episode.episode_id)

    remaining_by_target: dict[int, list[SelectedItem]] = {index: [] for index in range(ICON_COUNT)}
    for trace in _stable_shuffle(trace_refs, seed=seed, salt="fill-traces"):
        if trace.episode_id not in used_episode_ids:
            remaining_by_target[trace.target_index].append(SelectedItem(source_mode="trace", ref=trace))
    for episode in _stable_shuffle(episode_refs, seed=seed, salt="fill-episodes"):
        if episode.episode_id not in used_episode_ids:
            remaining_by_target[episode.target_index].append(
                SelectedItem(source_mode="episode_oracle", ref=episode)
            )

    target_counts: Counter[int] = Counter(item.target_index for item in selected)
    while len(selected) < limit:
        available_targets = [
            target_index
            for target_index, candidates in remaining_by_target.items()
            if candidates
        ]
        if not available_targets:
            break
        next_target = min(
            available_targets,
            key=lambda target_index: (target_counts[target_index], target_index),
        )
        item = remaining_by_target[next_target].pop(0)
        if len(selected) >= limit:
            break
        if item.source_mode == "trace":
            trace = item.ref
            assert isinstance(trace, TraceRef)
            if trace.episode_id in used_episode_ids:
                continue
            used_episode_ids.add(trace.episode_id)
        else:
            episode = item.ref
            assert isinstance(episode, EpisodeRef)
            if episode.episode_id in used_episode_ids:
                continue
            used_episode_ids.add(episode.episode_id)
        selected.append(item)
        target_counts[item.target_index] += 1
    return selected[:limit]


def _oracle_trajectory_type_for_position(position: int) -> str:
    return {
        0: "direct_success",
        1: "left_to_right_explore",
        2: "right_to_left_explore",
        3: "random_explore",
        4: "nearest_neighbor_explore",
    }[position % 5]


def _rebalance_for_smoke_trajectory_coverage(
    selected: list[SelectedItem],
    episode_refs: list[EpisodeRef],
    *,
    seed: int,
) -> list[SelectedItem]:
    if len(selected) < 15:
        return selected
    required = {
        "direct_success",
        "left_to_right_explore",
        "right_to_left_explore",
        "random_explore",
        "nearest_neighbor_explore",
    }
    present = {
        _oracle_trajectory_type_for_position(index)
        for index, item in enumerate(selected)
        if item.source_mode == "episode_oracle"
    }
    missing = sorted(required - present)
    if not missing:
        return selected

    used_episode_ids = {item.episode_id for item in selected}
    replacements = [
        episode
        for episode in _stable_shuffle(episode_refs, seed=seed, salt="smoke-oracle-fill")
        if episode.episode_id not in used_episode_ids
    ]
    if len(replacements) < len(missing):
        return selected

    rebalanced = list(selected)
    protected_targets = {item.target_index for item in selected}
    replacement_index = 0
    for trajectory_type in missing:
        desired_mod = {
            "direct_success": 0,
            "left_to_right_explore": 1,
            "right_to_left_explore": 2,
            "random_explore": 3,
            "nearest_neighbor_explore": 4,
        }[trajectory_type]
        replace_position = next(
            (
                index
                for index, item in reversed(list(enumerate(rebalanced)))
                if index % 5 == desired_mod
                and item.source_mode == "trace"
                and sum(1 for other in rebalanced if other.target_index == item.target_index) > 1
            ),
            None,
        )
        if replace_position is None:
            replace_position = next(
                (
                    index
                    for index, item in reversed(list(enumerate(rebalanced)))
                    if item.source_mode == "trace"
                    and item.target_index not in protected_targets
                ),
                None,
            )
        if replace_position is None:
            replace_position = next(
                (
                    index
                    for index, item in reversed(list(enumerate(rebalanced)))
                    if item.source_mode == "trace"
                    and sum(1 for other in rebalanced if other.target_index == item.target_index) > 1
                ),
                None,
            )
        if replace_position is None:
            continue
        rebalanced[replace_position] = SelectedItem(
            source_mode="episode_oracle",
            ref=replacements[replacement_index],
        )
        replacement_index += 1
        if replacement_index >= len(replacements):
            break
    return rebalanced


def _hit_icon_index(
    centers: list[list[int]],
    x: float,
    y: float,
) -> int | None:
    half_size = ICON_SIZE_PX / 2 + ICON_HITBOX_TOLERANCE_PX
    hits = [
        index
        for index, (center_x, center_y) in enumerate(centers)
        if abs(float(center_x) - x) <= half_size
        and abs(float(center_y) - y) <= half_size
    ]
    if len(hits) != 1:
        return None
    return hits[0]


def _clean_trace_action(action: dict[str, Any]) -> dict[str, Any] | None:
    kind = action.get("kind")
    if kind == "move_to":
        x_value = action.get("x")
        y_value = action.get("y")
        if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
            return None
        return {"kind": "move_to", "x": int(round(float(x_value))), "y": int(round(float(y_value)))}
    if kind in {"mouse_down", "mouse_up"}:
        return {"kind": kind}
    return None


def _trace_clean_success_sequence_error(
    lines: tuple[dict[str, Any], ...],
    *,
    centers: list[list[int]],
    target_index: int,
) -> str | None:
    actions = [_clean_trace_action(line.get("action", {})) for line in lines]
    if any(action is None for action in actions):
        return "invalid_action_shape"
    cleaned = [action for action in actions if action is not None]
    if len(cleaned) < 3:
        return "too_few_actions"
    if [action["kind"] for action in cleaned[-2:]] != ["mouse_down", "mouse_up"]:
        return "missing_terminal_click"
    if any(action["kind"] != "move_to" for action in cleaned[:-2]):
        return "non_move_before_click"
    move_actions = list(cleaned[:-2])
    if not move_actions:
        return "missing_hover_action"
    if len(move_actions) > ICON_COUNT:
        return "too_many_hover_actions"
    hover_indices: list[int] = []
    for action in move_actions:
        hover_index = _hit_icon_index(centers, float(action["x"]), float(action["y"]))
        if hover_index is None:
            return "move_to_outside_icon_hitbox"
        hover_indices.append(hover_index)
    if len(set(hover_indices)) != len(hover_indices):
        return "duplicate_hover_candidate"
    if hover_indices[-1] != target_index:
        return "terminal_hover_not_target"
    if target_index in hover_indices[:-1]:
        return "target_hovered_before_terminal"
    return None


def _trace_observation_sequence_error(
    lines: tuple[dict[str, Any], ...],
    *,
    initial_metadata: dict[str, Any],
) -> str | None:
    initial_path = lines[0].get("initial_screenshot_path")
    if not isinstance(initial_path, str) or not initial_path or not Path(initial_path).exists():
        return "missing_initial_screenshot"

    seen_paths = {str(Path(initial_path))}
    expected_episode = initial_metadata.get("episode_id")
    expected_target = initial_metadata.get("target_idx")
    expected_centers = initial_metadata.get("icon_centers_xy")
    for expected_index, line in enumerate(lines):
        line_index = line.get("index")
        if isinstance(line_index, int) and line_index != expected_index:
            return "trace_index_out_of_sequence"
        obs_path = line.get("obs_screenshot_path")
        if not isinstance(obs_path, str) or not obs_path or not Path(obs_path).exists():
            return "missing_obs_screenshot"
        normalized_obs_path = str(Path(obs_path))
        if normalized_obs_path in seen_paths:
            return "reused_obs_screenshot"
        seen_paths.add(normalized_obs_path)
        obs_metadata = line.get("obs_metadata")
        if isinstance(obs_metadata, dict):
            if (
                obs_metadata.get("episode_id") != expected_episode
                or obs_metadata.get("target_idx") != expected_target
                or obs_metadata.get("icon_centers_xy") != expected_centers
            ):
                return "obs_metadata_mismatch"
    return None


def _left_to_right_order(centers: list[list[int]]) -> list[int]:
    return sorted(range(len(centers)), key=lambda index: (centers[index][0], centers[index][1]))


def _right_to_left_order(centers: list[list[int]]) -> list[int]:
    return sorted(range(len(centers)), key=lambda index: (-centers[index][0], centers[index][1]))


def _nearest_neighbor_order(
    centers: list[list[int]],
    *,
    target_index: int,
    seed: int,
    salt: str,
) -> list[int]:
    rng = random.Random(f"{seed}:{salt}:nearest")
    wrong_indices = [index for index in range(len(centers)) if index != target_index]
    current = rng.choice(wrong_indices)
    order = [current]
    remaining = set(range(len(centers))) - {current}
    while remaining:
        current_x, current_y = centers[current]
        current = min(
            remaining,
            key=lambda index: (centers[index][0] - current_x) ** 2 + (centers[index][1] - current_y) ** 2,
        )
        order.append(current)
        remaining.remove(current)
    return order


def _ensure_target_at_success_rank(
    order: list[int],
    *,
    target_index: int,
    success_rank: int,
) -> list[int]:
    success_rank = max(1, min(ICON_COUNT, success_rank))
    wrong_needed = success_rank - 1
    wrong_prefix = [index for index in order if index != target_index]
    if len(wrong_prefix) < wrong_needed:
        wrong_prefix.extend(
            index
            for index in range(ICON_COUNT)
            if index != target_index and index not in wrong_prefix
        )
    prefix = wrong_prefix[:wrong_needed]
    return prefix + [target_index]


def _order_until_target(order: list[int], *, target_index: int) -> list[int]:
    if target_index not in order:
        raise ValueError(f"Target index {target_index} is missing from scan order")
    target_position = order.index(target_index)
    return order[: target_position + 1]


def build_scan_order(
    *,
    centers: list[list[int]],
    target_index: int,
    record_index: int,
    seed: int,
    salt: str,
    success_rank: int | None = None,
    scan_policy: ScanPolicy = "mixed",
) -> tuple[str, list[int]]:
    rank = success_rank if success_rank is not None else 1 + (record_index % ICON_COUNT)
    if scan_policy == "random":
        order = list(range(len(centers)))
        random.Random(f"{seed}:{salt}:random").shuffle(order)
        return "random_explore", _ensure_target_at_success_rank(
            order,
            target_index=target_index,
            success_rank=rank,
        )
    if scan_policy == "left_to_right":
        order = _left_to_right_order(centers)
        return "left_to_right_explore", _order_until_target(
            order,
            target_index=target_index,
        )
    if scan_policy != "mixed":
        raise ValueError(f"Unsupported scan policy: {scan_policy}")

    mode_index = record_index % 5
    if mode_index == 0:
        order = list(range(len(centers)))
        return "direct_or_ranked_success", _ensure_target_at_success_rank(
            order,
            target_index=target_index,
            success_rank=rank,
        )
    if mode_index == 1:
        order = _left_to_right_order(centers)
        return "left_to_right_explore", _ensure_target_at_success_rank(
            order,
            target_index=target_index,
            success_rank=rank,
        )
    if mode_index == 2:
        order = _right_to_left_order(centers)
        return "right_to_left_explore", _ensure_target_at_success_rank(
            order,
            target_index=target_index,
            success_rank=rank,
        )
    if mode_index == 3:
        order = list(range(len(centers)))
        random.Random(f"{seed}:{salt}:random").shuffle(order)
        return "random_explore", _ensure_target_at_success_rank(
            order,
            target_index=target_index,
            success_rank=rank,
        )
    order = _nearest_neighbor_order(
        centers,
        target_index=target_index,
        seed=seed,
        salt=salt,
    )
    return "nearest_neighbor_explore", _ensure_target_at_success_rank(
        order,
        target_index=target_index,
        success_rank=rank,
    )


def _observation_step(image_path: str, cursor_xy: tuple[float, float] | None = None) -> dict[str, Any]:
    step: dict[str, Any] = {"type": "observation", "image_path": str(Path(image_path).resolve())}
    if cursor_xy is not None:
        step["cursor_xy"] = [cursor_xy[0], cursor_xy[1]]
    return step


def _cursor_polygon(x: float, y: float) -> list[tuple[float, float]]:
    return [
        (x, y),
        (x, y + 26),
        (x + 6, y + 21),
        (x + 11, y + 33),
        (x + 17, y + 30),
        (x + 12, y + 18),
        (x + 21, y + 18),
    ]


def overlay_cursor_on_image(
    source_path: Path,
    output_path: Path,
    *,
    cursor_xy: tuple[float, float],
) -> Path:
    """Draw a visible mouse pointer with its hotspot aligned to cursor_xy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rendered = overlay_cursor_on_image_memory(image, cursor_xy=cursor_xy)
    rendered.save(output_path)
    return output_path


def overlay_cursor_on_image_memory(
    image: Image.Image,
    *,
    cursor_xy: tuple[float, float],
) -> Image.Image:
    """Draw the canonical cursor overlay without creating a filesystem artifact."""
    rgba = image.convert("RGBA")
    draw = ImageDraw.Draw(rgba)
    x, y = cursor_xy
    polygon = _cursor_polygon(x, y)
    shadow = [(px + 2, py + 2) for px, py in polygon]
    draw.polygon(shadow, fill=(0, 0, 0, 96))
    draw.line(shadow + [shadow[0]], fill=(0, 0, 0, 120), width=4, joint="curve")
    draw.polygon(polygon, fill=(255, 255, 255, 255))
    draw.line(polygon + [polygon[0]], fill=(0, 0, 0, 255), width=3, joint="curve")
    draw.line([(x, y), (x, y + 5)], fill=(0, 0, 0, 255), width=2)
    return rgba.convert("RGB")


def _move_action_step(
    *,
    x: float,
    y: float,
    thought: str,
    candidate_index: int,
) -> dict[str, Any]:
    step = {
        "type": "action",
        "kind": "move_to",
        "x": int(round(x)),
        "y": int(round(y)),
        "metadata": {"candidate_index": candidate_index},
    }
    set_canonical_assistant_response(
        step,
        build_canonical_assistant_response(
            thought=thought,
            action={"kind": "move_to", "x": int(round(x)), "y": int(round(y))},
        ),
    )
    return step


def _button_action_step(kind: Literal["mouse_down", "mouse_up"], thought: str) -> dict[str, Any]:
    step = {
        "type": "action",
        "kind": kind,
    }
    set_canonical_assistant_response(
        step,
        build_canonical_assistant_response(thought=thought, action={"kind": kind}),
    )
    return step


def _action_from_step(step: dict[str, Any]) -> dict[str, Any]:
    if step["kind"] == "move_to":
        return {"kind": "move_to", "x": step["x"], "y": step["y"]}
    return {"kind": step["kind"]}


def _make_click_event(
    *,
    attempt_index: int,
    cursor_xy: tuple[float, float] | None,
    centers: list[list[int]],
    target_index: int,
    outcome: Literal["exploration", "success"],
) -> dict[str, Any]:
    hit_index = None
    if cursor_xy is not None:
        hit_index = _hit_icon_index(centers, float(cursor_xy[0]), float(cursor_xy[1]))
    if hit_index is None:
        status = "miss"
    elif hit_index == target_index:
        status = "correct_icon"
    else:
        status = "wrong_icon"
    event: dict[str, Any] = {
        "attempt_index": attempt_index,
        "click_hit_status": status,
        "outcome": outcome,
        "cursor_xy": [cursor_xy[0], cursor_xy[1]] if cursor_xy is not None else None,
        "hit_candidate_index": hit_index,
    }
    return event


def build_episode_record(
    *,
    episode: EpisodeRef,
    output_dir: Path,
    renderer: BrowserRenderer | PlaceholderRenderer,
    record_index: int,
    seed: int,
    success_rank: int | None = None,
    include_diagnostic_clicks: bool = True,
    thought_pools: dict[str, list[str]] | None = None,
    scan_policy: ScanPolicy = "mixed",
) -> dict[str, Any]:
    del output_dir, include_diagnostic_clicks
    centers = episode.meta["icon_centers_xy"]
    target_index = episode.target_index
    trajectory_type, scan_order = build_scan_order(
        centers=centers,
        target_index=target_index,
        record_index=record_index,
        seed=seed,
        salt=episode.episode_id,
        success_rank=success_rank,
        scan_policy=scan_policy,
    )
    record_id = f"ten_choice_captcha_{episode.split}_{episode.episode_id}_{trajectory_type}"
    steps: list[dict[str, Any]] = []
    reset_image = renderer.reset(record_id, episode)
    steps.append(_observation_step(reset_image))

    click_events: list[dict[str, Any]] = []
    cursor_xy: tuple[float, float] | None = None
    step_index = 0
    action_thought_index = 0

    for scan_position, candidate_index in enumerate(scan_order, start=1):
        center_x, center_y = centers[candidate_index]
        pool_name = "move_hover" if scan_position == 1 else "wrong_icon"
        thought = _thought(
            pool_name,
            seed=seed,
            record_id=record_id,
            step_index=action_thought_index,
            thought_pools=thought_pools,
        )
        action = _move_action_step(
            x=center_x,
            y=center_y,
            thought=thought,
            candidate_index=candidate_index,
        )
        steps.append(action)
        action_thought_index += 1
        cursor_xy = (float(action["x"]), float(action["y"]))
        step_index += 1
        image_path = renderer.step(
            record_id,
            action,
            episode,
            step_index=step_index,
            cursor_xy=cursor_xy,
        )
        steps.append(_observation_step(image_path, cursor_xy=cursor_xy))

    steps.append(
        _button_action_step(
            "mouse_down",
            _thought(
                "correct_icon",
                seed=seed,
                record_id=record_id,
                step_index=action_thought_index,
                thought_pools=thought_pools,
            ),
        )
    )
    action_thought_index += 1
    step_index += 1
    renderer.step(
        record_id,
        {"kind": "mouse_down"},
        episode,
        step_index=step_index,
        cursor_xy=cursor_xy,
        capture=False,
    )
    steps.append(_observation_step(image_path, cursor_xy=cursor_xy))

    click_events.append(
        _make_click_event(
            attempt_index=len(click_events) + 1,
            cursor_xy=cursor_xy,
            centers=centers,
            target_index=target_index,
            outcome="success",
        )
    )

    steps.append(
        _button_action_step(
            "mouse_up",
            _thought(
                "mouse_up",
                seed=seed,
                record_id=record_id,
                step_index=action_thought_index,
                thought_pools=thought_pools,
            ),
        )
    )
    step_index += 1
    renderer.step(
        record_id,
        {"kind": "mouse_up"},
        episode,
        step_index=step_index,
        cursor_xy=cursor_xy,
        capture=False,
    )

    wrong_hover_count = sum(1 for index in scan_order if index != target_index)
    return {
        "id": record_id,
        "source": "hover_reveal",
        "instruction": _sanitize_text(episode.meta["instruction"]),
        "metadata": {
            "task_type": TASK_TYPE,
            "base_task_type": BASE_TASK_TYPE,
            "source_mode": "episode_oracle",
            "candidate_count": ICON_COUNT,
            "protocol_track": "captcha_mixed_left_click",
            "action_space_type": "mixed_primitive_macro",
            "action_paradigm": "mixed",
            "task_action_kinds": list(DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS),
            "protocol_version": PROTOCOL_VERSION,
            "target_index": target_index,
            "target_label": episode.meta.get("target_label"),
            "candidate_labels": list(episode.meta.get("labels", [])),
            "hover_reveal_episode_id": episode.episode_id,
            "hover_reveal_split": episode.split,
            "source_episode_dir": str(episode.episode_dir),
            "source_trace_path": None,
            "icon_centers_xy": centers,
            "icon_asset": episode.meta.get("icon_asset"),
            "icon_source_name": episode.meta.get("icon_source_name"),
            "trajectory_type": trajectory_type,
            "scan_order": scan_order,
            "success_rank": len(scan_order),
            "candidate_checks_before_success": len(scan_order),
            "wrong_hover_count": wrong_hover_count,
            "click_events": click_events,
            "click_hit_status": click_events[-1]["click_hit_status"] if click_events else "correct_icon",
            "outcome": "success",
            "viewport": episode.meta.get("viewport", [VIEWPORT_WIDTH, VIEWPORT_HEIGHT]),
        },
        "steps": steps,
    }


def _cursor_overlay_output_path(
    *,
    source_path: Path,
    output_dir: Path,
    record_id: str,
    step_index: int,
    cursor_xy: tuple[float, float],
) -> Path:
    suffix = ".png"
    digest = hashlib.sha1(
        f"{source_path.resolve()}:{cursor_xy[0]:.3f}:{cursor_xy[1]:.3f}:{step_index}".encode("utf-8")
    ).hexdigest()[:12]
    return output_dir / "images" / "cursor_overlay" / record_id / f"cursor_{step_index:04d}_{digest}{suffix}"


def apply_cursor_overlays_to_records(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    cursor_observation_count = 0
    no_cursor_observation_count = 0
    overlay_image_count = 0
    missing_source_count = 0
    examples: list[dict[str, Any]] = []
    for record in records:
        for step_index, step in enumerate(record.get("steps", [])):
            if step.get("type") != "observation":
                continue
            cursor_values = step.get("cursor_xy")
            if not isinstance(cursor_values, list) or len(cursor_values) < 2:
                no_cursor_observation_count += 1
                continue
            x_value, y_value = cursor_values[0], cursor_values[1]
            if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                no_cursor_observation_count += 1
                continue
            cursor_xy = (float(x_value), float(y_value))
            source_path = Path(str(step["image_path"]))
            cursor_observation_count += 1
            if not source_path.exists():
                missing_source_count += 1
                continue
            output_path = _cursor_overlay_output_path(
                source_path=source_path,
                output_dir=output_dir,
                record_id=str(record["id"]),
                step_index=step_index,
                cursor_xy=cursor_xy,
            )
            overlay_cursor_on_image(source_path, output_path, cursor_xy=cursor_xy)
            step["image_path"] = str(output_path.resolve())
            overlay_image_count += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "record_id": record["id"],
                        "step_index": step_index,
                        "source_image_path": str(source_path.resolve()),
                        "overlay_image_path": str(output_path.resolve()),
                        "cursor_xy": [cursor_xy[0], cursor_xy[1]],
                    }
                )
    return {
        "version": CURSOR_OVERLAY_VERSION,
        "cursor_observation_count": cursor_observation_count,
        "no_cursor_observation_count": no_cursor_observation_count,
        "overlay_image_count": overlay_image_count,
        "missing_source_image_count": missing_source_count,
        "hotspot_alignment": "arrow polygon first vertex and black hotspot mark are drawn at cursor_xy",
        "reuse_policy": "output filename hash includes source image path, cursor_xy, and step index",
        "examples": examples,
    }


def _copy_trace_image(source: str, output_dir: Path, record_id: str, index: int) -> str:
    source_path = Path(source)
    suffix = source_path.suffix or ".png"
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    target = output_dir / "images" / "trace" / record_id / f"obs_{index:04d}_{digest}{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)
    return str(target.resolve())


def _trace_candidate_labels(metadata: dict[str, Any], target_index: int) -> list[str]:
    labels = metadata.get("labels") or metadata.get("candidate_labels")
    if isinstance(labels, list) and len(labels) == ICON_COUNT:
        return [str(label) for label in labels]
    target_label = str(metadata.get("target_label", "This is the target"))
    generated = [f"This is trace option {index + 1}" for index in range(ICON_COUNT)]
    generated[target_index] = target_label
    return generated


def build_trace_record(
    *,
    trace: TraceRef,
    output_dir: Path,
    seed: int = 42,
    thought_pools: dict[str, list[str]] | None = None,
    success_rank: int | None = None,
) -> dict[str, Any] | None:
    metadata = trace.lines[0].get("initial_metadata")
    if not isinstance(metadata, dict):
        return None
    centers = metadata["icon_centers_xy"]
    digest = hashlib.sha1(str(trace.trace_path).encode("utf-8")).hexdigest()[:12]
    record_id = f"ten_choice_captcha_trace_{trace.episode_id}_{digest}"
    steps: list[dict[str, Any]] = []
    first_image = _copy_trace_image(
        str(trace.lines[0]["initial_screenshot_path"]),
        output_dir,
        record_id,
        0,
    )
    steps.append(_observation_step(first_image))

    previous_hover_index: int | None = None
    wrong_hover_count = 0
    cursor_xy: tuple[float, float] | None = None
    hover_order: list[int] = []
    click_events: list[dict[str, Any]] = []
    for index, line in enumerate(trace.lines, start=1):
        action = _clean_trace_action(line.get("action", {}))
        if action is None:
            return None
        if action["kind"] == "move_to":
            hover_index = _hit_icon_index(centers, float(action["x"]), float(action["y"]))
            if hover_index is None:
                return None
            if previous_hover_index is None:
                pool_name = "move_hover"
            elif previous_hover_index == trace.target_index:
                pool_name = "correct_icon"
            else:
                pool_name = "wrong_icon"
            thought = _thought(
                pool_name,
                seed=seed,
                record_id=record_id,
                step_index=index,
                thought_pools=thought_pools,
            )
            if hover_index != trace.target_index:
                wrong_hover_count += 1
            previous_hover_index = hover_index
            hover_order.append(hover_index)
            cursor_xy = (float(action["x"]), float(action["y"]))
            steps.append(
                _move_action_step(
                    x=float(action["x"]),
                    y=float(action["y"]),
                    thought=thought,
                    candidate_index=hover_index,
                )
            )
        elif action["kind"] == "mouse_down":
            steps.append(
                _button_action_step(
                    "mouse_down",
                    _thought(
                        "mouse_down",
                        seed=seed,
                        record_id=record_id,
                        step_index=index,
                        thought_pools=thought_pools,
                    ),
                )
            )
            click_events.append(
                _make_click_event(
                    attempt_index=len(click_events) + 1,
                    cursor_xy=cursor_xy,
                    centers=centers,
                    target_index=trace.target_index,
                    outcome="success",
                )
            )
        else:
            steps.append(
                _button_action_step(
                    "mouse_up",
                    _thought(
                        "mouse_up",
                        seed=seed,
                        record_id=record_id,
                        step_index=index,
                        thought_pools=thought_pools,
                    ),
                )
            )
        image_path = _copy_trace_image(str(line["obs_screenshot_path"]), output_dir, record_id, index)
        steps.append(_observation_step(image_path, cursor_xy=cursor_xy))

    return {
        "id": record_id,
        "source": "hover_reveal_trace",
        "instruction": _sanitize_text(trace.lines[0]["obs_instruction"]),
        "metadata": {
            "task_type": TASK_TYPE,
            "base_task_type": BASE_TASK_TYPE,
            "source_mode": "trace",
            "candidate_count": ICON_COUNT,
            "protocol_track": "captcha_mixed_left_click",
            "action_space_type": "mixed_primitive_macro",
            "action_paradigm": "mixed",
            "task_action_kinds": list(DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS),
            "protocol_version": PROTOCOL_VERSION,
            "target_index": trace.target_index,
            "target_label": metadata.get("target_label"),
            "candidate_labels": _trace_candidate_labels(metadata, trace.target_index),
            "hover_reveal_episode_id": trace.episode_id,
            "hover_reveal_split": trace.source_split,
            "source_trace_path": str(trace.trace_path),
            "source_episode_dir": (
                str(trace.source_episode_dir)
                if trace.source_episode_dir is not None
                else None
            ),
            "icon_centers_xy": centers,
            "icon_asset": metadata.get("icon_asset"),
            "icon_source_name": metadata.get("icon_source_name"),
            "trajectory_type": "trace_success",
            "scan_order": hover_order,
            "success_rank": success_rank or len(hover_order),
            "candidate_checks_before_success": success_rank or len(hover_order),
            "wrong_hover_count": wrong_hover_count,
            "click_events": click_events,
            "click_hit_status": click_events[-1]["click_hit_status"] if click_events else "correct_icon",
            "outcome": "success",
        },
        "steps": steps,
    }


def _record_group_key(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    for key in ("hover_reveal_episode_id", "source_episode_dir", "source_trace_path"):
        value = metadata.get(key)
        if value:
            return str(value)
    return str(record["id"])


def _record_is_official_test(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata", {})
    source_split = metadata.get("hover_reveal_split")
    source_episode_dir = str(metadata.get("source_episode_dir", ""))
    return source_split == "test" or "/test/" in source_episode_dir


def _record_primary_observation_hash(record: dict[str, Any]) -> str:
    for step in record.get("steps", []):
        if step.get("type") != "observation":
            continue
        image_path = step.get("image_path")
        if isinstance(image_path, str) and Path(image_path).exists():
            return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    return ""


def _split_exclusion(
    record: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    item: dict[str, Any] = {
        "reason": reason,
        "id": record.get("id"),
        "hover_reveal_episode_id": metadata.get("hover_reveal_episode_id"),
        "hover_reveal_split": metadata.get("hover_reveal_split"),
    }
    if metadata.get("source_trace_path") is not None:
        item["source_trace_path"] = metadata.get("source_trace_path")
    if metadata.get("source_episode_dir") is not None:
        item["source_episode_dir"] = metadata.get("source_episode_dir")
    return item


def _assign_group(
    *,
    split: dict[str, list[dict[str, Any]]],
    split_hashes: dict[str, set[str]],
    group: list[dict[str, Any]],
    group_hashes: set[str],
    preferred_split: str,
    allowed_splits: tuple[str, ...],
    excluded: list[dict[str, Any]],
) -> None:
    conflicting_splits = [
        split_name
        for split_name, hashes in split_hashes.items()
        if group_hashes & hashes
    ]
    if not conflicting_splits:
        chosen_split = preferred_split if preferred_split in allowed_splits else allowed_splits[0]
    elif len(conflicting_splits) == 1 and conflicting_splits[0] in allowed_splits:
        chosen_split = conflicting_splits[0]
    else:
        for record in group:
            excluded.append(_split_exclusion(record, "image_hash_cross_split_overlap"))
        return

    split[chosen_split].extend(group)
    split_hashes[chosen_split].update(group_hashes)


def split_records_with_exclusions(
    records: list[dict[str, Any]],
    *,
    seed: int,
    split_policy: SplitPolicy = "train_only",
) -> SplitResult:
    if split_policy not in ("train_val_test", "train_val_only", "train_only"):
        raise ValueError(f"Unsupported split policy: {split_policy}")
    split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    split_hashes: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    excluded: list[dict[str, Any]] = []
    if not records:
        return SplitResult(split=split, excluded=excluded)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source_split = record.get("metadata", {}).get("hover_reveal_split")
        if source_split == "test":
            key = _record_group_key(record)
        else:
            primary_hash = _record_primary_observation_hash(record)
            key = f"{_record_group_key(record)}:{primary_hash}:{record.get('id')}"
        grouped.setdefault(key, []).append(record)

    groups = _stable_shuffle(grouped.values(), seed=seed, salt="episode-split")
    official_test_groups = [group for group in groups if any(_record_is_official_test(record) for record in group)]
    non_test_groups = [group for group in groups if group not in official_test_groups]

    total = len(records)
    desired_val_count = (
        max(1, int(round(total * 0.10)))
        if total >= 3 and split_policy in {"train_val_test", "train_val_only"}
        else 0
    )
    desired_test_count = (
        max(1, int(round(total * 0.10)))
        if total >= 3 and split_policy == "train_val_test"
        else 0
    )

    for group in official_test_groups:
        if split_policy != "train_val_test":
            for record in group:
                excluded.append(_split_exclusion(record, "official_test_reserved_for_external_eval"))
            continue
        group_hashes = set().union(*(_record_observation_hashes(record) for record in group))
        _assign_group(
            split=split,
            split_hashes=split_hashes,
            group=group,
            group_hashes=group_hashes,
            preferred_split="test",
            allowed_splits=("test",),
            excluded=excluded,
        )

    for group in non_test_groups:
        if len(split["test"]) < desired_test_count:
            preferred_split = "test"
        elif len(split["val"]) < desired_val_count:
            preferred_split = "val"
        else:
            preferred_split = "train"
        group_hashes = set().union(*(_record_observation_hashes(record) for record in group))
        _assign_group(
            split=split,
            split_hashes=split_hashes,
            group=group,
            group_hashes=group_hashes,
            preferred_split=preferred_split,
            allowed_splits=("train", "val", "test"),
            excluded=excluded,
        )

    return SplitResult(
        split={
            name: sorted(items, key=lambda record: record["id"])
            for name, items in split.items()
        },
        excluded=excluded,
    )


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    split_policy: SplitPolicy = "train_only",
) -> dict[str, list[dict[str, Any]]]:
    return split_records_with_exclusions(records, seed=seed, split_policy=split_policy).split


def sanitize_trainable_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in record.get("metadata", {}).items()
        if key in TRAINABLE_METADATA_KEYS and value is not None
    }
    steps: list[dict[str, Any]] = []
    for step in record.get("steps", []):
        if step.get("type") == "observation":
            sanitized_step = {
                "type": "observation",
                "image_path": step["image_path"],
            }
            if "cursor_xy" in step:
                sanitized_step["cursor_xy"] = step["cursor_xy"]
            steps.append(sanitized_step)
            continue
        if step.get("type") == "action":
            sanitized_step = {
                "type": "action",
                "kind": step["kind"],
            }
            if step.get("kind") == "move_to":
                sanitized_step["x"] = step["x"]
                sanitized_step["y"] = step["y"]
                action_payload = {
                    "kind": "move_to",
                    "x": sanitized_step["x"],
                    "y": sanitized_step["y"],
                }
            else:
                action_payload = {"kind": step["kind"]}
            thought = extract_think_text(step)
            if thought is not None:
                set_canonical_assistant_response(
                    sanitized_step,
                    build_canonical_assistant_response(thought=thought, action=action_payload),
                )
            steps.append(sanitized_step)
    return {
        "id": record["id"],
        "source": record["source"],
        "instruction": record["instruction"],
        "metadata": metadata,
        "steps": steps,
    }


def _audit_record(record: dict[str, Any], *, split_name: str) -> dict[str, Any]:
    return {
        "id": record["id"],
        "split": split_name,
        "source": record["source"],
        "metadata": record.get("metadata", {}),
        "observation_sha256": sorted(_record_observation_hashes(record)),
        "action_kind_counts": dict(sorted(_count_actions([record]).items())),
    }


def _augment_record_variant(
    base_record: dict[str, Any],
    *,
    variant_index: int,
    success_rank: int,
    seed: int,
    thought_pools: dict[str, list[str]],
    scan_policy: ScanPolicy,
) -> dict[str, Any]:
    record = copy.deepcopy(base_record)
    metadata = record["metadata"]
    episode_id = str(metadata["hover_reveal_episode_id"])
    target_index = int(metadata["target_index"])
    centers = metadata["icon_centers_xy"]
    trajectory_type, scan_order = build_scan_order(
        centers=centers,
        target_index=target_index,
        record_index=variant_index,
        seed=seed,
        salt=f"{episode_id}:variant:{variant_index}",
        success_rank=success_rank,
        scan_policy=scan_policy,
    )
    base_id = record["id"]
    record["id"] = f"{base_id}_rank{success_rank:02d}_aug{variant_index:04d}"
    metadata["trajectory_type"] = f"{trajectory_type}_augmented"
    metadata["scan_order"] = scan_order
    metadata["success_rank"] = success_rank
    metadata["candidate_checks_before_success"] = success_rank
    metadata["wrong_hover_count"] = max(0, success_rank - 1)
    metadata["source_mode"] = "episode_oracle_augmented"
    metadata["click_events"] = []
    metadata["click_hit_status"] = "correct_icon"
    metadata["outcome"] = "success"

    reset_observation = next(
        step
        for step in record["steps"]
        if step.get("type") == "observation"
    )
    final_cursor = centers[target_index]
    steps: list[dict[str, Any]] = [reset_observation]
    action_index = 0
    cursor_xy: tuple[float, float] | None = None
    for scan_position, candidate_index in enumerate(scan_order, start=1):
        center_x, center_y = centers[candidate_index]
        pool = "move_hover" if scan_position == 1 else "wrong_icon"
        steps.append(
            _move_action_step(
                x=center_x,
                y=center_y,
                thought=_thought(
                    pool,
                    seed=seed,
                    record_id=record["id"],
                    step_index=action_index,
                    thought_pools=thought_pools,
                ),
                candidate_index=candidate_index,
            )
        )
        action_index += 1
        cursor_xy = (float(center_x), float(center_y))
        steps.append(dict(reset_observation, cursor_xy=[cursor_xy[0], cursor_xy[1]]))

    cursor_xy = (float(final_cursor[0]), float(final_cursor[1]))
    down = _button_action_step(
        "mouse_down",
        _thought(
            "correct_icon",
            seed=seed,
            record_id=record["id"],
            step_index=action_index,
            thought_pools=thought_pools,
        ),
    )
    steps.append(down)
    action_index += 1
    metadata["click_events"].append(
        _make_click_event(
            attempt_index=len(metadata["click_events"]) + 1,
            cursor_xy=cursor_xy,
            centers=centers,
            target_index=target_index,
            outcome="success",
        )
    )
    steps.append(dict(reset_observation, cursor_xy=[cursor_xy[0], cursor_xy[1]]))
    steps.append(
        _button_action_step(
            "mouse_up",
            _thought(
                "mouse_up",
                seed=seed,
                record_id=record["id"],
                step_index=action_index,
                thought_pools=thought_pools,
            ),
        )
    )
    steps.append(dict(reset_observation, cursor_xy=[cursor_xy[0], cursor_xy[1]]))
    record["steps"] = steps
    return record


def augment_records_to_target(
    records: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
    thought_pools: dict[str, list[str]],
    scan_policy: ScanPolicy = "mixed",
) -> list[dict[str, Any]]:
    if target_count <= len(records):
        return records[:target_count]
    if not records:
        return records
    augmented = list(records)
    base_candidates = _stable_shuffle(
        [
            record
            for record in records
            if record.get("metadata", {}).get("hover_reveal_split") != "test"
            and record.get("metadata", {}).get("source_mode") == "episode_oracle"
        ]
        or records,
        seed=seed,
        salt="augment-base",
    )
    variant_index = 0
    while len(augmented) < target_count:
        base = base_candidates[variant_index % len(base_candidates)]
        success_rank = 1 + (len(augmented) % ICON_COUNT)
        augmented.append(
            _augment_record_variant(
                base,
                variant_index=variant_index,
                success_rank=success_rank,
                seed=seed,
                thought_pools=thought_pools,
                scan_policy=scan_policy,
            )
        )
        variant_index += 1
    return augmented


def _flatten_split(split: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return sorted(
        (record for records in split.values() for record in records),
        key=lambda record: record["id"],
    )


def _split_episode_overlaps(split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    episodes = {
        name: {
            str(record.get("metadata", {}).get("hover_reveal_episode_id"))
            for record in records
            if record.get("metadata", {}).get("hover_reveal_episode_id") is not None
        }
        for name, records in split.items()
    }
    return {
        "train_val": len(episodes["train"] & episodes["val"]),
        "train_test": len(episodes["train"] & episodes["test"]),
        "val_test": len(episodes["val"] & episodes["test"]),
    }


def _split_metadata_value_overlaps(
    split: dict[str, list[dict[str, Any]]],
    key: str,
) -> dict[str, int]:
    values = {
        name: {
            str(record.get("metadata", {}).get(key))
            for record in records
            if record.get("metadata", {}).get(key) is not None
        }
        for name, records in split.items()
    }
    return {
        "train_val": len(values["train"] & values["val"]),
        "train_test": len(values["train"] & values["test"]),
        "val_test": len(values["val"] & values["test"]),
    }


def _split_image_hash_overlaps(split: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    hashes = {
        name: set().union(*(_record_observation_hashes(record) for record in records))
        if records
        else set()
        for name, records in split.items()
    }
    return {
        "train_val": len(hashes["train"] & hashes["val"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "val_test": len(hashes["val"] & hashes["test"]),
    }


def _summary_for_records(
    *,
    records: list[dict[str, Any]],
    split: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    dataset_root: Path,
    episode_refs: list[EpisodeRef],
    trace_refs: list[TraceRef],
    render_mode: RenderMode,
    requested_max_records: int | None,
    excluded_count: int,
    seed: int,
    scan_policy: ScanPolicy,
    split_policy: SplitPolicy,
    thought_pool_config: dict[str, Any],
    tooltip_html_policy: dict[str, Any],
    cursor_overlay_report: dict[str, Any],
) -> dict[str, Any]:
    target_counts = Counter(str(record["metadata"]["target_index"]) for record in records)
    success_rank_counts = Counter(str(record["metadata"].get("success_rank")) for record in records)
    trajectory_counts = Counter(record["metadata"].get("trajectory_type", "unknown") for record in records)
    source_mode_counts = Counter(record["metadata"].get("source_mode", "unknown") for record in records)
    wrong_hover_counts = Counter(str(record["metadata"].get("wrong_hover_count", 0)) for record in records)
    click_status_counts: Counter[str] = Counter()
    final_click_status_counts: Counter[str] = Counter()
    click_outcome_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    candidate_label_counts: Counter[str] = Counter()
    old_fixed_label_occurrences = 0
    for record in records:
        metadata = record["metadata"]
        target_label = metadata.get("target_label")
        if isinstance(target_label, str):
            target_label_counts[target_label] += 1
        for label in metadata.get("candidate_labels", []):
            if isinstance(label, str):
                candidate_label_counts[label] += 1
                if label in OLD_FIXED_TOOLTIP_LABELS:
                    old_fixed_label_occurrences += 1
        click_events = record["metadata"].get("click_events", [])
        for event in click_events:
            click_status_counts[str(event.get("click_hit_status"))] += 1
            click_outcome_counts[str(event.get("outcome"))] += 1
        if click_events:
            final_click_status_counts[str(click_events[-1].get("click_hit_status"))] += 1
    action_counts = _count_actions(records)
    sft_examples = sum(action_counts.values())
    record_count = len(records)
    episode_split_counts = Counter(ref.split for ref in episode_refs)
    return {
        "dataset": "ten_choice_captcha_sft",
        "base_environment": "hover_reveal",
        "output_dir": str(output_dir),
        "dataset_root": str(dataset_root),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "scan_policy": scan_policy,
        "split_policy": split_policy,
        "records": record_count,
        "requested_max_records": requested_max_records,
        "sft_examples": sft_examples,
        "candidate_count": ICON_COUNT,
        "target_index_counts": dict(sorted(target_counts.items())),
        "success_rank_definition": "candidate_checks_before_success; the ordinal candidate confirmation that reaches the correct icon",
        "success_rank_distribution": dict(sorted(success_rank_counts.items())),
        "trajectory_type_counts": dict(sorted(trajectory_counts.items())),
        "source_mode_counts": dict(sorted(source_mode_counts.items())),
        "wrong_hover_count_distribution": dict(sorted(wrong_hover_counts.items())),
        "click_hit_status_counts": dict(sorted(click_status_counts.items())),
        "final_click_hit_status_counts": dict(sorted(final_click_status_counts.items())),
        "click_outcome_counts": dict(sorted(click_outcome_counts.items())),
        "target_label_unique_count": len(target_label_counts),
        "target_label_top_counts": dict(target_label_counts.most_common(20)),
        "candidate_label_unique_count": len(candidate_label_counts),
        "candidate_label_top_counts": dict(candidate_label_counts.most_common(20)),
        "old_fixed_label_occurrences": old_fixed_label_occurrences,
        "action_kind_counts": dict(sorted(action_counts.items())),
        "split_counts": {name: len(items) for name, items in split.items()},
        "split_sft_examples": {
            name: sum(_count_actions(items).values())
            for name, items in split.items()
        },
        "trace_discovery": {
            "valid_success_trace_count": len(trace_refs),
            "used_trace_count": source_mode_counts.get("trace", 0),
        },
        "hover_reveal_reuse": {
            "available_episode_count": len(episode_refs),
            "available_episode_split_counts": dict(sorted(episode_split_counts.items())),
            "used_episode_oracle_count": source_mode_counts.get("episode_oracle", 0),
            "used_trace_count": source_mode_counts.get("trace", 0),
        },
        "render_mode": render_mode,
        "audit_path": str(output_dir / "audit.jsonl"),
        "excluded_count": excluded_count,
        "thought_pools": {
            "path": str(output_dir / "thought_pools.json"),
            "source": thought_pool_config.get("source"),
            "gpt": thought_pool_config.get("gpt"),
            "pool_sizes": thought_pool_config.get("pool_sizes"),
            "dedupe_strategy": thought_pool_config.get("dedupe_strategy"),
        },
        "quality_checks": {
            "split_episode_overlap_counts": _split_episode_overlaps(split),
            "split_source_episode_dir_overlap_counts": _split_metadata_value_overlaps(
                split,
                "source_episode_dir",
            ),
            "split_source_trace_path_overlap_counts": _split_metadata_value_overlaps(
                split,
                "source_trace_path",
            ),
            "split_image_sha256_overlap_counts": _split_image_hash_overlaps(split),
            "official_test_records_in_train": sum(
                1 for record in split["train"] if _record_is_official_test(record)
            ),
            "trainable_metadata_policy": {
                "records_jsonl_is_sanitized": True,
                "oracle_fields_written_to_audit_jsonl": True,
            },
        },
        "tooltip_no_fade_policy": tooltip_html_policy,
        "cursor_overlay": cursor_overlay_report,
        "training_strategy": (
            "single-click success only: each trainable record may hover over wrong candidates while scanning, "
            "then performs exactly one terminal click; that click must be correct_icon with outcome success. "
            "wrong_icon thoughts describe visible wrong labels and continued scanning, never clicking a wrong candidate."
        ),
        "old_dataset_review_constraints": {
            "old_split_path": "artifacts/datasets/new_paradigm_final_sft_20260506_split/",
            "not_reused_as_ten_choice": True,
            "old_split_ten_choice_records": 0,
            "legacy_ten_choice_paths": [
                "artifacts/datasets/ten_choice_captcha_sft_20260508/",
                "artifacts/datasets/ten_choice_captcha_sft_20260508_full/",
            ],
            "legacy_ten_choice_difference": (
                "2026-05-08 ten-choice exports allowed intermediate wrong_icon or miss diagnostic clicks "
                "followed by corrective actions; this single-click export keeps wrong candidates as hover-only "
                "exploration and allows exactly one terminal correct_icon success click in trainable records. "
                "Old screenshots may also show hover tooltip fade-out residue; this export rewrites episode HTML "
                "so non-hover tooltips disappear immediately."
            ),
            "known_old_risks_addressed": [
                "new records require metadata.task_type == ten_choice_captcha",
                "candidate_count is fixed to 10 with candidate labels and audit-only target oracle",
                "coordinates are validated against decoded image bounds",
                "image SHA overlap is checked across splits",
                "instruction email-like PII is sanitized",
            ],
        },
        "smoke_warning": (
            "This is a smoke-scale export for pipeline checks only and must not be used as formal SFT data."
            if record_count < FORMAL_TARGET_MIN_RECORDS
            else ""
        ),
        "thought_label_alignment": (
            "browser screenshots are captured after each move_to, so wrong/correct click thoughts follow a visible HoverReveal label"
            if render_mode == "browser"
            else "placeholder render mode is for offline tests; use browser mode for visible HoverReveal labels"
        ),
        "formal_target": {
            "min_records": FORMAL_TARGET_MIN_RECORDS,
            "max_records": FORMAL_TARGET_MAX_RECORDS,
            "shortfall_to_min": max(0, FORMAL_TARGET_MIN_RECORDS - record_count),
            "shortfall_to_requested_max": (
                max(0, requested_max_records - record_count)
                if requested_max_records is not None
                else 0
            ),
        },
    }


def _write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Ten-Choice Captcha SFT Dataset",
        "",
        "This dataset reuses or generates local HoverReveal episodes with ten candidate icons.",
        "",
        "Trainable manifests are sanitized for Qwen3-VL-8B SFT; oracle and provenance fields are audit-only.",
        "",
        f"- Records: {summary['records']}",
        f"- SFT examples: {summary['sft_examples']}",
        f"- Seed: `{summary.get('seed')}`",
        f"- Render mode: `{summary['render_mode']}`",
        f"- Source modes: `{summary['source_mode_counts']}`",
        f"- Target index counts: `{summary['target_index_counts']}`",
        f"- Target label unique count: {summary.get('target_label_unique_count')}",
        f"- Candidate label unique count: {summary.get('candidate_label_unique_count')}",
        f"- Old fixed label occurrences: {summary.get('old_fixed_label_occurrences')}",
        f"- Success-rank distribution: `{summary.get('success_rank_distribution', {})}`",
        f"- Click-hit status counts: `{summary.get('click_hit_status_counts', {})}`",
        f"- Cursor overlay: `{summary.get('cursor_overlay', {})}`",
        f"- Action kind counts: `{summary['action_kind_counts']}`",
        f"- Excluded/audit records: `{summary['excluded_count']}` excluded, audit at `audit.jsonl`",
        "",
        "`records.jsonl`, `train.jsonl`, `val.jsonl`, and `test.jsonl` are sanitized trainable records.",
        "Oracle fields such as `target_index`, `target_label`, `icon_centers_xy`, source paths, and action candidate indices are written only to `audit.jsonl` or `excluded.jsonl`.",
        "",
        "Success rank is `candidate_checks_before_success`: the ordinal candidate confirmation that reaches the correct icon.",
        "Every trainable record has exactly one `mouse_down` and one `mouse_up`, as the final two actions, and that single click is a successful `correct_icon` click.",
        "`wrong_icon` thoughts are used only after hovering a wrong candidate to continue scanning. Wrong or miss clicks are not present in trainable success records.",
        "Tooltip CSS is regenerated or rewritten so non-hover tooltips use `visibility: hidden`, `opacity: 0`, and `transition: none`; hover entry may fade in but mouse leave snaps hidden immediately.",
        "",
        "Compared with the 2026-05-08/2026-05-09 fixed-label ten-choice exports, this dataset removes fixed `This is the correct icon` / `This is the wrong icon` tooltip labels. Correctness is determined only by exact equality between the visible tooltip label and the instruction target label.",
        "Observation images with `cursor_xy` are post-processed with a visible mouse pointer whose hotspot is aligned at `cursor_xy`.",
        "",
        "The previous general GUI SFT split at `artifacts/datasets/new_paradigm_final_sft_20260506_split/` is not reused as ten-choice data. It has zero `ten_choice_captcha` records and lacks the required ten-candidate oracle/audit structure.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def detect_tooltip_regions_in_images(output_dir: Path) -> dict[str, Any]:
    return _detect_tooltip_regions_in_images(
        output_dir,
        settings=_VALIDATION_SETTINGS,
    )


def _tooltip_region_count(image_path: Path) -> int:
    return _validation_tooltip_region_count(
        image_path,
        settings=_VALIDATION_SETTINGS,
    )


def validate_cursor_overlays(output_dir: Path) -> dict[str, Any]:
    return _validate_cursor_overlays(
        output_dir,
        settings=_VALIDATION_SETTINGS,
    )


def run_tooltip_dom_hover_check(
    episode_refs: list[EpisodeRef],
    *,
    output_dir: Path,
    headless: bool = True,
) -> dict[str, Any]:
    return _run_tooltip_dom_hover_check(
        episode_refs,
        output_dir=output_dir,
        settings=_VALIDATION_SETTINGS,
        headless=headless,
    )


def validate_exported_dataset(
    output_dir: Path,
    *,
    episode_refs: list[EpisodeRef] | None = None,
    headless: bool = True,
) -> dict[str, Any]:
    return _validate_exported_dataset(
        output_dir,
        settings=_VALIDATION_SETTINGS,
        episode_refs=episode_refs,
        headless=headless,
    )



def export_dataset(
    *,
    output_dir: Path,
    dataset_root: Path | None = None,
    icon_dir: Path = DEFAULT_ICON_DIR,
    max_records: int | None = FORMAL_TARGET_MIN_RECORDS,
    seed: int = 42,
    render_mode: RenderMode = "browser",
    include_traces: bool = True,
    trace_root: Path | None = None,
    generate_if_missing: bool = True,
    headless: bool = True,
    scan_policy: ScanPolicy = "mixed",
    split_policy: SplitPolicy = "train_only",
) -> dict[str, Any]:
    dataset_root = dataset_root or source_episodes_root()
    trace_root = trace_root or default_trace_root()
    output_dir.mkdir(parents=True, exist_ok=True)
    thought_pool_config = build_thought_pool_config(
        seed=seed,
        gpt_model=os.environ.get("OPENAI_MODEL"),
    )
    thought_pools = thought_pool_config["pools"]
    (output_dir / "thought_pools.json").write_text(
        json.dumps(thought_pool_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    episode_refs = ensure_hover_reveal_episodes(
        dataset_root=dataset_root,
        icon_dir=icon_dir,
        min_records=max_records or FORMAL_TARGET_MIN_RECORDS,
        generate_if_missing=generate_if_missing,
        seed=seed,
    )
    tooltip_html_policy = ensure_latest_tooltip_html(episode_refs)
    if tooltip_html_policy["stale_after_rewrite_count"]:
        raise ValueError(
            "Failed to rewrite HoverReveal tooltip CSS for all episodes: "
            + ", ".join(tooltip_html_policy["stale_after_rewrite_examples"])
        )
    episode_refs = list_episode_refs(dataset_root)
    trace_discovery = (
        discover_trace_refs_with_exclusions(trace_root, episode_refs=episode_refs)
        if include_traces
        else TraceDiscoveryResult(trace_refs=[], excluded=[])
    )
    trace_refs = trace_discovery.trace_refs
    selectable_episode_refs = (
        episode_refs
        if split_policy == "train_val_test"
        else [ref for ref in episode_refs if ref.split != "test"]
    )
    selectable_trace_refs = (
        trace_refs
        if split_policy == "train_val_test"
        else [ref for ref in trace_refs if ref.source_split != "test"]
    )
    selected = select_items(
        selectable_episode_refs,
        selectable_trace_refs,
        max_records=max_records,
        seed=seed,
    )
    if max_records is not None and 10 <= max_records <= 20 and scan_policy == "mixed":
        selected = _rebalance_for_smoke_trajectory_coverage(
            selected,
            selectable_episode_refs,
            seed=seed,
        )
    if not selected:
        raise ValueError(
            f"No ten-choice HoverReveal episodes or valid traces found under {dataset_root}"
        )

    renderer: BrowserRenderer | PlaceholderRenderer
    if render_mode == "browser":
        renderer = BrowserRenderer(output_dir, headless=headless)
    elif render_mode == "placeholder":
        renderer = PlaceholderRenderer(output_dir)
    else:
        raise ValueError(f"Unsupported render mode: {render_mode}")

    full_records: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(selected):
            if item.source_mode == "trace":
                trace = item.ref
                assert isinstance(trace, TraceRef)
                record = build_trace_record(
                    trace=trace,
                    output_dir=output_dir,
                    seed=seed,
                    thought_pools=thought_pools,
                    success_rank=1 + (index % ICON_COUNT),
                )
                if record is not None:
                    full_records.append(record)
                continue
            episode = item.ref
            assert isinstance(episode, EpisodeRef)
            full_records.append(
                build_episode_record(
                    episode=episode,
                    output_dir=output_dir,
                    renderer=renderer,
                    record_index=index,
                    seed=seed,
                    success_rank=1 + (index % ICON_COUNT),
                    include_diagnostic_clicks=True,
                    thought_pools=thought_pools,
                    scan_policy=scan_policy,
                )
            )
    finally:
        renderer.close()

    full_records.sort(key=lambda record: record["id"])
    if max_records is not None and len(full_records) < max_records:
        full_records = augment_records_to_target(
            full_records,
            target_count=max_records,
            seed=seed,
            thought_pools=thought_pools,
            scan_policy=scan_policy,
        )
    full_records = full_records[:max_records] if max_records is not None else full_records
    full_records.sort(key=lambda record: record["id"])
    initial_cursor_overlay_report = apply_cursor_overlays_to_records(
        full_records,
        output_dir=output_dir,
    )
    split_result = split_records_with_exclusions(
        full_records,
        seed=seed,
        split_policy=split_policy,
    )
    split = split_result.split
    kept_full_records = _flatten_split(split)
    trainable_records = [sanitize_trainable_record(record) for record in kept_full_records]
    trainable_split = {
        name: [sanitize_trainable_record(record) for record in records]
        for name, records in split.items()
    }
    audit_records = [
        _audit_record(record, split_name=split_name)
        for split_name, records in split.items()
        for record in records
    ]
    excluded = trace_discovery.excluded + split_result.excluded
    _write_jsonl(trainable_records, output_dir / "records.jsonl")
    for split_name, split_records_value in trainable_split.items():
        _write_jsonl(split_records_value, output_dir / f"{split_name}.jsonl")
    _write_jsonl(audit_records, output_dir / "audit.jsonl")
    _write_jsonl(excluded, output_dir / "excluded.jsonl")
    summary = _summary_for_records(
        records=kept_full_records,
        split=split,
        output_dir=output_dir,
        dataset_root=dataset_root,
        episode_refs=episode_refs,
        trace_refs=trace_refs,
        render_mode=render_mode,
        requested_max_records=max_records,
        excluded_count=len(excluded),
        seed=seed,
        scan_policy=scan_policy,
        split_policy=split_policy,
        thought_pool_config=thought_pool_config,
        tooltip_html_policy=tooltip_html_policy,
        cursor_overlay_report=initial_cursor_overlay_report,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_dir, summary)
    _write_examples_zh(output_dir)
    validation_report = validate_exported_dataset(
        output_dir,
        episode_refs=episode_refs,
        headless=headless,
    )
    _write_cursor_overlay_report(output_dir, validation_report["cursor_overlay"])
    _write_validation_report(output_dir, validation_report)
    _write_full_sample_cases(output_dir)
    if validation_report["status"] != "passed":
        raise ValueError(
            "Ten-choice SFT export validation failed: "
            + "; ".join(validation_report["errors"][:5])
        )
    return summary


def default_output_dir() -> Path:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return sft_root() / f"ten_choice_captcha_sft_{date}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ten-choice CAPTCHA SFT records from local HoverReveal episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=source_episodes_root())
    parser.add_argument("--icon-dir", type=Path, default=DEFAULT_ICON_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--trace-root", type=Path, default=default_trace_root())
    parser.add_argument("--max-records", type=int, default=FORMAL_TARGET_MIN_RECORDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Generate 20 records for a quick coverage check.")
    parser.add_argument(
        "--render-mode",
        choices=("browser", "placeholder"),
        default="browser",
        help="Use browser for real HoverReveal screenshots; placeholder is intended for unit tests.",
    )
    parser.add_argument(
        "--scan-policy",
        choices=("mixed", "random", "left_to_right"),
        default="mixed",
        help=(
            "Candidate hover order strategy for oracle/augmented records. "
            "mixed preserves the historical five-way coverage; random uses a seeded random candidate order; "
            "left_to_right scans by increasing x, with increasing y as the tie-break."
        ),
    )
    parser.add_argument(
        "--split-policy",
        choices=("train_val_test", "train_val_only", "train_only"),
        default="train_only",
        help=(
            "How to write train/val/test JSONL splits. train_only keeps external fixed eval sets "
            "separate and writes empty val.jsonl/test.jsonl."
        ),
    )
    parser.add_argument("--no-include-traces", action="store_true")
    parser.add_argument("--no-generate-if-missing", action="store_true")
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    max_records = 20 if args.smoke else args.max_records
    summary = export_dataset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        icon_dir=args.icon_dir,
        max_records=max_records,
        seed=args.seed,
        render_mode=args.render_mode,
        include_traces=not args.no_include_traces,
        trace_root=args.trace_root,
        generate_if_missing=not args.no_generate_if_missing,
        headless=not args.headed,
        scan_policy=args.scan_policy,
        split_policy=args.split_policy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
