from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from ...data.jsonl_io import read_jsonl


@dataclass(frozen=True)
class TenChoiceValidationSettings:
    task_type: str
    icon_count: int
    supported_action_kinds: frozenset[str]
    oracle_field_keys: frozenset[str]
    old_fixed_tooltip_labels: frozenset[str]
    tooltip_rgb: tuple[int, int, int]
    tooltip_rgb_tolerance: int
    tooltip_min_region_pixels: int
    cursor_overlay_version: str
    viewport_width: int
    viewport_height: int


class EpisodeRefLike(Protocol):
    episode_id: str
    episode_dir: Path
    meta: dict[str, Any]


def count_actions(records: Iterable[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for step in record.get("steps", []):
            if step.get("type") == "action":
                counts[str(step.get("kind"))] += 1
    return counts


def record_observation_hashes(record: Mapping[str, Any]) -> set[str]:
    hashes: set[str] = set()
    for step in record.get("steps", []):
        if step.get("type") != "observation":
            continue
        image_path = step.get("image_path")
        if not isinstance(image_path, str):
            continue
        path = Path(image_path)
        if path.exists():
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def _iter_observation_paths(record: Mapping[str, Any]) -> Iterable[Path]:
    for step in record.get("steps", []):
        if step.get("type") == "observation" and isinstance(step.get("image_path"), str):
            yield Path(step["image_path"])


def _iter_action_steps(record: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for step in record.get("steps", []):
        if step.get("type") == "action":
            yield step


def _tooltip_region_count(
    image_path: Path,
    *,
    settings: TenChoiceValidationSettings,
) -> int:
    import numpy as np
    from scipy import ndimage

    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.int16)
    target = np.asarray(settings.tooltip_rgb, dtype=np.int16)
    mask = np.all(np.abs(array - target) <= settings.tooltip_rgb_tolerance, axis=2)
    if not mask.any():
        return 0
    labels, _label_count = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    count = 0
    for label_index, bounds in enumerate(objects, start=1):
        if bounds is None:
            continue
        region_pixels = int((labels[bounds] == label_index).sum())
        if region_pixels < settings.tooltip_min_region_pixels:
            continue
        y_slice, x_slice = bounds
        width = x_slice.stop - x_slice.start
        height = y_slice.stop - y_slice.start
        aspect_ratio = width / float(height) if height else 0.0
        if not (120 <= width <= 230 and 18 <= height <= 48 and aspect_ratio >= 2.2):
            continue
        count += 1
    return count


def detect_tooltip_regions_in_images(
    output_dir: Path,
    *,
    settings: TenChoiceValidationSettings,
) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    records = read_jsonl(output_dir / "records.jsonl")
    image_paths = sorted(
        {
            path.resolve()
            for record in records
            for path in _iter_observation_paths(record)
        }
    )
    distribution: Counter[str] = Counter()
    multi_tooltip_examples: list[dict[str, Any]] = []
    max_regions = 0
    workers = min(16, max(1, (os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        counts = list(
            executor.map(
                lambda image_path: _tooltip_region_count(
                    image_path,
                    settings=settings,
                ),
                image_paths,
            )
        )
    for path, count in zip(image_paths, counts):
        max_regions = max(max_regions, count)
        bucket = str(count) if count < 2 else "multi"
        distribution[bucket] += 1
        if count > 1 and len(multi_tooltip_examples) < 10:
            multi_tooltip_examples.append(
                {
                    "image_path": str(path),
                    "tooltip_region_count": count,
                }
            )
    return {
        "checked_image_count": len(image_paths),
        "tooltip_region_count_distribution": dict(sorted(distribution.items())),
        "zero_tooltip_image_count": distribution.get("0", 0),
        "one_tooltip_image_count": distribution.get("1", 0),
        "multi_tooltip_image_count": sum(
            value
            for key, value in distribution.items()
            if key not in {"0", "1"}
        ),
        "max_tooltip_regions": max_regions,
        "multi_tooltip_examples": multi_tooltip_examples,
        "detector": {
            "method": "connected components over dark tooltip background pixels",
            "rgb": list(settings.tooltip_rgb),
            "tolerance": settings.tooltip_rgb_tolerance,
            "min_region_pixels": settings.tooltip_min_region_pixels,
            "shape_filter": "120<=width<=230, 18<=height<=48, aspect_ratio>=2.2",
        },
    }


def _cursor_hotspot_present(image_path: Path, cursor_xy: list[float]) -> bool:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        x = int(round(float(cursor_xy[0])))
        y = int(round(float(cursor_xy[1])))
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                px = min(max(0, x + dx), rgb.width - 1)
                py = min(max(0, y + dy), rgb.height - 1)
                pixel = rgb.getpixel((px, py))
                if sum(pixel) <= 80:
                    return True
    return False


def validate_cursor_overlays(
    output_dir: Path,
    *,
    settings: TenChoiceValidationSettings,
) -> dict[str, Any]:
    records = read_jsonl(output_dir / "records.jsonl")
    cursor_observation_count = 0
    no_cursor_observation_count = 0
    cursor_overlay_missing_count = 0
    invalid_cursor_examples: list[dict[str, Any]] = []
    cursor_image_paths: set[str] = set()
    for record in records:
        for step_index, step in enumerate(record.get("steps", [])):
            if step.get("type") != "observation":
                continue
            cursor_xy = step.get("cursor_xy")
            if not isinstance(cursor_xy, list) or len(cursor_xy) < 2:
                no_cursor_observation_count += 1
                continue
            cursor_observation_count += 1
            image_path = Path(str(step.get("image_path", "")))
            cursor_image_paths.add(str(image_path))
            valid = image_path.exists() and _cursor_hotspot_present(
                image_path,
                cursor_xy,
            )
            if not valid:
                cursor_overlay_missing_count += 1
                if len(invalid_cursor_examples) < 10:
                    invalid_cursor_examples.append(
                        {
                            "record_id": record.get("id"),
                            "step_index": step_index,
                            "image_path": str(image_path),
                            "cursor_xy": cursor_xy,
                        }
                    )
    return {
        "version": settings.cursor_overlay_version,
        "cursor_observation_count": cursor_observation_count,
        "no_cursor_observation_count": no_cursor_observation_count,
        "cursor_overlay_image_count": len(cursor_image_paths),
        "cursor_overlay_missing_count": cursor_overlay_missing_count,
        "invalid_cursor_examples": invalid_cursor_examples,
        "hotspot_detector": (
            "requires at least one near-black pixel in the 3x3 neighborhood "
            "around cursor_xy"
        ),
    }


def _dom_visible_tooltips(page: Any) -> list[dict[str, Any]]:
    return page.evaluate(
        """() => Array.from(document.querySelectorAll('.icon-button')).flatMap((button, index) => {
          const style = window.getComputedStyle(button, '::after');
          const text = style.content.replace(/^"|"$/g, '');
          const visible = style.visibility !== 'hidden' && Number.parseFloat(style.opacity || '0') > 0.05;
          return visible ? [{ index, text, opacity: Number.parseFloat(style.opacity || '0'), visibility: style.visibility }] : [];
        })"""
    )


def run_tooltip_dom_hover_check(
    episode_refs: list[EpisodeRefLike],
    *,
    output_dir: Path,
    settings: TenChoiceValidationSettings,
    headless: bool = True,
) -> dict[str, Any]:
    candidate = next(
        (
            episode
            for episode in episode_refs
            if isinstance(episode.meta.get("labels"), list)
            and len(episode.meta["labels"]) == settings.icon_count
            and 0
            <= int(episode.meta.get("target_idx", -1))
            < settings.icon_count
            and any(
                label != episode.meta.get("target_label")
                for label in episode.meta["labels"]
            )
        ),
        None,
    )
    if candidate is None:
        return {
            "status": "skipped",
            "reason": (
                "no episode with one target label and at least one non-target label"
            ),
            "visible_tooltip_count_after_wrong_to_correct_hover": None,
            "visible_tooltip_texts_after_wrong_to_correct_hover": [],
        }

    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        if exc.name not in {"playwright", "playwright.sync_api"}:
            raise
        return {
            "status": "skipped",
            "reason": (
                "Playwright browser validation is unavailable; install the "
                "project browser dependencies and Chromium to run this check"
            ),
            "visible_tooltip_count_after_wrong_to_correct_hover": None,
            "visible_tooltip_texts_after_wrong_to_correct_hover": [],
        }

    labels = list(candidate.meta["labels"])
    centers = list(candidate.meta["icon_centers_xy"])
    target_label = str(candidate.meta["target_label"])
    wrong_index = next(
        index for index, label in enumerate(labels) if label != target_label
    )
    correct_index = next(
        index for index, label in enumerate(labels) if label == target_label
    )
    wrong_x, wrong_y = centers[wrong_index]
    correct_x, correct_y = centers[correct_index]
    html_path = candidate.episode_dir / "index.html"
    screenshot_dir = output_dir / "tooltip_dom_check"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = (
        screenshot_dir / f"{candidate.episode_id}_wrong_to_correct.jpg"
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=headless)
        try:
            page = browser.new_page(
                viewport={
                    "width": settings.viewport_width,
                    "height": settings.viewport_height,
                }
            )
            page.goto(
                html_path.resolve().as_uri(),
                wait_until="load",
                timeout=10_000,
            )
            page.mouse.move(float(wrong_x), float(wrong_y))
            page.wait_for_timeout(160)
            wrong_visible = _dom_visible_tooltips(page)
            page.mouse.move(float(correct_x), float(correct_y))
            page.wait_for_timeout(160)
            correct_visible = _dom_visible_tooltips(page)
            page.screenshot(
                path=str(screenshot_path),
                type="jpeg",
                quality=82,
                full_page=False,
            )
        finally:
            browser.close()

    return {
        "status": (
            "passed"
            if len(correct_visible) == 1
            and correct_visible[0]["text"] == target_label
            else "failed"
        ),
        "episode_id": candidate.episode_id,
        "wrong_candidate_index": wrong_index,
        "correct_candidate_index": correct_index,
        "wrong_label": labels[wrong_index],
        "target_label": target_label,
        "wrong_hover_visible_tooltips": wrong_visible,
        "visible_tooltips_after_wrong_to_correct_hover": correct_visible,
        "visible_tooltip_count_after_wrong_to_correct_hover": len(correct_visible),
        "visible_tooltip_texts_after_wrong_to_correct_hover": [
            item.get("text") for item in correct_visible
        ],
        "screenshot_path": str(screenshot_path.resolve()),
        "policy": (
            "after moving from wrong candidate to correct candidate, only the "
            "correct pseudo-tooltip may remain visible"
        ),
    }


def write_examples_zh(output_dir: Path, *, sample_count: int = 6) -> None:
    records = read_jsonl(output_dir / "records.jsonl")
    audit_by_id = {
        record["id"]: record
        for record in read_jsonl(output_dir / "audit.jsonl")
    }
    lines = [
        "# 十选一 CAPTCHA SFT 样例",
        "",
        (
            "以下样例来自本次导出的真实记录。训练清单不包含 oracle 字段；"
            "这里仅引用图片、动作摘要和审计中的非训练展示信息。"
        ),
        "",
    ]
    for record in records[:sample_count]:
        audit = audit_by_id.get(record["id"], {})
        metadata = audit.get("metadata", {})
        observations = list(_iter_observation_paths(record))
        actions = list(_iter_action_steps(record))
        current_observation: Path | None = observations[0] if observations else None
        correct_hover_image: Path | None = current_observation
        for step in record["steps"]:
            if step.get("type") == "observation" and isinstance(
                step.get("image_path"),
                str,
            ):
                current_observation = Path(step["image_path"])
                continue
            if step.get("type") == "action" and step.get("kind") == "mouse_down":
                correct_hover_image = current_observation
                break
        rel_image = (
            os.path.relpath(correct_hover_image, output_dir)
            if correct_hover_image is not None
            else ""
        )
        lines.extend(
            [
                f"## 样例 {record['id']}",
                "",
                f"- 图片: ![]({rel_image})",
                f"- instruction: {record['instruction']}",
                f"- split: {audit.get('split')}",
                f"- success_rank: {metadata.get('success_rank')}",
                f"- target_index: {metadata.get('target_index')}",
                f"- candidate_count: {metadata.get('candidate_count')}",
                f"- 动作数: {dict(sorted(count_actions([record]).items()))}",
                (
                    f"- 唯一点击状态: {metadata.get('click_hit_status')} / "
                    f"{metadata.get('outcome')}"
                ),
                (
                    "- 图片帧: 最终正确候选的悬停截图，视觉检测确认不含双 "
                    "tooltip 残影。"
                ),
                "",
                "动作摘要:",
            ]
        )
        for index, action in enumerate(actions[:12], start=1):
            if action.get("kind") == "move_to":
                lines.append(
                    f"{index}. move_to ({action.get('x')}, {action.get('y')})"
                )
            else:
                lines.append(f"{index}. {action.get('kind')}")
        if len(actions) > 12:
            lines.append(f"... 共 {len(actions)} 个动作")
        lines.append("")
    (output_dir / "examples_zh.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _image_as_base64_data_uri(path: Path) -> str:
    import base64

    mime = (
        "image/jpeg"
        if path.suffix.lower() in {".jpg", ".jpeg"}
        else "image/png"
    )
    return (
        f"data:{mime};base64,"
        + base64.b64encode(path.read_bytes()).decode("ascii")
    )


def write_full_sample_cases(
    output_dir: Path,
    *,
    sample_count: int = 3,
) -> None:
    records = read_jsonl(output_dir / "records.jsonl")[:sample_count]
    audit_by_id = {
        record["id"]: record
        for record in read_jsonl(output_dir / "audit.jsonl")
    }
    md_lines = [
        "# 十选一 CAPTCHA 完整真实样例",
        "",
        "以下样例展示完整 observation 图片序列；图片来自带鼠标 icon 后的新导出路径。",
        "",
    ]
    html_parts = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">',
        "<title>十选一 CAPTCHA 完整样例</title>",
        (
            "<style>body{font-family:Arial,sans-serif;margin:24px;color:#111827} "
            "img{max-width:100%;border:1px solid #d1d5db;margin:8px 0} "
            "section{margin-bottom:36px} "
            "code{background:#f3f4f6;padding:2px 4px}</style>"
        ),
        "</head><body><h1>十选一 CAPTCHA 完整真实样例</h1>",
    ]
    for case_index, record in enumerate(records, start=1):
        audit = audit_by_id.get(record["id"], {})
        metadata = audit.get("metadata", {})
        observations = [
            Path(step["image_path"])
            for step in record["steps"]
            if step.get("type") == "observation"
        ]
        md_lines.extend(
            [
                f"## 样例 {case_index}: {record['id']}",
                "",
                f"- instruction: {record['instruction']}",
                f"- split: {audit.get('split')}",
                f"- success_rank: {metadata.get('success_rank')}",
                f"- target_label: {metadata.get('target_label')}",
                f"- candidate_labels: {metadata.get('candidate_labels')}",
                "",
            ]
        )
        html_parts.extend(
            [
                (
                    f"<section><h2>样例 {case_index}: "
                    f"<code>{record['id']}</code></h2>"
                ),
                (
                    f"<p><strong>instruction:</strong> "
                    f"{record['instruction']}</p>"
                ),
                (
                    f"<p><strong>success_rank:</strong> "
                    f"{metadata.get('success_rank')} &nbsp; "
                    f"<strong>target_label:</strong> "
                    f"{metadata.get('target_label')}</p>"
                ),
            ]
        )
        for obs_index, image_path in enumerate(observations, start=1):
            rel = os.path.relpath(image_path, output_dir)
            md_lines.extend(
                [f"### Observation {obs_index}", f"![]({rel})", ""]
            )
            html_parts.append(
                f'<h3>Observation {obs_index}</h3>'
                f'<img src="{_image_as_base64_data_uri(image_path)}" '
                f'alt="observation {obs_index}">'
            )
        html_parts.append("</section>")
    html_parts.append("</body></html>")
    (output_dir / "sample_cases_zh_full.md").write_text(
        "\n".join(md_lines),
        encoding="utf-8",
    )
    (output_dir / "sample_cases_zh_full_embedded.html").write_text(
        "\n".join(html_parts),
        encoding="utf-8",
    )


def write_cursor_overlay_report(
    output_dir: Path,
    report: Mapping[str, Any],
) -> None:
    (output_dir / "cursor_overlay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Cursor Overlay Report",
        "",
        f"- Version: `{report.get('version')}`",
        (
            f"- Cursor observations: "
            f"{report.get('cursor_observation_count')}"
        ),
        (
            f"- Observations without cursor_xy: "
            f"{report.get('no_cursor_observation_count')}"
        ),
        (
            f"- Overlay images: "
            f"{report.get('overlay_image_count', report.get('cursor_overlay_image_count'))}"
        ),
        (
            f"- Missing source images: "
            f"{report.get('missing_source_image_count', 0)}"
        ),
        (
            f"- Missing/invalid cursor overlays: "
            f"{report.get('cursor_overlay_missing_count', 0)}"
        ),
        (
            f"- Hotspot alignment: "
            f"{report.get('hotspot_alignment', report.get('hotspot_detector'))}"
        ),
        f"- Reuse policy: {report.get('reuse_policy', '')}",
        "",
    ]
    (output_dir / "cursor_overlay_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def validate_exported_dataset(
    output_dir: Path,
    *,
    settings: TenChoiceValidationSettings,
    episode_refs: list[EpisodeRefLike] | None = None,
    headless: bool = True,
) -> dict[str, Any]:
    from ...train.qwen3_vl_sft import (
        format_sft_action_context,
        load_sft_examples,
    )

    errors: list[str] = []
    records = read_jsonl(output_dir / "records.jsonl")
    split_records_map = {
        name: read_jsonl(output_dir / f"{name}.jsonl")
        for name in ("train", "val", "test")
    }
    audit_records = read_jsonl(output_dir / "audit.jsonl")
    excluded_records = read_jsonl(output_dir / "excluded.jsonl")
    audit_by_id = {record["id"]: record for record in audit_records}

    if len(audit_by_id) != len(audit_records):
        errors.append("duplicate ids in audit.jsonl")
    if {record["id"] for record in records} != set(audit_by_id):
        errors.append("records.jsonl ids do not match audit.jsonl ids")
    split_id_sets = {
        name: {record["id"] for record in items}
        for name, items in split_records_map.items()
    }
    record_ids = {record["id"] for record in records}
    if record_ids != set().union(*split_id_sets.values()):
        errors.append("records.jsonl ids do not equal train/val/test union")

    for first, second in (
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ):
        overlap = split_id_sets[first] & split_id_sets[second]
        if overlap:
            errors.append(f"id overlap {first}/{second}: {len(overlap)}")

    trainable_blob = "\n".join(
        (output_dir / name).read_text(encoding="utf-8")
        for name in ("records.jsonl", "train.jsonl", "val.jsonl", "test.jsonl")
    )
    oracle_leaks = [
        key
        for key in settings.oracle_field_keys
        if f'"{key}"' in trainable_blob
    ]
    if oracle_leaks:
        errors.append(
            "oracle fields leaked into trainable manifests: "
            + ", ".join(oracle_leaks)
        )

    image_sizes: dict[str, tuple[int, int]] = {}
    all_action_counts = count_actions(records)
    click_status_counts: Counter[str] = Counter()
    final_click_status_counts: Counter[str] = Counter()
    click_outcome_counts: Counter[str] = Counter()
    success_rank_counts: Counter[str] = Counter()
    target_index_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    candidate_label_counts: Counter[str] = Counter()
    old_fixed_label_occurrences = 0

    for record in records:
        metadata = audit_by_id.get(record["id"], {}).get("metadata", {})
        if metadata.get("task_type") != settings.task_type:
            errors.append(
                f"{record['id']}: task_type is not {settings.task_type}"
            )
        if metadata.get("candidate_count") != settings.icon_count:
            errors.append(
                f"{record['id']}: candidate_count is not {settings.icon_count}"
            )
        target_index = metadata.get("target_index")
        if (
            not isinstance(target_index, int)
            or not 0 <= target_index < settings.icon_count
        ):
            errors.append(f"{record['id']}: target_index out of range")
            target_index = -1
        else:
            target_index_counts[str(target_index)] += 1
        centers = metadata.get("icon_centers_xy")
        if (
            not isinstance(centers, list)
            or len(centers) != settings.icon_count
            or len(
                {
                    tuple(center)
                    for center in centers
                    if isinstance(center, list) and len(center) == 2
                }
            )
            != settings.icon_count
        ):
            errors.append(f"{record['id']}: invalid or duplicate icon centers")
            centers = []
        labels = metadata.get("candidate_labels")
        if (
            not isinstance(labels, list)
            or len(labels) != settings.icon_count
        ):
            errors.append(f"{record['id']}: invalid candidate label mapping")
        elif (
            isinstance(target_index, int)
            and 0 <= target_index < settings.icon_count
        ):
            if len(set(labels)) != settings.icon_count:
                errors.append(
                    f"{record['id']}: candidate labels are not unique"
                )
            if labels[target_index] != metadata.get("target_label"):
                errors.append(f"{record['id']}: target label mapping mismatch")
            for label in labels:
                if isinstance(label, str):
                    candidate_label_counts[label] += 1
                    if label in settings.old_fixed_tooltip_labels:
                        old_fixed_label_occurrences += 1
            if isinstance(metadata.get("target_label"), str):
                target_label_counts[str(metadata["target_label"])] += 1
        scan_order = metadata.get("scan_order")
        success_rank = metadata.get("success_rank")
        if (
            not isinstance(success_rank, int)
            or not 1 <= success_rank <= settings.icon_count
        ):
            errors.append(f"{record['id']}: success_rank out of range")
        else:
            success_rank_counts[str(success_rank)] += 1
        expected_success_rank = success_rank
        if metadata.get("source_mode") == "trace":
            expected_success_rank = (
                len(scan_order)
                if isinstance(scan_order, list)
                else success_rank
            )
        if (
            not isinstance(scan_order, list)
            or len(scan_order) != expected_success_rank
            or len(set(scan_order)) != len(scan_order)
            or any(
                not isinstance(index, int)
                or not 0 <= index < settings.icon_count
                for index in scan_order
            )
            or (
                isinstance(target_index, int)
                and scan_order[-1] != target_index
            )
        ):
            errors.append(
                f"{record['id']}: invalid scan_order/success_rank alignment"
            )
        click_events = metadata.get("click_events")
        if not isinstance(click_events, list) or not click_events:
            errors.append(f"{record['id']}: missing click_events")
        else:
            for event in click_events:
                status = event.get("click_hit_status")
                if status not in {"correct_icon", "wrong_icon", "miss"}:
                    errors.append(
                        f"{record['id']}: invalid click_hit_status {status!r}"
                    )
                else:
                    click_status_counts[status] += 1
                click_outcome_counts[str(event.get("outcome"))] += 1
            final_status = click_events[-1].get("click_hit_status")
            final_click_status_counts[str(final_status)] += 1
            if len(click_events) != 1:
                errors.append(
                    f"{record['id']}: expected exactly one click_event"
                )
            only_event = click_events[-1]
            if (
                only_event.get("click_hit_status") != "correct_icon"
                or only_event.get("outcome") != "success"
            ):
                errors.append(
                    f"{record['id']}: single click is not successful correct_icon"
                )

        action_steps = list(_iter_action_steps(record))
        action_kinds = [step.get("kind") for step in action_steps]
        if action_kinds.count("mouse_down") != 1:
            errors.append(
                f"{record['id']}: expected exactly one mouse_down"
            )
        if action_kinds.count("mouse_up") != 1:
            errors.append(f"{record['id']}: expected exactly one mouse_up")
        if action_kinds[-2:] != ["mouse_down", "mouse_up"]:
            errors.append(f"{record['id']}: click is not terminal")

        previous_was_observation = False
        current_image_size: tuple[int, int] | None = None
        for step in record.get("steps", []):
            if step.get("type") == "observation":
                previous_was_observation = True
                path = Path(step.get("image_path", ""))
                if not path.exists():
                    errors.append(f"{record['id']}: missing image {path}")
                    current_image_size = None
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                    if (
                        width <= 0
                        or height <= 0
                        or (
                            Image.MAX_IMAGE_PIXELS is not None
                            and width * height > Image.MAX_IMAGE_PIXELS
                        )
                    ):
                        errors.append(
                            f"{record['id']}: invalid image dimensions {path}"
                        )
                    image_sizes[str(path)] = (width, height)
                    current_image_size = (width, height)
                except Exception as exc:
                    errors.append(
                        f"{record['id']}: image decode failed {path}: {exc}"
                    )
                    current_image_size = None
                continue
            if step.get("type") != "action":
                continue
            if not previous_was_observation:
                errors.append(
                    f"{record['id']}: action without preceding observation"
                )
            previous_was_observation = False
            if step.get("kind") not in settings.supported_action_kinds:
                errors.append(
                    f"{record['id']}: unsupported action kind "
                    f"{step.get('kind')}"
                )
            if step.get("kind") == "move_to":
                if current_image_size is None:
                    errors.append(
                        f"{record['id']}: move_to has unknown image size"
                    )
                else:
                    x_value, y_value = step.get("x"), step.get("y")
                    if (
                        not isinstance(x_value, (int, float))
                        or not isinstance(y_value, (int, float))
                        or not 0 <= float(x_value) < current_image_size[0]
                        or not 0 <= float(y_value) < current_image_size[1]
                    ):
                        errors.append(
                            f"{record['id']}: move_to coordinate out of bounds"
                        )

    split_episodes = {
        name: {
            audit_by_id[record["id"]]["metadata"].get(
                "hover_reveal_episode_id"
            )
            for record in items
            if record["id"] in audit_by_id
        }
        for name, items in split_records_map.items()
    }
    split_episode_overlaps = {
        "train_val": len(
            split_episodes["train"] & split_episodes["val"]
        ),
        "train_test": len(
            split_episodes["train"] & split_episodes["test"]
        ),
        "val_test": len(split_episodes["val"] & split_episodes["test"]),
    }
    for name, count in split_episode_overlaps.items():
        if count:
            errors.append(f"episode overlap {name}: {count}")

    split_hashes = {
        name: (
            set().union(
                *(record_observation_hashes(record) for record in items)
            )
            if items
            else set()
        )
        for name, items in split_records_map.items()
    }
    split_image_sha_overlaps = {
        "train_val": len(split_hashes["train"] & split_hashes["val"]),
        "train_test": len(split_hashes["train"] & split_hashes["test"]),
        "val_test": len(split_hashes["val"] & split_hashes["test"]),
    }
    for name, count in split_image_sha_overlaps.items():
        if count:
            errors.append(f"image sha overlap {name}: {count}")

    official_test_records_in_train_val = [
        record["id"]
        for split_name in ("train", "val")
        for record in split_records_map[split_name]
        if audit_by_id.get(record["id"], {})
        .get("metadata", {})
        .get("hover_reveal_split")
        == "test"
    ]
    if official_test_records_in_train_val:
        errors.append(
            "official test in train/val: "
            + str(len(official_test_records_in_train_val))
        )

    if not {
        str(index) for index in range(1, settings.icon_count + 1)
    }.issubset(success_rank_counts):
        errors.append("success_rank coverage does not include 1..10")
    if not {
        str(index) for index in range(settings.icon_count)
    }.issubset(target_index_counts):
        errors.append("target_index coverage does not include 0..9")
    bad_click_statuses = {
        status: count
        for status, count in click_status_counts.items()
        if status != "correct_icon" and count > 0
    }
    if bad_click_statuses:
        errors.append(
            "trainable click_hit_status contains non-correct statuses: "
            f"{bad_click_statuses}"
        )
    if click_status_counts["correct_icon"] != len(records):
        errors.append(
            "click_hit_status correct_icon count does not equal record count"
        )
    if click_outcome_counts != Counter({"success": len(records)}):
        errors.append(
            "click outcomes are not single success per record: "
            f"{dict(click_outcome_counts)}"
        )
    if old_fixed_label_occurrences:
        errors.append(
            "old fixed tooltip labels remain in candidate_labels: "
            f"{old_fixed_label_occurrences}"
        )

    loader_check: dict[str, Any] = {}
    prompt_action_pairs: set[str] = set()
    duplicate_prompt_action_pairs = 0
    for split_name in ("train", "val", "test"):
        examples = load_sft_examples([output_dir / f"{split_name}.jsonl"])
        loader_check[f"{split_name}_examples"] = len(examples)
        loader_check[f"{split_name}_action_counts"] = dict(
            sorted(
                Counter(
                    example.action["kind"] for example in examples
                ).items()
            )
        )
        for example in examples:
            signature = json.dumps(
                {
                    "instruction": example.instruction,
                    "image_path": str(example.image_path),
                    "context": format_sft_action_context(example.context),
                    "action": example.action,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if signature in prompt_action_pairs:
                duplicate_prompt_action_pairs += 1
            prompt_action_pairs.add(signature)

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "checked_records": len(records),
        "checked_audit_records": len(audit_records),
        "checked_excluded_records": len(excluded_records),
        "candidate_count_check": {
            "expected": settings.icon_count,
            "passed": not any(
                "candidate_count" in error for error in errors
            ),
        },
        "target_index_counts": dict(sorted(target_index_counts.items())),
        "success_rank_distribution": dict(
            sorted(success_rank_counts.items())
        ),
        "click_hit_status_counts": dict(sorted(click_status_counts.items())),
        "final_click_hit_status_counts": dict(
            sorted(final_click_status_counts.items())
        ),
        "click_outcome_counts": dict(sorted(click_outcome_counts.items())),
        "action_kind_counts": dict(sorted(all_action_counts.items())),
        "single_click_policy": {
            "exactly_one_mouse_down_per_record": not any(
                "expected exactly one mouse_down" in error
                for error in errors
            ),
            "exactly_one_mouse_up_per_record": not any(
                "expected exactly one mouse_up" in error
                for error in errors
            ),
            "click_actions_are_terminal": not any(
                "click is not terminal" in error for error in errors
            ),
            "only_correct_icon_success_clicks": not any(
                "trainable click_hit_status contains non-correct statuses"
                in error
                or "single click is not successful correct_icon" in error
                or "click outcomes are not single success per record" in error
                for error in errors
            ),
        },
        "split_counts": {
            name: len(items)
            for name, items in split_records_map.items()
        },
        "split_episode_overlap_counts": split_episode_overlaps,
        "split_image_sha256_overlap_counts": split_image_sha_overlaps,
        "official_test_records_in_train_val": (
            official_test_records_in_train_val
        ),
        "oracle_leak_check": {
            "trainable_oracle_leak_count": len(oracle_leaks),
            "leaked_fields": oracle_leaks,
        },
        "duplicate_prompt_action_pair_count": (
            duplicate_prompt_action_pairs
        ),
        "image_count": len(image_sizes),
        "label_diversity": {
            "target_label_unique_count": len(target_label_counts),
            "candidate_label_unique_count": len(candidate_label_counts),
            "target_label_top_counts": dict(
                target_label_counts.most_common(20)
            ),
            "candidate_label_top_counts": dict(
                candidate_label_counts.most_common(20)
            ),
            "old_fixed_label_occurrences": old_fixed_label_occurrences,
        },
        "loader_check": loader_check,
    }
    cursor_overlay_report = validate_cursor_overlays(
        output_dir,
        settings=settings,
    )
    report["cursor_overlay"] = cursor_overlay_report
    if cursor_overlay_report["cursor_overlay_missing_count"] != 0:
        errors.append(
            "cursor overlay validation found missing cursor icons: "
            + str(cursor_overlay_report["cursor_overlay_missing_count"])
        )
    tooltip_visual_detection = detect_tooltip_regions_in_images(
        output_dir,
        settings=settings,
    )
    report["tooltip_visual_detection"] = tooltip_visual_detection
    if tooltip_visual_detection["multi_tooltip_image_count"] != 0:
        errors.append(
            "visual tooltip detection found images with multiple tooltip "
            "regions: "
            + str(tooltip_visual_detection["multi_tooltip_image_count"])
        )

    if episode_refs:
        tooltip_dom_detection = run_tooltip_dom_hover_check(
            episode_refs,
            output_dir=output_dir,
            settings=settings,
            headless=headless,
        )
    else:
        tooltip_dom_detection = {
            "status": "skipped",
            "reason": "episode refs were not provided",
            "visible_tooltip_count_after_wrong_to_correct_hover": None,
            "visible_tooltip_texts_after_wrong_to_correct_hover": [],
        }
    report["tooltip_dom_detection"] = tooltip_dom_detection
    if tooltip_dom_detection.get("status") == "failed":
        errors.append("DOM tooltip wrong-to-correct hover check failed")
    if (
        tooltip_dom_detection.get("status") == "passed"
        and tooltip_dom_detection.get(
            "visible_tooltip_count_after_wrong_to_correct_hover"
        )
        != 1
    ):
        errors.append(
            "DOM tooltip visible count after wrong-to-correct hover is not 1"
        )
    report["status"] = "passed" if not errors else "failed"
    report["errors"] = errors
    return report


def write_validation_report(
    output_dir: Path,
    report: Mapping[str, Any],
) -> None:
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Ten-Choice CAPTCHA SFT Validation Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Checked records: {report['checked_records']}",
        f"- Split counts: `{report['split_counts']}`",
        (
            f"- Success-rank distribution: "
            f"`{report['success_rank_distribution']}`"
        ),
        (
            f"- Target-index distribution: "
            f"`{report['target_index_counts']}`"
        ),
        (
            f"- Click-hit status counts: "
            f"`{report['click_hit_status_counts']}`"
        ),
        (
            f"- Click outcome counts: "
            f"`{report.get('click_outcome_counts', {})}`"
        ),
        (
            f"- Single-click policy: "
            f"`{report.get('single_click_policy', {})}`"
        ),
        (
            f"- Oracle leak count: "
            f"{report['oracle_leak_check']['trainable_oracle_leak_count']}"
        ),
        (
            f"- Episode overlap counts: "
            f"`{report['split_episode_overlap_counts']}`"
        ),
        (
            f"- Image SHA overlap counts: "
            f"`{report['split_image_sha256_overlap_counts']}`"
        ),
        (
            f"- Duplicate expanded prompt-action pairs: "
            f"{report.get('duplicate_prompt_action_pair_count')}"
        ),
        f"- Label diversity: `{report.get('label_diversity', {})}`",
        f"- Cursor overlay: `{report.get('cursor_overlay', {})}`",
        (
            f"- Tooltip visual detection: "
            f"`{report.get('tooltip_visual_detection', {})}`"
        ),
        (
            f"- Tooltip DOM detection: "
            f"`{report.get('tooltip_dom_detection', {})}`"
        ),
        f"- Loader check: `{report['loader_check']}`",
        "",
    ]
    if report["errors"]:
        lines.append("## Errors")
        lines.extend(f"- {error}" for error in report["errors"])
    else:
        lines.append("No validation errors were found.")
    (output_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
