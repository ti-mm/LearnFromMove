from __future__ import annotations

import argparse
import html
import json
import math
import random
import shutil
import textwrap
from pathlib import Path
from typing import Any

from ...protocol_tracks import build_ten_choice_instruction
from .icon_assets import DEFAULT_ICON_DIR
from .paths import source_episodes_root

CORRECT_TOOLTIP = "This is the correct icon"
WRONG_TOOLTIP = "This is the wrong icon"
TOOLTIP_HIDE_POLICY = "instant_disappear_on_mouse_leave_viewport_safe"
OLD_FIXED_TOOLTIP_LABELS = {CORRECT_TOOLTIP, WRONG_TOOLTIP}

LABEL_POOL = [
    "This is the cat",
    "This is the dog",
    "This is the moon",
    "This is the river",
    "This is the golden crown",
    "This is the winter rose",
    "This is the silent bell",
    "This is the blue comet",
    "This is the paper lantern",
    "This is the hidden key",
    "This is the silver bridge",
    "This is the green valley",
    "This is the bright candle",
    "This is the red kite",
    "This is the quiet harbor",
    "This is the morning star",
    "This is the glass tower",
    "This is the copper coin",
    "This is the velvet mask",
    "This is the marble gate",
    "This is the old compass",
    "This is the cedar box",
    "This is the white lily",
    "This is the amber leaf",
    "This is the crystal cup",
    "This is the gentle rain",
    "This is the painted drum",
    "This is the lantern boat",
    "This is the velvet rope",
    "This is the cloud atlas",
    "This is the garden wall",
    "This is the brass horn",
    "This is the ruby ring",
    "This is the violet shell",
    "This is the orange sail",
    "This is the ivory tower",
    "This is the iron lantern",
    "This is the maple door",
    "This is the little crown",
    "This is the dawn feather",
    "This is the quiet lute",
    "This is the green orchard",
    "This is the blue ribbon",
    "This is the silver arrow",
    "This is the hidden garden",
    "This is the fair Verona",
    "This is the summer cloud",
    "This is the noble page",
    "This is the midnight bell",
    "This is the wandering star",
    "This is the gentle flame",
    "This is the royal cup",
    "This is the misty hill",
    "This is the bright falcon",
    "This is the willow shade",
    "This is the orchard gate",
    "This is the pearl button",
    "This is the quiet book",
    "This is the scarlet thread",
    "This is the blue fountain",
    "This is the silver mirror",
    "This is the winter lantern",
    "This is the golden apple",
    "This is the small anchor",
    "This is the mossy stone",
    "This is the velvet cloak",
    "This is the clear brook",
    "This is the bronze wheel",
    "This is the painted fan",
    "This is the twilight rose",
    "This is the ringing bell",
    "This is the cedar harp",
    "This is the crystal moon",
    "This is the quiet crown",
    "This is the paper moon",
    "This is the bright river",
    "This is the hidden rose",
    "This is the silver lantern",
    "This is the garden key",
    "This is the winter comet",
]

SUPPORTED_ICON_SUFFIXES = {
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}

VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 800
PAPER_VIEWPORT_WIDTH = 1280
PAPER_VIEWPORT_HEIGHT = 720
ICON_SIZE_PX = 92
ICON_COUNT = 10
CANVAS_MARGIN_PX = 24
INSTRUCTION_TOP_PX = 28
INSTRUCTION_HEIGHT_PX = 56
CANVAS_TOP_PX = 116
CANVAS_BOTTOM_MARGIN_PX = 24
ICON_PADDING_PX = 26
ICON_MIN_DISTANCE_PX = 118
TOOLTIP_EDGE_GUARD_PX = 170
TOOLTIP_TOP_GUARD_PX = 74
TOOLTIP_MAX_WIDTH_PX = 320
TOOLTIP_HORIZONTAL_PADDING_PX = 24
TOOLTIP_VERTICAL_PADDING_PX = 16
TOOLTIP_LINE_HEIGHT_PX = 16
TOOLTIP_OFFSET_PX = 12


def collect_icon_assets(icon_root: Path) -> list[Path]:
    """Return supported icon assets from *icon_root* in deterministic order."""
    if not icon_root.exists():
        raise FileNotFoundError(
            f"hover_reveal icon directory is missing: {icon_root}. "
            f"Add icon files under {icon_root} before generating episodes."
        )

    assets = sorted(
        path
        for path in icon_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_ICON_SUFFIXES
    )
    if not assets:
        supported = ", ".join(sorted(SUPPORTED_ICON_SUFFIXES))
        raise ValueError(
            f"No icon assets found in {icon_root}. "
            f"Supported extensions: {supported}."
        )
    return assets


def compute_canvas_rect(
    *,
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> dict[str, int]:
    """Return the large colored interaction area under the instruction."""
    if viewport_width <= 2 * CANVAS_MARGIN_PX:
        raise ValueError(f"viewport width is too small: {viewport_width}")
    if viewport_height <= CANVAS_TOP_PX + CANVAS_BOTTOM_MARGIN_PX:
        raise ValueError(f"viewport height is too small: {viewport_height}")
    return {
        "left": CANVAS_MARGIN_PX,
        "top": CANVAS_TOP_PX,
        "width": viewport_width - 2 * CANVAS_MARGIN_PX,
        "height": viewport_height - CANVAS_TOP_PX - CANVAS_BOTTOM_MARGIN_PX,
    }


def compute_instruction_rect(
    *,
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> dict[str, int]:
    del viewport_height
    if viewport_width <= 108:
        raise ValueError(f"viewport width is too small: {viewport_width}")
    return {
        "left": 54,
        "top": INSTRUCTION_TOP_PX,
        "width": viewport_width - 108,
        "height": INSTRUCTION_HEIGHT_PX,
    }


def _icon_rects_overlap(
    first: dict[str, int],
    second: dict[str, int],
    *,
    icon_size_px: int = ICON_SIZE_PX,
) -> bool:
    return not (
        first["left"] + icon_size_px <= second["left"]
        or second["left"] + icon_size_px <= first["left"]
        or first["top"] + icon_size_px <= second["top"]
        or second["top"] + icon_size_px <= first["top"]
    )


def compute_icon_positions(
    *,
    rng: random.Random,
    canvas_rect: dict[str, int],
    icon_count: int = ICON_COUNT,
    require_non_overlap: bool = False,
) -> list[dict[str, int]]:
    """Return spaced icon positions, optionally requiring disjoint rectangles."""
    positions: list[dict[str, int]] = []
    max_left = canvas_rect["width"] - ICON_SIZE_PX - ICON_PADDING_PX
    max_top = canvas_rect["height"] - ICON_SIZE_PX - ICON_PADDING_PX
    attempts = 0
    while len(positions) < icon_count and attempts < 5000:
        attempts += 1
        candidate = {
            "left": rng.randint(ICON_PADDING_PX, max_left),
            "top": rng.randint(ICON_PADDING_PX, max_top),
        }
        candidate_center_x = candidate["left"] + ICON_SIZE_PX / 2
        candidate_center_y = candidate["top"] + ICON_SIZE_PX / 2
        overlaps = False
        for placed in positions:
            placed_center_x = placed["left"] + ICON_SIZE_PX / 2
            placed_center_y = placed["top"] + ICON_SIZE_PX / 2
            if (
                require_non_overlap and _icon_rects_overlap(candidate, placed)
            ) or math.dist(
                (candidate_center_x, candidate_center_y),
                (placed_center_x, placed_center_y),
            ) < ICON_MIN_DISTANCE_PX:
                overlaps = True
                break
        if not overlaps:
            positions.append(candidate)

    if len(positions) != icon_count:
        columns = 5
        rows = math.ceil(icon_count / columns)
        horizontal_gap = max(
            ICON_PADDING_PX,
            (canvas_rect["width"] - columns * ICON_SIZE_PX) // (columns + 1),
        )
        vertical_gap = max(
            ICON_PADDING_PX,
            (canvas_rect["height"] - rows * ICON_SIZE_PX) // (rows + 1),
        )
        positions = []
        for index in range(icon_count):
            row = index // columns
            column = index % columns
            positions.append(
                {
                    "left": horizontal_gap + column * (ICON_SIZE_PX + horizontal_gap),
                    "top": vertical_gap + row * (ICON_SIZE_PX + vertical_gap),
                }
            )
    return positions


def icon_centers(
    positions: list[dict[str, int]],
    *,
    canvas_rect: dict[str, int],
) -> list[list[int]]:
    return [
        [
            canvas_rect["left"] + position["left"] + ICON_SIZE_PX // 2,
            canvas_rect["top"] + position["top"] + ICON_SIZE_PX // 2,
        ]
        for position in positions
    ]


def build_oracle_sequence(target_idx: int, centers: list[list[int]]) -> list[dict[str, Any]]:
    """Hover over each icon, return to target, then click and finish."""
    steps: list[dict[str, Any]] = []
    for center_x, center_y in centers:
        steps.append({"kind": "move_to", "x": center_x, "y": center_y})

    target_x, target_y = centers[target_idx]
    steps.append({"kind": "move_to", "x": target_x, "y": target_y})
    steps.append({"kind": "left_click"})
    return steps


def build_instruction(target_label: str) -> str:
    return build_ten_choice_instruction(target_label)


def sample_candidate_labels(rng: random.Random, *, count: int = ICON_COUNT) -> list[str]:
    pool = [
        label
        for label in dict.fromkeys(LABEL_POOL)
        if label not in OLD_FIXED_TOOLTIP_LABELS and len(label) <= 34
    ]
    if len(pool) < count:
        raise ValueError(f"Label pool has only {len(pool)} usable labels; need {count}")
    return rng.sample(pool, count)


def episode_html_has_instant_tooltip_hide(html: str) -> bool:
    """Return whether HoverReveal tooltip CSS uses the latest safe policy."""
    hover_marker = ".icon-button:hover::after"
    return (
        ".icon-button::after" in html
        and hover_marker in html
        and "data-tooltip-x" in html
        and "data-tooltip-y" in html
        and '.icon-button[data-tooltip-y="below"]::after' in html
        and '.icon-button[data-tooltip-x="left"]::after' in html
        and '.icon-button[data-tooltip-x="right"]::after' in html
        and "width: max-content;" in html
        and "min-width: min(160px, calc(100vw - 32px));" in html
        and "max-width: min(320px, calc(100vw - 32px));" in html
        and "line-height: 18px;" in html
        and "overflow-wrap: anywhere;" in html
        and "opacity: 0;" in html
        and "visibility: hidden;" in html
        and "transition: none;" in html
        and "opacity: 1;" in html
        and "visibility: visible;" in html
        and "transition: opacity 0.12s ease;" in html
        and html.index("transition: none;") < html.index(hover_marker)
        and html.index("visibility: hidden;") < html.index(hover_marker)
    )


def tooltip_anchor_for_position(
    position: dict[str, int],
    *,
    canvas_rect: dict[str, int] | None = None,
) -> dict[str, str]:
    """Return data attributes that keep the tooltip inside the page/canvas."""
    canvas_rect = canvas_rect or compute_canvas_rect()
    center_x = int(position["left"]) + ICON_SIZE_PX / 2
    if center_x < TOOLTIP_EDGE_GUARD_PX:
        tooltip_x = "left"
    elif center_x > canvas_rect["width"] - TOOLTIP_EDGE_GUARD_PX:
        tooltip_x = "right"
    else:
        tooltip_x = "center"
    tooltip_y = "below" if int(position["top"]) < TOOLTIP_TOP_GUARD_PX else "above"
    return {"x": tooltip_x, "y": tooltip_y}


def tooltip_rect_for_position(
    *,
    label: str,
    position: dict[str, int],
    canvas_rect: dict[str, int],
) -> dict[str, int]:
    """Return a conservative absolute tooltip rectangle for bounds checks."""
    estimated_text_width = max(1, len(label)) * 8
    inner_width = TOOLTIP_MAX_WIDTH_PX - TOOLTIP_HORIZONTAL_PADDING_PX
    line_count = max(1, math.ceil(estimated_text_width / inner_width))
    width = min(
        TOOLTIP_MAX_WIDTH_PX,
        estimated_text_width + TOOLTIP_HORIZONTAL_PADDING_PX,
    )
    height = TOOLTIP_VERTICAL_PADDING_PX + line_count * TOOLTIP_LINE_HEIGHT_PX
    anchor = tooltip_anchor_for_position(position, canvas_rect=canvas_rect)
    if anchor["x"] == "left":
        local_left = int(position["left"])
    elif anchor["x"] == "right":
        local_left = int(position["left"]) + ICON_SIZE_PX - width
    else:
        local_left = int(round(position["left"] + ICON_SIZE_PX / 2 - width / 2))
    if anchor["y"] == "below":
        local_top = int(position["top"]) + ICON_SIZE_PX + TOOLTIP_OFFSET_PX
    else:
        local_top = int(position["top"]) - TOOLTIP_OFFSET_PX - height
    return {
        "left": canvas_rect["left"] + local_left,
        "top": canvas_rect["top"] + local_top,
        "width": width,
        "height": height,
    }


def _rect_is_inside(inner: dict[str, int], outer: dict[str, int]) -> bool:
    return (
        inner["left"] >= outer["left"]
        and inner["top"] >= outer["top"]
        and inner["left"] + inner["width"] <= outer["left"] + outer["width"]
        and inner["top"] + inner["height"] <= outer["top"] + outer["height"]
    )


def validate_hover_reveal_episode_geometry(
    meta: dict[str, Any],
    *,
    expected_viewport: tuple[int, int] | None = None,
) -> list[str]:
    """Validate all visible episode geometry against its declared viewport."""
    errors: list[str] = []
    raw_viewport = meta.get("viewport")
    if not (
        isinstance(raw_viewport, list)
        and len(raw_viewport) == 2
        and all(isinstance(value, int) and value > 0 for value in raw_viewport)
    ):
        return ["invalid viewport"]
    viewport = (int(raw_viewport[0]), int(raw_viewport[1]))
    if expected_viewport is not None and viewport != expected_viewport:
        errors.append(f"viewport mismatch: {viewport} != {expected_viewport}")
    viewport_rect = {
        "left": 0,
        "top": 0,
        "width": viewport[0],
        "height": viewport[1],
    }
    canvas_rect = meta.get("canvas_rect")
    instruction_rect = meta.get("instruction_rect")
    if not isinstance(canvas_rect, dict) or not all(
        isinstance(canvas_rect.get(key), int)
        for key in ("left", "top", "width", "height")
    ):
        errors.append("invalid canvas_rect")
        return errors
    if not isinstance(instruction_rect, dict) or not all(
        isinstance(instruction_rect.get(key), int)
        for key in ("left", "top", "width", "height")
    ):
        errors.append("invalid instruction_rect")
    elif not _rect_is_inside(instruction_rect, viewport_rect):
        errors.append("instruction_rect exceeds viewport")
    if not _rect_is_inside(canvas_rect, viewport_rect):
        errors.append("canvas_rect exceeds viewport")

    labels = meta.get("labels")
    positions_xy = meta.get("icon_positions_xy")
    centers_xy = meta.get("icon_centers_xy")
    icon_size_px = meta.get("icon_size_px", ICON_SIZE_PX)
    if not isinstance(labels, list) or len(labels) != ICON_COUNT:
        errors.append("invalid labels")
        return errors
    if not isinstance(icon_size_px, int) or icon_size_px <= 0:
        errors.append("invalid icon_size_px")
        return errors
    if not isinstance(positions_xy, list) or len(positions_xy) != ICON_COUNT:
        errors.append("invalid icon_positions_xy")
        return errors
    if not isinstance(centers_xy, list) or len(centers_xy) != ICON_COUNT:
        errors.append("invalid icon_centers_xy")
        return errors

    absolute_rects: list[dict[str, int]] = []
    for index, raw_position in enumerate(positions_xy):
        if not (
            isinstance(raw_position, list)
            and len(raw_position) == 2
            and all(isinstance(value, int) for value in raw_position)
        ):
            errors.append(f"button {index}: invalid position")
            continue
        absolute_rect = {
            "left": int(raw_position[0]),
            "top": int(raw_position[1]),
            "width": icon_size_px,
            "height": icon_size_px,
        }
        local_position = {
            "left": absolute_rect["left"] - canvas_rect["left"],
            "top": absolute_rect["top"] - canvas_rect["top"],
        }
        absolute_rects.append(absolute_rect)
        if not _rect_is_inside(absolute_rect, canvas_rect):
            errors.append(f"button {index}: exceeds canvas")
        if not _rect_is_inside(absolute_rect, viewport_rect):
            errors.append(f"button {index}: exceeds viewport")
        center = centers_xy[index]
        expected_center = [
            absolute_rect["left"] + icon_size_px // 2,
            absolute_rect["top"] + icon_size_px // 2,
        ]
        if center != expected_center:
            errors.append(
                f"center {index}: {center!r} != expected {expected_center!r}"
            )
        tooltip_rect = tooltip_rect_for_position(
            label=str(labels[index]),
            position=local_position,
            canvas_rect=canvas_rect,
        )
        if not _rect_is_inside(tooltip_rect, canvas_rect):
            errors.append(f"tooltip {index}: exceeds canvas")
        if not _rect_is_inside(tooltip_rect, viewport_rect):
            errors.append(f"tooltip {index}: exceeds viewport")

    for first_index, first in enumerate(absolute_rects):
        for second_index in range(first_index + 1, len(absolute_rects)):
            second = absolute_rects[second_index]
            if _icon_rects_overlap(first, second, icon_size_px=icon_size_px):
                errors.append(
                    f"buttons {first_index} and {second_index}: overlap"
                )
    instruction = meta.get("instruction")
    target_label = meta.get("target_label")
    if not isinstance(instruction, str) or not instruction.strip():
        errors.append("invalid instruction")
    elif not isinstance(target_label, str) or target_label not in instruction:
        errors.append("instruction does not contain target_label")
    return errors


def _positions_from_meta(meta: dict[str, Any]) -> list[dict[str, int]]:
    canvas_rect = meta.get("canvas_rect") or compute_canvas_rect()
    if isinstance(meta.get("icon_positions_xy"), list) and len(meta["icon_positions_xy"]) == ICON_COUNT:
        return [
            {
                "left": int(round(float(position[0]) - float(canvas_rect["left"]))),
                "top": int(round(float(position[1]) - float(canvas_rect["top"]))),
            }
            for position in meta["icon_positions_xy"]
        ]
    centers = meta.get("icon_centers_xy")
    if isinstance(centers, list) and len(centers) == ICON_COUNT:
        return [
            {
                "left": int(round(float(center[0]) - float(canvas_rect["left"]) - ICON_SIZE_PX / 2)),
                "top": int(round(float(center[1]) - float(canvas_rect["top"]) - ICON_SIZE_PX / 2)),
            }
            for center in centers
        ]
    raise ValueError("Cannot reconstruct HoverReveal icon positions from meta.json")


def rewrite_episode_html_with_latest_tooltip_css(episode_dir: Path) -> bool:
    """Rewrite an episode index.html from meta.json when it uses stale tooltip CSS."""
    index_path = episode_dir / "index.html"
    if index_path.exists() and episode_html_has_instant_tooltip_hide(index_path.read_text(encoding="utf-8")):
        return False
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    raw_viewport = meta.get("viewport", [VIEWPORT_WIDTH, VIEWPORT_HEIGHT])
    viewport_width = int(raw_viewport[0])
    viewport_height = int(raw_viewport[1])
    html = build_html(
        labels=list(meta["labels"]),
        target_label=str(meta.get("target_label", meta["labels"][int(meta.get("target_idx", 0))])),
        icon_asset=str(meta.get("icon_asset", "assets/icon.svg")),
        positions=_positions_from_meta(meta),
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    meta["instruction"] = build_instruction(str(meta.get("target_label", meta["labels"][int(meta.get("target_idx", 0))])))
    index_path.write_text(html, encoding="utf-8")
    meta["version"] = max(int(meta.get("version", 1)), 3)
    meta["tooltip_position"] = "adaptive"
    meta["tooltip_hide_policy"] = TOOLTIP_HIDE_POLICY
    (episode_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return True


def _icon_button_html(
    *,
    index: int,
    label: str,
    position: dict[str, int],
    icon_asset: str,
    canvas_rect: dict[str, int],
) -> str:
    anchor = tooltip_anchor_for_position(position, canvas_rect=canvas_rect)
    return (
        f'      <button class="icon-button" data-label="{html.escape(label, quote=True)}" '
        f'data-tooltip-x="{anchor["x"]}" data-tooltip-y="{anchor["y"]}" '
        f'style="left:{position["left"]}px; top:{position["top"]}px;" '
        f'onclick="selectIcon({index})">'
        f'<img src="{html.escape(icon_asset, quote=True)}" alt="" draggable="false">'
        f"</button>"
    )


def build_html(
    *,
    labels: list[str],
    target_label: str,
    icon_asset: str,
    positions: list[dict[str, int]],
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> str:
    """Return the episode HTML with multiple visually identical icons."""
    canvas_rect = compute_canvas_rect(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    instruction_rect = compute_instruction_rect(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    icons_html = "\n".join(
        _icon_button_html(
            index=index,
            label=label,
            position=position,
            icon_asset=icon_asset,
            canvas_rect=canvas_rect,
        )
        for index, (label, position) in enumerate(zip(labels, positions))
    )
    target_label_html = html.escape(target_label)
    target_label_js = json.dumps(target_label)
    return textwrap.dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width={viewport_width}, initial-scale=1.0">
          <title>Hover Reveal</title>
          <style>
            * {{ box-sizing: border-box; }}
            body {{
              margin: 0;
              min-height: 100vh;
              overflow: auto;
              display: flex;
              justify-content: center;
              align-items: flex-start;
              font-family: Arial, sans-serif;
              background: #ffffff;
              color: #0f172a;
            }}
            .page {{
              position: relative;
              width: min(100vw, {viewport_width}px);
              height: {viewport_height}px;
              flex: 0 0 auto;
            }}
            .instruction {{
              position: absolute;
              left: {instruction_rect["left"]}px;
              top: {instruction_rect["top"]}px;
              width: {instruction_rect["width"]}px;
              text-align: center;
              font-size: 29px;
              line-height: {instruction_rect["height"]}px;
              font-weight: 700;
              white-space: nowrap;
            }}
            .instruction strong {{
              color: #1d4ed8;
            }}
            .canvas {{
              position: absolute;
              left: {canvas_rect["left"]}px;
              top: {canvas_rect["top"]}px;
              width: {canvas_rect["width"]}px;
              height: {canvas_rect["height"]}px;
              border-radius: 28px;
              border: 2px solid rgba(15, 23, 42, 0.12);
              background:
                linear-gradient(135deg, #dff7ea 0%, #d6ecff 52%, #efe0ff 100%);
              box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.45),
                0 12px 30px rgba(15, 23, 42, 0.08);
              overflow: hidden;
            }}
            .icon-button {{
              position: absolute;
              width: {ICON_SIZE_PX}px;
              height: {ICON_SIZE_PX}px;
              border: 0;
              padding: 0;
              border-radius: 0;
              background: transparent;
              cursor: pointer;
              display: flex;
              align-items: center;
              justify-content: center;
              box-shadow: none;
              transition: transform 0.15s ease;
            }}
            .icon-button:hover {{
              transform: translateY(-2px) scale(1.04);
            }}
            .icon-button img {{
              width: 100%;
              height: 100%;
              object-fit: contain;
              pointer-events: none;
              user-select: none;
              filter: drop-shadow(0 10px 16px rgba(15, 23, 42, 0.16));
            }}
            .icon-button::after {{
              content: attr(data-label);
              position: absolute;
              left: 50%;
              bottom: calc(100% + 12px);
              transform: translateX(-50%);
              padding: 8px 12px;
              border-radius: 999px;
              background: #0f172a;
              color: #ffffff;
              font-size: 13px;
              font-weight: 600;
              line-height: 18px;
              width: max-content;
              min-width: min(160px, calc(100vw - 32px));
              max-width: min(320px, calc(100vw - 32px));
              white-space: normal;
              overflow-wrap: anywhere;
              text-align: center;
              opacity: 0;
              visibility: hidden;
              pointer-events: none;
              transition: none;
              z-index: 20;
            }}
            .icon-button[data-tooltip-y="below"]::after {{
              top: calc(100% + 12px);
              bottom: auto;
            }}
            .icon-button[data-tooltip-x="left"]::after {{
              left: 0;
              transform: none;
            }}
            .icon-button[data-tooltip-x="right"]::after {{
              left: auto;
              right: 0;
              transform: none;
            }}
            .icon-button:hover::after {{
              opacity: 1;
              visibility: visible;
              transition: opacity 0.12s ease;
            }}
            .icon-button.selected {{
              box-shadow:
                0 18px 30px rgba(15, 23, 42, 0.16),
                inset 0 0 0 3px #f97316;
            }}
            .feedback {{
              position: absolute;
              left: 50%;
              bottom: 18px;
              transform: translateX(-50%);
              min-width: 140px;
              padding: 10px 18px;
              border-radius: 999px;
              background: rgba(15, 23, 42, 0.08);
              color: #334155;
              font-size: 16px;
              font-weight: 700;
              text-align: center;
            }}
            .feedback.correct {{
              background: #dcfce7;
              color: #166534;
            }}
            .feedback.wrong {{
              background: #fee2e2;
              color: #b91c1c;
            }}
          </style>
        </head>
        <body>
          <main class="page">
            <div class="instruction">
              Click the icon that displays "<strong>{target_label_html}</strong>".
            </div>
            <section class="canvas">
        {icons_html}
            </section>
            <div id="feedback" class="feedback">Choose one icon</div>
            <div id="result" style="display:none"></div>
          </main>
          <script>
            let answered = false;
            const icons = Array.from(document.querySelectorAll(".icon-button"));
            const feedback = document.getElementById("feedback");

            function selectIcon(index) {{
              if (answered) {{
                return;
              }}
              icons.forEach((icon, iconIndex) => {{
                icon.classList.toggle("selected", iconIndex === index);
              }});
              const label = icons[index].dataset.label;
              answered = true;
              if (label === {target_label_js}) {{
                feedback.textContent = "Correct";
                feedback.className = 'feedback correct';
              }} else {{
                feedback.textContent = "Wrong";
                feedback.className = 'feedback wrong';
              }}
              document.getElementById("result").textContent =
                JSON.stringify({{ clicked: index, label: label }});
            }}
          </script>
        </body>
        </html>
        """
    )


def write_episode(
    *,
    episode_id: str,
    split: str,
    rng: random.Random,
    icon_assets: list[Path],
    split_dir: Path,
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> None:
    """Generate one episode directory."""
    episode_dir = split_dir / episode_id
    assets_dir = episode_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    labels = sample_candidate_labels(rng, count=ICON_COUNT)
    target_idx = rng.randrange(ICON_COUNT)
    target_label = labels[target_idx]
    canvas_rect = compute_canvas_rect(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    instruction_rect = compute_instruction_rect(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    positions = compute_icon_positions(
        rng=rng,
        canvas_rect=canvas_rect,
        icon_count=ICON_COUNT,
    )
    centers = icon_centers(positions, canvas_rect=canvas_rect)
    absolute_positions = [
        [
            canvas_rect["left"] + position["left"],
            canvas_rect["top"] + position["top"],
        ]
        for position in positions
    ]

    source_icon = rng.choice(icon_assets)
    copied_name = f"icon{source_icon.suffix.lower()}"
    icon_asset = f"assets/{copied_name}"
    shutil.copy2(source_icon, assets_dir / copied_name)

    html = build_html(
        labels=labels,
        target_label=target_label,
        icon_asset=icon_asset,
        positions=positions,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )
    meta = {
        "episode_id": episode_id,
        "version": 3,
        "split": split,
        "instruction": build_instruction(target_label),
        "target_label": target_label,
        "target_idx": target_idx,
        "labels": labels,
        "layout": "horizontal",
        "icon_size_px": ICON_SIZE_PX,
        "gap_px": None,
        "tooltip_position": "adaptive",
        "tooltip_hide_policy": TOOLTIP_HIDE_POLICY,
        "icon_centers_xy": centers,
        "icon_positions_xy": absolute_positions,
        "canvas_rect": canvas_rect,
        "instruction_rect": instruction_rect,
        "icon_asset": icon_asset,
        "icon_source_name": source_icon.name,
        "viewport": [viewport_width, viewport_height],
        "coupling_level": "high",
        "oracle_primitive_sequence": build_oracle_sequence(target_idx, centers),
    }

    (episode_dir / "index.html").write_text(html, encoding="utf-8")
    (episode_dir / "meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def generate_dataset(
    *,
    output_dir: Path,
    icon_root: Path,
    train_count: int = 1000,
    test_count: int = 500,
    seed: int = 42,
    viewport_width: int = VIEWPORT_WIDTH,
    viewport_height: int = VIEWPORT_HEIGHT,
) -> dict[str, list[str]]:
    """Generate hover-reveal episodes under *output_dir*."""
    icon_assets = collect_icon_assets(icon_root)
    rng = random.Random(seed)
    generated: dict[str, list[str]] = {"train": [], "test": []}

    split_specs = [
        ("train", train_count, 1),
        ("test", test_count, train_count + 1),
    ]
    for split, count, start_index in split_specs:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for index in range(start_index, start_index + count):
            episode_id = f"hr_{index:04d}"
            write_episode(
                episode_id=episode_id,
                split=split,
                rng=rng,
                icon_assets=icon_assets,
                split_dir=split_dir,
                viewport_width=viewport_width,
                viewport_height=viewport_height,
            )
            generated[split].append(episode_id)
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the hover_reveal icon-hover benchmark dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--icon-dir",
        type=Path,
        default=DEFAULT_ICON_DIR,
        help="Directory containing icon assets to sample from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=source_episodes_root(),
        help="Output root for generated episode directories.",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=1000,
        help="Number of train episodes to generate.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=500,
        help="Number of test episodes to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument("--viewport-width", type=int, default=VIEWPORT_WIDTH)
    parser.add_argument("--viewport-height", type=int, default=VIEWPORT_HEIGHT)
    args = parser.parse_args(argv)

    generated = generate_dataset(
        output_dir=args.output_dir,
        icon_root=args.icon_dir,
        train_count=args.train_count,
        test_count=args.test_count,
        seed=args.seed,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
    )
    print(
        textwrap.dedent(
            f"""\
            Generated hover_reveal dataset:
              train: {len(generated['train'])}
              test:  {len(generated['test'])}
              icons: {args.icon_dir}
              out:   {args.output_dir}
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
