from __future__ import annotations

import copy
import html
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...domains.slot_drag.paths import formal_benchmark_root as slot_drag_root
from ...domains.ten_choice.paths import real_environment_hd720_root
from ...domains.third_person_drag.paths import (
    formal_benchmark_root as third_person_drag_root,
)
from ...integrations.storage import storage_path
from .contracts import EXPLORATION_BENCHMARK_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ROTATION_SOURCE = storage_path(
    "artifacts",
    "datasets",
    "lujiahao_rotation_outer_fixed_center_ood150_hd720_20260821",
)
DEFAULT_ROTATION_BACKGROUND_METADATA = storage_path(
    "artifacts",
    "background_pools",
    "rotation_openimages_20260604",
    "test_150_metadata.json",
)
TEN_CHOICE_SENSITIVITY_BANDS = {
    "low": (500, 800),
    "high": (1200, 1500),
}
FPS_DIRECTION_XY = (1, 1)
DRAG_SHARED_LAYOUT_RESET_COUNT = 100
DRAG_OFFSCREEN_RESET_COUNT = 50


@dataclass(frozen=True)
class ExplorationSourceRoots:
    ten_choice: Path
    slot_drag: Path
    third_person_drag: Path
    rotation_dataset: Path
    rotation_background_metadata: Path

    @classmethod
    def defaults(cls) -> "ExplorationSourceRoots":
        return cls(
            ten_choice=real_environment_hd720_root(),
            slot_drag=slot_drag_root() / "test",
            third_person_drag=third_person_drag_root() / "test",
            rotation_dataset=DEFAULT_ROTATION_SOURCE,
            rotation_background_metadata=DEFAULT_ROTATION_BACKGROUND_METADATA,
        )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_clean_rotation_backgrounds(metadata_path: Path) -> dict[str, Path]:
    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"rotation background metadata must be a list: {metadata_path}")
    backgrounds: dict[str, Path] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"invalid rotation background metadata row: {record!r}")
        url = str(record.get("localPath") or "")
        source = Path(str(record.get("sourceAbsolutePath") or ""))
        if not url or not source.is_file():
            raise FileNotFoundError(
                f"clean rotation background is unavailable for {url!r}: {source}"
            )
        if url in backgrounds:
            raise ValueError(f"duplicate rotation background URL: {url}")
        backgrounds[url] = source
    return backgrounds


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _assert_directly_unique_files(paths: list[Path], *, label: str) -> None:
    seen_by_size: dict[int, list[tuple[Path, bytes]]] = {}
    for path in paths:
        payload = path.read_bytes()
        for previous_path, previous_payload in seen_by_size.setdefault(
            len(payload), []
        ):
            if payload == previous_payload:
                raise ValueError(
                    f"NO-GO: {label} contains duplicate bytes: "
                    f"{previous_path.name} == {path.name}"
                )
        seen_by_size[len(payload)].append((path, payload))


def _split(index: int, count: int) -> str:
    if count == 1:
        return "smoke"
    return "iid" if index < count // 2 else "ood"


def _family_pair_id(family: str, index: int) -> str:
    prefix = {"ten_choice": "ten", "rotation": "rot", "drag": "drag"}[family]
    return f"ed-{prefix}-{index + 1:04d}"


def _episode_id(pair_id: str, variant: str) -> str:
    return f"{pair_id}--{variant.replace('_', '-')}"


def _model_xy_for_screen_xy(
    xy: tuple[float, float],
    viewport: tuple[int, int],
) -> list[float]:
    return [
        round(xy[0] / viewport[0] * 1000.0, 6),
        round(xy[1] / viewport[1] * 1000.0, 6),
    ]


def _bimodal_sensitivity(index: int) -> tuple[str, float]:
    band = "low" if index % 2 == 0 else "high"
    lower, upper = TEN_CHOICE_SENSITIVITY_BANDS[band]
    milli = lower + ((index * 73 + 29) % (upper - lower + 1))
    return band, round(milli / 1000.0, 3)


def _ten_choice_html(meta: dict[str, Any], *, instruction: str) -> str:
    canvas = dict(meta["canvas_rect"])
    labels = list(meta["labels"])
    positions = list(meta["icon_positions_xy"])
    buttons: list[str] = []
    for index, (label, position) in enumerate(zip(labels, positions)):
        left = float(position[0]) - float(canvas["left"])
        top = float(position[1]) - float(canvas["top"])
        buttons.append(
            "<button class=\"icon-button\" "
            f"data-index=\"{index}\" data-label=\"{html.escape(str(label), quote=True)}\" "
            f"style=\"left:{left:g}px;top:{top:g}px\" "
            f"onclick=\"selectIcon({index})\">"
            "<img src=\"assets/icon.svg\" alt=\"\" draggable=\"false\"></button>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=1280, initial-scale=1">
  <title>Ten-choice selection</title>
  <style>
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; }}
    body {{ font-family: Arial, sans-serif; background: #fff; color: #0f172a; }}
    .page {{ position: relative; width: 1280px; height: 720px; }}
    .instruction {{ position: absolute; left: 54px; top: 28px; width: 1172px;
      text-align: center; font-size: 29px; line-height: 56px; font-weight: 700;
      white-space: nowrap; }}
    .instruction strong {{ color: #1d4ed8; }}
    .canvas {{ position: absolute; left: 24px; top: 116px; width: 1232px; height: 580px;
      overflow: hidden; border-radius: 28px; border: 2px solid rgba(15,23,42,.12);
      background: linear-gradient(135deg,#dff7ea 0%,#d6ecff 52%,#efe0ff 100%);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.45),0 12px 30px rgba(15,23,42,.08); }}
    .icon-scene {{ position: absolute; inset: 0; transform: translate(0px,0px); }}
    .icon-button {{ position: absolute; width: 92px; height: 92px; border: 0; padding: 0;
      background: transparent; cursor: pointer; display: flex; align-items: center;
      justify-content: center; }}
    .icon-button img {{ width: 100%; height: 100%; object-fit: contain; pointer-events: none;
      user-select: none; filter: drop-shadow(0 10px 16px rgba(15,23,42,.16)); }}
    .icon-button::after {{ content: attr(data-label); position: absolute; left: 50%;
      bottom: calc(100% + 12px); transform: translateX(-50%); padding: 8px 12px;
      border-radius: 999px; background: #0f172a; color: #fff; font-size: 13px;
      font-weight: 600; line-height: 18px; width: max-content; max-width: 320px;
      white-space: normal; overflow-wrap: anywhere; text-align: center;
      opacity: 0; visibility: hidden; pointer-events: none; z-index: 20; }}
    .icon-button.tooltip-active {{ z-index: 21; }}
    .icon-button.tooltip-active::after {{ opacity: 1; visibility: visible; }}
    .icon-button.tooltip-shifted::after {{ margin-left: var(--tooltip-dx);
      margin-bottom: var(--tooltip-dy); }}
    .icon-button:hover, .icon-button.center-hover {{ transform: translateY(-2px) scale(1.04); }}
    .icon-button.selected {{ box-shadow: inset 0 0 0 3px #f97316; }}
    .feedback {{ position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%);
      min-width: 140px; padding: 10px 18px; border-radius: 999px;
      background: rgba(15,23,42,.08); color: #334155; font-size: 16px;
      font-weight: 700; text-align: center; }}
    .feedback.correct {{ background: #dcfce7; color: #166534; }}
    .feedback.wrong {{ background: #fee2e2; color: #b91c1c; }}
    .reticle {{ position: fixed; left: 640px; top: 360px; width: 8px; height: 8px;
      margin: -4px 0 0 -4px; border: 1px solid #7f1d1d; border-radius: 50%;
      background: #ef4444; box-shadow: 0 0 0 1px rgba(255,255,255,.9);
      pointer-events: none; z-index: 100; }}
    body[data-interaction-marker="mouse_icon"] .reticle {{ display: none; }}
  </style>
</head>
<body>
  <main class="page">
    <div class="instruction">{instruction}</div>
    <section class="canvas"><div class="icon-scene">{''.join(buttons)}</div></section>
    <div id="feedback" class="feedback">Choose one icon</div>
    <div id="result" hidden></div>
    <div class="reticle" aria-hidden="true"></div>
  </main>
  <script>
    (() => {{
      let answered = false;
      let offsetX = 0;
      let offsetY = 0;
      let firstPerson = false;
      const scene = document.querySelector('.icon-scene');
      const canvas = document.querySelector('.canvas');
      const icons = Array.from(document.querySelectorAll('.icon-button'));
      const feedback = document.getElementById('feedback');
      const viewportCenter = {{x: 640, y: 360}};
      document.body.dataset.interactionMarker = 'red_dot';

      function showTooltip(icon) {{
        icons.forEach(item => item.classList.toggle('tooltip-active', item === icon));
        if (!icon) return;
        // Keep the original pseudo-element, including its inherited 1.04 scale.
        // Only overflowing tooltips receive a positional adjustment.
        icon.classList.remove('tooltip-shifted');
        const style = getComputedStyle(icon, '::after');
        const rect = icon.getBoundingClientRect();
        const scale = rect.width / icon.offsetWidth;
        const width = (parseFloat(style.width) + parseFloat(style.paddingLeft) +
          parseFloat(style.paddingRight)) * scale;
        const height = (parseFloat(style.height) + parseFloat(style.paddingTop) +
          parseFloat(style.paddingBottom)) * scale;
        const left = rect.left + parseFloat(style.left) * scale - width / 2;
        const top = rect.bottom - parseFloat(style.bottom) * scale - height;
        const bounds = canvas.getBoundingClientRect();
        const minY = bounds.top + canvas.clientTop;
        const maxY = minY + canvas.clientHeight - height;
        const y = Math.max(minY, Math.min(top, maxY));
        // The inner canvas has rounded corners. Guard them only when necessary.
        const corner = y < minY + 26 || y + height > minY + canvas.clientHeight - 26;
        const minX = bounds.left + canvas.clientLeft + (corner ? 26 : 0);
        const maxX = bounds.right - canvas.clientLeft - width - (corner ? 26 : 0);
        const x = Math.max(minX, Math.min(left, maxX));
        if (x !== left || y !== top) {{
          icon.style.setProperty('--tooltip-dx', `${{(x - left) / scale}}px`);
          icon.style.setProperty('--tooltip-dy', `${{(top - y) / scale}}px`);
          icon.classList.add('tooltip-shifted');
        }}
      }}

      icons.forEach(icon => {{
        icon.addEventListener('mouseenter', () => {{
          if (!firstPerson) showTooltip(icon);
        }});
        icon.addEventListener('mouseleave', () => {{
          if (!firstPerson) showTooltip(null);
        }});
      }});

      window.__explorationSetTenChoiceMode = (variant) => {{
        firstPerson = variant === 'ten_choice_first_person';
        showTooltip(null);
        updateCenterHover();
        if (!firstPerson) showTooltip(icons.find(icon => icon.matches(':hover')));
      }};

      window.__explorationSetInteractionMarker = (marker) => {{
        document.body.dataset.interactionMarker = marker === 'mouse_icon'
          ? 'mouse_icon'
          : 'red_dot';
      }};

      window.selectIcon = (index) => {{
        if (answered) return;
        icons.forEach((icon, iconIndex) => icon.classList.toggle('selected', iconIndex === index));
        const label = icons[index].dataset.label;
        const correct = label === {json.dumps(str(meta['target_label']))};
        answered = correct;
        feedback.textContent = correct ? 'Correct' : 'Wrong - try again';
        feedback.className = `feedback ${{correct ? 'correct' : 'wrong'}}`;
        document.getElementById('result').textContent = JSON.stringify({{clicked:index,label}});
      }};

      function updateCenterHover() {{
        let hovered = null;
        for (const icon of icons) {{
          const rect = icon.getBoundingClientRect();
          const hit = viewportCenter.x >= rect.left && viewportCenter.x <= rect.right &&
            viewportCenter.y >= rect.top && viewportCenter.y <= rect.bottom;
          icon.classList.toggle('center-hover', firstPerson && hit);
          if (hit) hovered = Number(icon.dataset.index);
        }}
        if (firstPerson) showTooltip(hovered === null ? null : icons[hovered]);
        return hovered;
      }}

      window.__explorationMoveScene = (delta) => {{
        offsetX += Number(delta.dx || 0);
        offsetY += Number(delta.dy || 0);
        scene.style.transform = `translate(${{offsetX}}px,${{offsetY}}px)`;
        const hoveredIndex = updateCenterHover();
        return {{offsetX, offsetY, hoveredIndex}};
      }};
      window.__explorationReadTenChoiceState = () => ({{
        offsetX, offsetY, hoveredIndex: updateCenterHover(),
        result: document.getElementById('result').textContent || ''
      }});
      updateCenterHover();
    }})();
  </script>
</body>
</html>
"""


def _common_episode(
    *,
    suite_id: str,
    pair_id: str,
    family: str,
    variant: str,
    exploration_level: str,
    index: int,
    count: int,
    instruction: str,
    viewport: list[int],
    shared_scene_config: dict[str, Any],
    environment_config: dict[str, Any],
    success_evaluator: dict[str, Any],
    canonical_case_path: str,
) -> dict[str, Any]:
    return {
        "suite_id": suite_id,
        "episode_id": _episode_id(pair_id, variant),
        "pair_id": pair_id,
        "family": family,
        "variant": variant,
        "exploration_level": exploration_level,
        "case_seed": 2026082000 + {"ten_choice": 0, "rotation": 1000, "drag": 2000}[family] + index,
        "split": _split(index, count),
        "viewport": viewport,
        "instruction": instruction,
        "shared_scene_config": copy.deepcopy(shared_scene_config),
        "environment_config": copy.deepcopy(environment_config),
        "success_evaluator": copy.deepcopy(success_evaluator),
        "canonical_case_path": canonical_case_path,
        "policy_observation_allowlist": ["suite_id", "pair_id", "family", "viewport"],
    }


def _ten_choice_episodes(
    *,
    suite_id: str,
    output_root: Path,
    source_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    source_dirs = sorted(
        path for path in source_root.iterdir() if path.is_dir() and (path / "meta.json").is_file()
    )
    if len(source_dirs) < count:
        raise ValueError(f"ten-choice source has {len(source_dirs)} cases, need {count}")
    episodes: list[dict[str, Any]] = []
    for index, source_dir in enumerate(source_dirs[:count]):
        source_meta = _read_json(source_dir / "meta.json")
        pair_id = _family_pair_id("ten_choice", index)
        case_dir = output_root / "cases/ten_choice" / pair_id
        target_label = str(source_meta["target_label"])
        instruction = f'Click the icon that displays "<strong>{html.escape(target_label)}</strong>".'
        icon_source = source_dir / str(source_meta.get("icon_asset", "assets/icon.svg"))
        _copy_file(icon_source, case_dir / "assets/icon.svg")
        html_text = _ten_choice_html(source_meta, instruction=instruction)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "index.html").write_text(html_text, encoding="utf-8")

        viewport = [int(value) for value in source_meta.get("viewport", [1280, 720])]
        sensitivity_band, sensitivity = _bimodal_sensitivity(index)
        direction_xy = list(FPS_DIRECTION_XY)
        shared = {
            "scene_schema": "paired_ten_choice_v1",
            "source_case_id": source_meta["episode_id"],
            "background": "ten_choice_gradient_canvas_v1",
            "icon_asset": f"cases/ten_choice/{pair_id}/assets/icon.svg",
            "labels": list(source_meta["labels"]),
            "icon_centers_xy": copy.deepcopy(source_meta["icon_centers_xy"]),
            "icon_positions_xy": copy.deepcopy(source_meta["icon_positions_xy"]),
            "canvas_rect": copy.deepcopy(source_meta["canvas_rect"]),
            "initial_cursor_or_reticle_xy": [viewport[0] / 2.0, viewport[1] / 2.0],
            "target_object": {"target_index": int(source_meta["target_idx"]), "target_label": target_label},
            "hidden_dynamics": {
                "sensitivity": sensitivity,
                "sensitivity_band": sensitivity_band,
                "direction_xy": direction_xy,
            },
            "html_path": f"cases/ten_choice/{pair_id}/index.html",
        }
        _write_json(case_dir / "meta.json", shared)
        common_config = {
            "semantic_action_unit": "one action followed by its newly rendered observation",
            "hidden_dynamics": copy.deepcopy(shared["hidden_dynamics"]),
            "minimum_steps_gate": None,
        }
        variants = (
            (
                "ten_choice_third_person",
                "L1",
                {
                    **common_config,
                    "action_response": "move free cursor to an absolute screen coordinate",
                    "responsive_coordinate_system": "screen",
                    "interaction_marker": "mouse_icon",
                    "reset_frame_contract": "shared_scene_with_exocentric_mouse_icon",
                },
                {"type": "clicked_target_icon", "terminal_event": "first_release"},
            ),
            (
                "ten_choice_first_person",
                "L2",
                {
                    **common_config,
                    "action_response": "move icon scene under a fixed center reticle",
                    "responsive_coordinate_system": "scene_offset_from_center",
                    "interaction_marker": "red_dot",
                    "reset_frame_contract": "shared_scene_with_egocentric_red_dot",
                },
                {
                    "type": "target_icon_inside_center_interaction_zone",
                    "terminal_event": "fixed_center_click_or_release",
                },
            ),
        )
        for variant, level, environment, evaluator in variants:
            episode = _common_episode(
                suite_id=suite_id,
                pair_id=pair_id,
                family="ten_choice",
                variant=variant,
                exploration_level=level,
                index=index,
                count=count,
                instruction=f'Click the icon that displays "{target_label}".',
                viewport=viewport,
                shared_scene_config=shared,
                environment_config=environment,
                success_evaluator=evaluator,
                canonical_case_path=f"cases/ten_choice/{pair_id}/meta.json",
            )
            if count == 150:
                episode["split"] = None
            episodes.append(episode)
    return episodes


def _rotation_episodes(
    *,
    suite_id: str,
    output_root: Path,
    source_root: Path,
    background_metadata: Path,
    count: int,
) -> list[dict[str, Any]]:
    audit_rows = _read_jsonl(source_root / "audit.jsonl")
    if len(audit_rows) < count:
        raise ValueError(
            f"NO-GO: rotation source has {len(audit_rows)} cases, need {count}; "
            "cycling source cases is forbidden"
        )
    selected_rows = audit_rows[:count]
    clean_backgrounds = _load_clean_rotation_backgrounds(background_metadata)
    selected_urls = [
        str(row["metadata"].get("source_background_image_url") or "")
        for row in selected_rows
    ]
    missing_urls = [url for url in selected_urls if url not in clean_backgrounds]
    if missing_urls:
        raise FileNotFoundError(
            f"clean rotation source backgrounds are missing from {background_metadata}: "
            f"{missing_urls[:3]}"
        )
    selected_assets = [
        clean_backgrounds[url] for url in selected_urls
    ]
    _assert_directly_unique_files(
        selected_assets,
        label="clean rotation source backgrounds",
    )
    episodes: list[dict[str, Any]] = []
    for index, source in enumerate(selected_rows):
        metadata = dict(source["metadata"])
        source_challenge = dict(metadata["raw_challenge"])
        source_url = str(metadata["source_background_image_url"])
        source_asset = clean_backgrounds[source_url]
        pair_id = _family_pair_id("rotation", index)
        asset_name = f"{pair_id}{source_asset.suffix.lower()}"
        destination = output_root / "assets/rotation" / asset_name
        _copy_file(source_asset, destination)

        challenge = {
            key: copy.deepcopy(value)
            for key, value in source_challenge.items()
            if key != "rotationRegion"
        }
        challenge["pairedRelativeRotation"] = True
        challenge["canonicalInitialRelativeDeg"] = (
            float(challenge["targetRotationDeg"])
            + (
                float(challenge["startSliderValue"])
                - float(challenge["targetSliderValue"])
            )
            * float(challenge["degreesPerSliderUnit"])
            * float(challenge["sensitivityScale"])
            * float(challenge["rotationDirection"])
        ) % 360.0
        challenge["rotationToleranceDeg"] = min(
            5.0,
            float(metadata.get("rotationToleranceDeg", 5.0)),
        )
        slider_box = copy.deepcopy(
            metadata.get("outer_fixed_center_render_audit", {})
            .get("page_state", {})
            .get("sliderBox", metadata.get("slider_box"))
        )
        viewport = [int(value) for value in metadata.get("viewport", [1280, 720])]
        shared = {
            "scene_schema": "paired_relative_rotation_v1",
            "source_case_id": metadata.get("source_record_id", source["id"]),
            "source_background_image_url": metadata.get(
                "source_background_image_url"
            ),
            "background_asset": f"assets/rotation/{asset_name}",
            "circle_center": copy.deepcopy(challenge["circleCenter"]),
            "circle_radius": challenge["circleRadius"],
            "initial_relative_angle_deg": challenge["canonicalInitialRelativeDeg"],
            "target_relative_angle_deg": challenge["targetRotationDeg"],
            "slider_geometry": slider_box,
            "slider_min_value": challenge["sliderMinValue"],
            "slider_max_value": challenge["sliderMaxValue"],
            "start_slider_value": challenge["startSliderValue"],
            "target_slider_value": challenge["targetSliderValue"],
            "degrees_per_slider_unit": challenge["degreesPerSliderUnit"],
            "sensitivity": challenge["sensitivityScale"],
            "rotation_direction": challenge["rotationDirection"],
            "tolerance_deg": challenge["rotationToleranceDeg"],
            "initial_cursor_or_reticle_xy": [0.0, 0.0],
            "replay_challenge": challenge,
        }
        case_dir = output_root / "cases/rotation" / pair_id
        _write_json(case_dir / "meta.json", shared)
        common_config = {
            "static_replay_renderer": "scripts/interaction/serve_captcha_static_replay.py",
            "interaction_marker": "mouse_icon",
            "reset_frame_contract": "identical_free_mouse_scene",
            "hidden_mapping": {
                "degrees_per_slider_unit": challenge["degreesPerSliderUnit"],
                "sensitivity": challenge["sensitivityScale"],
                "rotation_direction": challenge["rotationDirection"],
                "target_slider_value": challenge["targetSliderValue"],
            },
            "minimum_steps_gate": None,
        }
        for variant, region in (("rotation_inner", "center"), ("rotation_outer", "outer")):
            episode = _common_episode(
                suite_id=suite_id,
                pair_id=pair_id,
                family="rotation",
                variant=variant,
                exploration_level="L2",
                index=index,
                count=count,
                instruction="Drag the slider to complete verification.",
                viewport=viewport,
                shared_scene_config=shared,
                environment_config={
                    **common_config,
                    "action_response": f"rotate only the {region} visual region",
                    "responsive_region": region,
                },
                success_evaluator={
                    "type": "relative_angular_error_on_release",
                    "responsive_region": region,
                    "tolerance_deg": challenge["rotationToleranceDeg"],
                },
                canonical_case_path=f"cases/rotation/{pair_id}/meta.json",
            )
            if count == 150:
                episode["split"] = None
            episodes.append(episode)
    return episodes


def _screen_to_world(
    xy: list[float],
    *,
    view_center: tuple[float, float],
    viewport: tuple[int, int],
    zoom: float,
) -> list[float]:
    return [
        round(view_center[0] + (float(xy[0]) - viewport[0] / 2.0) / zoom, 6),
        round(view_center[1] + (float(xy[1]) - viewport[1] / 2.0) / zoom, 6),
    ]


def _shape_half_extent(radius: float, shape: str) -> tuple[float, float]:
    if shape == "wide_rectangle":
        return (radius * 1.38, radius * 0.72)
    if shape == "notched_rectangle":
        return (radius * 1.36, radius * 0.76)
    if shape == "trapezoid":
        return (radius * 1.14, radius * 0.88)
    return (radius, radius)


def _offscreen_first_person_view_center(
    *,
    piece_world_xy: list[float],
    target_world_xy: list[float],
    viewport: tuple[int, int],
    world_size: tuple[int, int],
    zoom: float,
    piece_radius: float,
    target_shape: str,
    slot_radius_scale: float,
) -> list[float] | None:
    half_w = viewport[0] / 2.0 / zoom
    half_h = viewport[1] / 2.0 / zoom
    center_bounds = (
        (min(half_w, world_size[0] - half_w), max(half_w, world_size[0] - half_w)),
        (min(half_h, world_size[1] - half_h), max(half_h, world_size[1] - half_h)),
    )

    def clamp(value: float, axis: int) -> float:
        lower, upper = center_bounds[axis]
        return min(max(float(value), lower), upper)

    outer_slot_radius = piece_radius * max(slot_radius_scale, 1.0) + 14.0
    target_extent = _shape_half_extent(outer_slot_radius, target_shape)
    piece_extent = _shape_half_extent(piece_radius + 12.0, target_shape)
    candidates = (
        (
            target_world_xy[0] + (viewport[0] / 2.0 + target_extent[0] + 2.0) / zoom,
            piece_world_xy[1],
        ),
        (
            target_world_xy[0] - (viewport[0] / 2.0 + target_extent[0] + 2.0) / zoom,
            piece_world_xy[1],
        ),
        (
            piece_world_xy[0],
            target_world_xy[1] + (viewport[1] / 2.0 + target_extent[1] + 2.0) / zoom,
        ),
        (
            piece_world_xy[0],
            target_world_xy[1] - (viewport[1] / 2.0 + target_extent[1] + 2.0) / zoom,
        ),
    )
    for raw_center in candidates:
        center = (clamp(raw_center[0], 0), clamp(raw_center[1], 1))
        piece_screen = (
            viewport[0] / 2.0 + (piece_world_xy[0] - center[0]) * zoom,
            viewport[1] / 2.0 + (piece_world_xy[1] - center[1]) * zoom,
        )
        target_screen = (
            viewport[0] / 2.0 + (target_world_xy[0] - center[0]) * zoom,
            viewport[1] / 2.0 + (target_world_xy[1] - center[1]) * zoom,
        )
        piece_visible = (
            piece_extent[0] <= piece_screen[0] <= viewport[0] - piece_extent[0]
            and piece_extent[1] <= piece_screen[1] <= viewport[1] - piece_extent[1]
        )
        target_fully_outside = (
            target_screen[0] + target_extent[0] < 0
            or target_screen[0] - target_extent[0] > viewport[0]
            or target_screen[1] + target_extent[1] < 0
            or target_screen[1] - target_extent[1] > viewport[1]
        )
        if piece_visible and target_fully_outside:
            return [round(center[0], 6), round(center[1], 6)]
    return None


def _spread_selection(indices: list[int], count: int) -> set[int]:
    if len(indices) < count:
        raise ValueError(f"need {count} eligible drag cases, found {len(indices)}")
    if count == 0:
        return set()
    return {indices[position * len(indices) // count] for position in range(count)}


def _drag_episodes(
    *,
    suite_id: str,
    output_root: Path,
    third_person_source: Path,
    slot_source: Path,
    count: int,
) -> list[dict[str, Any]]:
    third_rows = [_read_json(path) for path in sorted(third_person_source.glob("*/meta.json"))]
    slot_rows = [_read_json(path) for path in sorted(slot_source.glob("*/meta.json"))]
    if count == 150:
        third_rows = sorted(third_rows, key=lambda row: (row.get("distribution") != "iid", row["episode_id"]))
    if len(third_rows) < count or not slot_rows:
        raise ValueError(
            f"drag sources are insufficient: third_person={len(third_rows)}, slot={len(slot_rows)}"
        )
    offscreen_centers: dict[int, list[float]] = {}
    if count == 150:
        eligible_by_split: dict[str, list[int]] = {"iid": [], "ood": []}
        for index, source in enumerate(third_rows[:count]):
            dynamics = slot_rows[index % len(slot_rows)]
            viewport = tuple(int(value) for value in source.get("viewport", [1280, 720]))
            zoom = float(dynamics.get("view_zoom", 1.0))
            base_view_center = (1100.0, 800.0)
            piece_world = _screen_to_world(
                source["piece_start_screen_xy"],
                view_center=base_view_center,
                viewport=viewport,
                zoom=zoom,
            )
            target_world = _screen_to_world(
                source["slot_center_screen_xy"],
                view_center=base_view_center,
                viewport=viewport,
                zoom=zoom,
            )
            center = _offscreen_first_person_view_center(
                piece_world_xy=piece_world,
                target_world_xy=target_world,
                viewport=viewport,
                world_size=(2200, 1600),
                zoom=zoom,
                piece_radius=float(source["piece"]["radius_px"]),
                target_shape=str(source["piece"]["shape"]),
                slot_radius_scale=float(source["slot_radius_scale"]),
            )
            distribution = str(source.get("distribution", ""))
            if center is not None and distribution in eligible_by_split:
                eligible_by_split[distribution].append(index)
                offscreen_centers[index] = center
        selected_offscreen = set()
        for distribution in ("iid", "ood"):
            selected_offscreen.update(
                _spread_selection(
                    eligible_by_split[distribution],
                    DRAG_OFFSCREEN_RESET_COUNT // 2,
                )
            )
        offscreen_centers = {
            index: center
            for index, center in offscreen_centers.items()
            if index in selected_offscreen
        }
        if len(offscreen_centers) != DRAG_OFFSCREEN_RESET_COUNT:
            raise ValueError(
                "drag offscreen reset selection must contain exactly "
                f"{DRAG_OFFSCREEN_RESET_COUNT} cases"
            )
    episodes: list[dict[str, Any]] = []
    for index, source in enumerate(third_rows[:count]):
        dynamics = slot_rows[index % len(slot_rows)]
        pair_id = _family_pair_id("drag", index)
        viewport = tuple(int(value) for value in source.get("viewport", [1280, 720]))
        view_center = (1100.0, 800.0)
        zoom = float(dynamics.get("view_zoom", 1.0))
        sensitivity = float(dynamics.get("sensitivity", 1.0))
        direction_xy = list(FPS_DIRECTION_XY)
        piece_screen = [float(value) for value in source["piece_start_screen_xy"]]
        slot_screen = [float(value) for value in source["slot_center_screen_xy"]]
        piece_world = _screen_to_world(
            piece_screen, view_center=view_center, viewport=viewport, zoom=zoom
        )
        slot_world = _screen_to_world(
            slot_screen, view_center=view_center, viewport=viewport, zoom=zoom
        )
        slot_options: list[dict[str, Any]] = []
        for option in source["slot_options"]:
            item = copy.deepcopy(option)
            screen_xy = [float(value) for value in option["center_screen_xy"]]
            item["center_screen_xy"] = screen_xy
            item["center_world_xy"] = _screen_to_world(
                screen_xy,
                view_center=view_center,
                viewport=viewport,
                zoom=zoom,
            )
            slot_options.append(item)
        instruction = "Place the solid colored shape into the matching gray outline."
        reset_group = (
            "egocentric_target_offscreen"
            if index in offscreen_centers
            else "shared_layout_different_interaction_marker"
        )
        canonical = {
            "episode_id": pair_id,
            "paired_scene_schema": "paired_drag_scene_v1",
            "source_case_id": source["episode_id"],
            "viewport": list(viewport),
            "world_size": [2200, 1600],
            "initial_cursor_screen_xy": [viewport[0] / 2.0, viewport[1] / 2.0],
            "initial_view_center_xy": list(view_center),
            "piece_start_screen_xy": piece_screen,
            "piece_start_world_xy": piece_world,
            "slot_center_screen_xy": slot_screen,
            "slot_center_world_xy": slot_world,
            "slot_options": slot_options,
            "piece": copy.deepcopy(source["piece"]),
            "slot_radius_scale": source["slot_radius_scale"],
            "tolerance_px": source["tolerance_px"],
            "instruction": instruction,
            "layout_family": source.get("layout_family"),
            "distribution": source.get("distribution", _split(index, count)),
            "source_split": source.get("split"),
            "sensitivity": sensitivity,
            "view_zoom": zoom,
            "movement_direction_xy": direction_xy,
            "reset_pairing_group": reset_group,
        }
        case_dir = output_root / "cases/drag" / pair_id
        _write_json(case_dir / "meta.json", canonical)
        shared = {
            "scene_schema": canonical["paired_scene_schema"],
            "background": "paired_drag_grid_v1",
            "piece": copy.deepcopy(canonical["piece"]),
            "piece_start_screen_xy": piece_screen,
            "piece_start_world_xy": piece_world,
            "slot_center_screen_xy": slot_screen,
            "slot_center_world_xy": slot_world,
            "slot_options": copy.deepcopy(slot_options),
            "initial_cursor_or_reticle_xy": canonical["initial_cursor_screen_xy"],
            "initial_view_center_xy": canonical["initial_view_center_xy"],
            "world_size": canonical["world_size"],
            "slot_radius_scale": canonical["slot_radius_scale"],
            "reset_pairing_group": reset_group,
            "hidden_dynamics": {
                "sensitivity": sensitivity,
                "view_zoom": zoom,
                "direction_xy": direction_xy,
            },
        }
        common_config = {
            "hidden_dynamics": copy.deepcopy(shared["hidden_dynamics"]),
            "minimum_steps_gate": None,
        }
        first_person_reset_config: dict[str, Any] = {
            "reset_frame_contract": reset_group,
            "interaction_marker": "red_dot",
        }
        if index in offscreen_centers:
            first_person_reset_config["initial_view_center_xy"] = offscreen_centers[index]
        variants = (
            (
                "drag_third_person",
                "L0",
                {
                    **common_config,
                    "action_response": "move a free cursor and directly drag in screen coordinates",
                    "responsive_coordinate_system": "screen",
                    "interaction_marker": "mouse_icon",
                    "reset_frame_contract": (
                        "shared_layout_with_third_person_mouse_icon"
                        if reset_group == "shared_layout_different_interaction_marker"
                        else "third_person_mouse_icon_visible_reference"
                    ),
                },
                {"type": "final_screen_geometry_match", "tolerance_px": source["tolerance_px"]},
            ),
            (
                "drag_first_person",
                "L2",
                {
                    **common_config,
                    "action_response": "move world view under a fixed center reticle",
                    "responsive_coordinate_system": "world_through_hidden_view_mapping",
                    **first_person_reset_config,
                },
                {"type": "final_world_geometry_match", "tolerance_px": source["tolerance_px"]},
            ),
        )
        for variant, level, environment, evaluator in variants:
            episode = _common_episode(
                suite_id=suite_id,
                pair_id=pair_id,
                family="drag",
                variant=variant,
                exploration_level=level,
                index=index,
                count=count,
                instruction=instruction,
                viewport=list(viewport),
                shared_scene_config=shared,
                environment_config=environment,
                success_evaluator=evaluator,
                canonical_case_path=f"cases/drag/{pair_id}/meta.json",
            )
            episode["split"] = str(source.get("distribution", episode["split"]))
            episodes.append(episode)
    return episodes


def validate_manifest_structure(manifest: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    episodes = list(manifest.get("episodes", []))
    implementation_tag = str(manifest.get("implementation_tag") or "6env-v1")
    required = {
        "suite_id",
        "pair_id",
        "family",
        "variant",
        "exploration_level",
        "case_seed",
        "split",
        "viewport",
        "shared_scene_config",
        "environment_config",
        "success_evaluator",
    }
    missing = [
        (episode.get("episode_id"), sorted(required - set(episode)))
        for episode in episodes
        if required - set(episode)
    ]
    if missing:
        raise ValueError(f"formal episodes are missing required fields: {missing[:3]}")

    counts = Counter(str(episode["variant"]) for episode in episodes)
    expected_variants = {contract.key for contract in EXPLORATION_BENCHMARK_VARIANTS}
    if set(counts) != expected_variants or set(counts.values()) != {expected_count}:
        raise ValueError(f"variant episode counts differ from {expected_count}: {dict(counts)}")
    for variant in expected_variants:
        ids = [episode["episode_id"] for episode in episodes if episode["variant"] == variant]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{variant} contains duplicate episode IDs")

    pair_sets: dict[str, set[str]] = {
        variant: {
            str(episode["pair_id"])
            for episode in episodes
            if episode["variant"] == variant
        }
        for variant in expected_variants
    }
    family_pairs = {
        "ten_choice": ("ten_choice_third_person", "ten_choice_first_person"),
        "rotation": ("rotation_inner", "rotation_outer"),
        "drag": ("drag_third_person", "drag_first_person"),
    }
    for family, variants in family_pairs.items():
        if pair_sets[variants[0]] != pair_sets[variants[1]]:
            raise ValueError(f"{family} variants do not have identical pair IDs")
        for pair_id in pair_sets[variants[0]]:
            pair = [
                episode
                for episode in episodes
                if episode["family"] == family and episode["pair_id"] == pair_id
            ]
            if len(pair) != 2:
                raise ValueError(f"{family}/{pair_id} does not contain exactly two variants")
            for field in (
                "suite_id",
                "pair_id",
                "case_seed",
                "split",
                "viewport",
                "instruction",
                "shared_scene_config",
                "canonical_case_path",
            ):
                if pair[0][field] != pair[1][field]:
                    raise ValueError(f"{family}/{pair_id} differs in shared field {field}")

    drag_counts = Counter(
        episode["split"]
        for episode in episodes
        if episode["variant"] == "drag_first_person"
    )
    family_split_counts = {
        family: Counter(
            episode["split"]
            for episode in episodes
            if episode["family"] == family
            and episode["variant"] == family_pairs[family][0]
        )
        for family in family_pairs
    }
    ten_choice_episodes = [
        episode
        for episode in episodes
        if episode["variant"] == "ten_choice_first_person"
    ]
    ten_choice_sensitivity_bands = Counter(
        episode["shared_scene_config"]["hidden_dynamics"].get("sensitivity_band")
        for episode in ten_choice_episodes
    )
    for episode in ten_choice_episodes:
        dynamics = episode["shared_scene_config"]["hidden_dynamics"]
        if tuple(dynamics.get("direction_xy", ())) != FPS_DIRECTION_XY:
            raise ValueError(
                f"{episode['pair_id']} ten-choice direction must use the fixed FPS mapping"
            )
        band = str(dynamics.get("sensitivity_band"))
        if band not in TEN_CHOICE_SENSITIVITY_BANDS:
            raise ValueError(f"invalid ten-choice sensitivity band: {band}")
        lower, upper = TEN_CHOICE_SENSITIVITY_BANDS[band]
        milli = round(float(dynamics["sensitivity"]) * 1000)
        if not lower <= milli <= upper:
            raise ValueError(
                f"ten-choice sensitivity {milli / 1000.0} is outside {band} band"
            )

    first_person_drag_episodes = [
        episode
        for episode in episodes
        if episode["variant"] == "drag_first_person"
    ]
    drag_reset_contracts = Counter(
        episode["environment_config"].get("reset_frame_contract")
        for episode in first_person_drag_episodes
    )
    drag_offscreen_splits = Counter()
    for episode in first_person_drag_episodes:
        dynamics = episode["shared_scene_config"]["hidden_dynamics"]
        if tuple(dynamics.get("direction_xy", ())) != FPS_DIRECTION_XY:
            raise ValueError(
                f"{episode['pair_id']} drag direction must use the fixed FPS mapping"
            )
        environment = episode["environment_config"]
        if environment.get("reset_frame_contract") != "egocentric_target_offscreen":
            continue
        drag_offscreen_splits[episode["split"]] += 1
        shared = episode["shared_scene_config"]
        view_center = environment.get("initial_view_center_xy")
        if not isinstance(view_center, list) or len(view_center) != 2:
            raise ValueError(
                f"{episode['pair_id']} is missing first-person initial view center"
            )
        viewport = tuple(int(value) for value in episode["viewport"])
        zoom = float(shared["hidden_dynamics"]["view_zoom"])
        target_world = shared["slot_center_world_xy"]
        target_screen = (
            viewport[0] / 2.0 + (float(target_world[0]) - float(view_center[0])) * zoom,
            viewport[1] / 2.0 + (float(target_world[1]) - float(view_center[1])) * zoom,
        )
        piece = shared["piece"]
        outer_radius = (
            float(piece["radius_px"]) * float(shared["slot_radius_scale"]) + 14.0
        )
        target_extent = _shape_half_extent(outer_radius, str(piece["shape"]))
        target_fully_outside = (
            target_screen[0] + target_extent[0] < 0
            or target_screen[0] - target_extent[0] > viewport[0]
            or target_screen[1] + target_extent[1] < 0
            or target_screen[1] - target_extent[1] > viewport[1]
        )
        if not target_fully_outside:
            raise ValueError(
                f"{episode['pair_id']} target is not fully outside the first-person reset frame"
            )

    if expected_count == 150:
        if drag_counts != {"iid": 75, "ood": 75}:
            raise ValueError(f"drag split must be 75 IID + 75 OOD, got {dict(drag_counts)}")
        for family in ("ten_choice", "rotation"):
            if family_split_counts[family] != {None: 150}:
                raise ValueError(
                    f"{family} test set must contain 150 episodes with null distribution, got "
                    f"{dict(family_split_counts[family])}"
                )
        if ten_choice_sensitivity_bands != {"low": 75, "high": 75}:
            raise ValueError(
                "ten-choice sensitivity must be 75 low + 75 high, got "
                f"{dict(ten_choice_sensitivity_bands)}"
            )
        shared_reset_contract = (
            "shared_layout_same_mouse_marker"
            if implementation_tag == "6env-v2"
            else "shared_layout_different_interaction_marker"
        )
        expected_reset_contracts = {
            shared_reset_contract: DRAG_SHARED_LAYOUT_RESET_COUNT,
            "egocentric_target_offscreen": DRAG_OFFSCREEN_RESET_COUNT,
        }
        if drag_reset_contracts != expected_reset_contracts:
            raise ValueError(
                f"drag reset contracts must be {expected_reset_contracts}, got "
                f"{dict(drag_reset_contracts)}"
            )
        if drag_offscreen_splits != {"iid": 25, "ood": 25}:
            raise ValueError(
                "offscreen drag cases must be 25 IID + 25 OOD, got "
                f"{dict(drag_offscreen_splits)}"
            )
    return {
        "episode_count": len(episodes),
        "implementation_tag": implementation_tag,
        "variant_counts": dict(sorted(counts.items())),
        "pair_counts": {
            family: len(pair_sets[variants[0]]) for family, variants in family_pairs.items()
        },
        "drag_split": dict(sorted(drag_counts.items())),
        "family_splits": {
            family: dict(sorted(counts.items()))
            for family, counts in family_split_counts.items()
        },
        "ten_choice_sensitivity_bands": dict(
            sorted(ten_choice_sensitivity_bands.items())
        ),
        "first_person_direction_xy": list(FPS_DIRECTION_XY),
        "drag_reset_contracts": dict(sorted(drag_reset_contracts.items())),
        "drag_offscreen_splits": dict(sorted(drag_offscreen_splits.items())),
    }


def build_suite(
    *,
    output_root: Path,
    suite_id: str,
    cases_per_family: int,
    sources: ExplorationSourceRoots | None = None,
) -> dict[str, Any]:
    if cases_per_family not in {1, 150}:
        raise ValueError("exploration-depth suites support one smoke pair or 150 formal pairs")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to replace non-empty suite directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_roots = ExplorationSourceRoots.defaults() if sources is None else sources
    episodes = [
        *_ten_choice_episodes(
            suite_id=suite_id,
            output_root=output_root,
            source_root=source_roots.ten_choice,
            count=cases_per_family,
        ),
        *_rotation_episodes(
            suite_id=suite_id,
            output_root=output_root,
            source_root=source_roots.rotation_dataset,
            background_metadata=source_roots.rotation_background_metadata,
            count=cases_per_family,
        ),
        *_drag_episodes(
            suite_id=suite_id,
            output_root=output_root,
            third_person_source=source_roots.third_person_drag,
            slot_source=source_roots.slot_drag,
            count=cases_per_family,
        ),
    ]
    manifest = {
        "schema": "gui_captcha_exploration_depth_manifest_v1",
        "suite_id": suite_id,
        "benchmark_axis": "environment_exploration_depth",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": cases_per_family == 150,
        "pair_count_per_family": cases_per_family,
        "case_count_per_variant": cases_per_family,
        "total_episode_count": len(episodes),
        "semantic_interaction_contract": (
            "count one state-changing action followed by its observation; mouse_down and mouse_up "
            "are gesture primitives and are not separate exploration interactions"
        ),
        "paired_reset_contract": {
            "ten_choice": (
                "150 of 150 pairs share the same scene outside the interaction marker; "
                "third-person uses a mouse icon and first-person uses a red dot"
            ),
            "rotation": "150 of 150 pairs remain pixel-identical free-mouse slider views",
            "drag": (
                "100 of 150 pairs share the same layout outside the interaction marker; "
                "all third-person frames use a mouse icon and all first-person frames use "
                "a red dot; 50 first-person targets start outside the reset frame"
            ),
        },
        "policy_observation_forbidden_fields": [
            "variant",
            "sensitivity",
            "target_index",
            "target_coordinates",
            "rotation_direction",
            "target_angle",
            "oracle_success_geometry",
        ],
        "episodes": sorted(episodes, key=lambda episode: (episode["variant"], episode["pair_id"])),
    }
    manifest["validation"] = validate_manifest_structure(
        manifest,
        expected_count=cases_per_family,
    )
    _write_json(output_root / "manifest.json", manifest)
    return manifest
