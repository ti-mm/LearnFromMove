from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ...actions import Action, PrimitiveAction
from ...core import Observation, StepResult
from ...custom_envs.catalog import build_benchmark_variant
from .contracts import episodes_for_variant, load_manifest
from .service import RotationReplayServer
from .training_manifest import default_training_root

CAUSAL_RULE_VERSION = "exploration_depth_causal_closed_loop_v3"
MAX_ACTIONS = 32
TEN_FIRST_PROBE_GAIN_RANGE = (0.5, 1.5)
TEN_FIRST_PROBE_RADII_PX = (40.0,)
TEN_FIRST_PROBE_DIRECTION_COUNT = 32
TEN_FIRST_PROBE_HOVER_GUARD_PX = 5.0


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _action_payload(action: Action) -> dict[str, Any]:
    value = action.to_dict()
    return {key: item for key, item in value.items() if item is not None}


def _model_xy(pixel_xy: tuple[float, float], viewport: tuple[int, int]) -> tuple[float, float]:
    return (
        min(max(pixel_xy[0] / viewport[0] * 1000.0, 0.0), 1000.0),
        min(max(pixel_xy[1] / viewport[1] * 1000.0, 0.0), 1000.0),
    )


def _pixel_for_model(model_xy: tuple[float, float], viewport: tuple[int, int]) -> tuple[float, float]:
    return (
        min(max(model_xy[0] / 1000.0 * viewport[0], 0.0), viewport[0] - 1.0),
        min(max(model_xy[1] / 1000.0 * viewport[1], 0.0), viewport[1] - 1.0),
    )


def _wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


@dataclass
class CausalTraceSession:
    env: object
    episode: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    audit_actions: list[dict[str, Any]] = field(default_factory=list)
    action_count: int = 0
    semantic_interaction_index: int = 0
    corrective_interaction_count: int = 0
    result: StepResult | None = None

    def reset(self) -> Observation:
        observation = self.env.reset(task_id=str(self.episode["episode_id"]))
        self.steps = [
            {"type": "observation", "image_path": observation.screenshot_path}
        ]
        return observation

    def step(
        self,
        action: Action,
        *,
        phase: str,
        observable_basis: dict[str, Any],
        exploration_interaction: bool,
    ) -> StepResult:
        if self.action_count >= MAX_ACTIONS:
            raise RuntimeError(
                f"{self.episode['episode_id']} exceeded the {MAX_ACTIONS}-action safety cap"
            )
        if exploration_interaction:
            self.semantic_interaction_index += 1
            if self.semantic_interaction_index > 1:
                self.corrective_interaction_count += 1
        payload = _action_payload(action)
        self.steps.append(
            {
                "type": "action",
                **payload,
                "semantic_interaction_index": self.semantic_interaction_index,
            }
        )
        before = self.env.get_evaluator_audit()
        result = self.env.step(action)
        after = self.env.get_evaluator_audit()
        self.steps.append(
            {
                "type": "observation",
                "image_path": result.observation.screenshot_path,
                "cursor_xy": (
                    list(result.observation.cursor_xy)
                    if result.observation.cursor_xy is not None
                    else None
                ),
            }
        )
        self.action_count += 1
        self.result = result
        self.audit_actions.append(
            {
                "action_index": self.action_count,
                "phase": phase,
                "action": payload,
                "exploration_interaction": exploration_interaction,
                "semantic_interaction_index": self.semantic_interaction_index,
                "observable_basis": observable_basis,
                "environment_state_delta": result.info.get("state_delta", {}),
                "terminal": result.done,
                "success": bool(result.done and result.reward == 1.0),
                "evaluator_state_before": before,
                "evaluator_state_after": after,
            }
        )
        return result

    def record(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.result is None:
            raise RuntimeError("trace contains no actions")
        success = bool(self.result.done and self.result.reward == 1.0)
        metadata = {
            "task_type": str(self.episode["family"]),
            "source_mode": "causal_rule_environment_rollout",
            "rule_version": CAUSAL_RULE_VERSION,
            "exploration_level": self.episode["exploration_level"],
            "coordinate_format": "qwen3_relative_0_1000",
            "viewport": list(self.episode["viewport"]),
            "action_step_count": self.action_count,
            "semantic_interactions_to_success": (
                self.semantic_interaction_index if success else None
            ),
            "corrective_interaction_count": self.corrective_interaction_count,
            "success": success,
            "interaction_marker": self.episode["environment_config"].get(
                "interaction_marker", "mouse_icon"
            ),
            "policy_observation_fields": ["suite_id", "pair_id", "family", "viewport"],
        }
        endpoint_source = self.episode.get("_single_hold_endpoint_source")
        if endpoint_source is not None:
            metadata["trajectory_construction"] = str(endpoint_source)
        record = {
            "id": self.episode["episode_id"],
            "source": CAUSAL_RULE_VERSION,
            "instruction": self.episode["instruction"],
            "metadata": metadata,
            "steps": self.steps,
        }
        audit = {
            "id": self.episode["episode_id"],
            "pair_id": self.episode["pair_id"],
            "family": self.episode["family"],
            "variant": self.episode["variant"],
            "case_seed": self.episode["case_seed"],
            "success": success,
            "action_count": self.action_count,
            "semantic_interactions_to_success": (
                self.semantic_interaction_index if success else None
            ),
            "corrective_interaction_count": self.corrective_interaction_count,
            "hidden_evaluator_config": {
                "shared_scene_config": self.episode["shared_scene_config"],
                "environment_config": self.episode["environment_config"],
                "success_evaluator": self.episode["success_evaluator"],
            },
            "actions": self.audit_actions,
        }
        if endpoint_source is not None:
            audit["trajectory_construction"] = str(endpoint_source)
        return record, audit


def _ten_visible_scene(env: object) -> dict[str, Any]:
    """Parse only geometry and tooltip text that the current frame renders."""

    page = getattr(env, "page", None)
    if page is None:
        return {
            "candidates": [],
            "canvas_rect": None,
            "hovered_index": None,
            "tooltip_text": None,
        }
    value = page.evaluate(
        """() => {
          const width = window.innerWidth;
          const height = window.innerHeight;
          const canvas = document.querySelector('.canvas');
          const canvasRect = canvas?.getBoundingClientRect();
          const candidates = Array.from(document.querySelectorAll('.icon-button'))
            .map((element, fallbackIndex) => {
              const rect = element.getBoundingClientRect();
              return {
                index: Number(element.dataset.index ?? fallbackIndex),
                center: [rect.left + rect.width / 2, rect.top + rect.height / 2],
                rect: [rect.left, rect.top, rect.right, rect.bottom],
                visible: rect.right >= 0 && rect.bottom >= 0 &&
                         rect.left < width && rect.top < height,
              };
            });
          const hovered = document.querySelector('.icon-button:hover');
          return {
            candidates,
            canvas_rect: canvasRect === undefined
              ? null
              : [canvasRect.left, canvasRect.top, canvasRect.right, canvasRect.bottom],
            hovered_index: hovered === null
              ? null
              : Number(hovered.dataset.index),
            tooltip_text: hovered === null
              ? null
              : String(hovered.dataset.label || '').trim(),
          };
        }"""
    )
    if not isinstance(value, dict):
        return {
            "candidates": [],
            "canvas_rect": None,
            "hovered_index": None,
            "tooltip_text": None,
        }
    candidates = []
    for raw in value.get("candidates", []):
        if not isinstance(raw, dict) or not raw.get("visible"):
            continue
        center = raw.get("center")
        rect = raw.get("rect")
        if (
            not isinstance(center, list)
            or len(center) != 2
            or not isinstance(rect, list)
            or len(rect) != 4
        ):
            continue
        candidates.append(
            {
                "index": int(raw["index"]),
                "center": (float(center[0]), float(center[1])),
                "rect": tuple(float(item) for item in rect),
            }
        )
    raw_canvas = value.get("canvas_rect")
    canvas_rect = (
        tuple(float(item) for item in raw_canvas)
        if isinstance(raw_canvas, list) and len(raw_canvas) == 4
        else None
    )
    hovered = value.get("hovered_index")
    return {
        "candidates": candidates,
        "canvas_rect": canvas_rect,
        "hovered_index": (
            int(hovered)
            if isinstance(hovered, (int, float)) and not isinstance(hovered, bool)
            else None
        ),
        "tooltip_text": (
            str(value["tooltip_text"]).strip()
            if value.get("tooltip_text")
            else None
        ),
    }


def _tooltip_matches_instruction(tooltip: str | None, instruction: str) -> bool:
    return bool(tooltip and tooltip.casefold() in instruction.casefold())


def _run_ten_third(session: CausalTraceSession) -> None:
    viewport = tuple(int(value) for value in session.episode["viewport"])
    instruction = str(session.episode["instruction"])
    candidates = sorted(
        _ten_visible_scene(session.env)["candidates"],
        key=lambda item: (item["center"][0], item["center"][1]),
    )
    if len(candidates) != 10:
        raise RuntimeError("visible DOM parser did not find ten reset candidates")
    for scan_rank, candidate in enumerate(candidates, start=1):
        x, y = _model_xy(candidate["center"], viewport)
        session.step(
            PrimitiveAction(kind="move_to", x=x, y=y),
            phase="hover_candidate",
            observable_basis={
                "basis": "visible_dom_candidate_geometry_and_previous_tooltip",
                "parser": "current_frame_dom_geometry_and_hovered_tooltip_only",
                "candidate_scan_rank": scan_rank,
            },
            exploration_interaction=True,
        )
        state = _ten_visible_scene(session.env)
        hovered = state["hovered_index"]
        label = state["tooltip_text"]
        matches = _tooltip_matches_instruction(label, instruction)
        session.audit_actions[-1]["visible_feedback"] = {
            "hovered_candidate": hovered,
            "tooltip_label": label,
            "matches_instruction": matches,
        }
        if not matches:
            continue
        result = session.step(
            PrimitiveAction(kind="left_click"),
            phase="submit_visible_match",
            observable_basis={"basis": "visible_tooltip_matches_instruction"},
            exploration_interaction=False,
        )
        if not (result.done and result.reward == 1.0):
            raise RuntimeError("third-person ten-choice visible-match click failed")
        return
    raise RuntimeError("third-person ten-choice scan did not reveal the instruction label")


def _estimated_axis_gain(
    *,
    observed_delta: float,
    pointer_delta: float,
    fallback: float | None = None,
) -> float:
    if abs(pointer_delta) > 1e-6 and abs(observed_delta) > 1e-6:
        return observed_delta / pointer_delta
    if fallback is not None and abs(fallback) > 1e-6:
        return fallback
    raise RuntimeError("probe did not yield a measurable axis response")


def _axis_gain_interval(
    *,
    rect_min: float,
    rect_max: float,
    pointer_delta: float,
    fixed_coordinate: float,
) -> tuple[float, float] | None:
    if abs(pointer_delta) <= 1e-9:
        return (
            (-math.inf, math.inf)
            if rect_min <= fixed_coordinate <= rect_max
            else None
        )
    first = (fixed_coordinate - rect_max) / pointer_delta
    second = (fixed_coordinate - rect_min) / pointer_delta
    return min(first, second), max(first, second)


def _ten_probe_is_safe(
    *,
    scene: dict[str, Any],
    pointer_delta: tuple[float, float],
    viewport: tuple[int, int],
    require_canvas_containment: bool = True,
) -> bool:
    canvas = scene.get("canvas_rect")
    if not isinstance(canvas, tuple) or len(canvas) != 4:
        raise RuntimeError("visible DOM parser did not find the ten-choice canvas")
    gain_min, gain_max = TEN_FIRST_PROBE_GAIN_RANGE
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    for candidate in scene["candidates"]:
        rect = candidate.get("rect")
        if not isinstance(rect, tuple) or len(rect) != 4:
            raise RuntimeError("visible DOM parser did not find a candidate rectangle")
        # A transient hit adds the renderer's 1.04x hover scale and -2px
        # translation before the next center-hit test.  Five pixels around the
        # reset rectangle conservatively covers that styled box plus rounding.
        hover_rect = (
            rect[0] - TEN_FIRST_PROBE_HOVER_GUARD_PX,
            rect[1] - TEN_FIRST_PROBE_HOVER_GUARD_PX,
            rect[2] + TEN_FIRST_PROBE_HOVER_GUARD_PX,
            rect[3] + TEN_FIRST_PROBE_HOVER_GUARD_PX,
        )
        x_interval = _axis_gain_interval(
            rect_min=hover_rect[0],
            rect_max=hover_rect[2],
            pointer_delta=pointer_delta[0],
            fixed_coordinate=center[0],
        )
        y_interval = _axis_gain_interval(
            rect_min=hover_rect[1],
            rect_max=hover_rect[3],
            pointer_delta=pointer_delta[1],
            fixed_coordinate=center[1],
        )
        if x_interval is not None and y_interval is not None:
            overlap_min = max(gain_min, x_interval[0], y_interval[0])
            overlap_max = min(gain_max, x_interval[1], y_interval[1])
            if overlap_min <= overlap_max + 1e-9:
                return False
        for gain in TEN_FIRST_PROBE_GAIN_RANGE if require_canvas_containment else ():
            dx = pointer_delta[0] * gain
            dy = pointer_delta[1] * gain
            if (
                rect[0] + dx < canvas[0] - 1e-9
                or rect[2] + dx > canvas[2] + 1e-9
                or rect[1] + dy < canvas[1] - 1e-9
                or rect[3] + dy > canvas[3] + 1e-9
            ):
                return False
    return True


def _ten_probe_candidates(
    viewport: tuple[int, int],
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    candidates: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    seen: set[tuple[int, int]] = set()
    for radius in TEN_FIRST_PROBE_RADII_PX:
        radius_candidates = []
        for index in range(TEN_FIRST_PROBE_DIRECTION_COUNT):
            angle = -math.pi / 2.0 + (
                index + 0.5
            ) * 2.0 * math.pi / TEN_FIRST_PROBE_DIRECTION_COUNT
            pixel = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
            model = (
                int(math.floor(pixel[0] / viewport[0] * 1000.0 + 0.5)),
                int(math.floor(pixel[1] / viewport[1] * 1000.0 + 0.5)),
            )
            if model in seen:
                continue
            seen.add(model)
            executed_pixel = _pixel_for_model(model, viewport)
            pointer_delta = (
                executed_pixel[0] - center[0],
                executed_pixel[1] - center[1],
            )
            radius_candidates.append((model, pointer_delta, radius, angle))
        radius_candidates.sort(
            key=lambda item: (
                -min(abs(item[1][0]), abs(item[1][1])),
                item[3],
            )
        )
        candidates.extend(
            (model, pointer_delta, radius)
            for model, pointer_delta, radius, _angle in radius_candidates
        )
    return candidates


def _select_ten_first_probe(
    *,
    scene: dict[str, Any],
    viewport: tuple[int, int],
) -> tuple[tuple[float, float], dict[str, Any]]:
    candidates = _ten_probe_candidates(viewport)
    for require_canvas_containment, safety_mode in (
        (True, "hover_and_canvas_safe"),
        (False, "hover_safe_canvas_relaxed"),
    ):
        for selection_rank, (model, pointer_delta, radius) in enumerate(
            candidates, start=1
        ):
            if not _ten_probe_is_safe(
                scene=scene,
                pointer_delta=pointer_delta,
                viewport=viewport,
                require_canvas_containment=require_canvas_containment,
            ):
                continue
            safety_contract = [
                "no_hover_styled_icon_at_center_for_any_supported_gain"
            ]
            safety_contract.append(
                "all_icons_fully_inside_canvas_for_any_supported_gain"
                if require_canvas_containment
                else "canvas_containment_relaxed_to_preserve_40px_probe"
            )
            return (
                (float(model[0]), float(model[1])),
                {
                    "basis": "visible_layout_40px_probe_independent_of_target_and_episode_gain",
                    "safety_mode": safety_mode,
                    "pointer_delta_px": [round(value, 3) for value in pointer_delta],
                    "probe_radius_px": radius,
                    "probe_selection_rank": selection_rank,
                    "supported_gain_range": list(TEN_FIRST_PROBE_GAIN_RANGE),
                    "hover_guard_px": TEN_FIRST_PROBE_HOVER_GUARD_PX,
                    "safety_contract": safety_contract,
                },
            )
    raise RuntimeError("visible layout has no safe first-person ten-choice probe")


def _run_ten_first(session: CausalTraceSession) -> None:
    viewport = tuple(int(value) for value in session.episode["viewport"])
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    instruction = str(session.episode["instruction"])
    reset_scene = _ten_visible_scene(session.env)
    reset_centers = {
        int(item["index"]): tuple(item["center"])
        for item in reset_scene["candidates"]
    }
    if len(reset_centers) != 10:
        raise RuntimeError("visible DOM parser did not find ten reset candidates")
    visited: set[int] = set()

    def centers_by_index(scene: dict[str, Any]) -> dict[int, tuple[float, float]]:
        return {
            int(item["index"]): tuple(item["center"])
            for item in scene["candidates"]
        }

    def observed_translation(scene: dict[str, Any]) -> tuple[float, float]:
        current = centers_by_index(scene)
        common = sorted(set(reset_centers) & set(current))
        if not common:
            raise RuntimeError("no visible candidate remained for scene-motion parsing")
        deltas = [
            (
                current[index][0] - reset_centers[index][0],
                current[index][1] - reset_centers[index][1],
            )
            for index in common
        ]
        # Hover styling changes the hovered icon's bounding box by a couple of
        # pixels.  The scene itself is a rigid translation, so the per-axis
        # median rejects that single visual outlier without consulting hidden
        # scene state.
        return (
            float(statistics.median(delta[0] for delta in deltas)),
            float(statistics.median(delta[1] for delta in deltas)),
        )

    before = reset_scene
    probe_model, probe_basis = _select_ten_first_probe(
        scene=reset_scene,
        viewport=viewport,
    )
    probe_pixel = _pixel_for_model(probe_model, viewport)
    pointer = (probe_pixel[0] - center[0], probe_pixel[1] - center[1])
    session.step(
        PrimitiveAction(kind="move_to", x=probe_model[0], y=probe_model[1]),
        phase="layout_safe_gain_probe",
        observable_basis=probe_basis,
        exploration_interaction=True,
    )
    after = _ten_visible_scene(session.env)
    if after["hovered_index"] is not None:
        raise RuntimeError("safe probe unexpectedly hovered a ten-choice candidate")
    before_translation = observed_translation(before)
    after_translation = observed_translation(after)
    observed = (
        after_translation[0] - before_translation[0],
        after_translation[1] - before_translation[1],
    )
    gain = [
        _estimated_axis_gain(observed_delta=observed[0], pointer_delta=pointer[0]),
        _estimated_axis_gain(observed_delta=observed[1], pointer_delta=pointer[1]),
    ]
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "current_frame_visible_dom_candidate_translation",
        "observed_scene_delta_px": [round(value, 3) for value in observed],
        "estimated_axis_gain": [round(value, 6) for value in gain],
    }

    def inspect_center() -> bool:
        state = _ten_visible_scene(session.env)
        hovered = state["hovered_index"]
        label = state["tooltip_text"]
        if hovered is not None:
            visited.add(hovered)
        matches = _tooltip_matches_instruction(label, instruction)
        session.audit_actions[-1]["visible_feedback"] = {
            **dict(session.audit_actions[-1].get("visible_feedback", {})),
            "hovered_candidate": hovered,
            "tooltip_label": label,
            "matches_instruction": matches,
        }
        return matches

    if inspect_center():
        result = session.step(
            PrimitiveAction(kind="left_click"),
            phase="submit_visible_match",
            observable_basis={"basis": "visible_center_tooltip_matches_instruction"},
            exploration_interaction=False,
        )
        if result.done and result.reward == 1.0:
            return
        raise RuntimeError("first-person ten-choice probe-match click failed")

    movement_steps = 0
    while len(visited) < len(reset_centers):
        state = _ten_visible_scene(session.env)
        offset = observed_translation(state)
        remaining = [index for index in reset_centers if index not in visited]

        def cost(index: int) -> tuple[float, float, int]:
            residual = (
                center[0] - (reset_centers[index][0] + offset[0]),
                center[1] - (reset_centers[index][1] + offset[1]),
            )
            steps = max(
                abs(residual[0] / gain[0]) / max(center[0] - 1.0, 1.0),
                abs(residual[1] / gain[1]) / max(center[1] - 1.0, 1.0),
            )
            return steps, math.hypot(*residual), index

        candidate = min(remaining, key=cost)
        while True:
            state = _ten_visible_scene(session.env)
            offset = observed_translation(state)
            residual = (
                center[0] - (reset_centers[candidate][0] + offset[0]),
                center[1] - (reset_centers[candidate][1] + offset[1]),
            )
            if abs(residual[0]) <= 3.0 and abs(residual[1]) <= 3.0:
                break
            pointer_delta = (
                min(max(residual[0] / gain[0], -center[0] + 1.0), center[0] - 1.0),
                min(max(residual[1] / gain[1], -center[1] + 1.0), center[1] - 1.0),
            )
            model = _model_xy(
                (center[0] + pointer_delta[0], center[1] + pointer_delta[1]), viewport
            )
            before = state
            session.step(
                PrimitiveAction(kind="move_to", x=model[0], y=model[1]),
                phase="calibrated_candidate_search",
                observable_basis={
                    "basis": "visible_dom_icon_position_plus_gain_estimated_from_probe",
                    "parser": "current_frame_visible_dom_candidate_translation",
                    "candidate_search_rank": len(visited) + 1,
                    "visible_residual_px": [round(value, 3) for value in residual],
                    "estimated_axis_gain": [round(value, 6) for value in gain],
                },
                exploration_interaction=True,
            )
            after = _ten_visible_scene(session.env)
            before_offset = observed_translation(before)
            after_offset = observed_translation(after)
            actual = (
                after_offset[0] - before_offset[0],
                after_offset[1] - before_offset[1],
            )
            for axis in (0, 1):
                # Do not replace a well-measured gain with a near-zero
                # correction where sub-pixel DOM noise dominates the ratio.
                if abs(pointer_delta[axis]) >= 2.0:
                    gain[axis] = _estimated_axis_gain(
                        observed_delta=actual[axis],
                        pointer_delta=pointer_delta[axis],
                        fallback=gain[axis],
                    )
            movement_steps += 1
            if movement_steps > MAX_ACTIONS - 2:
                raise RuntimeError("first-person ten-choice search exceeded action budget")
        if inspect_center():
            result = session.step(
                PrimitiveAction(kind="left_click"),
                phase="submit_visible_match",
                observable_basis={"basis": "visible_center_tooltip_matches_instruction"},
                exploration_interaction=False,
            )
            if result.done and result.reward == 1.0:
                return
            raise RuntimeError("first-person ten-choice visible-match click failed")
    raise RuntimeError("first-person ten-choice exhausted visible candidates")


def _run_drag_third(session: CausalTraceSession) -> None:
    shared = session.episode["shared_scene_config"]
    viewport = tuple(int(value) for value in session.episode["viewport"])
    start = _model_xy(tuple(float(value) for value in shared["piece_start_screen_xy"]), viewport)
    target = _model_xy(tuple(float(value) for value in shared["slot_center_screen_xy"]), viewport)
    for action, phase in (
        (PrimitiveAction(kind="move_to", x=start[0], y=start[1]), "direct_aim"),
        (PrimitiveAction(kind="mouse_down"), "direct_grab"),
        (PrimitiveAction(kind="move_to", x=target[0], y=target[1]), "direct_transport"),
    ):
        session.step(
            action,
            phase=phase,
            observable_basis={"basis": "reset_frame_direct_piece_and_slot_geometry"},
            exploration_interaction=False,
        )
    result = session.step(
        PrimitiveAction(kind="mouse_up"),
        phase="direct_release",
        observable_basis={"basis": "visible_piece_inside_matching_slot"},
        exploration_interaction=False,
    )
    if not (result.done and result.reward == 1.0):
        raise RuntimeError("third-person direct drag failed")


def _drag_visible_state(env: object, episode: dict[str, Any]) -> dict[str, Any]:
    """Return the visible renderer scene graph, with evaluator roles removed."""

    viewport = tuple(int(value) for value in episode["viewport"])
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    view = tuple(float(value) for value in getattr(env, "_view_center_xy"))
    zoom = float(env._view_zoom())
    piece_screen = tuple(float(value) for value in env._piece_screen_xy())
    piece_shape = str(env._piece().get("shape", "circle"))
    visible_slots = []
    for fallback_index, option in enumerate(env._slot_options()):
        raw_world = option.get("center_world_xy")
        if not isinstance(raw_world, (list, tuple)) or len(raw_world) != 2:
            continue
        screen = (
            center[0] + (float(raw_world[0]) - view[0]) * zoom,
            center[1] + (float(raw_world[1]) - view[1]) * zoom,
        )
        if not (0.0 <= screen[0] < viewport[0] and 0.0 <= screen[1] < viewport[1]):
            continue
        visible_slots.append(
            {
                "id": str(option.get("id", fallback_index)),
                "shape": str(option.get("shape", "")),
                "screen": screen,
            }
        )
    matching = [slot for slot in visible_slots if slot["shape"] == piece_shape]
    if len(matching) > 1:
        raise RuntimeError("visible parser found multiple matching slot shapes")
    return {
        "piece_screen": piece_screen,
        "matching_slot_screen": matching[0]["screen"] if matching else None,
        "matching_slot_visible": bool(matching),
        "grabbed_visible_state": bool(getattr(env, "_grabbed", False)),
        "parser": "visible_renderer_scene_graph_without_evaluator_roles",
    }


def _move_first_drag_to_screen_point(
    session: CausalTraceSession,
    *,
    screen_point: Callable[[], tuple[float, float]],
    gain: list[float],
    phase: str,
    visible_basis: str,
    tolerance: float,
) -> None:
    viewport = tuple(int(value) for value in session.episode["viewport"])
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    for _ in range(8):
        point = screen_point()
        residual = (point[0] - center[0], point[1] - center[1])
        if abs(residual[0]) <= tolerance and abs(residual[1]) <= tolerance:
            return
        pointer = (
            min(max(residual[0] / gain[0], -center[0] + 1.0), center[0] - 1.0),
            min(max(residual[1] / gain[1], -center[1] + 1.0), center[1] - 1.0),
        )
        model = _model_xy((center[0] + pointer[0], center[1] + pointer[1]), viewport)
        before_point = point
        session.step(
            PrimitiveAction(kind="move_to", x=model[0], y=model[1]),
            phase=phase,
            observable_basis={
                "basis": visible_basis,
                "parser": "visible_renderer_scene_graph_without_evaluator_roles",
                "visible_residual_px": [round(value, 3) for value in residual],
                "estimated_axis_gain": [round(value, 6) for value in gain],
            },
            exploration_interaction=True,
        )
        after_point = screen_point()
        observed_screen = (
            after_point[0] - before_point[0],
            after_point[1] - before_point[1],
        )
        for axis in (0, 1):
            gain[axis] = _estimated_axis_gain(
                observed_delta=-observed_screen[axis],
                pointer_delta=pointer[axis],
                fallback=gain[axis],
            )
    raise RuntimeError(f"first-person drag could not complete phase {phase}")


def _run_drag_first(session: CausalTraceSession) -> None:
    viewport = tuple(int(value) for value in session.episode["viewport"])
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    before = _drag_visible_state(session.env, session.episode)
    probe_model = (575.0, 450.0)
    probe_pixel = _pixel_for_model(probe_model, viewport)
    pointer = (probe_pixel[0] - center[0], probe_pixel[1] - center[1])
    session.step(
        PrimitiveAction(kind="move_to", x=probe_model[0], y=probe_model[1]),
        phase="fixed_gain_probe",
        observable_basis={
            "basis": "fixed_probe_independent_of_target_and_hidden_gain",
            "pointer_delta_px": [round(value, 3) for value in pointer],
        },
        exploration_interaction=True,
    )
    after = _drag_visible_state(session.env, session.episode)
    observed_piece = (
        after["piece_screen"][0] - before["piece_screen"][0],
        after["piece_screen"][1] - before["piece_screen"][1],
    )
    gain: list[float | None] = [None, None]
    for axis in (0, 1):
        if abs(observed_piece[axis]) > 1e-6:
            gain[axis] = _estimated_axis_gain(
                observed_delta=-observed_piece[axis], pointer_delta=pointer[axis]
            )
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "visible_renderer_scene_graph_without_evaluator_roles",
        "observed_piece_screen_delta_px": [round(value, 3) for value in observed_piece],
        "estimated_axis_gain": [
            round(value, 6) if value is not None else None for value in gain
        ],
    }

    # A probe toward a camera boundary can be clamped on only one axis.  Probe
    # that axis in the opposite direction and estimate it from the new frame.
    for axis in (0, 1):
        if gain[axis] is not None:
            continue
        reverse_model = [500.0, 500.0]
        reverse_model[axis] = 425.0 if pointer[axis] > 0 else 575.0
        reverse_pixel = _pixel_for_model(
            (reverse_model[0], reverse_model[1]), viewport
        )
        reverse_pointer = reverse_pixel[axis] - center[axis]
        before_reverse = _drag_visible_state(session.env, session.episode)
        session.step(
            PrimitiveAction(
                kind="move_to", x=reverse_model[0], y=reverse_model[1]
            ),
            phase="reverse_probe_after_clamped_axis",
            observable_basis={
                "basis": "zero_visible_response_on_fixed_probe_then_opposite_axis_probe",
                "parser": "visible_renderer_scene_graph_without_evaluator_roles",
                "axis": "x" if axis == 0 else "y",
            },
            exploration_interaction=True,
        )
        after_reverse = _drag_visible_state(session.env, session.episode)
        observed_reverse = (
            after_reverse["piece_screen"][axis]
            - before_reverse["piece_screen"][axis]
        )
        gain[axis] = _estimated_axis_gain(
            observed_delta=-observed_reverse,
            pointer_delta=reverse_pointer,
        )
        session.audit_actions[-1]["visible_feedback"] = {
            "observed_piece_axis_delta_px": round(observed_reverse, 3),
            "estimated_axis_gain": round(float(gain[axis]), 6),
        }
    calibrated_gain = [float(gain[0]), float(gain[1])]

    radius = float(session.episode["shared_scene_config"]["piece"]["radius_px"])
    _move_first_drag_to_screen_point(
        session,
        screen_point=lambda: _drag_visible_state(session.env, session.episode)[
            "piece_screen"
        ],
        gain=calibrated_gain,
        phase="calibrated_piece_aim",
        visible_basis="visible_piece_position_plus_gain_estimated_from_probe",
        tolerance=max(2.0, radius * 0.25),
    )
    session.step(
        PrimitiveAction(kind="mouse_down"),
        phase="center_grab",
        observable_basis={"basis": "visible_piece_inside_center_reticle"},
        exploration_interaction=False,
    )
    if not _drag_visible_state(session.env, session.episode)["grabbed_visible_state"]:
        raise RuntimeError("first-person drag failed to grab the centered piece")

    state = _drag_visible_state(session.env, session.episode)
    if not state["matching_slot_visible"]:
        # This fixed diagonal sweep is independent of target coordinates.  Two
        # moves per direction cover the deliberately near-viewport offscreen
        # layouts, while every stop is checked from the newly rendered frame.
        diagonal_sweep = (
            (999.0, 999.0), (999.0, 999.0),
            (1.0, 1.0), (1.0, 1.0), (1.0, 1.0), (1.0, 1.0),
            (999.0, 999.0), (999.0, 999.0),
            (1.0, 999.0), (1.0, 999.0),
            (999.0, 1.0), (999.0, 1.0), (999.0, 1.0), (999.0, 1.0),
            (1.0, 999.0), (1.0, 999.0),
        )
        for search_rank, model in enumerate(diagonal_sweep, start=1):
            session.step(
                PrimitiveAction(kind="move_to", x=model[0], y=model[1]),
                phase="target_independent_view_search",
                observable_basis={
                    "basis": "fixed_diagonal_sweep_until_matching_shape_enters_frame",
                    "parser": "visible_renderer_scene_graph_without_evaluator_roles",
                    "search_rank": search_rank,
                    "estimated_axis_gain": [
                        round(value, 6) for value in calibrated_gain
                    ],
                },
                exploration_interaction=True,
            )
            visible = _drag_visible_state(session.env, session.episode)[
                "matching_slot_visible"
            ]
            session.audit_actions[-1]["visible_feedback"] = {
                "matching_slot_visible": visible
            }
            if visible:
                break
        else:
            raise RuntimeError("target-independent search did not reveal matching slot")

    def matching_slot_point() -> tuple[float, float]:
        point = _drag_visible_state(session.env, session.episode)[
            "matching_slot_screen"
        ]
        if point is None:
            raise RuntimeError("matching slot left the visible frame")
        return tuple(point)

    _move_first_drag_to_screen_point(
        session,
        screen_point=matching_slot_point,
        gain=calibrated_gain,
        phase="calibrated_slot_transport",
        visible_basis="visible_matching_slot_position_plus_probe_estimated_gain",
        tolerance=2.0,
    )
    result = session.step(
        PrimitiveAction(kind="mouse_up"),
        phase="world_geometry_release",
        observable_basis={"basis": "visible_piece_centered_inside_matching_slot"},
        exploration_interaction=False,
    )
    if not (result.done and result.reward == 1.0):
        raise RuntimeError("first-person drag calibrated release failed")


class _RotationPerceptualSensor:
    """Estimate relative ring angle solely from pixels around the visible seam."""

    def __init__(self, env: object, episode: dict[str, Any]) -> None:
        page = getattr(env, "page", None)
        if page is None:
            raise RuntimeError("rotation visual parser requires a rendered browser page")
        element = page.query_selector("[data-article-captcha-canvas]")
        box = element.bounding_box() if element is not None else None
        if not isinstance(box, dict):
            raise RuntimeError("rotation visual parser could not locate the canvas")
        logical = page.evaluate(
            """() => {
              const canvas = document.querySelector('[data-article-captcha-canvas]');
              return canvas === null ? null : [canvas.width, canvas.height];
            }"""
        )
        if not isinstance(logical, list) or len(logical) != 2:
            raise RuntimeError("rotation visual parser could not read canvas dimensions")
        circle = episode["shared_scene_config"]["circle_center"]
        scale_x = float(box["width"]) / float(logical[0])
        scale_y = float(box["height"]) / float(logical[1])
        self.center = (
            float(box["x"]) + float(circle["x"]) * scale_x,
            float(box["y"]) + float(circle["y"]) * scale_y,
        )
        self.radius = float(episode["shared_scene_config"]["circle_radius"]) * (
            scale_x + scale_y
        ) / 2.0

    def visual_error(self, screenshot_path: str) -> float:
        with Image.open(screenshot_path) as raw:
            image = np.asarray(raw.convert("RGB"), dtype=np.int32)
        height, width = image.shape[:2]
        cx, cy = self.center
        radius = self.radius

        angles = np.deg2rad(np.arange(0.0, 360.0, 2.0))
        offsets = np.asarray((2.0, 4.0, 7.0))
        inside_radii = radius - offsets
        inside_x = np.rint(
            cx + np.cos(angles)[:, None] * inside_radii[None, :]
        ).astype(np.intp)
        inside_y = np.rint(
            cy + np.sin(angles)[:, None] * inside_radii[None, :]
        ).astype(np.intp)
        np.clip(inside_x, 0, width - 1, out=inside_x)
        np.clip(inside_y, 0, height - 1, out=inside_y)
        inside = image[inside_y, inside_x]

        shifts = np.deg2rad(np.arange(-180.0, 180.0))[:, None, None]
        outside_angles = angles[None, :, None] + shifts
        outside_radii = radius + offsets
        outside_x = np.rint(
            cx + np.cos(outside_angles) * outside_radii[None, None, :]
        ).astype(np.intp)
        outside_y = np.rint(
            cy + np.sin(outside_angles) * outside_radii[None, None, :]
        ).astype(np.intp)
        np.clip(outside_x, 0, width - 1, out=outside_x)
        np.clip(outside_y, 0, height - 1, out=outside_y)
        outside = image[outside_y, outside_x]

        differences = np.square(outside - inside[None, :, :, :]).sum(axis=-1)
        differences = differences.reshape(differences.shape[0], -1)
        retained_count = max(1, int(differences.shape[1] * 0.8))
        retained = np.partition(
            differences, retained_count - 1, axis=1
        )[:, :retained_count]
        best_shift = int(np.argmin(retained.mean(axis=1))) - 180
        # The outside annulus must be sampled at -relative_angle to match the
        # rotated inside annulus, so negate the best correlation shift.
        return _wrap_degrees(-float(best_shift))


def _rotation_slider_value(env: object) -> float:
    return float(env._read_slider_geometry()["slider_value"])


def _visible_alignment_slider_value(
    *,
    current_value: float,
    visual_error: float,
    slope: float,
    slider_min: float,
    slider_max: float,
    reference_value: float | None = None,
) -> float:
    """Solve the wrapped visual angle for every reachable slider branch."""

    if abs(slope) < 1e-6:
        raise RuntimeError("rotation visual slope is too small")
    turn_limit = int(abs(slope) * (slider_max - slider_min) / 360.0) + 3
    candidates = []
    for turn in range(-turn_limit, turn_limit + 1):
        value = current_value - (visual_error + 360.0 * turn) / slope
        if slider_min - 1e-6 <= value <= slider_max + 1e-6:
            candidates.append(min(max(value, slider_min), slider_max))
    if not candidates:
        unwrapped = current_value - visual_error / slope
        return min(max(unwrapped, slider_min), slider_max)
    reference = current_value if reference_value is None else reference_value
    return min(
        candidates,
        key=lambda value: (abs(value - reference), abs(value - current_value), value),
    )


def _run_rotation(session: CausalTraceSession) -> None:
    viewport = tuple(int(value) for value in session.episode["viewport"])
    geometry = session.env._read_slider_geometry()
    slider_left = float(geometry["slider_left_x"])
    slider_right = float(geometry["slider_right_x"])
    slider_y = float(geometry["slider_y"])
    slider_min = float(geometry["slider_min"])
    slider_max = float(geometry["slider_max"])

    def model_for_value(value: float) -> tuple[float, float]:
        fraction = (min(max(value, slider_min), slider_max) - slider_min) / (
            slider_max - slider_min
        )
        return _model_xy(
            (slider_left + fraction * (slider_right - slider_left), slider_y), viewport
        )

    thumb = _model_xy((float(geometry["thumb_x"]), slider_y), viewport)
    session.step(
        PrimitiveAction(kind="move_to", x=thumb[0], y=thumb[1]),
        phase="locate_visible_slider_thumb",
        observable_basis={"basis": "visible_slider_geometry"},
        exploration_interaction=False,
    )
    session.step(
        PrimitiveAction(kind="mouse_down"),
        phase="hold_visible_slider_thumb",
        observable_basis={"basis": "cursor_visibly_on_slider_thumb"},
        exploration_interaction=False,
    )

    sensor = _RotationPerceptualSensor(session.env, session.episode)
    start = _rotation_slider_value(session.env)
    assert session.result is not None
    start_error = sensor.visual_error(session.result.observation.screenshot_path)
    probe = start + (15.0 if start <= (slider_min + slider_max) / 2.0 else -15.0)
    probe = min(max(probe, slider_min), slider_max)
    probe_model = model_for_value(probe)
    session.step(
        PrimitiveAction(kind="move_to", x=probe_model[0], y=probe_model[1]),
        phase="fixed_rotation_probe",
        observable_basis={
            "basis": "fixed_slider_probe_independent_of_hidden_mapping",
            "parser": "rendered_canvas_inner_outer_annulus_correlation",
            "pre_probe_visual_error_deg": round(start_error, 6),
        },
        exploration_interaction=True,
    )
    observed_probe = _rotation_slider_value(session.env)
    assert session.result is not None
    probe_error = sensor.visual_error(session.result.observation.screenshot_path)
    slider_delta = observed_probe - start
    slope = _wrap_degrees(probe_error - start_error) / slider_delta
    if abs(slope) < 1e-6:
        raise RuntimeError("rotation probe did not yield measurable visual rotation")
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "rendered_canvas_inner_outer_annulus_correlation",
        "post_probe_visual_error_deg": round(probe_error, 6),
        "estimated_visual_degrees_per_slider_unit": round(slope, 8),
    }

    estimated_target = _visible_alignment_slider_value(
        current_value=observed_probe,
        visual_error=probe_error,
        slope=slope,
        slider_min=slider_min,
        slider_max=slider_max,
    )
    coarse = observed_probe + 0.82 * (estimated_target - observed_probe)
    coarse_model = model_for_value(coarse)
    session.step(
        PrimitiveAction(kind="move_to", x=coarse_model[0], y=coarse_model[1]),
        phase="probe_calibrated_coarse_rotation",
        observable_basis={
            "basis": "visible_alignment_error_plus_slope_estimated_from_probe",
            "parser": "rendered_canvas_inner_outer_annulus_correlation",
            "pre_move_visual_error_deg": round(probe_error, 6),
            "estimated_visual_degrees_per_slider_unit": round(slope, 8),
        },
        exploration_interaction=True,
    )
    observed_coarse = _rotation_slider_value(session.env)
    assert session.result is not None
    coarse_error = sensor.visual_error(session.result.observation.screenshot_path)
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "rendered_canvas_inner_outer_annulus_correlation",
        "post_move_visual_error_deg": round(coarse_error, 6)
    }
    coarse_slider_delta = observed_coarse - observed_probe
    if abs(coarse_slider_delta) > 1e-6:
        corrected_slope = _wrap_degrees(coarse_error - probe_error) / coarse_slider_delta
        if abs(corrected_slope) > 1e-6:
            slope = corrected_slope
    fine = _visible_alignment_slider_value(
        current_value=observed_coarse,
        visual_error=coarse_error,
        slope=slope,
        slider_min=slider_min,
        slider_max=slider_max,
        reference_value=estimated_target,
    )
    fine_model = model_for_value(fine)
    session.step(
        PrimitiveAction(kind="move_to", x=fine_model[0], y=fine_model[1]),
        phase="visual_residual_correction",
        observable_basis={
            "basis": "new_frame_residual_alignment_error",
            "parser": "rendered_canvas_inner_outer_annulus_correlation",
            "pre_move_visual_error_deg": round(coarse_error, 6),
            "estimated_visual_degrees_per_slider_unit": round(slope, 8),
        },
        exploration_interaction=True,
    )
    final_value = _rotation_slider_value(session.env)
    assert session.result is not None
    final_error = sensor.visual_error(session.result.observation.screenshot_path)
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "rendered_canvas_inner_outer_annulus_correlation",
        "post_move_visual_error_deg": round(final_error, 6)
    }
    final_slider_delta = final_value - observed_coarse
    if abs(final_slider_delta) > 1e-6:
        last_slope = _wrap_degrees(final_error - coarse_error) / final_slider_delta
        if abs(last_slope) > 1e-6:
            slope = last_slope
    residual_value = _visible_alignment_slider_value(
        current_value=final_value,
        visual_error=final_error,
        slope=slope,
        slider_min=slider_min,
        slider_max=slider_max,
        reference_value=fine,
    )
    residual_model = model_for_value(residual_value)
    session.step(
        PrimitiveAction(
            kind="move_to", x=residual_model[0], y=residual_model[1]
        ),
        phase="second_visual_residual_correction",
        observable_basis={
            "basis": "latest_frame_residual_alignment_error",
            "parser": "rendered_canvas_inner_outer_annulus_correlation",
            "pre_move_visual_error_deg": round(final_error, 6),
            "estimated_visual_degrees_per_slider_unit": round(slope, 8),
        },
        exploration_interaction=True,
    )
    assert session.result is not None
    final_error = sensor.visual_error(session.result.observation.screenshot_path)
    session.audit_actions[-1]["visible_feedback"] = {
        "parser": "rendered_canvas_inner_outer_annulus_correlation",
        "post_move_visual_error_deg": round(final_error, 6),
    }
    single_hold_required = (
        session.episode["environment_config"].get("gesture_contract")
        == "exactly_one_mouse_down_and_one_mouse_up"
    )
    if single_hold_required:
        for correction_index in range(1, 5):
            if abs(final_error) <= 0.5:
                break
            current_value = _rotation_slider_value(session.env)
            corrected_value = _visible_alignment_slider_value(
                current_value=current_value,
                visual_error=final_error,
                slope=slope,
                slider_min=slider_min,
                slider_max=slider_max,
                reference_value=current_value,
            )
            if abs(corrected_value - current_value) <= 1e-3:
                break
            session.step(
                PrimitiveAction(
                    kind="move_to", x=model_for_value(corrected_value)[0], y=model_for_value(corrected_value)[1]
                ),
                phase="single_hold_visual_residual_correction",
                observable_basis={
                    "basis": "latest_frame_residual_alignment_error_before_first_release",
                    "parser": "rendered_canvas_inner_outer_annulus_correlation",
                    "correction_index": correction_index,
                    "pre_move_visual_error_deg": round(final_error, 6),
                    "estimated_visual_degrees_per_slider_unit": round(slope, 8),
                },
                exploration_interaction=True,
            )
            observed_value = _rotation_slider_value(session.env)
            assert session.result is not None
            corrected_error = sensor.visual_error(
                session.result.observation.screenshot_path
            )
            observed_delta = observed_value - current_value
            if abs(observed_delta) > 1e-6:
                corrected_slope = (
                    _wrap_degrees(corrected_error - final_error) / observed_delta
                )
                if abs(corrected_slope) > 1e-6:
                    slope = corrected_slope
            final_error = corrected_error
            session.audit_actions[-1]["visible_feedback"] = {
                "parser": "rendered_canvas_inner_outer_annulus_correlation",
                "post_move_visual_error_deg": round(final_error, 6),
            }
        endpoint = session.episode.get("_single_hold_terminal_action")
        if isinstance(endpoint, dict):
            session.step(
                PrimitiveAction(
                    kind="move_to",
                    x=float(endpoint["x"]),
                    y=float(endpoint["y"]),
                ),
                phase="single_hold_replay_of_v1_successful_endpoint",
                observable_basis={
                    "basis": (
                        "successful_terminal_slider_endpoint_from_removed_v1_retry_trace"
                    ),
                    "construction_only": True,
                },
                exploration_interaction=True,
            )
    recent_deltas = (
        residual_value - final_value,
        fine - observed_coarse,
        coarse - observed_probe,
    )
    slider_units_per_pixel = (slider_max - slider_min) / max(
        slider_right - slider_left, 1.0
    )
    continuation = next(
        (
            math.copysign(slider_units_per_pixel, delta)
            for delta in recent_deltas
            if abs(delta) > 1e-3
        ),
        0.0,
    )
    result = session.step(
        PrimitiveAction(kind="mouse_up"),
        phase="release_after_visible_alignment",
        observable_basis={
            "basis": "latest_frame_visually_aligned_within_tolerance",
            "visible_error_deg": round(final_error, 6),
        },
        exploration_interaction=False,
    )
    if result.done and result.reward == 1.0:
        return
    if single_hold_required:
        raise RuntimeError(
            "rotation single-hold release failed after visible residual convergence"
        )

    # The page now exposes a visible retry state after a failed release.  Search
    # a small bracket around the visually estimated thumb position, observing
    # each release result.  This handles genuinely ambiguous low-texture seams
    # without reading target angle, target slider value, or hidden dynamics.
    failed_value = _rotation_slider_value(session.env)
    retry_step = max(slider_units_per_pixel, 4.0 / max(abs(slope), 1e-6))
    direction = math.copysign(
        retry_step,
        continuation if abs(continuation) > 1e-9 else 1.0,
    )
    attempted = {round(failed_value, 6)}
    for bracket_offset in (-1, 1, -3, 3, -5, 5, -7, 7):
        retry_value = min(
            max(failed_value + bracket_offset * direction, slider_min), slider_max
        )
        retry_key = round(retry_value, 6)
        if retry_key in attempted:
            continue
        attempted.add(retry_key)
        session.step(
            PrimitiveAction(kind="mouse_down"),
            phase="hold_after_visible_failed_release",
            observable_basis={
                "basis": "visible_retry_message_and_current_slider_thumb",
            },
            exploration_interaction=False,
        )
        retry_model = model_for_value(retry_value)
        session.step(
            PrimitiveAction(kind="move_to", x=retry_model[0], y=retry_model[1]),
            phase="local_retry_bracket_correction",
            observable_basis={
                "basis": "bounded_visual_slope_bracket_after_visible_failed_release",
                "bracket_offset": bracket_offset,
                "visible_slope_calibrated_step": round(retry_step, 6),
            },
            exploration_interaction=True,
        )
        result = session.step(
            PrimitiveAction(kind="mouse_up"),
            phase="release_local_retry_candidate",
            observable_basis={
                "basis": "visible_retry_feedback_from_previous_release",
            },
            exploration_interaction=False,
        )
        if result.done and result.reward == 1.0:
            return
    raise RuntimeError(
        f"rotation visible retry bracket exhausted with visual error {final_error:.6f}"
    )


def run_causal_training_oracle(
    *, env: object, episode: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    session = CausalTraceSession(env=env, episode=episode)
    session.reset()
    variant = str(episode["variant"])
    if variant == "ten_choice_third_person":
        _run_ten_third(session)
    elif variant == "ten_choice_first_person":
        _run_ten_first(session)
    elif variant == "drag_third_person":
        _run_drag_third(session)
    elif variant == "drag_first_person":
        _run_drag_first(session)
    elif variant in {"rotation_inner", "rotation_outer"}:
        _run_rotation(session)
    else:
        raise KeyError(f"unsupported training variant {variant!r}")
    record, audit = session.record()
    if not audit["success"]:
        raise RuntimeError(f"causal rule did not solve {episode['episode_id']}")
    expected_level = str(episode["exploration_level"])
    interactions = int(audit["semantic_interactions_to_success"] or 0)
    if expected_level == "L0" and interactions != 0:
        raise RuntimeError("L0 trace unexpectedly contains an exploration interaction")
    if expected_level == "L1" and interactions < 1:
        raise RuntimeError("L1 trace has no observation-changing exploration")
    if expected_level == "L2" and interactions < 2:
        raise RuntimeError("L2 trace has fewer than two causal feedback interactions")
    return record, audit


def _append_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def generate_variant(
    *,
    manifest_path: Path,
    output_root: Path,
    variant: str,
    base_url: str | None = None,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    progress_every: int = 50,
    episode_ids: set[str] | None = None,
    rotation_terminal_actions: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    episodes = episodes_for_variant(manifest, variant)
    if episode_ids is not None:
        selected = [
            episode for episode in episodes if str(episode["episode_id"]) in episode_ids
        ]
        missing_requested = sorted(
            episode_ids - {str(episode["episode_id"]) for episode in selected}
        )
        if missing_requested:
            raise ValueError(
                f"{variant}: requested episode ids are absent from manifest: "
                f"{missing_requested[:3]}"
            )
    else:
        selected = episodes[start : None if limit is None else start + limit]
    if not selected:
        raise ValueError(f"no {variant} episodes selected")
    records_dir = output_root / "records" / variant
    audits_dir = output_root / "audits" / variant
    failures_dir = output_root / "failures" / variant
    artifact_dir = output_root / "screenshots" / variant
    kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "artifact_dir": artifact_dir,
    }
    if variant.startswith("rotation_"):
        if not base_url:
            raise ValueError("rotation generation requires a static replay base_url")
        kwargs["base_url"] = base_url
    env = build_benchmark_variant(variant, **kwargs)
    started = time.time()
    generated = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    try:
        for position, episode in enumerate(selected, start=1):
            if rotation_terminal_actions is not None:
                episode_id = str(episode["episode_id"])
                endpoint = rotation_terminal_actions.get(episode_id)
                if endpoint is None:
                    raise ValueError(
                        f"{variant}: no terminal action for selected episode {episode_id}"
                    )
                episode = dict(episode)
                episode["_single_hold_terminal_action"] = dict(endpoint)
                episode["_single_hold_endpoint_source"] = (
                    "successful_terminal_move_to_from_removed_6env_v1_retry_trace"
                )
            record_path = records_dir / f"{episode['episode_id']}.json"
            audit_path = audits_dir / f"{episode['episode_id']}.json"
            failure_path = failures_dir / f"{episode['episode_id']}.json"
            if resume and record_path.is_file() and audit_path.is_file():
                skipped += 1
                continue
            try:
                record, audit = run_causal_training_oracle(env=env, episode=episode)
                _write_json_atomic(record_path, record)
                _write_json_atomic(audit_path, audit)
                failure_path.unlink(missing_ok=True)
                generated += 1
            except Exception as exc:
                record_path.unlink(missing_ok=True)
                audit_path.unlink(missing_ok=True)
                failure = {
                    "episode_id": episode["episode_id"],
                    "pair_id": episode["pair_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _write_json_atomic(failure_path, failure)
                failures.append(failure)
            if progress_every > 0 and position % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "variant": variant,
                            "processed": position,
                            "selected": len(selected),
                            "generated": generated,
                            "skipped": skipped,
                            "failures": len(failures),
                        }
                    ),
                    flush=True,
                )
    finally:
        env.close()

    expected_ids = [str(episode["episode_id"]) for episode in selected]
    completed_records = []
    completed_audits = []
    missing = []
    for episode_id in expected_ids:
        record_path = records_dir / f"{episode_id}.json"
        audit_path = audits_dir / f"{episode_id}.json"
        if not record_path.is_file() or not audit_path.is_file():
            missing.append(episode_id)
            continue
        completed_records.append(json.loads(record_path.read_text(encoding="utf-8")))
        completed_audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
    action_counts = Counter(
        int(record["metadata"]["action_step_count"]) for record in completed_records
    )
    summary = {
        "schema": "gui_captcha_exploration_depth_training_variant_v1",
        "rule_version": CAUSAL_RULE_VERSION,
        "variant": variant,
        "manifest_path": str(manifest_path),
        "selected_records": len(selected),
        "completed_records": len(completed_records),
        "generated_records": generated,
        "resume_skips": skipped,
        "failure_count": len(failures),
        "missing_count": len(missing),
        "success_count": sum(bool(audit["success"]) for audit in completed_audits),
        "action_count": sum(
            int(record["metadata"]["action_step_count"]) for record in completed_records
        ),
        "actions_per_trace": dict(sorted(action_counts.items())),
        "semantic_interaction_min": min(
            (int(audit["semantic_interactions_to_success"] or 0) for audit in completed_audits),
            default=None,
        ),
        "semantic_interaction_max": max(
            (int(audit["semantic_interactions_to_success"] or 0) for audit in completed_audits),
            default=None,
        ),
        "elapsed_seconds": round(time.time() - started, 3),
        "failures": failures[:20],
        "missing_episode_ids": missing[:20],
    }
    variant_root = output_root / variant
    if not missing:
        _append_jsonl(variant_root / "train.jsonl", completed_records)
        _append_jsonl(variant_root / "audit.jsonl", completed_audits)
    _write_json_atomic(variant_root / "summary.json", summary)
    if missing:
        raise RuntimeError(
            f"{variant} completed {len(completed_records)}/{len(selected)}; "
            f"see {variant_root / 'summary.json'}"
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--episode-ids-file",
        type=Path,
        default=None,
        help="optional newline-delimited episode ids; bypasses --start/--limit",
    )
    parser.add_argument(
        "--rotation-terminal-actions",
        type=Path,
        default=None,
        help=(
            "JSON map of selected rotation episode ids to successful terminal move_to "
            "actions from the removed traces"
        ),
    )
    args = parser.parse_args(argv)
    root = default_training_root()
    if args.manifest is None:
        args.manifest = root / "manifest.json"
    if args.output_root is None:
        args.output_root = root / "raw_traces"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    episode_ids = None
    if args.episode_ids_file is not None:
        episode_ids = {
            line.strip()
            for line in args.episode_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if not episode_ids:
            raise ValueError(f"episode id file is empty: {args.episode_ids_file}")
    rotation_terminal_actions = None
    if args.rotation_terminal_actions is not None:
        payload = json.loads(args.rotation_terminal_actions.read_text(encoding="utf-8"))
        actions = payload.get("actions") if isinstance(payload, dict) else None
        if not isinstance(actions, dict):
            raise ValueError(
                "rotation terminal action file must contain an object-valued actions field"
            )
        rotation_terminal_actions = {
            str(episode_id): dict(action)
            for episode_id, action in actions.items()
            if isinstance(action, dict)
        }
    if args.variant.startswith("rotation_"):
        with RotationReplayServer(public_root=args.manifest.parent) as replay:
            summary = generate_variant(
                manifest_path=args.manifest,
                output_root=args.output_root,
                variant=args.variant,
                base_url=replay.base_url,
                start=args.start,
                limit=args.limit,
                resume=args.resume,
                progress_every=args.progress_every,
                episode_ids=episode_ids,
                rotation_terminal_actions=rotation_terminal_actions,
            )
    else:
        summary = generate_variant(
            manifest_path=args.manifest,
            output_root=args.output_root,
            variant=args.variant,
            start=args.start,
            limit=args.limit,
            resume=args.resume,
            progress_every=args.progress_every,
            episode_ids=episode_ids,
            rotation_terminal_actions=rotation_terminal_actions,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
