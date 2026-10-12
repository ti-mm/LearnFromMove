from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...actions import Action, AtomicAction, PrimitiveAction
from ...core import Observation, StepResult


def _action_payload(action: Action | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return action.to_dict() if hasattr(action, "to_dict") else {"kind": action.kind}


def _meaningful_delta(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-9
    if isinstance(value, dict):
        return any(_meaningful_delta(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_meaningful_delta(item) for item in value)
    return False


@dataclass
class ExplorationTraceSession:
    env: object
    episode: dict[str, Any]
    output_dir: Path
    frames: list[dict[str, Any]] = field(default_factory=list, init=False)
    audit_frames: list[dict[str, Any]] = field(default_factory=list, init=False)
    semantic_interaction_index: int = field(default=0, init=False)

    def reset(self) -> Observation:
        observation = self.env.reset(task_id=str(self.episode["episode_id"]))
        self.frames = []
        self.audit_frames = []
        self.semantic_interaction_index = 0
        self._record(observation=observation, result=None, action=None)
        return observation

    def step(self, action: Action) -> StepResult:
        result = self.env.step(action)
        state_delta = result.info.get("state_delta", {})
        if (
            isinstance(action, PrimitiveAction)
            and action.kind == "move_to"
            and not result.done
            and _meaningful_delta(state_delta)
        ):
            self.semantic_interaction_index += 1
        self._record(observation=result.observation, result=result, action=action)
        return result

    def _record(
        self,
        *,
        observation: Observation,
        result: StepResult | None,
        action: Action | None,
    ) -> None:
        success = bool(result and result.done and result.reward == 1.0)
        frame = {
            "suite_id": self.episode["suite_id"],
            "pair_id": self.episode["pair_id"],
            "family": self.episode["family"],
            "frame_index": len(self.frames),
            "semantic_interaction_index": self.semantic_interaction_index,
            "action": _action_payload(action),
            "observation_path": observation.screenshot_path,
            "environment_state_delta": (
                {} if result is None else dict(result.info.get("state_delta", {}))
            ),
            "terminal_success": success,
            "interactions_to_success": (
                self.semantic_interaction_index if success else None
            ),
            "corrective_interaction_count": max(
                self.semantic_interaction_index - 1,
                0,
            ),
        }
        self.frames.append(frame)
        get_audit = getattr(self.env, "get_evaluator_audit", None)
        audit = get_audit() if callable(get_audit) else {}
        self.audit_frames.append(
            {
                "frame_index": frame["frame_index"],
                "semantic_interaction_index": self.semantic_interaction_index,
                "evaluator_only": audit,
            }
        )

    def write(self) -> dict[str, str]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "trace.jsonl"
        audit_path = self.output_dir / "audit.jsonl"
        trace_path.write_text(
            "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in self.frames),
            encoding="utf-8",
        )
        audit_path.write_text(
            "".join(json.dumps(frame, ensure_ascii=False) + "\n" for frame in self.audit_frames),
            encoding="utf-8",
        )
        return {"trace_path": str(trace_path), "audit_path": str(audit_path)}


def _model_xy(
    pixel_xy: tuple[float, float],
    viewport: tuple[int, int],
) -> tuple[float, float]:
    return (
        min(max(pixel_xy[0] / viewport[0] * 1000.0, 0.0), 1000.0),
        min(max(pixel_xy[1] / viewport[1] * 1000.0, 0.0), 1000.0),
    )


def _move_scene_to_target(
    session: ExplorationTraceSession,
    *,
    env: object,
    target_center: tuple[float, float],
    sensitivity: float,
    direction: tuple[float, float],
    max_moves: int = 8,
) -> None:
    viewport = (int(session.episode["viewport"][0]), int(session.episode["viewport"][1]))
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    for _ in range(max_moves):
        state = env.get_evaluator_audit()["scene_state"]
        if state.get("hoveredIndex") == session.episode["shared_scene_config"][
            "target_object"
        ]["target_index"]:
            return
        offset = (float(state.get("offsetX", 0.0)), float(state.get("offsetY", 0.0)))
        wanted = (center[0] - target_center[0], center[1] - target_center[1])
        residual = (wanted[0] - offset[0], wanted[1] - offset[1])
        pointer_delta = (
            residual[0] / sensitivity / direction[0],
            residual[1] / sensitivity / direction[1],
        )
        pixel = (
            min(max(center[0] + pointer_delta[0], 0.0), viewport[0] - 1.0),
            min(max(center[1] + pointer_delta[1], 0.0), viewport[1] - 1.0),
        )
        model = _model_xy(pixel, viewport)
        session.step(PrimitiveAction(kind="move_to", x=model[0], y=model[1]))
    raise RuntimeError("first-person ten-choice oracle could not center the target icon")


def _move_first_person_view(
    session: ExplorationTraceSession,
    *,
    env: object,
    target_world_xy: tuple[float, float],
    sensitivity: float,
    zoom: float,
    direction: tuple[float, float],
    max_moves: int = 8,
) -> None:
    viewport = (int(session.episode["viewport"][0]), int(session.episode["viewport"][1]))
    center = (viewport[0] / 2.0, viewport[1] / 2.0)
    for _ in range(max_moves):
        current = tuple(float(value) for value in env._view_center_xy)
        residual = (target_world_xy[0] - current[0], target_world_xy[1] - current[1])
        if abs(residual[0]) <= 1.0 and abs(residual[1]) <= 1.0:
            return
        pointer_delta = (
            residual[0] * zoom / sensitivity / direction[0],
            residual[1] * zoom / sensitivity / direction[1],
        )
        pixel = (
            min(max(center[0] + pointer_delta[0], 0.0), viewport[0] - 1.0),
            min(max(center[1] + pointer_delta[1], 0.0), viewport[1] - 1.0),
        )
        model = _model_xy(pixel, viewport)
        session.step(PrimitiveAction(kind="move_to", x=model[0], y=model[1]))
    raise RuntimeError("first-person drag oracle could not reach the requested world point")


def run_scripted_oracle(
    *,
    env: object,
    episode: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run a deterministic evaluator-side oracle without any model calls."""

    session = ExplorationTraceSession(env=env, episode=episode, output_dir=output_dir)
    session.reset()
    variant = str(episode["variant"])
    shared = episode["shared_scene_config"]

    if variant == "drag_third_person":
        viewport = tuple(int(value) for value in episode["viewport"])
        start = _model_xy(tuple(shared["piece_start_screen_xy"]), viewport)
        end = _model_xy(tuple(shared["slot_center_screen_xy"]), viewport)
        result = session.step(AtomicAction(kind="drag", points=[start, end]))

    elif variant == "drag_first_person":
        dynamics = shared["hidden_dynamics"]
        sensitivity = float(dynamics["sensitivity"])
        zoom = float(dynamics["view_zoom"])
        direction = tuple(float(value) for value in dynamics["direction_xy"])
        session.step(PrimitiveAction(kind="move_to", x=575.0, y=450.0))
        _move_first_person_view(
            session,
            env=env,
            target_world_xy=tuple(shared["piece_start_world_xy"]),
            sensitivity=sensitivity,
            zoom=zoom,
            direction=direction,
        )
        session.step(PrimitiveAction(kind="mouse_down"))
        _move_first_person_view(
            session,
            env=env,
            target_world_xy=tuple(shared["slot_center_world_xy"]),
            sensitivity=sensitivity,
            zoom=zoom,
            direction=direction,
        )
        result = session.step(PrimitiveAction(kind="mouse_up"))

    elif variant == "ten_choice_third_person":
        viewport = tuple(int(value) for value in episode["viewport"])
        target_index = int(shared["target_object"]["target_index"])
        target = tuple(float(value) for value in shared["icon_centers_xy"][target_index])
        model = _model_xy(target, viewport)
        session.step(PrimitiveAction(kind="move_to", x=model[0], y=model[1]))
        result = session.step(PrimitiveAction(kind="left_click"))

    elif variant == "ten_choice_first_person":
        dynamics = shared["hidden_dynamics"]
        target_index = int(shared["target_object"]["target_index"])
        target = tuple(float(value) for value in shared["icon_centers_xy"][target_index])
        session.step(PrimitiveAction(kind="move_to", x=600.0, y=420.0))
        _move_scene_to_target(
            session,
            env=env,
            target_center=target,
            sensitivity=float(dynamics["sensitivity"]),
            direction=tuple(float(value) for value in dynamics["direction_xy"]),
        )
        result = session.step(PrimitiveAction(kind="left_click"))

    elif variant in {"rotation_inner", "rotation_outer"}:
        geometry = env._read_slider_geometry()
        viewport = tuple(int(value) for value in episode["viewport"])
        thumb = _model_xy(
            (float(geometry["thumb_x"]), float(geometry["slider_y"])),
            viewport,
        )
        session.step(PrimitiveAction(kind="move_to", x=thumb[0], y=thumb[1]))
        session.step(PrimitiveAction(kind="mouse_down"))
        slider_left = float(geometry["slider_left_x"])
        slider_right = float(geometry["slider_right_x"])
        slider_y = float(geometry["slider_y"])
        target_value = float(shared["target_slider_value"])
        slider_min = float(shared["slider_min_value"])
        slider_max = float(shared["slider_max_value"])
        target_fraction = (target_value - slider_min) / (slider_max - slider_min)
        target_x = slider_left + target_fraction * (slider_right - slider_left)
        probe_x = slider_left + (0.2 if target_fraction > 0.5 else 0.8) * (
            slider_right - slider_left
        )
        probe = _model_xy((probe_x, slider_y), viewport)
        target = _model_xy((target_x, slider_y), viewport)
        session.step(PrimitiveAction(kind="move_to", x=probe[0], y=probe[1]))
        session.step(PrimitiveAction(kind="move_to", x=target[0], y=target[1]))
        result = session.step(PrimitiveAction(kind="mouse_up"))

    else:  # pragma: no cover - catalog is a closed six-variant set
        raise KeyError(f"unsupported exploration-depth variant {variant!r}")

    paths = session.write()
    success = bool(result.done and result.reward == 1.0)
    return {
        "variant": variant,
        "episode_id": episode["episode_id"],
        "success": success,
        "semantic_interactions_to_success": (
            session.semantic_interaction_index if success else None
        ),
        "corrective_interaction_count": max(session.semantic_interaction_index - 1, 0),
        "frame_count": len(session.frames),
        **paths,
    }
