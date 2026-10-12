from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...actions import Action, AtomicAction, PrimitiveAction, WebAction
from ...benchmarks.exploration_depth.contracts import (
    default_formal_manifest_path,
    episodes_for_variant,
    load_manifest,
    policy_observation_metadata,
    resolve_suite_path,
)
from ...core import Observation, StepResult
from ...envs.browser_base import overlay_cursor
from ...shared.coordinates import relative_bin_to_pixel_xy
from .environment import HoverRevealEnv


def _is_release(action: Action) -> bool:
    return (
        isinstance(action, AtomicAction)
        and action.kind in {"click", "drag"}
    ) or (
        isinstance(action, PrimitiveAction)
        and action.kind in {"mouse_up", "left_click"}
    ) or (
        isinstance(action, WebAction)
        and action.kind == "left_double"
    )


@dataclass
class PairedHoverRevealEnv(HoverRevealEnv):
    """Policy-safe paired L1 view of the existing HoverReveal environment."""

    manifest_path: Path | None = None
    benchmark_variant: str = field(
        default="ten_choice_third_person",
        init=False,
    )
    _paired_episode: dict[str, Any] | None = field(default=None, init=False)
    _last_evaluator_audit: dict[str, Any] = field(default_factory=dict, init=False)
    _paired_manifest_cache: dict[str, Any] | None = field(default=None, init=False)
    _paired_episode_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.manifest_path = (
            default_formal_manifest_path()
            if self.manifest_path is None
            else Path(self.manifest_path)
        )

    def _manifest(self) -> dict[str, Any]:
        assert self.manifest_path is not None
        if self._paired_manifest_cache is None:
            self._paired_manifest_cache = load_manifest(self.manifest_path)
        return self._paired_manifest_cache

    def _episode(self, requested: str) -> dict[str, Any]:
        if not self._paired_episode_index:
            for episode in episodes_for_variant(self._manifest(), self.benchmark_variant):
                self._paired_episode_index[str(episode["episode_id"])] = episode
                self._paired_episode_index[str(episode["pair_id"])] = episode
        try:
            return copy.deepcopy(self._paired_episode_index[requested])
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.benchmark_variant} training/formal episode {requested!r}"
            ) from exc

    def list_task_ids(self, limit: int | None = None) -> list[str]:
        ids = [
            str(episode["episode_id"])
            for episode in episodes_for_variant(self._manifest(), self.benchmark_variant)
        ]
        return ids[:limit] if limit is not None else ids

    def resolve_task_instance_id(self, task_type: str) -> str:
        return task_type

    def _read_scene_state(self) -> dict[str, Any]:
        if self.page is None:
            return {}
        try:
            value = self.page.evaluate(
                """() => {
                    const state = window.__explorationReadTenChoiceState?.() || {};
                    if (document.body.dataset.interactionMarker !== 'mouse_icon') {
                        return state;
                    }
                    const hovered = document.querySelector('.icon-button:hover');
                    return {
                        ...state,
                        hoveredIndex: hovered === null
                            ? null
                            : Number(hovered.dataset.index),
                    };
                }"""
            )
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _safe_observation(self, screenshot_path: str) -> Observation:
        assert self._paired_episode is not None
        return Observation(
            instruction=str(self._paired_episode["instruction"]),
            screenshot_path=screenshot_path,
            size_px=(self.viewport_width, self.viewport_height),
            cursor_xy=self.cursor_xy,
            metadata=policy_observation_metadata(self._paired_episode),
        )

    def _capture_screenshot(self, *, phase: str) -> str | None:
        screenshot_path = super()._capture_screenshot(phase=phase)
        if (
            screenshot_path
            and self._paired_episode is not None
            and self._paired_episode["environment_config"].get("interaction_marker")
            == "mouse_icon"
        ):
            path = Path(screenshot_path)
            overlay_cursor(
                path,
                path,
                cursor_xy=self.cursor_xy,
                keep_body_in_frame=True,
            )
        return screenshot_path

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        self._ensure_started()
        requested = task_id or task_type
        if requested is None:
            requested = self.list_task_ids(limit=1)[0]
        episode = self._episode(str(requested))
        self._paired_episode = episode
        shared = dict(episode["shared_scene_config"])
        target = dict(shared["target_object"])
        self._current_meta = {
            "target_idx": int(target["target_index"]),
            "target_label": str(target["target_label"]),
        }
        self.viewport_width, self.viewport_height = (
            int(episode["viewport"][0]),
            int(episode["viewport"][1]),
        )
        if self.page is not None:
            self.page.set_viewport_size(
                {"width": self.viewport_width, "height": self.viewport_height}
            )
        self._current_task_type = str(episode["pair_id"])
        self._current_task_id = str(episode["episode_id"])
        self._screenshot_index = 0
        assert self.manifest_path is not None
        html_path = resolve_suite_path(self.manifest_path, shared["html_path"])
        if self.page is not None:
            self.page.goto(
                html_path.resolve().as_uri(),
                wait_until="load",
                timeout=self.navigation_timeout_ms,
            )
            try:
                self.page.wait_for_function(
                    "() => document.fonts.status === 'loaded' && "
                    "Array.from(document.images).every(image => image.complete)",
                    timeout=10_000,
                )
            except Exception:
                pass
            center = (self.viewport_width / 2.0, self.viewport_height / 2.0)
            self.page.mouse.move(*center)
            self.page.evaluate(
                "marker => window.__explorationSetInteractionMarker?.(marker)",
                episode["environment_config"]["interaction_marker"],
            )
            self.page.evaluate(
                "variant => window.__explorationSetTenChoiceMode?.(variant)",
                episode.get("scene_variant", self.benchmark_variant),
            )
        self.cursor_xy = (self.viewport_width / 2.0, self.viewport_height / 2.0)
        screenshot_path = self._capture_screenshot(phase="reset") or ""
        self.last_observation = self._safe_observation(screenshot_path)
        self._last_evaluator_audit = self._build_audit(success=None)
        return self.last_observation

    def _build_audit(self, *, success: bool | None) -> dict[str, Any]:
        assert self._paired_episode is not None
        shared = self._paired_episode["shared_scene_config"]
        return {
            "suite_id": self._paired_episode["suite_id"],
            "pair_id": self._paired_episode["pair_id"],
            "variant": self.benchmark_variant,
            "target_object": shared["target_object"],
            "hidden_dynamics": shared["hidden_dynamics"],
            "scene_state": self._read_scene_state(),
            "terminal_success": success,
        }

    def get_evaluator_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._last_evaluator_audit))

    def step(self, action: Action) -> StepResult:
        before = self._read_scene_state()
        before_cursor = self.cursor_xy
        raw = super().step(action)
        after = self._read_scene_state()
        success = bool(raw.reward == 1.0) if raw.done else None
        self._last_evaluator_audit = self._build_audit(success=success)
        info = {
            "executed_kind": action.kind,
            "success": bool(success) if raw.done else False,
            "state_delta": {
                "cursor_moved": raw.observation.cursor_xy != before_cursor,
                "center_hover_changed": before.get("hoveredIndex")
                != after.get("hoveredIndex"),
            },
        }
        raw.observation.metadata = policy_observation_metadata(self._paired_episode or {})
        self.last_observation = raw.observation
        return StepResult(
            observation=raw.observation,
            reward=raw.reward,
            done=raw.done,
            info=info,
            action=action,
        )


@dataclass
class FirstPersonTenChoiceEnv(PairedHoverRevealEnv):
    """L2 ten-choice environment with a fixed reticle and movable scene."""

    benchmark_variant: str = field(
        default="ten_choice_first_person",
        init=False,
    )
    _button_is_down: bool = field(default=False, init=False)

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        observation = super().reset(task_type=task_type, task_id=task_id)
        self._button_is_down = False
        return observation

    def _move_scene(self, action: PrimitiveAction) -> dict[str, Any]:
        assert action.x is not None and action.y is not None
        assert self._paired_episode is not None
        pixel_xy = relative_bin_to_pixel_xy(
            action.x,
            action.y,
            (self.viewport_width, self.viewport_height),
        )
        center = (self.viewport_width / 2.0, self.viewport_height / 2.0)
        pointer_delta = (pixel_xy[0] - center[0], pixel_xy[1] - center[1])
        dynamics = self._paired_episode["shared_scene_config"]["hidden_dynamics"]
        sensitivity = float(dynamics["sensitivity"])
        direction = dynamics["direction_xy"]
        scene_delta = {
            "dx": pointer_delta[0] * sensitivity * float(direction[0]),
            "dy": pointer_delta[1] * sensitivity * float(direction[1]),
        }
        if self.page is not None:
            self.page.evaluate(
                "delta => window.__explorationMoveScene(delta)",
                scene_delta,
            )
            self.page.mouse.move(*center)
        self.cursor_xy = center
        return {
            "scene_moved": bool(scene_delta["dx"] or scene_delta["dy"]),
            "observation_changed": bool(scene_delta["dx"] or scene_delta["dy"]),
        }

    def _submit_at_center(self, *, require_down: bool) -> tuple[bool, dict[str, Any]]:
        if self.page is None:
            return False, self._read_scene_state()
        center = (self.viewport_width / 2.0, self.viewport_height / 2.0)
        self.page.mouse.move(*center)
        if require_down:
            if self._button_is_down:
                self.page.mouse.up()
        else:
            self.page.mouse.down()
            self.page.mouse.up()
        self._button_is_down = False
        state = self._read_scene_state()
        success, _ = self._fetch_task_result()
        assert self._current_meta is not None
        geometric_success = state.get("hoveredIndex") == self._current_meta["target_idx"]
        return bool(success and geometric_success), state

    def step(self, action: Action) -> StepResult:
        if self._paired_episode is None:
            raise RuntimeError("reset() must be called before step()")
        before = self._read_scene_state()
        done = False
        success: bool | None = None
        state_delta: dict[str, Any] = {"observation_changed": False}

        if isinstance(action, PrimitiveAction):
            if action.kind == "move_to":
                state_delta = self._move_scene(action)
            elif action.kind == "mouse_down":
                if self.page is not None:
                    self.page.mouse.down()
                self._button_is_down = True
            elif action.kind == "mouse_up":
                success, _ = self._submit_at_center(require_down=True)
                done = True
            elif action.kind == "left_click":
                success, _ = self._submit_at_center(require_down=False)
                done = True
            elif action.kind == "done":
                success = False
                done = True
        elif isinstance(action, AtomicAction):
            if action.kind == "click":
                model_x, model_y = action.points[0]
                state_delta = self._move_scene(
                    PrimitiveAction(kind="move_to", x=model_x, y=model_y)
                )
                state_delta["atomic_move_then_click"] = True
                success, _ = self._submit_at_center(require_down=False)
            else:
                success = False
            done = True
        elif isinstance(action, WebAction):
            if action.kind == "wait":
                pass
            elif action.kind in {"left_double", "finished"}:
                success, _ = self._submit_at_center(require_down=False)
                done = True
            else:
                raise ValueError(f"unsupported first-person ten-choice action {action.kind!r}")

        screenshot_path = self._capture_screenshot(
            phase="success" if success else ("failure" if done else "step")
        ) or (self.last_observation.screenshot_path if self.last_observation else "")
        observation = self._safe_observation(screenshot_path)
        self.last_observation = observation
        after = self._read_scene_state()
        state_delta["center_hover_changed"] = (
            before.get("hoveredIndex") != after.get("hoveredIndex")
        )
        self._last_evaluator_audit = self._build_audit(success=success)
        return StepResult(
            observation=observation,
            reward=1.0 if success else (0.0 if done else None),
            done=done,
            info={
                "executed_kind": action.kind,
                "success": bool(success),
                "state_delta": state_delta,
            },
            action=action,
        )
