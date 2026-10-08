from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ...actions import Action, AtomicAction, PrimitiveAction, WebAction
from ...core import Observation, StepResult
from ...shared.coordinates import relative_bin_to_pixel_xy
from ..slot_drag.environment import (
    DEFAULT_SLOT_RADIUS_SCALE,
    DRAG_GRAB_HIT_TEST_VERSION,
    _coerce_pair,
    _draw_hollow_slot,
    _draw_shape,
    _point_hits_drag_piece,
)
from .paths import environment_run_root, formal_benchmark_root

DEFAULT_VIEWPORT = (1280, 720)
DEFAULT_TOLERANCE_PX = 38.0

_DATASET_HINT = (
    "Set GUI_CAPTCHA_STORAGE_ROOT, then run PYTHONPATH=src python -m "
    "gui_agent_captcha.domains.third_person_drag.dataset_v2 "
    "--mode test --validate-oracles"
)


def _clamp_screen_xy(
    xy: tuple[float, float],
    *,
    viewport_size: tuple[int, int],
    margin: float = 0.0,
) -> tuple[float, float]:
    width, height = viewport_size
    safe_margin = max(float(margin), 0.0)
    return (
        min(max(float(xy[0]), safe_margin), max(safe_margin, float(width) - safe_margin)),
        min(max(float(xy[1]), safe_margin), max(safe_margin, float(height) - safe_margin)),
    )


def _draw_free_cursor(
    draw: ImageDraw.ImageDraw,
    *,
    cursor_xy: tuple[float, float],
) -> None:
    x, y = cursor_xy
    outer = [
        (x, y),
        (x + 3, y + 25),
        (x + 9, y + 18),
        (x + 15, y + 31),
        (x + 21, y + 28),
        (x + 15, y + 16),
        (x + 26, y + 15),
    ]
    inner = [
        (x + 2, y + 4),
        (x + 5, y + 20),
        (x + 10, y + 14),
        (x + 16, y + 25),
        (x + 18, y + 24),
        (x + 12, y + 13),
        (x + 21, y + 12),
    ]
    draw.polygon(outer, fill=(15, 23, 42, 255))
    draw.polygon(inner, fill=(248, 250, 252, 255))


@dataclass
class ThirdPersonDragCaptchaEnv:
    """Third-person shape-drag CAPTCHA with a free screen cursor.

    The whole workspace stays visible and the camera never moves. ``move_to``
    uses the repository's normal 0-1000 screenshot-relative coordinate format
    to place the cursor at the corresponding screen point. While the button is
    held after a successful grab, the piece follows the cursor with exactly the
    same screen-space displacement. There is no center lock and no sensitivity
    or view-zoom conversion.
    """

    dataset_root: Path = field(
        default_factory=lambda: formal_benchmark_root() / "test"
    )
    artifact_dir: Path = field(default_factory=environment_run_root)
    viewport_width: int = DEFAULT_VIEWPORT[0]
    viewport_height: int = DEFAULT_VIEWPORT[1]
    screenshot_subdir: str = "browser"
    _current_meta: dict[str, Any] | None = field(default=None, init=False)
    _current_task_id: str | None = field(default=None, init=False)
    _screenshot_index: int = field(default=0, init=False)
    _piece_screen_xy: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    _button_is_down: bool = field(default=False, init=False)
    _grabbed: bool = field(default=False, init=False)
    _grab_offset_screen_xy: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    cursor_xy: tuple[float, float] | None = field(default=None, init=False)
    last_observation: Observation | None = field(default=None, init=False)

    def _viewport_size(self) -> tuple[int, int]:
        return (int(self.viewport_width), int(self.viewport_height))

    def _episode_dir(self, episode_id: str) -> Path:
        return self.dataset_root / episode_id

    def _load_episode(self, episode_id: str) -> dict[str, Any]:
        meta_path = self._episode_dir(episode_id) / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"third_person_drag_captcha episode {episode_id!r} was not found under "
                f"{self.dataset_root}. Generate it with: {_DATASET_HINT}"
            )
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def list_task_ids(self, limit: int | None = None) -> list[str]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"third_person_drag_captcha dataset directory is missing: {self.dataset_root}. "
                f"Generate it with: {_DATASET_HINT}"
            )
        ids = sorted(
            path.name
            for path in self.dataset_root.iterdir()
            if path.is_dir() and (path / "meta.json").is_file()
        )
        return ids[:limit] if limit is not None else ids

    def _piece(self) -> dict[str, Any]:
        assert self._current_meta is not None
        return dict(self._current_meta.get("piece", {}))

    def _piece_radius(self) -> float:
        try:
            return float(self._piece().get("radius_px", 34.0))
        except (TypeError, ValueError):
            return 34.0

    def _slot_radius_scale(self) -> float:
        assert self._current_meta is not None
        try:
            return max(float(self._current_meta.get("slot_radius_scale", DEFAULT_SLOT_RADIUS_SCALE)), 1.0)
        except (TypeError, ValueError):
            return DEFAULT_SLOT_RADIUS_SCALE

    def _tolerance_px(self) -> float:
        assert self._current_meta is not None
        try:
            return float(self._current_meta.get("tolerance_px", DEFAULT_TOLERANCE_PX))
        except (TypeError, ValueError):
            return DEFAULT_TOLERANCE_PX

    def _slot_center_screen_xy(self) -> tuple[float, float]:
        assert self._current_meta is not None
        return _coerce_pair(
            self._current_meta.get(
                "slot_center_screen_xy",
                self._current_meta.get("slot_center_xy"),
            ),
            default=(960.0, 360.0),
        )

    def _slot_options(self) -> list[dict[str, Any]]:
        assert self._current_meta is not None
        options = self._current_meta.get("slot_options")
        if isinstance(options, list) and options:
            return [dict(option) for option in options if isinstance(option, dict)]
        piece = self._piece()
        return [
            {
                "id": "target",
                "role": "target",
                "center_screen_xy": list(self._slot_center_screen_xy()),
                "shape": piece.get("shape", "square"),
                "color": "#94a3b8",
            }
        ]

    def _piece_slot_distance_px(self) -> float:
        return math.dist(self._piece_screen_xy, self._slot_center_screen_xy())

    def _is_success(self) -> bool:
        return self._piece_slot_distance_px() <= self._tolerance_px()

    def _cursor_hits_piece(self) -> bool:
        if self.cursor_xy is None:
            return False
        return _point_hits_drag_piece(
            self.cursor_xy,
            center=self._piece_screen_xy,
            kind=str(self._piece().get("shape", "square")),
            radius=self._piece_radius(),
            viewport=self._viewport_size(),
        )

    def _screenshot_dir(self) -> Path:
        task_id = (self._current_task_id or "unknown-task-id").replace("/", "_")
        return (
            self.artifact_dir
            / self.screenshot_subdir
            / "ThirdPersonDragCaptcha"
            / task_id
        )

    def _render_background(self, draw: ImageDraw.ImageDraw) -> None:
        width, height = self._viewport_size()
        draw.rectangle([0, 0, width, height], fill=(239, 246, 255, 255))
        draw.rounded_rectangle(
            [24, 24, width - 24, height - 24],
            radius=28,
            fill=(248, 250, 252, 245),
            outline=(100, 116, 139, 120),
            width=3,
        )
        for x in range(80, width - 40, 80):
            draw.line([(x, 56), (x, height - 56)], fill=(148, 163, 184, 34), width=1)
        for y in range(80, height - 40, 80):
            draw.line([(56, y), (width - 56, y)], fill=(148, 163, 184, 34), width=1)

    def _render_screenshot(self, *, phase: str) -> str:
        assert self._current_meta is not None
        viewport = self._viewport_size()
        image = Image.new("RGBA", viewport, (239, 246, 255, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        self._render_background(draw)

        piece = self._piece()
        shape = str(piece.get("shape", "square"))
        radius = self._piece_radius()
        for option in self._slot_options():
            slot_screen = _coerce_pair(
                option.get("center_screen_xy"),
                default=self._slot_center_screen_xy(),
            )
            _draw_hollow_slot(
                draw,
                center=slot_screen,
                kind=str(option.get("shape", shape)),
                radius=radius,
                color=str(option.get("color", "#94a3b8")),
                target=str(option.get("role", "")) == "target",
                radius_scale=self._slot_radius_scale(),
            )

        _draw_shape(
            draw,
            center=self._piece_screen_xy,
            kind=shape,
            radius=radius,
            fill=str(piece.get("color", "#2563eb")),
            outline=str(piece.get("outline", "#111827")),
            width=4,
        )
        if self._grabbed and self.cursor_xy is not None:
            draw.line([self.cursor_xy, self._piece_screen_xy], fill=(6, 182, 212, 120), width=3)
        if self.cursor_xy is not None:
            _draw_free_cursor(draw, cursor_xy=self.cursor_xy)

        output_path = self._screenshot_dir() / f"{phase}-{self._screenshot_index:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path, compress_level=1)
        self._screenshot_index += 1
        return str(output_path)

    def _observation_metadata(self, episode_id: str) -> dict[str, Any]:
        piece = self._piece()
        return {
            "benchmark": "ThirdPersonDragCaptcha",
            "task_type": "third_person_drag_captcha",
            "task_id": episode_id,
            "episode_id": episode_id,
            "perspective": "third_person",
            "coordinate_contract": "qwen3_relative_0_1000_absolute_pointer",
            "cursor_lock": "none",
            "camera_motion": False,
            "movement_mapping": "absolute_screen_position_direct",
            "drag_screen_displacement_scale": 1.0,
            "viewport": [self.viewport_width, self.viewport_height],
            "piece_shape": piece.get("shape"),
            "piece_color": piece.get("color"),
            "piece_color_name": piece.get("color_name"),
            "movable_object_description": self._current_meta.get("movable_object_description"),
            "target_slot_description": self._current_meta.get("target_slot_description"),
            "distractor_slot_count": sum(
                1 for option in self._slot_options() if option.get("role") != "target"
            ),
            "slot_radius_scale": self._slot_radius_scale(),
            "tolerance_px": self._tolerance_px(),
        }

    def reset(self, *, task_type: str | None = None, task_id: str | None = None) -> Observation:
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"third_person_drag_captcha dataset directory is missing: {self.dataset_root}. "
                f"Generate it with: {_DATASET_HINT}"
            )
        episode_id = task_id or task_type
        if episode_id is None:
            task_ids = self.list_task_ids(limit=1)
            if not task_ids:
                raise ValueError(f"No third_person_drag_captcha episodes found in {self.dataset_root}")
            episode_id = task_ids[0]

        self._current_meta = self._load_episode(episode_id)
        viewport = self._current_meta.get("viewport")
        if isinstance(viewport, (list, tuple)) and len(viewport) == 2:
            self.viewport_width = int(viewport[0])
            self.viewport_height = int(viewport[1])
        self._current_task_id = episode_id
        self._screenshot_index = 0
        self._button_is_down = False
        self._grabbed = False
        self._grab_offset_screen_xy = (0.0, 0.0)
        self._piece_screen_xy = _clamp_screen_xy(
            _coerce_pair(
                self._current_meta.get(
                    "piece_start_screen_xy",
                    self._current_meta.get("piece_start_xy"),
                ),
                default=(260.0, 360.0),
            ),
            viewport_size=self._viewport_size(),
            margin=self._piece_radius(),
        )
        self.cursor_xy = _clamp_screen_xy(
            _coerce_pair(
                self._current_meta.get("initial_cursor_screen_xy"),
                default=(64.0, 64.0),
            ),
            viewport_size=self._viewport_size(),
        )
        screenshot_path = self._render_screenshot(phase="reset")
        instruction = str(
            self._current_meta.get(
                "instruction",
                "Drag the solid colored shape into the matching gray outline.",
            )
        )
        self.last_observation = Observation(
            instruction=instruction,
            screenshot_path=screenshot_path,
            size_px=self._viewport_size(),
            cursor_xy=self.cursor_xy,
            metadata=self._observation_metadata(episode_id),
        )
        return self.last_observation

    def _apply_absolute_move(self, model_x: float, model_y: float) -> dict[str, Any]:
        old_cursor = self.cursor_xy or (0.0, 0.0)
        old_piece = self._piece_screen_xy
        pixel_xy = relative_bin_to_pixel_xy(model_x, model_y, self._viewport_size())
        self.cursor_xy = _clamp_screen_xy(pixel_xy, viewport_size=self._viewport_size())
        if self._button_is_down and self._grabbed:
            self._piece_screen_xy = _clamp_screen_xy(
                (
                    self.cursor_xy[0] + self._grab_offset_screen_xy[0],
                    self.cursor_xy[1] + self._grab_offset_screen_xy[1],
                ),
                viewport_size=self._viewport_size(),
                margin=self._piece_radius(),
            )
        cursor_delta = (
            self.cursor_xy[0] - old_cursor[0],
            self.cursor_xy[1] - old_cursor[1],
        )
        piece_delta = (
            self._piece_screen_xy[0] - old_piece[0],
            self._piece_screen_xy[1] - old_piece[1],
        )
        return {
            "model_coordinate_format": "relative_0_1000",
            "model_xy": (float(model_x), float(model_y)),
            "pixel_xy": self.cursor_xy,
            "cursor_before_xy": old_cursor,
            "cursor_after_xy": self.cursor_xy,
            "cursor_delta_px": cursor_delta,
            "piece_delta_px": piece_delta,
            "drag_screen_displacement_scale": 1.0,
            "camera_moved": False,
            "view_conversion_applied": False,
        }

    def _begin_cursor_grab(self) -> dict[str, Any]:
        self._button_is_down = True
        self._grabbed = self._cursor_hits_piece()
        if self._grabbed and self.cursor_xy is not None:
            self._grab_offset_screen_xy = (
                self._piece_screen_xy[0] - self.cursor_xy[0],
                self._piece_screen_xy[1] - self.cursor_xy[1],
            )
        return {
            "grab_started": self._grabbed,
            "grab_offset_screen_xy": self._grab_offset_screen_xy,
        }

    def _apply_atomic_drag_macro(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        self._button_is_down = False
        self._grabbed = False
        self._grab_offset_screen_xy = (0.0, 0.0)
        start_x, start_y = points[0]
        start_info = self._apply_absolute_move(start_x, start_y)
        grab_info = self._begin_cursor_grab()
        move_infos = [self._apply_absolute_move(x, y) for x, y in points[1:]]
        cursor_delta = tuple(
            sum(float(info["cursor_delta_px"][axis]) for info in move_infos)
            for axis in (0, 1)
        )
        piece_delta = tuple(
            sum(float(info["piece_delta_px"][axis]) for info in move_infos)
            for axis in (0, 1)
        )
        return {
            "model_coordinate_format": "relative_0_1000",
            "model_points": [(float(x), float(y)) for x, y in points],
            "pixel_points": [
                start_info["pixel_xy"],
                *(info["pixel_xy"] for info in move_infos),
            ],
            "atomic_drag_macro": True,
            "grab_started": grab_info["grab_started"],
            "cursor_delta_px": cursor_delta,
            "piece_delta_px": piece_delta,
            "drag_screen_displacement_scale": 1.0,
            "camera_moved": False,
            "view_conversion_applied": False,
        }

    def _make_observation(self, *, phase: str) -> Observation:
        assert self._current_task_id is not None
        assert self._current_meta is not None
        screenshot_path = self._render_screenshot(phase=phase)
        obs = Observation(
            instruction=str(self._current_meta.get("instruction", "")),
            screenshot_path=screenshot_path,
            size_px=self._viewport_size(),
            cursor_xy=self.cursor_xy,
            metadata=self._observation_metadata(self._current_task_id),
        )
        self.last_observation = obs
        return obs

    def _info(
        self,
        *,
        action: Action,
        executed_kind: str,
        success: bool,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        info = {
            "success": success,
            "executed_kind": executed_kind,
            "perspective": "third_person",
            "cursor_locked": False,
            "cursor_xy": self.cursor_xy,
            "camera_moved": False,
            "piece_screen_xy": self._piece_screen_xy,
            "slot_screen_xy": self._slot_center_screen_xy(),
            "piece_slot_distance_px": self._piece_slot_distance_px(),
            "tolerance_px": self._tolerance_px(),
            "button_is_down": self._button_is_down,
            "grabbed": self._grabbed,
            "grab_hit_test_version": DRAG_GRAB_HIT_TEST_VERSION,
            "action": action.to_dict() if hasattr(action, "to_dict") else repr(action),
        }
        if extra:
            info.update(extra)
        return info

    def refresh_observation(self) -> Observation:
        """Render a fresh frame without executing or scoring an action."""

        if self._current_meta is None:
            raise RuntimeError("reset() must be called before refresh_observation()")
        return self._make_observation(phase="refresh")

    def step(self, action: Action) -> StepResult:
        if self._current_meta is None:
            raise RuntimeError("reset() must be called before step()")

        reward = 0.0
        done = False
        success = False
        phase = "step"
        extra: dict[str, Any] = {}
        executed_kind = getattr(action, "kind", "unknown")

        if isinstance(action, PrimitiveAction):
            if action.kind == "move_to":
                extra = self._apply_absolute_move(float(action.x), float(action.y))
            elif action.kind == "mouse_down":
                extra = self._begin_cursor_grab()
            elif action.kind == "mouse_up":
                success = self._button_is_down and self._grabbed and self._is_success()
                reward = 1.0 if success else 0.0
                done = True
                phase = "success" if success else "failure"
                self._button_is_down = False
                self._grabbed = False
            elif action.kind == "left_click":
                done = True
                phase = "failure"
                self._button_is_down = False
                self._grabbed = False
            elif action.kind == "done":
                success = self._is_success()
                reward = 1.0 if success else 0.0
                done = True
                phase = "success" if success else "failure"
        elif isinstance(action, AtomicAction):
            if action.kind == "drag":
                extra = self._apply_atomic_drag_macro(action.points)
                success = self._button_is_down and self._grabbed and self._is_success()
                reward = 1.0 if success else 0.0
                done = True
                phase = "success" if success else "failure"
                self._button_is_down = False
                self._grabbed = False
            else:
                done = True
                phase = "failure"
        elif isinstance(action, WebAction):
            if action.kind == "wait":
                duration_s = 1.0 if action.duration_s is None else action.duration_s
                time.sleep(duration_s)
                extra = {
                    "web_action_effect": "wait_observe",
                    "duration_s": duration_s,
                }
            elif action.kind == "finished":
                success = self._is_success()
                reward = 1.0 if success else 0.0
                done = True
                phase = "success" if success else "failure"
            else:
                raise ValueError(
                    "ThirdPersonDragCaptchaEnv does not support "
                    f"UI-Venus web action {action.kind!r}"
                )

        obs = self._make_observation(phase=phase)
        info = self._info(
            action=action,
            executed_kind=executed_kind,
            success=success,
            extra=extra,
        )
        return StepResult(observation=obs, reward=reward, done=done, info=info, action=action)

    def close(self) -> None:
        return None
