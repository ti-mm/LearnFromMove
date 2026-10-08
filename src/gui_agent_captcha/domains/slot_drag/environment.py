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
from .paths import environment_run_root, formal_benchmark_root

DEFAULT_VIEWPORT = (1280, 800)
DEFAULT_WORLD_SIZE = (2200, 1600)
DEFAULT_TOLERANCE_PX = 26.0
DEFAULT_SLOT_RADIUS_SCALE = 1.35
DRAG_GRAB_HIT_TEST_VERSION = "legacy_circle_union_rendered_piece_v1"

_DATASET_HINT = (
    "Set GUI_CAPTCHA_STORAGE_ROOT, then run "
    "PYTHONPATH=src python -m gui_agent_captcha.domains.slot_drag.dataset "
    "--test-count 150"
)


def _center_for_size(size_px: tuple[int, int]) -> tuple[float, float]:
    return (size_px[0] / 2.0, size_px[1] / 2.0)


def _coerce_pair(value: Any, *, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default
    return default


def _clamp_view_center(
    center_xy: tuple[float, float],
    *,
    world_size: tuple[int, int],
    viewport_size: tuple[int, int],
    view_zoom: float,
) -> tuple[float, float]:
    zoom = max(float(view_zoom), 1e-6)
    half_w = viewport_size[0] / 2.0 / zoom
    half_h = viewport_size[1] / 2.0 / zoom
    min_x = min(half_w, world_size[0] - half_w)
    max_x = max(half_w, world_size[0] - half_w)
    min_y = min(half_h, world_size[1] - half_h)
    max_y = max(half_h, world_size[1] - half_h)
    return (
        min(max(float(center_xy[0]), min_x), max_x),
        min(max(float(center_xy[1]), min_y), max_y),
    )


def _clamp_world_xy(xy: tuple[float, float], world_size: tuple[int, int]) -> tuple[float, float]:
    return (
        min(max(float(xy[0]), 0.0), float(world_size[0])),
        min(max(float(xy[1]), 0.0), float(world_size[1])),
    )


def _screen_xy_for_world_xy(
    world_xy: tuple[float, float],
    *,
    view_center_xy: tuple[float, float],
    viewport_size: tuple[int, int],
    view_zoom: float,
) -> tuple[float, float]:
    center_x, center_y = _center_for_size(viewport_size)
    zoom = max(float(view_zoom), 1e-6)
    return (
        center_x + (world_xy[0] - view_center_xy[0]) * zoom,
        center_y + (world_xy[1] - view_center_xy[1]) * zoom,
    )


def _shape_polygon(kind: str, *, center: tuple[float, float], radius: float) -> list[tuple[float, float]]:
    cx, cy = center
    if kind == "triangle":
        angles = (-90, 30, 150)
    elif kind == "diamond":
        angles = (-90, 0, 90, 180)
    elif kind == "pentagon":
        angles = tuple(-90 + index * 72 for index in range(5))
    elif kind == "hexagon":
        angles = (-90, -30, 30, 90, 150, 210)
    elif kind == "cross":
        arm = radius * 0.36
        return [
            (cx - arm, cy - radius),
            (cx + arm, cy - radius),
            (cx + arm, cy - arm),
            (cx + radius, cy - arm),
            (cx + radius, cy + arm),
            (cx + arm, cy + arm),
            (cx + arm, cy + radius),
            (cx - arm, cy + radius),
            (cx - arm, cy + arm),
            (cx - radius, cy + arm),
            (cx - radius, cy - arm),
            (cx - arm, cy - arm),
        ]
    elif kind == "trapezoid":
        half_top = radius * 0.66
        half_bottom = radius * 1.14
        half_height = radius * 0.88
        return [
            (cx - half_top, cy - half_height),
            (cx + half_top, cy - half_height),
            (cx + half_bottom, cy + half_height),
            (cx - half_bottom, cy + half_height),
        ]
    elif kind == "star":
        points: list[tuple[float, float]] = []
        for index in range(10):
            point_radius = radius if index % 2 == 0 else radius * 0.44
            angle = math.radians(-90 + index * 36)
            points.append(
                (
                    cx + math.cos(angle) * point_radius,
                    cy + math.sin(angle) * point_radius,
                )
            )
        return points
    elif kind == "l_shape":
        inner = radius * 0.28
        return [
            (cx - radius, cy - radius),
            (cx - inner, cy - radius),
            (cx - inner, cy + inner),
            (cx + radius, cy + inner),
            (cx + radius, cy + radius),
            (cx - radius, cy + radius),
        ]
    elif kind == "wide_rectangle":
        half_w = radius * 1.38
        half_h = radius * 0.72
        return [
            (cx - half_w, cy - half_h),
            (cx + half_w, cy - half_h),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
    elif kind == "notched_rectangle":
        half_w = radius * 1.36
        half_h = radius * 0.76
        notch = radius * 0.42
        return [
            (cx - half_w, cy - half_h),
            (cx + half_w - notch, cy - half_h),
            (cx + half_w, cy - half_h + notch),
            (cx + half_w, cy + half_h),
            (cx - half_w, cy + half_h),
        ]
    else:
        angles = (0, 90, 180, 270)
    return [
        (
            cx + math.cos(math.radians(angle)) * radius,
            cy + math.sin(math.radians(angle)) * radius,
        )
        for angle in angles
    ]


def _draw_shape(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    kind: str,
    radius: float,
    fill: str | tuple[int, int, int, int] | None,
    outline: str | tuple[int, int, int, int],
    width: int = 3,
) -> None:
    x, y = center
    if kind == "circle":
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill, outline=outline, width=width)
    elif kind == "square":
        draw.rounded_rectangle(
            [x - radius, y - radius, x + radius, y + radius],
            radius=max(3, int(radius * 0.18)),
            fill=fill,
            outline=outline,
            width=width,
        )
    elif kind in {"wide_rectangle", "notched_rectangle"}:
        polygon = _shape_polygon(kind, center=center, radius=radius)
        draw.polygon(polygon, fill=fill, outline=outline)
        draw.line(polygon + [polygon[0]], fill=outline, width=width)
    else:
        polygon = _shape_polygon(kind, center=center, radius=radius)
        draw.polygon(polygon, fill=fill, outline=outline)
        draw.line(polygon + [polygon[0]], fill=outline, width=width)


def _point_hits_drag_piece(
    point_xy: tuple[float, float],
    *,
    center: tuple[float, float],
    kind: str,
    radius: float,
    viewport: tuple[int, int],
) -> bool:
    """Preserve every legacy circle hit and add the rendered solid/outline pixels.

    Keep the legacy distance test on continuous coordinates. Only the added
    shape region is rasterized; shadows are excluded and rendering is unchanged.
    """
    if math.dist(point_xy, center) <= radius + 4:
        return True
    pixel = (math.floor(point_xy[0]), math.floor(point_xy[1]))
    if not (0 <= pixel[0] < viewport[0] and 0 <= pixel[1] < viewport[1]):
        return False
    mask = Image.new("1", viewport)
    _draw_shape(ImageDraw.Draw(mask), center=center, kind=kind, radius=radius,
                fill="white", outline="white", width=4)
    return bool(mask.getpixel(pixel))


def _draw_hollow_slot(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    kind: str,
    radius: float,
    color: str,
    target: bool,
    radius_scale: float = DEFAULT_SLOT_RADIUS_SCALE,
) -> None:
    slot_radius = radius * max(float(radius_scale), 1.0)
    outline = (71, 85, 105, 190)
    inner = (148, 163, 184, 46)
    _draw_shape(draw, center=center, kind=kind, radius=slot_radius + 10, fill=(255, 255, 255, 205), outline=outline, width=6)
    _draw_shape(draw, center=center, kind=kind, radius=max(radius + 4, slot_radius - 4), fill=inner, outline=(30, 41, 59, 105), width=3)


def _draw_crosshair(draw: ImageDraw.ImageDraw, *, center: tuple[float, float]) -> None:
    x, y = center
    draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(15, 23, 42, 185))
    draw.ellipse(
        [x - 3, y - 3, x + 3, y + 3],
        fill=(248, 250, 252, 245),
        outline=(15, 23, 42, 245),
        width=1,
    )
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(244, 63, 94, 255))


@dataclass
class SlotDragGameEnv:
    """First-person shape-to-slot CAPTCHA with a center-locked reticle.

    The reticle never leaves the screen center, so it acts as the agent's fixed
    first-person interaction point while the world moves around it. A
    coordinate-bearing move_to is interpreted as a relative pointer
    displacement from center, and the world camera moves by ``displacement *
    sensitivity / view_zoom``. To solve an episode, move the object under the
    center reticle, mouse_down to grab it, move the matching slot under the
    center reticle while holding, and mouse_up.
    """

    dataset_root: Path | None = None
    artifact_dir: Path | None = None
    viewport_width: int = DEFAULT_VIEWPORT[0]
    viewport_height: int = DEFAULT_VIEWPORT[1]
    screenshot_subdir: str = "browser"
    _current_meta: dict[str, Any] | None = field(default=None, init=False)
    _current_task_type: str | None = field(default=None, init=False)
    _current_task_id: str | None = field(default=None, init=False)
    _screenshot_index: int = field(default=0, init=False)
    _view_center_xy: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    _piece_world_xy: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    _button_is_down: bool = field(default=False, init=False)
    _grabbed: bool = field(default=False, init=False)
    _grab_offset_world_xy: tuple[float, float] = field(default=(0.0, 0.0), init=False)
    cursor_xy: tuple[float, float] | None = field(default=None, init=False)
    last_observation: Observation | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.dataset_root = (
            formal_benchmark_root() / "test"
            if self.dataset_root is None
            else Path(self.dataset_root)
        )
        self.artifact_dir = (
            environment_run_root()
            if self.artifact_dir is None
            else Path(self.artifact_dir)
        )

    def _viewport_size(self) -> tuple[int, int]:
        return (int(self.viewport_width), int(self.viewport_height))

    def _cursor_center_xy(self) -> tuple[float, float]:
        return _center_for_size(self._viewport_size())

    def _episode_dir(self, episode_id: str) -> Path:
        return self.dataset_root / episode_id

    def _load_episode(self, episode_id: str) -> dict[str, Any]:
        meta_path = self._episode_dir(episode_id) / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"slot_drag_game episode {episode_id!r} was not found under "
                f"{self.dataset_root}. Generate it with: {_DATASET_HINT}"
            )
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def list_task_ids(self, limit: int | None = None) -> list[str]:
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"slot_drag_game dataset directory is missing: {self.dataset_root}. "
                f"Generate it with: {_DATASET_HINT}"
            )
        ids = sorted(
            path.name
            for path in self.dataset_root.iterdir()
            if path.is_dir() and (path / "meta.json").is_file()
        )
        return ids[:limit] if limit is not None else ids

    def _world_size(self) -> tuple[int, int]:
        assert self._current_meta is not None
        raw = self._current_meta.get("world_size", list(DEFAULT_WORLD_SIZE))
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return (int(raw[0]), int(raw[1]))
        return DEFAULT_WORLD_SIZE

    def _piece(self) -> dict[str, Any]:
        assert self._current_meta is not None
        return dict(self._current_meta.get("piece", {}))

    def _slot_center_world_xy(self) -> tuple[float, float]:
        assert self._current_meta is not None
        return _coerce_pair(
            self._current_meta.get("slot_center_world_xy", self._current_meta.get("slot_center_xy")),
            default=(1100.0, 800.0),
        )

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

    def _sensitivity(self) -> float:
        assert self._current_meta is not None
        try:
            return float(self._current_meta.get("sensitivity", 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _view_zoom(self) -> float:
        assert self._current_meta is not None
        try:
            return max(float(self._current_meta.get("view_zoom", 1.0)), 1e-6)
        except (TypeError, ValueError):
            return 1.0

    def _piece_slot_distance_px(self) -> float:
        return math.dist(self._piece_world_xy, self._slot_center_world_xy())

    def _is_success(self) -> bool:
        return self._piece_slot_distance_px() <= self._tolerance_px()

    def _piece_screen_xy(self) -> tuple[float, float]:
        return _screen_xy_for_world_xy(
            self._piece_world_xy,
            view_center_xy=self._view_center_xy,
            viewport_size=self._viewport_size(),
            view_zoom=self._view_zoom(),
        )

    def _slot_screen_xy(self) -> tuple[float, float]:
        return _screen_xy_for_world_xy(
            self._slot_center_world_xy(),
            view_center_xy=self._view_center_xy,
            viewport_size=self._viewport_size(),
            view_zoom=self._view_zoom(),
        )

    def _piece_center_distance_px(self) -> float:
        return math.dist(self._piece_screen_xy(), self._cursor_center_xy())

    def _slot_center_distance_px(self) -> float:
        return math.dist(self._slot_screen_xy(), self._cursor_center_xy())

    def _crosshair_hits_piece(self) -> bool:
        return _point_hits_drag_piece(
            self._cursor_center_xy(),
            center=self._piece_screen_xy(),
            kind=str(self._piece().get("shape", "circle")),
            radius=self._piece_radius(),
            viewport=self._viewport_size(),
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
                "center_world_xy": list(self._slot_center_world_xy()),
                "shape": piece.get("shape", "circle"),
                "color": piece.get("color", "#f97316"),
            }
        ]

    def _screenshot_dir(self) -> Path:
        task_type = self._current_task_type or "unknown-task-type"
        task_id = (self._current_task_id or "unknown-task-id").replace("/", "_")
        return self.artifact_dir / self.screenshot_subdir / "SlotDragGame" / task_type / task_id

    def _render_background(self, draw: ImageDraw.ImageDraw) -> None:
        viewport = self._viewport_size()
        center_x, center_y = self._cursor_center_xy()
        view_x, view_y = self._view_center_xy
        zoom = self._view_zoom()
        for y in range(0, viewport[1], 40):
            fill = (224, 242, 254, 170) if (y // 40) % 2 == 0 else (241, 245, 249, 210)
            draw.rectangle([0, y, viewport[0], min(viewport[1], y + 40)], fill=fill)
        grid = int(self._current_meta.get("grid_spacing_px", 160)) if self._current_meta else 160
        visible_half_w = center_x / zoom
        visible_half_h = center_y / zoom
        start_x = int((view_x - visible_half_w) // grid * grid)
        end_x = int(view_x + visible_half_w + grid)
        for world_x in range(start_x, end_x + 1, grid):
            screen_x = center_x + (world_x - view_x) * zoom
            draw.line([(screen_x, 0), (screen_x, viewport[1])], fill=(100, 116, 139, 70), width=1)
        start_y = int((view_y - visible_half_h) // grid * grid)
        end_y = int(view_y + visible_half_h + grid)
        for world_y in range(start_y, end_y + 1, grid):
            screen_y = center_y + (world_y - view_y) * zoom
            draw.line([(0, screen_y), (viewport[0], screen_y)], fill=(100, 116, 139, 70), width=1)

    def _render_screenshot(self, *, phase: str) -> str:
        assert self._current_meta is not None
        viewport = self._viewport_size()
        image = Image.new("RGBA", viewport, (239, 246, 255, 255))
        draw = ImageDraw.Draw(image, "RGBA")
        self._render_background(draw)

        piece = self._piece()
        shape = str(piece.get("shape", "circle"))
        radius = self._piece_radius()
        for option in self._slot_options():
            slot_world = _coerce_pair(option.get("center_world_xy"), default=self._slot_center_world_xy())
            slot_screen = _screen_xy_for_world_xy(
                slot_world,
                view_center_xy=self._view_center_xy,
                viewport_size=viewport,
                view_zoom=self._view_zoom(),
            )
            if slot_screen[0] < -100 or slot_screen[0] > viewport[0] + 100 or slot_screen[1] < -100 or slot_screen[1] > viewport[1] + 100:
                continue
            _draw_hollow_slot(
                draw,
                center=slot_screen,
                kind=str(option.get("shape", shape)),
                radius=radius,
                color=str(option.get("color", piece.get("color", "#f97316"))),
                target=str(option.get("role", "")) == "target",
                radius_scale=self._slot_radius_scale(),
            )

        piece_screen = self._piece_screen_xy()
        shadow_offset = 5 if self._grabbed else 8
        _draw_shape(
            draw,
            center=(piece_screen[0] + shadow_offset, piece_screen[1] + shadow_offset),
            kind=shape,
            radius=radius,
            fill=(15, 23, 42, 54),
            outline=(15, 23, 42, 10),
            width=1,
        )
        _draw_shape(
            draw,
            center=piece_screen,
            kind=shape,
            radius=radius,
            fill=str(piece.get("color", "#f97316")),
            outline=str(piece.get("outline", "#111827")),
            width=4,
        )
        if self._grabbed:
            draw.line([self._cursor_center_xy(), piece_screen], fill=(6, 182, 212, 120), width=3)
        _draw_crosshair(draw, center=self._cursor_center_xy())

        output_path = self._screenshot_dir() / f"{phase}-{self._screenshot_index:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path, compress_level=1)
        self._screenshot_index += 1
        return str(output_path)

    def _observation_metadata(self, episode_id: str) -> dict[str, Any]:
        piece = self._piece()
        return {
            "benchmark": "SlotDragGame",
            "task_type": "slot_drag_game",
            "task_id": episode_id,
            "episode_id": episode_id,
            "perspective": "first_person",
            "coordinate_contract": "qwen3_relative_0_1000_center_locked",
            "cursor_lock": "center",
            "sensitivity": self._sensitivity(),
            "view_zoom": self._view_zoom(),
            "viewport": [self.viewport_width, self.viewport_height],
            "world_size": list(self._world_size()),
            "piece_shape": piece.get("shape"),
            "piece_color": piece.get("color"),
            "piece_color_name": piece.get("color_name"),
            "movable_object_description": self._current_meta.get("movable_object_description"),
            "target_slot_description": self._current_meta.get("target_slot_description"),
            "target_slot_world_xy": list(self._slot_center_world_xy()),
            "distractor_slot_count": sum(1 for option in self._slot_options() if option.get("role") != "target"),
            "slot_radius_scale": self._slot_radius_scale(),
            "tolerance_px": self._tolerance_px(),
        }

    def reset(self, *, task_type: str | None = None, task_id: str | None = None) -> Observation:
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"slot_drag_game dataset directory is missing: {self.dataset_root}. "
                f"Generate it with: {_DATASET_HINT}"
            )
        episode_id = task_type or task_id
        if episode_id is None:
            task_ids = self.list_task_ids(limit=1)
            if not task_ids:
                raise ValueError(f"No slot_drag_game episodes found in {self.dataset_root}")
            episode_id = task_ids[0]

        self._current_meta = self._load_episode(episode_id)
        viewport = self._current_meta.get("viewport")
        if isinstance(viewport, list) and len(viewport) == 2:
            self.viewport_width = int(viewport[0])
            self.viewport_height = int(viewport[1])
        self._current_task_type = episode_id
        self._current_task_id = episode_id
        self._screenshot_index = 0
        self._button_is_down = False
        self._grabbed = False
        self._grab_offset_world_xy = (0.0, 0.0)
        self._piece_world_xy = _coerce_pair(
            self._current_meta.get("piece_start_world_xy", self._current_meta.get("piece_start_xy")),
            default=(900.0, 760.0),
        )
        initial_center = _coerce_pair(
            self._current_meta.get("initial_view_center_xy", self._current_meta.get("initial_cursor_xy")),
            default=(self._world_size()[0] / 2.0, self._world_size()[1] / 2.0),
        )
        self._view_center_xy = _clamp_view_center(
            initial_center,
            world_size=self._world_size(),
            viewport_size=self._viewport_size(),
            view_zoom=self._view_zoom(),
        )
        self.cursor_xy = self._cursor_center_xy()
        screenshot_path = self._render_screenshot(phase="reset")
        instruction = str(
            self._current_meta.get(
                "instruction",
                "Place the solid colored shape into the matching gray outline using the center reticle.",
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

    def _apply_pointer_displacement(self, model_x: float, model_y: float) -> dict[str, Any]:
        pixel_xy = relative_bin_to_pixel_xy(model_x, model_y, self._viewport_size())
        center_x, center_y = self._cursor_center_xy()
        pointer_delta = (pixel_xy[0] - center_x, pixel_xy[1] - center_y)
        sensitivity = self._sensitivity()
        zoom = self._view_zoom()
        view_delta = (
            pointer_delta[0] * sensitivity / zoom,
            pointer_delta[1] * sensitivity / zoom,
        )
        old_view_center = self._view_center_xy
        self._view_center_xy = _clamp_view_center(
            (old_view_center[0] + view_delta[0], old_view_center[1] + view_delta[1]),
            world_size=self._world_size(),
            viewport_size=self._viewport_size(),
            view_zoom=zoom,
        )
        if self._button_is_down and self._grabbed:
            self._piece_world_xy = _clamp_world_xy(
                (
                    self._view_center_xy[0] + self._grab_offset_world_xy[0],
                    self._view_center_xy[1] + self._grab_offset_world_xy[1],
                ),
                self._world_size(),
            )
        self.cursor_xy = self._cursor_center_xy()
        return {
            "model_coordinate_format": "relative_0_1000",
            "model_xy": (float(model_x), float(model_y)),
            "pixel_xy": pixel_xy,
            "pointer_delta_px": pointer_delta,
            "sensitivity": sensitivity,
            "view_zoom": zoom,
            "view_delta_world": view_delta,
            "view_center_before": old_view_center,
            "view_center_after": self._view_center_xy,
        }

    def _apply_drag_displacement(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        pixel_points = [
            relative_bin_to_pixel_xy(x, y, self._viewport_size()) for x, y in points
        ]
        start_px = pixel_points[0]
        end_px = pixel_points[-1]
        pointer_delta = (end_px[0] - start_px[0], end_px[1] - start_px[1])
        sensitivity = self._sensitivity()
        zoom = self._view_zoom()
        view_delta = (
            pointer_delta[0] * sensitivity / zoom,
            pointer_delta[1] * sensitivity / zoom,
        )
        old_view_center = self._view_center_xy
        for previous_px, current_px in zip(pixel_points, pixel_points[1:]):
            segment_delta = (
                current_px[0] - previous_px[0],
                current_px[1] - previous_px[1],
            )
            self._view_center_xy = _clamp_view_center(
                (
                    self._view_center_xy[0] + segment_delta[0] * sensitivity / zoom,
                    self._view_center_xy[1] + segment_delta[1] * sensitivity / zoom,
                ),
                world_size=self._world_size(),
                viewport_size=self._viewport_size(),
                view_zoom=zoom,
            )
            if self._button_is_down and self._grabbed:
                self._piece_world_xy = _clamp_world_xy(
                    (
                        self._view_center_xy[0] + self._grab_offset_world_xy[0],
                        self._view_center_xy[1] + self._grab_offset_world_xy[1],
                    ),
                    self._world_size(),
                )
        self.cursor_xy = self._cursor_center_xy()
        return {
            "model_coordinate_format": "relative_0_1000",
            "model_points": [(float(x), float(y)) for x, y in points],
            "pixel_points": pixel_points,
            "pointer_delta_px": pointer_delta,
            "sensitivity": sensitivity,
            "view_zoom": zoom,
            "view_delta_world": view_delta,
            "view_center_before": old_view_center,
            "view_center_after": self._view_center_xy,
        }

    def _begin_center_grab(self) -> dict[str, Any]:
        self._button_is_down = True
        self._grabbed = self._crosshair_hits_piece()
        if self._grabbed:
            self._grab_offset_world_xy = (
                self._piece_world_xy[0] - self._view_center_xy[0],
                self._piece_world_xy[1] - self._view_center_xy[1],
            )
        return {"grab_started": self._grabbed}

    def _apply_atomic_drag_macro(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        """Execute drag as a full press-drag-release macro in center-lock space.

        The start point is first aimed under the fixed center reticle. While
        held, the view then moves by the screen-space delta from start to end.
        The action is terminal: only a correct one-shot placement succeeds.
        """
        had_prior_button = self._button_is_down
        had_prior_grab = self._grabbed
        self._button_is_down = False
        self._grabbed = False
        self._grab_offset_world_xy = (0.0, 0.0)
        start_x, start_y = points[0]
        aim_info = self._apply_pointer_displacement(start_x, start_y)
        grab_info = self._begin_center_grab()
        drag_info = self._apply_drag_displacement(points)
        aim_delta = aim_info["pointer_delta_px"]
        drag_delta = drag_info["pointer_delta_px"]
        return {
            "model_coordinate_format": "relative_0_1000",
            "model_points": drag_info["model_points"],
            "pixel_points": drag_info["pixel_points"],
            "atomic_drag_macro": True,
            "sensitivity": drag_info["sensitivity"],
            "view_zoom": drag_info["view_zoom"],
            "reticle_recenter_between_drag_points": False,
            "cleared_prior_button_state": had_prior_button or had_prior_grab,
            "pre_grab_delta_from_center_px": aim_delta,
            "held_delta_start_to_end_px": drag_delta,
            "net_internal_pointer_delta_px": (
                aim_delta[0] + drag_delta[0],
                aim_delta[1] + drag_delta[1],
            ),
            "aim_start": aim_info,
            "grab_started": grab_info["grab_started"],
            "drag_delta": drag_info,
        }

    def _make_observation(self, *, phase: str) -> Observation:
        assert self._current_task_id is not None
        assert self._current_meta is not None
        screenshot_path = self._render_screenshot(phase=phase)
        obs = Observation(
            instruction=str(self._current_meta.get("instruction", "")),
            screenshot_path=screenshot_path,
            size_px=self._viewport_size(),
            cursor_xy=self._cursor_center_xy(),
            metadata=self._observation_metadata(self._current_task_id),
        )
        self.last_observation = obs
        return obs

    def _info(self, *, action: Action, executed_kind: str, success: bool = False, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        info = {
            "success": success,
            "executed_kind": executed_kind,
            "cursor_locked": True,
            "cursor_xy": self._cursor_center_xy(),
            "view_center_xy": self._view_center_xy,
            "piece_world_xy": self._piece_world_xy,
            "piece_screen_xy": self._piece_screen_xy(),
            "slot_center_world_xy": self._slot_center_world_xy(),
            "slot_screen_xy": self._slot_screen_xy(),
            "piece_center_distance_px": self._piece_center_distance_px(),
            "target_slot_distance_px": self._slot_center_distance_px(),
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
                extra = self._apply_pointer_displacement(float(action.x), float(action.y))
            elif action.kind == "mouse_down":
                extra = self._begin_center_grab()
            elif action.kind == "mouse_up":
                success = self._button_is_down and self._grabbed and self._is_success()
                reward = 1.0 if success else 0.0
                done = True
                phase = "success" if success else "failure"
                self._button_is_down = False
                self._grabbed = False
            elif action.kind == "left_click":
                success = False
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
                    f"SlotDragGameEnv does not support UI-Venus web action {action.kind!r}"
                )

        obs = self._make_observation(phase=phase)
        info = self._info(action=action, executed_kind=executed_kind, success=success, extra=extra)
        return StepResult(observation=obs, reward=reward, done=done, info=info, action=action)

    def close(self) -> None:
        return None
