from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ...actions import Action
from ...core import Observation, StepResult
from ...domains.slot_drag.environment import (
    DRAG_GRAB_HIT_TEST_VERSION,
    SlotDragGameEnv,
    _coerce_pair,
    _draw_crosshair,
    _draw_hollow_slot,
    _draw_shape,
    _screen_xy_for_world_xy,
)
from ...domains.third_person_drag.environment import (
    ThirdPersonDragCaptchaEnv,
    _draw_free_cursor,
)
from .contracts import (
    default_formal_manifest_path,
    episodes_for_variant,
    load_manifest,
    policy_observation_metadata,
)


def render_paired_drag_scene(
    *,
    viewport: tuple[int, int],
    piece: dict[str, Any],
    piece_xy: tuple[float, float],
    slot_options: list[dict[str, Any]],
    slot_radius_scale: float,
    interaction_xy: tuple[float, float],
    interaction_marker: str,
    grabbed: bool,
) -> Image.Image:
    """Single renderer used by both paired drag coordinate systems."""

    image = Image.new("RGBA", viewport, (239, 246, 255, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle(
        [24, 24, viewport[0] - 24, viewport[1] - 24],
        radius=28,
        fill=(248, 250, 252, 245),
        outline=(100, 116, 139, 120),
        width=3,
    )
    for x in range(80, viewport[0] - 40, 80):
        draw.line([(x, 56), (x, viewport[1] - 56)], fill=(148, 163, 184, 34), width=1)
    for y in range(80, viewport[1] - 40, 80):
        draw.line([(56, y), (viewport[0] - 56, y)], fill=(148, 163, 184, 34), width=1)

    shape = str(piece.get("shape", "square"))
    radius = float(piece.get("radius_px", 34.0))
    for option in slot_options:
        slot_xy = _coerce_pair(option.get("screen_xy"), default=(960.0, 360.0))
        _draw_hollow_slot(
            draw,
            center=slot_xy,
            kind=str(option.get("shape", shape)),
            radius=radius,
            color=str(option.get("color", "#94a3b8")),
            target=str(option.get("role", "")) == "target",
            radius_scale=slot_radius_scale,
        )
    shadow = 5 if grabbed else 8
    _draw_shape(
        draw,
        center=(piece_xy[0] + shadow, piece_xy[1] + shadow),
        kind=shape,
        radius=radius,
        fill=(15, 23, 42, 54),
        outline=(15, 23, 42, 10),
        width=1,
    )
    _draw_shape(
        draw,
        center=piece_xy,
        kind=shape,
        radius=radius,
        fill=str(piece.get("color", "#2563eb")),
        outline=str(piece.get("outline", "#111827")),
        width=4,
    )
    if grabbed:
        draw.line([interaction_xy, piece_xy], fill=(6, 182, 212, 120), width=3)
    if interaction_marker == "mouse_icon":
        _draw_free_cursor(draw, cursor_xy=interaction_xy)
    elif interaction_marker == "red_dot":
        _draw_crosshair(draw, center=interaction_xy)
    else:
        raise ValueError(f"unsupported interaction marker: {interaction_marker!r}")
    return image.convert("RGB")


class _PairedDragMixin:
    manifest_path: Path | None
    benchmark_variant: str
    _paired_episode: dict[str, Any] | None
    _last_evaluator_audit: dict[str, Any]
    _paired_manifest_cache: dict[str, Any] | None
    _paired_episode_index: dict[str, dict[str, Any]]

    def _configure_pairing(self) -> None:
        self.manifest_path = (
            default_formal_manifest_path()
            if self.manifest_path is None
            else Path(self.manifest_path)
        )
        self.dataset_root = self.manifest_path.parent / "cases/drag"

    def _manifest(self) -> dict[str, Any]:
        assert self.manifest_path is not None
        if self._paired_manifest_cache is None:
            self._paired_manifest_cache = load_manifest(self.manifest_path)
        return self._paired_manifest_cache

    def list_task_ids(self, limit: int | None = None) -> list[str]:
        ids = [
            str(episode["episode_id"])
            for episode in episodes_for_variant(self._manifest(), self.benchmark_variant)
        ]
        return ids[:limit] if limit is not None else ids

    def resolve_task_instance_id(self, task_type: str) -> str:
        return task_type

    def _load_pair(self, requested: str | None) -> dict[str, Any]:
        if requested is None:
            requested = self.list_task_ids(limit=1)[0]
        if not self._paired_episode_index:
            for episode in episodes_for_variant(self._manifest(), self.benchmark_variant):
                self._paired_episode_index[str(episode["episode_id"])] = episode
                self._paired_episode_index[str(episode["pair_id"])] = episode
        try:
            episode = copy.deepcopy(self._paired_episode_index[str(requested)])
        except KeyError as exc:
            raise KeyError(
                f"unknown {self.benchmark_variant} training/formal episode {requested!r}"
            ) from exc
        self._paired_episode = episode
        return episode

    def _paired_metadata(self) -> dict[str, Any]:
        assert self._paired_episode is not None
        return policy_observation_metadata(self._paired_episode)

    def _audit(self, *, success: bool | None) -> dict[str, Any]:
        assert self._paired_episode is not None
        shared = self._paired_episode["shared_scene_config"]
        payload = {
            "suite_id": self._paired_episode["suite_id"],
            "pair_id": self._paired_episode["pair_id"],
            "variant": self.benchmark_variant,
            "hidden_dynamics": shared["hidden_dynamics"],
            "success_evaluator": self._paired_episode["success_evaluator"],
            "terminal_success": success,
            "grab_hit_test_version": DRAG_GRAB_HIT_TEST_VERSION,
        }
        if isinstance(self, PairedFirstPersonDragEnv):
            payload["state"] = {
                "view_center_world_xy": list(self._view_center_xy),
                "piece_world_xy": list(self._piece_world_xy),
                "target_world_xy": list(self._slot_center_world_xy()),
                "target_screen_xy": list(self._slot_screen_xy()),
            }
        else:
            payload["state"] = {
                "cursor_screen_xy": list(self.cursor_xy or (0.0, 0.0)),
                "piece_screen_xy": list(self._piece_screen_xy),
                "target_screen_xy": list(self._slot_center_screen_xy()),
            }
        return payload

    def get_evaluator_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._last_evaluator_audit))


@dataclass
class PairedThirdPersonDragEnv(_PairedDragMixin, ThirdPersonDragCaptchaEnv):
    """L0 paired drag variant using the existing screen-coordinate evaluator."""

    manifest_path: Path | None = None
    benchmark_variant: str = field(default="drag_third_person", init=False)
    _paired_episode: dict[str, Any] | None = field(default=None, init=False)
    _last_evaluator_audit: dict[str, Any] = field(default_factory=dict, init=False)
    _paired_manifest_cache: dict[str, Any] | None = field(default=None, init=False)
    _paired_episode_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._configure_pairing()

    def _observation_metadata(self, episode_id: str) -> dict[str, Any]:
        return self._paired_metadata()

    def _render_screenshot(self, *, phase: str) -> str:
        assert self._current_meta is not None
        options = []
        for option in self._slot_options():
            rendered = dict(option)
            rendered["screen_xy"] = list(
                _coerce_pair(
                    option.get("center_screen_xy"),
                    default=self._slot_center_screen_xy(),
                )
            )
            options.append(rendered)
        image = render_paired_drag_scene(
            viewport=self._viewport_size(),
            piece=self._piece(),
            piece_xy=self._piece_screen_xy,
            slot_options=options,
            slot_radius_scale=self._slot_radius_scale(),
            interaction_xy=self.cursor_xy or (
                self.viewport_width / 2.0,
                self.viewport_height / 2.0,
            ),
            interaction_marker="mouse_icon",
            grabbed=self._grabbed,
        )
        output_path = self._screenshot_dir() / f"{phase}-{self._screenshot_index:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, compress_level=1)
        self._screenshot_index += 1
        return str(output_path)

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        episode = self._load_pair(task_id or task_type)
        observation = super().reset(task_id=str(episode["pair_id"]))
        observation.instruction = str(episode["instruction"])
        observation.metadata = self._paired_metadata()
        self._current_task_id = str(episode["episode_id"])
        self.last_observation = observation
        self._last_evaluator_audit = self._audit(success=None)
        return observation

    def step(self, action: Action) -> StepResult:
        before_cursor = self.cursor_xy
        before_piece = self._piece_screen_xy
        raw = super().step(action)
        success = bool(raw.reward == 1.0) if raw.done else None
        raw.observation.metadata = self._paired_metadata()
        raw.observation.instruction = str(self._paired_episode["instruction"])
        self._last_evaluator_audit = self._audit(success=success)
        raw.info = {
            "executed_kind": action.kind,
            "success": bool(success),
            "state_delta": {
                "cursor_delta_px": [
                    (self.cursor_xy or (0.0, 0.0))[axis]
                    - (before_cursor or (0.0, 0.0))[axis]
                    for axis in (0, 1)
                ],
                "piece_delta_px": [
                    self._piece_screen_xy[axis] - before_piece[axis]
                    for axis in (0, 1)
                ],
                "camera_moved": False,
            },
        }
        self.last_observation = raw.observation
        return raw


@dataclass
class PairedFirstPersonDragEnv(_PairedDragMixin, SlotDragGameEnv):
    """L2 paired drag variant using the existing world-coordinate evaluator."""

    manifest_path: Path | None = None
    benchmark_variant: str = field(default="drag_first_person", init=False)
    _paired_episode: dict[str, Any] | None = field(default=None, init=False)
    _last_evaluator_audit: dict[str, Any] = field(default_factory=dict, init=False)
    _paired_manifest_cache: dict[str, Any] | None = field(default=None, init=False)
    _paired_episode_index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._configure_pairing()
        if self.artifact_dir is None:
            from ...domains.slot_drag.paths import environment_run_root

            self.artifact_dir = environment_run_root()
        else:
            self.artifact_dir = Path(self.artifact_dir)

    def _observation_metadata(self, episode_id: str) -> dict[str, Any]:
        return self._paired_metadata()

    def _load_episode(self, episode_id: str) -> dict[str, Any]:
        meta = super()._load_episode(episode_id)
        if self._paired_episode is None:
            return meta
        config = self._paired_episode["environment_config"]
        initial_view_center = config.get("initial_view_center_xy")
        if initial_view_center is not None:
            meta["initial_view_center_xy"] = list(initial_view_center)
        if "world_size" in config:
            meta["world_size"] = list(config["world_size"])
        if "world_translation_xy" in config:
            # Translate only this first-person instance, after its camera override.
            # Shared case metadata and screen coordinates also serve third person.
            dx, dy = config["world_translation_xy"]
            for key in (
                "initial_view_center_xy", "piece_start_world_xy", "slot_center_world_xy"
            ):
                x, y = meta[key]
                meta[key] = [x + dx, y + dy]
            for option in meta["slot_options"]:
                x, y = option["center_world_xy"]
                option["center_world_xy"] = [x + dx, y + dy]
        return meta

    def _uses_strict_reset_frame(self, phase: str) -> bool:
        return bool(
            phase == "reset"
            and self._paired_episode is not None
            and self._paired_episode["environment_config"].get(
                "reset_frame_contract"
            )
            in {
                "shared_layout_different_interaction_marker",
                "shared_layout_same_mouse_marker",
            }
        )

    def _render_screenshot(self, *, phase: str) -> str:
        assert self._current_meta is not None
        strict_reset = self._uses_strict_reset_frame(phase)
        options = []
        for option in self._slot_options():
            rendered = dict(option)
            if strict_reset and "center_screen_xy" in option:
                rendered["screen_xy"] = list(option["center_screen_xy"])
            else:
                world_xy = _coerce_pair(
                    option.get("center_world_xy"),
                    default=self._slot_center_world_xy(),
                )
                rendered["screen_xy"] = list(
                    _screen_xy_for_world_xy(
                        world_xy,
                        view_center_xy=self._view_center_xy,
                        viewport_size=self._viewport_size(),
                        view_zoom=self._view_zoom(),
                    )
                )
            options.append(rendered)
        piece_xy = (
            _coerce_pair(
                self._current_meta.get("piece_start_screen_xy"),
                default=self._piece_screen_xy(),
            )
            if strict_reset
            else self._piece_screen_xy()
        )
        image = render_paired_drag_scene(
            viewport=self._viewport_size(),
            piece=self._piece(),
            piece_xy=piece_xy,
            slot_options=options,
            slot_radius_scale=self._slot_radius_scale(),
            interaction_xy=self._cursor_center_xy(),
            interaction_marker=str(
                self._paired_episode["environment_config"].get(
                    "interaction_marker", "red_dot"
                )
            ),
            grabbed=self._grabbed,
        )
        output_path = self._screenshot_dir() / f"{phase}-{self._screenshot_index:04d}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, compress_level=1)
        self._screenshot_index += 1
        return str(output_path)

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        episode = self._load_pair(task_id or task_type)
        observation = super().reset(task_id=str(episode["pair_id"]))
        observation.instruction = str(episode["instruction"])
        observation.metadata = self._paired_metadata()
        self._current_task_id = str(episode["episode_id"])
        self.last_observation = observation
        self._last_evaluator_audit = self._audit(success=None)
        return observation

    def _direction_adjusted_model_xy(self, x: float, y: float) -> tuple[float, float]:
        assert self._paired_episode is not None
        direction = self._paired_episode["shared_scene_config"]["hidden_dynamics"][
            "direction_xy"
        ]
        return (
            500.0 + (float(x) - 500.0) * float(direction[0]),
            500.0 + (float(y) - 500.0) * float(direction[1]),
        )

    def _apply_pointer_displacement(self, model_x: float, model_y: float) -> dict[str, Any]:
        adjusted = self._direction_adjusted_model_xy(model_x, model_y)
        return super()._apply_pointer_displacement(*adjusted)

    def _apply_drag_displacement(self, points: list[tuple[float, float]]) -> dict[str, Any]:
        adjusted = [self._direction_adjusted_model_xy(x, y) for x, y in points]
        return super()._apply_drag_displacement(adjusted)

    def step(self, action: Action) -> StepResult:
        before_view = self._view_center_xy
        before_piece = self._piece_world_xy
        raw = super().step(action)
        success = bool(raw.reward == 1.0) if raw.done else None
        raw.observation.metadata = self._paired_metadata()
        raw.observation.instruction = str(self._paired_episode["instruction"])
        self._last_evaluator_audit = self._audit(success=success)
        raw.info = {
            "executed_kind": action.kind,
            "success": bool(success),
            "state_delta": {
                "view_moved": self._view_center_xy != before_view,
                "piece_moved": self._piece_world_xy != before_piece,
                "reticle_moved": False,
            },
        }
        self.last_observation = raw.observation
        return raw
