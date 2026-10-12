from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from ...actions import Action
from ...benchmarks.exploration_depth.contracts import (
    default_formal_manifest_path,
    episodes_for_variant,
    load_manifest,
    policy_observation_metadata,
)
from ...core import Observation, StepResult
from ...envs.browser_base import overlay_cursor
from .environment import InteractionCaptchaEnv


@dataclass
class PairedRotationEnv(InteractionCaptchaEnv):
    """Paired relative-angle contract layered onto the static replay env."""

    manifest_path: Path | None = None
    benchmark_variant: str = field(default="rotation_inner", init=False)
    navigate_on_launch: bool = False
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

    def _read_rotation_state(self) -> dict[str, Any]:
        if self.page is None:
            return {}
        try:
            value = self.page.evaluate(
                """() => ({
                  sliderValue: Number(document.querySelector('[data-article-captcha-slider]')?.value),
                  rotationRegion: document.querySelector('[data-article-captcha-gate]')?.dataset.rotationRegion,
                  gateState: document.querySelector('[data-article-captcha-gate]')?.dataset.gateState,
                  attempt: window.__guiAgentCaptchaLastAttempt || null
                })"""
            )
            return dict(value) if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _wait_until_page_ready(self) -> None:
        if self.page is None:
            return
        try:
            self.page.wait_for_selector(
                "[data-article-captcha-canvas]",
                state="visible",
                timeout=15_000,
            )
            self.page.wait_for_function(
                "() => window.__guiAgentCaptchaStaticReplayReady === true",
                timeout=15_000,
            )
        except Exception:
            return

    def _screenshot(self, *, phase: str) -> str:
        if self.page is None:
            return ""
        pair_id = (
            str(self._paired_episode["pair_id"])
            if self._paired_episode is not None
            else "unknown-pair"
        )
        output_path = (
            self.artifact_dir
            / "exploration_depth"
            / "rotation"
            / pair_id
            / self.benchmark_variant
            / f"{phase}-{self._screenshot_index:04d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(output_path), full_page=False, timeout=15_000)
        overlay_cursor(output_path, output_path, cursor_xy=self.cursor_xy)
        self._screenshot_index += 1
        return str(output_path)

    def _replay_spec(self, episode: dict[str, Any]) -> dict[str, Any]:
        shared = episode["shared_scene_config"]
        challenge = copy.deepcopy(shared["replay_challenge"])
        challenge["rotationRegion"] = (
            "outer" if self.benchmark_variant == "rotation_outer" else "center"
        )
        return {
            "backgroundImageUrl": "/" + str(shared["background_asset"]).lstrip("/"),
            "sliderBox": copy.deepcopy(shared["slider_geometry"]),
            "challenge": challenge,
        }

    def _safe_observation(self, raw: Observation) -> Observation:
        assert self._paired_episode is not None
        return Observation(
            instruction=str(self._paired_episode["instruction"]),
            screenshot_path=raw.screenshot_path,
            size_px=raw.size_px,
            cursor_xy=raw.cursor_xy,
            metadata=policy_observation_metadata(self._paired_episode),
        )

    def _audit(self, *, success: bool | None) -> dict[str, Any]:
        assert self._paired_episode is not None
        shared = self._paired_episode["shared_scene_config"]
        return {
            "suite_id": self._paired_episode["suite_id"],
            "pair_id": self._paired_episode["pair_id"],
            "variant": self.benchmark_variant,
            "hidden_mapping": self._paired_episode["environment_config"]["hidden_mapping"],
            "target_relative_angle_deg": shared["target_relative_angle_deg"],
            "tolerance_deg": shared["tolerance_deg"],
            "rotation_state": self._read_rotation_state(),
            "terminal_success": success,
        }

    def get_evaluator_audit(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._last_evaluator_audit))

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        requested = task_id or task_type
        if requested is None:
            requested = self.list_task_ids(limit=1)[0]
        episode = self._episode(str(requested))
        self._paired_episode = episode
        self.challenge_seed = int(episode["case_seed"])
        self.viewport_width, self.viewport_height = (
            int(episode["viewport"][0]),
            int(episode["viewport"][1]),
        )
        service_origin = self.base_url.split("?", 1)[0]
        query = urlencode(
            {"replaySpec": json.dumps(self._replay_spec(episode), separators=(",", ":"))}
        )
        self.base_url = f"{service_origin}?{query}"
        raw = super().reset(
            task_type=str(episode["pair_id"]),
            task_id=str(episode["episode_id"]),
        )
        observation = self._safe_observation(raw)
        self.last_observation = observation
        self._last_evaluator_audit = self._audit(success=None)
        return observation

    def step(self, action: Action) -> StepResult:
        before = self._read_rotation_state()
        raw = super().step(action)
        after = self._read_rotation_state()
        success = bool(raw.reward == 1.0) if raw.done else None
        # The replay page keeps the slider enabled and visibly says "Try
        # again" after an unsuccessful release.  Reflect that browser
        # contract: only a successful release is terminal, so a policy can
        # observe the failed attempt and make a causal correction.
        done = bool(raw.done and raw.reward == 1.0)
        observation = self._safe_observation(raw.observation)
        self.last_observation = observation
        self._last_evaluator_audit = self._audit(success=success)
        return StepResult(
            observation=observation,
            reward=raw.reward,
            done=done,
            info={
                "executed_kind": action.kind,
                "success": bool(success),
                "state_delta": {
                    "slider_position_changed": before.get("sliderValue")
                    != after.get("sliderValue"),
                    "visual_rotation_changed": before.get("sliderValue")
                    != after.get("sliderValue"),
                },
            },
            action=action,
        )


@dataclass
class PairedInnerRotationEnv(PairedRotationEnv):
    benchmark_variant: str = field(default="rotation_inner", init=False)


@dataclass
class PairedOuterRotationEnv(PairedRotationEnv):
    benchmark_variant: str = field(default="rotation_outer", init=False)
