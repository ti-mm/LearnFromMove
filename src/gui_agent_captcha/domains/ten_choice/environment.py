from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...actions import Action, AtomicAction, PrimitiveAction, WebAction
from ...core import Observation, StepResult
from ...envs.browser_base import BrowserBenchmarkEnv
from ...protocol_tracks import build_ten_choice_instruction
from .icon_assets import DEFAULT_ICON_DIR
from .paths import environment_run_root, source_episodes_root

_DATASET_HINT = (
    "Set GUI_CAPTCHA_STORAGE_ROOT, then run PYTHONPATH=src python -m "
    "gui_agent_captcha.domains.ten_choice.dataset "
    f"--icon-dir {DEFAULT_ICON_DIR}"
)


@dataclass
class HoverRevealEnv(BrowserBenchmarkEnv):
    """Benchmark environment for hover-reveal icon-selection tasks.

    Ten visually identical icons are displayed; each reveals a unique text
    label when the mouse hovers over it.  The task is to click the icon whose
    label matches the instruction.

    Episodes are served directly from local *file://* URLs — no HTTP server is
    required.  The agent must use ``move_to`` primitives to discover each
    icon's label.  Any action that releases the left mouse button
    (``mouse_up``, ``left_click``, ``click``, or ``drag``) commits the single
    CAPTCHA attempt.

    Answer checking reads the DOM element ``<div id="result">`` that the HTML's
    *onclick* handler writes to, so no coordinate math is required.

    This task is designed as a high-coupling diagnostic: static screenshots
    are information-theoretically uninformative (all icons look identical), so
    the ``base``, ``feedback``, ``primitives``, and ``compute_matched``
    conditions should achieve ~10% (chance level), while the ``both``
    condition should approach ~90%+ by hovering each icon to reveal its label.
    """

    artifact_dir: Path = field(default_factory=environment_run_root)
    dataset_root: Path = field(
        default_factory=lambda: source_episodes_root() / "test"
    )
    # base_url is accepted for BrowserBenchmarkEnv interface compatibility
    # but is never used — episodes load from file:// URLs.
    base_url: str = "file://"
    enable_playwright: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    _current_meta: dict[str, Any] | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # BrowserBenchmarkEnv overrides
    # ------------------------------------------------------------------

    def _wait_for_service_ready(self) -> None:
        """No-op: local file:// episodes need no HTTP readiness check."""
        return

    def _entry_url(self) -> str:
        """Return about:blank for initial browser launch.

        The real episode URL is set in reset() on each call.
        """
        return "about:blank"

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"hover_reveal dataset directory is missing: {self.dataset_root}. "
                f"Generate it with: {_DATASET_HINT}"
            )
        self._ensure_started()
        episode_id = task_type or task_id
        if episode_id is None:
            candidates = sorted(
                p.name for p in self.dataset_root.iterdir() if p.is_dir()
            )
            if not candidates:
                raise ValueError(
                    f"No episode directories found in {self.dataset_root}"
                )
            episode_id = candidates[0]

        ep_dir = self.dataset_root / episode_id
        if not ep_dir.exists():
            raise FileNotFoundError(
                f"hover_reveal episode {episode_id!r} was not found in {self.dataset_root}. "
                f"Generate the dataset with: {_DATASET_HINT}"
            )
        self._current_meta = json.loads((ep_dir / "meta.json").read_text())
        self._current_task_type = episode_id
        self._current_task_id = episode_id
        self._screenshot_index = 0

        if self.page is not None:
            html_url = (ep_dir / "index.html").resolve().as_uri()
            self.page.goto(
                html_url,
                wait_until="load",
                timeout=self.navigation_timeout_ms,
            )
        self._reset_cursor_to_playwright_origin()

        target_label = str(self._current_meta.get("target_label", "")).strip()
        instruction = (
            build_ten_choice_instruction(target_label)
            if target_label
            else self._current_meta["instruction"]
        )
        screenshot_path = self._capture_screenshot(phase="reset") or ""
        self.last_observation = Observation(
            instruction=instruction,
            screenshot_path=screenshot_path,
            size_px=(self.viewport_width, self.viewport_height),
            cursor_xy=self.cursor_xy,
            metadata={
                "episode_id": episode_id,
                "target_idx": self._current_meta["target_idx"],
                "target_label": self._current_meta["target_label"],
                "icon_centers_xy": self._current_meta["icon_centers_xy"],
                "icon_asset": self._current_meta.get("icon_asset"),
                "icon_source_name": self._current_meta.get("icon_source_name"),
                "task_type": episode_id,
                "task_id": episode_id,
            },
        )
        return self.last_observation

    def _fetch_task_result(self) -> tuple[bool | None, dict[str, Any]]:
        """Read click result from DOM and compare against ground truth.

        The HTML's ``selectIcon(idx)`` onclick handler writes
        ``JSON.stringify({clicked: idx})`` into ``<div id="result">``.
        We read that element via ``page.evaluate`` and compare against
        ``target_idx`` from the episode sidecar.

        Returns ``(None, {})`` when the page has not been clicked yet or
        when running without a real Playwright page (e.g. in unit tests).
        """
        if self._current_meta is None or self.page is None:
            return None, {}
        evaluate = getattr(self.page, "evaluate", None)
        if evaluate is None:
            return None, {}
        try:
            raw: str = evaluate(
                "() => document.getElementById('result').textContent"
            )
            if not raw:
                return None, {}
            data = json.loads(raw)
            clicked_idx = data.get("clicked")
            target_idx = self._current_meta["target_idx"]
            correct = clicked_idx == target_idx
            return correct, {
                "clicked_idx": clicked_idx,
                "target_idx": target_idx,
                "target_label": self._current_meta.get("target_label"),
            }
        except Exception:
            return None, {}

    def step(self, action: Action) -> StepResult:
        result = super().step(action)
        is_atomic_release = (
            isinstance(action, AtomicAction) and action.kind in {"click", "drag"}
        )
        is_primitive_release = (
            isinstance(action, PrimitiveAction)
            and action.kind in {"mouse_up", "left_click"}
        )
        is_web_release = isinstance(action, WebAction) and action.kind == "left_double"
        if not (is_atomic_release or is_primitive_release or is_web_release):
            return result

        success, result_info = self._fetch_task_result()
        strict_success = bool(success)
        info = dict(result.info)
        info.update(result_info)
        info["success"] = strict_success
        info["strict_first_release"] = True
        # Kept for compatibility with existing result aggregation.
        info["strict_first_click"] = True
        if success is None:
            info["result_missing"] = True
        return StepResult(
            observation=result.observation,
            reward=1.0 if strict_success else 0.0,
            done=True,
            info=info,
            action=action,
        )
