"""
BenchmarkEnv adapter for interaction rotation CAPTCHA.

Wraps the local static_replay page (or a compatible live page) as a BenchmarkEnv
so it can be driven by any Policy via the standard reset/step interface.  Default
base_url points to localhost:4321.

Observation  : viewport screenshot (1280×720) — VLM coordinates == page coords.
Atomic action : drag  → execute slider drag; first release decides success/fail
                submit → terminate episode
Primitive seq : move_to / mouse_down / mouse_up / done
                mouse_up triggers CAPTCHA check and terminates success/fail.
                done action → always terminates.
Click actions : click / left_click → execute the click, then terminate as failure.
"""
from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from ...actions import Action, AtomicAction, PrimitiveAction, WebAction
from ...core import Observation, StepResult
from ...envs.browser_base import (
    BrowserBenchmarkEnv,
    overlay_cursor,
    relative_bin_to_pixel_xy,
)
from ...protocol_tracks import ROTATION_TASK_REQUIREMENT
from .online_contract import INTERACTION_ONLINE_VIEWPORT

# Legacy slider geometry used by the environment's fallback path.
SLIDER_LEFT_X  = 248   # actual x=247.8
SLIDER_RIGHT_X = 1032  # actual x=247.8+784.4=1032.2
SLIDER_Y       = 701   # actual y=700.6  (was 745 — WRONG)
SLIDER_THUMB_DIAMETER_PX = 28.0


def _seeded_math_random_init_script(seed: int) -> str:
    seed = seed & 0xFFFFFFFF
    return f"""
(() => {{
  let state = {seed};
  Math.random = () => {{
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  }};
}})();
"""


def _coerce_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


@dataclass
class InteractionCaptchaEnv(BrowserBenchmarkEnv):
    """Live interaction.com rotation-puzzle CAPTCHA environment."""

    base_url: str = "http://localhost:4321/"
    enable_playwright: bool = True
    headless: bool = True
    viewport_width: int = INTERACTION_ONLINE_VIEWPORT[0]
    viewport_height: int = INTERACTION_ONLINE_VIEWPORT[1]
    service_ready_timeout_s: float = 120.0
    service_ready_path: str = "/"
    entry_path: str = "/"
    app_root: Any = None   # no local server
    challenge_seed: int | None = None
    navigate_on_launch: bool = True

    _slider_box: dict | None = field(default=None, init=False)
    _slider_geometry: dict[str, float] | None = field(default=None, init=False)
    _mouse_is_down: bool = field(default=False, init=False)
    _slider_drag_active: bool = field(default=False, init=False)
    _challenge_audit_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------ #
    # Overrides                                                            #
    # ------------------------------------------------------------------ #

    def _launch_browser_session(self) -> None:
        from ...envs.browser_base import load_sync_playwright

        sync_playwright = load_sync_playwright()
        self._playwright = sync_playwright().start()
        browser_launcher = getattr(self._playwright, self.browser_name)
        self._browser = browser_launcher.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        self._context = self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        if self.challenge_seed is not None:
            self._context.add_init_script(
                _seeded_math_random_init_script(int(self.challenge_seed)),
            )
        self._context.route("**/*", self._route_browser_request)
        self.page = self._context.new_page()
        if self.navigate_on_launch:
            self.page.goto(
                self._entry_url(),
                wait_until="load",
                timeout=self.navigation_timeout_ms,
            )
            self._wait_until_page_ready()

    @staticmethod
    def _is_allowed_browser_request_url(url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme in {"about", "blob", "data"}:
            return True
        if parsed.scheme not in {"http", "https", "ws", "wss"}:
            return False
        hostname = parsed.hostname
        if hostname == "localhost":
            return True
        if hostname is None:
            return False
        try:
            return ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            return False

    def _route_browser_request(self, route: Any, request: Any) -> None:
        if self._is_allowed_browser_request_url(str(request.url)):
            route.continue_()
        else:
            route.abort("blockedbyclient")

    def _wait_for_service_ready(self) -> None:
        # For local dev server, do a simple HTTP check; for remote, skip (403 to urllib).
        if "localhost" in self.base_url or "127.0.0.1" in self.base_url:
            import urllib.error
            import urllib.request

            deadline = time.monotonic() + self.service_ready_timeout_s
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                try:
                    urllib.request.urlopen(
                        self.base_url,
                        timeout=min(60.0, max(1.0, remaining)),
                    )
                    return
                except (urllib.error.URLError, OSError) as exc:
                    last_error = exc
                    time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            raise RuntimeError(
                f"Local captcha server not reachable at {self.base_url} after "
                f"{self.service_ready_timeout_s:.1f}s"
            ) from last_error
        return

    def _wait_until_page_ready(self) -> None:
        if self.page is None:
            return
        try:
            self.page.wait_for_selector(
                "[data-article-captcha-canvas]", state="visible", timeout=15_000
            )
        except Exception:
            pass
        try:
            self.page.wait_for_function(
                """
                () => {
                  if (window.__guiAgentCaptchaStaticReplayReady === true) {
                    return true;
                  }
                  const status = document.querySelector("[data-article-captcha-status]");
                  const state = status?.dataset?.state;
                  return Boolean(state && state !== "loading");
                }
                """,
                timeout=15_000,
            )
        except Exception:
            pass
        time.sleep(1.0)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def resolve_task_instance_id(self, task_type: str) -> str | None:
        if self.challenge_seed is None:
            return None
        return f"{task_type}:seed={int(self.challenge_seed)}"

    def _screenshot(self, *, phase: str) -> str:
        if self.page is None:
            return ""
        out = (
            self.artifact_dir
            / "interaction"
            / f"{phase}-{self._screenshot_index:04d}.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        self.page.screenshot(path=str(out), full_page=False)
        try:
            overlay_cursor(out, out, cursor_xy=self.cursor_xy)
        except Exception:
            pass
        self._screenshot_index += 1
        return str(out)

    def _get_slider_box(self) -> dict:
        if self._slider_box is not None:
            return self._slider_box
        if self.page is None:
            return {"x": SLIDER_LEFT_X, "y": SLIDER_Y, "width": 764, "height": 20}
        el = self.page.query_selector("[data-article-captcha-slider]")
        fallback = {"x": SLIDER_LEFT_X, "y": SLIDER_Y, "width": 764, "height": 20}
        self._slider_box = (el.bounding_box() if el else None) or fallback
        return self._slider_box

    def _read_slider_geometry(self) -> dict[str, float]:
        fallback_box = {"x": SLIDER_LEFT_X, "y": SLIDER_Y - 10, "width": SLIDER_RIGHT_X - SLIDER_LEFT_X, "height": 20}
        box = fallback_box
        slider_min = 0.0
        slider_max = 100.0
        slider_value = 0.0

        if self.page is not None:
            el = self.page.query_selector("[data-article-captcha-slider]")
            raw_box = (el.bounding_box() if el else None) or fallback_box
            box = {
                "x": _coerce_float(raw_box.get("x"), fallback_box["x"]),
                "y": _coerce_float(raw_box.get("y"), fallback_box["y"]),
                "width": _coerce_float(raw_box.get("width"), fallback_box["width"]),
                "height": _coerce_float(raw_box.get("height"), fallback_box["height"]),
            }
            try:
                slider_min = _coerce_float(
                    self.page.evaluate("document.querySelector('[data-article-captcha-slider]')?.min"),
                    0.0,
                )
                slider_max = _coerce_float(
                    self.page.evaluate("document.querySelector('[data-article-captcha-slider]')?.max"),
                    100.0,
                )
                slider_value = _coerce_float(
                    self.page.evaluate("document.querySelector('[data-article-captcha-slider]')?.value"),
                    slider_min,
                )
            except Exception:
                slider_min = 0.0
                slider_max = 100.0
                slider_value = slider_min

        slider_left_x = box["x"]
        slider_right_x = box["x"] + box["width"]
        slider_y = box["y"] + box["height"] / 2
        slider_range = slider_max - slider_min
        if slider_range <= 0:
            fraction = 0.5
        else:
            fraction = (slider_value - slider_min) / slider_range
            fraction = min(max(fraction, 0.0), 1.0)
        thumb_diameter = min(SLIDER_THUMB_DIAMETER_PX, float(box["width"]))
        thumb_left_x = slider_left_x + fraction * (float(box["width"]) - thumb_diameter)
        thumb_x = thumb_left_x + thumb_diameter / 2.0
        thumb_top_y = slider_y - thumb_diameter / 2.0

        geometry = {
            "slider_left_x": slider_left_x,
            "slider_right_x": slider_right_x,
            "slider_y": slider_y,
            "slider_min": slider_min,
            "slider_max": slider_max,
            "slider_value": slider_value,
            "thumb_x": thumb_x,
            "action_thumb_x": thumb_x,
            "thumb_left_x": thumb_left_x,
            "thumb_top_y": thumb_top_y,
            "thumb_diameter": thumb_diameter,
        }
        self._slider_box = box
        self._slider_geometry = geometry
        return geometry

    def _slider_value_for_pixel_x(self, x: float, geometry: dict[str, float]) -> float:
        slider_left = float(geometry["slider_left_x"])
        slider_right = float(geometry["slider_right_x"])
        slider_min = float(geometry["slider_min"])
        slider_max = float(geometry["slider_max"])
        slider_width = max(1.0, slider_right - slider_left)
        fraction = min(max((float(x) - slider_left) / slider_width, 0.0), 1.0)
        return slider_min + fraction * (slider_max - slider_min)

    def _slider_hit_test(self, x: float, y: float) -> dict[str, Any]:
        geometry = self._slider_geometry or self._read_slider_geometry()
        box = self._slider_box
        if box is None:
            return {
                "hit": False,
                "pixel_xy": [float(x), float(y)],
                "reason": "slider_geometry_unavailable",
            }

        hit_box = {
            "x": float(geometry["thumb_left_x"]),
            "y": float(geometry["thumb_top_y"]),
            "width": float(geometry["thumb_diameter"]),
            "height": float(geometry["thumb_diameter"]),
        }
        hit = (
            hit_box["x"] <= float(x) <= hit_box["x"] + hit_box["width"]
            and hit_box["y"] <= float(y) <= hit_box["y"] + hit_box["height"]
        )
        return {
            "hit": hit,
            "pixel_xy": [float(x), float(y)],
            "hit_box": hit_box,
            "slider_y": float(geometry["slider_y"]),
            "target": "current_slider_thumb",
        }

    def _dispatch_slider_input_for_drag(self, x: float) -> dict[str, float] | None:
        if self.page is None:
            return None
        geometry = self._slider_geometry or self._read_slider_geometry()
        value = self._slider_value_for_pixel_x(x, geometry)
        self.page.eval_on_selector(
            "[data-article-captcha-slider]",
            """(slider, value) => {
                const descriptor = Object.getOwnPropertyDescriptor(
                    HTMLInputElement.prototype,
                    "value",
                );
                if (descriptor && descriptor.set) {
                    descriptor.set.call(slider, String(value));
                } else {
                    slider.value = String(value);
                }
                slider.dispatchEvent(new Event("input", { bubbles: true }));
            }""",
            float(value),
        )
        self._slider_geometry = None
        return {
            "pixel_x": float(x),
            "value": float(value),
            "slider_left_x": float(geometry["slider_left_x"]),
            "slider_right_x": float(geometry["slider_right_x"]),
        }

    def _dispatch_slider_pointer_down(self) -> bool:
        if self.page is None:
            return False
        self.page.eval_on_selector(
            "[data-article-captcha-slider]",
            """(slider) => {
                slider.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
                if (typeof PointerEvent !== "undefined") {
                    slider.dispatchEvent(new PointerEvent("pointerdown", {
                        bubbles: true,
                        pointerId: 1,
                    }));
                }
            }""",
        )
        return True

    def _dispatch_document_pointer_up(self) -> bool:
        if self.page is None:
            return False
        self.page.evaluate(
            """() => {
                if (typeof PointerEvent !== "undefined") {
                    document.dispatchEvent(new PointerEvent("pointerup", {
                        bubbles: true,
                        pointerId: 1,
                    }));
                }
                document.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
            }"""
        )
        return True

    def _build_instruction(self) -> str:
        return ROTATION_TASK_REQUIREMENT

    def _current_background_image_url(self) -> str:
        if self.page is None:
            return ""
        try:
            return str(
                self.page.evaluate(
                    "document.querySelector('[data-article-captcha-gate]')"
                    "?.dataset.currentBackgroundImageUrl ?? ''"
                )
                or ""
            )
        except Exception:
            return ""

    def _observation_metadata(self) -> dict[str, Any]:
        geometry = self._read_slider_geometry()
        return {
            "task_type": self._current_task_type or "rotation_captcha",
            "task_id": self._current_task_id or "interaction_live",
            "slider_box": self._get_slider_box(),
            **geometry,
            "mouse_is_down": self._mouse_is_down,
            "background_image_url": self._current_background_image_url(),
            **self._challenge_audit_metadata,
        }

    def _dom_status(self) -> tuple[str, str]:
        if self.page is None:
            return "unknown", "unknown"
        gate = self.page.evaluate(
            "document.querySelector('[data-article-captcha-gate]')"
            "?.dataset.gateState ?? 'unknown'"
        )
        state = self.page.evaluate(
            "document.querySelector('[data-article-captcha-status]')"
            "?.dataset.state ?? 'unknown'"
        )
        return str(gate), str(state)

    def _status_text(self) -> str:
        if self.page is None:
            return ""
        return self.page.text_content("[data-article-captcha-status]") or ""

    def _check_and_record(self, info: dict) -> bool:
        """Read DOM success state, populate info dict, return success bool."""
        gate, state = self._dom_status()
        success = gate == "passed" or state == "success"
        info["success"] = success
        info["gate_state"] = gate
        info["status_state"] = state
        info["status_text"] = self._status_text()
        return success

    def _record_forced_failure(self, info: dict, *, reason: str) -> bool:
        """Record the DOM state while enforcing an environment-level failure."""
        gate, state = self._dom_status()
        info["success"] = False
        info["gate_state"] = gate
        info["status_state"] = state
        info["status_text"] = self._status_text()
        info["dom_would_report_success"] = gate == "passed" or state == "success"
        info["failure_reason"] = reason
        return False

    # ------------------------------------------------------------------ #
    # BenchmarkEnv protocol                                                #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        self._ensure_started()
        self._current_task_type = task_type or "rotation_captcha"
        self._current_task_id = task_id or "interaction_live"
        self._screenshot_index = 0
        self._slider_box = None
        self._slider_geometry = None
        self._mouse_is_down = False
        self._slider_drag_active = False

        if self.page is not None:
            # Clear captcha-passed flag so the puzzle shows again.
            # Use goto instead of reload so we recover from error/blank page states.
            try:
                self.page.evaluate("sessionStorage.clear()")
            except Exception:
                pass  # page may be on about:blank or error page — that's fine
            try:
                self.page.goto(
                    self.base_url,
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout_ms,
                )
            except Exception:
                # One retry on navigation failure
                time.sleep(2)
                self.page.goto(
                    self.base_url,
                    wait_until="domcontentloaded",
                    timeout=self.navigation_timeout_ms,
                )
            self._wait_until_page_ready()
        self._reset_cursor_to_playwright_origin()

        ss = self._screenshot(phase="reset")
        metadata = self._observation_metadata()

        self.last_observation = Observation(
            instruction=self._build_instruction(),
            screenshot_path=ss,
            size_px=(self.viewport_width, self.viewport_height),
            cursor_xy=self.cursor_xy,
            metadata=metadata,
        )
        return self.last_observation

    def step(self, action: Action) -> StepResult:  # type: ignore[override]
        done = False
        success: bool | None = None
        info: dict[str, Any] = {"executed_kind": action.kind}
        pixel_xy: tuple[float, float] | None = None

        if isinstance(action, WebAction) and self.page is None:
            raise RuntimeError("Rotation browser page is not configured")

        if self.page is not None:

            # ── Atomic actions ──────────────────────────────────────────
            if isinstance(action, AtomicAction):
                if action.kind == "drag":
                    geometry = self._slider_geometry or self._read_slider_geometry()
                    model_start = action.points[0]
                    (x1, y1) = relative_bin_to_pixel_xy(
                        float(model_start[0]),
                        float(model_start[1]),
                        (self.viewport_width, self.viewport_height),
                    )
                    pixel_points = [
                        relative_bin_to_pixel_xy(
                            float(model_x),
                            float(model_y),
                            (self.viewport_width, self.viewport_height),
                        )
                        for model_x, model_y in action.points
                    ]
                    info.update({
                        "model_coordinate_format": "relative_0_1000",
                        "model_points": [
                            (float(model_x), float(model_y))
                            for model_x, model_y in action.points
                        ],
                        "pixel_points": pixel_points,
                    })
                    start_hit_test = self._slider_hit_test(x1, y1)
                    info["slider_start_hit_test"] = start_hit_test
                    if abs(float(x1) - geometry["action_thumb_x"]) > 30:
                        info.setdefault("coordinate_warnings", []).append({
                            "field": "drag_start_x",
                            "provided": float(x1),
                            "expected_thumb_x": geometry["action_thumb_x"],
                            "reason": "far_from_current_thumb",
                        })
                    self.cursor_xy = tuple(float(value) for value in pixel_points[-1])
                    if start_hit_test["hit"]:
                        m = self.page.mouse
                        m.move(x1, y1)
                        m.down()
                        for x, y in pixel_points[1:]:
                            m.move(x, y)
                        m.up()
                        time.sleep(0.8)
                        success = self._check_and_record(info)
                    else:
                        info["native_drag_dispatched"] = False
                        success = self._record_forced_failure(
                            info,
                            reason="drag_start_missed_slider",
                        )
                    done = True
                elif action.kind == "click":
                    model_x, model_y = action.points[0]
                    x, y = relative_bin_to_pixel_xy(
                        float(model_x),
                        float(model_y),
                        (self.viewport_width, self.viewport_height),
                    )
                    info.update({
                        "model_coordinate_format": "relative_0_1000",
                        "model_points": [(float(model_x), float(model_y))],
                        "pixel_points": [(x, y)],
                    })
                    m = self.page.mouse
                    m.move(x, y)
                    m.down()
                    m.up()
                    self.cursor_xy = (float(x), float(y))
                    time.sleep(0.3)
                    done = True
                    success = self._record_forced_failure(
                        info,
                        reason="click_is_terminal_failure",
                    )

                elif action.kind == "submit":
                    done = True
                    success = self._check_and_record(info)

            # ── Primitive actions ────────────────────────────────────────
            elif isinstance(action, PrimitiveAction):
                if action.kind == "move_to" and action.x is not None:
                    x, y = relative_bin_to_pixel_xy(
                        float(action.x),
                        float(action.y or 0),
                        (self.viewport_width, self.viewport_height),
                    )
                    pixel_xy = (x, y)
                    self.page.mouse.move(x, y)
                    self.cursor_xy = (x, y)
                    info.update({
                        "model_coordinate_format": "relative_0_1000",
                        "model_xy": (float(action.x), float(action.y or 0)),
                        "pixel_xy": pixel_xy,
                    })
                    if self._mouse_is_down and self._slider_drag_active:
                        dispatch_info = self._dispatch_slider_input_for_drag(x)
                        if dispatch_info is not None:
                            info["slider_input_dispatch"] = dispatch_info
                    elif self._mouse_is_down:
                        info["slider_input_dispatch_skipped"] = (
                            "mouse_down_did_not_hit_slider"
                        )

                elif action.kind == "mouse_down":
                    if self._mouse_is_down:
                        info["ignored_redundant_mouse_down"] = True
                    else:
                        cursor_xy = self.cursor_xy or (0.0, 0.0)
                        hit_test = self._slider_hit_test(*cursor_xy)
                        info["slider_hit_test"] = hit_test
                        self.page.mouse.down()
                        if hit_test["hit"]:
                            try:
                                info["slider_pointer_down_dispatch"] = (
                                    self._dispatch_slider_pointer_down()
                                )
                            except Exception:
                                info["slider_pointer_down_dispatch"] = False
                            self._slider_drag_active = True
                        else:
                            info["slider_pointer_down_dispatch"] = False
                            self._slider_drag_active = False
                        self._mouse_is_down = True

                elif action.kind == "mouse_up":
                    slider_drag_was_active = self._slider_drag_active
                    info["slider_drag_was_active"] = slider_drag_was_active
                    self.page.mouse.up()
                    if slider_drag_was_active:
                        try:
                            info["document_pointer_up_dispatch"] = (
                                self._dispatch_document_pointer_up()
                            )
                        except Exception:
                            info["document_pointer_up_dispatch"] = False
                    else:
                        info["document_pointer_up_dispatch"] = False
                    self._mouse_is_down = False
                    self._slider_drag_active = False
                    time.sleep(0.8)   # wait for captcha check animation
                    if slider_drag_was_active:
                        success = self._check_and_record(info)
                    else:
                        success = self._record_forced_failure(
                            info,
                            reason="mouse_up_without_slider_drag",
                        )
                    done = True

                elif action.kind == "left_click":
                    self.page.mouse.down()
                    self.page.mouse.up()
                    self._mouse_is_down = False
                    self._slider_drag_active = False
                    time.sleep(0.8)
                    success = self._record_forced_failure(
                        info,
                        reason="click_is_terminal_failure",
                    )
                    done = True

                elif action.kind == "done":
                    # Explicit termination (give up or confirm success).
                    if self._mouse_is_down:
                        self.page.mouse.up()
                        self._mouse_is_down = False
                        self._slider_drag_active = False
                        time.sleep(0.5)
                    done = True
                    success = self._check_and_record(info)

            elif isinstance(action, WebAction):
                self._dispatch_web(action)
                info.update(self._coordinate_info(action, None))
                if action.kind == "left_double":
                    done = True
                    success = self._record_forced_failure(
                        info,
                        reason="double_click_is_terminal_failure",
                    )
                elif action.kind == "finished":
                    done = True
                    success = self._check_and_record(info)

        ss = self._screenshot(phase="step")
        obs = Observation(
            instruction=self._build_instruction(),
            screenshot_path=ss or (
                self.last_observation.screenshot_path if self.last_observation else ""
            ),
            size_px=(self.viewport_width, self.viewport_height),
            cursor_xy=self.cursor_xy,
            metadata=self._observation_metadata(),
        )
        self.last_observation = obs

        return StepResult(
            observation=obs,
            reward=1.0 if success is True else (0.0 if success is False else None),
            done=done,
            info=info,
            action=action,
        )

    def refresh_observation(self) -> Observation:
        """Capture a fresh frame without executing or scoring an action."""

        if self.last_observation is None:
            raise RuntimeError("reset() must be called before refresh_observation()")
        screenshot_path = self._screenshot(phase="refresh")
        refreshed = Observation(
            instruction=self._build_instruction(),
            screenshot_path=screenshot_path or self.last_observation.screenshot_path,
            size_px=(self.viewport_width, self.viewport_height),
            cursor_xy=self.cursor_xy,
            metadata=self._observation_metadata(),
        )
        self.last_observation = refreshed
        return refreshed

    def close(self) -> None:
        errors: list[tuple[str, Exception]] = []
        for resource_name, close_method in (
            ("page", "close"),
            ("_context", "close"),
            ("_browser", "close"),
            ("_playwright", "stop"),
        ):
            resource = getattr(self, resource_name, None)
            setattr(self, resource_name, None)
            if resource is None:
                continue
            try:
                getattr(resource, close_method)()
            except Exception as exc:
                errors.append((resource_name, exc))

        self._started = False
        self._mouse_is_down = False
        self._slider_drag_active = False
        self._slider_box = None
        self._slider_geometry = None
        try:
            self.stop_local_server()
        except Exception as exc:
            errors.append(("_server_process", exc))

        if errors:
            details = "; ".join(
                f"{resource_name}: {error}" for resource_name, error in errors
            )
            raise RuntimeError(f"failed to close Interaction browser resources: {details}") from errors[0][1]
