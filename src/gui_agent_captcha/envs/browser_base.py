from __future__ import annotations

import importlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from PIL import Image, ImageDraw

from ..actions import Action, AtomicAction, PrimitiveAction, WebAction
from ..core import Observation, StepResult
from ..integrations.storage import storage_path

RELATIVE_COORDINATE_MIN = 0.0
RELATIVE_COORDINATE_MAX = 1000.0
_PLAYWRIGHT_PROCESS_START_LOCK = Lock()


def relative_bin_to_pixel_xy(
    x: float,
    y: float,
    size_px: tuple[int, int],
) -> tuple[float, float]:
    """Map Qwen3-VL 0-1000 relative coordinate bins to viewport pixels."""
    width, height = size_px
    if width <= 0 or height <= 0:
        return float(x), float(y)
    x_bin = min(max(float(x), RELATIVE_COORDINATE_MIN), RELATIVE_COORDINATE_MAX)
    y_bin = min(max(float(y), RELATIVE_COORDINATE_MIN), RELATIVE_COORDINATE_MAX)
    return (
        min((x_bin / RELATIVE_COORDINATE_MAX) * width, float(width - 1)),
        min((y_bin / RELATIVE_COORDINATE_MAX) * height, float(height - 1)),
    )


def load_sync_playwright():
    try:
        module = importlib.import_module("playwright.sync_api")
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is not installed. Install gui-agent-captcha[browser] "
            "and run playwright install chromium."
        ) from exc
    return module.sync_playwright


def _cursor_polygons(
    x: float,
    y: float,
    size: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (outer_black, inner_fill) polygon point-lists for an arrow cursor.

    The arrow tip (hot-spot) is placed at (x, y).  ``size`` controls the total
    height of the cursor in pixels; scale it to the image resolution so the
    cursor is always visible.

    Shape: classic upper-left pointing arrow with a rectangular tail notch.
    """
    s = size / 20.0  # normalised scale — shape is designed at size=20

    outer: list[tuple[float, float]] = [
        (x,           y),             # tip
        (x,           y + 18 * s),    # left bottom
        (x + 5 * s,   y + 13 * s),    # left notch
        (x + 7 * s,   y + 18 * s),    # tail bottom
        (x + 9.5 * s, y + 16.5 * s),  # tail right
        (x + 7 * s,   y + 11.5 * s),  # right notch
        (x + 12 * s,  y + 11.5 * s),  # right shoulder
    ]

    # Inset by ~1.5 units to produce the white interior
    d = 1.5 * s
    inner: list[tuple[float, float]] = [
        (x + d,           y + d),
        (x + d,           y + 16 * s),
        (x + 5.5 * s,     y + 12 * s),
        (x + 7 * s,       y + 16 * s),
        (x + 8.5 * s,     y + 15 * s),
        (x + 7 * s,       y + 10.5 * s),
        (x + 11 * s,      y + 10.5 * s),
    ]

    return outer, inner


def render_cursor_image(
    source_image: Image.Image,
    *,
    cursor_xy: tuple[float, float] | None,
    size: int | None = None,
    inner_fill: tuple[int, int, int, int] = (150, 150, 150, 255),
    keep_body_in_frame: bool = False,
) -> Image.Image:
    """Return an RGB copy of *source_image* with an arrow cursor overlaid.

    The cursor hot-spot (tip) is placed at *cursor_xy*.  When *size* is ``None``
    it is auto-scaled to ~2 % of the image height so the cursor is visible on
    both small and high-resolution screenshots.
    """
    image = source_image.convert("RGBA")
    if cursor_xy is not None:
        if size is None:
            size = max(16, int(image.height * 0.02))
        draw = ImageDraw.Draw(image)
        x, y = cursor_xy
        outer, inner = _cursor_polygons(x, y, size)
        if keep_body_in_frame:
            all_points = outer + inner
            flip_x = max(point[0] for point in all_points) > image.width - 1
            flip_y = max(point[1] for point in all_points) > image.height - 1
            if flip_x:
                outer = [(2 * x - point_x, point_y) for point_x, point_y in outer]
                inner = [(2 * x - point_x, point_y) for point_x, point_y in inner]
            if flip_y:
                outer = [(point_x, 2 * y - point_y) for point_x, point_y in outer]
                inner = [(point_x, 2 * y - point_y) for point_x, point_y in inner]
        draw.polygon(outer, fill=(0, 0, 0, 255))
        draw.polygon(inner, fill=inner_fill)
    return image.convert("RGB")


def overlay_cursor(
    source_path: Path,
    output_path: Path,
    *,
    cursor_xy: tuple[float, float] | None,
    size: int | None = None,
    inner_fill: tuple[int, int, int, int] = (150, 150, 150, 255),
    keep_body_in_frame: bool = False,
) -> Path:
    """Overlay an arrow-cursor sprite onto *source_path* and save to *output_path*."""
    with Image.open(source_path) as source_image:
        image = render_cursor_image(
            source_image,
            cursor_xy=cursor_xy,
            size=size,
            inner_fill=inner_fill,
            keep_body_in_frame=keep_body_in_frame,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if output_path.suffix.lower() == ".png":
        save_kwargs["compress_level"] = 1
    image.save(output_path, **save_kwargs)
    return output_path


class MouseProtocol(Protocol):
    def move(self, x: float, y: float) -> None: ...
    def down(self, *, button: str = "left") -> None: ...
    def up(self, *, button: str = "left") -> None: ...


class PageProtocol(Protocol):
    mouse: MouseProtocol

    def goto(
        self,
        url: str,
        *,
        wait_until: str = "load",
        timeout: int = 10_000,
    ) -> Any: ...

    def screenshot(self, *, path: str) -> Any: ...
    def close(self) -> Any: ...


@dataclass
class BrowserActionExecutor:
    page: PageProtocol

    def dispatch(
        self,
        action: PrimitiveAction,
        *,
        size_px: tuple[int, int],
    ) -> tuple[float, float] | None:
        if action.kind == "move_to":
            assert action.x is not None and action.y is not None
            pixel_xy = relative_bin_to_pixel_xy(action.x, action.y, size_px)
            self.page.mouse.move(pixel_xy[0], pixel_xy[1])
            return pixel_xy
        elif action.kind == "mouse_down":
            self.page.mouse.down()
        elif action.kind == "mouse_up":
            self.page.mouse.up()
        elif action.kind == "left_click":
            self.page.mouse.down()
            self.page.mouse.up()
        elif action.kind == "left_double":
            self.page.mouse.down()
            self.page.mouse.up()
            self.page.mouse.down()
            self.page.mouse.up()
        elif action.kind == "right_single":
            self.page.mouse.down(button="right")
            self.page.mouse.up(button="right")
        elif action.kind == "done":
            return
        else:
            raise ValueError(f"Unsupported primitive action {action.kind}")


@dataclass
class BrowserBenchmarkEnv:
    base_url: str
    artifact_dir: Path = field(
        default_factory=lambda: storage_path("runs", "browser", "environment"),
    )
    app_root: Path | None = None
    enable_playwright: bool = False
    browser_name: str = "chromium"
    headless: bool = True
    navigation_timeout_ms: int = 10_000
    service_ready_timeout_s: float = 15.0
    service_ready_poll_interval_s: float = 0.25
    service_ready_path: str = "/"
    entry_path: str = "/"
    screenshot_subdir: str = "browser"
    python_executable: str = "python"
    viewport_width: int = 1280
    viewport_height: int = 1800
    screenshot_format: str = "png"
    screenshot_quality: int | None = None
    page: PageProtocol | None = None
    cursor_xy: tuple[float, float] | None = None
    last_observation: Observation | None = None
    _server_process: subprocess.Popen[str] | None = field(default=None, init=False)
    _playwright: Any = field(default=None, init=False)
    _browser: Any = field(default=None, init=False)
    _context: Any = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _screenshot_index: int = field(default=0, init=False)
    _current_task_type: str | None = field(default=None, init=False)
    _current_task_id: str | None = field(default=None, init=False)

    def _request_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        if payload is None:
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode())
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())

    def start_local_server(self, app_root: Path, *, python_executable: str = "python") -> None:
        if self._server_process is not None:
            return
        self._server_process = subprocess.Popen(
            [python_executable, "app.py"],
            cwd=app_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop_local_server(self) -> None:
        if self._server_process is None:
            return
        self._server_process.terminate()
        self._server_process.wait(timeout=5)
        self._server_process = None

    def _service_ready_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.service_ready_path.lstrip('/')}"

    def _entry_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.entry_path.lstrip('/')}"

    def _wait_for_service_ready(self) -> None:
        deadline = time.time() + self.service_ready_timeout_s
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self._service_ready_url()):
                    return
            except urllib.error.URLError as exc:
                last_error = exc
                time.sleep(self.service_ready_poll_interval_s)
        raise RuntimeError(
            f"Timed out waiting for service readiness at {self._service_ready_url()} "
            f"after {self.service_ready_timeout_s:.2f}s"
        ) from last_error

    def _wait_until_page_ready(self) -> None:
        return

    def _benchmark_name(self) -> str:
        name = self.__class__.__name__
        return name[:-3] if name.endswith("Env") else name

    def _puzzle_api_path(self) -> str:
        return "api/get_puzzle"

    def _screenshot_dir(self) -> Path:
        task_type = self._current_task_type or "unknown-task-type"
        task_id = (self._current_task_id or "unknown-task-id").replace("/", "_")
        return (
            self.artifact_dir
            / self.screenshot_subdir
            / self._benchmark_name()
            / task_type
            / task_id
        )

    def _capture_screenshot(self, *, phase: str) -> str | None:
        if self.page is None or not self.enable_playwright:
            return None
        screenshot_format = self.screenshot_format.lower()
        suffix = "jpg" if screenshot_format == "jpeg" else screenshot_format
        output_path = self._screenshot_dir() / f"{phase}-{self._screenshot_index:04d}.{suffix}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"path": str(output_path), "full_page": False}
        if screenshot_format != "png":
            kwargs["type"] = screenshot_format
        if self.screenshot_quality is not None:
            kwargs["quality"] = self.screenshot_quality
        self.page.screenshot(**kwargs)
        self._screenshot_index += 1
        return str(output_path)

    def _build_observation_metadata(
        self,
        *,
        payload: dict[str, Any],
        task_type: str | None,
        task_id: str | None,
    ) -> dict[str, Any]:
        return {
            "task_type": task_type,
            "task_id": task_id,
            "raw_payload": payload,
        }

    def _launch_browser_session(self) -> None:
        sync_playwright = load_sync_playwright()
        # uvloop cannot safely spawn multiple Playwright driver processes at
        # the same instant from separate event-loop threads in one Ray worker.
        with _PLAYWRIGHT_PROCESS_START_LOCK:
            self._playwright = sync_playwright().start()
        browser_launcher = getattr(self._playwright, self.browser_name)
        self._browser = browser_launcher.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        self._context = self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        self.page = self._context.new_page()
        self.page.goto(
            self._entry_url(),
            wait_until="load",
            timeout=self.navigation_timeout_ms,
        )
        self._wait_until_page_ready()

    def _ensure_started(self) -> None:
        if self._started:
            return
        if self.app_root is not None:
            self.start_local_server(self.app_root, python_executable=self.python_executable)
        self._wait_for_service_ready()
        if self.enable_playwright:
            self._launch_browser_session()
        self._started = True

    def start(self) -> None:
        self._ensure_started()

    def _reset_cursor_to_playwright_origin(self) -> None:
        """Start each episode at Playwright's deterministic mouse origin.

        Playwright initializes its tracked mouse position to ``(0, 0)``. A page
        object is reused across benchmark episodes, however, and navigation does
        not reset that position. Explicitly moving to the origin keeps the real
        browser cursor and the observation's ``cursor_xy`` in sync.
        """

        self.cursor_xy = (0.0, 0.0)
        if self.page is not None:
            self.page.mouse.move(*self.cursor_xy)

    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation:
        self._ensure_started()
        self._current_task_type = task_type
        self._current_task_id = task_id
        self._screenshot_index = 0
        payload = self._request_json(self._puzzle_api_path())
        screenshot_path = self._capture_screenshot(phase="reset") or str(
            payload.get("image_path", ""),
        )
        self.last_observation = Observation(
            instruction=payload.get("prompt", ""),
            screenshot_path=screenshot_path,
            size_px=tuple(payload.get("image_size", [0, 0])),
            cursor_xy=self.cursor_xy,
            metadata=self._build_observation_metadata(
                payload=payload,
                task_type=task_type,
                task_id=task_id,
            ),
        )
        return self.last_observation

    def _fetch_task_result(self) -> tuple[bool | None, dict]:
        try:
            payload = self._request_json("api/last_result")
            if payload.get("has_result"):
                return bool(payload.get("correct")), payload
            return None, {}
        except Exception:
            return None, {}

    def _dispatch_atomic(self, action: AtomicAction) -> None:
        if self.page is None:
            raise RuntimeError("Browser page is not configured")
        mouse = self.page.mouse
        if action.kind == "click":
            model_x, model_y = action.points[0]
            x, y = relative_bin_to_pixel_xy(model_x, model_y, self._observation_size_px())
            mouse.move(x, y)
            mouse.down()
            mouse.up()
            self.cursor_xy = (x, y)
        elif action.kind == "drag":
            pixel_points = [
                relative_bin_to_pixel_xy(x, y, self._observation_size_px())
                for x, y in action.points
            ]
            x1, y1 = pixel_points[0]
            mouse.move(x1, y1)
            mouse.down()
            for x, y in pixel_points[1:]:
                mouse.move(x, y)
            mouse.up()
            self.cursor_xy = pixel_points[-1]
        elif action.kind == "submit":
            return

    def _dispatch_web(self, action: WebAction) -> None:
        if self.page is None:
            raise RuntimeError("Browser page is not configured")
        if action.kind in {"left_double", "right_single"}:
            model_x, model_y = action.points[0]
            x, y = relative_bin_to_pixel_xy(model_x, model_y, self._observation_size_px())
            self.page.mouse.move(x, y)
            if action.kind == "left_double":
                double_click = getattr(self.page.mouse, "dblclick", None)
                if callable(double_click):
                    double_click(x, y)
                else:
                    for _ in range(2):
                        self.page.mouse.down()
                        self.page.mouse.up()
            else:
                click = getattr(self.page.mouse, "click", None)
                if not callable(click):
                    raise RuntimeError("Browser mouse does not support right click")
                click(x, y, button="right")
            self.cursor_xy = (x, y)
            return
        if action.kind == "scroll":
            wheel = getattr(self.page.mouse, "wheel", None)
            if not callable(wheel):
                raise RuntimeError("Browser mouse does not support scrolling")
            wheel(0, -720 if action.direction == "up" else 720)
            return
        if action.kind in {"type", "hotkey", "press_enter"}:
            keyboard = getattr(self.page, "keyboard", None)
            if keyboard is None:
                raise RuntimeError("Browser page does not expose a keyboard")
            if action.kind == "type":
                keyboard.type(action.content or "")
                keyboard.press("Enter")
            elif action.kind == "hotkey":
                aliases = {
                    "ctrl": "Control",
                    "alt": "Alt",
                    "shift": "Shift",
                    "meta": "Meta",
                }
                key = "+".join(
                    aliases.get(part.casefold(), part)
                    for part in (action.key or "").split("+")
                )
                keyboard.press(key)
            else:
                keyboard.press("Enter")
            return
        if action.kind == "wait":
            duration_s = 1.0 if action.duration_s is None else action.duration_s
            wait_for_timeout = getattr(self.page, "wait_for_timeout", None)
            if callable(wait_for_timeout):
                wait_for_timeout(round(duration_s * 1000))
            else:
                time.sleep(duration_s)
            return
        if action.kind == "finished":
            return
        raise ValueError(
            f"BrowserBenchmarkEnv does not support UI-Venus web action {action.kind!r}"
        )

    def _observation_size_px(self) -> tuple[int, int]:
        if self.last_observation is not None and self.last_observation.size_px != (0, 0):
            return self.last_observation.size_px
        return self.viewport_width, self.viewport_height

    def _primitive_pixel_xy(
        self,
        action: PrimitiveAction,
    ) -> tuple[float, float] | None:
        if action.kind != "move_to":
            return None
        assert action.x is not None and action.y is not None
        return relative_bin_to_pixel_xy(action.x, action.y, self._observation_size_px())

    def _coordinate_info(
        self,
        action: Action,
        pixel_xy: tuple[float, float] | None,
    ) -> dict[str, Any]:
        if isinstance(action, PrimitiveAction) and action.kind == "move_to":
            assert action.x is not None and action.y is not None
            return {
                "model_coordinate_format": "relative_0_1000",
                "model_xy": (float(action.x), float(action.y)),
                "pixel_xy": pixel_xy,
            }
        if isinstance(action, AtomicAction) and action.kind in {"click", "drag"}:
            model_points = [(float(x), float(y)) for x, y in action.points]
            return {
                "model_coordinate_format": "relative_0_1000",
                "model_points": model_points,
                "pixel_points": [
                    relative_bin_to_pixel_xy(x, y, self._observation_size_px())
                    for x, y in model_points
                ],
            }
        if isinstance(action, WebAction) and action.points:
            model_points = [(float(x), float(y)) for x, y in action.points]
            return {
                "model_coordinate_format": "relative_0_1000",
                "model_points": model_points,
                "pixel_points": [
                    relative_bin_to_pixel_xy(x, y, self._observation_size_px())
                    for x, y in model_points
                ],
            }
        return {}

    def refresh_observation(self) -> Observation:
        """Capture a fresh frame without executing or scoring an action."""

        if self.last_observation is None:
            raise RuntimeError("reset() must be called before refresh_observation()")
        screenshot_path = self._capture_screenshot(phase="refresh")
        refreshed = Observation(
            instruction=self.last_observation.instruction,
            screenshot_path=screenshot_path or self.last_observation.screenshot_path,
            size_px=self.last_observation.size_px,
            cursor_xy=self.cursor_xy,
            metadata=dict(self.last_observation.metadata),
        )
        self.last_observation = refreshed
        return refreshed

    def step(self, action: Action) -> StepResult:
        pixel_xy: tuple[float, float] | None = None
        if self.page is not None:
            if isinstance(action, PrimitiveAction):
                pixel_xy = BrowserActionExecutor(self.page).dispatch(
                    action,
                    size_px=self._observation_size_px(),
                )
                if action.kind == "move_to":
                    self.cursor_xy = pixel_xy
            elif isinstance(action, AtomicAction):
                self._dispatch_atomic(action)
            elif isinstance(action, WebAction):
                self._dispatch_web(action)
            else:  # pragma: no cover - Action is a closed union
                raise TypeError(f"Unsupported browser action object: {action!r}")
        observation = self.last_observation or Observation("", "", (0, 0), self.cursor_xy, {})
        screenshot_path = self._capture_screenshot(phase="step")
        updated = Observation(
            instruction=observation.instruction,
            screenshot_path=screenshot_path or observation.screenshot_path,
            size_px=observation.size_px,
            cursor_xy=self.cursor_xy,
            metadata=dict(observation.metadata),
        )
        self.last_observation = updated
        is_primitive_done = isinstance(action, PrimitiveAction) and action.kind == "done"
        is_atomic_submit = isinstance(action, AtomicAction) and action.kind == "submit"
        is_web_finished = isinstance(action, WebAction) and action.kind == "finished"
        done = is_primitive_done or is_atomic_submit or is_web_finished
        info: dict[str, Any] = {"executed_kind": action.kind}
        info.update(self._coordinate_info(action, pixel_xy))
        reward: float | None = None
        if done:
            success, _result_info = self._fetch_task_result()
            if success is not None:
                info["success"] = success
                reward = 1.0 if success else 0.0
        return StepResult(
            observation=updated,
            reward=reward,
            done=done,
            info=info,
            action=action,
        )

    def close(self) -> None:
        for resource_name in ("page", "_context", "_browser"):
            resource = getattr(self, resource_name, None)
            if resource is not None:
                try:
                    resource.close()
                finally:
                    setattr(self, resource_name, None)
        if self._playwright is not None:
            try:
                self._playwright.stop()
            finally:
                self._playwright = None
        self._started = False
        self.stop_local_server()
