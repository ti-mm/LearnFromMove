from __future__ import annotations

import argparse
import html
import importlib.util
import json
import mimetypes
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

from ...actions import PrimitiveAction
from ...custom_envs import build_benchmark_variant
from .contracts import (
    EXPLORATION_BENCHMARK_VARIANTS,
    default_formal_manifest_path,
    load_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SHOWCASE_TEMPLATE = Path(__file__).with_name("showcase.html")
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_DEMO_SESSIONS = 1


def _rotation_page() -> str:
    script_path = REPO_ROOT / "scripts/interaction/serve_captcha_static_replay.py"
    spec = importlib.util.spec_from_file_location("exploration_rotation_replay", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load rotation replay server: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.build_page())


def _episode_by_variant(manifest: dict[str, Any], variant: str) -> dict[str, Any]:
    episodes = [episode for episode in manifest["episodes"] if episode["variant"] == variant]
    if not episodes:
        raise KeyError(variant)
    return dict(sorted(episodes, key=lambda row: row["pair_id"])[0])


def _public_catalog(manifest: dict[str, Any]) -> dict[str, Any]:
    display = {
        "ten_choice_third_person": ("Ten-Choice", "Exocentric", "L1"),
        "ten_choice_first_person": ("Ten-Choice", "Egocentric", "L2"),
        "rotation_inner": ("Rotation", "Inner Ring", "L2"),
        "rotation_outer": ("Rotation", "Outer Ring", "L2"),
        "drag_third_person": ("Drag-and-Drop", "Exocentric", "L0"),
        "drag_first_person": ("Drag-and-Drop", "Egocentric", "L2"),
    }
    variants = []
    for variant, (family_name, variant_name, level) in display.items():
        episode = _episode_by_variant(manifest, variant)
        family = str(episode["family"])
        variants.append(
            {
                "key": variant,
                "family": family,
                "family_name": family_name,
                "variant_name": variant_name,
                "exploration_level": level,
                "instruction": str(episode["instruction"]),
                "reset_frame": f"paired_reset_examples/{family}/{variant}.png",
            }
        )
    return {
        "suite_id": manifest["suite_id"],
        "total_episode_count": manifest["total_episode_count"],
        "pair_count_per_family": manifest["pair_count_per_family"],
        "variants": variants,
    }


def build_index(manifest: dict[str, Any]) -> str:
    catalog = json.dumps(_public_catalog(manifest), ensure_ascii=False, separators=(",", ":"))
    catalog = catalog.replace("<", "\\u003c").replace(">", "\\u003e")
    return SHOWCASE_TEMPLATE.read_text(encoding="utf-8").replace(
        "__EXPLORATION_DEPTH_CATALOG__",
        catalog,
        1,
    )


@dataclass
class _DemoSession:
    environment: Any
    observation: Any
    frame_index: int = 0
    interaction_index: int = 0
    done: bool = False
    success: bool = False


class DemoManager:
    """Run real benchmark environments on one worker thread for browser previews."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        rotation_base_url: str,
        artifact_root: Path,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.rotation_base_url = rotation_base_url
        self.artifact_root = artifact_root.resolve()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="exploration-showcase-env",
        )
        self._sessions: OrderedDict[str, _DemoSession] = OrderedDict()
        self._closed = False
        self._submit_lock = threading.Lock()

    def _submit(self, function: Any, *args: Any) -> Any:
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("demo manager is closed")
            future = self._executor.submit(function, *args)
        return future.result(timeout=45)

    def reset(self, variant: str) -> dict[str, Any]:
        return self._submit(self._reset, variant)

    def _reset(self, variant: str) -> dict[str, Any]:
        allowed = {contract.key for contract in EXPLORATION_BENCHMARK_VARIANTS}
        if variant not in allowed:
            raise ValueError(f"unknown benchmark variant: {variant}")
        while len(self._sessions) >= _MAX_DEMO_SESSIONS:
            _, stale = self._sessions.popitem(last=False)
            self._close_environment(stale.environment)
        session_id = uuid4().hex
        artifact_dir = self.artifact_root / session_id
        kwargs: dict[str, Any] = {
            "manifest_path": self.manifest_path,
            "artifact_dir": artifact_dir,
        }
        if variant.startswith("rotation_"):
            kwargs["base_url"] = self.rotation_base_url
        environment = build_benchmark_variant(variant, **kwargs)
        try:
            observation = environment.reset()
        except Exception:
            self._close_environment(environment)
            raise
        session = _DemoSession(environment=environment, observation=observation)
        self._sessions[session_id] = session
        return self._payload(session_id, session)

    def step(self, session_id: str, action_payload: dict[str, Any]) -> dict[str, Any]:
        return self._submit(self._step, session_id, action_payload)

    def _step(self, session_id: str, action_payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(session_id)
        if session.done:
            raise RuntimeError("episode is terminal; reset the environment")
        kind = str(action_payload.get("kind", ""))
        if kind not in {"move_to", "mouse_down", "mouse_up", "left_click", "done"}:
            raise ValueError(f"unsupported demo action: {kind}")
        if kind == "move_to":
            x = _relative_coordinate(action_payload.get("x"), "x")
            y = _relative_coordinate(action_payload.get("y"), "y")
            action = PrimitiveAction(kind="move_to", x=x, y=y)
        else:
            action = PrimitiveAction(kind=kind)
        result = session.environment.step(action)
        session.observation = result.observation
        session.frame_index += 1
        session.interaction_index += 1
        session.done = bool(result.done)
        session.success = bool(result.reward == 1.0) if result.done else False
        self._sessions.move_to_end(session_id)
        return self._payload(session_id, session)

    def frame_path(self, session_id: str) -> Path:
        return self._submit(self._frame_path, session_id)

    def _frame_path(self, session_id: str) -> Path:
        return Path(self._session(session_id).observation.screenshot_path).resolve()

    def _session(self, session_id: str) -> _DemoSession:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise KeyError("unknown or expired demo session") from error

    @staticmethod
    def _payload(session_id: str, session: _DemoSession) -> dict[str, Any]:
        observation = session.observation
        cursor = observation.cursor_xy
        return {
            "session_id": session_id,
            "instruction": str(observation.instruction),
            "viewport": [int(observation.size_px[0]), int(observation.size_px[1])],
            "cursor_xy": list(cursor) if cursor is not None else None,
            "frame_index": session.frame_index,
            "interaction_index": session.interaction_index,
            "frame_url": f"api/demo/frame/{session_id}?v={session.frame_index}",
            "done": session.done,
            "success": session.success,
        }

    def close(self) -> None:
        with self._submit_lock:
            if self._closed:
                return
            future = self._executor.submit(self._close)
            self._closed = True
        future.result(timeout=30)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _close(self) -> None:
        for session in self._sessions.values():
            self._close_environment(session.environment)
        self._sessions.clear()

    @staticmethod
    def _close_environment(environment: Any) -> None:
        try:
            environment.close()
        except Exception:
            traceback.print_exc()


def _relative_coordinate(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1000]")
    coordinate = float(value)
    if not 0.0 <= coordinate <= 1000.0:
        raise ValueError(f"{name} must be in [0, 1000]")
    return coordinate


def _canonical_route_path(path: str) -> str:
    """Strip an optional frontend-preview proxy prefix from application routes."""

    markers = (
        "/api/",
        "/paired_reset_examples/",
        "/assets/",
        "/cases/",
        "/rotation-replay",
        "/live/ten-choice",
        "/manifest.json",
        "/favicon.ico",
    )
    for marker in markers:
        index = path.find(marker)
        if index >= 0:
            return path[index:]
    return path


def make_handler(manifest_path: Path) -> type[BaseHTTPRequestHandler]:
    suite_root = manifest_path.parent.resolve()
    manifest = load_manifest(manifest_path)
    index = build_index(manifest).encode("utf-8")
    rotation_page = _rotation_page().encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route_path = _canonical_route_path(parsed.path)
            if route_path in {"/", "/index.html", "/preview", "/preview/"}:
                return self._send(index, "text/html; charset=utf-8")
            if route_path == "/api/health":
                return self._send(b'{"status":"ok"}\n', "application/json")
            if route_path in {"/api/manifest", "/manifest.json"}:
                return self._send(manifest_path.read_bytes(), "application/json")
            if route_path == "/rotation-replay":
                return self._send(rotation_page, "text/html; charset=utf-8")
            if route_path == "/live/ten-choice":
                variant = parse_qs(parsed.query).get("variant", [""])[0]
                try:
                    episode = _episode_by_variant(manifest, variant)
                except KeyError:
                    return self._error(HTTPStatus.NOT_FOUND, "unknown ten-choice variant")
                case_path = suite_root / episode["shared_scene_config"]["html_path"]
                case_relative = Path(episode["shared_scene_config"]["html_path"])
                case_href = f"/{case_relative.parent}/"
                page = case_path.read_text(encoding="utf-8").replace(
                    "<head>",
                    f'<head><base href="{html.escape(case_href, quote=True)}">',
                    1,
                )
                return self._send(page.encode("utf-8"), "text/html; charset=utf-8")
            if route_path.startswith("/api/demo/frame/"):
                session_id = route_path.rsplit("/", 1)[-1]
                try:
                    frame_path = self._demo_manager().frame_path(session_id)
                    if not frame_path.is_file():
                        raise FileNotFoundError(frame_path)
                except (KeyError, FileNotFoundError):
                    return self._error(HTTPStatus.NOT_FOUND, "demo frame not found")
                content_type = mimetypes.guess_type(str(frame_path))[0] or "image/png"
                return self._send(frame_path.read_bytes(), content_type)
            if route_path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
                return

            relative = route_path.lstrip("/")
            candidate = (suite_root / relative).resolve()
            try:
                candidate.relative_to(suite_root)
            except ValueError:
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            if candidate.is_file():
                content_type = mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
                return self._send(candidate.read_bytes(), content_type)
            if "text/html" in self.headers.get("Accept", ""):
                return self._send(index, "text/html; charset=utf-8")
            return self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            route_path = _canonical_route_path(parsed.path)
            try:
                payload = self._read_json()
                if route_path == "/api/demo/reset":
                    response = self._demo_manager().reset(str(payload.get("variant", "")))
                    return self._json(response)
                if route_path == "/api/demo/action":
                    session_id = str(payload.get("session_id", ""))
                    action = payload.get("action")
                    if not session_id or not isinstance(action, dict):
                        raise ValueError("session_id and action are required")
                    response = self._demo_manager().step(session_id, action)
                    return self._json(response)
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            except (KeyError, ValueError) as error:
                return self._error(HTTPStatus.BAD_REQUEST, str(error))
            except RuntimeError as error:
                return self._error(HTTPStatus.CONFLICT, str(error))
            except Exception as error:
                traceback.print_exc()
                return self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"environment preview failed: {type(error).__name__}",
                )

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("invalid Content-Length") from error
            if length <= 0 or length > _MAX_REQUEST_BYTES:
                raise ValueError("request body must be non-empty and at most 64 KiB")
            try:
                payload = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                raise ValueError("request body must be valid JSON") from error
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _demo_manager(self) -> DemoManager:
            manager = getattr(self.server, "demo_manager", None)
            if not isinstance(manager, DemoManager):
                raise RuntimeError("environment preview is unavailable")
            return manager

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
            return self._send(body, "application/json", status=status)

        def _error(self, status: HTTPStatus, message: str) -> None:
            return self._json({"error": message, "status": int(status)}, status=status)

        def _send(
            self,
            body: bytes,
            content_type: str,
            *,
            status: HTTPStatus = HTTPStatus.OK,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: object) -> None:
            print(f"{self.address_string()} - {format_string % args}", flush=True)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the exploration-depth benchmark showcase")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    manifest_path = (args.manifest or default_formal_manifest_path()).resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(manifest_path))
    server.daemon_threads = True
    rotation_origin = f"http://127.0.0.1:{server.server_port}/rotation-replay"
    manager = DemoManager(
        manifest_path=manifest_path,
        rotation_base_url=rotation_origin,
        artifact_root=manifest_path.parent / "showcase_runs",
    )
    server.demo_manager = manager  # type: ignore[attr-defined]
    print(f"Exploration-depth showcase: http://{args.host}:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        manager.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
