from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class RotationReplayServer:
    public_root: Path
    port: int = 0
    process: subprocess.Popen[str] | None = field(default=None, init=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> "RotationReplayServer":
        if self.port == 0:
            self.port = find_free_local_port()
        script = REPO_ROOT / "scripts/interaction/serve_captcha_static_replay.py"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--public-root",
                str(self.public_root),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.monotonic() + 15.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(self.base_url, timeout=1.0):
                    return self
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        self.stop()
        raise RuntimeError(f"rotation replay server did not start at {self.base_url}") from last_error

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.process = None

    def __enter__(self) -> "RotationReplayServer":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.stop()
