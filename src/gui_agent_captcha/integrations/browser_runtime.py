from __future__ import annotations

import fcntl
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class OnlineInfrastructureError(RuntimeError):
    """Raised when a rollout cannot produce a policy-valid sample."""

    reward: None = None

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        errors: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.errors = errors


class BrowserSlotTimeout(OnlineInfrastructureError):
    """Raised when no cross-process browser slot becomes available."""


@dataclass
class BrowserSlotLease:
    slot_index: int
    path: Path
    _fd: int | None = field(repr=False)

    @property
    def released(self) -> bool:
        return self._fd is None

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def __enter__(self) -> BrowserSlotLease:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


@dataclass(frozen=True)
class BrowserSlotPool:
    root: Path
    capacity: int = 4
    acquire_timeout_s: float = 300.0
    poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("browser slot capacity must be positive")
        if self.acquire_timeout_s < 0:
            raise ValueError("browser slot timeout cannot be negative")
        if self.poll_interval_s <= 0:
            raise ValueError("browser slot poll interval must be positive")

    def acquire(self) -> BrowserSlotLease:
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.acquire_timeout_s
        while True:
            for slot_index in range(self.capacity):
                path = self.root / f"browser-{slot_index:03d}.lock"
                fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                except Exception:
                    os.close(fd)
                    raise

                os.ftruncate(fd, 0)
                os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
                return BrowserSlotLease(
                    slot_index=slot_index,
                    path=path,
                    _fd=fd,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BrowserSlotTimeout(
                    f"timed out after {self.acquire_timeout_s:.3f}s waiting for "
                    f"one of {self.capacity} browser slots in {self.root}",
                    attempts=0,
                )
            time.sleep(min(self.poll_interval_s, remaining))


def ensure_verl_processor_rope_binding(processor: Any) -> None:
    """Mirror verl.hf_processor's Qwen position-id binding when needed."""

    if processor is None or hasattr(processor, "get_rope_index"):
        return
    if processor.__class__.__name__ != "Qwen3VLProcessor":
        return

    import types

    from transformers import AutoConfig
    from transformers.models.qwen3_vl import Qwen3VLModel

    name_or_path = getattr(getattr(processor, "tokenizer", None), "name_or_path", None)
    if not name_or_path:
        raise ValueError("Qwen3VLProcessor is missing tokenizer.name_or_path")
    processor.config = AutoConfig.from_pretrained(
        name_or_path,
        trust_remote_code=True,
    )
    processor.get_rope_index = types.MethodType(
        Qwen3VLModel.get_rope_index,
        processor,
    )
    if hasattr(Qwen3VLModel, "get_vision_position_ids"):
        processor.get_vision_position_ids = types.MethodType(
            Qwen3VLModel.get_vision_position_ids,
            processor,
        )
