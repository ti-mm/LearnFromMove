from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import Action


@dataclass
class Observation:
    instruction: str
    screenshot_path: str
    size_px: tuple[int, int]
    cursor_xy: tuple[float, float] | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    observation: Observation
    reward: float | None
    done: bool
    info: dict[str, Any] = field(default_factory=dict)
    action: Action | None = None


class BenchmarkEnv(Protocol):
    def reset(
        self,
        *,
        task_type: str | None = None,
        task_id: str | None = None,
    ) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...

    def close(self) -> None: ...


class Policy(Protocol):
    def predict(
        self,
        obs: Observation,
        history: list[StepResult],
    ) -> Action: ...
