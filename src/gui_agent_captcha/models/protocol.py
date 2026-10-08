from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..actions import Action, PrimitiveAction
from ..core import Observation, StepResult


@runtime_checkable
class PolicyBackend(Protocol):
    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action: ...


@runtime_checkable
class PlanCapableBackend(PolicyBackend, Protocol):
    def predict_plan(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> list[PrimitiveAction]: ...
