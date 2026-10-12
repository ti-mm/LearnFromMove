from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..actions import Action
from ..core import Observation, Policy, StepResult
from ..models.protocol import PlanCapableBackend, PolicyBackend


@dataclass
class BasePolicy(Policy):
    backend: PolicyBackend
    allowed_kinds: tuple[str, ...]
    condition_name: str
    budget: int | None = None
    _cached_initial_obs: Observation | None = field(default=None, init=False)

    def reset(self) -> None:
        self._cached_initial_obs = None

    def _predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str] | None = None,
        budget: int | None = None,
        condition_name: str | None = None,
    ) -> Action:
        return self.backend.predict_action(
            obs,
            history,
            allowed_kinds=allowed_kinds or self.allowed_kinds,
            budget=self.budget if budget is None else budget,
            condition=condition_name or self.condition_name,
        )


def maybe_plan_actions(
    backend: PolicyBackend,
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None,
    condition_name: str,
) -> list[Action] | None:
    if not isinstance(backend, PlanCapableBackend):
        return None
    try:
        return list(
            backend.predict_plan(
                obs,
                history,
                allowed_kinds=allowed_kinds,
                budget=budget,
                condition=condition_name,
            ),
        )
    except Exception:
        return None
