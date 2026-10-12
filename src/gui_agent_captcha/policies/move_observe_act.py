from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..actions import Action
from ..core import Observation, StepResult
from .base import BasePolicy, maybe_plan_actions

PRIMITIVE_KINDS = ("move_to", "mouse_down", "mouse_up", "left_click", "done")


@dataclass
class PrimitiveSequencePolicy(BasePolicy):
    allowed_kinds: tuple[str, ...] = PRIMITIVE_KINDS
    condition_name: str = "primitives"
    budget: int | None = 1
    _plan: deque[Action] = field(default_factory=deque, init=False)

    def reset(self) -> None:
        super().reset()
        self._plan.clear()

    def predict(self, obs: Observation, history: list[StepResult]) -> Action:
        if not self._plan:
            planned = maybe_plan_actions(
                self.backend,
                obs,
                history,
                allowed_kinds=self.allowed_kinds,
                budget=self.budget,
                condition_name=self.condition_name,
            )
            if planned:
                self._plan.extend(planned)
            else:
                self._plan.append(
                    self._predict_action(
                        obs,
                        history,
                    ),
                )
        return self._plan.popleft()


@dataclass
class ClosedLoopPrimitivePolicy(BasePolicy):
    allowed_kinds: tuple[str, ...] = PRIMITIVE_KINDS
    condition_name: str = "both"
    budget: int | None = 4

    def predict(self, obs: Observation, history: list[StepResult]) -> Action:
        return self._predict_action(obs, history)
