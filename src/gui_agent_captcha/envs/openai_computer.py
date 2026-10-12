from __future__ import annotations

import copy
from typing import Any

from ..actions import AtomicAction, PrimitiveAction
from ..core import Observation, StepResult
from ..models.openai_native_computer import (
    OPENAI_COMPUTER_ACTION_KINDS,
    OpenAIComputerAction,
    validate_native_action,
)

SIX_ACTION_ENVIRONMENT_ACTIONS = ("move", "click", "drag")


def _normalized_point(
    x_value: Any,
    y_value: Any,
    *,
    observation: Observation,
) -> tuple[float, float]:
    width, height = observation.size_px
    return float(x_value) * 1000.0 / width, float(y_value) * 1000.0 / height


def _has_modifiers(native: dict[str, Any]) -> bool:
    return bool(native.get("keys"))


def _environment_action(
    action: OpenAIComputerAction,
    observation: Observation,
) -> PrimitiveAction | AtomicAction | None:
    native = action.action
    if _has_modifiers(native):
        return None
    if action.kind == "move":
        x, y = _normalized_point(native["x"], native["y"], observation=observation)
        return PrimitiveAction(kind="move_to", x=x, y=y)
    if action.kind == "click" and native["button"] == "left":
        return AtomicAction(
            kind="click",
            points=[
                _normalized_point(native["x"], native["y"], observation=observation)
            ],
        )
    if action.kind == "drag":
        return AtomicAction(
            kind="drag",
            points=[
                _normalized_point(point["x"], point["y"], observation=observation)
                for point in native["path"]
            ],
        )
    return None


def _no_environment_effect_result(
    action: OpenAIComputerAction,
    observation: Observation,
) -> StepResult:
    return StepResult(
        observation=observation,
        reward=None,
        done=False,
        info={
            "executed_kind": action.kind,
            "native_tool_type": action.tool_type,
            "native_action": action.to_dict(),
            "environment_effect": "none",
            "environment_actions": [],
            "state_delta": {},
        },
        action=action,  # type: ignore[arg-type]
    )


def execute_openai_computer_action(
    environment: object,
    action: OpenAIComputerAction,
    observation: Observation,
) -> StepResult:
    """Execute native computer actions and mouse functions in the same environment."""

    if action.tool_type == "function":
        if action.kind not in {"mouse_down", "mouse_up"} or action.action != {"type": action.kind}:
            raise ValueError("mouse function must be mouse_down or mouse_up without arguments")
        internal_action = PrimitiveAction(kind=action.kind)  # type: ignore[arg-type]
    elif action.tool_type == "computer":
        if action.kind not in OPENAI_COMPUTER_ACTION_KINDS:
            raise ValueError(f"unexpected OpenAI computer action {action.kind!r}")
        validate_native_action(action.action, size_px=observation.size_px)
        internal_action = _environment_action(action, observation)
    else:
        raise ValueError(f"unexpected tool_type {action.tool_type!r}")
    if internal_action is None:
        return _no_environment_effect_result(action, observation)

    step = getattr(environment, "step", None)
    if not callable(step):
        raise RuntimeError("environment does not expose step()")
    result = step(internal_action)
    if not isinstance(result, StepResult):
        raise RuntimeError("environment step() returned an invalid value")
    info = dict(result.info)
    info.update(
        {
            "executed_kind": action.kind,
            "native_tool_type": action.tool_type,
            "native_action": copy.deepcopy(action.action),
            "environment_effect": (
                "mouse_function_adapter" if action.tool_type == "function" else "six_action_adapter"
            ),
            "environment_actions": [internal_action.to_dict()],
        }
    )
    result.info = info
    result.action = action  # type: ignore[assignment]
    return result


__all__ = ["SIX_ACTION_ENVIRONMENT_ACTIONS", "execute_openai_computer_action"]
