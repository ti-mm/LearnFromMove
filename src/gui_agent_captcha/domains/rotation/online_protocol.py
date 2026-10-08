from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gui_agent_captcha.actions import PrimitiveAction

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
RELATIVE_COORDINATE_MIN = 0.0
RELATIVE_COORDINATE_MAX = 1000.0


class ResponseParseError(ValueError):
    """Raised when a sampled model response violates the online action schema."""


class ProtocolViolation(ValueError):
    """Raised when a parsed action is illegal in the current mouse state."""


class TrajectoryOutcome(str, Enum):
    SUCCESS = "success"
    BROWSER_FAILURE = "browser_failure"
    PARSE_ERROR = "parse_error"
    PROTOCOL_ERROR = "protocol_error"
    MAX_STEPS = "max_steps"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class ParsedInteractionResponse:
    thought: str
    action: PrimitiveAction


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ResponseParseError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ResponseParseError(f"non-finite JSON number is forbidden: {value}")


def _decode_exact_json_object(blob: str) -> dict[str, Any]:
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    try:
        value, end = decoder.raw_decode(blob)
    except (json.JSONDecodeError, ResponseParseError) as exc:
        if isinstance(exc, ResponseParseError):
            raise
        raise ResponseParseError("response does not contain one valid JSON object") from exc
    if blob[end:].strip():
        raise ResponseParseError("trailing content after the JSON action is forbidden")
    if not isinstance(value, dict):
        raise ResponseParseError("the response JSON value must be an object")
    return value


def _coordinate(value: Any, *, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise ResponseParseError(f"move_to {field_name} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ResponseParseError(f"move_to {field_name} must be finite")
    if not RELATIVE_COORDINATE_MIN <= number <= RELATIVE_COORDINATE_MAX:
        raise ResponseParseError(
            f"move_to {field_name} must be within "
            f"[{RELATIVE_COORDINATE_MIN:g}, {RELATIVE_COORDINATE_MAX:g}]"
        )
    return number


def _parse_action(payload: dict[str, Any]) -> PrimitiveAction:
    if set(payload) != {"action"}:
        raise ResponseParseError("top-level JSON must contain exactly the action key")
    action = payload["action"]
    if not isinstance(action, dict):
        raise ResponseParseError("action must be a JSON object")

    kind = action.get("kind")
    if kind == "move_to":
        if set(action) != {"kind", "x", "y"}:
            raise ResponseParseError("move_to must contain exactly kind, x, and y")
        return PrimitiveAction(
            kind="move_to",
            x=_coordinate(action["x"], field_name="x"),
            y=_coordinate(action["y"], field_name="y"),
        )
    if kind in {"mouse_down", "mouse_up"}:
        if set(action) != {"kind"}:
            raise ResponseParseError(f"{kind} must contain exactly the kind key")
        return PrimitiveAction(kind=kind)
    raise ResponseParseError(f"unsupported action kind: {kind!r}")


def parse_interaction_response(response: str) -> ParsedInteractionResponse:
    """Parse the exact SFT response contract into one safe primitive action."""

    if not isinstance(response, str):
        raise ResponseParseError("model response must be text")
    stripped = response.strip()
    if not stripped.startswith(THINK_OPEN_TAG):
        raise ResponseParseError("response must start with <think>")

    close_index = stripped.find(THINK_CLOSE_TAG, len(THINK_OPEN_TAG))
    if close_index < 0:
        raise ResponseParseError("response is missing </think>")
    thought = stripped[len(THINK_OPEN_TAG) : close_index].strip()
    if not thought:
        raise ResponseParseError("reasoning inside <think> must be non-empty")
    if THINK_OPEN_TAG in thought or THINK_CLOSE_TAG in thought:
        raise ResponseParseError("nested think tags are forbidden")

    json_blob = stripped[close_index + len(THINK_CLOSE_TAG) :].strip()
    if not json_blob:
        raise ResponseParseError("response is missing the JSON action")
    payload = _decode_exact_json_object(json_blob)
    return ParsedInteractionResponse(thought=thought, action=_parse_action(payload))


@dataclass
class OnlineProtocolState:
    max_steps: int = 6
    step_count: int = field(default=0, init=False)
    cursor_has_moved: bool = field(default=False, init=False)
    mouse_is_down: bool = field(default=False, init=False)
    is_terminal: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @property
    def reached_max_steps(self) -> bool:
        return self.step_count >= self.max_steps and not self.is_terminal

    def accept(self, action: PrimitiveAction) -> None:
        """Validate and record one action without mutating state on rejection."""

        if self.is_terminal:
            raise ProtocolViolation("actions after terminal state are forbidden")
        if self.reached_max_steps:
            raise ProtocolViolation("maximum step count has already been reached")

        if action.kind == "move_to":
            next_cursor_has_moved = True
            next_mouse_is_down = self.mouse_is_down
            next_terminal = False
        elif action.kind == "mouse_down":
            if not self.cursor_has_moved:
                raise ProtocolViolation("mouse_down requires a prior move_to")
            if self.mouse_is_down:
                raise ProtocolViolation("repeated mouse_down is forbidden")
            next_cursor_has_moved = self.cursor_has_moved
            next_mouse_is_down = True
            next_terminal = False
        elif action.kind == "mouse_up":
            if not self.mouse_is_down:
                raise ProtocolViolation("mouse_up requires an active mouse_down")
            next_cursor_has_moved = self.cursor_has_moved
            next_mouse_is_down = False
            next_terminal = True
        else:
            raise ProtocolViolation(f"unsupported online action kind: {action.kind!r}")

        self.cursor_has_moved = next_cursor_has_moved
        self.mouse_is_down = next_mouse_is_down
        self.is_terminal = next_terminal
        self.step_count += 1


def trajectory_reward(outcome: TrajectoryOutcome) -> float | None:
    """Return policy reward, or None when infrastructure invalidates the sample."""

    rewards: dict[TrajectoryOutcome, float | None] = {
        TrajectoryOutcome.SUCCESS: 1.0,
        TrajectoryOutcome.BROWSER_FAILURE: 0.0,
        TrajectoryOutcome.PARSE_ERROR: -0.2,
        TrajectoryOutcome.PROTOCOL_ERROR: -0.2,
        TrajectoryOutcome.MAX_STEPS: -0.2,
        TrajectoryOutcome.INFRA_ERROR: None,
    }
    return rewards[outcome]
