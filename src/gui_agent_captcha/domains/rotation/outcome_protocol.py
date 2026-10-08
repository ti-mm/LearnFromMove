from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from gui_agent_captcha.domains.rotation.action_contract import (
    CanonicalRotationAction,
    CanonicalRotationResponse,
    TaskActionError,
    action_sequence_has_loop,
    parse_rotation_response,
    require_primitive_action,
    thought_has_excessive_repetition,
)
from gui_agent_captcha.integrations.online_rl import (
    ActionBudgetViolationV4,
    LoopViolationV4,
    NoProgressViolationV4,
    PolicyActionViolationV4,
    ProtocolViolationV4,
)


def parse_online_response_v4(response: str) -> CanonicalRotationResponse:
    """Use the one shared final-SFT parser without an online-specific fork."""

    return parse_rotation_response(response)


def trajectory_format_valid_v4(responses: Sequence[str]) -> bool:
    """Return whether every response in a non-empty trajectory is canonical."""

    if not responses:
        return False
    try:
        for response in responses:
            parse_online_response_v4(response)
    except ValueError:
        return False
    return True


@dataclass
class OnlineProtocolStateV4:
    max_steps: int = 8
    cursor: tuple[int, int] | None = field(default=None, init=False)
    mouse_down: bool = field(default=False, init=False)
    held_move: bool = field(default=False, init=False)
    released: bool = field(default=False, init=False)
    step_count: int = field(default=0, init=False)
    _actions: list[CanonicalRotationAction] = field(default_factory=list, init=False)
    _responses: list[str] = field(default_factory=list, init=False)
    _thoughts: list[str] = field(default_factory=list, init=False)
    _screenshot_fingerprints: list[str] = field(default_factory=list, init=False)
    _screenshot_state_snapshots: list[tuple[Any, ...]] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    @property
    def action_history(self) -> tuple[CanonicalRotationAction, ...]:
        return tuple(self._actions)

    @property
    def response_history(self) -> tuple[str, ...]:
        return tuple(self._responses)

    @property
    def thought_history(self) -> tuple[str, ...]:
        return tuple(self._thoughts)

    @property
    def screenshot_fingerprints(self) -> tuple[str, ...]:
        return tuple(self._screenshot_fingerprints)

    @property
    def reached_action_budget(self) -> bool:
        return self.step_count >= self.max_steps and not self.released

    def snapshot(self) -> dict[str, Any]:
        return {
            "cursor": list(self.cursor) if self.cursor is not None else None,
            "mouse_down": self.mouse_down,
            "held_move": self.held_move,
            "released": self.released,
            "step_count": self.step_count,
            "action_history": [action.to_dict() for action in self._actions],
            "response_history": list(self._responses),
            "screenshot_fingerprints": list(self._screenshot_fingerprints),
        }

    def _progress_snapshot(self) -> tuple[Any, ...]:
        return (
            self.cursor,
            self.mouse_down,
            self.held_move,
            self.released,
            tuple(action.to_dict().items() for action in self._actions),
        )

    def record_screenshot(self, fingerprint: str) -> None:
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("screenshot fingerprint must be non-empty text")
        snapshot = self._progress_snapshot()
        if (
            self._screenshot_fingerprints
            and self._screenshot_fingerprints[-1] == fingerprint
            and self._screenshot_state_snapshots[-1] == snapshot
        ):
            raise NoProgressViolationV4(
                "screenshot fingerprint and public protocol state are unchanged"
            )
        self._screenshot_fingerprints.append(fingerprint)
        self._screenshot_state_snapshots.append(snapshot)

    def accept_response(self, response: str) -> CanonicalRotationResponse:
        if self.released:
            raise ProtocolViolationV4("actions after mouse_up are forbidden")
        if self.reached_action_budget:
            raise ActionBudgetViolationV4("maximum action budget has been reached")

        parsed = parse_online_response_v4(response)
        normalized_thought = " ".join(parsed.thought.split()).casefold()
        if normalized_thought in {
            " ".join(thought.split()).casefold() for thought in self._thoughts
        }:
            raise LoopViolationV4("repeated think is forbidden")
        if thought_has_excessive_repetition(parsed.thought):
            raise LoopViolationV4("repetitive think is forbidden")
        try:
            action = require_primitive_action(parsed.action)
        except TaskActionError as exc:
            raise PolicyActionViolationV4(str(exc)) from exc

        if action.kind == "move_to":
            assert action.x is not None and action.y is not None
            if self.cursor == (action.x, action.y):
                raise LoopViolationV4("repeated move_to is forbidden")
        elif action.kind == "mouse_down":
            if self.cursor is None:
                raise ProtocolViolationV4("mouse_down requires a prior move_to")
            if self.mouse_down:
                raise ProtocolViolationV4("repeated mouse_down is forbidden")
        elif action.kind == "mouse_up":
            if not self.mouse_down:
                raise ProtocolViolationV4("mouse_up requires an active mouse_down")
            if not self.held_move:
                raise ProtocolViolationV4("mouse_up requires at least one held move_to")

        candidate = (*self._actions, action)
        if action_sequence_has_loop(candidate):
            if len(candidate) >= 4:
                raise LoopViolationV4("A-B-A-B action cycle is forbidden")
            raise LoopViolationV4("repeated action is forbidden")

        if action.kind == "move_to":
            assert action.x is not None and action.y is not None
            self.cursor = (action.x, action.y)
            if self.mouse_down:
                self.held_move = True
        elif action.kind == "mouse_down":
            self.mouse_down = True
            self.held_move = False
        else:
            self.mouse_down = False
            self.released = True

        self._actions.append(action)
        self._responses.append(response)
        self._thoughts.append(parsed.thought)
        self.step_count += 1
        return parsed
