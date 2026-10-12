from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from ..rotation.action_contract import (
    THINK_CLOSE_TAG,
    THINK_OPEN_TAG,
    ResponseFormatError,
    TaskActionError,
    _decode_exact_json_object,
    normalize_prefilled_think_response,
)

SLOT_DRAG_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
)
SLOT_DRAG_PRIMITIVE_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
)
SlotDragActionKind = Literal[
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
]


@dataclass(frozen=True)
class CanonicalSlotDragAction:
    kind: SlotDragActionKind
    x: float | None = None
    y: float | None = None
    points: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "move_to":
            return {"kind": self.kind, "x": self.x, "y": self.y}
        if self.kind == "drag":
            return {
                "kind": self.kind,
                "points": [[x, y] for x, y in self.points],
            }
        return {"kind": self.kind}


@dataclass(frozen=True)
class CanonicalSlotDragResponse:
    thought: str
    action: CanonicalSlotDragAction
    normalized_text: str
    restored_prefilled_think: bool


def _coordinate(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResponseFormatError(f"{field_name} must be a JSON number")
    coordinate = float(value)
    if not math.isfinite(coordinate):
        raise ResponseFormatError(f"{field_name} must be finite")
    if not 0.0 <= coordinate <= 1000.0:
        raise ResponseFormatError(f"{field_name} must be within [0, 1000]")
    return coordinate


def _parse_action_payload(payload: Mapping[str, Any]) -> CanonicalSlotDragAction:
    if set(payload) != {"action"}:
        raise ResponseFormatError("top-level JSON must contain exactly the action key")
    raw_action = payload["action"]
    if not isinstance(raw_action, dict):
        raise ResponseFormatError("action must be a JSON object")
    kind = raw_action.get("kind")
    if kind == "move_to":
        if set(raw_action) != {"kind", "x", "y"}:
            raise ResponseFormatError("move_to must contain exactly kind, x, and y")
        return CanonicalSlotDragAction(
            kind="move_to",
            x=_coordinate(raw_action["x"], field_name="move_to x"),
            y=_coordinate(raw_action["y"], field_name="move_to y"),
        )
    if kind in {"mouse_down", "mouse_up", "left_click"}:
        if set(raw_action) != {"kind"}:
            raise ResponseFormatError(f"{kind} must contain exactly the kind key")
        return CanonicalSlotDragAction(kind=kind)
    if kind == "drag":
        if set(raw_action) != {"kind", "points"}:
            raise ResponseFormatError("drag must contain exactly kind and points")
        raw_points = raw_action["points"]
        if not isinstance(raw_points, list) or len(raw_points) != 2:
            raise ResponseFormatError("drag must contain exactly two coordinate points")
        points: list[tuple[float, float]] = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, list) or len(point) != 2:
                raise ResponseFormatError(f"drag point {index} must be an [x, y] pair")
            points.append(
                (
                    _coordinate(point[0], field_name=f"drag point {index} x"),
                    _coordinate(point[1], field_name=f"drag point {index} y"),
                )
            )
        return CanonicalSlotDragAction(kind="drag", points=tuple(points))
    raise ResponseFormatError(f"unsupported action kind: {kind!r}")


def parse_slot_drag_response(response: str) -> CanonicalSlotDragResponse:
    """Parse the exact SlotDrag v2 SFT wire format, including float coordinates."""

    normalized, restored = normalize_prefilled_think_response(response)
    if not normalized.startswith(THINK_OPEN_TAG):
        raise ResponseFormatError("response must start with <think>")
    if normalized.count(THINK_OPEN_TAG) != 1:
        raise ResponseFormatError("response must contain exactly one <think> tag")
    if normalized.count(THINK_CLOSE_TAG) != 1:
        raise ResponseFormatError("response must contain exactly one </think> tag")
    close_index = normalized.find(THINK_CLOSE_TAG, len(THINK_OPEN_TAG))
    thought = normalized[len(THINK_OPEN_TAG) : close_index]
    if not thought.strip():
        raise ResponseFormatError("reasoning inside <think> must be non-empty")
    if THINK_OPEN_TAG in thought or THINK_CLOSE_TAG in thought:
        raise ResponseFormatError("nested think tags are forbidden")
    json_blob = normalized[close_index + len(THINK_CLOSE_TAG) :].strip()
    if not json_blob or not json_blob.startswith("{"):
        raise ResponseFormatError("</think> must be followed by one JSON object")
    if THINK_OPEN_TAG in json_blob or THINK_CLOSE_TAG in json_blob:
        raise ResponseFormatError("JSON remainder contains think tags")
    action = _parse_action_payload(_decode_exact_json_object(json_blob))
    return CanonicalSlotDragResponse(
        thought=thought,
        action=action,
        normalized_text=normalized,
        restored_prefilled_think=restored,
    )


def require_slot_drag_primitive_action(
    action: CanonicalSlotDragAction,
) -> CanonicalSlotDragAction:
    if action.kind not in SLOT_DRAG_PRIMITIVE_ACTION_KINDS:
        raise TaskActionError(
            f"{action.kind} is format-valid but forbidden in primitive SlotDrag RL"
        )
    return action


def canonicalize_slot_drag_data_action(
    value: Any,
) -> CanonicalSlotDragAction:
    if not isinstance(value, Mapping):
        raise ValueError("canonical SlotDrag dataset action must be a mapping")
    compact = {
        str(key): item
        for key, item in value.items()
        if item is not None
    }
    try:
        return _parse_action_payload({"action": compact})
    except ResponseFormatError as exc:
        raise ValueError(f"invalid canonical SlotDrag dataset action: {exc}") from exc


def slot_drag_actions_equal(
    left: CanonicalSlotDragAction,
    right: CanonicalSlotDragAction,
) -> bool:
    if left.kind != right.kind:
        return False
    if left.kind == "move_to":
        if left.x is None or left.y is None or right.x is None or right.y is None:
            return False
        return math.isclose(left.x, right.x, abs_tol=1e-9) and math.isclose(
            left.y,
            right.y,
            abs_tol=1e-9,
        )
    if left.kind == "drag":
        return left.points == right.points
    return True


def slot_drag_action_sequence_has_loop(
    actions: Sequence[CanonicalSlotDragAction],
) -> bool:
    if len(actions) >= 2 and slot_drag_actions_equal(actions[-1], actions[-2]):
        return True
    if len(actions) >= 4:
        return slot_drag_actions_equal(
            actions[-1],
            actions[-3],
        ) and slot_drag_actions_equal(actions[-2], actions[-4])
    return False


def slot_drag_trajectory_format_valid(responses: Sequence[str]) -> bool:
    if not responses:
        return False
    try:
        for response in responses:
            parse_slot_drag_response(response)
    except ValueError:
        return False
    return True
