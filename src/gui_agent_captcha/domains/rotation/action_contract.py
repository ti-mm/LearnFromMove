from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
RELATIVE_COORDINATE_MIN = 0
RELATIVE_COORDINATE_MAX = 1000
ROTATION_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
)
ROTATION_PRIMITIVE_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up")
RotationActionKind = Literal[
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
]


class ResponseFormatError(ValueError):
    """Raised when a response violates the exact final-SFT wire format."""


class TaskActionError(ValueError):
    """Raised when a format-valid action is illegal for the primitive episode."""


@dataclass(frozen=True)
class CanonicalRotationAction:
    kind: RotationActionKind
    x: int | None = None
    y: int | None = None
    points: tuple[tuple[int, int], ...] = ()

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
class CanonicalRotationResponse:
    thought: str
    action: CanonicalRotationAction
    normalized_text: str
    restored_prefilled_think: bool


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ResponseFormatError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ResponseFormatError(f"non-finite JSON number is forbidden: {value}")


def _decode_exact_json_object(blob: str) -> dict[str, Any]:
    decoder = json.JSONDecoder(
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    try:
        value, end = decoder.raw_decode(blob)
    except ResponseFormatError:
        raise
    except json.JSONDecodeError as exc:
        raise ResponseFormatError("response does not contain one valid JSON object") from exc
    if blob[end:].strip():
        raise ResponseFormatError("trailing content after the JSON action is forbidden")
    if not isinstance(value, dict):
        raise ResponseFormatError("the response JSON value must be an object")
    return value


def _coordinate(value: Any, *, field_name: str) -> int:
    if type(value) is not int:
        raise ResponseFormatError(f"{field_name} must be a JSON integer")
    if not RELATIVE_COORDINATE_MIN <= value <= RELATIVE_COORDINATE_MAX:
        raise ResponseFormatError(
            f"{field_name} must be within "
            f"[{RELATIVE_COORDINATE_MIN}, {RELATIVE_COORDINATE_MAX}]"
        )
    return value


def _parse_action_payload(payload: Mapping[str, Any]) -> CanonicalRotationAction:
    if set(payload) != {"action"}:
        raise ResponseFormatError("top-level JSON must contain exactly the action key")
    raw_action = payload["action"]
    if not isinstance(raw_action, dict):
        raise ResponseFormatError("action must be a JSON object")

    kind = raw_action.get("kind")
    if kind == "move_to":
        if set(raw_action) != {"kind", "x", "y"}:
            raise ResponseFormatError("move_to must contain exactly kind, x, and y")
        return CanonicalRotationAction(
            kind="move_to",
            x=_coordinate(raw_action["x"], field_name="move_to x"),
            y=_coordinate(raw_action["y"], field_name="move_to y"),
        )
    if kind in {"mouse_down", "mouse_up", "left_click"}:
        if set(raw_action) != {"kind"}:
            raise ResponseFormatError(f"{kind} must contain exactly the kind key")
        return CanonicalRotationAction(kind=kind)
    if kind == "drag":
        if set(raw_action) != {"kind", "points"}:
            raise ResponseFormatError("drag must contain exactly kind and points")
        raw_points = raw_action["points"]
        if not isinstance(raw_points, list) or len(raw_points) != 2:
            raise ResponseFormatError("drag must contain exactly two coordinate points")
        points: list[tuple[int, int]] = []
        for index, point in enumerate(raw_points):
            if not isinstance(point, list) or len(point) != 2:
                raise ResponseFormatError(f"drag point {index} must be an [x, y] pair")
            points.append(
                (
                    _coordinate(point[0], field_name=f"drag point {index} x"),
                    _coordinate(point[1], field_name=f"drag point {index} y"),
                )
            )
        return CanonicalRotationAction(kind="drag", points=tuple(points))
    raise ResponseFormatError(f"unsupported action kind: {kind!r}")


def normalize_prefilled_think_response(response: str) -> tuple[str, bool]:
    """Restore only the prompt-owned opening tag used by the final checkpoint."""

    if not isinstance(response, str):
        raise ResponseFormatError("model response must be text")
    stripped = response.strip()
    if stripped.startswith(THINK_OPEN_TAG):
        return stripped, False
    if not stripped:
        raise ResponseFormatError("model response must not be empty")
    if THINK_OPEN_TAG in stripped:
        raise ResponseFormatError("nested or misplaced <think> tags are forbidden")
    if stripped.count(THINK_CLOSE_TAG) != 1:
        raise ResponseFormatError(
            "prefilled-think suffix must contain exactly one </think> tag"
        )
    close_index = stripped.find(THINK_CLOSE_TAG)
    thought = stripped[:close_index]
    remainder = stripped[close_index + len(THINK_CLOSE_TAG) :].strip()
    if not thought.strip():
        raise ResponseFormatError("reasoning inside <think> must be non-empty")
    if THINK_OPEN_TAG in remainder or THINK_CLOSE_TAG in remainder:
        raise ResponseFormatError("think tags after </think> are forbidden")
    if not remainder.startswith("{"):
        raise ResponseFormatError("</think> must be followed by one JSON object")
    _decode_exact_json_object(remainder)
    return f"{THINK_OPEN_TAG}{thought}{THINK_CLOSE_TAG}\n{remainder}", True


def parse_rotation_response(response: str) -> CanonicalRotationResponse:
    """Parse exactly one non-empty think block and one canonical action object."""

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
    if not json_blob:
        raise ResponseFormatError("response is missing the JSON action")
    if not json_blob.startswith("{"):
        raise ResponseFormatError("</think> must be followed by one JSON object")
    if THINK_OPEN_TAG in json_blob or THINK_CLOSE_TAG in json_blob:
        raise ResponseFormatError("JSON remainder contains think tags")
    action = _parse_action_payload(_decode_exact_json_object(json_blob))
    return CanonicalRotationResponse(
        thought=thought,
        action=action,
        normalized_text=normalized,
        restored_prefilled_think=restored,
    )


def canonicalize_data_action(value: Any) -> CanonicalRotationAction:
    """Normalize an Arrow-expanded action while rejecting semantic aliases."""

    if not isinstance(value, Mapping):
        raise ValueError("canonical dataset action must be a mapping")
    compact = {
        str(key): item
        for key, item in value.items()
        if item is not None
    }
    try:
        return _parse_action_payload({"action": compact})
    except ResponseFormatError as exc:
        raise ValueError(f"invalid canonical dataset action: {exc}") from exc


def require_primitive_action(action: CanonicalRotationAction) -> CanonicalRotationAction:
    if action.kind not in ROTATION_PRIMITIVE_ACTION_KINDS:
        raise TaskActionError(
            f"{action.kind} is format-valid but forbidden in the primitive rotation episode"
        )
    return action


def actions_equal(
    left: CanonicalRotationAction,
    right: CanonicalRotationAction,
    *,
    coordinate_tolerance: float = 0.0,
) -> bool:
    if left.kind != right.kind:
        return False
    if left.kind == "move_to":
        if left.x is None or left.y is None or right.x is None or right.y is None:
            return False
        return math.hypot(left.x - right.x, left.y - right.y) <= coordinate_tolerance
    if left.kind == "drag":
        return left.points == right.points
    return True


_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？\n]+")


def thought_has_excessive_repetition(thought: str) -> bool:
    """Detect generic repeated prose without blacklisting any fixed phrase."""

    words = [match.group(0).lower() for match in _WORD_RE.finditer(thought)]
    if len(words) < 24:
        return False

    sentences = [
        " ".join(_WORD_RE.findall(sentence.lower()))
        for sentence in _SENTENCE_SPLIT_RE.split(thought)
    ]
    substantial = [sentence for sentence in sentences if len(sentence.split()) >= 6]
    if len(substantial) >= 2 and len(set(substantial)) < len(substantial):
        return True

    width = 4
    ngrams = [tuple(words[index : index + width]) for index in range(len(words) - width + 1)]
    if len(ngrams) >= 20:
        unique_ratio = len(set(ngrams)) / len(ngrams)
        max_occurrences = max(ngrams.count(ngram) for ngram in set(ngrams))
        if unique_ratio < 0.55 and max_occurrences >= 3:
            return True

    for block_width in range(4, min(20, len(words) // 3) + 1):
        tail = words[-block_width:]
        if words[-2 * block_width : -block_width] != tail:
            continue
        if words[-3 * block_width : -2 * block_width] == tail:
            return True
    return False


def action_sequence_has_loop(actions: Sequence[CanonicalRotationAction]) -> bool:
    """Reject an exact no-op or an exact A-B-A-B action cycle."""

    if len(actions) >= 2 and actions_equal(actions[-1], actions[-2]):
        return True
    if len(actions) >= 4:
        return actions_equal(actions[-1], actions[-3]) and actions_equal(
            actions[-2], actions[-4]
        )
    return False
