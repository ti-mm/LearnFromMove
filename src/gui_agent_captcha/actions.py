from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

Point: TypeAlias = tuple[float, float]


@dataclass(frozen=True)
class PrimitiveAction:
    kind: Literal[
        "move_to",
        "mouse_down",
        "mouse_up",
        "left_click",
        "left_double",
        "right_single",
        "done",
    ]
    x: float | None = None
    y: float | None = None
    # Optional structured answer carried by `done` for non-coordinate tasks
    # (e.g. OCW number/multiselect types). Ignored for all other kinds.
    answer: object = None

    def __post_init__(self) -> None:
        if self.kind == "move_to" and (self.x is None or self.y is None):
            raise ValueError("move_to requires both x and y coordinates")
        if self.kind != "move_to" and (self.x is not None or self.y is not None):
            raise ValueError(f"{self.kind} does not accept coordinates")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "x": self.x, "y": self.y}
        if self.kind == "done" and self.answer is not None:
            d["answer"] = self.answer
        return d


@dataclass(frozen=True)
class AtomicAction:
    kind: Literal["click", "drag", "submit"]
    points: list[Point] = field(default_factory=list)
    # Optional structured answer for non-coordinate tasks (e.g. OCW index/count types).
    # Not validated; defaults to None (backward compatible).
    answer: object = None

    def __post_init__(self) -> None:
        if self.kind == "drag":
            if len(self.points) < 2:
                raise ValueError(
                    f"drag requires at least 2 point(s), got {len(self.points)}",
                )
            return
        expected = {"click": 1, "submit": 0}[self.kind]
        if len(self.points) != expected:
            raise ValueError(
                f"{self.kind} requires {expected} point(s), got {len(self.points)}",
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "kind": self.kind,
            "points": list(self.points),
        }
        if self.answer is not None:
            d["answer"] = self.answer
        return d


@dataclass(frozen=True)
class WebAction:
    kind: Literal[
        "left_double",
        "right_single",
        "scroll",
        "type",
        "hotkey",
        "wait",
        "finished",
        "launch",
        "call_user",
        "long_press",
        "press_back",
        "press_home",
        "press_enter",
        "press_recent",
    ]
    points: list[Point] = field(default_factory=list)
    direction: str | None = None
    content: str | None = None
    key: str | None = None
    app: str | None = None
    url: str | None = None
    duration_s: float | None = None

    def __post_init__(self) -> None:
        expected_points = {
            "left_double": 1,
            "right_single": 1,
            "scroll": None,
            "type": 0,
            "hotkey": 0,
            "wait": 0,
            "finished": 0,
            "launch": 0,
            "call_user": 0,
            "long_press": 1,
            "press_back": 0,
            "press_home": 0,
            "press_enter": 0,
            "press_recent": 0,
        }[self.kind]
        if expected_points is not None and expected_points != len(self.points):
            raise ValueError(
                f"{self.kind} requires {expected_points} point(s), got {len(self.points)}",
            )
        if self.kind == "scroll" and not self.direction:
            raise ValueError("scroll requires direction")
        if self.kind == "type" and self.content is None:
            raise ValueError("type requires content")
        if self.kind == "hotkey" and self.key is None:
            raise ValueError("hotkey requires key")
        if self.kind == "finished" and self.content is None:
            raise ValueError("finished requires content")
        if self.kind == "call_user" and self.content is None:
            raise ValueError("call_user requires content")
        if self.kind == "launch" and self.app is None and self.url is None:
            raise ValueError("launch requires app or url")
        if self.duration_s is not None:
            if self.kind != "wait":
                raise ValueError(f"{self.kind} does not accept duration_s")
            if (
                isinstance(self.duration_s, bool)
                or not isinstance(self.duration_s, (int, float))
                or not math.isfinite(float(self.duration_s))
                or self.duration_s < 0
            ):
                raise ValueError("wait duration_s must be a finite non-negative number")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind, "points": list(self.points)}
        if self.direction is not None:
            d["direction"] = self.direction
        if self.content is not None:
            d["content"] = self.content
        if self.key is not None:
            d["key"] = self.key
        if self.app is not None:
            d["app"] = self.app
        if self.url is not None:
            d["url"] = self.url
        if self.duration_s is not None:
            d["duration_s"] = self.duration_s
        return d


Action: TypeAlias = PrimitiveAction | AtomicAction | WebAction


@dataclass
class ActionResult:
    action: Action
    success: bool
    metadata: dict[str, Any] = field(default_factory=dict)
