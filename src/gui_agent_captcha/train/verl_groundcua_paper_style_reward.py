"""Fail closed if VERL attempts to score a GroundCUA AgentLoop trajectory offline."""

from __future__ import annotations

from typing import Any


def compute_score(*args: Any, **kwargs: Any) -> float:
    """GroundCUA terminal rewards are owned by the static VERL AgentLoop.

    The AgentLoop emits only `-0.2`, `0.0`, or `1.0` after the complete strict
    trajectory has terminated. Calling an offline hook would lose trajectory state
    and reopen the reward-hacking paths this release intentionally excludes.
    """

    del args, kwargs
    raise RuntimeError(
        "offline reward scoring is forbidden for GroundCUA paper-style VERL RL; "
        "the AgentLoop must supply the terminal reward"
    )
