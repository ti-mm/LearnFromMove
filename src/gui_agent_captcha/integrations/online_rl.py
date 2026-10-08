from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProtocolViolationV4(ValueError):
    """A format-valid primitive action is illegal in the current mouse state."""


class PolicyActionViolationV4(ValueError):
    """A format-valid action is outside the primitive online task policy."""


class LoopViolationV4(ValueError):
    """The sampled response repeats an action or reasoning loop."""


class NoProgressViolationV4(ValueError):
    """The public screenshot and protocol state made no progress."""


class ActionBudgetViolationV4(ValueError):
    """The policy requested another action after exhausting the budget."""


class V4TrajectoryOutcome(str, Enum):
    SUCCESS = "success"
    BROWSER_FAILURE = "browser_failure"
    PARSE_ERROR = "parse_error"
    PROTOCOL_ERROR = "protocol_error"
    POLICY_ACTION_ERROR = "policy_action_error"
    LOOP = "loop"
    NO_PROGRESS = "no_progress"
    MAX_STEPS = "max_steps"
    INFRA_ERROR = "infra_error"
    GENERATION_LENGTH = "generation_length"
    CONTEXT_BUDGET = "context_budget"


@dataclass(frozen=True)
class OnlineRewardBreakdownV4:
    reward: float | None
    category: str
    policy_valid: bool
    format_valid: bool | None
    browser_success: bool
    terminal_source: str
    components: dict[str, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "category": self.category,
            "policy_valid": self.policy_valid,
            "format_valid": self.format_valid,
            "browser_success": self.browser_success,
            "terminal_source": self.terminal_source,
            "components": dict(self.components),
        }

    def to_verl_reward_extra_info(self) -> dict[str, Any]:
        """Return scalar-only validation metrics accepted by verl."""

        def scalar(value: Any) -> Any:
            return "none" if value is None else value

        payload = {
            "reward": scalar(self.reward),
            "category": self.category,
            "policy_valid": self.policy_valid,
            "format_valid": scalar(self.format_valid),
            "browser_success": self.browser_success,
            "terminal_source": self.terminal_source,
        }
        payload.update(
            {
                f"component_{name}": scalar(value)
                for name, value in self.components.items()
            }
        )
        return payload


OUTCOME_SUCCESS_REWARD_V4 = 1.0
POLICY_FAILURE_REWARD_V4 = 0.0
TRAJECTORY_FORMAT_REWARD_V4 = 0.1


_OUTCOME_CATEGORIES: dict[V4TrajectoryOutcome, tuple[str, bool]] = {
    V4TrajectoryOutcome.SUCCESS: ("browser_success", True),
    V4TrajectoryOutcome.BROWSER_FAILURE: ("browser_failure", True),
    V4TrajectoryOutcome.PARSE_ERROR: ("parse_error", True),
    V4TrajectoryOutcome.PROTOCOL_ERROR: ("protocol_error", True),
    V4TrajectoryOutcome.POLICY_ACTION_ERROR: ("policy_action_error", True),
    V4TrajectoryOutcome.LOOP: ("loop", True),
    V4TrajectoryOutcome.NO_PROGRESS: ("no_progress", True),
    V4TrajectoryOutcome.MAX_STEPS: ("max_steps", True),
    V4TrajectoryOutcome.INFRA_ERROR: ("infrastructure_invalid", False),
    V4TrajectoryOutcome.GENERATION_LENGTH: ("generation_length", True),
    V4TrajectoryOutcome.CONTEXT_BUDGET: ("context_budget", True),
}


def reward_breakdown_for_outcome(
    outcome: V4TrajectoryOutcome,
    *,
    terminal_source: str = "none",
    format_valid: bool | None = None,
) -> OnlineRewardBreakdownV4:
    category, policy_valid = _OUTCOME_CATEGORIES[outcome]
    if outcome is V4TrajectoryOutcome.INFRA_ERROR:
        return OnlineRewardBreakdownV4(
            reward=None,
            category=category,
            policy_valid=policy_valid,
            format_valid=None,
            browser_success=False,
            terminal_source=terminal_source,
            components={
                "outcome_success": None,
                "policy_failure": None,
                "trajectory_format": None,
            },
        )

    if format_valid is None:
        format_valid = outcome is not V4TrajectoryOutcome.PARSE_ERROR
    is_success = outcome is V4TrajectoryOutcome.SUCCESS
    outcome_reward = OUTCOME_SUCCESS_REWARD_V4 if is_success else 0.0
    failure_reward = POLICY_FAILURE_REWARD_V4
    format_reward = TRAJECTORY_FORMAT_REWARD_V4 if format_valid else 0.0
    return OnlineRewardBreakdownV4(
        reward=outcome_reward + failure_reward + format_reward,
        category=category,
        policy_valid=policy_valid,
        format_valid=format_valid,
        browser_success=outcome is V4TrajectoryOutcome.SUCCESS,
        terminal_source=terminal_source,
        components={
            "outcome_success": outcome_reward,
            "policy_failure": failure_reward,
            "trajectory_format": format_reward,
        },
    )
