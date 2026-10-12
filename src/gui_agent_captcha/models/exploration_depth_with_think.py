from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..benchmarks.exploration_depth.training_no_think import (
    IMAGE_HISTORY_MAX,
    SIX_ACTION_KINDS,
)
from ..benchmarks.exploration_depth.training_with_think import (
    PROMPT_CONTRACT,
    RESPONSE_CONTRACT,
    _think_action_prompt,
)
from ..core import Observation, StepResult
from ..train.qwen3_vl_sft import (
    SftPromptBuildResult,
    build_action_context_from_history,
    fold_image_history_paths,
)
from .exploration_depth_no_think import _exact_assistant_responses
from .qwen3_vl_sft_local import Qwen3VLSftLocalBackend


EXPLORATION_DEPTH_WITH_THINK_EVAL_PROMPT_CONTRACT = PROMPT_CONTRACT
EXPLORATION_DEPTH_WITH_THINK_RESPONSE_CONTRACT = RESPONSE_CONTRACT


def build_exploration_depth_with_think_eval_prompt(
    *,
    obs: Observation,
    history: list[StepResult],
    image_history_paths: list[Path],
    images_to_keep: int = IMAGE_HISTORY_MAX,
    budget: int | None,
) -> SftPromptBuildResult:
    action_history = tuple(
        step.action.to_dict() for step in history if step.action is not None
    )
    response_history = _exact_assistant_responses(history)
    if len(action_history) != len(response_history):
        raise ValueError("action and exact assistant-response history lengths differ")
    if len(image_history_paths) != len(action_history) + 1:
        raise ValueError(
            "runtime image history must contain one pre-action observation per "
            f"action plus the current observation; images={len(image_history_paths)} "
            f"actions={len(action_history)}"
        )

    retained_paths = fold_image_history_paths(
        tuple(image_history_paths),
        images_to_keep=images_to_keep,
    )
    first_retained_index = len(image_history_paths) - len(retained_paths)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _think_action_prompt(obs.instruction)}
    ]

    def append_previous_response(action_index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {action_index + 1} assistant response "
                    f"(context only):\n{response_history[action_index]}\n"
                ),
            }
        )

    for action_index in range(first_retained_index):
        append_previous_response(action_index)
    for retained_offset, path in enumerate(retained_paths):
        content.append({"type": "image", "image": str(path)})
        action_index = first_retained_index + retained_offset
        if action_index < len(action_history):
            append_previous_response(action_index)
    action_index = len(action_history)
    history_text = (
        f"{action_index} previous assistant response(s) are included in "
        "chronological order."
        if action_index
        else "No previous actions are present in this trajectory."
    )
    content.append(
        {
            "type": "text",
            "text": "\n".join(
                (
                    history_text,
                    "The last attached image is the current observation.",
                    f"Return the Think block and JSON action for step {action_index + 1}.",
                )
            ),
        }
    )

    context = build_action_context_from_history(
        action_history=action_history,
        total_actions=budget,
        cursor_xy=obs.cursor_xy,
        button_state="up",
        task_type=str((obs.metadata or {}).get("family") or "") or None,
        allowed_kinds=SIX_ACTION_KINDS,
        budget_remaining=(
            max(0, budget - len(action_history)) if budget is not None else None
        ),
    )
    return SftPromptBuildResult(
        messages=[{"role": "user", "content": content}],
        image_paths=retained_paths,
        context=context,
        prompt_contract=EXPLORATION_DEPTH_WITH_THINK_EVAL_PROMPT_CONTRACT,
    )


@dataclass
class Qwen35ExplorationDepthWithThinkLocalBackend(Qwen3VLSftLocalBackend):
    """Local backend matching the six-variant visible-Think training prompt."""

    enable_thinking: bool = True

    def _build_prompt(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        image_history_paths: list[Path],
        allowed_kinds: Iterable[str],
        budget: int | None,
    ) -> SftPromptBuildResult:
        allowed = tuple(allowed_kinds)
        if allowed != SIX_ACTION_KINDS:
            raise ValueError(
                "exploration-depth evaluation requires the common six-action space; "
                f"got {allowed}"
            )
        return build_exploration_depth_with_think_eval_prompt(
            obs=obs,
            history=history,
            image_history_paths=image_history_paths,
            images_to_keep=self.image_history_max,
            budget=budget,
        )
