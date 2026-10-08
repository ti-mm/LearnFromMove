"""VERL AgentLoop aliases for third-person exploration-depth RL tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ...integrations.verl_browser import VERL_AVAILABLE_V4
from .first_person_online_agent_loop import (
    ExplorationDepthFirstPersonNoThinkAgentLoopV1,
    ExplorationDepthFirstPersonWithThinkAgentLoopV1,
    RUNTIME_TASKS,
    first_person_env_factory,
    normalize_first_person_task_config,
    normalize_first_person_with_think_task_config,
)
from .third_person_online_dataset import AGENT_NAME, TASKS

WITH_THINK_AGENT_NAME = "exploration_depth_third_person_with_think_online_v1"
WITH_THINK_DATA_SOURCE = "exploration_depth_third_person_online_grpo_withthink_v1"

if not set(TASKS).issubset(RUNTIME_TASKS):  # pragma: no cover - import contract
    raise RuntimeError("third-person tasks are missing from the shared AgentLoop runtime")


def normalize_third_person_task_config(
    task_type: Any,
    task_config: Any,
) -> dict[str, Any]:
    normalized = normalize_first_person_task_config(task_type, task_config)
    if normalized["benchmark_variant"] not in TASKS:
        raise ValueError("third-person task_config selects a non-third-person task")
    return normalized


def normalize_third_person_with_think_task_config(
    task_type: Any,
    task_config: Any,
) -> dict[str, Any]:
    normalized = normalize_first_person_with_think_task_config(task_type, task_config)
    if normalized["benchmark_variant"] not in TASKS:
        raise ValueError("third-person task_config selects a non-third-person task")
    return normalized


def third_person_env_factory(
    *,
    task: str,
    manifest_path: Path,
    episode_id: str,
) -> Any:
    if task not in TASKS:
        raise ValueError(f"unsupported third-person task: {task!r}")
    return first_person_env_factory(
        task=task,
        manifest_path=manifest_path,
        episode_id=episode_id,
    )


if not VERL_AVAILABLE_V4:

    class ExplorationDepthThirdPersonNoThinkAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("ExplorationDepthThirdPersonNoThinkAgentLoopV1 requires VERL")

    class ExplorationDepthThirdPersonWithThinkAgentLoopV1:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("ExplorationDepthThirdPersonWithThinkAgentLoopV1 requires VERL")

else:
    from verl.experimental.agent_loop.agent_loop import register

    @register(AGENT_NAME)
    class ExplorationDepthThirdPersonNoThinkAgentLoopV1(
        ExplorationDepthFirstPersonNoThinkAgentLoopV1
    ):
        """No-Think six-action loop for third-person ten-choice and drag."""

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_third_person_task_config(task_type, task_config)

    @register(WITH_THINK_AGENT_NAME)
    class ExplorationDepthThirdPersonWithThinkAgentLoopV1(
        ExplorationDepthFirstPersonWithThinkAgentLoopV1
    ):
        """Visible-Think six-action loop for third-person ten-choice and drag."""

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            return normalize_third_person_with_think_task_config(
                task_type,
                task_config,
            )
