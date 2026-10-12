from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from gui_agent_captcha.domains.rotation.action_contract import (
    ROTATION_ACTION_KINDS,
    ROTATION_PRIMITIVE_ACTION_KINDS,
    ResponseFormatError,
)
from gui_agent_captcha.domains.rotation.agent_loop import (
    default_browser_slot_pool,
)
from gui_agent_captcha.domains.rotation.outcome_protocol import (
    OnlineProtocolStateV4,
    trajectory_format_valid_v4,
)
from gui_agent_captcha.integrations.browser_runtime import (
    BrowserSlotPool,
)
from gui_agent_captcha.integrations.browser_trajectory import (
    ONLINE_V4_CONTEXT_WINDOW,
    ONLINE_V4_IMAGE_HISTORY_MAX,
    ONLINE_V4_VIEWPORT,
    ContextBudgetErrorV4,
    EnvFactoryV4,
    PublicBrowserObservationV4,
    TurnGeneratorV4,
)
from gui_agent_captcha.integrations.browser_trajectory import (
    BrowserTrajectoryRunnerV4 as DomainNeutralBrowserTrajectoryRunnerV4,
)
from gui_agent_captcha.integrations.verl_browser import (
    VERL_AVAILABLE_V4,
    VerlBrowserAgentLoopV4,
)
from gui_agent_captcha.protocol_tracks import ROTATION_TASK_REQUIREMENT

from .qwen3_vl_sft import (
    FROZEN_FINAL_SFT_RUNTIME_SELF_HISTORY_CONTRACT,
    MOUSE_CAPTCHA_ACTION_KINDS,
    QWEN3_RELATIVE_COORDINATE_FORMAT,
    SftPromptBuildResult,
    build_prompt_messages_from_history,
)

ONLINE_V4_NAVIGATION_TIMEOUT_MS = 60_000


def build_runtime_self_history_prompt_v4(
    *,
    observations: Sequence[PublicBrowserObservationV4],
    state: OnlineProtocolStateV4,
    max_steps: int,
) -> SftPromptBuildResult:
    if not observations:
        raise ValueError("a live prompt requires at least one public observation")
    if len(observations) != len(state.action_history) + 1:
        raise ValueError("live observations must contain one state after every action")
    current = observations[-1]
    if current.size_px != ONLINE_V4_VIEWPORT:
        raise ValueError("V4 live prompt requires an exact 1280x720 observation")
    build = build_prompt_messages_from_history(
        instruction=ROTATION_TASK_REQUIREMENT,
        image_path=Path(current.screenshot_path),
        image_paths=tuple(Path(item.screenshot_path) for item in observations),
        action_history=tuple(action.to_dict() for action in state.action_history),
        assistant_response_history=state.response_history,
        total_actions=max_steps,
        cursor_xy=state.cursor,
        button_state="down" if state.mouse_down else "up",
        task_type="rotation_captcha",
        allowed_kinds=MOUSE_CAPTCHA_ACTION_KINDS,
        budget_remaining=max(0, max_steps - state.step_count),
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
        image_size_px=ONLINE_V4_VIEWPORT,
        images_to_keep=ONLINE_V4_IMAGE_HISTORY_MAX,
        frozen_final_sft_runtime=True,
    )
    if any(message.get("role") == "system" for message in build.messages):
        raise RuntimeError("V4 live rotation prompt must not contain a system message")
    if [message.get("role") for message in build.messages] != ["user"]:
        raise RuntimeError("V4 live rotation prompt must remain one multimodal user turn")
    return replace(
        build,
        prompt_contract=FROZEN_FINAL_SFT_RUNTIME_SELF_HISTORY_CONTRACT,
    )


class BrowserTrajectoryRunnerV4(DomainNeutralBrowserTrajectoryRunnerV4):
    """Rotation defaults for the domain-neutral browser trajectory runner."""

    def __init__(
        self,
        *,
        env_factory: EnvFactoryV4,
        turn_generator: TurnGeneratorV4,
        artifact_root: Path,
        slot_pool: BrowserSlotPool | None = None,
        max_steps: int = 8,
        max_infra_retries: int = 2,
        id_factory: Callable[[], str] | None = None,
        task_type: str = "rotation_captcha",
        prompt_builder: Callable[..., SftPromptBuildResult] = (
            build_runtime_self_history_prompt_v4
        ),
        protocol_state_factory: Callable[[int], Any] | None = None,
        trajectory_format_validator: Callable[[Sequence[str]], bool] = (
            trajectory_format_valid_v4
        ),
    ) -> None:
        super().__init__(
            env_factory=env_factory,
            turn_generator=turn_generator,
            artifact_root=artifact_root,
            slot_pool=slot_pool or default_browser_slot_pool(),
            task_type=task_type,
            prompt_builder=prompt_builder,
            protocol_state_factory=protocol_state_factory
            or (lambda max_steps: OnlineProtocolStateV4(max_steps=max_steps)),
            trajectory_format_validator=trajectory_format_validator,
            response_format_errors=(ResponseFormatError,),
            max_steps=max_steps,
            max_infra_retries=max_infra_retries,
            id_factory=id_factory,
            viewport=ONLINE_V4_VIEWPORT,
            non_retryable_errors=(ContextBudgetErrorV4,),
            thread_name_prefix="interaction-v4",
        )


def _default_interaction_env_factory_v4(seed: int, artifact_dir: Path) -> Any:
    from gui_agent_captcha.domains.rotation.environment import InteractionCaptchaEnv

    return InteractionCaptchaEnv(
        base_url=os.environ.get("INTERACTION_CAPTCHA_URL", "http://localhost:4321/"),
        artifact_dir=artifact_dir,
        challenge_seed=seed,
        enable_playwright=True,
        headless=os.environ.get("INTERACTION_BROWSER_HEADLESS", "1") != "0",
        viewport_width=ONLINE_V4_VIEWPORT[0],
        viewport_height=ONLINE_V4_VIEWPORT[1],
        navigate_on_launch=False,
        navigation_timeout_ms=int(
            os.environ.get(
                "INTERACTION_NAVIGATION_TIMEOUT_MS",
                str(ONLINE_V4_NAVIGATION_TIMEOUT_MS),
            )
        ),
    )


if not VERL_AVAILABLE_V4:
    from gui_agent_captcha.integrations.verl_browser import VERL_IMPORT_ERROR_V4

    class InteractionBrowserAgentLoopV4:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "InteractionBrowserAgentLoopV4 requires the project VERL environment"
            ) from VERL_IMPORT_ERROR_V4

else:
    from verl.experimental.agent_loop.agent_loop import register

    @register("interaction_browser_online_v4")
    class InteractionBrowserAgentLoopV4(VerlBrowserAgentLoopV4):
        """Rotation VERL adapter for exact self-history browser trajectories."""

        def __init__(
            self,
            *args: Any,
            env_factory: EnvFactoryV4 | None = None,
            artifact_root: str | Path | None = None,
            slot_pool: BrowserSlotPool | None = None,
            id_factory: Callable[[], str] | None = None,
            max_steps: int = 8,
            max_infra_retries: int = 2,
            context_window: int = ONLINE_V4_CONTEXT_WINDOW,
            generation_safety_reserve_tokens: int = 32,
            **kwargs: Any,
        ) -> None:
            super().__init__(
                *args,
                env_factory=env_factory or _default_interaction_env_factory_v4,
                artifact_root=(
                    artifact_root
                    or os.environ.get(
                        "INTERACTION_ONLINE_V4_ARTIFACT_ROOT",
                        "artifacts/runs/interaction_browser_online_grpo_v4",
                    )
                ),
                slot_pool=slot_pool or default_browser_slot_pool(),
                prompt_contract=FROZEN_FINAL_SFT_RUNTIME_SELF_HISTORY_CONTRACT,
                id_factory=id_factory,
                max_steps=max_steps,
                max_infra_retries=max_infra_retries,
                context_window=context_window,
                generation_safety_reserve_tokens=generation_safety_reserve_tokens,
                **kwargs,
            )

        @staticmethod
        def _normalize_task_config(
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            if task_type not in {None, "rotation_captcha"}:
                raise ValueError("V4 AgentLoop only supports rotation_captcha")
            if hasattr(task_config, "item") and not isinstance(task_config, Mapping):
                task_config = task_config.item()
            if not isinstance(task_config, Mapping):
                raise ValueError("online V4 row is missing task_config")
            normalized = dict(task_config)
            required = {
                "background_pool",
                "viewport",
                "max_steps",
                "coordinate_format",
                "format_action_kinds",
                "executable_action_kinds",
            }
            if set(normalized) != required:
                raise ValueError(
                    f"task_config keys must be exactly {sorted(required)!r}"
                )
            if list(normalized["viewport"]) != list(ONLINE_V4_VIEWPORT):
                raise ValueError("online V4 viewport must be exactly 1280x720")
            if normalized["coordinate_format"] != "qwen_relative_0_1000":
                raise ValueError("unexpected online V4 coordinate format")
            if tuple(normalized["format_action_kinds"]) != ROTATION_ACTION_KINDS:
                raise ValueError(
                    "online V4 format grammar must expose move_to, mouse_down, "
                    "mouse_up, left_click, and drag"
                )
            if tuple(normalized["executable_action_kinds"]) != (
                *ROTATION_PRIMITIVE_ACTION_KINDS,
            ):
                raise ValueError("online V4 execution permits only primitive actions")
            return normalized

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            del task_config
            return BrowserTrajectoryRunnerV4(
                env_factory=self.env_factory,
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
            )
