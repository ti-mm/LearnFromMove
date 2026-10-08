from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from gui_agent_captcha.integrations.browser_runtime import (
    BrowserSlotPool,
    ensure_verl_processor_rope_binding,
)
from gui_agent_captcha.integrations.browser_trajectory import (
    ONLINE_V4_CONTEXT_WINDOW,
    ONLINE_V4_IMAGE_MAX_PIXELS,
    ONLINE_V4_VIEWPORT,
    BrowserTrajectoryResultV4,
    BrowserTrajectoryRunnerV4,
    ContextBudgetErrorV4,
    DynamicGenerationBudgetV4,
    EnvFactoryV4,
    GeneratedTurnV4,
    OnlineInfrastructureErrorV4,
    RuntimeGenerationRequestV4,
    TurnGeneratorV4,
)


def policy_sample_fingerprint_v4(
    *,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    image_fingerprints: Sequence[str],
    mm_processor_kwargs: Mapping[str, Any],
    prompt_contract: str,
) -> str:
    payload = {
        "prompt_ids": list(prompt_ids),
        "response_ids": list(response_ids),
        "image_fingerprints": list(image_fingerprints),
        "mm_processor_kwargs": dict(mm_processor_kwargs),
        "prompt_contract": prompt_contract,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _EncodedRuntimePromptV4:
    prompt_ids: list[int]
    images: list[Image.Image]
    messages: list[dict[str, Any]]


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
    from verl.utils.chat_template import apply_chat_template
    from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
    from verl.workers.rollout.replica import TokenOutput
except ModuleNotFoundError as exc:
    VERL_AVAILABLE_V4 = False
    VERL_IMPORT_ERROR_V4 = exc

    class VerlBrowserAgentLoopV4:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "VerlBrowserAgentLoopV4 requires the project VERL environment"
            ) from VERL_IMPORT_ERROR_V4

else:
    VERL_AVAILABLE_V4 = True

    class VerlBrowserAgentLoopV4(AgentLoopBase):
        """Domain-neutral VERL adapter for browser trajectory step samples."""

        def __init__(
            self,
            *args: Any,
            env_factory: EnvFactoryV4,
            artifact_root: str | Path,
            slot_pool: BrowserSlotPool,
            prompt_contract: str,
            id_factory: Callable[[], str] | None = None,
            max_steps: int = 8,
            max_infra_retries: int = 2,
            context_window: int = ONLINE_V4_CONTEXT_WINDOW,
            generation_safety_reserve_tokens: int = 32,
            max_generation_tokens: int | None = None,
            prompt_owned_think_opening: bool = True,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if self.processor is None:
                raise ValueError("V4 online GRPO requires a multimodal processor")
            ensure_verl_processor_rope_binding(self.processor)
            self.max_steps = int(max_steps)
            self.max_infra_retries = int(max_infra_retries)
            self.context_window = int(context_window)
            if self.context_window != ONLINE_V4_CONTEXT_WINDOW:
                raise ValueError("V4 online GRPO context window must be exactly 16384")
            self.max_generation_tokens = max_generation_tokens
            self.generation_budget = DynamicGenerationBudgetV4(
                context_window=self.context_window,
                response_capacity=int(self.rollout_config.response_length),
                safety_reserve_tokens=int(generation_safety_reserve_tokens),
                max_generation_tokens=max_generation_tokens,
            )
            self.prompt_length = int(self.rollout_config.prompt_length)
            self.response_length = int(self.rollout_config.response_length)
            self.max_model_len = int(self.rollout_config.max_model_len)
            if self.prompt_length + self.response_length != self.context_window:
                raise ValueError("VERL prompt_length + response_length must equal 16384")
            if self.max_model_len != self.context_window:
                raise ValueError("rollout.max_model_len must equal 16384")
            self.env_factory = env_factory
            self.artifact_root = Path(artifact_root)
            self.slot_pool = slot_pool
            self.id_factory = id_factory or (lambda: uuid4().hex)
            self.prompt_contract = prompt_contract
            self.prompt_owned_think_opening = bool(prompt_owned_think_opening)
            self.last_trajectory_result: BrowserTrajectoryResultV4 | None = None

        def _load_prompt_images(
            self,
            image_paths: Sequence[Path],
        ) -> list[Image.Image]:
            images: list[Image.Image] = []
            for path in image_paths:
                with Image.open(path) as source:
                    image = source.convert("RGB").copy()
                if image.size != ONLINE_V4_VIEWPORT:
                    raise OnlineInfrastructureErrorV4(
                        f"V4 prompt image must be 1280x720, got {image.size!r}",
                        attempts=1,
                    )
                images.append(image)
            return images

        def _encode_runtime_prompt(
            self,
            prompt: Any,
        ) -> _EncodedRuntimePromptV4:
            images = self._load_prompt_images(prompt.image_paths)
            raw_prompt = apply_chat_template(
                self.processor,
                prompt.messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.apply_chat_template_kwargs,
            )
            mm_kwargs = dict(self.mm_processor_kwargs or {})
            mm_kwargs["truncation"] = False
            mm_kwargs["max_pixels"] = ONLINE_V4_IMAGE_MAX_PIXELS
            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[raw_prompt],
                images=images,
                mm_processor_kwargs=mm_kwargs,
            )
            input_ids = (
                model_inputs["input_ids"]
                if isinstance(model_inputs, Mapping)
                else model_inputs.input_ids
            )
            return _EncodedRuntimePromptV4(
                prompt_ids=list(normalize_token_ids(input_ids)),
                images=images,
                messages=prompt.messages,
            )

        def _normalize_task_config(
            self,
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            raise NotImplementedError

        def _make_trajectory_runner(
            self,
            *,
            generate_turn: TurnGeneratorV4,
            task_config: Mapping[str, Any],
            effective_max_steps: int,
        ) -> BrowserTrajectoryRunnerV4:
            raise NotImplementedError

        async def run(
            self,
            sampling_params: dict[str, Any],
            **kwargs: Any,
        ) -> list[AgentLoopOutput]:
            seed_value = kwargs.get("seed")
            if hasattr(seed_value, "item"):
                seed_value = seed_value.item()
            if not isinstance(seed_value, int) or isinstance(seed_value, bool):
                raise ValueError(f"online V4 row has invalid seed: {seed_value!r}")
            seed = int(seed_value)
            task_config = self._normalize_task_config(
                kwargs.get("task_type"),
                kwargs.get("task_config"),
            )
            effective_max_steps = int(task_config["max_steps"])
            if not 0 < effective_max_steps <= self.max_steps:
                raise ValueError("task max_steps exceeds the V4 AgentLoop limit")

            generate_seconds = 0.0
            num_preempted = 0

            async def generate_turn(
                request: RuntimeGenerationRequestV4,
                base_sampling: dict[str, Any],
            ) -> GeneratedTurnV4:
                nonlocal generate_seconds
                nonlocal num_preempted
                encoded = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._encode_runtime_prompt,
                    request.prompt,
                )
                if len(encoded.prompt_ids) > self.prompt_length:
                    raise ContextBudgetErrorV4(
                        "actual V4 prompt exceeds VERL prompt_length; truncation is forbidden: "
                        f"{len(encoded.prompt_ids)} > {self.prompt_length}"
                    )
                per_turn = self.generation_budget.sampling_params(
                    base_sampling,
                    prompt_token_count=len(encoded.prompt_ids),
                    accumulated_response_token_count=(
                        request.accumulated_response_token_count
                    ),
                )
                generation_budget = int(per_turn["max_tokens"])
                started = time.perf_counter()
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request.request_id,
                    prompt_ids=encoded.prompt_ids,
                    sampling_params=per_turn,
                    image_data=encoded.images,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs={
                        "truncation": False,
                        "max_pixels": ONLINE_V4_IMAGE_MAX_PIXELS,
                    },
                )
                generate_seconds += time.perf_counter() - started
                num_preempted += int(output.num_preempted or 0)
                token_ids = list(output.token_ids)
                if len(token_ids) > generation_budget:
                    raise OnlineInfrastructureErrorV4(
                        "V4 rollout server exceeded the requested generation budget: "
                        f"{len(token_ids)} > {generation_budget}",
                        attempts=1,
                    )
                logprobs = (
                    [float(value) for value in output.log_probs]
                    if output.log_probs is not None
                    else None
                )
                if logprobs is None:
                    raise OnlineInfrastructureErrorV4(
                        "V4 rollout server did not return token logprobs",
                        attempts=1,
                    )
                raw_text = self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                )
                if (
                    request.accumulated_response_token_count + len(token_ids)
                    > self.response_length
                ):
                    raise ContextBudgetErrorV4(
                        "accumulated V4 PPO response exceeds response_length; "
                        "truncation is forbidden"
                    )
                return GeneratedTurnV4(
                    text=raw_text,
                    token_ids=token_ids,
                    logprobs=logprobs,
                    extra_fields={
                        **dict(output.extra_fields),
                        "prompt_owned_think_opening": self.prompt_owned_think_opening,
                        "prompt_ids": list(encoded.prompt_ids),
                        "prompt_images": list(encoded.images),
                        "prompt_messages": encoded.messages,
                        "prompt_token_count": len(encoded.prompt_ids),
                        "generation_budget": generation_budget,
                        "image_fingerprints": [
                            item.screenshot_fingerprint
                            for item in request.observations[-len(encoded.images) :]
                        ],
                    },
                )

            runner = self._make_trajectory_runner(
                generate_turn=generate_turn,
                task_config=task_config,
                effective_max_steps=effective_max_steps,
            )
            runner.terminate_on_generation_limit = self.max_generation_tokens is not None
            result = await runner.run(seed=seed, sampling_params=sampling_params)
            self.last_trajectory_result = result
            if not result.turns:
                raise OnlineInfrastructureErrorV4(
                    "V4 trajectory ended before a prompt was encoded",
                    attempts=result.infra_retry_count + 1,
                )
            reward = result.reward_breakdown.reward
            if reward is None:
                raise OnlineInfrastructureErrorV4(
                    "infrastructure-invalid result cannot enter PPO",
                    attempts=result.infra_retry_count + 1,
                )
            step_count = len(result.turns)
            step_outputs: list[AgentLoopOutput] = []
            for step_index, turn in enumerate(result.turns):
                prompt_ids = list(turn.extra_fields["prompt_ids"])
                prompt_images = list(turn.extra_fields["prompt_images"])
                image_fingerprints = list(turn.extra_fields["image_fingerprints"])
                if len(prompt_ids) + len(turn.token_ids) > self.context_window:
                    raise ContextBudgetErrorV4(
                        "V4 step sample exceeds the 16384 context budget"
                    )
                mm_kwargs = {
                    "truncation": False,
                    "max_pixels": ONLINE_V4_IMAGE_MAX_PIXELS,
                }
                sample_fingerprint = policy_sample_fingerprint_v4(
                    prompt_ids=prompt_ids,
                    response_ids=turn.token_ids,
                    image_fingerprints=image_fingerprints,
                    mm_processor_kwargs=mm_kwargs,
                    prompt_contract=self.prompt_contract,
                )
                extra_fields = {
                    key: value
                    for key, value in turn.extra_fields.items()
                    if key not in {
                        "prompt_ids",
                        "prompt_images",
                        "image_fingerprints",
                    }
                }
                extra_fields.update(
                    {
                        "step_sample_contract": "independent_browser_step_samples_v1",
                        "fingerprint_policy": "rollout_actor_ref_fail_closed_v1",
                        "processor_fingerprint": sample_fingerprint,
                        "ppo_fingerprint": sample_fingerprint,
                        "image_fingerprints": image_fingerprints,
                        "trajectory_id": result.trajectory_id,
                        "request_id": result.request_id,
                        "seed": seed,
                        "task_config": task_config,
                        "step_index": step_index,
                        "step_count": step_count,
                        "trajectory_loss_weight": 1.0 / step_count,
                        "trajectory_outcome": result.outcome.value,
                        "terminal_source": result.terminal_source,
                        "reward_breakdown": result.reward_breakdown.to_dict(),
                        "reward_extra_info": (
                            result.reward_breakdown.to_verl_reward_extra_info()
                        ),
                        "artifact_dir": str(result.artifact_dir),
                        "response": turn.text,
                        "action": (
                            result.steps[step_index].action
                            if step_index < len(result.steps)
                            else None
                        ),
                        "screenshot_paths": [
                            observation.screenshot_path
                            for observation in result.observations
                        ],
                        "screenshot_fingerprints": [
                            observation.screenshot_fingerprint
                            for observation in result.observations
                        ],
                        "prompt_contract": self.prompt_contract,
                        "turn_scores": [],
                        "tool_rewards": [],
                    }
                )
                step_outputs.append(
                    AgentLoopOutput(
                        prompt_ids=prompt_ids,
                        response_ids=list(turn.token_ids),
                        response_mask=[1] * len(turn.token_ids),
                        response_logprobs=list(turn.logprobs or ()),
                        multi_modal_data={"images": prompt_images},
                        mm_processor_kwargs=mm_kwargs,
                        reward_score=(
                            float(reward) if step_index == step_count - 1 else None
                        ),
                        num_turns=1,
                        metrics=AgentLoopMetrics(
                            generate_sequences=generate_seconds / step_count,
                            num_preempted=num_preempted,
                        ),
                        extra_fields=extra_fields,
                    )
                )
            return step_outputs
