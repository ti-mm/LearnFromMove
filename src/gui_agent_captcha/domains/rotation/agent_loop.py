from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from gui_agent_captcha.actions import PrimitiveAction
from gui_agent_captcha.core import Observation, StepResult
from gui_agent_captcha.domains.rotation.online_contract import (
    INTERACTION_ONLINE_IMAGE_MAX_PIXELS,
    INTERACTION_ONLINE_VIEWPORT,
)
from gui_agent_captcha.domains.rotation.online_protocol import (
    OnlineProtocolState,
    ProtocolViolation,
    ResponseParseError,
    TrajectoryOutcome,
    parse_interaction_response,
    trajectory_reward,
)
from gui_agent_captcha.integrations import browser_runtime
from gui_agent_captcha.protocol_tracks import ROTATION_TASK_REQUIREMENT
from gui_agent_captcha.train.qwen3_vl_sft import (
    QWEN3_RELATIVE_COORDINATE_FORMAT,
    build_prompt_messages_from_history,
)


@dataclass(frozen=True)
class PublicBrowserObservation:
    """Only the observation fields that are permitted to reach model code."""

    screenshot_path: str
    size_px: tuple[int, int]
    step_index: int
    is_terminal: bool


@dataclass(frozen=True)
class GeneratedTurn:
    text: str
    token_ids: list[int]
    logprobs: list[float] | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


STRICT_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up")
ONLINE_ROTATION_INSTRUCTION = ROTATION_TASK_REQUIREMENT


def build_initial_online_messages(
    observation: PublicBrowserObservation,
    *,
    max_steps: int,
) -> list[dict[str, Any]]:
    if observation.step_index != 0:
        raise ValueError("initial online observation must have step_index=0")
    prompt = build_prompt_messages_from_history(
        instruction=ONLINE_ROTATION_INSTRUCTION,
        image_path=Path(observation.screenshot_path),
        image_paths=(Path(observation.screenshot_path),),
        action_history=(),
        total_actions=max_steps,
        cursor_xy=None,
        button_state="up",
        task_type="rotation_captcha",
        allowed_kinds=STRICT_ACTION_KINDS,
        budget_remaining=max_steps,
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
        image_size_px=observation.size_px,
        images_to_keep=1,
    )
    if any(message.get("role") == "system" for message in prompt.messages):
        raise RuntimeError("online rotation prompt must not contain a system message")
    return prompt.messages


def build_incremental_observation_message(
    observation: PublicBrowserObservation,
    *,
    max_steps: int,
) -> dict[str, Any]:
    remaining = max(0, max_steps - observation.step_index)
    text = (
        f"This is the live browser observation produced by action step "
        f"{observation.step_index}."
    )
    if observation.is_terminal:
        text += " The browser episode is terminal; no further action is requested."
    elif remaining == 0:
        text += " The public action budget is exhausted; no further action is requested."
    else:
        text += (
            f" {remaining} action step(s) remain. Use this current screenshot and the "
            "visible interaction history to return the next <think>...</think> block "
            "and exactly one JSON action."
        )
    return {
        "role": "user",
        "content": [
            {"type": "image", "image": observation.screenshot_path},
            {"type": "text", "text": text},
        ],
    }


def build_online_messages(
    observations: Sequence[PublicBrowserObservation],
    turns: Sequence[GeneratedTurn],
    *,
    max_steps: int,
    request_next_action: bool | None = None,
) -> list[dict[str, Any]]:
    del request_next_action
    if not observations:
        raise ValueError("online transcript requires an initial observation")
    if len(observations) not in {len(turns), len(turns) + 1}:
        raise ValueError(
            "online transcript must alternate one model turn with at most one new observation"
        )
    messages = build_initial_online_messages(observations[0], max_steps=max_steps)
    for index, turn in enumerate(turns):
        messages.append({"role": "assistant", "content": turn.text})
        observation_index = index + 1
        if observation_index < len(observations):
            messages.append(
                build_incremental_observation_message(
                    observations[observation_index],
                    max_steps=max_steps,
                )
            )
    return messages


@dataclass(frozen=True)
class PreparedTranscript:
    prompt_ids: list[int]
    messages: list[dict[str, Any]]
    images: list[Image.Image]


@dataclass(frozen=True)
class EncodedTranscript:
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    images: list[Image.Image]
    messages: list[dict[str, Any]]
    mm_processor_kwargs: dict[str, Any]


class OnlineTranscriptCodec:
    """Build a loss-aligned online transcript from full chat-template prefixes."""

    def __init__(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        max_steps: int,
        apply_chat_template_kwargs: Mapping[str, Any] | None,
        mm_processor_kwargs: Mapping[str, Any] | None,
    ) -> None:
        if processor is None:
            raise ValueError("online multimodal GRPO requires a processor")
        browser_runtime.ensure_verl_processor_rope_binding(processor)
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self.mm_processor_kwargs = dict(mm_processor_kwargs or {})
        self.mm_processor_kwargs["truncation"] = False
        for key in tuple(self.mm_processor_kwargs):
            if "max_pixels" not in key.lower():
                continue
            value = self.mm_processor_kwargs[key]
            try:
                normalized_value = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{key} must be an integer pixel cap") from exc
            if normalized_value != INTERACTION_ONLINE_IMAGE_MAX_PIXELS:
                raise ValueError(
                    f"{key} must be exactly {INTERACTION_ONLINE_IMAGE_MAX_PIXELS}, "
                    f"got {value!r}"
                )
            if key != "max_pixels":
                del self.mm_processor_kwargs[key]
        self.mm_processor_kwargs["max_pixels"] = INTERACTION_ONLINE_IMAGE_MAX_PIXELS
        self._image_cache: dict[str, Image.Image] = {}
        self.reset()

    def reset(self) -> None:
        self._initial_prompt_ids: list[int] | None = None
        self._sequence_ids: list[int] = []
        self._response_mask: list[int] = []
        self._response_logprobs: list[float] = []
        self._requires_fragment_continuations = False

    def _images_for_messages(self, messages: Sequence[Mapping[str, Any]]) -> list[Image.Image]:
        paths: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, Mapping) or item.get("type") != "image":
                    continue
                path = item.get("image")
                if not isinstance(path, str) or not path:
                    raise RuntimeError("online image message is missing an image path")
                paths.append(path)

        images: list[Image.Image] = []
        for path_string in paths:
            image = self._image_cache.get(path_string)
            if image is None:
                path = Path(path_string)
                if not path.is_file():
                    raise FileNotFoundError(f"online browser screenshot is missing: {path}")
                with Image.open(path) as source:
                    image = source.convert("RGB").copy()
                if image.size != INTERACTION_ONLINE_VIEWPORT:
                    raise RuntimeError(
                        "online browser screenshot must remain 1280x720, "
                        f"got {image.size!r} for {path}"
                    )
                self._image_cache[path_string] = image
            images.append(image)
        return images

    def _tokenize(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
    ) -> tuple[list[int], list[Image.Image]]:
        try:
            from verl.utils.chat_template import apply_chat_template
            from verl.utils.tokenizer import (
                build_multimodal_processor_inputs,
                normalize_token_ids,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "OnlineTranscriptCodec tokenization requires the verl environment"
            ) from exc

        images = self._images_for_messages(messages)
        raw_prompt = apply_chat_template(
            self.processor,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.apply_chat_template_kwargs,
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[raw_prompt],
            images=images,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        if isinstance(model_inputs, Mapping):
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = model_inputs.input_ids
        return normalize_token_ids(input_ids), images

    def _append_template_delta(self, full_ids: list[int]) -> bool:
        current_length = len(self._sequence_ids)
        if full_ids[:current_length] != self._sequence_ids:
            return False
        delta = full_ids[current_length:]
        self._sequence_ids.extend(delta)
        self._response_mask.extend([0] * len(delta))
        self._response_logprobs.extend([0.0] * len(delta))
        return True

    def _continuation_fragment_ids(
        self,
        user_message: dict[str, Any] | None,
        *,
        add_generation_prompt: bool,
    ) -> list[int]:
        try:
            from verl.utils.chat_template import apply_chat_template
            from verl.utils.tokenizer import (
                build_multimodal_processor_inputs,
                normalize_token_ids,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "online continuation tokenization requires the verl environment"
            ) from exc

        sentinel = "__GUI_CAPTCHA_ASSISTANT_SENTINEL_7f24c9__"
        fragment_messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": sentinel},
        ]
        if user_message is not None:
            fragment_messages.append(user_message)
        rendered = apply_chat_template(
            self.processor,
            fragment_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.apply_chat_template_kwargs,
        )
        sentinel_index = rendered.find(sentinel)
        if sentinel_index < 0:
            raise browser_runtime.OnlineInfrastructureError(
                "chat template did not preserve the continuation sentinel",
                attempts=1,
            )
        suffix_text = rendered[sentinel_index + len(sentinel) :]
        fragment_images = (
            self._images_for_messages([user_message])
            if user_message is not None
            else []
        )
        model_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[suffix_text],
            images=fragment_images,
            mm_processor_kwargs=self.mm_processor_kwargs,
        )
        if isinstance(model_inputs, Mapping):
            input_ids = model_inputs["input_ids"]
        else:
            input_ids = model_inputs.input_ids
        return normalize_token_ids(input_ids)

    def _append_fragment_delta(
        self,
        user_message: dict[str, Any] | None,
        *,
        add_generation_prompt: bool,
    ) -> None:
        delta = self._continuation_fragment_ids(
            user_message,
            add_generation_prompt=add_generation_prompt,
        )
        self._sequence_ids.extend(delta)
        self._response_mask.extend([0] * len(delta))
        self._response_logprobs.extend([0.0] * len(delta))

    def prepare_generation(
        self,
        observations: Sequence[PublicBrowserObservation],
        turns: Sequence[GeneratedTurn],
    ) -> PreparedTranscript:
        if len(observations) != len(turns) + 1:
            raise ValueError("generation requires exactly one current observation after history")
        messages = build_online_messages(
            observations,
            turns,
            max_steps=self.max_steps,
        )
        prompt_ids, images = self._tokenize(messages, add_generation_prompt=True)
        if self._initial_prompt_ids is None:
            self._initial_prompt_ids = list(prompt_ids)
            self._sequence_ids = list(prompt_ids)
        else:
            if self._requires_fragment_continuations:
                self._append_fragment_delta(
                    messages[-1],
                    add_generation_prompt=True,
                )
                prompt_ids = list(self._sequence_ids)
            elif not self._append_template_delta(prompt_ids):
                # Qwen3.5's stock chat template intentionally drops reasoning
                # from assistant turns once a later user turn exists. Preserve
                # the actual sampled prefix and append only the new public user
                # observation/template suffix instead.
                self._requires_fragment_continuations = True
                self._append_fragment_delta(
                    messages[-1],
                    add_generation_prompt=True,
                )
                prompt_ids = list(self._sequence_ids)
        return PreparedTranscript(
            prompt_ids=list(prompt_ids),
            messages=messages,
            images=images,
        )

    def record_generation(
        self,
        prepared: PreparedTranscript,
        turn: GeneratedTurn,
    ) -> None:
        if prepared.prompt_ids != self._sequence_ids:
            raise browser_runtime.OnlineInfrastructureError(
                "LLM prompt ids diverged from the accumulated online transcript",
                attempts=1,
            )
        if turn.logprobs is None:
            raise browser_runtime.OnlineInfrastructureError(
                "rollout server did not return token logprobs",
                attempts=1,
            )
        if len(turn.logprobs) != len(turn.token_ids):
            raise browser_runtime.OnlineInfrastructureError(
                "rollout token/logprob lengths are inconsistent",
                attempts=1,
            )
        self._sequence_ids.extend(turn.token_ids)
        self._response_mask.extend([1] * len(turn.token_ids))
        self._response_logprobs.extend(float(value) for value in turn.logprobs)

    def finalize(
        self,
        observations: Sequence[PublicBrowserObservation],
        turns: Sequence[GeneratedTurn],
    ) -> EncodedTranscript:
        if self._initial_prompt_ids is None:
            raise RuntimeError("cannot finalize a transcript before generation")
        messages = build_online_messages(
            observations,
            turns,
            max_steps=self.max_steps,
        )
        full_ids, images = self._tokenize(messages, add_generation_prompt=False)
        if self._requires_fragment_continuations:
            user_message = messages[-1] if messages[-1].get("role") == "user" else None
            self._append_fragment_delta(
                user_message,
                add_generation_prompt=False,
            )
        elif not self._append_template_delta(full_ids):
            self._requires_fragment_continuations = True
            user_message = messages[-1] if messages[-1].get("role") == "user" else None
            self._append_fragment_delta(
                user_message,
                add_generation_prompt=False,
            )
        response_ids = self._sequence_ids[len(self._initial_prompt_ids) :]
        if len(response_ids) != len(self._response_mask):
            raise AssertionError("online response ids and mask are misaligned")
        if len(response_ids) != len(self._response_logprobs):
            raise AssertionError("online response ids and logprobs are misaligned")
        return EncodedTranscript(
            prompt_ids=list(self._initial_prompt_ids),
            response_ids=list(response_ids),
            response_mask=list(self._response_mask),
            response_logprobs=list(self._response_logprobs),
            images=images,
            messages=messages,
            mm_processor_kwargs=dict(self.mm_processor_kwargs),
        )


@dataclass(frozen=True)
class BrowserTrajectoryResult:
    trajectory_id: str
    request_id: str
    seed: int
    artifact_dir: Path
    outcome: TrajectoryOutcome
    reward: float
    observations: list[PublicBrowserObservation]
    turns: list[GeneratedTurn]
    actions: list[PrimitiveAction]
    terminal_info: dict[str, Any]
    infra_retry_count: int = 0
    cleanup_ok: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


EnvFactory = Callable[[int, Path], Any]
TurnGenerator = Callable[..., Awaitable[GeneratedTurn]]


def default_browser_slot_pool() -> browser_runtime.BrowserSlotPool:
    capacity = int(os.environ.get("INTERACTION_BROWSER_CONCURRENCY", "4"))
    root = Path(
        os.environ.get(
            "INTERACTION_BROWSER_SLOT_DIR",
            "/tmp/gui-captcha-interaction-browser-slots",
        )
    )
    timeout_s = float(os.environ.get("INTERACTION_BROWSER_SLOT_TIMEOUT_S", "300"))
    return browser_runtime.BrowserSlotPool(
        root=root,
        capacity=capacity,
        acquire_timeout_s=timeout_s,
    )


class BrowserTrajectoryRunner:
    """Execute one real stateful browser trajectory with infra-safe retries."""

    def __init__(
        self,
        *,
        env_factory: EnvFactory,
        turn_generator: TurnGenerator,
        artifact_root: Path,
        slot_pool: browser_runtime.BrowserSlotPool | None = None,
        max_steps: int = 6,
        max_infra_retries: int = 2,
        id_factory: Callable[[], str] | None = None,
        task_type: str = "rotation_captcha",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if max_infra_retries < 0:
            raise ValueError("max_infra_retries cannot be negative")
        self.env_factory = env_factory
        self.turn_generator = turn_generator
        self.artifact_root = Path(artifact_root)
        self.slot_pool = slot_pool or default_browser_slot_pool()
        self.max_steps = max_steps
        self.max_infra_retries = max_infra_retries
        self.id_factory = id_factory or (lambda: uuid4().hex)
        self.task_type = task_type

    async def _on_browser_thread(
        self,
        executor: ThreadPoolExecutor,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            partial(function, *args, **kwargs),
        )

    def _public_observation(
        self,
        observation: Observation,
        *,
        step_index: int,
        is_terminal: bool,
    ) -> PublicBrowserObservation:
        if observation.size_px != INTERACTION_ONLINE_VIEWPORT:
            raise RuntimeError(
                "online GRPO requires a raw 1280x720 browser observation, "
                f"got {observation.size_px!r}"
            )
        if not observation.screenshot_path:
            raise RuntimeError("browser observation is missing a screenshot path")
        return PublicBrowserObservation(
            screenshot_path=str(observation.screenshot_path),
            size_px=INTERACTION_ONLINE_VIEWPORT,
            step_index=step_index,
            is_terminal=is_terminal,
        )

    def _result(
        self,
        *,
        trajectory_id: str,
        request_id: str,
        seed: int,
        artifact_dir: Path,
        outcome: TrajectoryOutcome,
        observations: list[PublicBrowserObservation],
        turns: list[GeneratedTurn],
        actions: list[PrimitiveAction],
        terminal_info: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> BrowserTrajectoryResult:
        reward = trajectory_reward(outcome)
        if reward is None:
            raise AssertionError("infrastructure outcomes must be raised, not returned")
        diagnostics: dict[str, Any] = {
            "num_actions": len(actions),
            "num_observations": len(observations),
            "num_model_turns": len(turns),
            "outcome": outcome.value,
        }
        if error is not None:
            diagnostics["policy_error"] = f"{type(error).__name__}: {error}"
        return BrowserTrajectoryResult(
            trajectory_id=trajectory_id,
            request_id=request_id,
            seed=seed,
            artifact_dir=artifact_dir,
            outcome=outcome,
            reward=reward,
            observations=observations,
            turns=turns,
            actions=actions,
            terminal_info=dict(terminal_info or {}),
            diagnostics=diagnostics,
        )

    @staticmethod
    def _screenshot_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_trajectory_audit(self, result: BrowserTrajectoryResult) -> None:
        screenshot_paths = [
            Path(observation.screenshot_path) for observation in result.observations
        ]
        screenshot_exists = [path.is_file() for path in screenshot_paths]
        payload = {
            "trajectory_id": result.trajectory_id,
            "request_id": result.request_id,
            "seed": result.seed,
            "outcome": result.outcome.value,
            "reward": result.reward,
            "cleanup_ok": result.cleanup_ok,
            "infra_retry_count": result.infra_retry_count,
            "num_model_turns": len(result.turns),
            "num_actions": len(result.actions),
            "num_observations": len(result.observations),
            "actions": [action.to_dict() for action in result.actions],
            "model_turns": [
                {
                    "text": turn.text,
                    "num_tokens": len(turn.token_ids),
                }
                for turn in result.turns
            ],
            "screenshot_paths": [str(path) for path in screenshot_paths],
            "screenshot_sizes": [
                list(observation.size_px) for observation in result.observations
            ],
            "screenshot_exists": screenshot_exists,
            "screenshot_sha256": [
                self._screenshot_sha256(path) if exists else None
                for path, exists in zip(screenshot_paths, screenshot_exists, strict=True)
            ],
            "trajectory_started_at_unix": result.diagnostics.get(
                "trajectory_started_at_unix"
            ),
            "trajectory_finished_at_unix": result.diagnostics.get(
                "trajectory_finished_at_unix"
            ),
            "trajectory_seconds": result.diagnostics.get("trajectory_seconds"),
            "diagnostics": dict(result.diagnostics),
        }
        (result.artifact_dir / "trajectory_result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    async def _run_attempt(
        self,
        *,
        env: Any,
        executor: ThreadPoolExecutor,
        trajectory_id: str,
        request_id: str,
        seed: int,
        artifact_dir: Path,
        sampling_params: dict[str, Any],
    ) -> BrowserTrajectoryResult:
        task_id = (
            env.resolve_task_instance_id(self.task_type)
            if hasattr(env, "resolve_task_instance_id")
            else f"{self.task_type}:seed={seed}"
        )
        initial_observation: Observation = await self._on_browser_thread(
            executor,
            env.reset,
            task_type=self.task_type,
            task_id=task_id,
        )
        observations = [
            self._public_observation(
                initial_observation,
                step_index=0,
                is_terminal=False,
            )
        ]
        turns: list[GeneratedTurn] = []
        actions: list[PrimitiveAction] = []
        protocol = OnlineProtocolState(max_steps=self.max_steps)

        while True:
            generated = await self.turn_generator(
                request_id=request_id,
                observations=tuple(observations),
                turns=tuple(turns),
                sampling_params=dict(sampling_params),
            )
            if not isinstance(generated, GeneratedTurn):
                raise TypeError("turn_generator must return GeneratedTurn")
            turns.append(generated)

            try:
                parsed = parse_interaction_response(generated.text)
            except ResponseParseError as exc:
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=TrajectoryOutcome.PARSE_ERROR,
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    error=exc,
                )

            try:
                protocol.accept(parsed.action)
            except ProtocolViolation as exc:
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=TrajectoryOutcome.PROTOCOL_ERROR,
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    error=exc,
                )

            step_result: StepResult = await self._on_browser_thread(
                executor,
                env.step,
                parsed.action,
            )
            if not isinstance(step_result, StepResult):
                raise TypeError("browser env.step must return StepResult")
            actions.append(parsed.action)
            public_observation = self._public_observation(
                step_result.observation,
                step_index=len(actions),
                is_terminal=bool(step_result.done),
            )
            if public_observation.screenshot_path == observations[-1].screenshot_path:
                raise RuntimeError(
                    "browser env.step did not produce a new screenshot observation"
                )
            observations.append(public_observation)

            if protocol.is_terminal:
                if not step_result.done:
                    raise RuntimeError(
                        "mouse_up did not produce a terminal browser result"
                    )
                success = step_result.info.get("success")
                if type(success) is not bool:
                    raise RuntimeError(
                        "terminal browser result is missing a boolean success field"
                    )
                outcome = (
                    TrajectoryOutcome.SUCCESS
                    if success
                    else TrajectoryOutcome.BROWSER_FAILURE
                )
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=outcome,
                    observations=observations,
                    turns=turns,
                    actions=actions,
                    terminal_info=step_result.info,
                )

            if step_result.done:
                raise RuntimeError(
                    f"browser terminated unexpectedly after {parsed.action.kind}"
                )
            if protocol.reached_max_steps:
                return self._result(
                    trajectory_id=trajectory_id,
                    request_id=request_id,
                    seed=seed,
                    artifact_dir=artifact_dir,
                    outcome=TrajectoryOutcome.MAX_STEPS,
                    observations=observations,
                    turns=turns,
                    actions=actions,
                )

    async def run(
        self,
        *,
        seed: int,
        sampling_params: dict[str, Any],
    ) -> BrowserTrajectoryResult:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("trajectory seed must be an integer")

        trajectory_started_at_unix = time.time()
        trajectory_started_at_perf = time.perf_counter()
        trajectory_id = str(self.id_factory())
        request_id = str(self.id_factory())
        trajectory_root = self.artifact_root / f"seed-{seed}" / trajectory_id
        trajectory_root.mkdir(parents=True, exist_ok=False)

        lease = await asyncio.to_thread(self.slot_pool.acquire)
        executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"interaction-{trajectory_id[:12]}",
        )
        infra_errors: list[str] = []
        try:
            total_attempts = self.max_infra_retries + 1
            for attempt_index in range(total_attempts):
                attempt_dir = trajectory_root / f"attempt-{attempt_index:02d}"
                attempt_dir.mkdir(parents=True, exist_ok=False)
                env: Any = None
                result: BrowserTrajectoryResult | None = None
                attempt_error: Exception | None = None
                try:
                    env = await self._on_browser_thread(
                        executor,
                        self.env_factory,
                        seed,
                        attempt_dir,
                    )
                    result = await self._run_attempt(
                        env=env,
                        executor=executor,
                        trajectory_id=trajectory_id,
                        request_id=request_id,
                        seed=seed,
                        artifact_dir=attempt_dir,
                        sampling_params=sampling_params,
                    )
                except Exception as exc:
                    attempt_error = exc
                finally:
                    if env is not None:
                        try:
                            await self._on_browser_thread(executor, env.close)
                        except Exception as close_exc:
                            if attempt_error is None:
                                attempt_error = close_exc
                            else:
                                attempt_error = RuntimeError(
                                    f"{attempt_error}; cleanup failed: {close_exc}"
                                )

                if attempt_error is None:
                    if result is None:
                        raise AssertionError("trajectory attempt ended without result")
                    diagnostics = dict(result.diagnostics)
                    diagnostics["infra_errors_before_success"] = list(infra_errors)
                    diagnostics.update(
                        {
                            "trajectory_started_at_unix": trajectory_started_at_unix,
                            "trajectory_finished_at_unix": time.time(),
                            "trajectory_seconds": time.perf_counter()
                            - trajectory_started_at_perf,
                        }
                    )
                    completed = replace(
                        result,
                        infra_retry_count=attempt_index,
                        cleanup_ok=True,
                        diagnostics=diagnostics,
                    )
                    self._write_trajectory_audit(completed)
                    return completed

                infra_errors.append(
                    f"attempt-{attempt_index:02d}: "
                    f"{type(attempt_error).__name__}: {attempt_error}"
                )

            raise browser_runtime.OnlineInfrastructureError(
                "online browser trajectory exhausted infrastructure retries: "
                + infra_errors[-1],
                attempts=total_attempts,
                errors=tuple(infra_errors),
            )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            lease.release()


def _default_interaction_env_factory(seed: int, artifact_dir: Path) -> Any:
    from gui_agent_captcha.domains.rotation.environment import InteractionCaptchaEnv

    return InteractionCaptchaEnv(
        base_url=os.environ.get("INTERACTION_CAPTCHA_URL", "http://localhost:4321/"),
        artifact_dir=artifact_dir,
        challenge_seed=seed,
        enable_playwright=True,
        headless=os.environ.get("INTERACTION_BROWSER_HEADLESS", "1") != "0",
        viewport_width=INTERACTION_ONLINE_VIEWPORT[0],
        viewport_height=INTERACTION_ONLINE_VIEWPORT[1],
    )


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    from verl.workers.rollout.replica import TokenOutput
except ModuleNotFoundError as exc:
    VERL_AVAILABLE = False
    VERL_IMPORT_ERROR = exc

    class InteractionBrowserAgentLoop:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "InteractionBrowserAgentLoop must be imported in the verl training environment"
            ) from VERL_IMPORT_ERROR

else:
    VERL_AVAILABLE = True

    @register("interaction_browser_online")
    class InteractionBrowserAgentLoop(AgentLoopBase):
        """verl AgentLoop for live HD720 Interaction rotation CAPTCHA rollouts."""

        def __init__(
            self,
            *args: Any,
            env_factory: EnvFactory | None = None,
            artifact_root: str | Path | None = None,
            slot_pool: browser_runtime.BrowserSlotPool | None = None,
            id_factory: Callable[[], str] | None = None,
            max_steps: int = 6,
            max_turn_tokens: int = 192,
            max_infra_retries: int = 2,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if self.processor is None:
                raise ValueError("InteractionBrowserAgentLoop requires a multimodal processor")
            if max_turn_tokens <= 0:
                raise ValueError("max_turn_tokens must be positive")
            self.max_steps = int(max_steps)
            self.max_turn_tokens = int(max_turn_tokens)
            self.max_infra_retries = int(max_infra_retries)
            self.prompt_length = int(self.rollout_config.prompt_length)
            self.response_length = int(self.rollout_config.response_length)
            self.max_model_len = int(
                self.rollout_config.get(
                    "max_model_len",
                    self.prompt_length + self.response_length,
                )
            )
            self.env_factory = env_factory or _default_interaction_env_factory
            self.artifact_root = Path(
                artifact_root
                or os.environ.get(
                    "INTERACTION_ONLINE_ARTIFACT_ROOT",
                    "artifacts/runs/interaction_browser_online_grpo",
                )
            )
            self.slot_pool = slot_pool or default_browser_slot_pool()
            self.id_factory = id_factory or (lambda: uuid4().hex)
            self.last_trajectory_result: BrowserTrajectoryResult | None = None

        def _validate_task_config(
            self,
            task_type: Any,
            task_config: Any,
        ) -> dict[str, Any]:
            if task_type is None:
                task_type = "rotation_captcha"
            if task_type != "rotation_captcha":
                raise ValueError(
                    f"InteractionBrowserAgentLoop only supports rotation_captcha, got {task_type!r}"
                )
            if hasattr(task_config, "item") and not isinstance(task_config, Mapping):
                task_config = task_config.item()
            if not isinstance(task_config, Mapping):
                raise ValueError("online GRPO row is missing public task_config")
            normalized = dict(task_config)
            required = {
                "background_pool",
                "viewport",
                "max_steps",
                "coordinate_format",
                "action_kinds",
            }
            if set(normalized) != required:
                raise ValueError(
                    f"task_config keys must be exactly {sorted(required)!r}"
                )
            if list(normalized["viewport"]) != list(INTERACTION_ONLINE_VIEWPORT):
                raise ValueError("online GRPO viewport must be exactly 1280x720")
            requested_max_steps = int(normalized["max_steps"])
            if requested_max_steps <= 0 or requested_max_steps > self.max_steps:
                raise ValueError(
                    "dataset max_steps must be positive and no larger than the AgentLoop cap"
                )
            if normalized["coordinate_format"] != "qwen_relative_0_1000":
                raise ValueError("unexpected coordinate format")
            if tuple(normalized["action_kinds"]) != STRICT_ACTION_KINDS:
                raise ValueError("online GRPO permits only move_to, mouse_down, and mouse_up")
            configured_pool = os.environ.get("INTERACTION_CAPTCHA_BACKGROUND_POOL_DIR")
            if configured_pool and normalized["background_pool"] != configured_pool:
                raise ValueError(
                    "dataset background_pool does not match the native Astro server: "
                    f"{normalized['background_pool']!r} != {configured_pool!r}"
                )
            return normalized

        async def run(
            self,
            sampling_params: dict[str, Any],
            **kwargs: Any,
        ) -> AgentLoopOutput:
            seed_value = kwargs.get("seed")
            if hasattr(seed_value, "item"):
                seed_value = seed_value.item()
            if not isinstance(seed_value, int) or isinstance(seed_value, bool):
                raise ValueError(f"online GRPO row has invalid seed: {seed_value!r}")
            seed = int(seed_value)
            task_config = self._validate_task_config(
                kwargs.get("task_type"),
                kwargs.get("task_config"),
            )
            effective_max_steps = int(task_config["max_steps"])

            codec = OnlineTranscriptCodec(
                processor=self.processor,
                tokenizer=self.tokenizer,
                max_steps=effective_max_steps,
                apply_chat_template_kwargs=self.apply_chat_template_kwargs,
                mm_processor_kwargs=self.mm_processor_kwargs,
            )
            active_initial_screenshot: str | None = None
            generate_seconds = 0.0
            num_preempted = 0
            rollout_extra_fields: dict[str, Any] = {}

            async def generate_turn(
                *,
                request_id: str,
                observations: Sequence[PublicBrowserObservation],
                turns: Sequence[GeneratedTurn],
                sampling_params: dict[str, Any],
            ) -> GeneratedTurn:
                nonlocal active_initial_screenshot
                nonlocal generate_seconds
                nonlocal num_preempted
                if not turns:
                    initial_screenshot = observations[0].screenshot_path
                    if active_initial_screenshot != initial_screenshot:
                        codec.reset()
                        active_initial_screenshot = initial_screenshot

                prepared = await asyncio.get_running_loop().run_in_executor(
                    None,
                    codec.prepare_generation,
                    observations,
                    turns,
                )
                per_turn_sampling = dict(sampling_params)
                per_turn_sampling.pop("max_new_tokens", None)
                per_turn_sampling["max_tokens"] = self.max_turn_tokens

                started = time.perf_counter()
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prepared.prompt_ids,
                    sampling_params=per_turn_sampling,
                    image_data=prepared.images,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs=codec.mm_processor_kwargs,
                )
                generate_seconds += time.perf_counter() - started
                num_preempted += int(output.num_preempted or 0)

                raw_text = self.tokenizer.decode(
                    output.token_ids,
                    skip_special_tokens=True,
                ).strip()
                if raw_text.startswith("<think>"):
                    full_text = raw_text
                else:
                    full_text = "<think>\n" + raw_text
                turn = GeneratedTurn(
                    text=full_text,
                    token_ids=list(output.token_ids),
                    logprobs=(
                        [float(value) for value in output.log_probs]
                        if output.log_probs is not None
                        else None
                    ),
                    extra_fields=dict(output.extra_fields),
                )
                codec.record_generation(prepared, turn)
                if not rollout_extra_fields:
                    rollout_extra_fields.update(output.extra_fields)
                else:
                    max_global_steps = output.extra_fields.get("max_global_steps")
                    if max_global_steps is not None:
                        rollout_extra_fields["max_global_steps"] = max_global_steps
                return turn

            runner = BrowserTrajectoryRunner(
                env_factory=self.env_factory,
                turn_generator=generate_turn,
                artifact_root=self.artifact_root,
                slot_pool=self.slot_pool,
                max_steps=effective_max_steps,
                max_infra_retries=self.max_infra_retries,
                id_factory=self.id_factory,
            )
            result = await runner.run(seed=seed, sampling_params=sampling_params)
            self.last_trajectory_result = result
            encoded = await asyncio.get_running_loop().run_in_executor(
                None,
                codec.finalize,
                tuple(result.observations),
                tuple(result.turns),
            )

            if len(encoded.prompt_ids) > self.prompt_length:
                raise browser_runtime.OnlineInfrastructureError(
                    "initial prompt exceeds the configured prompt budget; truncation is forbidden: "
                    f"{len(encoded.prompt_ids)} > {self.prompt_length}",
                    attempts=result.infra_retry_count + 1,
                )
            if len(encoded.response_ids) > self.response_length:
                raise browser_runtime.OnlineInfrastructureError(
                    "online response exceeds the configured response budget; truncation is forbidden: "
                    f"{len(encoded.response_ids)} > {self.response_length}",
                    attempts=result.infra_retry_count + 1,
                )
            total_model_tokens = len(encoded.prompt_ids) + len(encoded.response_ids)
            if total_model_tokens > self.max_model_len:
                raise browser_runtime.OnlineInfrastructureError(
                    "online trajectory exceeds max_model_len; truncation is forbidden: "
                    f"{total_model_tokens} > {self.max_model_len}",
                    attempts=result.infra_retry_count + 1,
                )

            extra_fields = dict(rollout_extra_fields)
            extra_fields.update(result.diagnostics)
            extra_fields.update(
                {
                    "trajectory_id": result.trajectory_id,
                    "request_id": result.request_id,
                    "seed": seed,
                    "task_config": task_config,
                    "trajectory_outcome": result.outcome.value,
                    "infra_retry_count": result.infra_retry_count,
                    "cleanup_ok": result.cleanup_ok,
                    "num_actions": len(result.actions),
                    "num_observations": len(result.observations),
                    "artifact_dir": str(result.artifact_dir),
                    "terminal_screenshot_path": result.observations[-1].screenshot_path,
                    "screenshot_paths": [
                        observation.screenshot_path for observation in result.observations
                    ],
                    "actions": [action.to_dict() for action in result.actions],
                    "transcript_messages": encoded.messages,
                    "turn_scores": [],
                    "tool_rewards": [],
                    "trajectory": {
                        "outcome": result.outcome.value,
                        "num_actions": len(result.actions),
                        "num_observations": len(result.observations),
                        "infra_retry_count": result.infra_retry_count,
                        "cleanup_ok": result.cleanup_ok,
                        "artifact_dir": str(result.artifact_dir),
                    },
                }
            )
            return AgentLoopOutput(
                prompt_ids=encoded.prompt_ids,
                response_ids=encoded.response_ids,
                response_mask=encoded.response_mask,
                response_logprobs=encoded.response_logprobs,
                multi_modal_data={"images": encoded.images},
                mm_processor_kwargs=encoded.mm_processor_kwargs,
                reward_score=result.reward,
                num_turns=len(encoded.messages),
                metrics=AgentLoopMetrics(
                    generate_sequences=generate_seconds,
                    num_preempted=num_preempted,
                ),
                extra_fields=extra_fields,
            )
