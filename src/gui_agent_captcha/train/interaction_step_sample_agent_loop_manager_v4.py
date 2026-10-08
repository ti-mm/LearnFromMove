from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import torch

STEP_SAMPLE_CONTRACT_V4 = "independent_browser_step_samples_v1"
FINGERPRINT_POLICY_V4 = "rollout_actor_ref_fail_closed_v1"


@dataclass(frozen=True)
class V4StepSampleRecord:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    image_fingerprints: tuple[str, ...]
    processor_fingerprint: str
    ppo_fingerprint: str
    multi_modal_data: Mapping[str, Any] | None = None
    mm_processor_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.prompt_ids or not self.response_ids:
            raise ValueError("V4 step samples require non-empty prompt and response IDs")
        if len(self.response_ids) != len(self.response_logprobs):
            raise ValueError("V4 step response/logprob lengths must match")
        if not self.image_fingerprints:
            raise ValueError("V4 step samples require at least one image fingerprint")
        if not self.processor_fingerprint or not self.ppo_fingerprint:
            raise ValueError("V4 step samples require processor and PPO fingerprints")


@dataclass(frozen=True)
class V4MaterializedStepSample:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    image_fingerprints: tuple[str, ...]
    processor_fingerprint: str
    ppo_fingerprint: str
    trajectory_id: str
    group_id: str
    step_index: int
    step_count: int
    trajectory_reward: float
    trajectory_loss_weight: float
    multi_modal_data: Mapping[str, Any] | None = None
    mm_processor_kwargs: Mapping[str, Any] | None = None


def materialize_step_samples_v4(
    records: Sequence[V4StepSampleRecord],
    *,
    trajectory_id: str,
    group_id: str,
    terminal_reward: float | None,
) -> tuple[V4MaterializedStepSample, ...]:
    if terminal_reward is None:
        raise ValueError("infrastructure-invalid V4 trajectory cannot enter PPO")
    if not records:
        raise ValueError("V4 trajectory produced no policy step samples")
    if not trajectory_id or not group_id:
        raise ValueError("V4 step samples require trajectory and group identity")

    step_count = len(records)
    weight = 1.0 / step_count
    samples: list[V4MaterializedStepSample] = []
    for step_index, record in enumerate(records):
        if record.processor_fingerprint != record.ppo_fingerprint:
            raise ValueError(
                "V4 rollout/processor/PPO fingerprint mismatch; optimizer input is invalid"
            )
        samples.append(
            V4MaterializedStepSample(
                prompt_ids=record.prompt_ids,
                response_ids=record.response_ids,
                response_logprobs=record.response_logprobs,
                image_fingerprints=record.image_fingerprints,
                processor_fingerprint=record.processor_fingerprint,
                ppo_fingerprint=record.ppo_fingerprint,
                trajectory_id=trajectory_id,
                group_id=group_id,
                step_index=step_index,
                step_count=step_count,
                trajectory_reward=float(terminal_reward),
                trajectory_loss_weight=weight,
                multi_modal_data=record.multi_modal_data,
                mm_processor_kwargs=record.mm_processor_kwargs,
            )
        )
    return tuple(samples)


def broadcast_trajectory_advantages_v4(
    *,
    trajectory_rewards: Mapping[str, tuple[str, float]],
    step_trajectory_ids: Sequence[str],
    epsilon: float = 1e-6,
) -> tuple[float, ...]:
    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    for group_id, reward in trajectory_rewards.values():
        grouped_rewards[group_id].append(float(reward))

    trajectory_advantages: dict[str, float] = {}
    for trajectory_id, (group_id, reward) in trajectory_rewards.items():
        rewards = grouped_rewards[group_id]
        if len(rewards) == 1:
            advantage = float(reward)
        else:
            mean = sum(rewards) / len(rewards)
            variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
            advantage = (float(reward) - mean) / (variance**0.5 + epsilon)
        trajectory_advantages[trajectory_id] = advantage
    try:
        return tuple(trajectory_advantages[item] for item in step_trajectory_ids)
    except KeyError as exc:
        raise ValueError(f"unknown V4 trajectory identity: {exc.args[0]}") from exc


def scale_broadcast_advantages_v4(
    advantages: Any,
    *,
    response_mask: Any,
    batch_keys: Sequence[str],
) -> Any:
    """Give every valid trajectory unit total step weight after GRPO broadcast."""

    if len(advantages) != len(batch_keys) or len(response_mask) != len(batch_keys):
        raise ValueError("V4 advantage rows and batch keys must have equal length")
    counts: dict[str, int] = defaultdict(int)
    session_keys: list[str] = []
    for key in batch_keys:
        fields = key.rsplit("_", 2)
        if len(fields) != 3:
            raise ValueError(f"unexpected V4 batch key: {key!r}")
        session_key = f"{fields[0]}_{fields[1]}"
        session_keys.append(session_key)
        counts[session_key] += 1
    weights = advantages.new_tensor(
        [1.0 / counts[session_key] for session_key in session_keys]
    ).unsqueeze(-1)
    return advantages * weights * response_mask.to(dtype=advantages.dtype)


def _validate_independent_outputs_v4(outputs: Sequence[Any]) -> None:
    if not outputs:
        raise ValueError("V4 online AgentLoop returned no independent step samples")
    trajectory_ids: set[str] = set()
    step_counts: set[int] = set()
    step_indices: list[int] = []
    for output in outputs:
        extra = output.extra_fields
        if extra.get("step_sample_contract") != STEP_SAMPLE_CONTRACT_V4:
            raise ValueError("V4 online AgentLoop returned the forbidden concatenated prototype")
        if extra.get("fingerprint_policy") != FINGERPRINT_POLICY_V4:
            raise ValueError("V4 online AgentLoop fingerprint policy is missing")
        processor_fingerprint = str(extra.get("processor_fingerprint") or "")
        ppo_fingerprint = str(extra.get("ppo_fingerprint") or "")
        if not processor_fingerprint or processor_fingerprint != ppo_fingerprint:
            raise ValueError("V4 rollout/actor/ref fingerprint mismatch")
        trajectory_ids.add(str(extra.get("trajectory_id") or ""))
        step_counts.add(int(extra.get("step_count", -1)))
        step_indices.append(int(extra.get("step_index", -1)))
        expected_weight = 1.0 / len(outputs)
        if abs(float(extra.get("trajectory_loss_weight", -1.0)) - expected_weight) > 1e-12:
            raise ValueError("V4 trajectory loss weight must be 1 / valid_step_count")
    if trajectory_ids == {""} or len(trajectory_ids) != 1:
        raise ValueError("V4 step samples must share one non-empty trajectory identity")
    if step_counts != {len(outputs)} or step_indices != list(range(len(outputs))):
        raise ValueError("V4 step samples have inconsistent ordering/count metadata")


try:
    import ray
    import transfer_queue as tq
    from tensordict import TensorDict
    from verl.protocol import DataProto
    from verl.trainer.main_ppo_sync import (
        AgentLoopManagerTQ,
        AgentLoopWorkerTQ,
        PPOTrainer,
    )
    from verl.trainer.ppo import padding_utils as ppo_padding_utils
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict
    from verl.workers.utils.padding import response_to_nested
except (ImportError, ModuleNotFoundError):
    AgentLoopManagerTQ = object  # type: ignore[assignment,misc]
    AgentLoopWorkerTQ = None  # type: ignore[assignment]


if AgentLoopWorkerTQ is None:

    class InteractionStepSampleAgentLoopManagerV4:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "V4 step-sample manager requires VERL main_ppo_sync and TransferQueue==0.1.6"
            )

else:
    _AgentLoopWorkerTQBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class

    _RAGGED_SEQUENCE_FIELDS_V4 = (
        "prompts",
        "responses",
        "attention_mask",
        "response_mask",
        "loss_mask",
        "input_ids",
        "position_ids",
        "rollout_log_probs",
        "rm_scores",
        "routed_experts",
    )

    def _v4_fields_to_tensordict(fields: list[dict[str, Any]]) -> TensorDict:
        """Keep sequence fields jagged even when every row has equal length."""

        packed = list_of_dict_to_tensordict(fields)
        for name in _RAGGED_SEQUENCE_FIELDS_V4:
            if name not in fields[0]:
                continue
            values = [field[name] for field in fields]
            if not all(isinstance(value, torch.Tensor) for value in values):
                continue
            packed[name] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
        return packed

    def _v4_force_validation_text_nested(data: TensorDict) -> TensorDict:
        """Restore the jagged contract after TransferQueue retrieves equal rows."""

        for name in ("prompts", "responses"):
            value = data[name]
            if value.is_nested:
                continue
            if value.ndim != 2:
                raise ValueError(f"V4 validation {name} must be a batched token tensor")
            data[name] = torch.nested.as_nested_tensor(
                list(value.unbind(0)), layout=torch.jagged
            )
        return data

    class _InteractionStepSampleAgentLoopWorkerV4(_AgentLoopWorkerTQBase):
        async def _agent_loop_postprocess(
            self,
            output: Any,
            validate: bool,
            **kwargs: Any,
        ) -> None:
            outputs = output if isinstance(output, list) else [output]
            _validate_independent_outputs_v4(outputs)

            await self._compute_score(outputs, kwargs=kwargs)

            final_output = outputs[-1]
            await self._compute_teacher_logprobs(
                final_output,
                prompt_ids=final_output.prompt_ids,
                response_ids=final_output.response_ids,
                validate=validate,
                sample_kwargs=kwargs,
            )

            if final_output.reward_score is not None:
                for prior_output in outputs[:-1]:
                    prior_output.reward_score = final_output.reward_score
                    prior_output.extra_fields["reward_extra_info"] = (
                        final_output.extra_fields["reward_extra_info"]
                    )

            keys: list[str] = []
            fields: list[dict[str, Any]] = []
            tags: list[dict[str, Any]] = []
            uid, session_id = kwargs["uid"], kwargs["session_id"]
            for index, sample in enumerate(outputs):
                prompts = torch.tensor(sample.prompt_ids, dtype=torch.int64)
                responses = torch.tensor(sample.response_ids, dtype=torch.int64)
                input_ids = torch.cat([prompts, responses], dim=0)
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                multi_modal_inputs = self._compute_multi_modal_inputs(sample, input_ids)
                position_ids = self._compute_position_ids(
                    input_ids.unsqueeze(0),
                    attention_mask.unsqueeze(0),
                    multi_modal_inputs,
                ).squeeze(0)

                keys.append(f"{uid}_{session_id}_{index}")
                field = sample.as_dict()
                field.update(kwargs)
                field.pop("multi_modal_data", None)
                field["loss_mask"] = field["response_mask"]
                field["input_ids"] = input_ids
                field["attention_mask"] = attention_mask
                field["position_ids"] = position_ids
                field["multi_modal_inputs"] = multi_modal_inputs
                fields.append(field)
                prompt_len = field["prompts"].size(0)
                response_len = field["responses"].size(0)
                tags.append(
                    {
                        "global_steps": kwargs["global_steps"],
                        "status": "success",
                        "prompt_len": prompt_len,
                        "response_len": response_len,
                        "seq_len": prompt_len + response_len,
                    }
                )

            await tq.async_kv_batch_put(
                keys=keys,
                fields=_v4_fields_to_tensordict(fields),
                tags=tags,
                partition_id="train" if not validate else "val",
            )


    InteractionStepSampleAgentLoopWorkerV4 = ray.remote(
        _InteractionStepSampleAgentLoopWorkerV4
    )

    class InteractionStepSampleAgentLoopManagerV4(AgentLoopManagerTQ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = InteractionStepSampleAgentLoopWorkerV4


    class InteractionStepSamplePPOTrainerV4(PPOTrainer):
        """Narrow sync trainer adapter for per-trajectory step-loss normalization."""

        def _balance_batch(
            self,
            batch: Any,
            metrics: dict,
            logging_prefix: str = "global_seqlen",
            keep_minibatch: bool = False,
        ) -> Any:
            # VERL may append equal-length synthetic samples to make the expanded
            # step-sample batch divisible by the 16-GPU training layout. Its
            # generic converter stacks those samples densely, while the FSDP
            # no-padding path requires every sequence field to remain jagged.
            with patch.object(
                ppo_padding_utils,
                "list_of_dict_to_tensordict",
                _v4_fields_to_tensordict,
            ):
                return super()._balance_batch(
                    batch,
                    metrics,
                    logging_prefix=logging_prefix,
                    keep_minibatch=keep_minibatch,
                )

        def _validate(self) -> dict[str, Any]:
            original_kv_batch_get = tq.kv_batch_get

            def validation_safe_kv_batch_get(*args: Any, **kwargs: Any) -> TensorDict:
                data = original_kv_batch_get(*args, **kwargs)
                selected = kwargs.get("select_fields")
                if selected is not None and set(selected) == {"prompts", "responses"}:
                    return _v4_force_validation_text_nested(data)
                return data

            with patch.object(tq, "kv_batch_get", validation_safe_kv_batch_get):
                return super()._validate()

        def _compute_advantage(self, batch: Any, metrics: dict) -> Any:
            batch = super()._compute_advantage(batch, metrics)
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["advantages", "returns", "response_mask"],
            )
            response_mask = data["response_mask"]
            padded = DataProto(batch=data.to_padded_tensor())
            scaled = scale_broadcast_advantages_v4(
                padded.batch["advantages"],
                response_mask=padded.batch["response_mask"],
                batch_keys=batch.keys,
            )
            output = {
                "advantages": response_to_nested(scaled, response_mask),
                "returns": response_to_nested(scaled, response_mask),
            }
            tq.kv_batch_put(
                keys=batch.keys,
                partition_id=batch.partition_id,
                fields=type(data)(output, batch_size=len(batch)),
            )
            return batch
