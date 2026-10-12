"""VERL sync-manager support for multi-turn GroundCUA static trajectories."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import torch


def _validate_step_outputs(outputs: Sequence[Any]) -> None:
    if not outputs:
        raise ValueError("GroundCUA AgentLoop returned no policy step samples")
    trajectory_ids: set[str] = set()
    step_counts: set[int] = set()
    step_indices: list[int] = []
    for output in outputs:
        extra = output.extra_fields
        trajectory_ids.add(str(extra.get("trajectory_id") or ""))
        step_counts.add(int(extra.get("step_count", -1)))
        step_indices.append(int(extra.get("step_index", -1)))
        expected_weight = 1.0 / len(outputs)
        if abs(float(extra.get("trajectory_loss_weight", -1.0)) - expected_weight) > 1e-12:
            raise ValueError("GroundCUA trajectory loss weight must equal 1 / step_count")
    if trajectory_ids == {""} or len(trajectory_ids) != 1:
        raise ValueError("GroundCUA step samples must share one nonempty trajectory identity")
    if step_counts != {len(outputs)} or step_indices != list(range(len(outputs))):
        raise ValueError("GroundCUA step samples have inconsistent ordering or count")
    terminal_reward = outputs[-1].reward_score
    if terminal_reward is None or not math.isfinite(float(terminal_reward)) or not any(
        math.isclose(float(terminal_reward), expected, rel_tol=0.0, abs_tol=1e-6)
        for expected in (-0.2, 0.0, 1.0)
    ):
        raise ValueError("GroundCUA terminal reward must be -0.2, 0.0, or 1.0")
    if any(output.reward_score is not None for output in outputs[:-1]):
        raise ValueError("only the terminal GroundCUA step may carry reward_score")


def _scale_trajectory_advantages(
    advantages: Any,
    *,
    response_mask: Any,
    batch_keys: Sequence[str],
) -> Any:
    if len(advantages) != len(batch_keys) or len(response_mask) != len(batch_keys):
        raise ValueError("GroundCUA advantage rows and batch keys must have equal length")
    counts: dict[str, int] = defaultdict(int)
    session_keys: list[str] = []
    for key in batch_keys:
        fields = key.rsplit("_", 2)
        if len(fields) != 3:
            raise ValueError(f"unexpected GroundCUA batch key: {key!r}")
        session_key = f"{fields[0]}_{fields[1]}"
        session_keys.append(session_key)
        counts[session_key] += 1
    weights = advantages.new_tensor(
        [1.0 / counts[session_key] for session_key in session_keys]
    ).unsqueeze(-1)
    return advantages * weights * response_mask.to(dtype=advantages.dtype)


def _groundcua_global_steps(prompt: dict[str, Any], trajectory: dict[str, Any]) -> int:
    """Resolve the replay-buffer step from VERL prompt or trajectory metadata."""

    value = prompt.get("global_steps")
    if isinstance(value, torch.Tensor):
        value = value.item()
    if value is None:
        value = trajectory.get("step", -1)
    return int(value)


_GROUND_CUA_RAGGED_FIELDS = (
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


def _groundcua_restore_jagged_sequences(data: Any) -> Any:
    """Restore jagged sequence fields after TransferQueue batches equal rows.

    TransferQueue 0.1.6 packs equal-shaped per-sample tensors with ``stack``.
    VERL's no-padding path instead requires sequence fields to expose jagged
    ``offsets()`` metadata, so convert only the known batched sequence fields.
    """

    for name in _GROUND_CUA_RAGGED_FIELDS:
        if name not in data:
            continue
        value = data[name]
        if not isinstance(value, torch.Tensor) or value.is_nested or value.ndim < 2:
            continue
        data[name] = torch.nested.as_nested_tensor(
            list(value.unbind(0)), layout=torch.jagged
        )
    return data


try:
    import ray
    import transfer_queue as tq
    from omegaconf import OmegaConf, open_dict
    from tensordict import TensorDict
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.protocol import DataProto
    from verl.trainer import main_ppo_sync as verl_main_ppo_sync
    from verl.trainer.main_ppo_sync import AgentLoopManagerTQ, AgentLoopWorkerTQ, PPOTrainer
    from verl.utils.tensordict_utils import list_of_dict_to_tensordict
    from verl.workers.utils.padding import response_to_nested
except (ImportError, ModuleNotFoundError):
    AgentLoopManagerTQ = object  # type: ignore[assignment,misc]
    AgentLoopWorkerTQ = None  # type: ignore[assignment]


if AgentLoopWorkerTQ is None:

    class GroundCUAStepSampleAgentLoopManager:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "GroundCUA step manager requires VERL main_ppo_sync and TransferQueue"
            )

    class GroundCUAStepSamplePPOTrainer:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "GroundCUA step trainer requires VERL main_ppo_sync and TransferQueue"
            )

else:
    _AgentLoopWorkerBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class

    _RAGGED_SEQUENCE_FIELDS = (
        "prompts",
        "responses",
        "response_mask",
        "loss_mask",
        "input_ids",
        "position_ids",
        "rollout_log_probs",
        "rm_scores",
        "routed_experts",
    )

    def _groundcua_fields_to_tensordict(fields: list[dict[str, Any]]) -> TensorDict:
        """Keep sequence fields jagged even when a batch happens to have equal lengths.

        VERL's generic converter stacks equal-shaped tensors.  The sync trainer
        relies on ``prompts`` and ``responses`` exposing ``offsets()`` whenever
        remove-padding is enabled, so equal-length responses must not silently
        become dense tensors.
        """

        packed = list_of_dict_to_tensordict(fields)
        for name in _RAGGED_SEQUENCE_FIELDS:
            if name not in fields[0]:
                continue
            values = [field[name] for field in fields]
            if not all(isinstance(value, torch.Tensor) for value in values):
                continue
            packed[name] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
        return packed

    class _GroundCUAStepSampleAgentLoopWorker(_AgentLoopWorkerBase):
        async def _run_prompt(
            self,
            prompt: dict[str, Any],
            sampling_params: dict[str, Any],
            trajectory: dict[str, Any],
            trace: bool = False,
        ) -> None:
            """Forward the trainer step needed by the TransferQueue replay key."""

            prompt = dict(prompt)
            prompt["global_steps"] = _groundcua_global_steps(prompt, trajectory)
            await super()._run_prompt(prompt, sampling_params, trajectory, trace)

        async def _agent_loop_postprocess(
            self, output: Any, validate: bool, **kwargs: Any
        ) -> None:
            outputs = output if isinstance(output, list) else [output]
            _validate_step_outputs(outputs)
            # This is the VERL 0.1.6 postprocess contract with one deliberate
            # difference: sequence fields are written as jagged tensors.
            output = outputs[-1]
            output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]
            await self._compute_score(outputs, kwargs=kwargs)
            await self._compute_teacher_logprobs(
                output,
                prompt_ids=output.prompt_ids,
                response_ids=output.response_ids,
                validate=validate,
                sample_kwargs=kwargs,
            )

            if output.reward_score is not None:
                for prior in outputs[:-1]:
                    prior.reward_score = output.reward_score
                    prior.extra_fields["reward_extra_info"] = output.extra_fields[
                        "reward_extra_info"
                    ]

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
                    input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
                ).squeeze(0)

                keys.append(f"{uid}_{session_id}_{index}")
                field = sample.as_dict()
                field.update(kwargs)
                field.pop("multi_modal_data", None)
                field["loss_mask"] = field["response_mask"]
                field["input_ids"] = input_ids
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
                fields=_groundcua_fields_to_tensordict(fields),
                tags=tags,
                partition_id="train" if not validate else "val",
            )

    GroundCUAStepSampleAgentLoopWorker = ray.remote(_GroundCUAStepSampleAgentLoopWorker)

    class GroundCUAStepSampleAgentLoopManager(AgentLoopManagerTQ):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.agent_loop_workers_class = GroundCUAStepSampleAgentLoopWorker

    class GroundCUAStepSamplePPOTrainer(PPOTrainer):
        """Normalize a terminal trajectory advantage across its generated turns."""

        def _init_dataloader(self) -> None:
            if list(self.config.data.val_files):
                raise ValueError("GroundCUA paper-style RL must not define validation data")
            if self.config.trainer.get("val_before_train", True):
                raise ValueError("GroundCUA paper-style RL must disable validation before training")
            if int(self.config.trainer.get("test_freq", 0)) > 0:
                raise ValueError("GroundCUA paper-style RL must disable periodic validation")

            self.train_dataset = verl_main_ppo_sync.create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                is_train=True,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
            self.val_dataset = None
            self.train_dataloader = StatefulDataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.data.get(
                    "gen_batch_size", self.config.data.train_batch_size
                ),
                num_workers=self.config.data["dataloader_num_workers"],
                drop_last=True,
                collate_fn=verl_main_ppo_sync.collate_fn,
                sampler=verl_main_ppo_sync.create_rl_sampler(
                    self.config.data, self.train_dataset
                ),
            )
            self.val_dataloader = ()
            total_training_steps = (
                len(self.train_dataloader) * self.config.trainer.total_epochs
            )
            if self.config.trainer.total_training_steps is not None:
                total_training_steps = self.config.trainer.total_training_steps
            self.total_training_steps = total_training_steps
            try:
                OmegaConf.set_struct(self.config, True)
                with open_dict(self.config):
                    if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                        self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                            total_training_steps
                        )
                    if OmegaConf.select(self.config, "critic.optim"):
                        self.config.critic.optim.total_training_steps = total_training_steps
            except Exception as exc:
                raise RuntimeError(
                    "failed to propagate GroundCUA total_training_steps"
                ) from exc

        def _compute_advantage(self, batch: Any, metrics: dict) -> Any:
            batch = super()._compute_advantage(batch, metrics)
            data = tq.kv_batch_get(
                keys=batch.keys,
                partition_id=batch.partition_id,
                select_fields=["advantages", "returns", "response_mask"],
            )
            response_mask = data["response_mask"]
            padded = DataProto(batch=data.to_padded_tensor())
            scaled = _scale_trajectory_advantages(
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


__all__ = [
    "GroundCUAStepSampleAgentLoopManager",
    "GroundCUAStepSamplePPOTrainer",
]
