"""VERL sync entrypoint for the GroundCUA paper-style static AgentLoop."""

from __future__ import annotations

from pprint import pprint
from typing import Any

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.trainer import main_ppo_sync
from verl.utils import transferqueue_utils
from verl.workers.engine_workers import ActorRolloutRefWorker

from .groundcua_paper_style_step_manager import (
    GroundCUAStepSamplePPOTrainer,
    _groundcua_restore_jagged_sequences,
)


_BaseTaskRunner = main_ppo_sync.TaskRunner.__ray_metadata__.modified_class


def _install_groundcua_transferqueue_restore() -> None:
    """Make VERL worker-side TransferQueue reads preserve jagged sequences."""

    marker = "_groundcua_original_async_meta_to_realdata"
    if hasattr(transferqueue_utils, marker):
        return
    original = transferqueue_utils._async_meta_to_realdata

    async def restore_after_read(meta: Any) -> Any:
        data = await original(meta)
        return _groundcua_restore_jagged_sequences(data)

    setattr(transferqueue_utils, marker, original)
    transferqueue_utils._async_meta_to_realdata = restore_after_read


class _GroundCUAActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args: Any, **kwargs: Any):
        _install_groundcua_transferqueue_restore()
        super().__init__(*args, **kwargs)


@ray.remote
class GroundCUAStepSampleTaskRunner(_BaseTaskRunner):
    def add_actor_rollout_worker(self, config: Any) -> None:
        super().add_actor_rollout_worker(config)
        for role, worker_cls in list(self.role_worker_mapping.items()):
            if worker_cls is not None and role in {
                main_ppo_sync.Role.ActorRollout,
                main_ppo_sync.Role.ActorRolloutRef,
            }:
                self.role_worker_mapping[role] = ray.remote(
                    _GroundCUAActorRolloutRefWorker
                )

    def run(self, config: Any) -> None:
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        original_kv_batch_get = tq.kv_batch_get

        def groundcua_kv_batch_get(*args: Any, **kwargs: Any) -> Any:
            return _groundcua_restore_jagged_sequences(
                original_kv_batch_get(*args, **kwargs)
            )

        tq.kv_batch_get = groundcua_kv_batch_get
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            trainer = GroundCUAStepSamplePPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            tq.kv_batch_get = original_kv_batch_get
            tq.close()


def main() -> None:
    main_ppo_sync.TaskRunner = GroundCUAStepSampleTaskRunner
    main_ppo_sync.main()


if __name__ == "__main__":
    main()
