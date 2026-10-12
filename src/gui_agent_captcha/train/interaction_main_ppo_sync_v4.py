from __future__ import annotations

from pprint import pprint
from typing import Any

import ray
import transfer_queue as tq
from omegaconf import OmegaConf
from verl.trainer import main_ppo_sync

from .interaction_step_sample_agent_loop_manager_v4 import (
    InteractionStepSamplePPOTrainerV4,
)

_BaseTaskRunner = main_ppo_sync.TaskRunner.__ray_metadata__.modified_class


@ray.remote
class InteractionStepSampleTaskRunnerV4(_BaseTaskRunner):
    """Run VERL sync PPO with V4 trajectory-normalized browser-step samples."""

    def run(self, config: Any) -> None:
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        try:
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.init_resource_pool_mgr(config)
            trainer = InteractionStepSamplePPOTrainerV4(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=self.resource_pool_manager,
            )
            trainer.init_workers()
            trainer.fit()
        finally:
            tq.close()


def main() -> None:
    # Reuse VERL's own Hydra entrypoint/config package, changing only the remote
    # TaskRunner that instantiates the V4 trajectory-normalizing PPO trainer.
    main_ppo_sync.TaskRunner = InteractionStepSampleTaskRunnerV4
    main_ppo_sync.main()


if __name__ == "__main__":
    main()
