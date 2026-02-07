# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.main_ppo import run_ppo as base_run_ppo

from recipe.hint.hint_trainer import RayHintTrainer


@hydra.main(config_path="config", config_name="hint_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


class TaskRunner(BaseTaskRunner):
    trainer_cls = RayHintTrainer


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head
    base_run_ppo(config, task_runner_class=task_runner_class)


if __name__ == "__main__":
    main()
