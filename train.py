"""
Example of a custom gym environment. Run this example for a demo.

This example shows the usage of:
  - a custom environment
  - Ray Tune for grid search to try different learning rates

You can visualize experiment results in ~/ray_results using TensorBoard.

Run example with defaults:
$ python custom_env.py
For CLI options:
$ python custom_env.py --help
"""
import argparse
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import os
import random

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
# )
# parser.add_argument(
#     "--framework",
#     choices=["tf", "tf2", "torch"],
#     default="torch",
#     help="The DL framework specifier.",
# )
# parser.add_argument(
#     "--as-test",
#     action="store_true",
#     help="Whether this script should be run as a test: --stop-reward must "
#     "be achieved within --stop-timesteps AND --stop-iters.",
# )
# parser.add_argument(
#     "--stop-iters", type=int, default=5000, help="Number of iterations to train."
# )
# parser.add_argument(
#     "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
# )
# parser.add_argument(
#     "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
# )
# parser.add_argument(
#     "--no-tune",
#     action="store_true",
#     help="Run without Tune using a manual train loop instead. In this case,"
#     "use PPO without grid search and no TensorBoard.",
# )
# parser.add_argument(
#     "--local-mode",
#     action="store_true",
#     help="Init Ray in local mode for easier debugging.",
# )

from swing_trader.env.env import StockEnv, Config, InitialConditions
from swing_trader.env.env import CloseVolumeState

StockEnv.set_state(CloseVolumeState)

HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
stock_config = Config(
    rollout_length=200,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn
from ray.rllib.models import ModelCatalog


class CustomTorchModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = nn.Sequential(
            SlimFC(int(np.product(obs_space.shape)), 256, activation_fn=nn.ReLU),
            SlimFC(256, 256, activation_fn=nn.ReLU),
            SlimFC(256, num_outputs, activation_fn=None)
        )

        self.value_branch = nn.Sequential(
            SlimFC(int(np.product(obs_space.shape)), 256, activation_fn=nn.ReLU),
            SlimFC(256, 1, activation_fn=None)
        )

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._last_out = self.torch_sub_model(self._last_flat_in)
        return self._last_out, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._last_flat_in is not None, "must call forward() first"
        return self.value_branch(self._last_flat_in).squeeze(1)
ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)


if __name__ == "__main__":
    # args = parser.parse_args()

    
    class args:
        local_mode = False
        no_tune = False
        stop_reward = 10
        stop_timesteps = 100000
        stop_iters = 5000
        as_test = False
        run = "PPO"
        framework = "torch"
        
    print(f"Running with following CLI options: {args}")

    ray.init(
        local_mode=args.local_mode,
        num_cpus=8,
    )

    
    
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        # or "corridor" if registered above
        .environment(StockEnv, env_config=stock_config)
        .training(model={
            "custom_model": "custom_torch_model",
            "custom_model_config": {},
        })
        .framework(args.framework)
        .rollouts(num_rollout_workers=1)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
        
    )

    

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            # stop training of the target train steps or reward are reached
            if (
                result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            # run_config=air.RunConfig(stop=stop),
            run_config=air.RunConfig(
                stop=stop,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=1,
                )
            ),
        )
        results = tuner.fit()
        
        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
