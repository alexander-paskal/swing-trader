from ray.rllib.core.rl_module.rl_module import RLModule
import os

rl_module = RLModule.from_checkpoint(
    "C:/Users/alexc/ray_results/PPO_2024-07-07_16-46-55/PPO_Env_0c24f_00000_0_2024-07-07_16-46-55/checkpoint_000000/policies/default_policy"
)

print()
