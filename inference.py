import ray
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
import gym
import numpy as np


from env import Env


# Register the custom environment
register_env("CustomEnv-v0", lambda config: Env(config))

# Initialize Ray with minimal resources
ray.init(num_cpus=1, num_gpus=0)

# Load the trained model
checkpoint_path = "C:/Users/alexc/ray_results/PPO_2024-07-08_06-05-24/PPO_Env_bd224_00000_0_2024-07-08_06-05-24/checkpoint_000000"
loaded_algo = Algorithm.from_checkpoint(checkpoint_path)

# get model
policy = loaded_algo.get_policy()
breakpoint()
model = policy.model
# Set up the environment for inference
env = Env()

# Perform inference
num_episodes = 10
for episode in range(num_episodes):
    obs, *_ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = loaded_algo.compute_single_action(obs)
        obs, reward, done, *_ = env.step(action)
        episode_reward += reward
    print(f"Episode {episode + 1} reward: {episode_reward}")

# Shut down Ray
ray.shutdown()


print()
