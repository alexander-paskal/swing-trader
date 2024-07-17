import ray
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
import gym
import numpy as np


from env import Env, Config, ICs
from train import CustomTorchModel
import datetime

# Register the custom environment
register_env("CustomEnv-v0", lambda config: Env(config))

# Initialize Ray with minimal resources
ray.init(num_cpus=1, num_gpus=0)

# Load the trained model
checkpoint_path = "C:/Users/alexc/ray_results/PPO_2024-07-14_22-43-34/PPO_Env_07f73_00000_0_2024-07-14_22-43-35/checkpoint_000002"
loaded_algo = Algorithm.from_checkpoint(checkpoint_path)

# get model
policy = loaded_algo.get_policy()
# breakpoint()
model = policy.model
# Set up the environment for inference

# from env import Config
stock_config = Config(
    rollout_length=64,
    market="QQQ",
    min_hold=2,
    state_history_length=49
)
ics = ICs(
    name="MSFT",
    date=datetime.datetime(2013,1,1)
)
env = Env(stock_config)

from ui.mpl import MPLCore

ui = MPLCore(env)
import time



for episode in range(1):
    obs, *_ = env.reset(randomize=False)
    done = False
    episode_reward = 0
    steps = 0
    while not done:
        action = loaded_algo.compute_single_action(obs)

        obs, reward, done, *_ = env.step(action)
        time.sleep(0.1)
        steps += 1
    

        # episode_reward += reward
    # print(f"Episode {episode + 1} reward: {episode_reward}")

# Shut down Ray
ray.shutdown()

history = env.history
buy_dates = {t["buy_date"] for t in env.history}
sell_dates = {t["sell_date"] for t in env.history}


env.reset(randomize=False)


ui = MPLCore(env)
import matplotlib.pyplot as plt
i = 0
while i < steps:
    if env.cur_date in buy_dates:
        ui.buy()
    if env.cur_date in sell_dates:
        ui.sell()
    
    ui.step()
    i += 1
    plt.pause(1)

plt.show()
print()
