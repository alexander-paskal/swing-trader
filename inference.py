import ray
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
import gym
import numpy as np


from swing_trader.env.env import StockEnv, Config, InitialConditions
from train import AttentionNetwork
import datetime

# Register the custom environment
register_env("CustomEnv-v0", lambda config: StockEnv(config))

# Initialize Ray with minimal resources
ray.init(num_cpus=1, num_gpus=0)

# Load the trained model
checkpoint_path = "C:/Users/alexc/ray_results/PPO_2024-07-27_17-42-35/PPO_StockEnv_23460_00000_0_2024-07-27_17-42-35/checkpoint_000010"
loaded_algo = Algorithm.from_checkpoint(checkpoint_path)

# get model
policy = loaded_algo.get_policy()
# breakpoint()
model = policy.model
# Set up the environment for inference

# from env import Config
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
ics = InitialConditions(
    name="MSFT",
    date=datetime.datetime(2013,1,1)
)
env = StockEnv(stock_config)

from ui.mpl import MPLCore

from swing_trader.env.utils import log_env
import time


summaries = []

for episode in range(30):
    obs, *_ = env.reset(randomize=True)
    done = False
    episode_reward = 0
    steps = 0
    while not done:
        action = loaded_algo.compute_single_action(obs)

        obs, reward, done, *_ = env.step(action)
        time.sleep(0.1)
        steps += 1
    
    # log_env(env)
        # episode_reward += reward
    # print(f"Episode {episode + 1} reward: {episode_reward}")
    summaries.append({
        "start_date": env.start_date,
        "ticker": env.data.ticker,
        "buy_and_hold": env.data.buy_and_hold(env.start_date, env.cur_date),
        "performance": env.reward,
        "history": env.history
    })
    print(summaries[-1])


print("Averages")
vals = [s["performance"] - s["buy_and_hold"] for s in summaries]
print("Average:", np.mean(vals))
print("Variance:", np.var(vals))
# for summary in summaries:
#     print(summary)

# Shut down Ray
ray.shutdown()

# history = env.history
# buy_dates = {t["buy_date"] for t in env.history}
# sell_dates = {t["sell_date"] for t in env.history}

# print(history)

#### uncomment when ui is working
# env.reset(randomize=False)



# ui = MPLCore(env)
# import matplotlib.pyplot as plt
# i = 0
# while i < steps:
#     if env.cur_date in buy_dates:
#         ui.buy()
#     if env.cur_date in sell_dates:
#         ui.sell()
    
#     ui.step()
#     i += 1
#     plt.pause(1)

# plt.show()
# print()
