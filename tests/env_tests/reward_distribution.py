from swing_trader.env.env import *
from swing_trader.env.data.data_model import NoDataException
import random

HISTORY = 12
OBS_SPACE = HISTORY * 6 + 1
ROLLOUT_LENGTH = 100
config = Config(
    rollout_length=ROLLOUT_LENGTH,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)

StockEnv.set_state(CloseVolumeState)

import gymnasium as gym
from ray.rllib.utils import check_env
from swing_trader.configs.buy_only_MLP import BuyOnlyEnv

gym.logger.set_level(40)
i = 0
rewards = []
while True:
    print(f"Env {i}")


    env = BuyOnlyEnv(
        config=config,
        ics=InitialConditions.from_random(raise_=False, n=config["rollout_length"] + ROLLOUT_LENGTH),
    )

    if i == 0:
        result = check_env(env)
        pass

    try:
        state, _ = env.reset()
    except NoDataException as e:
        print(e)
        continue
    terminated = False
    while not terminated:
        state_arr,reward,terminated,truncated,infos = env.step([random.random()])
    i += 1
    rewards.append(reward)

    if i > 500:
        break

import matplotlib.pyplot as plt

plt.hist(rewards)
plt.show()
print()