from env import *

config = Config(
    rollout_length=64,
    market="QQQ",
    min_hold=2,
    state_history_length=49
)

import gymnasium as gym
gym.logger.set_level(40)
for i in range(10000):
    print(f"Env {i}")
    env = Env(config=config)
    env.reset()
    terminated = False
    while not terminated:
        state_arr,reward,terminated,truncated,infos = env.step([0, 0])
    i += 1
