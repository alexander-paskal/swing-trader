from swing_trader.env.env import *
from swing_trader.env.data.data_model import NoDataException

HISTORY = 12
OBS_SPACE = HISTORY * 6 + 1
config = Config(
    rollout_length=10,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)

StockEnv.set_state(CloseVolumeState)

import gymnasium as gym
# from ray.rllib.utils import check_env
from stable_baselines3.common.env_checker import check_env


gym.logger.set_level(40)
i = 0
while True:
    print(f"Env {i}")


    env = StockEnv(
        config=config,
        ics=InitialConditions.from_random(raise_=False, n=config["rollout_length"] + 10),
        # state_cls=CloseVolumeState
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
        state_arr,reward,terminated,truncated,infos = env.step([0, 0])
    i += 1
