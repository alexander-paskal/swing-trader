from swing_trader.env.env import *
from swing_trader.env.data.data_model import NoDataException
import gymnasium as gym
import random



HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
config = Config(
    rollout_length=300,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)
# StockEnv.set_state(CloseVolumeState)

gym.logger.set_level(40)


# find a valid env

env = StockEnv(
    config=config,
    ics=InitialConditions(
        name="MSFT",
        date="2018-02-01"
    ),
)
state, _ = env.reset(randomize=False)
terminated = False
env.print_out()


ACTIONS = [
    ("2018-02-05", 1),
    ("2018-03-13", -1),
    ("2018-03-22", 1),
    ("2018-10-01", -1),
    ("2018-12-22", 1),
]

i = 0
j = 0
while True:
    
    action = 0
    try:
        next_action_date, next_action_value = ACTIONS[j]
    except IndexError:
        # print("index error")
        next_action_date = "9999-12-31"
    
    if env.cur_date >= next_action_date:
        print(f"Action: {env.cur_date}, {next_action_value}")
        action = next_action_value
        j += 1

    state_arr,reward,terminated,truncated,infos = env.step([action])
    # log_env(env)
    
    if terminated:
        break

    i += 1


env.render()
