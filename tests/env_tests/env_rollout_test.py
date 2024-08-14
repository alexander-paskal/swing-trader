from swing_trader.env.env import *
from swing_trader.env.data.data_model import NoDataException
import gymnasium as gym
import random



HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
config = Config(
    rollout_length=50,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)
# StockEnv.set_state(CloseVolumeState)

def random_action():
    return [random.random() - 0.5]

class RandomActionWalk:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.action = 0
        self.steps = 0
        
    def step(self):
        self.action += (random.random() - 0.5) * 0.1
        self.steps += 1
        if self.steps > 10:
            self.reset()

def log_env(env: StockEnv):
    print(f"""
    cur_date:  {env.cur_date}
    end_date:  {env.end_date}
    n_steps:   {env.n_steps}
    history:   {env.history}
    reward:    {env.reward}
    next_open: {env.data.get_price_on_close(env.cur_date)}
    is_holding:{env.is_holding}
    state:     {None}
    """)

gym.logger.set_level(40)



while True:
    env = StockEnv(
        config=config,
        ics=InitialConditions.from_random(raise_=False, n=config["rollout_length"] + 10),
    )
    try:
        state, _ = env.reset()
    except NoDataException as e:
        print(e)
        continue
    break

terminated = False

log_env(env)
i = 0
walk = RandomActionWalk()
while True:
    
    # if i % 20 == 0:
    #     action = random_action()
    # else:
    #     action = [-1, -1]
    walk.step()

    print("Action:", walk.action)
    state_arr,reward,terminated,truncated,infos = env.step([walk.action])
    log_env(env)
    
    if terminated:
        break

    i += 1

import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, 1)
fig.tight_layout()
dates = [d.as_datetime for d in env.date_history]
axs[0].bar(dates, env.reward_history)
axs[0].set_title("Rewards")
axs[0].set_xticklabels([])
axs[1].plot(dates, env.performance_history)
axs[1].set_title("Performance")
axs[1].set_xticklabels([])
axs[2].plot(dates, env.price_history)
axs[2].set_title("Price")
axs[2].set_xticklabels(axs[1].get_xticklabels(), rotation=60)
plt.show()
