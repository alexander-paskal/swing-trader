from swing_trader.configs.buy_only_MLP import BuyOnlyEnv
from swing_trader.env.env import *
from swing_trader.env.data.data_model import NoDataException
import gymnasium as gym
import random



HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
config = Config(
    rollout_length=250,
    market="QQQ",
    min_hold=2,
    state_history_length=HISTORY,
    action_space = 2,
    observation_space = OBS_SPACE
)
StockEnv.set_state(CloseVolumeState)

def random_action():
    return [random.random()]

def dont_buy():
    return [0]

def log_env(env: StockEnv):
    print(f"""
    cur_date:  {env.cur_date}
    end_date:  {env.end_date}
    n_steps:   {env.n_steps}
    history:   {env.history}
    reward:    {env.reward}
    next_open: {env.data.get_price_on_close(env.cur_date)}
    is_holding:{env.is_holding}
    state:     {env.state}
    """)

gym.logger.set_level(40)



while True:
    env = BuyOnlyEnv(
        config=config,
        ics=InitialConditions.from_random(raise_=False, n=config["rollout_length"] + 10),
        # state_cls=CloseVolumeState
    )
    try:
        state, _ = env.reset()
    except NoDataException as e:
        print(e)
        continue
    break

terminated = False

log_env(env)
while True:
    
    # action = random_action()
    action = dont_buy()
    print(action)
    state_arr,reward,terminated,truncated,infos = env.step(action)
    log_env(env)
    
    if terminated:
        break

