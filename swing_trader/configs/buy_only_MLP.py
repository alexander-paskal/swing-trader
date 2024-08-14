from swing_trader.configs.config import RunConfig
from swing_trader.env.env import StockEnv, Config
from swing_trader.env.actions import Action
from swing_trader.env.rewards import Reward, Performance, BuyAndHold
from swing_trader.model.mlp import CustomTorchModel
import numpy as np
from typing_extensions import Self


class BuyOnlyAction(Action):
    buy: bool
    sell: bool

    def __init__(self, buy: bool):
        self.buy = buy
        self.sell = False

    def serialize(self) -> np.array:
        if self.buy:
            return np.array([1])
        else:
            return np.array([0])

    @classmethod
    def from_array(cls, arr: np.array) -> Self:
        if arr[0] > 0.5:
            return cls(True)
        
        return cls(False)

class PerformanceVsBuyHoldReward(Reward):
    def __init__(self, **kwargs):
        self.performance = Performance(**kwargs)
        self.buy_and_hold = BuyAndHold(**kwargs)

        self.is_holding = kwargs['is_holding']
    
    def value(self) -> float:
        
        # if buy, reward is buy_and_hold - 1
        # if sell, reward is 1 - buy_and_hold
        if self.is_holding:
            reward = self.performance.value() - 1
        
        else:
            reward = self.buy_and_hold.value() - 1

        reward = min([reward, 5])
        reward = max([reward, -5])
        return reward

class BuyOnlyEnv(StockEnv):
    """Only one buy permitted. Reward computed for N days after"""
    
    action_cls = BuyOnlyAction
    reward_cls = PerformanceVsBuyHoldReward

    def _step_sim(self):
        """Jumps right to the end date"""
        self.cur_date = self.end_date
        self.n_steps += 1

    def _check_if_finished(self):
        """Finishes immediately"""
        if self.n_steps > 0:
            return True
        
        return super()._check_if_finished()



HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
config = RunConfig(
    model=CustomTorchModel,
    model_config={},
    env=BuyOnlyEnv,
    env_config=Config(
        rollout_length=200,
        market="QQQ",
        min_hold=2,
        state_history_length=HISTORY,
        action_space = 1,
        observation_space = OBS_SPACE
    ),
    num_cpus=8,
    num_rollout_workers=7,
)