from swing_trader.env.states import DWMState, CompositeState, State, CloseVolumeState
from swing_trader.env.rewards import Reward, PerformanceDifference
from swing_trader.env.actions import Action, BuySellAction, BuySellSingleAction
from swing_trader.env.utils import Date, weekdays_after, BuyEvent, SellEvent
from swing_trader.env.data.data_model import DataModel

import numpy as np
from typing import *
from typing_extensions import TypedDict, Self
import gymnasium as gym

import random
from dataclasses import dataclass
import os

__all__ = [
    'BuyEvent',
    'SellEvent',
    'Reward',
    'Action',
    'Config',
    'StockEnv',
    'InitialConditions',
    'CloseVolumeState',
    'ConfigError'
]


class Config(TypedDict):
    rollout_length: int
    market: str
    min_hold: int
    state_history_length: int
    clip_reward: float


DATA_DIR = "data/daily"

class InitialConditions(TypedDict):
    name: str
    date: Date

    @classmethod
    def from_random(cls, n: int = 50, raise_: bool = True) -> Self:
        names = list(os.listdir(DATA_DIR))
        name = random.choice(names)
        
        # pick a random ticker
        ticker = name.split("-")[0]

        # load the data for that ticker
        try:
            data = DataModel(ticker, freqs=['daily', 'weekly', 'monthly'])
        except Exception as e:
            if raise_:
                raise e
            else:
                print(f"Skipping {name}")
                return cls.from_random(n, raise_=raise_)

        min_date, max_date = data.get_date_bounds()

        daily = data.daily[data.daily.index > min_date.as_timestamp]
        daily = daily[daily.index < max_date.as_timestamp].iloc[:-n, :]

        if daily.empty:
            print(f"Skipping {name}")
            return cls.from_random(n, raise_=raise_)
        
        return {
            "name": ticker,
            "date": Date(random.choice(daily.index))
        }


class ConfigError(Exception):
    pass

class StockEnv(gym.Env):

    state_cls: Type[State] = CloseVolumeState
    action_cls: Type[Action] = BuySellSingleAction
    reward_cls: Type[Reward] = PerformanceDifference

    def __init__(
            self,
            config: Config, 
            *,
            ics: Optional[InitialConditions] = None, 
            
        ):
        self.config: Config = config
        if ics is None:
            ics = InitialConditions.from_random(n=config["rollout_length"], raise_ = False)
        self.ics: InitialConditions = ics
        
        self.data: DataModel = None
        self.cur_date: Date = None
        self.start_date: Date = None
        self.end_date: Date = None
        self.n_steps: int = 0
        self.history: List[Union[BuyEvent, SellEvent]] = []
        self.reward_history: List[float] = []
        self.action_history: List[Action] = []
        self.state_history: List[State] = []
        self.performance_history: List[float] = []
        self.date_history: List[Date] = []
        self.price_history: List[float] = []
        self.is_holding: bool = False

        # self.set_state(state_cls)
        # self.set_action(action_cls)
        # self.set_reward(reward_cls)

        if "action_space" not in config or "observation_space" not in config:
            raise ConfigError("config needs action or observation space")
        
        self.action_space = gym.spaces.Box(low=0., high=1., shape=(config.get("action_space"),), dtype=float)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=-np.inf, shape=(config.get("observation_space"), ), dtype=float)

    @classmethod
    def set_state(cls, state_cls: TypeVar):
        assert hasattr(state_cls, "timeframe"), "States must denote a timeframe for updates"
        cls.state_cls = state_cls

    @classmethod
    def set_action(cls, action_cls: TypeVar):
        cls.action_cls = action_cls
    
    @classmethod
    def set_reward(cls, reward_cls: TypeVar):
        cls.reward_cls = reward_cls
    
    def reset(self, *, seed=None, options=None, randomize: bool = True):
        if randomize:
            self._randomize_ics()
        
        self.data: DataModel = self._load_data()  # TODO

        self.cur_date = Date(self.ics["date"])
        self.start_date = Date(self.ics["date"])
        self.end_date = self.data.get_n_ticks_after(self.state_cls.timeframe, self.start_date, self.config["rollout_length"])
        self.is_holding = False
        self.history = []
        self.n_steps = 0

        # return State.serialize(self.state()), {"env_state": "reset"}
        return self.state, {}

    def _randomize_ics(self):
        self.ics = InitialConditions.from_random(n=self.config["rollout_length"], raise_ = False)
    
    def _load_data(self) -> DataModel:
        return DataModel(self.ics["name"], freqs=['daily', 'weekly', 'monthly'])

    @property
    def name(self):
        return self.ics["name"]

    def step(self, action: np.array):

        self._process_action(action)
        self._step_sim()

        self.terminated = self._check_if_finished()
        self.truncated = self.terminated and self.n_steps < self.config["rollout_length"]
        self.infos = {}

        self.state_history.append(self.state)
        self.action_history.append(action)
        self.reward_history.append(self.reward)
        self.performance_history.append(self.performance)
        self.date_history.append(self.cur_date)
        self.price_history.append(self.data.get_price_on_close(self.cur_date))

        return (
            self.state,
            self.reward,
            self.terminated,
            self.truncated,
            self.infos
        )

    def _process_action(self, action: np.array):
        """Processes a numpy action into the state"""
        action = self.action_cls.from_array(action)

        if action.buy:
            if self.is_holding:  # duplicate buy
                return
            self.is_holding = True
            self.history.append(BuyEvent(
                date=self.cur_date,
                price=self.data.get_price_on_open(self.cur_date)
            ))

        if action.sell:
            if not self.is_holding:  # duplicate sell
                return
            
            self.is_holding = False
            self.history.append(SellEvent(
                date=self.cur_date,
                price=self.data.get_price_on_open(self.cur_date)
            ))
    
    def _step_sim(self):
        """Processes a sim step. Increments the data"""
        self.cur_date = self.data.get_next_tick(self.state_cls.timeframe, self.cur_date)
        self.n_steps += 1

    def _check_if_finished(self):
        """
        Check if the sim has reached termination. This occurs if:

            - the current date of data is the last day
            - the rollout length has been reached

        """
        if self.cur_date == self.end_date:
            return True
        
        if self.cur_date == self.data.end_date():
            return True
        
        if self.n_steps == self.config["rollout_length"]:
            return True
        return False
    
    @property
    def state(self) -> np.array:
        """
        Returns a state value.
        The state is:

            [STATE_ARR, holding] where holding is either 0 or 1
        
        """

        state_arr = self.state_cls(self.cur_date, self.data, self.config["state_history_length"]).array
        state_arr = np.concatenate([
            state_arr, np.array([float(self.is_holding)]),
        ], dtype=np.float32)
        return state_arr
    
    @property
    def reward(self) -> float:
        """
        Calculates a reward based on the history of the environment
        """
        reward = self.reward_cls(**{
            "cur_date": self.cur_date,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "history": self.history,
            "reward_history": self.reward_history,
            "action_history": self.action_history,
            "state_history": self.state_history,
            "performance_history": self.performance_history,
            "date_history": self.date_history,
            "is_holding": self.is_holding,
            "is_finished": self._check_if_finished(),
            "ics": self.ics,
            "data": self.data
        })

        return reward.value()
    
    @property
    def performance(self) -> float:
        """
        Computes the performance based on the trade history
        """
        history = self.history.copy()
        if len(self.history) % 2 == 1:
            current_price = self.data.access("daily", self.cur_date, length=1).loc[self.cur_date.as_timestamp, "Close"]
            history.append(SellEvent(
                date=self.cur_date,
                price=current_price
            ))
        
        multiplier = 1
        for i in range(0, len(history), 2):
            buy = history[i]
            sell = history[i+1]
            assert isinstance(buy, BuyEvent), "unordered buy"
            assert isinstance(sell, SellEvent), "unordered sell"
            multiplier *= sell.price / buy.price

        return multiplier - 1


    def render(self):
        """
        Plots the performance
        """
        import matplotlib.pyplot as plt
        plt.grid()
        plt.tight_layout()
        fig, axs = plt.subplots(2, 2)
        axs = axs.flatten()

        dates = [d.as_datetime for d in self.date_history]
        axs[0].bar(dates, self.reward_history)
        axs[0].set_title("Rewards")
        axs[0].set_xticklabels([])
        axs[0].grid()
        axs[1].plot(dates, self.performance_history)
        axs[1].set_title("Performance")
        axs[1].set_xticklabels([])
        axs[1].grid()
        axs[2].plot(dates, self.price_history)
        axs[2].set_title("Price")
        axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=60)
        axs[2].grid()

        cum_rewards = [
            sum(self.reward_history[:i+1]) for i in range(len(dates))
        ]
        axs[3].plot(dates, cum_rewards)
        axs[3].set_title("Reward")
        axs[3].set_xticklabels(axs[3].get_xticklabels(), rotation=60)
        axs[3].grid()
        
        for event in self.history:
            if isinstance(event, BuyEvent):
                axs[2].scatter([event.date], [event.price], c="green")
            elif isinstance(event, SellEvent):
                axs[2].scatter([event.date], [event.price], c="red")

        plt.show()

    def print_out(self):
        print(f"""
    cur_date:  {self.cur_date}
    end_date:  {self.end_date}
    n_steps:   {self.n_steps}
    history:   {self.history}
    reward:    {self.reward}
    next_open: {self.data.get_price_on_close(self.cur_date)}
    is_holding:{self.is_holding}
    state:     {None}
         """)