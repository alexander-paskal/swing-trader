from swing_trader.env.states import DWMState, CompositeState, State
from swing_trader.env.utils import Date, weekdays_after
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


@dataclass
class BuyEvent:
    date: Date
    price: float

@dataclass
class SellEvent:
    date: Date
    price: float
    days_held: int = 0

class Reward:
    def __init__(self, **kwargs):
        # "cur_date": self.cur_date,
        # "start_date": self.start_date,
        # "end_date": self.end_date,
        # "history": self.history,
        # "is_holding": self.is_holding,
        # "ics": self.ics,
        # "data": self.data


        if "data" not in kwargs:
            raise ValueError("Reward needs data")
        
        if "cur_date" not in kwargs:
            raise ValueError("Reward needs cur_date")
        
        if "history" not in kwargs:
            raise ValueError("Reward needs history")
        
        if "is_finished" not in kwargs:
            raise ValueError("Reward needs is_finished")
        
        if "is_holding" not in kwargs:
            raise ValueError("Reward needs is_holding")

        self.data: DataModel = kwargs["data"]
        self.cur_date: Date = kwargs["cur_date"]
        self.is_finished: Date = kwargs["is_finished"]
        self.history: List[Union[BuyEvent, SellEvent]] = kwargs["history"]
        self.is_holding: bool = kwargs["is_holding"]

    def value(self):
        if not self.is_finished:
            return 0
        
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
        
        if multiplier > 3:
            return 3
        return multiplier


class Config(TypedDict):
    rollout_length: int
    market: str
    min_hold: int
    state_history_length: int
    clip_reward: float


class Action:
    buy: bool
    sell: bool
    def __init__(self, buy: bool = False, sell: bool = False):
        self.buy = buy
        self.sell = sell
    
    def serialize(self) -> np.array:
        return np.array([
            float(self.buy),
            float(self.sell)
        ])

    @classmethod
    def from_array(cls, arr: np.array) -> Self:
        d = {"buy": False, "sell": False}

        if arr[0] > 0.5:
            d["buy"] = True
            return cls(**d)
        
        if arr[1] > 0.5:
            d["sell"] = True
            return cls(**d)

        return cls(**d)

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



class CloseVolumeState(CompositeState):

    timeframe = "daily"

    def __init__(self, date: Date, data: DataModel, ticks: int):

        s1 = DWMState(date, data, {"column": "Close", "normalize": True, "ticks": ticks})
        s2 = DWMState(date, data, {"column": "Volume", "log": True, "divide": 10, "ticks": ticks})
        
        super().__init__(s1, s2)

class ConfigError(Exception):
    pass

class StockEnv(gym.Env):

    state_cls: Type[State] = CloseVolumeState
    action_cls: Type[Action] = Action
    reward_cls: Type[Reward] = Reward

    def __init__(
            self,
            config: Config, 
            *,
            ics: Optional[InitialConditions] = None, 
            state_cls: Type[State] = State,
            action_cls: Type[Action] = Action, 
            reward_cls: Type[Reward] = Reward,
            
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
        reward = Reward(**{
            "cur_date": self.cur_date,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "history": self.history,
            "is_holding": self.is_holding,
            "is_finished": self._check_if_finished(),
            "ics": self.ics,
            "data": self.data
        })

        return reward.value()


