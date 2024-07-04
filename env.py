import pandas as pd
import os
import datetime
from typing import *
from typing_extensions import TypedDict, Self
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import random




# SEED = 0
SEED = None
DATA_DIR = "data/nasdaq"
NAME_PATTERN = "{}-weekly.csv"
DT_PATTERN = "%Y-%M-%d"

if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

class Action(TypedDict):
    buy: bool
    sell: bool

    @classmethod
    def space(cls) -> Box:
        return Box(
            low=np.zeros(2),
            high=np.ones(2),
        )

    def serialize(self) -> np.array:
        return np.array([
            float(self["buy"]),
            float(self["sell"])
        ])

class State(TypedDict):
    bought: bool
    high: float
    low: float
    open: float
    close: float
    volume: float
    
    @classmethod
    def space(cls, history=0) -> Box:
        l = (history + 1) * 6
        return Box(
            low=-np.inf*np.ones(l),
            high=np.inf*np.ones(l),
        )

    @classmethod
    def serialize(cls, self) -> np.array:

        
        return np.array([
            float(self["bought"]),
            self["high"] / 1000,
            self["low"] / 1000,
            self["open"] / 1000,
            self["close"] / 1000,
            np.log10(self["volume"] + 1.01),
        ])
    
    @classmethod
    def from_serialized(cls, arr: np.array) -> Self:

        (bought, high, low, open, close, vol) = arr

        return cls({
            "bought": bool(bought),
            "high": high * 1000,
            "low": low * 1000,
            "open": open * 1000,
            "close": close * 1000,
            "volume": 10**vol
        }) 
    
    @classmethod
    def null(cls) -> Self:
        return cls({
            "bought": 0,
            "high": 0,
            "low": 0,
            "open": 0,
            "close": 0,
            "volume": -1
        })

class ICs(TypedDict):
    name: str
    date: datetime.datetime

    @classmethod
    def from_random(cls) -> Self:
        names = list(os.listdir(DATA_DIR))
        name = random.choice(names)
        
        with open(os.path.join(DATA_DIR, name)) as f:
            lines = f.readlines()

        if len(lines) < 50:
            print(f"Skipping {name}")
            return cls.from_random()

        
        cols = lines[0].replace(" ","").split(",")
        line_ind = random.randint(0, len(lines) - 1)
        line = lines[line_ind].split(",")

        record = {k: v for k, v in zip(cols, line)}
        
        if record["Open"] == 0 or record["Volume"] == 0:
           line = lines[line_ind + 1].split(",")
           record = {k: v for k, v in zip(cols, line)}


        date = record["Date"].split(" ")[0]
        try:
            date = datetime.datetime.strptime(date, DT_PATTERN)
        except ValueError:
            print(f"Skipping {name}")
            return cls.from_random()
        
        return cls({
            "name": name.replace("-weekly.csv",""),
            "date": date
        })

class Transaction(TypedDict):
    name: str
    buy_date: datetime.datetime
    sell_date: datetime.datetime
    buy_price: float
    sell_price: float
    ticks_held: int

class Config(TypedDict):
    rollout_length: int
    market: str
    min_hold: int
    state_history_length: int
    clip_reward: float

class Env(gym.Env):

    def __init__(self, config: Optional[dict] = None, ics: Optional[ICs] = None):

        # set initial conditions
        self.ics: ICs = dict()
        if ics is None:
            self._randomize_ics()
        else:
            self.ics = ics

        
        # set config values
        config = {} if config is None else config
        self.config = config
        self.rollout_length = config.get("rollout_length") if "rollout_length" in config else 50
        self.market = config.get("market") if "market" in config else "QQQ"
        self.min_hold = config.get("min_hold") if "min_hold" in config else 1
        self.state_history_length = config.get("state_history_length") if "state_history_length" in config else 0
        self.clip_reward = config.get("clip_reward") if "clip_reward" in config else 1

        # Gym components
        self.action_space = Action.space()
        self.observation_space = State.space(self.state_history_length)


        # set internals
        self.reset(randomize = False)
        self.market_df, self.market_records, self.market_raw_data, self.market_raw_records, self.market_raw_index = self._load_market()

    def _randomize_ics(self):
        """
        Randomizes initial conditions
        """
        self.ics = ICs.from_random()
    
    def _load_data(self) -> Tuple[List[Dict], pd.DataFrame]:

        name = self.ics["name"]
        date = self.ics["date"]

        # load df by name
        fname = NAME_PATTERN.format(name)
        data: pd.DataFrame = pd.read_csv(os.path.join(DATA_DIR, fname))
        
        # filter df by date
        date_data: pd.Series = data["Date"]
        date_df = date_data.str.split(" ", expand=True) # ???
        data["Date"] = date_df[0]
        data["Date"] = pd.to_datetime(data["Date"])

        raw_data = data
        
        data = data[data["Date"] > date]
        raw_index = data.index[0]

        records = data.to_dict("records")
        raw_records = raw_data.to_dict("records")
        return data, records, raw_data, raw_records, raw_index

    def _load_market(self) -> Tuple[List[Dict], pd.DataFrame]:
        name = self.ics["name"]
        self.ics["name"] = self.market
        data, records, raw_data, raw_records, raw_index = self._load_data()
        self.ics["name"] = name
        return data, records, raw_data, raw_records, raw_index

    def reset(self, *, seed=None, options=None, randomize: bool = True):
        if randomize:
            self._randomize_ics()
        self.df, self.records, self.raw_df, self.raw_records, self.raw_index = self._load_data()
        self.ind = 0
        self.holding = False
        self.buy_date = None
        self.buy_price = None
        self.sell_date = None
        self.sell_price = None
        self.multiplier = 1
        self.ticks_holding = 0
        self.history = []

        # return State.serialize(self.state()), {"env_state": "reset"}
        return self.state_arr(), {}
        
    def step(
            self,
            action: float
    ):

        self.dump("log.json")
        buy, sell = action
    
        if all([
            self.ind >= len(self.records) - 1,
            buy > 0.5 ,
            not self.holding,
            self.records[self.ind + 1]["Open"] > 0
        ]):
            
            self.holding = True
            self.buy_date = self.records[self.ind]["Date"]
            self.buy_price = self.records[self.ind+1]["Open"]
            

        elif sell > 0.5 and \
                self.holding and \
                self.ticks_holding > self.min_hold:

            self.sell_date = self.records[self.ind+1]["Date"]
            self.sell_price = self.records[self.ind+1]["Open"]    
            multiplier = self.sell_price / self.buy_price
            self.multiplier *= multiplier
            
            self.history.append(Transaction({
                "name": self.ics["name"],
                "buy_date": self.buy_date,
                "sell_date": self.sell_date,
                "buy_price": self.buy_price,
                "sell_price": self.sell_price,
                "ticks_held": self.ticks_holding
            }))
            
            # reset variables
            self.holding = False
            self.buy_date = None
            self.buy_price = None
            self.sell_date = None
            self.sell_price = None
            self.ticks_holding = 0

        elif self.holding:
            self.ticks_holding += 1 

        # step the sim
        self.ind += 1

        terminated = False
        truncated = False
        reward = 0
        infos = {"history": self.history}
        if any([
            self.ind >= len(self.records) - 1,
            self.ind >= self.rollout_length - 1
        ]):
            terminated = True
            truncated = True if self.ind == len(self.records) - 1 else False
            if self.holding:
                cur_price = self.records[self.ind]["Close"]
                multiplier = cur_price / self.buy_price
                self.multiplier *= multiplier
                self.history.append(Transaction({
                    "name": self.ics["name"],
                    "buy_date": self.buy_date,
                    "sell_date": self.records[self.ind]["Date"],
                    "buy_price": self.buy_price,
                    "sell_price": cur_price,
                    "ticks_held": self.ticks_holding,
                }))

            reward = self.reward()

        state_arr = self.state_arr()

        return (
            state_arr,
            reward,
            terminated,
            truncated,
            infos,
        )
    
    def market_multiplier(self) -> float:
        market_open = self.market_records[0]["Open"]
        market_close = self.market_records[self.ind]["Close"]
        return market_close / market_open

    def reward(self) -> float:
        """
        Our return - market return - 2 percent fees
        """

        reward =  self.multiplier - self.market_multiplier() - 0.02
        if self.clip_reward:
            return min([reward, self.clip_reward])
    
    def state(self) -> State:
        record = self.records[self.ind]
        state: State = State({
            "high": record["High"],
            "low": record["Low"],
            "open": record["Open"],
            "close": record["Close"],
            "volume": record["Volume"],
            "bought": self.holding
        })
        
        return state
    
    def state_arr(self) -> np.array:
        state_arr = State.serialize(self.state())
        if self.state_history_length > 0:
            state_history_arr = np.concatenate([
                State.serialize(s) for s in self.state_history(self.state_history_length)
            ])
            state_arr = np.concatenate([state_history_arr, state_arr])
        
        return state_arr

    def state_history(self, length: int) -> list[State]:
        og_ind = self.ind
        states = []
        for ind in range(self.ind - length, self.ind):
            self.ind = ind
            if ind < 0:
                state = State.null()
            else:
                state = self.state()
            states.append(state)
        
        self.ind = og_ind
        return states
    
    def start_date(self) -> datetime.datetime:
        return self.ics["date"]
    
    def end_date(self) -> datetime.datetime:
        if len(self.records) <= self.rollout_length:
            return self.records[-1]["Date"]
        
        return self.records[self.rollout_length]["Date"]

    def dump(self, path: str):
        """
        Dumps to json
        """
        import json
        f = open(path, "w")
        json.dump({
            "name": self.ics["name"],
            "start_date": self.ics["date"],
            "end_date": self.end_date(),
            # "ics": self.ics,
            "config": self.config,
            "history": self.history,
            "multiplier": self.multiplier,
            "reward": self.reward(),
            "market_multiplier": self.market_multiplier(),
            "raw_index": self.raw_index
        }, f, sort_keys=True, indent=4, default=str)
        f.close()