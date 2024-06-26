import pandas as pd
import os
import datetime
from typing import *
from typing_extensions import TypedDict, Self
import gym
import gymnasium
import numpy as np
import random




SEED = 0
DATA_DIR = "data"
NAME_PATTERN = "{}-weekly.csv"
DT_PATTERN = "%Y-%M-%D"

if SEED:
    random.seed(SEED)
    np.random.seed(SEED)

class Action(TypedDict):
    buy: bool
    sell: bool

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
    
    def serialize(self) -> np.array:

        return np.array([
            float(self["bought"]),
            self["high"] / 1000,
            self["low"] / 1000,
            self["open"] / 1000,
            self["close"] / 1000,
            np.log10(self["volume"]),
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

class ICs(TypedDict):
    name: str
    date: datetime.datetime

    @classmethod
    def from_random(cls) -> Self:
        names = list(os.listdir(DATA_DIR))
        name = random.choice(names)
        
        with open(os.path.join(DATA_DIR, name)) as f:
            lines = f.readlines()

        cols = lines[0].replace(" ","").split(",")
        line = random.choice(lines).split(",")

        record = {k: v for k, v in zip(cols, line)}
        
        date = record["date"].split(" ")[0]
        date = datetime.datetime.strptime(date, DT_PATTERN)

        return cls({
            "name": name.replace("-weekly.csv",""),
            "date": date
        })

class Env(gym.Env):

    def __init__(self, config: Optional[dict] = None, ics: Optional[ICs] = None):

        # set initial conditions
        self.ics: ICs = dict()
        if ics is None:
            self._randomize_ics()
        else:
            self.ics = ics

        # set config values
        self.rollout_length = config.get("rollout_length") if "rollout_length" in config else 50
        self.market = config.get("market") if "market" in config else "QQQ"
        self.min_hold = config.get("min_hold") if "min_hold" in config else 1
        

        # set internals
        self.reset(randomize = False)
        self.market_df, self.market_records = self._load_market()

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
        data: pd.DataFrame = pd.read_csv(fname)
        
        # filter df by date
        date_data: pd.Series = data["data"]
        data["Date"] = date_data.str.split(" ")[0]  # ???
        data["Date"] = data["Date"].dt.to_timestamp(how=DT_PATTERN)
        data = data[data["Date"] > date]

        records = data.to_dict("records")
        return data, records

    def _load_market(self) -> Tuple[List[Dict], pd.DataFrame]:
        name = self.ics["name"]
        self.ics["name"] = self.market
        data, records = self._load_data()
        self.ics["name"] = name
        return data, records

    def reset(self, *, randomize: bool = False):
        if randomize:
            self._randomize_ics()
        self.df, self.records = self._load_data()
        self.ind = 0
        self.holding = False
        self.buy_date = None
        self.sell_date = None
        self.multiplier = 1
        self.ticks_holding = 0
        
    def step(
            self,
            action: float
    ):

        buy, sell = action

        if buy > 0.5 and not self.holding:
            self.holding = True

            # TODO set buy variables
        
        elif sell > 0.5 and \
                self.holding and \
                self.ticks_holding > self.min_hold:

            self.holding = False
            self.ticks_holding = 0
            self.sell_date = self.records[self.ind]["Date"]

        elif self.holding:
            self.ticks_holding += 1 

        # step the sim
        self.ind += 1

        record = self.records[self.ind]
        state: State = State({
            "high": record["High"],
            "low": record["Low"],
            "open": record["Open"],
            "close": record["Close"],
            "volume": record["Volume"],
            "bought": self.holding
        })

        return 