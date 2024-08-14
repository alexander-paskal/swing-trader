import numpy as np
import pandas as pd
import gymnasium as gym
from typing import *
from typing_extensions import Self

from swing_trader.env.data import DataModel
from swing_trader.env.utils import Date
from swing_trader.env.states.base import State

__all__ = ['DWMState']

class DWMState(State):
    timeframe: str = "daily"
    ticks: int = 5
    column: str = "Close"
    divide: float = 1.
    normalize: bool = False  # normalize to 0-1
    zero_center: bool = False
    log: bool = False
    daily: pd.DataFrame
    weekly: pd.DataFrame
    monthly: pd.DataFrame

    def __init__(self, date: Date, data: DataModel, cfg: Optional[Dict] = None):
        
        if cfg is None:
            cfg = {}

        if "ticks" in cfg:  # number of ticks for each timeframe
            self.ticks = cfg["ticks"]
        
        if "normalize" in cfg:
            self.normalize = cfg["normalize"]
        
        if "column" in cfg:
            self.column = cfg["column"]

        if "log" in cfg:
            self.log = cfg["log"]
        
        if "zero_center" in cfg:
            self.zero_center = cfg["zero_center"]
        
        if "divide" in cfg:
            self.divide = cfg["divide"]

        self.data = data
        self.date = date

        # get daily
        daily = self.data.access(freq="daily", date=date, length=self.ticks)

        # get weekly
        weekly = self.data.access(freq="weekly", date=date, length=self.ticks + 1)
        weekly = weekly.iloc[:-1, :]

        # get monthly
        monthly = self.data.access(freq="monthly", date=date, length=self.ticks + 1)
        monthly = monthly.iloc[:-1, :]

        self.daily = daily
        self.monthly = monthly
        self.weekly = weekly

    @classmethod
    def from_data(cls, date: Date, data: DataModel, **kwargs):
        
        s = cls.__new__()
        s.__init__(date, data, **kwargs)
        return s
    
    def null(self) -> Self:
        return np.zeros(self.ticks * 3)
    
    def space(self) -> gym.Space:
        raise NotImplementedError
    
    @property
    def array(self) -> np.array:
        
        arr = np.zeros(self.ticks * 3)

        eday = self.ticks
        arr[
            eday-self.daily.shape[0]:eday
        ] = self.daily[self.column].array.astype(float)

        eweek = self.ticks * 2
        arr[
            eweek-self.weekly.shape[0]:eweek
        ] = self.weekly[self.column].array.astype(float)

        emonth = self.ticks * 3
        arr[
            emonth-self.monthly.shape[0]:emonth
        ] = self.monthly[self.column].array.astype(float)

        old_arr = arr
        if self.log:
            mask = arr > 0 
            arr[mask] = np.log10(arr[mask])
        
        if self.divide:
            arr = arr / self.divide

        if self.normalize:
            arr = arr / np.max(arr)

        if self.zero_center:
            arr = arr - np.mean(arr)

        
        if np.any(np.isnan(arr)):
            raise ValueError("nans in state")

        return arr
