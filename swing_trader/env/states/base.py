"""
Implements a Base Class for States
"""
import numpy as np
import gymnasium as gym
from typing_extensions import Self

from swing_trader.env.data import DataModel
from swing_trader.env.utils import Date

class State:
    
    timeframe: str
    data: DataModel
    date: Date
    
    @classmethod
    def from_data(cls, date: Date, data: DataModel):
        raise NotImplementedError
    
    def null(self) -> np.array:
        raise NotImplementedError
    
    def space(self) -> gym.Space:
        raise NotImplementedError
    
    @property
    def array(self) -> np.array:
        raise NotImplementedError

