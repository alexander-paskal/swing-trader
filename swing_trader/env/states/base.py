"""
Implements a Base Class for States
"""
import numpy as np
import gymnasium as gym
from typing_extensions import Self

from swing_trader.env.data import DataModel
from swing_trader.env.utils import Date

class State:
    

    @classmethod
    def from_data(cls, date: Date, data: DataModel):
        raise NotImplementedError
    
    def serialize(self) -> np.array:
        raise NotImplementedError
    
    @classmethod
    def from_serialized(cls, arr: np.array) -> Self:
        raise NotImplementedError
    
    @classmethod
    def null(cls) -> Self:
        raise NotImplementedError
    
    @classmethod
    def space(cls, history=0) -> gym.Box:
        raise NotImplementedError
    


