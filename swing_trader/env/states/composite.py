from swing_trader.env.states.base import State


import numpy as np
import gymnasium as gym
from typing_extensions import Self
from typing import *

from swing_trader.env.data import DataModel
from swing_trader.env.utils import Date

__all__ = ['CompositeState']

# TODO - figure out how this is gonna work

class CompositeState(State):
    
    timeframe: str = "daily"
    states: List[State]

    def __init__(self, *states):

        self.states = states

        data = None
        date = None
        for s in self.states:
            if date is not None and s.date != date:
                raise ValueError("Mismatching Dates")
            data = s.data
            date = s.date
        
        self.data = data
        self.date = date

    @classmethod
    def from_data(cls, date: Date, data: DataModel, *state_classes, **kwargs):
        raise NotImplementedError
    
    def null(self) -> np.array:
        return [s.null() for s in self.states]
    
    @classmethod
    def space(cls) -> gym.Space:
        raise NotImplementedError
    
    @property
    def array(self) -> np.array:
        arr =  np.concatenate([s.array for s in self.states])
        return arr
