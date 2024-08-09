from typing import List, Union
from swing_trader.env.utils import Date, BuyEvent, SellEvent
from swing_trader.env.data.data_model import DataModel
from swing_trader.env.rewards.base import Reward

class BuyAndHold(Reward):
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
        
        if "end_date" not in kwargs:
            raise ValueError("Reward needs end_date")
        
        if "start_date" not in kwargs:
            raise ValueError("Reward needs start_date")

        self.data: DataModel = kwargs["data"]
        self.end_date: Date = kwargs["end_date"]
        self.start_date: Date = kwargs["start_date"]

    def value(self):
        """Computes the multiplier if had just bought and held"""
        multiplier = self.data.get_price_on_close(self.end_date) / self.data.get_price_on_open(self.start_date)
        return multiplier
