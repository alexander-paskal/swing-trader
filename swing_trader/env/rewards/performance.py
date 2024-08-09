from typing import List, Union
from swing_trader.env.utils import Date, BuyEvent, SellEvent
from swing_trader.env.data.data_model import DataModel
from swing_trader.env.rewards.base import Reward

class Performance(Reward):
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

        # num trades - factor this out
        # num_trades_penalty = 0.01 * len(self.history) // 2
        # if not self.is_finished:
        #     return num_trades_penalty
        
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
        
        # reward clipping - factor this out
        if multiplier > 3:
            return 3
        
        # num trades penalty - factor this out
        # multiplier -= num_trades_penalty

        return multiplier
