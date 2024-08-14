import numpy as np
from typing_extensions import Self

class BuySellSingleAction:
    buy: bool
    sell: bool
    def __init__(self, buy: bool = False, sell: bool = False):
        self.buy = buy
        self.sell = sell
    
    def serialize(self) -> np.array:
        return np.array([
            1 if self.buy else -1 if self.sell else 0,
        ])

    @classmethod
    def from_array(cls, arr: np.array) -> Self:
        d = {"buy": False, "sell": False}

        if arr[0] > 0:
            d["buy"] = True
        elif arr[0] < 0:
            d["sell"] = True
        
        return cls(**d)
