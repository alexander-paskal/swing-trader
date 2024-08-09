import numpy as np
from typing_extensions import Self

class BuySellAction:
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