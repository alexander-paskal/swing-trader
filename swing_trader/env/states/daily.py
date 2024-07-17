from swing_trader.env.states.base import *


class DailyState:
    bought: bool
    high: float
    low: float
    open: float
    close: float
    volume: float
    
    @classmethod
    def space(cls, history=0) -> gym.Box:
        l = (history + 1) * 6
        return gym.Box(
            low=-np.inf*np.ones(l),
            high=np.inf*np.ones(l),
            dtype=np.float32
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
