from swing_trader.env.rewards.base import Reward


class NumTradesPenalty(Reward):
    def __init__(self, **kwargs):
        if "history" not in kwargs:
            raise ValueError("Reward needs history")
        
        if "num_trades_penalty_weight" in kwargs:
            self.num_trades_penalty_weight = kwargs["num_trades_penalty_weight"]
        else:
            self.num_trades_penalty_weight = 0.01    
    
        self.history = kwargs["history"]
    
    def value(self) -> float:
        num_trades_penalty = self.num_trades_penalty_weight * (len(self.history) // 2)
        return num_trades_penalty