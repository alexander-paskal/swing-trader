from swing_trader.env.rewards.base import Reward
from typing import List, Type

class CompositeReward(Reward):
    
    components: List[Reward]
    
    def value(self) -> float:
        return sum([c.value() for c in self.components])