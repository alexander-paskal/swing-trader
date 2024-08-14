from swing_trader.env.rewards.performance import Performance


class PerformanceDifference(Performance):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if "performance_history" not in kwargs:
            raise ValueError("Reward needs performance_history")

        self.performance_history = kwargs['performance_history']

    def value(self) -> float:

        reward = super().value()

        if len(self.performance_history) == 0:
            return 0
        
        prev_performance = self.performance_history[-1]

        return reward - prev_performance