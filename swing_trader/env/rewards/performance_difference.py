from swing_trader.env.rewards.performance import Performance


class PerformanceDifference(Performance):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        if "reward_history" not in kwargs:
            raise ValueError("Reward needs reward_history")

        self.reward_history = kwargs['reward_history']

    def value(self) -> float:

        reward = super().value()

        if len(self.reward_history) == 0:
            return reward
        
        prev_reward = self.reward_history[-1]

        return reward - prev_reward