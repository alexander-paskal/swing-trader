from swing_trader.configs.config import RunConfig
from swing_trader.env.env import StockEnv, Config, InitialConditions
from swing_trader.model.mlp import CustomTorchModel
from swing_trader.env.data import DataModel
import random
from swing_trader.env.utils import Date

HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1
data = DataModel("QQQ", ["daily"])
data.set_date_bounds("2000-01-01", "2018-12-31")
DATES = [Date(d) for d in data.daily.index]


class TestEnv1(StockEnv):
    def _randomize_ics(self):
        self.ics = InitialConditions(
            name="Synthetic",
            date=Date(random.choice(DATES))
        )

    def reset(self, *, seed=None, options=None, randomize: bool = True):
        if randomize:
            self._randomize_ics()
        
        self.data: DataModel = DataModel.synthetic()

        self.cur_date = Date(self.ics["date"])
        self.start_date = Date(self.ics["date"])
        self.end_date = self.data.get_n_ticks_after(self.state_cls.timeframe, self.start_date, self.config["rollout_length"])
        self.is_holding = False
        self.history = []
        self.n_steps = 0

        return self.state, {}
    
config = RunConfig(
    model=CustomTorchModel,
    model_config={},
    env=TestEnv1,
    env_config=Config(
        rollout_length=300,
        market=None,
        min_hold=2,
        state_history_length=HISTORY,
        action_space = 1,
        observation_space = OBS_SPACE,
        scale_reward=0.01
    ),
    num_cpus=2
)