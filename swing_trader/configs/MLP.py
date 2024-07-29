from swing_trader.configs.config import RunConfig
from swing_trader.env.env import StockEnv, Config
from swing_trader.model.mlp import CustomTorchModel

HISTORY = 40
OBS_SPACE = HISTORY * 6 + 1

config = RunConfig(
    model=CustomTorchModel,
    model_config={},
    env=StockEnv,
    env_config=Config(
        rollout_length=200,
        market="QQQ",
        min_hold=2,
        state_history_length=HISTORY,
        action_space = 2,
        observation_space = OBS_SPACE
    ),
    num_cpus=2
)