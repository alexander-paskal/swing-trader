# Configs setup
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--config", type=str, default="TestEnv"
)

args = parser.parse_args()

import swing_trader.configs as MyConfigs

config: MyConfigs.config.RunConfig = getattr(MyConfigs, args.config)


# Stable Baselines Setup
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env
import torch as th

env = config.env(config.env_config)
env.reset()
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 128, 64, 32], vf=[256, 128, 64, 32]))
model = PPO(
    policy="MlpPolicy",
    policy_kwargs=policy_kwargs,
    env=env,
    n_steps=128,
    verbose=1, 
    tensorboard_log="tb_logs/ppo_stock_env/", 
    max_grad_norm=config.max_grad_norm,
) 
    # Train the agent and display a progress bar
model.learn(
    total_timesteps=300_000,
    progress_bar=True, 
    # callback=eval_callback
    
)
# Save the agent
model.save("stock_env")