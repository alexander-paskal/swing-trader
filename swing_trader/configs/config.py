from typing_extensions import TypedDict
from typing import *
from dataclasses import dataclass
# Reward, State, Action 

@dataclass
class RunConfig:

    model: TypeVar  # the type of the model
    model_config: Dict  # the init args of the model
    env: TypeVar  # the type of the environment
    env_config: Dict  # the init args of the environment

    # base args
    train_type: str = "PPO"
    local_mode: bool = False
    no_tune: bool = False

    # performance
    num_cpus: int = 1
    num_gpus: int = 1
    num_rollout_workers: int = 1
    torch_compile_worker: bool = False,  # this may make debugging difficult
    torch_compile_worker_dynamo_backend: str ="ipex",
    torch_compile_worker_dynamo_mode: str ="default",
    
    # training
    stop_training_iteration: int  = 5000# after how many training runs to terminate
    stop_timesteps_total: int = 100000 # after how many timesteps to stop the reward
    stop_episode_reward_mean: float = 10 # what time to stop the reward

    # checkpointing
    checkpoint_frequency: int = 1 # how often to checkpoint

    # hyperparameters
    max_grad_norm: float = 0.5