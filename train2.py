
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--config", type=str
)

args = parser.parse_args()



# from configs.config import RunConfig
import swing_trader.configs as MyConfigs

config: MyConfigs.config.RunConfig = getattr(MyConfigs, args.config)


# breakpoint()

import ray
from ray import air, tune
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import get_trainable_cls
from ray.rllib.models import ModelCatalog

torch, nn = try_import_torch()


ModelCatalog.register_custom_model("MyCustomModel", config.model)


ray.init(
    local_mode=config.local_mode,
    num_cpus=config.num_cpus
)

RayConfig = (
    get_trainable_cls(config.train_type)  # usuall PPO
    .get_default_config()
    .environment(
        config.env, env_config=config.env_config
    )
    .training(model={
        "custom_model": "MyCustomModel",
        "custom_model_config": config.model_config
    })
    .framework(
        "torch",
        torch_compile_worker=config.torch_compile_worker,
        torch_compile_worker_dynamo_backend=config.torch_compile_worker_dynamo_backend,
        torch_compile_worker_dynamo_mode=config.torch_compile_worker_dynamo_mode,
    )
    .rollouts(
        num_rollout_workers=config.num_rollout_workers
    )
    .resources(
        num_gpus=config.num_gpus
    )
)


# right now only support running with the tuner
tuner = tune.Tuner(
    config.train_type,  # 'PPO'
    param_space=RayConfig.to_dict(),
    run_config=air.RunConfig(
        stop={
            "training_iteration": config.stop_training_iteration,
            "timesteps_total": config.stop_timesteps_total,
            "episode_reward_mean": config.stop_episode_reward_mean
        }
    )
)

tuner.fit()

ray.shutdown()