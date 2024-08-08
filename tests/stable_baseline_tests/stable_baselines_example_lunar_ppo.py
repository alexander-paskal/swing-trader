import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.lunar_lander import LunarLander
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.env_util import make_vec_env

TRAIN = True
EVAL = False
RENDER = True



from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


class FastLunarLander(LunarLander):
    def reset(self, *args, **kwargs):
        self._step_counter = 0
        return super().reset(*args, **kwargs)

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = super().step(*args, **kwargs)
        self._step_counter += 1

        # if self._step_counter > 2000:
        #     reward = reward - 10
        
        # elif self._step_counter > 1500:
        #     reward = reward - 5
        # elif self._step_counter > 1000:
        #     reward = reward - 1
        
        # if terminated:
        #     print('Final Steps:', self._step_counter)

        return obs, reward, terminated, truncated, info

# Create environment
print("Building the Env")
# env = gym.make("LunarLander-v2", render_mode="rgb_array")
# env = FastLunarLander(render_mode="rgb_array")
env = make_vec_env("LunarLander-v2", n_envs=8)

# callbacks
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
eval_callback = EvalCallback(
    env,
    eval_freq=1000,
    # callback_after_eval=stop_train_callback,
    verbose=1, 
    best_model_save_path=".",
    deterministic=False
)

# # Instantiate the agent
if TRAIN:
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="tb_logs/ppo_ll/", max_grad_norm=1)
    # model = A2C(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="tb_logs/a2c_ll/", max_grad_norm=1)
    model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="tb_logs/a2c_ll/", max_grad_norm=2)
    # Train the agent and display a progress bar
    model.learn(
        total_timesteps=200_000,
        progress_bar=True, 
        # callback=eval_callback
    )
    # Save the agent
    model.save("ppo_lunar")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
print("Loading the Model")
model = PPO.load("ppo_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
if EVAL:
    print("Evaluating the Agent")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10, deterministic=False)
    print(
    f"""
        Mean Reward: {mean_reward}
        STD Reward : {std_reward}
    """
    )

# Enjoy trained agent
if RENDER:
    import matplotlib.pyplot as plt
    import numpy as np
    vec_env = model.get_env()
    obs = vec_env.reset()
    print("beginning render")
    ax = None
    dones = np.array([False])
    i = 0

    while not dones[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        
        arr = vec_env.render("rgb_array")
        if ax is None:
            print("No Ax")
            ax = plt.imshow(arr)
        else:
            ax.set_array(arr)
        # plt.draw()
        plt.title(f"Step {i}")
        plt.pause(1/60)
        i += 1
        
    print("Done Rendering")