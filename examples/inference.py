import ray
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_env
import gym
import numpy as np

# Assuming you have a custom environment
class CustomEnv(gym.Env):
    def __init__(self, config=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,))

    def reset(self):
        return np.random.rand(4)

    def step(self, action):
        next_state = np.random.rand(4)
        reward = 1 if action == 1 else 0
        done = np.random.rand() > 0.95
        return next_state, reward, done, {}

# Register the custom environment
register_env("CustomEnv-v0", lambda config: CustomEnv(config))

# Initialize Ray with minimal resources
ray.init(num_cpus=1, num_gpus=0)

# Load the trained model
checkpoint_path = "/path/to/your/checkpoint"
loaded_algo = Algorithm.from_checkpoint(checkpoint_path)

# Set up the environment for inference
env = CustomEnv()

# Perform inference
num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = loaded_algo.compute_single_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
    print(f"Episode {episode + 1} reward: {episode_reward}")

# Shut down Ray
ray.shutdown()