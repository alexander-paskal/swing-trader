import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

TRAIN = False
EVAL = False
RENDER = True

# Create environment
print("Building the Env")
env = gym.make("LunarLander-v2", render_mode="rgb_array")

# # Instantiate the agent
if TRAIN:
    model = DQN("MlpPolicy", env, verbose=1)
    # Train the agent and display a progress bar
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    # Save the agent
    model.save("dqn_lunar")
    del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
print("Loading the Model")
model = DQN.load("dqn_lunar", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
if EVAL:
    print("Evaluating the Agent")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
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