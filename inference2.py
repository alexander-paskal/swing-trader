from stable_baselines3 import PPO

model = PPO.load("stock_env")

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
    
    # arr = vec_env.render("rgb_array")
    # if ax is None:
    #     print("No Ax")
    #     ax = plt.imshow(arr)
    # else:
    #     ax.set_array(arr)
    # # plt.draw()
    # plt.title(f"Step {i}")
    # plt.pause(1/60)
    # i += 1

vec_env.render()
print("Done Rendering")