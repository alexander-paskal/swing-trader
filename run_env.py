from env import Env, Config, State, Action, ICs

config = Config(
    rollout_length=64,
    market="QQQ",
    min_hold=2,
    state_history_length=49
)

actions = [
    (1, 0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,1),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (1,0),
    
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,1),    
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (1,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,1),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (1,0),
    
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,1),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (1,0),
    
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    (0,0),
    
    (0,0),
    (0,0),
    (0,0),
]
env = Env(config)

state, infos = env.reset()

for a in actions:
    state, reward, terminated, truncated, infos = env.step(a)
    
    if terminated:
        break

print(
f"""
    name: {env.ics['name']}
    history: {env.history}
    return: {env.multiplier}
    market: {env.market_multiplier()}
    reward: {reward}
"""
    )

env.dump("env.json")
print()