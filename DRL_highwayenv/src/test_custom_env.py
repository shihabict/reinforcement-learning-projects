import numpy as np

from DRL_highwayenv.src.traffic_aware_environment import LaneKeepingEnv

# from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv

# assume LaneKeepingEnv is already defined in the notebook or imported
env = LaneKeepingEnv(render_mode="human")   # or render_mode="rgb_array"

obs, info = env.reset()
total = 0

for _ in range(1000):
    # for ContinuousAction, sample returns an array like [steer] (or [steer, accel])
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward

    if terminated or truncated:
        obs, info = env.reset()

print("Random rollout reward:", total)
print("Obs shape:", np.array(obs).shape, "Action space:", env.action_space)
env.close()
