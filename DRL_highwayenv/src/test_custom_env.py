import numpy as np
import highway_env  # ensures rendering deps are loaded (and env utilities)

from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv

# assume LaneKeepingEnv is already defined in the notebook or imported
env = LaneKeepingEnv(render_mode="human")   # or render_mode="rgb_array"

obs, info = env.reset()

for _ in range(1000):
    # for ContinuousAction, sample returns an array like [steer] (or [steer, accel])
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()

env.close()
