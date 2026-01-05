# from stable_baselines3 import SAC
# from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv
#
# env = LaneKeepingEnv(render_mode="human")
# model = SAC.load("runs_lka/run_005/models/sac_lane_keeping_steps_300000_run_005.zip")
#
# obs, info = env.reset()
# episode_reward = 0.0
#
# for _ in range(100000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     episode_reward += float(reward)
#
#     if terminated or truncated:
#         break
#
# print("Episode reward:", episode_reward)
# env.close()



import time
import numpy as np
from stable_baselines3 import SAC

from DRL_highwayenv.src.traffic_aware_environment import LaneKeepingEnv

# from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv

env = LaneKeepingEnv(render_mode="human")
# Optional: make sure it runs in real-time if your env supports it
env.configure({"duration": 30, "policy_frequency": 50, "real_time_rendering": True, "initial_speed": 15, "road_length": 600})

model = SAC.load("runs_lka/run_007/models/sac_lane_keeping_steps_300000_run_007.zip")

N_EPISODES = 50

returns = []
lengths = []

for ep in range(N_EPISODES):
    obs, info = env.reset()
    ep_return = 0.0
    ep_len = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ep_return += float(reward)
        ep_len += 1

        if terminated or truncated:
            break

    returns.append(ep_return)
    lengths.append(ep_len)
    print(f"Episode {ep+1:03d} | return={ep_return:.2f} | steps={ep_len}")

print("\nSummary:")
print(f"Return mean={np.mean(returns):.2f}, std={np.std(returns):.2f}")
print(f"Length mean={np.mean(lengths):.1f} steps")

env.close()
