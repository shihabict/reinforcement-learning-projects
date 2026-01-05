# # from stable_baselines3 import SAC
# # from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv
# #
# # env = LaneKeepingEnv(render_mode="human")
# # model = SAC.load("runs_lka/run_005/models/sac_lane_keeping_steps_300000_run_005.zip")
# #
# # obs, info = env.reset()
# # episode_reward = 0.0
# #
# # for _ in range(100000):
# #     action, _ = model.predict(obs, deterministic=True)
# #     obs, reward, terminated, truncated, info = env.step(action)
# #     episode_reward += float(reward)
# #
# #     if terminated or truncated:
# #         break
# #
# # print("Episode reward:", episode_reward)
# # env.close()
#
#
#
# import time
# import numpy as np
# from stable_baselines3 import SAC
#
# from DRL_highwayenv.src.traffic_aware_environment import LaneKeepingEnv
#
# # from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv
#
# env = LaneKeepingEnv(render_mode="human")
# # Optional: make sure it runs in real-time if your env supports it
# env.configure({"duration": 30, "policy_frequency": 50, "real_time_rendering": True, "initial_speed": 15, "road_length": 600})
#
# model = SAC.load("runs_lka/run_007/models/sac_lane_keeping_steps_300000_run_007.zip")
#
# N_EPISODES = 50
#
# returns = []
# lengths = []
#
# for ep in range(N_EPISODES):
#     obs, info = env.reset()
#     ep_return = 0.0
#     ep_len = 0
#
#     while True:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)
#
#         ep_return += float(reward)
#         ep_len += 1
#
#         if terminated or truncated:
#             break
#
#     returns.append(ep_return)
#     lengths.append(ep_len)
#     print(f"Episode {ep+1:03d} | return={ep_return:.2f} | steps={ep_len}")
#
# print("\nSummary:")
# print(f"Return mean={np.mean(returns):.2f}, std={np.std(returns):.2f}")
# print(f"Length mean={np.mean(lengths):.1f} steps")
#
# env.close()


import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from DRL_highwayenv.src.traffic_aware_environment import LaneKeepingEnv

env = LaneKeepingEnv(render_mode="human")
env.configure({
    "duration": 30,
    "policy_frequency": 50,
    "real_time_rendering": True,
    "initial_speed": 15,
    "road_length": 600
})

model = SAC.load("runs_lka/run_008/models/sac_lane_keeping_steps_300000_run_008.zip")

N_EPISODES = 5  # set to 50 if you want, plotting 50 episodes may be heavy

all_returns = []
all_lengths = []

for ep in range(N_EPISODES):
    obs, info = env.reset()
    ep_return = 0.0
    ep_len = 0

    # ---- logs for plotting ----
    t_log = []
    ego_v_log = []
    front_v_log = []
    gap_log = []

    dt = 1.0 / env.config["policy_frequency"]  # step time in seconds

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        ego = env.vehicle
        ego_speed = float(ego.speed)

        # immediate front vehicle in same lane
        front, _ = env.road.neighbour_vehicles(ego, lane_index=ego.lane_index)
        if front is not None:
            front_speed = float(front.speed)
            gap = float(front.position[0] - ego.position[0] - front.LENGTH)
        else:
            front_speed = np.nan
            gap = np.nan

        # log
        t_log.append(ep_len * dt)
        ego_v_log.append(ego_speed)
        front_v_log.append(front_speed)
        gap_log.append(gap)

        ep_return += float(reward)
        ep_len += 1

        if terminated or truncated:
            break

    all_returns.append(ep_return)
    all_lengths.append(ep_len)

    print(f"Episode {ep+1:03d} | return={ep_return:.2f} | steps={ep_len}")

    # ---- Plot for this episode ----
    plt.figure()
    plt.plot(t_log, ego_v_log, label="Ego speed (m/s)")
    plt.plot(t_log, front_v_log, label="Front vehicle speed (m/s)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Speed Profile (Episode {ep+1:03d})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional: also plot gap
    plt.figure()
    plt.plot(t_log, gap_log, label="Gap to front (m)")
    plt.xlabel("Time (s)")
    plt.ylabel("Gap (m)")
    plt.title(f"Gap Profile (Episode {ep+1:03d})")
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nSummary:")
print(f"Return mean={np.mean(all_returns):.2f}, std={np.std(all_returns):.2f}")
print(f"Length mean={np.mean(all_lengths):.1f} steps")

env.close()
