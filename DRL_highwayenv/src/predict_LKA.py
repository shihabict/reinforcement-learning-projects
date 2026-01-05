from stable_baselines3 import SAC
from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv

env = LaneKeepingEnv(render_mode="human")
model = SAC.load("runs_lka/run_005/models/sac_lane_keeping_steps_300000_run_005.zip")

obs, info = env.reset()
episode_reward = 0.0

for _ in range(100000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += float(reward)

    if terminated or truncated:
        break

print("Episode reward:", episode_reward)
env.close()
