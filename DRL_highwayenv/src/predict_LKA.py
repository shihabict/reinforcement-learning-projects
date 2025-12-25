from stable_baselines3 import SAC
from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv

env = LaneKeepingEnv(render_mode="human")
model = SAC.load("sac_lane_keeping")

obs, info = env.reset()
episode_reward = 0.0

for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += float(reward)

    if terminated or truncated:
        break

print("Episode reward:", episode_reward)
env.close()
