from stable_baselines3 import SAC
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv


log_dir = "./logs_lka"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    env = LaneKeepingEnv(render_mode=None)
    # Optional: tweak config quickly
    env.configure({"duration": 20, "policy_frequency": 5})
    return Monitor(env, log_dir)

vec_env = make_vec_env(make_env, n_envs=4)  # parallel envs helps

model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=200000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=1,
    gradient_steps=1,
)

model.learn(total_timesteps=300_000)
model.save("sac_lane_keeping")