from stable_baselines3 import SAC
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# from DRL_highwayenv.src.custom_LKA_env import LaneKeepingEnv
from DRL_highwayenv.src.episod_callback import EpisodeRewardCSVCallback
from DRL_highwayenv.src.traffic_aware_environment import LaneKeepingEnv
from DRL_highwayenv.src.utils import get_next_run_dir

# log_dir = "./logs_lka"
run_dir, log_dir, model_dir, run_id = get_next_run_dir(base_dir="runs_lka", prefix="run_")
print(f"Saving this run to: {run_dir}")

def make_env():
    env = LaneKeepingEnv(render_mode=None)
    # Optional: tweak config quickly
    env.configure({"duration": 20, "policy_frequency": 5})
    return Monitor(env, log_dir)


callback = EpisodeRewardCSVCallback(csv_path=os.path.join(log_dir, "lka_rewards.csv"))
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
training_timesteps = 300_000
model.learn(total_timesteps=training_timesteps, callback=callback)
save_name = f"sac_lane_keeping_steps_{training_timesteps}_run_{run_id:03d}"
model.save(os.path.join(model_dir, save_name))

print(f"Model saved to: {os.path.join(model_dir, save_name)}")