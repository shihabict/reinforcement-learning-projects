import pandas as pd
import matplotlib.pyplot as plt

log_dir = "./logs_lka"

df = pd.read_csv(f"runs_lka/run_005/logs/lka_rewards.csv")

plt.figure()
plt.plot(df["timesteps"], df["episode_reward"])
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Episode Reward During Training")
plt.grid(True)
plt.savefig(f"{log_dir}/reward_vs_episod.png")
plt.show()

