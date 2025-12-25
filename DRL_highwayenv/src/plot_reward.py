import os, glob
import pandas as pd
import matplotlib.pyplot as plt

run_dir = "./logs_lka/"  # <-- put your actual run folder here

# Read all monitor files (important for n_envs=4)
files = glob.glob(os.path.join(run_dir, "*monitor*.csv")) + glob.glob(os.path.join(run_dir, "*.csv"))
files = sorted(list(set(files)))

dfs = []
for f in files:
    # SB3 monitor has a JSON comment line starting with '#'
    df = pd.read_csv(f, comment="#")
    if set(["r", "l", "t"]).issubset(df.columns):
        dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

# data columns:
# r = episode reward, l = episode length, t = time (seconds since training start)
data = data.sort_values("t").reset_index(drop=True)

# Plot episode reward vs episode index
plt.figure()
plt.plot(data["r"].values, alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Episode reward")
plt.title("Training reward (Monitor)")
plt.grid(True)
plt.show()

# Optional: moving average
window = 50
if len(data) >= window:
    ma = data["r"].rolling(window).mean()
    plt.figure()
    plt.plot(ma)
    plt.xlabel("Episode")
    plt.ylabel(f"Reward (moving avg, window={window})")
    plt.title("Smoothed training reward")
    plt.grid(True)
    plt.show()
