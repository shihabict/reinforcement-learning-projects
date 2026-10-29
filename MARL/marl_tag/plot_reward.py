import numpy as np
import matplotlib.pyplot as plt
import os

def smooth(x, window=50):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def plot_history(history, output_path, max_steps=64):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, role in enumerate(["tagger", "runner"]):
        axes[0, col].plot(smooth(history[role]["rewards"]))
        axes[0, col].set_title(f"{role}: episode reward")
        axes[1, col].plot(smooth(history[role]["lengths"]))
        axes[1, col].axhline(max_steps, linestyle="--", color="gray")
        axes[1, col].set_title(f"{role}: episode length")
        axes[1, col].set_xlabel("episodes (over all rounds)")
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "history_plot.pdf")
    plt.savefig(output_path, dpi=300)