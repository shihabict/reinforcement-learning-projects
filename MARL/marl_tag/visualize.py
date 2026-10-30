import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from env import TagEnv


def draw(ax, env, step, message="", pause=True):
    """Draw one frame: an empty grid with the tagger (red) and runner (blue)."""
    n = env.grid_size
    ax.clear()
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)              # flipped so row 0 is at the top
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid(True)
    
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor")
    ax.tick_params(which="minor", length=0)
    

    tr, tc = env.game.tagger_pos
    rr, rc = env.game.runner_pos
    ax.scatter(tc, tr, s=900, c="red", label="Tagger")
    ax.scatter(rc, rr, s=500, c="blue", label="Runner")
    ax.legend(loc="upper right")
    ax.set_title(f"Step {step}  {message}")
    if pause:
        plt.pause(0.001)                       # lets the window refresh



from PIL import Image

def record(trainer, path="tag.gif", episodes=3, seed=None, tagger_random=False,
           frame_ms=300, hold_ms=1000):
    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))
    print(f"Using base seed {seed}")

    env = TagEnv(trainer.grid_size, trainer.max_steps, train_tagger=True)
    env.set_opponent(trainer.models["runner"])
    tagger = trainer.models["tagger"]

    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)   # smaller = lighter GIF
    frames, durations = [], []

    def grab(duration):
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        frames.append(img)
        durations.append(duration)

    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        draw(ax, env, step=0, message="start", pause=False)
        grab(frame_ms)

        terminated = truncated = False
        while not (terminated or truncated):
            if tagger_random:
                action = env.action_space.sample()
            else:
                action, _ = tagger.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            draw(ax, env, step=env.steps, pause=False)
            grab(frame_ms)

        message = "CAUGHT!" if terminated else "TIME UP"
        draw(ax, env, step=env.steps, message=message, pause=False)
        grab(hold_ms)                                # hold the last frame longer

    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)       # loop=0 repeats forever
    plt.close(fig)
    print(f"Saved {len(frames)} frames to {path}")


def watch(trainer, episodes=5, delay=0.1, seed=None, tagger_random=False):
    if seed is None:
        seed = int(np.random.randint(0, 1_000_000))
        
    env = TagEnv(trainer.grid_size, trainer.max_steps, train_tagger=True)
    env.set_opponent(trainer.models["runner"])
    tagger = trainer.models["tagger"]

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        draw(ax, env, step=0, message="start")
        plt.pause(delay)

        terminated = truncated = False
        while not (terminated or truncated):
            if tagger_random:
                action = env.action_space.sample()
            else:
                action, _ = tagger.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(int(action))

            draw(ax, env, step=env.steps, message="")
            plt.pause(delay)

        # final frame
        message = "CAUGHT!" if terminated else "TIME UP"
        draw(ax, env, step=env.steps, message=message)
        plt.pause(1.0)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    from trainer import SelfPlayTrainer
    trainer = SelfPlayTrainer()
    trainer.load_models()
    watch(trainer)
    
    # record(trainer, "random_tagger.gif", episodes=3, tagger_random=True, seed=500548)