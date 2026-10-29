import numpy as np
from env import TagEnv

def play(tagger=None, runner=None, episodes=200, grid_size=8, max_steps=64, seed=0):
    env = TagEnv(grid_size=grid_size, max_steps=max_steps, train_tagger=True)
    env.set_opponent(runner)

    catches, lengths, rewards = [], [], []

    for i in range(episodes):
        obs,_ = env.reset(seed=seed+i)
        total_reward = 0 

        terminated = truncated = False


        # loop untill the episode ends
        while not (terminated or truncated):
            if tagger is None:
                action = env.action_space.sample()
            else:
                action, _ = tagger.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward

        catches.append(terminated)      # terminated means the runner was caught
        lengths.append(env.steps)
        rewards.append(total_reward)

    return {
        "catch_rate": float(np.mean(catches)),
        "mean_length": float(np.mean(lengths)),
        "mean_reward": float(np.mean(rewards)),
    } 


def cross_play(trainer, episodes=200, seed=0):
    # print 2x2 matrix of cross-play results
    taggers = {"random": None, "trainer": trainer.models["tagger"]}
    runners = {"random": None, "trainer": trainer.models["runner"]}

    print(f"{'tagger':<10}{'runner':<10}{'catch rate':>12}{'mean len':>10}{'mean rew':>10}")
    for t_name, t in taggers.items():
        for r_name, r in runners.items():
            s = play(t, r, episodes, trainer.grid_size, trainer.max_steps, seed)
            print(f"{t_name:<10}{r_name:<10}{s['catch_rate']:>12.2%}"
                  f"{s['mean_length']:>10.1f}{s['mean_reward']:>10.1f}")


if __name__ == "__main__":
    from trainer import SelfPlayTrainer
    trainer = SelfPlayTrainer()
    trainer.load_models()
    cross_play(trainer)