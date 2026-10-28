import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game import TagGame


class TagEnv(gym.Env):
    """Single-agent view of tag. The opponent acts inside step()."""

    def __init__(self, grid_size=8, max_steps=64, train_tagger=True, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.game = TagGame(grid_size, seed)
        self.max_steps = max_steps
        self.train_tagger = train_tagger   # True: learner is tagger, opponent is runner
        self.opponent = None               # set later with set_opponent()
        self.steps = 0                     # note: "steps", not "step" (that's the method)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)

    def set_opponent(self, model):
        self.opponent = model

    def _get_obs(self):
        """[tagger_row, tagger_col, runner_row, runner_col], scaled to [0, 1]."""
        obs = np.concatenate([self.game.tagger_pos, self.game.runner_pos])
        return obs.astype(np.float32) / (self.grid_size - 1)

    def _opponent_action(self, obs):
        """Ask the opponent model for an action, or act randomly if there is none."""
        if self.opponent is None:
            return int(self.action_space.sample())
        action, _ = self.opponent.predict(obs, deterministic=False)  # predict returns (action, state)
        return int(action)

    def _reward(self, tagged):
        """Zero-sum reward from the learner's point of view."""
        if self.train_tagger:
            return float(self.max_steps) if tagged else -1.0
        return -float(self.max_steps) if tagged else 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.seed(seed)           # makes reset(seed=...) reproducible
        self.game.reset()
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        # Opponent chooses its action from the same observation the learner saw
        opponent_action = self._opponent_action(self._get_obs())

        # Map actions to roles 
        if self.train_tagger:
            tagger_action, runner_action = action, opponent_action
        else:
            tagger_action, runner_action = opponent_action, action

        # Move both agents
        self.game.step(tagger_action, runner_action)

        # Outcome
        tagged = self.game.is_tagged()
        reward = self._reward(tagged)
        terminated = tagged                                   # the game ended
        truncated = (not tagged) and self.steps >= self.max_steps  # the time limit hit

        return self._get_obs(), reward, terminated, truncated, {}



if __name__ == "__main__":
    
    from stable_baselines3.common.env_checker import check_env
    check_env(TagEnv(), warn=True)

    # role mapping
    env = TagEnv()
    env.reset()
    env.game.tagger_pos = np.array([0, 0])
    env.game.runner_pos = np.array([0, 2])
    env.opponent = type("Stay",(),{"predict": lambda self, obs, deterministic=False: (2, None)})()
    obs, reward, terminated, truncated, info = env.step(3)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

    # Runner role
    env = TagEnv(train_tagger=False)
    env.reset()
    env.game.tagger_pos = np.array([0, 0])
    env.game.runner_pos = np.array([0, 2])
    env.opponent = type("Stay",(),{"predict": lambda self, obs, deterministic=False: (2, None)})()
    obs, reward, terminated, truncated, info = env.step(2)
    print("Observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)