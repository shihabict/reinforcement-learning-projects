import csv
import os
from stable_baselines3.common.callbacks import BaseCallback

class EpisodeRewardCSVCallback(BaseCallback):
    """
    Save episode rewards/lengths to a CSV during training.
    Works with VecEnv (n_envs >= 1).
    """

    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        # columns: global timestep, env index, episode reward, episode length
        self._writer.writerow(["timesteps", "env_idx", "episode_reward", "episode_length"])
        self._file.flush()

    def _on_step(self) -> bool:
        # infos is a list with length = n_envs
        infos = self.locals.get("infos", [])
        for env_idx, info in enumerate(infos):
            # SB3 puts episode stats here when an episode ends (no Monitor needed)
            ep_info = info.get("episode")
            if ep_info is not None:
                ep_r = ep_info.get("r")
                ep_l = ep_info.get("l")
                self._writer.writerow([self.num_timesteps, env_idx, ep_r, ep_l])
                self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
