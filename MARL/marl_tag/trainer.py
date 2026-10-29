import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from plot_reward import plot_history
from env import TagEnv
from utils import pick_device, set_seed



class EpisodeLogger(BaseCallback):
    """Collects the reward and length of every finished episode."""
    def __init__(self):
        super().__init__()
        self.rewards = []
        self.lengths = []

    def _on_step(self):
        for info in self.locals["infos"]:      # one info per parallel env
            if "episode" in info:              # present only when an episode just ended
                self.rewards.append(info["episode"]["r"])
                self.lengths.append(info["episode"]["l"])
        return True   


class SelfPlayTrainer:
    def __init__(self, grid_size=8, max_steps=64, n_envs=8, n_steps=128, model_dir="models", device=None):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.model_dir = model_dir
        self.device = device or pick_device()
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.models = {"tagger": None, "runner": None}
        self.history = {
            "tagger": {
                "rewards": [],
                "lengths": []
            },
            "runner": {
                "rewards": [],
                "lengths": []
            }
        }
        
    
    def _make_env(self, role):
        train_tagger = role == "tagger"
        
        return make_vec_env(
            lambda: TagEnv(grid_size=self.grid_size, max_steps=self.max_steps, train_tagger=train_tagger),
            n_envs=self.n_envs,
        )   
    
    def _train_role(self, role, timesteps):
        opponent_role = "runner" if role == "tagger" else "tagger"
        env = self._make_env(role)
        logger = EpisodeLogger()

        # Give the frozen opponent to every env (None means random in TagEnv)
        if self.models[opponent_role] is not None:
            env.env_method("set_opponent", self.models[opponent_role])
            

        if self.models[role] is None:
            self.models[role] = PPO("MlpPolicy", env, n_steps=self.n_steps,
                                    device=self.device, verbose=0)
        else:
            self.models[role].set_env(env)

        print(env.get_attr("opponent"))

        self.models[role].learn(total_timesteps=timesteps, reset_num_timesteps=False, callback=logger)

        self.history[role]["rewards"].extend(logger.rewards)
        self.history[role]["lengths"].extend(logger.lengths)

        
        env.close()
        
    def train(self, timesteps_per_role=10000, n_iterations=10):
        for iteration in range(n_iterations):
            print(f"Starting iteration {iteration + 1}/{n_iterations}")
            for role in ["tagger", "runner"]:
                print(f"Training role: {role}")
                self._train_role(role, timesteps_per_role)
        self.save()
    
    
    def save(self):
        os.makedirs(self.model_dir, exist_ok=True)
        for role, model in self.models.items():
            model.save(os.path.join(self.model_dir, role))

        with open(os.path.join(self.model_dir, "history.json"), "w") as f:
            json.dump(self.history, f)


            
    def load_models(self):
        for role in ["tagger", "runner"]:
            model_path = os.path.join(self.model_dir, f"{role}.zip")
            if os.path.exists(model_path):
                self.models[role] = PPO.load(model_path, device=self.device)
                print(f"Loaded model for role: {role}")
            else:
                print(f"No saved model found for role: {role}") 
    
    
    
if __name__ == "__main__":
    set_seed(0)
    trainer = SelfPlayTrainer(grid_size=8, max_steps=64, n_envs=8, n_steps=128)
    trainer.train(timesteps_per_role=20000, n_iterations=15)

    plot_history(trainer.history, output_path="figures")