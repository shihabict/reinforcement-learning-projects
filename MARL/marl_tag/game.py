import numpy as np

class TagGame:

    # Initialize the game actions
    MOVES = {0: (-1,0),  # up
             1: (1,0),  # down
             2: (0,-1),  # left
             3: (0,1)}  # right 

    def __init__(self, grid_size=8, seed=None):
        self.grid_size = grid_size
        self.rng = np.random.default_rng(seed) 
        self.tagger_pos = None
        self.runner_pos = None

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        # place tagger and runner at random positions
        while True:
            self.tagger_pos = self.rng.integers(low=0, high=self.grid_size, size=2)
            self.runner_pos = self.rng.integers(low=0, high=self.grid_size, size=2)
            if not np.array_equal(self.tagger_pos, self.runner_pos):
                break

    def _move(self, pos, action):
        # return the new position after applying the action
        new_pos = pos + np.array(self.MOVES[int(action)])
        new_pos = np.clip(new_pos, 0, self.grid_size - 1) # ensure the new position stays within the grid boundaries
        return new_pos

    def step(self, tagger_action, runner_action):
        # move both agents at the same time
        self.tagger_pos = self._move(self.tagger_pos, tagger_action)
        self.runner_pos = self._move(self.runner_pos, runner_action)
        return self.tagger_pos, self.runner_pos, np.array_equal(self.tagger_pos, self.runner_pos)

    def is_tagged(self):
        # return True if both in the same cell, False otherwise
        return bool(np.array_equal(self.tagger_pos, self.runner_pos))

    
if __name__ == "__main__":
    game = TagGame(grid_size=8, seed=42)
    game.reset()
    print("Tagger position:", game.tagger_pos)
    print("Runner position:", game.runner_pos)

    for _ in range(20):
        game.step(tagger_action=0, runner_action=0)
    print("Tagger position:", game.tagger_pos)
    print("Runner position:", game.runner_pos)