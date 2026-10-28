import random
import numpy as np
import torch



def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    set_seed(42)
    print("Seed set to 42")
    print("Picked device:", pick_device())
