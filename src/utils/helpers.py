import os
import torch
import numpy as np
import random

class AverageMeter:
    """Tracks and computes the running average of values (e.g., loss or accuracy)."""

    def __init__(self):
        self.reset()

    def update(self, value: float, n: int = 1):
        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_model(model, save_path):
    makedirs(os.path.dirname(save_path))
    torch.save(model.state_dict(), save_path)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True