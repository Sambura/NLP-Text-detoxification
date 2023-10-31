import transformers
import torch
import random
import numpy as np

def seed_everything(seed):
    if seed is None: return None
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None