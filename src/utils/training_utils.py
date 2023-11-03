import transformers
import torch
import random
import numpy as np
import typing

def seed_everything(seed: typing.Optional[int]) -> None:
    """
    Seed random generators for transformers, numpy, torch, and python's random
    module with the specified seed
    """
    if seed is None: return None
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None