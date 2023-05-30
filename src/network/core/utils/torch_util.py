import random
import numpy as np

import torch


def set_random_seed(seed):
    """
    Set the random seed of numpy, random, and torch package
    The seed is not fixed if it is less than zero.
    """
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
