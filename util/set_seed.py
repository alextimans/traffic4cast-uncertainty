import os
import random

import numpy
import torch


def set_seed(seed: int = None) -> int:

    # https://pytorch.org/docs/stable/notes/randomness.html

    random.seed(seed)
    numpy.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True # Randomness but better performance
    torch.backends.cudnn.deterministic = False # Randomness but better performance
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    return seed
