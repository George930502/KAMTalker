import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def init_seeds(cuda_deterministic=True):
    """
    Initialize random seeds for reproducibility.
    """
    seed = 42   # Fixed seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Speed-reproducibility tradeoff
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def master_only(func):
    """
    Apply this function only on the master GPU.
    Since we're in single GPU, it always executes the function.
    """

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def is_master():
    """
    Check if the current process is the master.
    In a single GPU setting, this is always True.
    """
    return True


@master_only
def master_only_print(*args, **kwargs):
    """
    Master-only print.
    In a single GPU setting, this always prints.
    """
    print(*args, **kwargs)
