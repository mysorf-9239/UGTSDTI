import os
import random

import numpy as np
import torch
from loguru import logger


def make_reproducible(seed: int = 42, strict_cudnn: bool = False):
    """
    Forces PyTorch, NumPy, and Python to use generic random seeds to maximize
    reproducibility across training runs.

    Args:
        seed: The integer seed to lock in.
        strict_cudnn: If True, makes CUDA convolution algorithms deterministic
                      (may significantly impact training performance).
    """
    logger.info(f"Locking RNG seeds to: {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

        if strict_cudnn:
            logger.warning("Strict CUDNN determinism enabled. Training will be slower.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
