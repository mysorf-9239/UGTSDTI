import os
import random

import numpy as np
import torch
from loguru import logger


def set_random_seed(seed: int = 42, strict_cudnn: bool = False) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: Integer seed value.
        strict_cudnn: If ``True``, forces deterministic cuDNN algorithms.
            Improves reproducibility at the cost of training throughput.
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
