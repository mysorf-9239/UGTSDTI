"""Seed management utilities for reproducible runs."""
from __future__ import annotations

import random
from typing import Any


def seed_everything(seed: int, *, deterministic: bool = False) -> dict[str, Any]:
    """Set Python/NumPy/torch seeds and return applied runtime notes."""
    random.seed(seed)
    notes = {
        "seed": seed,
        "deterministic": deterministic,
        "assumption": "Cross-platform reproducibility remains best-effort.",
    }

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
    except ImportError:
        pass

    return notes


def seed_worker(worker_id: int, base_seed: int) -> int:
    """Derive and apply a worker-specific seed."""
    worker_seed = int(base_seed) + int(worker_id)
    seed_everything(worker_seed, deterministic=False)
    return worker_seed
