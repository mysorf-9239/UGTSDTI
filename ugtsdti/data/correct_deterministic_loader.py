"""Correct deterministic data loading approach.

Follows proper DataLoader semantics without data transformation.
"""

from __future__ import annotations

import random
from typing import Any

# Try to import torch, but don't fail if not available
try:
    import numpy as np
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    np = None


def seed_worker(worker_id: int) -> None:
    """Seed function for PyTorch DataLoader workers.

    Args:
        worker_id: The worker ID assigned by PyTorch DataLoader
    """
    # Use global seed + worker_id for deterministic worker seeding
    try:
        import torch

        # Get initial seed (could be from global context)
        base_seed = torch.initial_seed() if hasattr(torch, "initial_seed") else 42
        worker_seed = base_seed + worker_id

        # Seed all random sources in worker
        random.seed(worker_seed)
        if np is not None:
            np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

        print(f"Worker {worker_id} seeded with {worker_seed}")
    except Exception:
        # Fallback if torch not available
        random.seed(worker_id + 42)
        if np is not None:
            np.random.seed(worker_id + 42)


def create_deterministic_generator(seed: int) -> "torch.Generator":
    """Create a deterministic PyTorch Generator.

    Args:
        seed: Random seed for the generator

    Returns:
        Deterministic torch.Generator instance
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for deterministic generator")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def create_deterministic_dataloader(
    dataset: Any, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, seed: int = 42, **kwargs
) -> "torch.utils.data.DataLoader":
    """Create a deterministic PyTorch DataLoader.

    This follows PROPER DataLoader semantics:
    - Uses PyTorch's built-in shuffling (not custom hash sorting)
    - Seeds workers and generator properly
    - Does NOT transform the underlying data

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size for loading
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Deterministic PyTorch DataLoader
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for deterministic DataLoader")

    # Create deterministic generator
    generator = create_deterministic_generator(seed)

    # Create DataLoader with deterministic settings
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Let PyTorch handle shuffling deterministically
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        **kwargs,
    )


def validate_deterministic_dataloader(dataloader: Any) -> bool:
    """Validate that DataLoader is configured for determinism.

    Args:
        dataloader: PyTorch DataLoader to validate

    Returns:
        True if DataLoader is deterministic, False otherwise
    """
    if not HAS_TORCH or torch is None:
        return False

    issues = []

    # Check for worker_init_fn
    if not hasattr(dataloader, "worker_init_fn") or dataloader.worker_init_fn is None:
        issues.append("Missing worker_init_fn for deterministic worker seeding")

    # Check for generator
    if not hasattr(dataloader, "generator") or dataloader.generator is None:
        issues.append("Missing generator for deterministic sampling")

    # Check deterministic settings
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        if not torch.backends.cudnn.deterministic:
            issues.append("cuDNN determinism not enabled")

    if issues:
        print("⚠️  DataLoader determinism issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


class DeterministicBatchSampler(torch.utils.data.Sampler):
    """Deterministic batch sampler that respects seed.

    This sampler ensures reproducible batch ordering when shuffle=True.
    """

    def __init__(self, data_source: Any, shuffle: bool = True, seed: int = 42):
        if not HAS_TORCH or torch is None:
            raise RuntimeError("PyTorch is required for DeterministicBatchSampler")

        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(data_source)

        # Generate deterministic indices
        generator = create_deterministic_generator(seed)
        indices = list(range(self.num_samples))

        if self.shuffle:
            generator.manual_seed(seed)
            indices = torch.randperm(self.num_samples, generator=generator).tolist()

        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.num_samples


def create_deterministic_sampler(dataset: Any, shuffle: bool = True, seed: int = 42) -> DeterministicBatchSampler:
    """Create a deterministic batch sampler.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        Deterministic batch sampler
    """
    return DeterministicBatchSampler(dataset, shuffle=shuffle, seed=seed)


# ✅ CORRECT APPROACH NOTES:
"""
PROPER DETERMINISM FOR DATALOADER:

✅ CORRECT: Use PyTorch's built-in deterministic shuffling
   DataLoader(shuffle=True, generator=seeded_generator, worker_init_fn=seed_worker)

❌ INCORRECT: Hash-based record sorting (data transformation)
   sorted(records, key=hash_based_key)  # This changes the data!

KEY INSIGHT:
- Determinism = same seed → same SAMPLE ORDER
- Determinism ≠ same seed → same DATA CONTENT
- Hash sorting = DATA TRANSFORMATION (changes content!)
"""
