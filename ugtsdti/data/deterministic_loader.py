"""Deterministic data loading utilities.

This module provides utilities to ensure data loading
doesn't break determinism when using PyTorch DataLoader.
"""

from __future__ import annotations

from typing import Any, Iterator

# Try to import torch and numpy, but don't fail if not available
try:
    import torch

    HAS_TORCH = True
    HAS_NUMPY = True
except ImportError:
    HAS_TORCH = False
    HAS_NUMPY = False


def seed_worker(worker_id: int) -> None:
    """Seed function for PyTorch DataLoader workers.

    This function should be passed to DataLoader as worker_init_fn
    to ensure each worker process is deterministically seeded.

    Args:
        worker_id: The worker ID assigned by PyTorch DataLoader
    """
    # Use global seed + worker_id for deterministic worker seeding
    try:
        # Get initial seed (could be from global context)
        base_seed = torch.initial_seed() if hasattr(torch, "initial_seed") else 42
        worker_seed = base_seed + worker_id
        torch.manual_seed(worker_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
    except Exception:
        pass  # Expected to fail if torch not available


def create_deterministic_generator(seed: int = 42) -> "torch.Generator":
    """Create a deterministic PyTorch random number generator.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Deterministic PyTorch Generator instance
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for deterministic generator")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class DeterministicBatchSampler:
    """Deterministic batch sampler for reproducible data loading."""

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
            # Use torch.randperm for deterministic shuffling
            indices = torch.randperm(self.num_samples, generator=generator).tolist()

        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


def create_deterministic_sampler(dataset: Any, shuffle: bool = True, seed: int = 42) -> DeterministicBatchSampler:
    """Create a deterministic batch sampler.

    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility

    Returns:
        DeterministicBatchSampler instance
    """
    return DeterministicBatchSampler(dataset, shuffle=shuffle, seed=seed)


def create_deterministic_dataloader(
    dataset: Any, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, seed: int = 42, **kwargs: Any
) -> Any:
    """Create a deterministic PyTorch DataLoader.

    This DataLoader ensures:
    - Deterministic shuffling (when enabled)
    - Deterministic worker seeding
    - Reproducible sampling

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        seed: Random seed for reproducibility
        **kwargs: Additional arguments passed to DataLoader

    Returns:
        Deterministic PyTorch DataLoader instance
    """
    if not HAS_TORCH or torch is None:
        raise RuntimeError("PyTorch is required for deterministic DataLoader")

    # Create deterministic generator for shuffling
    generator = create_deterministic_generator(seed)

    # Create sampler
    sampler = create_deterministic_sampler(dataset, shuffle=shuffle, seed=seed)

    # Create DataLoader with deterministic settings
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: seed_worker(seed + worker_id),
        generator=generator,
        sampler=sampler,
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

    # Check for deterministic configuration indicators
    has_generator = hasattr(dataloader, "generator") and dataloader.generator is not None
    has_worker_init = hasattr(dataloader, "worker_init_fn") and dataloader.worker_init_fn is not None
    has_sampler = hasattr(dataloader, "sampler") and dataloader.sampler is not None

    return has_generator and has_worker_init and has_sampler
