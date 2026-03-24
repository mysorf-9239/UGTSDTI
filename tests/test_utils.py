"""Tests for utility functions: set_random_seed reproducibility."""

import numpy as np
import torch

from ugtsdti.utils.seed import set_random_seed


def test_seed_makes_torch_reproducible():
    set_random_seed(42)
    a = torch.randn(10)

    set_random_seed(42)
    b = torch.randn(10)

    assert torch.allclose(a, b)


def test_seed_makes_numpy_reproducible():
    set_random_seed(42)
    a = np.random.rand(10)

    set_random_seed(42)
    b = np.random.rand(10)

    assert np.allclose(a, b)


def test_different_seeds_produce_different_results():
    set_random_seed(0)
    a = torch.randn(10)

    set_random_seed(1)
    b = torch.randn(10)

    assert not torch.allclose(a, b)


def test_seed_does_not_crash_with_strict_cudnn():
    """strict_cudnn=True must not crash (even on CPU-only machines)."""
    set_random_seed(42, strict_cudnn=True)
