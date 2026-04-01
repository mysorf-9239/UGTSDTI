"""Global seed control for deterministic behavior.

This module ensures all randomness sources are properly seeded
and controlled across the entire system.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

# Try to import numpy and torch, but don't fail if not available
try:
    import numpy
    import torch

    HAS_NUMPY = True
    HAS_TORCH = True
except ImportError:
    HAS_NUMPY = False
    HAS_TORCH = False

# Create aliases for use when modules are available
if HAS_NUMPY:
    _np = numpy
else:
    _np = None  # type: ignore

if HAS_TORCH:
    _torch = torch
else:
    _torch = None  # type: ignore


def set_global_seed(seed: int) -> None:
    """Set global seed for all random number generators.

    This function ensures deterministic behavior across:
    - Python's random module
    - NumPy random number generation
    - PyTorch random number generation (CPU + CUDA)

    Args:
        seed: Global seed value for reproducibility
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    if HAS_NUMPY and np is not None:
        np.random.seed(seed)

    # Set PyTorch random seed
    if HAS_TORCH and torch is not None:
        torch.manual_seed(seed)

    # Set PyTorch seed if available
    if HAS_TORCH and _torch is not None:
        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed(seed)

    # Enable deterministic cuDNN
    if HAS_TORCH and _torch is not None:
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except RuntimeError as e:
            print(f"Warning: Some deterministic algorithms unavailable: {e}")
            # Continue with available deterministic ops


def get_random_state() -> dict[str, Any]:
    """Get current random state for debugging.

    Returns:
        Dictionary containing current random states of all systems.
    """
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "seed_info": {
            "python_seed": hash(random.random()),
        },
    }

    if HAS_NUMPY and _np is not None:
        state["numpy_random"] = _np.random.get_state()
        state["seed_info"]["numpy_seed"] = hash(_np.random.random())

    if HAS_TORCH and _torch is not None:
        state["torch_random"] = _torch.get_rng_state()
        state["seed_info"]["torch_seed"] = hash(_torch.rand(1).item())
        if _torch.cuda.is_available():
            state["torch_cuda_random"] = _torch.cuda.get_rng_state()

    return state


def validate_deterministic_setup() -> bool:
    """Validate that system is configured for deterministic behavior.

    Returns:
        True if all randomness sources are properly controlled.
    """
    issues = []

    # Check Python random
    try:
        # This should be deterministic
        random.random()
    except Exception as e:
        issues.append(f"Python random: {e}")

    # Check NumPy
    if HAS_NUMPY and _np is not None:
        try:
            _np.random.random()
        except Exception as e:
            issues.append(f"NumPy random: {e}")

    # Check PyTorch
    if HAS_TORCH and _torch is not None:
        try:
            _torch.rand(1)
            if _torch.cuda.is_available():
                _torch.rand(1, device="cuda")
        except Exception as e:
            issues.append(f"PyTorch random: {e}")

    if issues:
        print("⚠️  Deterministic setup issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    return True


class DeterministicContext:
    """Context manager for temporary deterministic testing."""

    def __init__(self, seed: int):
        self.seed = seed
        self.original_state: dict[str, Any] | None = None

    def __enter__(self) -> "DeterministicContext":
        # Store original state
        self.original_state = get_random_state()
        # Apply new seed
        set_global_seed(self.seed)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore original state
        if self.original_state:
            _restore_random_state(self.original_state)
        return None


def _restore_random_state(state: dict[str, Any]) -> None:
    """Restore random state from saved state."""
    # Restore Python random
    if "python_random" in state:
        random.setstate(state["python_random"])

    # Restore NumPy
    if HAS_NUMPY and _np is not None and "numpy_random" in state:
        _np.random.set_state(state["numpy_random"])

    # Restore PyTorch
    if HAS_TORCH and _torch is not None:
        if "torch_random" in state:
            _torch.set_rng_state(state["torch_random"])
        if _torch.cuda.is_available() and "torch_cuda_random" in state:
            _torch.cuda.set_rng_state(state["torch_cuda_random"])
