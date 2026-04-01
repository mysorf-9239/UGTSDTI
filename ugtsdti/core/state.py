"""State and StateWriter — the single communication medium between pipeline stages.

Design rules (REQ-STATE-001, REQ-STATE-002):
- State is a write-once, single-producer, monotonically-growing key-value store.
- Only StateWriter.commit() may add keys; nodes/plugins MUST NOT mutate State directly.
- Overwriting an existing key raises KeyCollisionError.
- Sensitive decision/postprocess surfaces are isolated on read to reduce silent mutation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ugtsdti.core.errors import KeyCollisionError


def _stable_hash(value: Any) -> Any:
    """Create a stable, hashable representation of any value.

    EXACT implementation for correctness - no performance optimizations.
    """
    import hashlib

    # Handle primitives directly
    if isinstance(value, (str, int, float, bool, type(None))):
        return value

    # Handle torch.Tensor
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # detach → cpu → contiguous
            tensor = value.detach().cpu().contiguous()
            # Hash FULL byte content using md5
            tensor_bytes = tensor.numpy().tobytes()
            md5_hash = hashlib.md5(tensor_bytes).hexdigest()
            # Include: ("tensor", shape, dtype, md5(bytes))
            return ("tensor", tuple(tensor.shape), str(tensor.dtype), md5_hash)
    except ImportError:
        pass

    # Handle numpy.ndarray
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            # contiguous
            if not value.flags.contiguous:
                value = value.copy()
            # md5(bytes)
            array_bytes = value.tobytes()
            md5_hash = hashlib.md5(array_bytes).hexdigest()
            # Include shape + dtype
            return ("numpy", tuple(value.shape), str(value.dtype), md5_hash)
    except ImportError:
        pass

    # Handle dict
    if isinstance(value, dict):
        # sorted by key
        sorted_items = []
        for key in sorted(value.keys(), key=str):
            sorted_items.append((key, _stable_hash(value[key])))
        return ("dict", tuple(sorted_items))

    # Handle list / tuple
    if isinstance(value, (list, tuple)):
        # recursive hash
        return ("list", tuple(_stable_hash(item) for item in value))

    # Handle set
    if isinstance(value, set):
        # sorted hash values
        sorted_hashes = sorted(str(_stable_hash(item)) for item in value)
        return ("set", tuple(sorted_hashes))

    # Fallback for other objects
    try:
        return ("object", type(value).__name__, str(value))
    except Exception:
        return ("object", type(value).__name__, "<unrepresentable>")


def _compute_fingerprint_from_store(store: dict[str, Any]) -> str:
    """Compute fingerprint from store using EXACT specification."""
    import hashlib

    # Iterate sorted(store.items())
    stable_items = []
    for key in sorted(store.keys()):
        # For each (k, v) → (k, stable_hash(v))
        stable_items.append((key, _stable_hash(store[key])))

    # Build tuple
    stable_tuple = tuple(stable_items)

    # fingerprint = md5(repr(tuple).encode())
    return hashlib.md5(repr(stable_tuple).encode()).hexdigest()


class State:
    """Read-only public surface for pipeline stages.

    Stages read from State via get/has/keys/snapshot.
    Writing is exclusively done through StateWriter.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self._fingerprint: str = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute a stable hash fingerprint of the current state."""
        return _compute_fingerprint_from_store(self._store)

    def get_fingerprint(self) -> str:
        """Get the current fingerprint of the state."""
        # Always compute fresh fingerprint to detect mutations
        return self._compute_fingerprint()

    def _update_fingerprint(self) -> None:
        """Update the fingerprint after legitimate state changes."""
        self._fingerprint = self._compute_fingerprint()

    # ------------------------------------------------------------------
    # Public read surface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Return an isolated copy of the value for *key*.

        ALWAYS isolates to ensure complete safety and prevent reference leaks.

        Raises KeyError if the key has not been committed yet.
        """
        if key not in self._store:
            raise KeyError(key)
        return _isolate_value(self._store[key])

    def get_isolated(self, key: str) -> Any:
        """Return an isolated copy of the committed value for *key*.

        NOTE: This method is now redundant with get() but kept for backward compatibility.
        """
        return self.get(key)  # Same as get() now

    def has(self, key: str) -> bool:
        """Return True if *key* has been committed to this State."""
        return key in self._store

    def keys(self) -> list[str]:
        """Return a sorted list of all committed keys."""
        return sorted(self._store.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated copy of the internal store.

        The returned dict is detached from State storage, including nested
        mutable values, so callers cannot mutate State through the snapshot.
        """
        return {key: _isolate_value(value) for key, value in self._store.items()}

    def snapshot_isolated(self) -> dict[str, Any]:
        """Return a per-key isolated snapshot for boundary-safe inspection."""
        return {key: self.get(key) for key in self.keys()}

    def _internal_store(self) -> dict[str, Any]:
        """Internal access to storage for StateWriter only.

        This method is deliberately named with underscore to indicate
        it's for internal use by trusted components only.
        """
        return self._store


class StateWriter:
    """Executor-only write surface for State.

    Only the stage executor should hold a StateWriter reference.
    Nodes and plugins receive a read-only State view.
    """

    def __init__(self, state: State) -> None:
        self._state = state
        # Maps key -> producer name for single-producer enforcement.
        self._producers: dict[str, str] = {}

    def commit(self, producer: str, outputs: dict[str, Any]) -> None:
        """Commit *outputs* into State on behalf of *producer*.

        Rules enforced:
        - write-once: a key already present raises KeyCollisionError.
        - single-producer: the same key from a different producer also raises
          KeyCollisionError (the key is already owned by the first producer).

        Args:
            producer: Identifier of the component committing these outputs
                      (e.g. node name, stage name).
            outputs:  Mapping of key -> value to commit.

        Raises:
            KeyCollisionError: If any key in *outputs* already exists in State.
        """
        # Validate the whole batch first so commit is atomic: a failed write
        # must not leave partial outputs in State.
        for key in outputs:
            if self._state.has(key):
                existing_producer = self._producers.get(key, "<unknown>")
                raise KeyCollisionError(
                    f"Key {key!r} already committed by producer {existing_producer!r}; "
                    f"producer {producer!r} cannot overwrite it.",
                    stage="state",
                    component=producer,
                    key=key,
                )

        for key, value in outputs.items():
            self._state._internal_store()[key] = _isolate_value(value)
            self._producers[key] = producer

        # Update fingerprint after successful commit
        self._state._update_fingerprint()


def _isolate_value(value: Any) -> Any:
    """Detach or copy mutable values at commit time to reduce silent mutation.

    Ensures complete isolation:
    - Tensors: detach() + clone() to break grad graph and shared memory
    - Arrays: copy() to break shared memory
    - Collections: deep copy with recursive isolation
    """
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # Detach from computation graph AND clone to break shared memory
            # This ensures no gradient leakage and complete isolation
            return value.detach().clone()
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.copy()
    except ImportError:
        pass

    if isinstance(value, dict):
        return {key: _isolate_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_isolate_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_isolate_value(item) for item in value)
    if isinstance(value, set):
        return {_isolate_value(item) for item in value}

    try:
        return deepcopy(value)
    except Exception:
        return value


def _should_isolate_on_read(key: str) -> bool:
    if key in {"teacher.logits", "student.logits", "logits", "gate.alpha"}:
        return True
    return key.startswith(("loss.", "metrics.", "diagnostics."))
