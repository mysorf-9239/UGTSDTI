"""State and StateWriter — the single communication medium between pipeline stages.

Design rules (REQ-STATE-001, REQ-STATE-002):
- State is a write-once, single-producer, monotonically-growing key-value store.
- Only StateWriter.commit() may add keys; nodes/plugins MUST NOT mutate State directly.
- Overwriting an existing key raises KeyCollisionError.
- snapshot() returns a shallow copy so callers cannot mutate internal storage.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from ugtsdti.core.errors import KeyCollisionError


class State:
    """Read-only public surface for pipeline stages.

    Stages read from State via get/has/keys/snapshot.
    Writing is exclusively done through StateWriter.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public read surface
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any:
        """Return the value for *key*.

        Raises KeyError if the key has not been committed yet.
        """
        return self._store[key]

    def has(self, key: str) -> bool:
        """Return True if *key* has been committed to this State."""
        return key in self._store

    def keys(self) -> list[str]:
        """Return a sorted list of all committed keys."""
        return sorted(self._store.keys())

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the internal store.

        The returned dict is independent of the internal store, so callers
        cannot mutate State by modifying the snapshot.
        """
        return dict(self._store)


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
            self._state._store[key] = _isolate_value(value)
            self._producers[key] = producer


def _isolate_value(value: Any) -> Any:
    """Detach or copy mutable values at commit time to reduce silent mutation."""
    try:
        import torch

        if isinstance(value, torch.Tensor):
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
