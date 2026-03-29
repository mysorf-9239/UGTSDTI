"""Operational runtime adaptation without semantic changes."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class RuntimeAdapter:
    """Resolve operational knobs for local runtime execution."""

    def adapt(self, cfg: dict[str, Any]) -> dict[str, Any]:
        runtime = dict(cfg.get("runtime", {}))
        adapted = {
            "device": runtime.get("device", "cpu"),
            "seed": int(runtime.get("seed", 0)),
            "deterministic": bool(runtime.get("deterministic", False)),
            "precision": runtime.get("precision", "fp32"),
            "num_workers": int(runtime.get("num_workers", 0)),
            "batch_size": int(runtime.get("batch_size", 32)),
            "artifacts_dir": str(Path(runtime.get("artifacts_dir", "artifacts")).resolve()),
            "checkpoint_dir": str(Path(runtime.get("checkpoint_dir", "checkpoints")).resolve()),
            "debug": bool(runtime.get("debug", False)),
        }
        return adapted
