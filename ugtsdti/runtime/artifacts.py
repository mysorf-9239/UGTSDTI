"""Artifact bundle writer for reproducibility outputs."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


class ArtifactWriter:
    """Materialize a run-level artifact bundle under ``artifacts/<run_id>/``."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self._root = Path(root)

    def write_bundle(
        self,
        *,
        identity: dict[str, Any],
        config: dict[str, Any],
        metrics: dict[str, Any],
        diagnostics: dict[str, Any],
        split_manifest: dict[str, Any],
        model_state: dict[str, Any],
        execution_trace: dict[str, Any] | None = None,
        state_boundary_summaries: dict[str, Any] | None = None,
        logs_dir: str | Path | None = None,
        bundle_kind: Literal["final", "snapshot"] = "final",
        snapshot_label: str | None = None,
    ) -> Path:
        run_id = str(identity["run_id"])
        run_root = self._root / run_id
        bundle_dir = run_root
        if bundle_kind == "snapshot":
            label = snapshot_label or "snapshot"
            bundle_dir = run_root / "snapshots" / label
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "logs").mkdir(exist_ok=True)

        self._write_yaml(bundle_dir / "config.yaml", config)
        self._write_json(bundle_dir / "identity.json", identity)
        self._write_json(bundle_dir / "metrics.json", metrics)
        self._write_json(bundle_dir / "diagnostics.json", diagnostics)
        self._write_json(bundle_dir / "split_manifest.json", split_manifest)
        self._write_model(bundle_dir / "model.pt", model_state)

        if execution_trace is not None:
            self._write_json(bundle_dir / "execution_trace.json", execution_trace)
        if state_boundary_summaries is not None:
            self._write_json(bundle_dir / "state_boundaries.json", state_boundary_summaries)
        if logs_dir is not None:
            self._copy_logs(Path(logs_dir), bundle_dir / "logs")

        self._write_manifest(run_root)
        return bundle_dir

    def _write_manifest(self, run_root: Path) -> None:
        snapshots_dir = run_root / "snapshots"
        snapshot_labels = (
            sorted(path.name for path in snapshots_dir.iterdir() if path.is_dir()) if snapshots_dir.exists() else []
        )
        payload = {
            "canonical_bundle": "." if (run_root / "identity.json").exists() else None,
            "latest_snapshot": snapshot_labels[-1] if snapshot_labels else None,
            "snapshot_labels": snapshot_labels,
        }
        self._write_json(run_root / "artifact_manifest.json", payload)

    def _write_yaml(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, yaml.safe_dump(_serialize_payload(payload), sort_keys=True))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(_serialize_payload(payload), sort_keys=True, indent=2))

    def _write_model(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            import torch

            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return
        except ImportError:
            self._write_json(path, payload)

    def _copy_logs(self, source: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for entry in source.iterdir():
            target = destination / entry.name
            if entry.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)

    def _atomic_write(self, path: Path, payload: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)


def _serialize_payload(value: Any) -> Any:
    """Convert common runtime objects into JSON/YAML-safe data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _serialize_payload(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serialize_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_payload(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_payload(value.to_dict())

    try:
        import torch

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                return float(tensor.item())
            return tensor.tolist()
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    return str(value)
