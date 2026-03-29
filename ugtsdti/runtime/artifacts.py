"""Artifact bundle writer for reproducibility outputs."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

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
    ) -> Path:
        run_id = str(identity["run_id"])
        bundle_dir = self._root / run_id
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

        return bundle_dir

    def _write_yaml(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, yaml.safe_dump(payload, sort_keys=True))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        self._atomic_write(path, json.dumps(payload, sort_keys=True, indent=2))

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
