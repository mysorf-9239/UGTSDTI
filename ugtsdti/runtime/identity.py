"""Experiment identity and reproducibility-key helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

_RUNTIME_OPERATIONAL_HASH_FIELDS = {
    "artifacts_dir",
    "checkpoint_dir",
    "checkpoint_path",
    "data_dir",
}


@dataclass(frozen=True)
class ExperimentIdentity:
    """Unique run identity kept separate from reproducibility tuple."""

    run_id: str
    config_hash: str
    git_commit: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_experiment_identity(config: dict[str, Any]) -> ExperimentIdentity:
    """Create a unique run identity for the current execution."""
    return ExperimentIdentity(
        run_id=uuid.uuid4().hex,
        config_hash=hash_config(config),
        git_commit=current_git_commit(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def hash_config(config: dict[str, Any]) -> str:
    payload = json.dumps(_canonicalize_config_for_hash(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def current_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_reproducibility_key(
    *,
    config_hash: str,
    dataset_version: str,
    preprocessing_version: str,
    split_version: str,
    seed: int,
) -> str:
    payload = {
        "config_hash": config_hash,
        "dataset_version": dataset_version,
        "preprocessing_version": preprocessing_version,
        "split_version": split_version,
        "seed": int(seed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonicalize_config_for_hash(config: dict[str, Any]) -> dict[str, Any]:
    canonical = deepcopy(config)
    runtime = canonical.get("runtime")
    if isinstance(runtime, dict):
        for key in _RUNTIME_OPERATIONAL_HASH_FIELDS:
            runtime.pop(key, None)
    normalized = _canonicalize_value(canonical)
    if isinstance(normalized, dict):
        return normalized
    return {"config": normalized}


def _canonicalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            canonical[str(key)] = _canonicalize_mapping_entry(str(key), item)
        return canonical
    if isinstance(value, list):
        return [_canonicalize_value(item) for item in value]
    return value


def _canonicalize_mapping_entry(key: str, value: Any) -> Any:
    canonical = _canonicalize_value(value)
    if not isinstance(canonical, list):
        return canonical

    if key in {"eval", "available", "uses", "enabled"}:
        return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))

    return canonical
