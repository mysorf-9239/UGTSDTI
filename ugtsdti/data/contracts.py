"""Data contracts for materialized datasets and split manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetVersion:
    """Version metadata for a materialized dataset snapshot."""

    dataset: str
    dataset_version: str
    preprocessing_version: str
    record_count: int
    feature_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["raw_version"] = self.dataset_version
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetVersion":
        dataset_version = payload.get("dataset_version", payload.get("raw_version"))
        if dataset_version is None:
            raise KeyError("dataset_version")
        return cls(
            dataset=str(payload["dataset"]),
            dataset_version=str(dataset_version),
            preprocessing_version=str(payload["preprocessing_version"]),
            record_count=int(payload["record_count"]),
            feature_keys=list(payload.get("feature_keys", [])),
        )

    @property
    def raw_version(self) -> str:
        """Backward-compatible alias for older artifacts/spec wording."""
        return self.dataset_version


@dataclass(frozen=True)
class SplitManifest:
    """Persistent metadata for deterministic split artifacts."""

    dataset: str
    preprocessing_version: str
    split_version: str
    seed: int
    scenarios: list[str]
    split_paths: dict[str, str]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitManifest":
        return cls(
            dataset=str(payload["dataset"]),
            preprocessing_version=str(payload["preprocessing_version"]),
            split_version=str(payload["split_version"]),
            seed=int(payload["seed"]),
            scenarios=list(payload.get("scenarios", [])),
            split_paths=dict(payload.get("split_paths", {})),
            counts={str(key): int(value) for key, value in dict(payload.get("counts", {})).items()},
        )
