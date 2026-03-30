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
    partitions: list[str] = field(default_factory=lambda: ["train", "val", "test"])
    scenario_partitions: dict[str, dict[str, str]] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    protocol_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SplitManifest":
        if "scenario_partitions" not in payload:
            split_paths = dict(payload.get("split_paths", {}))
            counts = {str(key): int(value) for key, value in dict(payload.get("counts", {})).items()}
            scenario_partitions = {str(scenario): {"test": str(path)} for scenario, path in split_paths.items()}
            nested_counts = {str(scenario): {"test": int(counts.get(str(scenario), 0))} for scenario in split_paths}
            partitions = ["test"]
            protocol_report = payload.get("protocol_report", {})
        else:
            scenario_partitions = {
                str(scenario): {str(partition): str(path) for partition, path in dict(partitions).items()}
                for scenario, partitions in dict(payload.get("scenario_partitions", {})).items()
            }
            nested_counts = {
                str(scenario): {str(partition): int(value) for partition, value in dict(partitions).items()}
                for scenario, partitions in dict(payload.get("counts", {})).items()
            }
            partitions = list(payload.get("partitions", ["train", "val", "test"]))
            protocol_report = dict(payload.get("protocol_report", {}))
        return cls(
            dataset=str(payload["dataset"]),
            preprocessing_version=str(payload["preprocessing_version"]),
            split_version=str(payload["split_version"]),
            seed=int(payload["seed"]),
            scenarios=list(payload.get("scenarios", [])),
            partitions=partitions,
            scenario_partitions=scenario_partitions,
            counts=nested_counts,
            protocol_report=protocol_report,
        )

    @property
    def split_paths(self) -> dict[str, str]:
        """Backward-compatible alias for legacy single-file manifests."""
        return {
            scenario: partitions["test"]
            for scenario, partitions in self.scenario_partitions.items()
            if "test" in partitions
        }
