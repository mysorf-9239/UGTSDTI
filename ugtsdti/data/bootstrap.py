"""Config-driven data bootstrap before artifact-backed runtime execution."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ugtsdti.core.errors import MissingRawSnapshotError, ProcessedSplitMismatchError, UGTSDTIError
from ugtsdti.data.acquisition import DataAcquisition
from ugtsdti.data.preprocessing import DataPreprocessor
from ugtsdti.data.splitting import DataSplitter
from ugtsdti.data.validate import DataValidator

_SMILES_ALPHABET = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
_PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"


@dataclass(frozen=True)
class BootstrapReport:
    """Summarize bootstrap behavior for logging and tests."""

    source_type: str
    auto_prepare: bool
    prepared: bool
    dataset_version_path: str
    split_manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "auto_prepare": self.auto_prepare,
            "prepared": self.prepared,
            "dataset_version_path": self.dataset_version_path,
            "split_manifest_path": self.split_manifest_path,
        }


class DataBootstrapOrchestrator:
    """Ensure required processed/split artifacts exist before runtime execution."""

    def ensure_artifacts(self, cfg: dict[str, Any], *, data_root: str | Path) -> BootstrapReport:
        data_cfg = dict(cfg.get("data", {}))
        source_cfg = dict(data_cfg.get("source", {}))
        source_type = str(source_cfg.get("type", "artifacts")).lower()
        auto_prepare = bool(source_cfg.get("auto_prepare", False))

        dataset = str(data_cfg.get("dataset", "dataset"))
        preprocessing_version = str(data_cfg.get("preprocessing_version", "v1"))
        split_version = str(data_cfg.get("split_version", "v1"))
        data_dir = Path(data_root)

        dataset_version_path = data_dir / "processed" / dataset / preprocessing_version / "dataset_version.json"
        split_manifest_path = data_dir / "splits" / dataset / preprocessing_version / split_version / "manifest.json"

        if self._artifacts_exist(dataset_version_path, split_manifest_path):
            self._validate_artifacts(dataset_version_path, split_manifest_path)
            return BootstrapReport(
                source_type=source_type,
                auto_prepare=auto_prepare,
                prepared=False,
                dataset_version_path=str(dataset_version_path),
                split_manifest_path=str(split_manifest_path),
            )

        if not auto_prepare:
            raise ProcessedSplitMismatchError(
                "Required processed/split artifacts are missing and data.source.auto_prepare is disabled.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key="artifacts",
                debug_payload={
                    "dataset_version_path": str(dataset_version_path),
                    "split_manifest_path": str(split_manifest_path),
                    "source_type": source_type,
                },
            )

        records = self._load_source_records(source_type=source_type, source_cfg=source_cfg, dataset=dataset)
        raw_root = data_dir / "raw"
        processed_root = data_dir / "processed"
        splits_root = data_dir / "splits"
        raw_snapshot = DataAcquisition(raw_root=raw_root).export_raw_snapshot(dataset, records=records)
        records_path, _ = DataPreprocessor(processed_root=processed_root).materialize(
            dataset,
            raw_snapshot=raw_snapshot,
            preprocessing_version=preprocessing_version,
            transform_fn=lambda row: self._transform_row(
                row,
                drug_max_len=int(source_cfg.get("drug_max_len", 64)),
                protein_max_len=int(source_cfg.get("protein_max_len", 512)),
                label_threshold=float(source_cfg.get("label_threshold", 7.0)),
                label_order=str(source_cfg.get("label_order", "descending")),
            ),
        )
        materialized_rows = self._read_jsonl(records_path)
        DataSplitter(splits_root=splits_root).create_splits(
            dataset,
            records=materialized_rows,
            preprocessing_version=preprocessing_version,
            split_version=split_version,
            seed=int(cfg.get("runtime", {}).get("seed", 0)),
        )
        self._validate_artifacts(dataset_version_path, split_manifest_path)
        return BootstrapReport(
            source_type=source_type,
            auto_prepare=auto_prepare,
            prepared=True,
            dataset_version_path=str(dataset_version_path),
            split_manifest_path=str(split_manifest_path),
        )

    def _artifacts_exist(self, dataset_version_path: Path, split_manifest_path: Path) -> bool:
        return dataset_version_path.exists() and split_manifest_path.exists()

    def _validate_artifacts(self, dataset_version_path: Path, split_manifest_path: Path) -> None:
        DataValidator().validate_artifacts(
            dataset_version_path=dataset_version_path,
            split_manifest_path=split_manifest_path,
        )

    def _load_source_records(
        self,
        *,
        source_type: str,
        source_cfg: dict[str, Any],
        dataset: str,
    ) -> list[dict[str, Any]]:
        if source_type == "artifacts":
            raise ProcessedSplitMismatchError(
                "Artifacts source cannot auto-prepare missing artifacts.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key="data.source.type",
            )
        if source_type == "csv":
            raw_csv = source_cfg.get("raw_csv")
            if not raw_csv:
                raise MissingRawSnapshotError(
                    "CSV bootstrap requires data.source.raw_csv.",
                    stage="data_bootstrap",
                    component="DataBootstrapOrchestrator",
                    key="data.source.raw_csv",
                )
            return self._read_csv_rows(Path(str(raw_csv)))
        if source_type == "pytdc":
            try:
                from tdc.multi_pred import DTI
            except Exception as exc:  # pragma: no cover - optional dependency path
                raise UGTSDTIError(
                    "PyTDC source requested but PyTDC is unavailable.",
                    stage="data_bootstrap",
                    component="DataBootstrapOrchestrator",
                    key="data.source.type",
                ) from exc
            dataset_name = str(source_cfg.get("tdc_name") or dataset).upper()
            dti = DTI(name=dataset_name)
            frame = dti.get_data()
            if hasattr(frame, "to_dict"):
                return list(frame.to_dict(orient="records"))
            raise UGTSDTIError(
                f"PyTDC dataset {dataset_name!r} did not return a tabular payload.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key=dataset_name,
            )
        raise UGTSDTIError(
            f"Unsupported data.source.type {source_type!r}.",
            stage="data_bootstrap",
            component="DataBootstrapOrchestrator",
            key="data.source.type",
        )

    def _read_csv_rows(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _transform_row(
        self,
        row: dict[str, Any],
        *,
        drug_max_len: int,
        protein_max_len: int,
        label_threshold: float,
        label_order: str,
    ) -> dict[str, Any]:
        drug_text = self._first_present(row, "drug", "Drug", "drug_seq", "SMILES", "Drug_SMILES")
        protein_text = self._first_present(row, "target", "Target", "protein_seq", "Target Sequence", "Target_Sequence")
        if not drug_text or not protein_text:
            raise UGTSDTIError(
                "Each bootstrap row must provide drug and protein sequence/text fields.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key="row",
            )
        raw_label = self._first_present(row, "labels", "label", "Label", "Y", "y")
        if raw_label is None:
            raise UGTSDTIError(
                "Each bootstrap row must provide a label column such as labels/Y/Label.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key="labels",
            )
        return {
            "drug_id": self._first_present(row, "drug_id", "Drug_ID", "drug", "Drug", default=drug_text),
            "protein_id": self._first_present(
                row,
                "protein_id",
                "Target_ID",
                "target",
                "Target",
                default=protein_text,
            ),
            "drug_seq": self._encode_text(str(drug_text), alphabet=_SMILES_ALPHABET, max_len=drug_max_len),
            "protein_seq": self._encode_text(str(protein_text), alphabet=_PROTEIN_ALPHABET, max_len=protein_max_len),
            "labels": self._binary_label(raw_label, threshold=label_threshold, order=label_order),
            "raw_label": self._maybe_float(raw_label),
        }

    def _first_present(self, row: dict[str, Any], *keys: str, default: Any | None = None) -> Any | None:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
        return default

    def _binary_label(self, value: Any, *, threshold: float, order: str) -> float:
        numeric = self._maybe_float(value)
        if numeric is None:
            text = str(value).strip().lower()
            if text in {"1", "true", "positive", "pos"}:
                return 1.0
            if text in {"0", "false", "negative", "neg"}:
                return 0.0
            raise UGTSDTIError(
                f"Could not interpret label value {value!r} as binary or numeric.",
                stage="data_bootstrap",
                component="DataBootstrapOrchestrator",
                key="labels",
            )
        if order == "descending":
            return 1.0 if numeric <= threshold else 0.0
        return 1.0 if numeric >= threshold else 0.0

    def _maybe_float(self, value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _encode_text(self, text: str, *, alphabet: str, max_len: int) -> list[int]:
        vocab = {char: index + 2 for index, char in enumerate(alphabet)}
        unknown = 1
        encoded = [vocab.get(char, unknown) for char in text[:max_len]]
        if len(encoded) < max_len:
            encoded.extend([0] * (max_len - len(encoded)))
        return encoded
