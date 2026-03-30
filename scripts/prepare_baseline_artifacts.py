"""Prepare artifact-backed baseline data from raw CSV or optional PyTDC."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ugtsdti.data.acquisition import DataAcquisition  # noqa: E402
from ugtsdti.data.preprocessing import DataPreprocessor  # noqa: E402
from ugtsdti.data.splitting import DataSplitter  # noqa: E402

_SMILES_ALPHABET = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
_PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare processed baseline artifacts for artifact-backed train/eval runs."
    )
    parser.add_argument("--source", choices=("csv", "pytdc"), default="csv")
    parser.add_argument("--dataset", default="davis")
    parser.add_argument("--tdc-name", default=None, help="Optional PyTDC dataset name. Defaults to upper(dataset).")
    parser.add_argument("--raw-csv", default=None, help="Raw CSV path when --source=csv.")
    parser.add_argument("--data-dir", default="data", help="Root data directory containing raw/processed/splits.")
    parser.add_argument("--preprocessing-version", default="v1")
    parser.add_argument("--split-version", default="v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drug-max-len", type=int, default=64)
    parser.add_argument("--protein-max-len", type=int, default=512)
    parser.add_argument("--label-threshold", type=float, default=30.0)
    parser.add_argument(
        "--label-order",
        choices=("descending", "ascending"),
        default="descending",
        help="Binary label rule for numeric affinity values.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data_dir = Path(args.data_dir).resolve()
    raw_root = data_dir / "raw"
    processed_root = data_dir / "processed"
    splits_root = data_dir / "splits"

    rows = _load_rows(
        source=args.source,
        dataset=str(args.dataset),
        raw_csv=args.raw_csv,
        tdc_name=args.tdc_name,
        label_threshold=float(args.label_threshold),
        label_order=str(args.label_order),
    )
    raw_snapshot = DataAcquisition(raw_root=raw_root).export_raw_snapshot(str(args.dataset), records=rows)
    records_path, dataset_version = DataPreprocessor(processed_root=processed_root).materialize(
        str(args.dataset),
        raw_snapshot=raw_snapshot,
        preprocessing_version=str(args.preprocessing_version),
        transform_fn=lambda row: _transform_row(
            row,
            drug_max_len=int(args.drug_max_len),
            protein_max_len=int(args.protein_max_len),
            label_threshold=float(args.label_threshold),
            label_order=str(args.label_order),
        ),
    )

    materialized_rows = _read_jsonl(records_path)
    manifest = DataSplitter(splits_root=splits_root).create_splits(
        str(args.dataset),
        records=materialized_rows,
        preprocessing_version=str(args.preprocessing_version),
        split_version=str(args.split_version),
        seed=int(args.seed),
    )

    summary = {
        "dataset": str(args.dataset),
        "source": str(args.source),
        "raw_snapshot": str(raw_snapshot),
        "records_path": str(records_path),
        "dataset_version_path": str(
            processed_root / str(args.dataset) / str(args.preprocessing_version) / "dataset_version.json"
        ),
        "split_manifest_path": str(
            splits_root
            / str(args.dataset)
            / str(args.preprocessing_version)
            / str(args.split_version)
            / "manifest.json"
        ),
        "record_count": int(dataset_version.record_count),
        "scenarios": list(manifest.scenarios),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_rows(
    *,
    source: str,
    dataset: str,
    raw_csv: str | None,
    tdc_name: str | None,
    label_threshold: float,
    label_order: str,
) -> list[dict[str, Any]]:
    if source == "csv":
        if not raw_csv:
            raise SystemExit("--raw-csv is required when --source=csv")
        return _read_csv_rows(Path(raw_csv))

    try:
        from tdc.multi_pred import DTI
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("PyTDC is not available. Install the acquisition extra or use --source=csv.") from exc

    dataset_name = str(tdc_name or dataset).upper()
    dti = DTI(name=dataset_name)
    if hasattr(dti, "binarize"):
        try:
            dti.binarize(threshold=label_threshold, order=label_order)
        except TypeError:
            dti.binarize(threshold=label_threshold)
    frame = dti.get_data()
    if hasattr(frame, "to_dict"):
        return list(frame.to_dict(orient="records"))
    raise SystemExit(f"PyTDC dataset {dataset_name!r} did not return a tabular payload.")


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _transform_row(
    row: dict[str, Any],
    *,
    drug_max_len: int,
    protein_max_len: int,
    label_threshold: float,
    label_order: str,
) -> dict[str, Any]:
    drug_text = _first_present(row, "drug", "Drug", "drug_seq", "SMILES", "Drug_SMILES")
    protein_text = _first_present(row, "target", "Target", "protein_seq", "Target Sequence", "Target_Sequence")
    if not drug_text or not protein_text:
        raise SystemExit("Each row must provide drug and protein sequence/text fields.")

    raw_label = _first_present(row, "labels", "label", "Label", "Y", "y")
    if raw_label is None:
        raise SystemExit("Each row must provide a label column such as labels/Y/Label.")

    label_value = _binary_label(raw_label, threshold=label_threshold, order=label_order)
    return {
        "drug_id": _first_present(row, "drug_id", "Drug_ID", "drug", "Drug", default=drug_text),
        "protein_id": _first_present(row, "protein_id", "Target_ID", "target", "Target", default=protein_text),
        "drug_seq": _encode_text(str(drug_text), alphabet=_SMILES_ALPHABET, max_len=drug_max_len),
        "protein_seq": _encode_text(str(protein_text), alphabet=_PROTEIN_ALPHABET, max_len=protein_max_len),
        "labels": label_value,
        "raw_label": _maybe_float(raw_label),
    }


def _first_present(row: dict[str, Any], *keys: str, default: Any | None = None) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _binary_label(value: Any, *, threshold: float, order: str) -> float:
    numeric = _maybe_float(value)
    if numeric is None:
        text = str(value).strip().lower()
        if text in {"1", "true", "positive", "pos"}:
            return 1.0
        if text in {"0", "false", "negative", "neg"}:
            return 0.0
        raise SystemExit(f"Could not interpret label value {value!r} as binary or numeric.")
    if order == "descending":
        return 1.0 if numeric <= threshold else 0.0
    return 1.0 if numeric >= threshold else 0.0


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _encode_text(text: str, *, alphabet: str, max_len: int) -> list[int]:
    vocab = {char: index + 2 for index, char in enumerate(alphabet)}
    unknown = 1
    encoded = [vocab.get(char, unknown) for char in text[:max_len]]
    if len(encoded) < max_len:
        encoded.extend([0] * (max_len - len(encoded)))
    return encoded


if __name__ == "__main__":
    raise SystemExit(main())
