from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import torch

from ugtsdti.data.datasets.tdc_dataset import normalize_tdc_column_name, normalize_tdc_split_method

if TYPE_CHECKING:
    import pandas as pd

try:
    from tdc.multi_pred import DTI
except ImportError:
    DTI = None


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def summarize_split_frames(
    split_dict: dict[str, pd.DataFrame], scenario_name: str, dataset_name: str
) -> dict[str, Any]:
    """Summarize overlap semantics directly from PyTDC split frames."""
    train_df = split_dict["train"]
    valid_df = split_dict.get("valid")
    test_df = split_dict.get("test")

    train_drugs = set(train_df["Drug"].tolist())
    train_targets = set(train_df["Target"].tolist())

    summary: dict[str, Any] = {
        "dataset_name": dataset_name,
        "scenario_name": scenario_name,
        "train_pairs": int(len(train_df)),
        "train_unique_drugs": int(len(train_drugs)),
        "train_unique_targets": int(len(train_targets)),
    }

    for split_name, split_df in (("valid", valid_df), ("test", test_df)):
        if split_df is None:
            continue
        eval_drugs = set(split_df["Drug"].tolist())
        eval_targets = set(split_df["Target"].tolist())
        drug_overlap = len(train_drugs & eval_drugs)
        target_overlap = len(train_targets & eval_targets)
        summary.update(
            {
                f"{split_name}_pairs": int(len(split_df)),
                f"{split_name}_unique_drugs": int(len(eval_drugs)),
                f"{split_name}_unique_targets": int(len(eval_targets)),
                f"{split_name}_drug_overlap_count": int(drug_overlap),
                f"{split_name}_target_overlap_count": int(target_overlap),
                f"{split_name}_drug_overlap_ratio": _safe_ratio(drug_overlap, len(eval_drugs)),
                f"{split_name}_target_overlap_ratio": _safe_ratio(target_overlap, len(eval_targets)),
            }
        )

    return summary


def probe_tdc_split(
    *,
    name: str,
    split_type: str,
    column_name: str | Sequence[str] | None,
    frac: list[float] | None,
    seed: int,
    scenario_name: str,
) -> dict[str, Any]:
    """Probe PyTDC split semantics without building cached model features."""
    if DTI is None:
        raise ImportError("PyTDC is not installed. Run `pip install PyTDC`.")

    dataset = DTI(name=name)
    split_kwargs: dict[str, Any] = {
        "method": normalize_tdc_split_method(split_type),
        "frac": frac or [0.8, 0.1, 0.1],
        "seed": seed,
    }
    normalized_column_name = normalize_tdc_column_name(column_name)
    if normalized_column_name is not None:
        split_kwargs["column_name"] = normalized_column_name

    split_dict = dataset.get_split(**split_kwargs)
    return summarize_split_frames(split_dict, scenario_name=scenario_name, dataset_name=name)


def write_probe_summary(summary: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(summary, output_path)
