from __future__ import annotations

from typing import Any


def _safe_ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def summarize_split_overlap(train_dataset: Any, eval_dataset: Any, split_name: str) -> dict[str, float | int | str]:
    """Summarize entity overlap between train and another split.

    This audit is used to sanity-check the intended cold-start protocol:
    - S1 should have overlap on both drugs and targets.
    - S2 should have zero drug overlap.
    - S3 should have zero target overlap.
    - S4 should have zero overlap on both.
    """
    train_drugs = set(getattr(train_dataset, "split_unique_smiles", []) or [])
    train_targets = set(getattr(train_dataset, "split_unique_fasta", []) or [])
    eval_drugs = set(getattr(eval_dataset, "split_unique_smiles", []) or [])
    eval_targets = set(getattr(eval_dataset, "split_unique_fasta", []) or [])

    drug_overlap = len(train_drugs & eval_drugs)
    target_overlap = len(train_targets & eval_targets)

    return {
        "split": split_name,
        "scenario": getattr(eval_dataset, "scenario_name", "unknown"),
        "train_drugs": len(train_drugs),
        "train_targets": len(train_targets),
        "eval_drugs": len(eval_drugs),
        "eval_targets": len(eval_targets),
        "drug_overlap_count": drug_overlap,
        "target_overlap_count": target_overlap,
        "drug_overlap_ratio": _safe_ratio(drug_overlap, len(eval_drugs)),
        "target_overlap_ratio": _safe_ratio(target_overlap, len(eval_targets)),
        "is_cold_drug": int(drug_overlap == 0),
        "is_cold_target": int(target_overlap == 0),
    }
