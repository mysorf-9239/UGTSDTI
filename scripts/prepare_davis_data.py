"""
Prepare DAVIS dataset từ data/raw/davis.tab thành format artifacts cho new-design pipeline.

4 scenarios theo chuẩn TDC:
  s1 — random split
  s2 — cold drug  (unseen drugs in val/test)
  s3 — cold target (unseen targets in val/test)
  s4 — cold both  (unseen drugs AND targets)

Usage:
    conda run -n ugtsdti python scripts/prepare_davis_data.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RAW_TAB = Path("data/raw/davis.tab")
PROCESSED_DIR = Path("data/processed/davis/v1")
SPLITS_DIR = Path("data/splits/davis/v1/v1")
SEED = 42
FRAC_TRAIN = 0.7
FRAC_VAL = 0.15
# FRAC_TEST   = 0.15 (remainder)

# Encoding alphabets (same as bootstrap.py)
SMILES_ALPHABET = "#%()+-./0123456789=@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz"
PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
DRUG_MAX_LEN = 64
PROTEIN_MAX_LEN = 512

# Label: Kd ≤ 30nM → positive (log10 scale: pKd ≥ 7.52 ≈ threshold 30)
# TDC convention: Y là Kd (nM), nhỏ hơn = tốt hơn
# Dùng threshold = 30 nM (Kd ≤ 30 → positive)
KD_THRESHOLD = 30.0


def encode_text(text: str, alphabet: str, max_len: int) -> list[int]:
    vocab = {c: i + 2 for i, c in enumerate(alphabet)}
    encoded = [vocab.get(c, 1) for c in str(text)[:max_len]]
    encoded += [0] * (max_len - len(encoded))
    return encoded


def make_record(row: pd.Series, scenario: str) -> dict:
    return {
        "drug_seq": encode_text(row["X1"], SMILES_ALPHABET, DRUG_MAX_LEN),
        "protein_seq": encode_text(row["X2"], PROTEIN_ALPHABET, PROTEIN_MAX_LEN),
        "drug_graph": None,  # GNN encoder handles missing via allow_missing_input
        "labels": 1.0 if float(row["Y"]) <= KD_THRESHOLD else 0.0,
        "scenario": scenario,
        "drug_id": str(row["ID1"]),
        "target_id": str(row["ID2"]),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n",
        encoding="utf-8",
    )


def split_random(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * FRAC_TRAIN)
    n_val = int(n * FRAC_VAL)
    return df[:n_train], df[n_train : n_train + n_val], df[n_train + n_val :]


def split_cold_drug(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drugs = df["ID1"].unique().tolist()
    random.seed(seed)
    random.shuffle(drugs)
    n = len(drugs)
    train_drugs = set(drugs[: int(n * FRAC_TRAIN)])
    val_drugs = set(drugs[int(n * FRAC_TRAIN) : int(n * (FRAC_TRAIN + FRAC_VAL))])
    train = df[df["ID1"].isin(train_drugs)]
    val = df[df["ID1"].isin(val_drugs)]
    test = df[~df["ID1"].isin(train_drugs | val_drugs)]
    return train, val, test


def split_cold_target(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = df["ID2"].unique().tolist()
    random.seed(seed)
    random.shuffle(targets)
    n = len(targets)
    train_targets = set(targets[: int(n * FRAC_TRAIN)])
    val_targets = set(targets[int(n * FRAC_TRAIN) : int(n * (FRAC_TRAIN + FRAC_VAL))])
    train = df[df["ID2"].isin(train_targets)]
    val = df[df["ID2"].isin(val_targets)]
    test = df[~df["ID2"].isin(train_targets | val_targets)]
    return train, val, test


def split_cold_both(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    drugs = df["ID1"].unique().tolist()
    targets = df["ID2"].unique().tolist()
    random.seed(seed)
    random.shuffle(drugs)
    random.shuffle(targets)
    nd, nt = len(drugs), len(targets)
    train_drugs = set(drugs[: int(nd * FRAC_TRAIN)])
    val_drugs = set(drugs[int(nd * FRAC_TRAIN) : int(nd * (FRAC_TRAIN + FRAC_VAL))])
    train_targets = set(targets[: int(nt * FRAC_TRAIN)])
    val_targets = set(targets[int(nt * FRAC_TRAIN) : int(nt * (FRAC_TRAIN + FRAC_VAL))])
    train = df[df["ID1"].isin(train_drugs) & df["ID2"].isin(train_targets)]
    val = df[df["ID1"].isin(val_drugs) & df["ID2"].isin(val_targets)]
    test = df[~df["ID1"].isin(train_drugs | val_drugs) & ~df["ID2"].isin(train_targets | val_targets)]
    return train, val, test


def main() -> None:
    print(f"Loading {RAW_TAB}...")
    df = pd.read_csv(RAW_TAB, sep="\t")
    print(f"  Total records: {len(df)}")

    pos = (df["Y"] <= KD_THRESHOLD).sum()
    print(f"  Positives (Kd ≤ {KD_THRESHOLD} nM): {pos} ({100*pos/len(df):.1f}%)")
    print(f"  Unique drugs: {df['ID1'].nunique()}, unique targets: {df['ID2'].nunique()}")

    scenarios = {
        "s1": split_random,
        "s2": split_cold_drug,
        "s3": split_cold_target,
        "s4": split_cold_both,
    }

    scenario_partitions: dict[str, dict[str, str]] = {}
    counts: dict[str, dict[str, int]] = {}

    for scenario, split_fn in scenarios.items():
        print(f"\nBuilding {scenario}...")
        train_df, val_df, test_df = split_fn(df, SEED)

        partitions: dict[str, list[dict]] = {
            "train": [make_record(row, scenario) for _, row in train_df.iterrows()],
            "val": [make_record(row, scenario) for _, row in val_df.iterrows()],
            "test": [make_record(row, scenario) for _, row in test_df.iterrows()],
        }

        scenario_partitions[scenario] = {}
        counts[scenario] = {}
        for partition, rows in partitions.items():
            path = SPLITS_DIR / scenario / f"{partition}.jsonl"
            write_jsonl(path, rows)
            scenario_partitions[scenario][partition] = str(path)
            counts[scenario][partition] = len(rows)
            pos_count = sum(1 for r in rows if r["labels"] == 1.0)
            print(f"  {partition}: {len(rows)} records, {pos_count} pos ({100*pos_count/max(len(rows),1):.1f}%)")

    # dataset_version.json
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset_version = {
        "dataset": "davis",
        "dataset_version": "davis-tdc-v1",
        "preprocessing_version": "v1",
        "record_count": len(df),
        "feature_keys": ["drug_seq", "protein_seq", "drug_graph", "labels", "scenario"],
        "kd_threshold_nm": KD_THRESHOLD,
        "seed": SEED,
    }
    (PROCESSED_DIR / "dataset_version.json").write_text(
        json.dumps(dataset_version, indent=2, sort_keys=True), encoding="utf-8"
    )

    # manifest.json
    manifest = {
        "dataset": "davis",
        "preprocessing_version": "v1",
        "split_version": "v1",
        "seed": SEED,
        "scenarios": list(scenarios.keys()),
        "partitions": ["train", "val", "test"],
        "scenario_partitions": scenario_partitions,
        "counts": counts,
        "protocol_report": {
            "protocol_version": "davis-tdc.v1",
            "kd_threshold_nm": KD_THRESHOLD,
            "s1": "random split",
            "s2": "cold drug",
            "s3": "cold target",
            "s4": "cold drug+target",
        },
    }
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print("\nDone.")
    print(f"  dataset_version → {PROCESSED_DIR / 'dataset_version.json'}")
    print(f"  manifest        → {SPLITS_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
