from unittest.mock import patch

import pandas as pd
import pytest
import torch

try:
    from ugtsdti.data.datasets.tdc_dataset import (
        TDCCachingDataset,
        normalize_tdc_column_name,
        normalize_tdc_split_method,
    )
    from ugtsdti.data.protocols.audit import summarize_split_overlap

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tdc_uses_train_based_global_indices_with_unknown_slot(MockAutoTokenizer, tmp_path):
    cache_dir = str(tmp_path / "data_cache_global_vocab")

    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 32)), "attention_mask": torch.ones((1, 32))}

    train_df = pd.DataFrame(
        {
            "Drug": ["CCO", "CCN"],
            "Target": ["AAAA", "BBBB"],
            "Y": [8.0, 6.0],
        }
    )
    valid_df = pd.DataFrame(
        {
            "Drug": ["CCC"],
            "Target": ["CCCC"],
            "Y": [9.0],
        }
    )

    with patch("ugtsdti.data.datasets.tdc_dataset.DTI") as MockDTI:
        instance = MockDTI.return_value
        instance.get_split.return_value = {"train": train_df, "valid": valid_df, "test": valid_df}

        train_dataset = TDCCachingDataset(
            name="DAVIS", split="train", split_type="cold_split", cache_dir=cache_dir, seed=42, frac=[0.8, 0.1, 0.1]
        )
        val_dataset = TDCCachingDataset(
            name="DAVIS", split="valid", split_type="cold_split", cache_dir=cache_dir, seed=42, frac=[0.8, 0.1, 0.1]
        )

    assert train_dataset.num_unique_drugs == 3
    assert train_dataset.num_unique_targets == 3
    assert train_dataset[0]["label"].item() == 1.0
    assert train_dataset[1]["label"].item() == 0.0
    assert val_dataset[0]["drug_index"].item() == train_dataset.unk_drug_index
    assert val_dataset[0]["target_index"].item() == train_dataset.unk_target_index
    assert val_dataset[0]["label"].item() == 1.0


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tdc_s4_passes_both_entity_columns_for_global_cold_split(MockAutoTokenizer, tmp_path):
    cache_dir = str(tmp_path / "data_cache_s4")

    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 32)), "attention_mask": torch.ones((1, 32))}

    split_df = pd.DataFrame(
        {
            "Drug": ["CCO", "CCN"],
            "Target": ["AAAA", "BBBB"],
            "Y": [1.0, 0.0],
        }
    )

    with patch("ugtsdti.data.datasets.tdc_dataset.DTI") as MockDTI:
        instance = MockDTI.return_value
        instance.get_split.return_value = {"train": split_df, "valid": split_df, "test": split_df}

        TDCCachingDataset(
            name="DAVIS",
            split="train",
            split_type="cold_split",
            column_name=["Drug", "Target"],
            cache_dir=cache_dir,
            seed=42,
            scenario_name="s4",
        )

    _, kwargs = instance.get_split.call_args
    assert kwargs["method"] == "cold_split"
    assert kwargs["column_name"] == ["Drug", "Target"]


def test_summarize_split_overlap_matches_scenario_semantics():
    class DummySplit:
        def __init__(self, scenario_name, drugs, targets):
            self.scenario_name = scenario_name
            self.split_unique_smiles = drugs
            self.split_unique_fasta = targets

    train = DummySplit("train", ["d1", "d2"], ["t1", "t2"])
    s2_eval = DummySplit("s2", ["d3", "d4"], ["t1", "t2"])
    s3_eval = DummySplit("s3", ["d1", "d2"], ["t3"])
    s4_eval = DummySplit("s4", ["d5"], ["t4"])

    s2_audit = summarize_split_overlap(train, s2_eval, split_name="test")
    s3_audit = summarize_split_overlap(train, s3_eval, split_name="test")
    s4_audit = summarize_split_overlap(train, s4_eval, split_name="test")

    assert s2_audit["drug_overlap_count"] == 0
    assert s2_audit["target_overlap_count"] == 2
    assert s3_audit["drug_overlap_count"] == 2
    assert s3_audit["target_overlap_count"] == 0
    assert s4_audit["drug_overlap_count"] == 0
    assert s4_audit["target_overlap_count"] == 0


def test_normalize_tdc_split_method_accepts_repo_aliases():
    assert normalize_tdc_split_method("random_split") == "random"
    assert normalize_tdc_split_method("random") == "random"
    assert normalize_tdc_split_method("cold_split") == "cold_split"


def test_normalize_tdc_column_name_accepts_multi_column_protocol():
    assert normalize_tdc_column_name(("Drug", "Target")) == ["Drug", "Target"]
    assert normalize_tdc_column_name(["Drug", "Target"]) == ["Drug", "Target"]


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tdc_bundle_metadata_captures_scenario_provenance(MockAutoTokenizer, tmp_path):
    cache_dir = str(tmp_path / "data_cache_metadata")

    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 32)), "attention_mask": torch.ones((1, 32))}

    split_df = pd.DataFrame(
        {
            "Drug": ["CCO", "CCN"],
            "Target": ["AAAA", "BBBB"],
            "Y": [1.0, 0.0],
        }
    )

    with patch("ugtsdti.data.datasets.tdc_dataset.DTI") as MockDTI:
        instance = MockDTI.return_value
        instance.get_split.return_value = {"train": split_df, "valid": split_df, "test": split_df}

        TDCCachingDataset(
            name="DAVIS",
            split="train",
            split_type="cold_split",
            column_name=["Drug", "Target"],
            cache_dir=cache_dir,
            seed=42,
            scenario_name="s4",
        )

    metadata = torch.load(tmp_path / "data_cache_metadata" / "DAVIS_s4_42" / "bundle_metadata.pt", weights_only=False)
    assert metadata["scenario_name"] == "s4"
    assert metadata["split_method"] == "cold_split"
    assert metadata["column_name"] == ["Drug", "Target"]
    assert metadata["splits"]["train"]["pairs"] == 2
    assert metadata["splits"]["valid"]["pairs"] == 2
