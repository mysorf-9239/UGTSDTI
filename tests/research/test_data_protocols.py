from unittest.mock import patch

import pandas as pd
import pytest
import torch

try:
    from ugtsdti.data.datasets.tdc_dataset import TDCCachingDataset

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
def test_tdc_s4_omits_column_name_for_global_cold_split(MockAutoTokenizer, tmp_path):
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
            column_name=None,
            cache_dir=cache_dir,
            seed=42,
            scenario_name="s4",
        )

    _, kwargs = instance.get_split.call_args
    assert kwargs["method"] == "cold_split"
    assert "column_name" not in kwargs
