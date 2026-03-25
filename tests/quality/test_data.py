from unittest.mock import patch

import pandas as pd
import pytest
import torch

from ugtsdti.data.transforms.chemistry import smiles_to_graph
from ugtsdti.data.transforms.sequence import ESMSequenceTokenizer

try:
    from torch_geometric.loader import DataLoader

    from ugtsdti.data.datasets.tdc_dataset import TDCCachingDataset

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def test_smiles_to_graph():
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    data = smiles_to_graph(smiles)

    assert data is not None
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")
    assert data.x.size(0) == 13
    assert data.x.size(1) == 7


def test_invalid_smiles():
    invalid_smiles = "invalid_chemistry"
    data = smiles_to_graph(invalid_smiles)
    assert data is None


@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_sequence_tokenizer(MockAutoTokenizer):
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 128)), "attention_mask": torch.ones((1, 128))}

    tokenizer = ESMSequenceTokenizer(model_name="facebook/esm2_t6_8M_UR50D", max_length=128)
    fasta = "MVLSPADKTN"
    tokens = tokenizer.encode(fasta)

    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert tokens["input_ids"].dim() == 1
    assert tokens["input_ids"].size(0) == 128
    assert tokens["attention_mask"].size(0) == 128


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tdc_caching_and_batching(MockAutoTokenizer, tmp_path):
    cache_dir = str(tmp_path / "data_cache")

    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 1024)), "attention_mask": torch.ones((1, 1024))}

    mock_df = pd.DataFrame(
        {
            "Drug": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"] * 2,
            "Target": ["MVLSPADKTN", "MFLS"] * 2,
            "Y": [7.0, 5.0, 6.0, 8.0],
        }
    )

    with patch("ugtsdti.data.datasets.tdc_dataset.DTI") as MockDTI:
        instance = MockDTI.return_value
        instance.get_split.return_value = {"train": mock_df, "valid": mock_df, "test": mock_df}

        dataset = TDCCachingDataset(
            name="DAVIS", split="train", split_type="cold_split", cache_dir=cache_dir, seed=42, frac=[0.8, 0.1, 0.1]
        )

    assert len(dataset) == 4
    sample = dataset[0]

    assert "drug" in sample
    assert "target_ids" in sample
    assert "label" in sample
    assert "drug_index" in sample
    assert "target_index" in sample

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    assert hasattr(batch["drug"], "batch")


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tdc_bundle_cache_builds_all_splits_once(MockAutoTokenizer, tmp_path):
    cache_dir = str(tmp_path / "data_cache_bundle")

    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 64)), "attention_mask": torch.ones((1, 64))}

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

        train_dataset = TDCCachingDataset(
            name="DAVIS", split="train", split_type="cold_split", cache_dir=cache_dir, seed=42, frac=[0.8, 0.1, 0.1]
        )
        valid_dataset = TDCCachingDataset(
            name="DAVIS", split="valid", split_type="cold_split", cache_dir=cache_dir, seed=42, frac=[0.8, 0.1, 0.1]
        )

    assert len(train_dataset) == 2
    assert len(valid_dataset) == 2
    assert instance.get_split.call_count == 1
    assert (tmp_path / "data_cache_bundle" / "DAVIS_s2_42" / "valid" / "dataset.pt").exists()
    assert (tmp_path / "data_cache_bundle" / "DAVIS_s2_42" / "test" / "dataset.pt").exists()
    assert (tmp_path / "data_cache_bundle" / "DAVIS_s2_42" / "bundle_metadata.pt").exists()
