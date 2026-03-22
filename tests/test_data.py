import os
import shutil
from unittest.mock import patch

import pandas as pd
import pytest
import torch

# Ensure utilities are importable
from ugtsdti.utils.chemistry import smiles_to_graph
from ugtsdti.utils.sequence import ESMSequenceTokenizer

# Requires PyTDC and torch_geometric
try:
    from torch_geometric.loader import DataLoader

    from ugtsdti.data.tdc_dataset import TDCCachingDataset

    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def test_smiles_to_graph():
    """Test RDKit SMILES -> PyG Data extraction."""
    # Aspirin
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    data = smiles_to_graph(smiles)

    assert data is not None
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")

    # Aspirin has 13 heavy atoms
    assert data.x.size(0) == 13
    assert data.x.size(1) == 7  # 7 features per atom


def test_invalid_smiles():
    """Test that invalid smiles are caught gracefully."""
    invalid_smiles = "invalid_chemistry"
    data = smiles_to_graph(invalid_smiles)
    assert data is None


@patch("ugtsdti.utils.sequence.AutoTokenizer")
def test_sequence_tokenizer(MockAutoTokenizer):
    """Test ESM Tokenization bounding and mapping (Mocked to avoid DNS errors)."""
    # Mock the return value of tokenizer(sequence)
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 128)), "attention_mask": torch.ones((1, 128))}

    tokenizer = ESMSequenceTokenizer(model_name="facebook/esm2_t6_8M_UR50D", max_length=128)

    # 10 Amino Acids
    fasta = "MVLSPADKTN"
    tokens = tokenizer.encode(fasta)

    assert "input_ids" in tokens
    assert "attention_mask" in tokens

    assert tokens["input_ids"].dim() == 1
    assert tokens["input_ids"].size(0) == 128  # Padded
    assert tokens["attention_mask"].size(0) == 128


@pytest.mark.skipif(not HAS_PYG, reason="PyG/PyTDC not installed in this environment.")
@patch("ugtsdti.utils.sequence.AutoTokenizer")
def test_tdc_caching_and_batching(MockAutoTokenizer):
    """Test full integration with PyTDC and PyG batching."""
    cache_dir = "./tests/data_cache"

    # Ensure clean slate
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    # Mock ESM Tokenizer inside to avoid DNS errors
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {"input_ids": torch.randint(0, 33, (1, 1024)), "attention_mask": torch.ones((1, 1024))}

    # We mock PyTDC's DTI.get_split to return a tiny Dataframe (4 rows) so the test is instant
    mock_df = pd.DataFrame(
        {
            "Drug": ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO"] * 2,
            "Target": ["MVLSPADKTN", "MFLS"] * 2,
            "Y": [7.0, 5.0, 6.0, 8.0],
        }
    )

    with patch("ugtsdti.data.tdc_dataset.DTI") as MockDTI:
        # Configure the mock to return a split dictionary
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
    assert "drug_index" in sample, "Loader must output transductive drug hash"
    assert "target_index" in sample, "Loader must output transductive target hash"

    # Test batching via PyG
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    # Ensure PyG collated the graphs correctly into a Batch object
    assert hasattr(batch["drug"], "batch")

    # Cleanup
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
