"""Unit tests for chemistry and sequence transforms.

Covers:
- smiles_to_graph: atom count, feature dims, bond features, isolated atoms,
  single-atom molecule, undirected edges, known molecules
- extract_atom_features / extract_bond_features: feature length, unknown vocab
- _safe_index: in-vocab and out-of-vocab
- ESMSequenceTokenizer: output shapes, padding, truncation (mocked)
"""

from unittest.mock import patch

import torch

from ugtsdti.data.transforms.chemistry import _safe_index, extract_atom_features, extract_bond_features, smiles_to_graph
from ugtsdti.data.transforms.sequence import ESMSequenceTokenizer

# ---------------------------------------------------------------------------
# _safe_index
# ---------------------------------------------------------------------------


def test_safe_index_in_vocab():
    vocab = [1, 6, 7, 8]
    assert _safe_index(vocab, 6) == 1


def test_safe_index_out_of_vocab_returns_last():
    vocab = [1, 6, 7, "OTHER"]
    assert _safe_index(vocab, 999) == len(vocab) - 1


def test_safe_index_first_element():
    vocab = ["a", "b", "c"]
    assert _safe_index(vocab, "a") == 0


# ---------------------------------------------------------------------------
# smiles_to_graph — known molecules
# ---------------------------------------------------------------------------


def test_aspirin_atom_count():
    """Aspirin (CC(=O)OC1=CC=CC=C1C(=O)O) has 13 heavy atoms."""
    data = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert data is not None
    assert data.x.size(0) == 13


def test_aspirin_feature_dim():
    data = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert data.x.size(1) == 7  # OGB-standard 7 atom features


def test_aspirin_bond_feature_dim():
    data = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    assert data.edge_attr.size(1) == 3  # 3 bond features


def test_ethanol_atom_count():
    """Ethanol (CCO) has 3 heavy atoms."""
    data = smiles_to_graph("CCO")
    assert data is not None
    assert data.x.size(0) == 3


def test_undirected_edges():
    """Each bond must appear twice (both directions)."""
    data = smiles_to_graph("CCO")  # 2 bonds → 4 directed edges
    assert data.edge_index.size(1) == 4


def test_single_atom_molecule():
    """Single atom (no bonds) must return valid graph with empty edge_index."""
    data = smiles_to_graph("[Na+]")
    assert data is not None
    assert data.x.size(0) == 1
    assert data.edge_index.size(1) == 0
    assert data.edge_attr.size(0) == 0


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not_a_molecule") is None
    assert smiles_to_graph("") is None
    assert smiles_to_graph("ZZZZZ") is None


def test_graph_x_dtype():
    """Node features must be long (integer indices)."""
    data = smiles_to_graph("CCO")
    assert data.x.dtype == torch.long


def test_graph_edge_index_dtype():
    data = smiles_to_graph("CCO")
    assert data.edge_index.dtype == torch.long


def test_graph_edge_attr_dtype():
    data = smiles_to_graph("CCO")
    assert data.edge_attr.dtype == torch.long


def test_edge_index_valid_range():
    """All edge indices must be within [0, num_atoms)."""
    data = smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")
    num_atoms = data.x.size(0)
    assert data.edge_index.max().item() < num_atoms
    assert data.edge_index.min().item() >= 0


# ---------------------------------------------------------------------------
# extract_atom_features
# ---------------------------------------------------------------------------


def test_extract_atom_features_length():
    """Must return exactly 7 features per atom."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    for atom in mol.GetAtoms():
        feats = extract_atom_features(atom)
        assert len(feats) == 7


def test_extract_atom_features_all_int():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    for atom in mol.GetAtoms():
        feats = extract_atom_features(atom)
        assert all(isinstance(f, int) for f in feats)


# ---------------------------------------------------------------------------
# extract_bond_features
# ---------------------------------------------------------------------------


def test_extract_bond_features_length():
    """Must return exactly 3 features per bond."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    for bond in mol.GetBonds():
        feats = extract_bond_features(bond)
        assert len(feats) == 3


def test_extract_bond_features_all_int():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("CCO")
    for bond in mol.GetBonds():
        feats = extract_bond_features(bond)
        assert all(isinstance(f, int) for f in feats)


# ---------------------------------------------------------------------------
# ESMSequenceTokenizer (mocked)
# ---------------------------------------------------------------------------


@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tokenizer_output_keys(MockAutoTokenizer):
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {
        "input_ids": torch.randint(0, 33, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
    }
    tok = ESMSequenceTokenizer(max_length=64)
    out = tok.encode("MVLSPADKTN")
    assert "input_ids" in out
    assert "attention_mask" in out


@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tokenizer_output_1d(MockAutoTokenizer):
    """encode() must return 1D tensors (squeezed from HF 2D output)."""
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {
        "input_ids": torch.randint(0, 33, (1, 64)),
        "attention_mask": torch.ones(1, 64, dtype=torch.long),
    }
    tok = ESMSequenceTokenizer(max_length=64)
    out = tok.encode("MVLSPADKTN")
    assert out["input_ids"].dim() == 1
    assert out["attention_mask"].dim() == 1


@patch("ugtsdti.data.transforms.sequence.AutoTokenizer")
def test_tokenizer_respects_max_length(MockAutoTokenizer):
    max_len = 128
    mock_instance = MockAutoTokenizer.from_pretrained.return_value
    mock_instance.return_value = {
        "input_ids": torch.randint(0, 33, (1, max_len)),
        "attention_mask": torch.ones(1, max_len, dtype=torch.long),
    }
    tok = ESMSequenceTokenizer(max_length=max_len)
    out = tok.encode("MVLSPADKTN")
    assert out["input_ids"].size(0) == max_len
    assert out["attention_mask"].size(0) == max_len
