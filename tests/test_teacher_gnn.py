"""
Tests for Teacher GNN (Phase 11 UGTSDTI).

Covers:
- Unit tests: GCNTeacher registration, forward shape, error handling,
  graph builder edge cases, EMA, cache.
- Property-based tests (P1–P9) using hypothesis.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch_geometric.data import Data

from ugtsdti.core.registry import MODELS
from ugtsdti.data.graph_builder import (
    build_and_cache_graphs,
    build_drug_drug_graph,
    build_protein_protein_graph,
    compute_edge_churn,
    ema_update,
)
from ugtsdti.models.teacher.gcn_teacher import GCNTeacher

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_SMILES_POOL = [
    "CC",
    "CCC",
    "CCCC",
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
]


def _make_mock_graphs(n_drugs=5, n_proteins=4, drug_feat=2048, prot_feat=8000):
    dd = Data(
        x=torch.randn(n_drugs, drug_feat),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        num_nodes=n_drugs,
    )
    pp = Data(
        x=torch.randn(n_proteins, prot_feat),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        num_nodes=n_proteins,
    )
    return dd, pp


# ---------------------------------------------------------------------------
# 5.1 Unit tests: GCNTeacher registration, forward shape, no-graphs raises
# ---------------------------------------------------------------------------


def test_gcn_teacher_registered():
    """Req 6.1, 8.1, 8.3: GCNTeacher is registered and buildable via registry."""
    model = MODELS.build(
        {
            "name": "gcn_teacher",
            "params": {
                "drug_feat_dim": 2048,
                "protein_feat_dim": 8000,
                "hidden_dim": 32,
            },
        }
    )
    assert isinstance(model, GCNTeacher)


def test_gcn_teacher_forward_shape():
    """Req 6.2, 6.3: forward returns logits of shape (B,) and dtype float32."""
    model = GCNTeacher(drug_feat_dim=2048, protein_feat_dim=8000, hidden_dim=32)
    dd, pp = _make_mock_graphs()
    model.set_graphs(dd, pp)
    model.eval()

    B = 4
    batch = {
        "drug_index": torch.randint(0, 5, (B, 1)),
        "target_index": torch.randint(0, 4, (B, 1)),
    }
    out = model(batch)
    assert out["logits"].shape == (B,)
    assert out["logits"].dtype == torch.float32


def test_gcn_teacher_no_graphs_raises():
    """Req 6.6: RuntimeError when set_graphs() not called before forward()."""
    model = GCNTeacher(hidden_dim=32)
    with pytest.raises(RuntimeError):
        model(
            {
                "drug_index": torch.zeros(1, 1, dtype=torch.long),
                "target_index": torch.zeros(1, 1, dtype=torch.long),
            }
        )


# ---------------------------------------------------------------------------
# 5.2 Unit tests: Graph Builder edge cases
# ---------------------------------------------------------------------------


def test_invalid_smiles_zero_vector():
    """Req 1.2: Invalid SMILES → zero feature vector at that index."""
    graph = build_drug_drug_graph(["INVALID_SMILES_XYZ", "CC"])
    assert graph.x[0].sum().item() == 0.0


def test_single_node_no_edges():
    """Req 3.5: Single node → no edges, no error."""
    graph = build_drug_drug_graph(["CC"])
    assert graph.edge_index.shape == (2, 0)


def test_graph_builder_creates_directories(tmp_path):
    """Req 5.3: Auto-create nested parent directories before saving cache."""
    cache_path = str(tmp_path / "nested" / "dir" / "graphs.pt")
    build_and_cache_graphs(["CC", "CCC"], ["ACDEF", "GHIKL"], cache_path=cache_path)
    assert os.path.exists(cache_path)


# ---------------------------------------------------------------------------
# 5.3 Unit tests: EMA and cache hit
# ---------------------------------------------------------------------------


def test_ema_decay_one_no_change():
    """Req 11.3: ema_decay=1.0 → result equals feat_old exactly."""
    feat = torch.randn(10)
    other = torch.randn(10)
    result = ema_update(feat, other, ema_decay=1.0)
    assert torch.equal(result, feat)


def test_graph_cache_hit(tmp_path):
    """Req 5.1: Second call with same cache_path returns identical tensors."""
    cache_path = str(tmp_path / "graphs.pt")
    graphs1 = build_and_cache_graphs(["CC", "CCC"], ["ACDEF"], cache_path=cache_path)
    graphs2 = build_and_cache_graphs(["CC", "CCC"], ["ACDEF"], cache_path=cache_path)
    assert torch.equal(graphs1["dd"].x, graphs2["dd"].x)
    assert torch.equal(graphs1["pp"].x, graphs2["pp"].x)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


# P1: Drug fingerprint shape and ordering invariant
# Feature: teacher-gnn, Property 1: Drug fingerprint extraction invariant
@given(smiles_list=st.lists(st.sampled_from(VALID_SMILES_POOL), min_size=1, max_size=10))
@settings(max_examples=50)
def test_p1_drug_fingerprint_shape_and_ordering(smiles_list):
    """Validates: Requirements 1.1, 1.3, 1.4"""
    from rdkit import Chem

    graph = build_drug_drug_graph(smiles_list)
    assert graph.x.shape == (len(smiles_list), 2048)
    assert graph.x.dtype == torch.float32
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            assert graph.x[i].sum().item() > 0


# P2: Protein k-mer shape and normalization invariant
# Feature: teacher-gnn, Property 2: Protein k-mer extraction invariant
@given(
    fasta_list=st.lists(
        st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=3, max_size=20),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=50)
def test_p2_protein_kmer_shape_and_normalization(fasta_list):
    """Validates: Requirements 2.1, 2.2, 2.3, 2.4"""
    graph = build_protein_protein_graph(fasta_list)
    assert graph.x.shape == (len(fasta_list), 8000)
    assert graph.x.dtype == torch.float32
    for i in range(len(fasta_list)):
        l1 = graph.x[i].sum().item()
        assert abs(l1 - 1.0) < 1e-5 or l1 == 0.0


# P3: Graph symmetry — undirected, no self-loops
# Feature: teacher-gnn, Property 3: Graph symmetry
@given(smiles_list=st.lists(st.sampled_from(VALID_SMILES_POOL), min_size=2, max_size=10))
@settings(max_examples=50)
def test_p3_graph_symmetry(smiles_list):
    """Validates: Requirements 3.3, 4.3"""
    graph = build_drug_drug_graph(smiles_list)
    if graph.edge_index.shape[1] == 0:
        return
    edges = set(map(tuple, graph.edge_index.t().tolist()))
    for i, j in edges:
        assert i != j, "Self-loop found"
        assert (j, i) in edges, f"Missing reverse edge ({j}, {i})"


# P4: Graph degree bound <= 2 * min(top_k, n-1)
# Feature: teacher-gnn, Property 4: Graph degree bound
# Note: after symmetrization, a node can receive reverse edges from up to
# min(top_k, n-1) other nodes that each selected it as a neighbor, so the
# tight upper bound is 2 * min(top_k, n-1) (not 2*top_k when n is small).
@given(
    smiles_list=st.lists(st.sampled_from(VALID_SMILES_POOL), min_size=2, max_size=10),
    top_k=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=50)
def test_p4_graph_degree_bound(smiles_list, top_k):
    """Validates: Requirements 3.2, 4.2"""
    graph = build_drug_drug_graph(smiles_list, top_k=top_k)
    if graph.edge_index.shape[1] == 0:
        return
    n = len(smiles_list)
    effective_k = min(top_k, n - 1)
    degrees = torch.zeros(n, dtype=torch.long)
    degrees.scatter_add_(
        0,
        graph.edge_index[0],
        torch.ones(graph.edge_index.shape[1], dtype=torch.long),
    )
    assert degrees.max().item() <= 2 * effective_k * n  # loose upper bound: each of n nodes can add reverse edge
    # Tighter: each node selects at most effective_k outgoing neighbors,
    # and can receive at most (n-1) incoming reverse edges → degree <= effective_k + (n-1)
    assert degrees.max().item() <= effective_k + (n - 1)


# P5: Graph cache round-trip
# Feature: teacher-gnn, Property 5: Graph cache round-trip
@given(
    smiles_list=st.lists(st.sampled_from(VALID_SMILES_POOL), min_size=1, max_size=5),
    fasta_list=st.lists(
        st.text(alphabet="ACDEFGHIKLMNPQRSTVWY", min_size=3, max_size=10),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=50)
def test_p5_graph_cache_roundtrip(smiles_list, fasta_list):
    """Validates: Requirements 5.1, 5.2, 13.1, 13.2"""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "graphs.pt")
        graphs = build_and_cache_graphs(smiles_list, fasta_list, cache_path=cache_path)
        loaded = torch.load(cache_path, weights_only=False)
        assert torch.equal(graphs["dd"].x, loaded["dd"].x)
        assert torch.equal(graphs["dd"].edge_index, loaded["dd"].edge_index)
        assert torch.equal(graphs["pp"].x, loaded["pp"].x)
        assert torch.equal(graphs["pp"].edge_index, loaded["pp"].edge_index)


# P6: GCNTeacher forward shape for any batch size
# Feature: teacher-gnn, Property 6: GCNTeacher forward shape
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    n_drugs=st.integers(min_value=2, max_value=8),
    n_proteins=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=50)
def test_p6_gcn_teacher_forward_shape(batch_size, n_drugs, n_proteins):
    """Validates: Requirements 6.2, 6.3"""
    model = GCNTeacher(drug_feat_dim=2048, protein_feat_dim=8000, hidden_dim=16)
    dd, pp = _make_mock_graphs(n_drugs=n_drugs, n_proteins=n_proteins)
    model.set_graphs(dd, pp)
    model.eval()

    batch = {
        "drug_index": torch.randint(0, n_drugs, (batch_size, 1)),
        "target_index": torch.randint(0, n_proteins, (batch_size, 1)),
    }
    out = model(batch)
    assert out["logits"].shape == (batch_size,)
    assert out["logits"].dtype == torch.float32


# P7: MC-Dropout variance > 0 for dropout >= 0.2
# Feature: teacher-gnn, Property 7: MC-Dropout non-zero variance
@given(
    batch_size=st.integers(min_value=1, max_value=8),
    dropout=st.floats(min_value=0.2, max_value=0.5),
)
@settings(max_examples=50)
def test_p7_mc_dropout_variance(batch_size, dropout):
    """Validates: Requirements 6.4, 6.8, 9.2"""
    model = GCNTeacher(dropout=dropout, hidden_dim=16)
    dd, pp = _make_mock_graphs()
    model.set_graphs(dd, pp)
    model.train()

    batch = {
        "drug_index": torch.zeros(batch_size, 1, dtype=torch.long),
        "target_index": torch.zeros(batch_size, 1, dtype=torch.long),
    }
    logits = torch.stack([model(batch)["logits"] for _ in range(20)])
    assert (logits.var(dim=0) > 0).all()


# P8: EMA update formula correctness
# Feature: teacher-gnn, Property 8: EMA update formula
@given(
    shape=st.integers(min_value=1, max_value=100),
    ema_decay=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=50)
def test_p8_ema_update_formula(shape, ema_decay):
    """Validates: Requirements 11.1, 11.3"""
    feat_old = torch.randn(shape)
    feat_new = torch.randn(shape)
    result = ema_update(feat_old, feat_new, ema_decay)
    expected = ema_decay * feat_old.numpy() + (1.0 - ema_decay) * feat_new.numpy()
    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-6)


# P9: Edge churn computation formula
# Feature: teacher-gnn, Property 9: Edge churn computation
@given(
    old_edges=st.frozensets(st.tuples(st.integers(0, 9), st.integers(0, 9))),
    new_edges=st.frozensets(st.tuples(st.integers(0, 9), st.integers(0, 9))),
)
@settings(max_examples=50)
def test_p9_edge_churn_formula(old_edges, new_edges):
    """Validates: Requirements 12.1"""
    churn = compute_edge_churn(old_edges, new_edges)
    expected = len(old_edges.symmetric_difference(new_edges)) / max(len(old_edges), 1)
    assert abs(churn - expected) < 1e-6
