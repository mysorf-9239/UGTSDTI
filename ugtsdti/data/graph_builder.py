"""Similarity-graph construction utilities for graph-based teacher models.

This module builds the transductive Drug-Drug and Protein-Protein graphs used by
teacher encoders such as :class:`ugtsdti.models.teacher.gcn_teacher.GCNTeacher`.
The builders are deterministic so scenario-aware caches can be reused safely
across repeated runs.
"""

from __future__ import annotations

import itertools
import os
from typing import Optional

import numpy as np
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from torch_geometric.data import Data


def _morgan_fingerprints(smiles_list: list[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprints for a list of SMILES. Invalid → zero vector."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # leave as zero vector
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps[i] = arr
    return fps


def _kmer_vectors(fasta_list: list[str], k: int = 3) -> np.ndarray:
    """Compute L1-normalized k-mer frequency vectors for a list of sequences."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    kmers = ["".join(p) for p in itertools.product(amino_acids, repeat=k)]
    kmer_to_idx = {km: idx for idx, km in enumerate(kmers)}
    dim = len(kmers)  # 20**k

    vecs = np.zeros((len(fasta_list), dim), dtype=np.float32)
    for i, seq in enumerate(fasta_list):
        for j in range(len(seq) - k + 1):
            kmer = seq[j : j + k]
            if kmer in kmer_to_idx:
                vecs[i, kmer_to_idx[kmer]] += 1.0
        total = vecs[i].sum()
        if total > 0:
            vecs[i] /= total
    return vecs


def _knn_edges(sim_matrix: np.ndarray, top_k: int) -> tuple[list[int], list[int]]:
    """Build undirected kNN edges from a pairwise similarity matrix."""
    n = sim_matrix.shape[0]
    src, dst = [], []
    for i in range(n):
        row = sim_matrix[i].copy()
        row[i] = -1.0  # exclude self
        # Get top-k indices
        k = min(top_k, n - 1)
        if k <= 0:
            continue
        top_indices = np.argpartition(row, -k)[-k:]
        for j in top_indices:
            src.append(i)
            dst.append(int(j))
            src.append(int(j))
            dst.append(i)
    return src, dst


def _deduplicate_edges(src: list[int], dst: list[int]) -> tuple[list[int], list[int]]:
    """Remove duplicate edges (keep unique (i,j) pairs)."""
    seen = set()
    new_src, new_dst = [], []
    for i, j in zip(src, dst, strict=True):
        if (i, j) not in seen:
            seen.add((i, j))
            new_src.append(i)
            new_dst.append(j)
    return new_src, new_dst


def build_drug_drug_graph(
    smiles_list: list[str],
    radius: int = 2,
    nbits: int = 2048,
    top_k: int = 10,
) -> Data:
    """Build a Drug-Drug similarity graph from SMILES strings.

    Args:
        smiles_list: List of SMILES strings. Invalid SMILES → zero feature vector.
        radius: Morgan fingerprint radius (default 2 = ECFP4).
        nbits: Fingerprint bit length (default 2048).
        top_k: Number of nearest neighbors per node.

    Returns:
        Data(x=FloatTensor(n, nbits), edge_index=LongTensor(2, E), num_nodes=n)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    n = len(smiles_list)
    fps_np = _morgan_fingerprints(smiles_list, radius=radius, nbits=nbits)

    # Build pairwise Tanimoto similarity
    sim_matrix = np.zeros((n, n), dtype=np.float32)
    if n > 0:
        # Convert to RDKit fingerprint objects for BulkTanimotoSimilarity
        rdkit_fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                rdkit_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits))
            else:
                rdkit_fps.append(None)

        for i in range(n):
            if rdkit_fps[i] is None:
                continue
            valid_fps = [(j, fp) for j, fp in enumerate(rdkit_fps) if fp is not None]
            if not valid_fps:
                continue
            indices, fps_list = zip(*valid_fps, strict=True)
            sims = DataStructs.BulkTanimotoSimilarity(rdkit_fps[i], list(fps_list))
            for j, s in zip(indices, sims, strict=True):
                sim_matrix[i, j] = s

    src, dst = _knn_edges(sim_matrix, top_k)
    src, dst = _deduplicate_edges(src, dst)

    if len(src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.from_numpy(fps_np)
    graph = Data(x=x, edge_index=edge_index, num_nodes=n)

    if graph.num_nodes != len(smiles_list):
        raise ValueError(f"num_nodes mismatch: graph.num_nodes={graph.num_nodes}, len(smiles_list)={len(smiles_list)}")
    return graph


def build_protein_protein_graph(
    fasta_list: list[str],
    k: int = 3,
    top_k: int = 10,
) -> Data:
    """Build a Protein-Protein similarity graph from amino acid sequences.

    Args:
        fasta_list: List of amino acid sequences.
        k: k-mer length (default 3 → 8000 dims).
        top_k: Number of nearest neighbors per node.

    Returns:
        Data(x=FloatTensor(n, 20**k), edge_index=LongTensor(2, E), num_nodes=n)
    """
    n = len(fasta_list)
    vecs = _kmer_vectors(fasta_list, k=k)

    # Pairwise cosine similarity
    if n > 1:
        sim_matrix = cosine_similarity(vecs).astype(np.float32)
    else:
        sim_matrix = np.ones((n, n), dtype=np.float32)

    src, dst = _knn_edges(sim_matrix, top_k)
    src, dst = _deduplicate_edges(src, dst)

    if len(src) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.from_numpy(vecs)
    graph = Data(x=x, edge_index=edge_index, num_nodes=n)

    if graph.num_nodes != len(fasta_list):
        raise ValueError(f"num_nodes mismatch: graph.num_nodes={graph.num_nodes}, len(fasta_list)={len(fasta_list)}")
    return graph


def build_and_cache_graphs(
    smiles_list: list[str],
    fasta_list: list[str],
    cache_path: str = "data/cache/similarity_graphs.pt",
) -> dict:
    """Build DD and PP graphs, caching to disk.

    Args:
        smiles_list: List of SMILES strings.
        fasta_list: List of amino acid sequences.
        cache_path: Path to cache file.

    Returns:
        {"dd": dd_graph, "pp": pp_graph}
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading graphs from cache: {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data

    logger.info("Building graphs from scratch...")
    dd_graph = build_drug_drug_graph(smiles_list)
    pp_graph = build_protein_protein_graph(fasta_list)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    graphs = {"dd": dd_graph, "pp": pp_graph}
    torch.save(graphs, cache_path)
    logger.info(f"Graphs saved to cache: {cache_path}")
    return graphs


def load_graphs(cache_path: str) -> tuple[Data, Data]:
    """Load cached DD and PP graphs.

    Args:
        cache_path: Path to the .pt cache file.

    Returns:
        (dd_graph, pp_graph)
    """
    data = torch.load(cache_path, weights_only=False)
    return data["dd"], data["pp"]


# ---------------------------------------------------------------------------
# Graph Stability Controls (Task 4)
# ---------------------------------------------------------------------------


def ema_update(feat_old: Tensor, feat_new: Tensor, ema_decay: float) -> Tensor:
    """Exponential Moving Average update for node features.

    Args:
        feat_old: Previous feature tensor.
        feat_new: New feature tensor.
        ema_decay: Decay factor in [0, 1]. 1.0 = no change.

    Returns:
        ema_decay * feat_old + (1 - ema_decay) * feat_new
    """
    return ema_decay * feat_old + (1.0 - ema_decay) * feat_new


def compute_edge_churn(old_edges: set, new_edges: set) -> float:
    """Compute the fraction of edges that changed between two edge sets.

    Args:
        old_edges: Set of (i, j) tuples representing old edges.
        new_edges: Set of (i, j) tuples representing new edges.

    Returns:
        len(symmetric_difference) / max(len(old_edges), 1)
    """
    return len(old_edges.symmetric_difference(new_edges)) / max(len(old_edges), 1)


def maybe_rebuild_graph(
    teacher,
    epoch: int,
    warmup_epochs: int,
    rebuild_every: int,
    ema_decay: float,
    smiles_list: list[str],
    fasta_list: list[str],
    cache_path: Optional[str] = None,
) -> None:
    """Conditionally rebuild the kNN graph with EMA smoothing.

    Args:
        teacher: GCNTeacher instance with dd_graph and pp_graph attributes.
        epoch: Current training epoch.
        warmup_epochs: Number of epochs to freeze graph topology.
        rebuild_every: Rebuild interval (0 = never rebuild).
        ema_decay: EMA decay factor for node features.
        smiles_list: List of SMILES strings.
        fasta_list: List of amino acid sequences.
        cache_path: Optional path to save rebuilt graphs.
    """
    if rebuild_every == 0:
        return
    if epoch < warmup_epochs:
        return
    if epoch % rebuild_every != 0:
        return

    # Get old edges as set
    old_edge_index = teacher.dd_graph.edge_index
    old_edges = set(map(tuple, old_edge_index.t().tolist()))

    # Build new graphs
    new_dd = build_drug_drug_graph(smiles_list)
    new_pp = build_protein_protein_graph(fasta_list)

    # Apply EMA to node features
    new_dd.x = ema_update(teacher.dd_graph.x, new_dd.x, ema_decay)
    new_pp.x = ema_update(teacher.pp_graph.x, new_pp.x, ema_decay)

    # Compute edge churn for DD graph
    new_edges = set(map(tuple, new_dd.edge_index.t().tolist()))
    churn = compute_edge_churn(old_edges, new_edges)

    # Update teacher graphs
    teacher.set_graphs(new_dd, new_pp)

    logger.info(f"Epoch {epoch}: Rebuilt graphs. DD edge churn = {churn:.4f}")

    # Log to WandB
    try:
        import wandb

        wandb.log({"graph/edge_churn": churn})
    except ImportError:
        pass
