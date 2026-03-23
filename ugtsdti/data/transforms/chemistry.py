from typing import Any, List, Optional

import torch
from loguru import logger

try:
    from rdkit import Chem
except ImportError:
    Chem = None

try:
    from torch_geometric.data import Data
except ImportError:
    Data = None

# -----------------------------------------------------------------------
# OGB-Standard Node & Edge Features Extraction
# -----------------------------------------------------------------------

# Allowed vocabularies for atomic features
ATOM_VOCAB: dict[str, list] = {
    "atomic_num": list(range(1, 119)),
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "num_radical_electrons": [0, 1, 2, 3, 4],
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

BOND_VOCAB: dict[str, list] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "is_conjugated": [False, True],
    "is_in_ring": [False, True],
}


def _safe_index(vocab: list, item: Any) -> int:
    """Return index of item in vocab, or the last index (usually 'OTHER') if not found."""
    try:
        return vocab.index(item)
    except ValueError:
        return len(vocab) - 1


def extract_atom_features(atom) -> List[int]:
    """Extract standard 7 features for a single RDKit atom."""
    return [
        _safe_index(ATOM_VOCAB["atomic_num"], atom.GetAtomicNum()),
        _safe_index(ATOM_VOCAB["degree"], atom.GetTotalDegree()),
        _safe_index(ATOM_VOCAB["formal_charge"], atom.GetFormalCharge()),
        _safe_index(ATOM_VOCAB["num_radical_electrons"], atom.GetNumRadicalElectrons()),
        _safe_index(ATOM_VOCAB["hybridization"], str(atom.GetHybridization())),
        _safe_index(ATOM_VOCAB["is_aromatic"], atom.GetIsAromatic()),
        _safe_index(ATOM_VOCAB["is_in_ring"], atom.IsInRing()),
    ]


def extract_bond_features(bond) -> List[int]:
    """Extract standard 3 features for a single RDKit bond."""
    return [
        _safe_index(BOND_VOCAB["bond_type"], str(bond.GetBondType())),
        _safe_index(BOND_VOCAB["is_conjugated"], bond.GetIsConjugated()),
        _safe_index(BOND_VOCAB["is_in_ring"], bond.IsInRing()),
    ]


def smiles_to_graph(smiles: str) -> Optional["Data"]:
    """
    Convert a SMILES string to a PyTorch Geometric Data object.
    Automatically handles invalid SMILES by returning None.
    """
    if Chem is None or Data is None:
        raise ImportError("RDKit and PyTorch Geometric are required to parse SMILES graphs.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"RDKit could not parse SMILES: {smiles}")
        return None

    # Node features
    atom_features_list = [extract_atom_features(atom) for atom in mol.GetAtoms()]  # type: ignore

    if len(atom_features_list) == 0:
        return None

    x = torch.tensor(atom_features_list, dtype=torch.long)

    # Edge features
    edge_index_list: list = []
    edge_attr_list: list = []

    for bond in mol.GetBonds():  # type: ignore
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_feat = extract_bond_features(bond)

        # Add both directions (undirected graph convention in PyG)
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [bond_feat, bond_feat]

    if len(edge_index_list) > 0:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
