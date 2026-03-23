import os
from typing import Sequence

import torch
from loguru import logger
from torch.utils.data import Dataset

from ugtsdti.core.registry import DATASETS

try:
    from tdc.multi_pred import DTI
except ImportError:
    DTI = None


@DATASETS.register("tdc_caching_dataset")
class TDCCachingDataset(Dataset):
    """
    SOTA Dataset for DTI prediction.
    - Integrates with Therapeutics Data Commons (PyTDC) for standardized benchmarks.
    - S1-S4 splits via TDC's native split functions.
    - Implements disk caching (.pt) to avoid re-computing RDKit/ESM features every run.
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        split_type: str = "cold_split",
        column_name: str = "Drug",
        frac: Sequence[float] | None = None,
        cache_dir: str = "./data/cache",
        seed: int = 42,
    ):
        super().__init__()

        if DTI is None:
            raise ImportError("PyTDC is not installed. Run `pip install PyTDC`.")

        self.name = name
        self.split = split
        self.cache_dir = os.path.join(cache_dir, f"{name}_{split_type}_{split}_{seed}")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(self.cache_dir, "dataset.pt")
        split_frac = list(frac) if frac is not None else [0.8, 0.1, 0.1]

        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached {split} dataset from {self.cache_file}")
            self.data = torch.load(self.cache_file)
        else:
            logger.info(f"Cache not found for {split}. Fetching {name} via PyTDC...")

            # Fetch data using PyTDC
            dataset = DTI(name=name)

            # Use PyTDC split mechanisms (handles Cold Start / S1-S4 equivalent)
            split_dict = dataset.get_split(method=split_type, column_name=column_name, frac=split_frac, seed=seed)
            raw_data = split_dict[split]

            logger.info(f"Processing and Caching {len(raw_data)} pairs...")
            self.data = self._build_sample_list(raw_data)
            torch.save(self.data, self.cache_file)
            logger.info(f"Saved cache to {self.cache_file}")

    def _build_sample_list(self, df) -> list:
        """Preprocess raw DataFrame rows into model-ready sample dicts.

        Each sample contains:
        - ``drug``: PyG molecular graph (RDKit atom/bond features).
        - ``target_ids`` / ``target_mask``: ESM token tensors.
        - ``label``: Affinity value as FloatTensor.
        - ``drug_node_id`` / ``target_node_id``: Deterministic integer IDs for
          Teacher transductive lookup (MD5 hash modulo large prime).
        """
        import hashlib

        from tqdm import tqdm

        from ugtsdti.data.transforms.chemistry import smiles_to_graph
        from ugtsdti.data.transforms.sequence import ESMSequenceTokenizer

        _HASH_MODULUS = 100_003  # large prime; collision rate ~0.01% on DAVIS

        samples = []
        tokenizer = ESMSequenceTokenizer()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing [{self.split}]"):
            drug_smiles: str = row["Drug"]
            target_fasta: str = row["Target"]
            affinity: float = float(row["Y"])

            # 1. SMILES → PyG molecular graph
            drug_graph = smiles_to_graph(drug_smiles)
            if drug_graph is None:
                continue  # skip unparseable SMILES

            # 2. FASTA → ESM token tensors
            target_tokens = tokenizer.encode(target_fasta)

            # 3. Deterministic node IDs for Teacher transductive lookup
            drug_node_id = int(hashlib.md5(drug_smiles.encode()).hexdigest(), 16) % _HASH_MODULUS
            target_node_id = int(hashlib.md5(target_fasta.encode()).hexdigest(), 16) % _HASH_MODULUS

            samples.append(
                {
                    "drug": drug_graph,
                    "target_ids": target_tokens["input_ids"],
                    "target_mask": target_tokens["attention_mask"],
                    "label": torch.tensor([affinity], dtype=torch.float32),
                    "drug_index": torch.tensor([drug_node_id], dtype=torch.long),
                    "target_index": torch.tensor([target_node_id], dtype=torch.long),
                }
            )

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return dict natively. PyG `DataLoader` automatically collates dict values recursively.
        return self.data[idx]
