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
            self.data = self._preprocess_and_cache(raw_data)
            torch.save(self.data, self.cache_file)
            logger.info(f"Saved cache to {self.cache_file}")

    def _preprocess_and_cache(self, df):
        """
        Converts raw SMILES and FASTA to PyG graphs and ESM tokens.
        """
        import hashlib

        from tqdm import tqdm

        from ugtsdti.utils.chemistry import smiles_to_graph
        from ugtsdti.utils.sequence import ESMSequenceTokenizer

        processed = []
        tokenizer = ESMSequenceTokenizer()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {self.split}"):
            drug_smiles = row["Drug"]
            target_fasta = row["Target"]
            y = float(row["Y"])

            # 1. SMILES to PyG Molecular Graph
            drug_graph = smiles_to_graph(drug_smiles)
            if drug_graph is None:
                continue  # Skip invalid SMILES that RDKit cannot parse

            # 2. FASTA to ESM Tokens
            target_tokens = tokenizer.encode(target_fasta)

            # 3. Deterministic Node IDs for Teacher Transductive Lookup
            # We use MD5 modulo a large prime (100003) to simulate a global sparse dictionary seamlessly
            d_hash = int(hashlib.md5(drug_smiles.encode()).hexdigest(), 16) % 100003
            t_hash = int(hashlib.md5(target_fasta.encode()).hexdigest(), 16) % 100003

            item = {
                "drug": drug_graph,
                "target_ids": target_tokens["input_ids"],
                "target_mask": target_tokens["attention_mask"],
                "label": torch.tensor([y], dtype=torch.float32),
                "drug_index": torch.tensor([d_hash], dtype=torch.long),
                "target_index": torch.tensor([t_hash], dtype=torch.long),
            }
            processed.append(item)

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return dict natively. PyG `DataLoader` automatically collates dict values recursively.
        return self.data[idx]
