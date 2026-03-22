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
        Converts raw SMILES and FASTA to features.
        In a real scenario, this applies RDKit for drugs and loads ESM caches for proteins.
        For boilerplate completion, we mock the feature dimensions.
        """
        processed = []
        for _, row in df.iterrows():
            y = float(row["Y"])

            # Mocking the heavy computation
            # D = rdkit_to_graph(drug_smiles)
            # T = self.esm_model.get_embedding(target_fasta)

            item = {
                "drug": torch.randn(128),  # Mock drug embedding
                "target": torch.randn(320),  # Mock ESM-2 8M embedding
                "label": torch.tensor(y, dtype=torch.float32),
            }
            processed.append(item)

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]["label"]
