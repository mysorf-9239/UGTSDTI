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
    """Dataset for DTI prediction backed by Therapeutics Data Commons (PyTDC).

    - Integrates with PyTDC for standardized benchmarks (DAVIS, KIBA, BindingDB).
    - S1-S4 cold-start splits via TDC's native ``get_split()`` mechanism.
    - Implements disk caching (.pt) to avoid re-computing RDKit/ESM features every run.

    Negative Sampling Policy
    ------------------------
    DAVIS (and most PyTDC DTI datasets) contains **only positive pairs** — there is
    no negative sampling applied here, and none is needed for standard benchmarking.

    Split-First Invariant
    ---------------------
    If negative sampling is ever added in the future, it **MUST** occur *per-split*
    (i.e., after ``dataset.get_split()`` returns the train/valid/test subsets).
    Global negative sampling before the split would cause data leakage between
    train, validation, and test sets, invalidating all metrics.

    Correct order::

        split_dict = dataset.get_split(...)   # split first
        for split_name, split_df in split_dict.items():
            split_df = add_negatives(split_df)  # then sample negatives per split
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        split_type: str = "cold_split",
        column_name: str | None = "Drug",
        frac: Sequence[float] | None = None,
        cache_dir: str = "./data/cache",
        seed: int = 42,
        binarize_labels: bool = True,
        affinity_threshold: float = 7.0,
        scenario_name: str | None = None,
    ):
        super().__init__()

        if DTI is None:
            raise ImportError("PyTDC is not installed. Run `pip install PyTDC`.")

        self.name = name
        self.split = split
        self.binarize_labels = binarize_labels
        self.affinity_threshold = affinity_threshold
        self.scenario_name = scenario_name or self._infer_scenario_name(split_type=split_type, column_name=column_name)

        self.cache_root = os.path.join(cache_dir, f"{name}_{self.scenario_name}_{seed}")
        self.cache_dir = os.path.join(self.cache_root, split)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_file = os.path.join(self.cache_dir, "dataset.pt")
        self.vocab_file = os.path.join(self.cache_root, "entity_vocab.pt")
        self.graph_cache_path = os.path.join(self.cache_root, "similarity_graphs.pt")
        split_frac = list(frac) if frac is not None else [0.8, 0.1, 0.1]

        # Public attributes: set after loading/building
        self.num_unique_drugs: int | None = None
        self.num_unique_targets: int | None = None
        self.unique_smiles: list[str] | None = None
        self.unique_fasta: list[str] | None = None
        self.unk_drug_index: int | None = None
        self.unk_target_index: int | None = None

        if os.path.exists(self.cache_file):
            logger.info(f"Loading cached {split} dataset from {self.cache_file}")
            cached = torch.load(self.cache_file, weights_only=False)
            self.data = cached["samples"]
            self.num_unique_drugs = cached.get("num_unique_drugs")
            self.num_unique_targets = cached.get("num_unique_targets")
            self.unique_smiles = cached.get("unique_smiles")
            self.unique_fasta = cached.get("unique_fasta")
            self.unk_drug_index = cached.get("unk_drug_index")
            self.unk_target_index = cached.get("unk_target_index")
        else:
            logger.info(f"Cache not found for {split}. Fetching {name} via PyTDC...")

            # Fetch data using PyTDC
            dataset = DTI(name=name)

            # Use PyTDC split mechanisms (handles Cold Start / S1-S4 equivalent)
            split_kwargs = {"method": split_type, "frac": split_frac, "seed": seed}
            if column_name is not None:
                split_kwargs["column_name"] = column_name
            split_dict = dataset.get_split(**split_kwargs)
            raw_data = split_dict[split]
            vocab = self._load_or_build_entity_vocab(split_dict)

            logger.info(f"Processing and Caching {len(raw_data)} pairs...")
            samples = self._build_sample_list(
                raw_data,
                drug_to_idx=vocab["drug_to_idx"],
                target_to_idx=vocab["target_to_idx"],
                unk_drug_index=vocab["unk_drug_index"],
                unk_target_index=vocab["unk_target_index"],
            )
            self.data = samples
            self.unique_smiles = vocab["unique_smiles"]
            self.unique_fasta = vocab["unique_fasta"]
            self.unk_drug_index = vocab["unk_drug_index"]
            self.unk_target_index = vocab["unk_target_index"]
            self.num_unique_drugs = len(self.unique_smiles) + 1
            self.num_unique_targets = len(self.unique_fasta) + 1

            torch.save(
                {
                    "samples": samples,
                    "unique_smiles": self.unique_smiles,
                    "unique_fasta": self.unique_fasta,
                    "num_unique_drugs": self.num_unique_drugs,
                    "num_unique_targets": self.num_unique_targets,
                    "unk_drug_index": self.unk_drug_index,
                    "unk_target_index": self.unk_target_index,
                },
                self.cache_file,
            )
            logger.info(
                f"Saved cache to {self.cache_file} "
                f"({self.num_unique_drugs} unique drugs, {self.num_unique_targets} unique targets)"
            )

    @staticmethod
    def _infer_scenario_name(split_type: str, column_name: str | None) -> str:
        """Derive a stable scenario/cache tag from split config."""
        if split_type == "random_split":
            return "s1"
        if split_type == "cold_split" and column_name == "Drug":
            return "s2"
        if split_type == "cold_split" and column_name == "Target":
            return "s3"
        if split_type == "cold_split" and column_name is None:
            return "s4"

        column_tag = "all" if column_name is None else str(column_name).lower()
        return f"{split_type}_{column_tag}"

    def _load_or_build_entity_vocab(self, split_dict) -> dict:
        """Use train split entities as the shared teacher namespace for all splits.

        Validation/test entities unseen in train are mapped to an explicit UNK slot.
        This preserves a stable transductive namespace without leaking val/test nodes
        into the teacher graph.
        """
        if os.path.exists(self.vocab_file):
            vocab = torch.load(self.vocab_file, weights_only=False)
            drug_to_idx = {s: i for i, s in enumerate(vocab["unique_smiles"])}
            target_to_idx = {f: i for i, f in enumerate(vocab["unique_fasta"])}
            vocab["drug_to_idx"] = drug_to_idx
            vocab["target_to_idx"] = target_to_idx
            return vocab

        train_df = split_dict["train"]
        unique_smiles = list(dict.fromkeys(train_df["Drug"].tolist()))
        unique_fasta = list(dict.fromkeys(train_df["Target"].tolist()))
        vocab = {
            "unique_smiles": unique_smiles,
            "unique_fasta": unique_fasta,
            "unk_drug_index": len(unique_smiles),
            "unk_target_index": len(unique_fasta),
        }
        torch.save(vocab, self.vocab_file)
        vocab["drug_to_idx"] = {s: i for i, s in enumerate(unique_smiles)}
        vocab["target_to_idx"] = {f: i for i, f in enumerate(unique_fasta)}
        return vocab

    def _prepare_label(self, raw_label: float) -> float:
        """Normalize labels to binary when the upstream dataset exposes affinity values."""
        if raw_label in (0.0, 1.0):
            return raw_label
        if not self.binarize_labels:
            return raw_label
        return float(raw_label > self.affinity_threshold)

    def _build_sample_list(
        self, df, drug_to_idx: dict, target_to_idx: dict, unk_drug_index: int, unk_target_index: int
    ):
        """Preprocess raw DataFrame rows into model-ready sample dicts.

        Each sample contains:
        - ``drug``: PyG molecular graph (RDKit atom/bond features).
        - ``target_ids`` / ``target_mask``: ESM token tensors.
        - ``label``: Affinity value as FloatTensor.
        - ``drug_index`` / ``target_index``: Sequential node indices (0..N-1) for
          Teacher transductive lookup. Index i corresponds to the i-th unique
          drug/protein in this split, matching the node ordering in DD/PP graphs.

        """
        from tqdm import tqdm

        from ugtsdti.data.transforms.chemistry import smiles_to_graph
        from ugtsdti.data.transforms.sequence import ESMSequenceTokenizer

        samples = []
        tokenizer = ESMSequenceTokenizer()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Preprocessing [{self.split}]"):
            drug_smiles: str = row["Drug"]
            target_fasta: str = row["Target"]
            label = self._prepare_label(float(row["Y"]))

            # 1. SMILES → PyG molecular graph
            drug_graph = smiles_to_graph(drug_smiles)
            if drug_graph is None:
                continue  # skip unparseable SMILES

            # 2. FASTA → ESM token tensors
            target_tokens = tokenizer.encode(target_fasta)

            # 3. Shared train-based node IDs for Teacher transductive lookup.
            #    Unseen val/test entities are mapped to an explicit UNK slot.
            drug_node_id = drug_to_idx.get(drug_smiles, unk_drug_index)
            target_node_id = target_to_idx.get(target_fasta, unk_target_index)

            samples.append(
                {
                    "drug": drug_graph,
                    "target_ids": target_tokens["input_ids"],
                    "target_mask": target_tokens["attention_mask"],
                    "label": torch.tensor([label], dtype=torch.float32),
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
