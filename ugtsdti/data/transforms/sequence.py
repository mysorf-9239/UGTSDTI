from typing import Dict

import torch
from loguru import logger

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class ESMSequenceTokenizer:
    """
    A unified wrapper around HuggingFace ESM Tokenizers for Protein Sequences.
    Converts FASTA strings into padded/truncated `input_ids` and `attention_mask`.
    """

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", max_length: int = 1024):
        if AutoTokenizer is None:
            raise ImportError("Please install `transformers` to use the ESMSequenceTokenizer.")

        self.model_name = model_name
        self.max_length = max_length
        logger.info(f"Loading ESM Tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Encode a single protein sequence.
        Args:
            sequence: Raw amino acid string (e.g., 'MVLSPADKTN...')
        Returns:
            A dictionary containing 'input_ids' and 'attention_mask' as 1D tensors.
        """
        # The ESM tokenizer expects spaces between amino acids occasionally depending on the exact build,
        # but the standard HF ESM models natively handle standard continuous FASTA strings.

        # Remove any invalid characters or whitespace
        sequence = sequence.strip().upper()

        encoded = self.tokenizer(
            sequence,
            add_special_tokens=True,  # <cls> and <eos>
            max_length=self.max_length,  # Truncate to max length
            padding="max_length",  # Pad rigidly to max length
            truncation=True,
            return_tensors="pt",
        )

        # HuggingFace returns 2D batch tensors (1, seq_len).
        # We squeeze them to 1D since Dataloader will collate them into batches later.
        return {"input_ids": encoded["input_ids"].squeeze(0), "attention_mask": encoded["attention_mask"].squeeze(0)}
