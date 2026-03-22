import torch.nn as nn
from loguru import logger

from ugtsdti.core.registry import MODELS

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel, AutoTokenizer = None, None


@MODELS.register("esm_student")
class ESMProteinStudent(nn.Module):
    """
    Protein Language Model Student.
    Uses HuggingFace 'transformers' to pull pre-trained ESM models for SOTA representations.
    """

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", freeze: bool = True):
        super().__init__()

        if AutoModel is None:
            raise ImportError("Transformers library not installed. Run `pip install transformers`.")

        logger.info(f"Loading pretrained PLM: {model_name}")
        # Note: In production, you typically cache PLM embeddings offline if 'freeze' is True.
        # This module demonstrates online extraction.
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            logger.info(f"Freezing weights for {model_name}")
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Determine output dim
        self.output_dim = self.encoder.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """
        input_ids: Tokenized FASTA sequences [Batch, SeqLen]
        """
        # [Batch, SeqLen, HiddenDim]
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Taking the <CLS> token representation (index 0) for the whole protein
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return cls_rep
