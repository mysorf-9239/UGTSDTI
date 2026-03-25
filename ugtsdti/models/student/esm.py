import torch.nn as nn
from loguru import logger

from ugtsdti.core.registry import MODELS

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel, AutoTokenizer = None, None


@MODELS.register("esm_student")
class ESMProteinStudent(nn.Module):
    """ESM-2 protein language model student encoder.

    Encodes protein sequences via a pretrained ESM-2 model (HuggingFace).
    Uses the ``[CLS]`` token representation as the sequence-level embedding.

    WARNING: Not tested end-to-end with the full training pipeline.
    Validate before using in experiments.

    Args:
        model_name: HuggingFace model identifier for ESM-2.
        hidden_dim: Output projection dimension for DTI prediction head.
        freeze: If ``True``, freezes ESM-2 weights (feature extraction mode).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        hidden_dim: int = 128,
        freeze: bool = True,
    ):
        super().__init__()

        if AutoModel is None:
            raise ImportError("Transformers library not installed. Run `pip install transformers`.")

        logger.info(f"Loading pretrained ESM-2: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)

        if freeze:
            logger.info(f"Freezing ESM-2 weights: {model_name}")
            for param in self.encoder.parameters():
                param.requires_grad = False

        esm_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(esm_hidden_size, hidden_dim)
        self.predictor = nn.Linear(hidden_dim, 1)

    def forward(self, batch: dict) -> dict:
        """Forward pass consuming the standard multimodal batch dict.

        Args:
            batch: Must contain ``target_ids`` (LongTensor [B, L]) and
                ``target_mask`` (LongTensor [B, L]).

        Returns:
            Dict with key ``logits`` of shape ``(B, 1)``.
        """
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        outputs = self.encoder(input_ids=target_ids, attention_mask=target_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [B, esm_hidden_size]

        protein_emb = self.projection(cls_emb)  # [B, hidden_dim]
        logits = self.predictor(protein_emb).squeeze(-1)  # [B]

        return {"logits": logits}
