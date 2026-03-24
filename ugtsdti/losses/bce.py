import torch.nn as nn

from ugtsdti.core.registry import LOSSES


@LOSSES.register("bce")
class BCELoss(nn.Module):
    """BCE with logits loss wrapper.

    Wraps ``nn.BCEWithLogitsLoss`` to accept the standard model output dict
    and align shapes before computing the loss.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, model_outputs: dict, y_true):
        logits = model_outputs["logits"].view(-1)  # ensure (B,)
        return self.bce(logits, y_true.view(-1))
