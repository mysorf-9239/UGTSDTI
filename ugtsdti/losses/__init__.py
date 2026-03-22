import torch.nn as nn

from ugtsdti.core.registry import LOSSES

from .distillation import KDDualLoss


# Register PyTorch's native BCEWithLogitsLoss to serve as the default plugin
@LOSSES.register("bce_with_logits")
class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, model_outputs: dict, y_true):
        # BCE only cares about the final logits predictor
        return self.bce(model_outputs["logits"], y_true)


__all__ = ["KDDualLoss", "BCEWithLogitsLossWrapper"]
