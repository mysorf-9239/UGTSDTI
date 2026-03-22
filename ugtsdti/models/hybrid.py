import torch
import torch.nn as nn

from ugtsdti.core.registry import MODELS


@MODELS.register("hybrid_dti")
class HybridDTIModel(nn.Module):
    """
    The top-level orchestration model for UGTSDTI.
    Automatically builds its subcomponents based on Hydra nested configs.
    """

    def __init__(self, student_cfg=None, teacher_cfg=None, fusion_cfg=None):
        super().__init__()

        self.student = MODELS.build(student_cfg) if student_cfg else None
        self.teacher = MODELS.build(teacher_cfg) if teacher_cfg else None
        self.fusion = MODELS.build(fusion_cfg) if fusion_cfg else None

        # Determine training mode based on instantiated components
        self.is_hybrid = (self.student is not None) and (self.teacher is not None) and (self.fusion is not None)

    def _estimate_epistemic_uncertainty(self, sub_model, x, mc_samples=5):
        """
        Calculates predictive variance using Monte Carlo Dropout.
        Executes N stochastic forward passes without tracking gradients to serve as Uncertainty.
        """
        was_training = sub_model.training
        sub_model.train()  # Force dropout layers active
        with torch.no_grad():
            preds = []
            for _ in range(mc_samples):
                preds.append(sub_model(x)["logits"])
            # Stack: [Batch, 1, M]
            var = torch.stack(preds, dim=-1).var(dim=-1)
        if not was_training:
            sub_model.eval()
        return var

    def forward(self, x):
        """
        Routing logic based on what components exist.
        Supports Only-Student, Only-Teacher, and Hybrid Fusion modes natively.
        """
        if self.is_hybrid:
            # 1. Base Forward
            s_out = self.student(x)["logits"]
            t_out = self.teacher(x)["logits"]

            # 2. Epistemic Uncertainty (Only if fusion requires it, e.g., pairgate_fusion)
            # We use duck-typing: if fusion has mc_samples, it needs variance
            if hasattr(self.fusion, "mc_samples") and self.fusion.mc_samples > 0:
                s_var = self._estimate_epistemic_uncertainty(self.student, x, self.fusion.mc_samples)
                t_var = self._estimate_epistemic_uncertainty(self.teacher, x, self.fusion.mc_samples)
                fusion_out = self.fusion(s_out, t_out, student_var=s_var, teacher_var=t_var)
            else:
                fusion_out = self.fusion(s_out, t_out)

            return {
                "logits": fusion_out,
                "student_logits": s_out,
                "teacher_logits": t_out,
            }

        elif self.student is not None:
            return self.student(x)

        elif self.teacher is not None:
            return self.teacher(x)

        else:
            raise ValueError("No valid sub-models instantiated in HybridDTIModel.")
