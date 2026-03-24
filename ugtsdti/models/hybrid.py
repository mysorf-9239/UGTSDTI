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

    def _mc_forward(self, branch: nn.Module, batch: dict, mc_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run N stochastic MC-Dropout passes and return mean logit and epistemic variance.

        Both ``mean_logit`` and ``epistemic_var`` are derived from the same ``mc_logits``
        tensor, ensuring mathematical consistency for PairGate fusion.

        Args:
            branch: Student or Teacher sub-model (must have Dropout layers).
            batch: Input batch dictionary.
            mc_samples: Number of stochastic forward passes.

        Returns:
            Tuple of ``(mean_logit, epistemic_var)`` where ``mean_logit`` has shape
            ``(B, 1)`` and ``epistemic_var`` has shape ``(B,)``.
        """
        training_mode = branch.training
        branch.train()  # activate dropout
        with torch.no_grad():
            mc_logits = torch.stack([branch(batch)["logits"] for _ in range(mc_samples)], dim=-1)  # [B, 1, N]
        if not training_mode:
            branch.eval()
        mean_logit = mc_logits.mean(dim=-1)  # [B, 1]
        epistemic_var = mc_logits.var(dim=-1).squeeze(-1)  # [B]
        return mean_logit, epistemic_var

    def forward(self, batch: dict) -> dict:
        """Route forward pass based on available sub-models.

        Supports three modes determined by config:
        - ``only_student``: student branch only.
        - ``only_teacher``: teacher branch only.
        - ``hybrid``: both branches fused via PairGate.
        """
        if self.is_hybrid:
            mc_samples = getattr(self.fusion, "mc_samples", 0)
            if mc_samples > 0 and not self.training:
                student_logits, student_var = self._mc_forward(self.student, batch, mc_samples)
                teacher_logits, teacher_var = self._mc_forward(self.teacher, batch, mc_samples)
                fused_logits = self.fusion(student_logits, teacher_logits, student_var, teacher_var)
            else:
                student_logits = self.student(batch)["logits"]
                teacher_logits = self.teacher(batch)["logits"]
                fused_logits = self.fusion(student_logits, teacher_logits)

            return {
                "logits": fused_logits,
                "student_logits": student_logits,
                "teacher_logits": teacher_logits,
            }

        elif self.student is not None:
            return self.student(batch)

        elif self.teacher is not None:
            return self.teacher(batch)

        else:
            raise ValueError("HybridDTIModel has no sub-models. Check student_cfg/teacher_cfg in config.")
