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

    def _estimate_epistemic_uncertainty(self, branch: nn.Module, batch: dict, mc_samples: int = 5) -> torch.Tensor:
        """Estimate epistemic uncertainty via Monte Carlo Dropout.

        Runs ``mc_samples`` stochastic forward passes with dropout active and
        returns the per-sample predictive variance across passes.

        Args:
            branch: Student or Teacher sub-model (must have Dropout layers).
            batch: Input batch dictionary.
            mc_samples: Number of stochastic forward passes.

        Returns:
            Predictive variance tensor of shape ``(B,)``.
        """
        training_mode = branch.training
        branch.train()  # activate dropout
        with torch.no_grad():
            mc_logits = [branch(batch)["logits"] for _ in range(mc_samples)]
        epistemic_var = torch.stack(mc_logits, dim=-1).var(dim=-1)
        if not training_mode:
            branch.eval()
        return epistemic_var

    def forward(self, batch: dict) -> dict:
        """Route forward pass based on available sub-models.

        Supports three modes determined by config:
        - ``only_student``: student branch only.
        - ``only_teacher``: teacher branch only.
        - ``hybrid``: both branches fused via PairGate.
        """
        if self.is_hybrid:
            student_logits = self.student(batch)["logits"]
            teacher_logits = self.teacher(batch)["logits"]

            if hasattr(self.fusion, "mc_samples") and self.fusion.mc_samples > 0:
                student_uncertainty = self._estimate_epistemic_uncertainty(self.student, batch, self.fusion.mc_samples)
                teacher_uncertainty = self._estimate_epistemic_uncertainty(self.teacher, batch, self.fusion.mc_samples)
                fused_logits = self.fusion(
                    student_logits,
                    teacher_logits,
                    student_var=student_uncertainty,
                    teacher_var=teacher_uncertainty,
                )
            else:
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
