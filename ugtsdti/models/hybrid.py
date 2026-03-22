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

    def forward(self, x):
        """
        Routing logic based on what components exist.
        In a real research project, this would handle complex Dict inputs
        (e.g., x['drug_seq'], x['prot_graph'], etc.).
        """
        # --- Simplified Forward ---
        if self.is_hybrid:
            # 1. Student Forward
            s_out = self.student(x)
            # 2. Teacher Forward
            t_out = self.teacher(x)
            # 3. Fusion Forward
            fusion_out = self.fusion(s_out, t_out)
            return fusion_out
        elif self.student is not None:
            return self.student(x)
        elif self.teacher is not None:
            return self.teacher(x)
        else:
            raise ValueError("No valid sub-models instantiated in HybridDTIModel.")
