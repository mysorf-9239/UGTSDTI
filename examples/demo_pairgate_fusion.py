"""
Demo: PairGate fusion with MC-Dropout uncertainty from both branches.

Shows how alpha (gate weight) shifts when teacher is uncertain vs confident.

Usage:
    conda run -n ugtsdti python examples/demo_pairgate_fusion.py
"""
import torch
from ugtsdti.models.fusion.pairgate import PairGateFusion

if __name__ == "__main__":
    B = 4
    gate = PairGateFusion(gate_hidden=16, mc_samples=5)
    gate.eval()

    student_logits = torch.randn(B, 1)
    teacher_logits = torch.randn(B, 1)

    # Case 1: teacher confident (low var), student uncertain (high var)
    # → gate should weight teacher more (alpha → 1)
    student_var = torch.tensor([0.8, 0.9, 0.7, 0.85])
    teacher_var = torch.tensor([0.05, 0.03, 0.04, 0.02])
    fused = gate(student_logits, teacher_logits, student_var, teacher_var)
    print(f"Teacher confident  → fused: {fused.squeeze().detach()}")

    # Case 2: teacher uncertain (high var), student confident (low var)
    # → gate should weight student more (alpha → 0)
    student_var2 = torch.tensor([0.02, 0.03, 0.01, 0.04])
    teacher_var2 = torch.tensor([0.9, 0.85, 0.8, 0.95])
    fused2 = gate(student_logits, teacher_logits, student_var2, teacher_var2)
    print(f"Teacher uncertain  → fused: {fused2.squeeze().detach()}")

    # Case 3: no uncertainty info → equal average fallback
    fused3 = gate(student_logits, teacher_logits)
    print(f"No uncertainty     → fused: {fused3.squeeze().detach()}")
