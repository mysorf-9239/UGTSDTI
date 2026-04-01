"""Tests for decision fallback edge cases and error handling."""

import pytest

from ugtsdti.core.context import ExecutionContext
from ugtsdti.core.errors import InvalidDecisionOutputError, MissingDependencyError
from ugtsdti.core.state import State, StateWriter
from ugtsdti.decision.gate import HardSelectionDecisionModule, SoftBlendingDecisionModule
from ugtsdti.decision.trust import TrustEstimator


class TestDecisionFallbackEdgeCases:
    """Test decision module fallback behavior in edge cases."""

    def test_no_teacher_no_student_fallback(self):
        """Test fallback when both teacher and student are missing."""
        state = State()
        StateWriter(state)

        # No teacher.logits or student.logits in state
        module = SoftBlendingDecisionModule(fallback={"no_teacher": "student", "no_student": "teacher"})

        # Should raise error - both missing
        with pytest.raises(InvalidDecisionOutputError, match="Missing required logits"):
            module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

    def test_no_teacher_with_student_fallback(self):
        """Test fallback when teacher is missing but student exists."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Only student logits available
        student_logits = torch.tensor([[0.2, 0.8]])
        writer.commit("roles", {"student.logits": student_logits})

        module = SoftBlendingDecisionModule(fallback={"no_teacher": "student"})

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should fallback to student
        assert torch.equal(outputs["logits"], student_logits)
        assert "gate.alpha" not in outputs, "No gate.alpha when only one branch"

    def test_no_student_with_teacher_fallback(self):
        """Test fallback when student is missing but teacher exists."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Only teacher logits available
        teacher_logits = torch.tensor([[0.3, 0.7]])
        writer.commit("roles", {"teacher.logits": teacher_logits})

        module = SoftBlendingDecisionModule(fallback={"no_student": "teacher"})

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should fallback to teacher
        assert torch.equal(outputs["logits"], teacher_logits)
        assert "gate.alpha" not in outputs, "No gate.alpha when only one branch"

    def test_invalid_fallback_configuration(self):
        """Test error handling with invalid fallback configuration."""
        state = State()
        StateWriter(state)

        # Invalid fallback key
        module = SoftBlendingDecisionModule(fallback={"invalid_key": "student"})

        with pytest.raises(InvalidDecisionOutputError):
            module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

    def test_fallback_with_uncertainty_missing(self):
        """Test fallback when uncertainty is required but missing."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Teacher and student available, but no uncertainty
        teacher_logits = torch.tensor([[0.6, 0.4]])
        student_logits = torch.tensor([[0.3, 0.7]])
        writer.commit("roles", {"teacher.logits": teacher_logits, "student.logits": student_logits})

        # Module that requires uncertainty
        trust_estimator = TrustEstimator(use_uncertainty=True)
        module = SoftBlendingDecisionModule(trust_estimator=trust_estimator)

        # Should raise error about missing uncertainty
        with pytest.raises(MissingDependencyError, match="uncertainty"):
            module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

    def test_fallback_with_uncertainty_available(self):
        """Test fallback when uncertainty is available."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # All required signals available
        teacher_logits = torch.tensor([[0.6, 0.4]])
        student_logits = torch.tensor([[0.3, 0.7]])
        teacher_var = torch.tensor([[0.1, 0.1]])
        student_var = torch.tensor([[0.2, 0.2]])

        writer.commit("roles", {"teacher.logits": teacher_logits, "student.logits": student_logits})
        writer.commit("interaction", {"teacher.var": teacher_var, "student.var": student_var})

        # Module that requires uncertainty
        trust_estimator = TrustEstimator(use_uncertainty=True)
        module = SoftBlendingDecisionModule(trust_estimator=trust_estimator)

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should work normally
        assert "logits" in outputs
        assert "gate.alpha" in outputs
        assert 0.0 <= outputs["gate.alpha"] <= 1.0, "gate.alpha should be bounded"

    def test_hard_selection_fallback(self):
        """Test hard selection decision module fallback."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Only student available
        student_logits = torch.tensor([[0.2, 0.8]])
        writer.commit("roles", {"student.logits": student_logits})

        module = HardSelectionDecisionModule(threshold=0.5, fallback={"no_teacher": "student"})

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should fallback to student
        assert torch.equal(outputs["logits"], student_logits)
        assert "gate.alpha" in outputs, "Hard selection should still produce gate.alpha"
        assert outputs["gate.alpha"] == 0.0 or outputs["gate.alpha"] == 1.0, "Hard selection should be 0 or 1"

    def test_fallback_with_no_uncertainty_fallback(self):
        """Test fallback when uncertainty is missing but fallback is configured."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Teacher and student available, but no uncertainty
        teacher_logits = torch.tensor([[0.6, 0.4]])
        student_logits = torch.tensor([[0.3, 0.7]])
        writer.commit("roles", {"teacher.logits": teacher_logits, "student.logits": student_logits})

        # Module that requires uncertainty but has fallback
        trust_estimator = TrustEstimator(use_uncertainty=True)
        module = SoftBlendingDecisionModule(trust_estimator=trust_estimator, fallback={"no_uncertainty": "student"})

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should fallback to student when uncertainty missing
        assert torch.equal(outputs["logits"], student_logits)
        assert "gate.alpha" not in outputs, "No gate.alpha in fallback mode"

    def test_fallback_priority_order(self):
        """Test that fallback priority is handled correctly."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Only student available
        student_logits = torch.tensor([[0.2, 0.8]])
        writer.commit("roles", {"student.logits": student_logits})

        # Multiple fallback configurations
        module = SoftBlendingDecisionModule(
            fallback={"no_teacher": "student", "no_student": "teacher", "no_uncertainty": "student"}
        )

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should use no_teacher fallback (student is available)
        assert torch.equal(outputs["logits"], student_logits)

    def test_fallback_with_corrupted_logits(self):
        """Test fallback when logits are corrupted or invalid."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Corrupted teacher logits (NaN)
        teacher_logits = torch.tensor([[float("nan"), 0.4]])
        student_logits = torch.tensor([[0.3, 0.7]])

        writer.commit("roles", {"teacher.logits": teacher_logits, "student.logits": student_logits})

        module = SoftBlendingDecisionModule(fallback={"no_teacher": "student"})

        # Should handle corrupted logits gracefully
        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should still produce some output
        assert "logits" in outputs
        # Behavior with NaN depends on implementation, but shouldn't crash

    def test_fallback_empty_state(self):
        """Test fallback behavior with completely empty state."""
        state = State()
        StateWriter(state)

        module = SoftBlendingDecisionModule(fallback={"no_teacher": "student", "no_student": "teacher"})

        # Should raise error - nothing to fallback to
        with pytest.raises(InvalidDecisionOutputError):
            module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

    def test_fallback_with_mixed_modalities(self):
        """Test fallback when modalities are different between branches."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        state = State()
        writer = StateWriter(state)

        # Student logits available, teacher missing
        student_logits = torch.tensor([[0.2, 0.8, 0.0]])  # 3-class (sequence-only)
        writer.commit("roles", {"student.logits": student_logits})

        module = SoftBlendingDecisionModule(fallback={"no_teacher": "student"})

        outputs = module.forward(state, ExecutionContext(mode="eval", seed=0, device="cpu", deterministic=False))

        # Should fallback to student with its modality
        assert torch.equal(outputs["logits"], student_logits)
        assert outputs["logits"].shape[1] == 3, "Should preserve student's modality"
