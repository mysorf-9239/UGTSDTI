import torch

from ugtsdti.losses.kd import KDLoss


def test_kd_loss():
    loss_fn = KDLoss(alpha=0.5)

    student_logits = torch.tensor([1.0, -1.0], requires_grad=True)
    teacher_logits = torch.tensor([2.0, -2.0])
    labels = torch.tensor([1.0, 0.0])

    model_outputs = {"logits": student_logits, "student_logits": student_logits, "teacher_logits": teacher_logits}

    loss = loss_fn(model_outputs, labels)

    assert loss.dim() == 0
    assert not torch.isnan(loss)

    loss.backward()
    assert student_logits.grad is not None
