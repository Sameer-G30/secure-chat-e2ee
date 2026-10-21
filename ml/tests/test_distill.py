"""Unit tests for the distillation loss (no model downloads)."""

import torch

from secure_chat_ml.distill import distillation_loss


def test_loss_is_zero_soft_term_when_student_matches_teacher() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]])
    labels = torch.tensor([0, 1])
    weights = torch.ones(2)
    soft_only = distillation_loss(logits, logits, labels, weights, alpha=0.0, temperature=2.0)
    assert float(soft_only) < 1e-6


def test_alpha_one_equals_hard_cross_entropy() -> None:
    student = torch.tensor([[0.3, 0.1], [0.2, 0.9]])
    teacher = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    labels = torch.tensor([1, 1])
    weights = torch.tensor([1.0, 2.0])
    loss = distillation_loss(student, teacher, labels, weights, alpha=1.0, temperature=3.0)
    expected = torch.nn.functional.cross_entropy(student, labels, weight=weights)
    assert torch.isclose(loss, expected)
