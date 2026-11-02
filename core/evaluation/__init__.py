"""Evaluation helpers for binary water masks and manual references."""

from core.evaluation.masks import load_reference_mask, normalize_reference_mask
from core.evaluation.metrics import EvaluationMetrics, evaluate_masks

__all__ = [
    "EvaluationMetrics",
    "evaluate_masks",
    "load_reference_mask",
    "normalize_reference_mask",
]
