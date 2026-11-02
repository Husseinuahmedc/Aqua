"""Mathematical metrics for predicted and manually annotated binary masks."""

from dataclasses import asdict, dataclass

import numpy as np

from core.water.regions import validate_binary_mask


@dataclass(frozen=True)
class EvaluationMetrics:
    """Confusion counts and overlap metrics for a binary foreground class."""

    intersection_over_union: float
    dice_coefficient: float
    precision: float | None
    recall: float | None
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    total_pixels: int

    def to_record(self) -> dict[str, float | int | None]:
        """Return a tabular representation suitable for CSV export."""
        return asdict(self)


def evaluate_masks(
    predicted_mask: np.ndarray,
    reference_mask: np.ndarray,
) -> EvaluationMetrics:
    """Compare two same-size 0/255 masks using the foreground as positive.

    IoU and Dice are 1.0 when both masks contain no foreground and 0.0 when
    exactly one is empty. Precision is ``None`` when there are no predicted
    positives; Recall is ``None`` when the reference contains no positives.
    """
    validate_binary_mask(predicted_mask)
    validate_binary_mask(reference_mask)
    if predicted_mask.shape != reference_mask.shape:
        raise ValueError(
            "Predicted and reference mask dimensions must match; "
            f"got {predicted_mask.shape} and {reference_mask.shape}."
        )

    predicted = predicted_mask == 255
    reference = reference_mask == 255
    true_positive = int(np.count_nonzero(predicted & reference))
    false_positive = int(np.count_nonzero(predicted & ~reference))
    false_negative = int(np.count_nonzero(~predicted & reference))
    true_negative = int(np.count_nonzero(~predicted & ~reference))
    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union else 1.0
    dice_denominator = 2 * true_positive + false_positive + false_negative
    dice = 2 * true_positive / dice_denominator if dice_denominator else 1.0
    predicted_positive = true_positive + false_positive
    reference_positive = true_positive + false_negative
    precision = (
        true_positive / predicted_positive if predicted_positive else None
    )
    recall = true_positive / reference_positive if reference_positive else None

    return EvaluationMetrics(
        intersection_over_union=float(iou),
        dice_coefficient=float(dice),
        precision=None if precision is None else float(precision),
        recall=None if recall is None else float(recall),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        true_negative=true_negative,
        total_pixels=int(predicted_mask.size),
    )
