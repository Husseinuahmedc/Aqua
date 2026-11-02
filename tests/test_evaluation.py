"""Tests for manual mask normalization and binary evaluation metrics."""

from pathlib import Path

import numpy as np
import pytest

from core.evaluation.masks import load_reference_mask, normalize_reference_mask
from core.evaluation.metrics import evaluate_masks
from core.preprocessing.image_io import save_image


def test_evaluation_metrics_have_known_expected_values() -> None:
    prediction = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    reference = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    metrics = evaluate_masks(prediction, reference)

    assert metrics.intersection_over_union == pytest.approx(1 / 3)
    assert metrics.dice_coefficient == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert (metrics.true_positive, metrics.false_positive) == (1, 1)
    assert (metrics.false_negative, metrics.true_negative) == (1, 1)
    assert metrics.total_pixels == 4


def test_empty_masks_use_documented_zero_denominator_convention() -> None:
    empty = np.zeros((2, 2), dtype=np.uint8)
    both_empty = evaluate_masks(empty, empty)
    one_empty = evaluate_masks(
        np.array([[0, 0], [0, 255]], dtype=np.uint8), empty
    )

    assert both_empty.intersection_over_union == 1.0
    assert both_empty.dice_coefficient == 1.0
    assert both_empty.precision is None
    assert both_empty.recall is None
    assert one_empty.intersection_over_union == 0.0
    assert one_empty.dice_coefficient == 0.0
    assert one_empty.precision == 0.0
    assert one_empty.recall is None


def test_evaluation_rejects_size_and_encoding_mismatch() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        evaluate_masks(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((3, 2), dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="0 and/or 255"):
        evaluate_masks(
            np.array([[0, 128]], dtype=np.uint8),
            np.zeros((1, 2), dtype=np.uint8),
        )


def test_reference_mask_normalizes_only_explicit_binary_encodings() -> None:
    binary = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    normalized = normalize_reference_mask(binary)

    assert normalized.dtype == np.uint8
    assert normalized.tolist() == [[0, 255], [255, 0]]
    with pytest.raises(ValueError, match="only 0/255 or explicitly binary"):
        normalize_reference_mask(np.array([[0, 128]], dtype=np.uint8))


def test_load_reference_mask_from_unicode_path(tmp_path: Path) -> None:
    path = tmp_path / "mask α.png"
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    save_image(path, mask, color_space="GRAY")

    loaded = load_reference_mask(path)

    np.testing.assert_array_equal(loaded, mask)


def test_reference_mask_rejects_non_uint8() -> None:
    with pytest.raises(TypeError, match="uint8"):
        normalize_reference_mask(np.zeros((2, 2), dtype=np.uint16))
