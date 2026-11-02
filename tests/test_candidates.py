"""Tests for experimental, geometry-only candidate-region proposals."""

import numpy as np
import pytest

from core.objects.candidates import extract_candidate_regions
from core.objects.visualization import draw_candidate_boxes


def test_candidate_regions_report_boxes_and_basic_shape_properties() -> None:
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[2:7, 3:11] = 255
    mask[12:18, 16:20] = 255

    candidates = extract_candidate_regions(mask, min_area_pixels=10)

    assert len(candidates) == 2
    first, second = candidates
    assert (first.index, first.x, first.y, first.width, first.height) == (1, 3, 2, 8, 5)
    assert first.area_pixels == 40
    assert first.aspect_ratio == pytest.approx(1.6)
    assert first.extent == pytest.approx(1.0)
    assert second.area_pixels == 24
    assert second.index == 2


def test_candidate_filters_apply_area_and_extent_bounds() -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    mask[8, 8] = 255

    candidates = extract_candidate_regions(
        mask,
        min_area_pixels=2,
        max_area_pixels=20,
        min_extent=0.8,
    )

    assert len(candidates) == 1
    assert candidates[0].area_pixels == 16


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_area_pixels": -1},
        {"min_area_pixels": 8, "max_area_pixels": 7},
        {"min_extent": -0.1},
        {"min_extent": 1.1},
    ],
)
def test_candidate_extraction_rejects_invalid_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        extract_candidate_regions(np.zeros((4, 4), dtype=np.uint8), **kwargs)


def test_empty_mask_returns_no_candidates() -> None:
    assert extract_candidate_regions(np.zeros((5, 7), dtype=np.uint8)) == ()


def test_candidate_overlay_draws_boxes_on_a_copy() -> None:
    image = np.zeros((12, 14, 3), dtype=np.uint8)
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[3:8, 4:10] = 255
    candidates = extract_candidate_regions(mask)

    overlay = draw_candidate_boxes(image, candidates)

    assert overlay.shape == image.shape
    assert np.any(overlay != image)
    assert np.count_nonzero(image) == 0
