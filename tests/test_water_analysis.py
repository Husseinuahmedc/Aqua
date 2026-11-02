"""Tests for pixel counts, region metrics, overlays, and mask comparisons."""

import numpy as np
import pytest

from core.water.analysis import analyze_water, compare_masks
from core.water.overlay import create_overlay
from core.water.regions import extract_regions


def test_water_analysis_reports_pixel_units_and_exact_regions() -> None:
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[0, 0:2] = 255
    mask[2:4, 3] = 255

    analysis = analyze_water(mask)

    assert analysis.total_pixels == 20
    assert analysis.water_pixels == 4
    assert analysis.coverage_percent == 20.0
    assert analysis.region_count == 2
    assert analysis.region_areas_pixels == (2, 2)


def test_diagonal_foreground_pixels_share_eight_connected_region() -> None:
    mask = np.eye(3, dtype=np.uint8) * 255
    regions = extract_regions(mask)

    assert len(regions) == 1
    assert regions[0].area_pixels == 3


def test_compare_masks_reports_signed_change_and_changed_pixels() -> None:
    first = np.zeros((3, 4), dtype=np.uint8)
    first[0, 0:2] = 255
    second = np.zeros_like(first)
    second[1, 0:3] = 255

    comparison = compare_masks(first, second)

    assert comparison.first_water_pixels == 2
    assert comparison.second_water_pixels == 3
    assert comparison.water_pixel_change == 1
    assert comparison.first_coverage_percent == pytest.approx(100 / 6)
    assert comparison.second_coverage_percent == pytest.approx(25.0)
    assert comparison.coverage_change_percentage_points == pytest.approx(25 - 100 / 6)
    assert comparison.changed_pixels == 5


def test_compare_masks_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="identical dimensions"):
        compare_masks(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=np.uint8),
        )


def test_overlay_tints_mask_without_mutating_source() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1:3, 2:4] = 255
    original = image.copy()

    overlay = create_overlay(image, mask)

    assert overlay.shape == image.shape
    assert np.all(overlay[mask == 0] == 0)
    assert np.any(overlay[mask == 255] != 0)
    np.testing.assert_array_equal(image, original)


def test_overlay_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions must match"):
        create_overlay(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((3, 4), dtype=np.uint8),
        )
