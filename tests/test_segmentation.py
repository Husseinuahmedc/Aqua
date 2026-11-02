"""Tests for the traditional HSV threshold and refinement pipeline."""

import numpy as np
import pytest

from core.water.regions import (
    extract_regions,
    remove_small_regions,
    validate_binary_mask,
)
from core.water.segmentation import SegmentationConfig, segment_water


def no_filter_config(**overrides: object) -> SegmentationConfig:
    values: dict[str, object] = {
        "hue_min": 110,
        "hue_max": 130,
        "saturation_min": 200,
        "saturation_max": 255,
        "value_min": 100,
        "value_max": 255,
        "blur_method": "none",
        "morphology_kernel_size": 1,
        "morphology_iterations": 0,
        "min_region_area": 0,
    }
    values.update(overrides)
    return SegmentationConfig(**values)  # type: ignore[arg-type]


def test_hsv_segmentation_generates_same_size_binary_mask_without_mutation() -> None:
    image = np.full((20, 24, 3), (0, 0, 255), dtype=np.uint8)
    image[5:11, 7:15] = (255, 0, 0)  # OpenCV BGR blue: HSV hue 120.
    original = image.copy()

    result = segment_water(image, no_filter_config())

    assert result.mask.shape == image.shape[:2]
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 255}
    assert int(np.count_nonzero(result.mask)) == 48
    assert len(result.regions) == 1
    assert result.regions[0].area_pixels == 48
    np.testing.assert_array_equal(image, original)


def test_hue_interval_can_wrap_across_zero() -> None:
    image = np.array(
        [[(0, 0, 255), (8, 0, 255), (255, 0, 0)]],
        dtype=np.uint8,
    )
    config = no_filter_config(
        hue_min=170,
        hue_max=5,
        saturation_min=50,
        min_region_area=0,
    )

    result = segment_water(image, config)

    assert result.mask.tolist() == [[255, 255, 0]]


def test_morphology_and_minimum_area_remove_small_components() -> None:
    image = np.full((24, 24, 3), (0, 0, 255), dtype=np.uint8)
    image[2:12, 2:12] = (255, 0, 0)
    image[18, 18] = (255, 0, 0)

    result = segment_water(
        image,
        no_filter_config(
            min_region_area=4,
            morphology_kernel_size=3,
            morphology_iterations=1,
        ),
    )

    assert len(result.regions) == 1
    assert result.regions[0].area_pixels >= 80
    assert result.mask[18, 18] == 0


def test_region_area_is_pixel_count_and_contour_is_extracted() -> None:
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    mask[5, 7] = 255

    regions = extract_regions(mask, min_area_pixels=2)

    assert len(regions) == 1
    assert regions[0].area_pixels == 6
    assert (regions[0].x, regions[0].y) == (1, 1)
    assert (regions[0].width, regions[0].height) == (3, 2)
    assert regions[0].contour is not None


def test_remove_small_regions_preserves_full_size_component() -> None:
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[0, 0] = 255
    mask[2:5, 2:5] = 255

    filtered = remove_small_regions(mask, 4)

    assert filtered.shape == mask.shape
    assert filtered.dtype == np.uint8
    assert filtered[0, 0] == 0
    assert np.count_nonzero(filtered) == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hue_min": -1},
        {"hue_max": 180},
        {"saturation_min": 200, "saturation_max": 100},
        {"value_min": 200, "value_max": 100},
        {"blur_method": "median", "blur_kernel_size": 4},
        {"morphology_kernel_size": 2},
        {"morphology_iterations": 11},
        {"min_region_area": -1},
    ],
)
def test_segmentation_config_rejects_invalid_parameters(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        no_filter_config(**kwargs)


def test_mask_validation_rejects_non_binary_encoding() -> None:
    mask = np.array([[0, 128]], dtype=np.uint8)
    with pytest.raises(ValueError, match="0 and/or 255"):
        validate_binary_mask(mask)
