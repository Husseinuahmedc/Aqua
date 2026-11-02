"""Tests for the independent traditional-DIP benchmark methods."""

import cv2
import numpy as np
import pytest

from core.water.experimental import (
    AdaptiveThresholdConfig,
    HSVColorConfig,
    MorphologyConfig,
    OtsuThresholdConfig,
    RGBThresholdConfig,
    refine_mask_morphology,
    segment_adaptive_threshold,
    segment_hsv_color_threshold,
    segment_otsu_threshold,
    segment_rgb_threshold,
)


def test_rgb_threshold_uses_declared_rgb_bounds_from_bgr_input() -> None:
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    image[:, :10] = (0, 0, 255)  # BGR red.
    image[:, 10:] = (255, 0, 0)  # BGR blue.
    config = RGBThresholdConfig(
        red_min=200,
        red_max=255,
        green_min=0,
        green_max=60,
        blue_min=0,
        blue_max=60,
    )

    mask = segment_rgb_threshold(image, config)

    assert np.all(mask[:, :10] == 255)
    assert np.all(mask[:, 10:] == 0)
    assert mask.shape == image.shape[:2]


def test_hsv_color_threshold_supports_wrapped_hue_interval() -> None:
    hsv = np.zeros((8, 12, 3), dtype=np.uint8)
    hsv[:, :6] = (175, 200, 200)
    hsv[:, 6:] = (5, 200, 200)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = segment_hsv_color_threshold(
        image,
        HSVColorConfig(
            hue_min=170,
            hue_max=10,
            saturation_min=100,
            value_min=100,
        ),
    )

    assert np.all(mask == 255)
    assert set(np.unique(mask)) == {255}


def test_adaptive_threshold_on_saturation_returns_binary_same_size_mask() -> None:
    hsv = np.zeros((64, 64, 3), dtype=np.uint8)
    hsv[:, :32] = (90, 10, 180)
    hsv[:, 32:] = (90, 220, 180)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    mask = segment_adaptive_threshold(
        image,
        AdaptiveThresholdConfig(
            channel="saturation",
            method="gaussian",
            block_size=15,
            constant=2,
            invert=True,
        ),
    )

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask) > 0


def test_otsu_threshold_separates_two_level_grayscale_fixture() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255

    mask = segment_otsu_threshold(
        image,
        OtsuThresholdConfig(channel="grayscale", invert=True),
    )

    assert np.all(mask[:, :16] == 255)
    assert np.all(mask[:, 16:] == 0)


def test_morphology_removes_isolated_noise_without_mutating_input() -> None:
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[3:12, 3:12] = 255
    mask[1, 1] = 255
    original = mask.copy()

    refined = refine_mask_morphology(
        mask,
        MorphologyConfig(operation="opening", kernel_size=3, iterations=1),
    )

    assert refined[1, 1] == 0
    assert refined[7, 7] == 255
    assert np.array_equal(mask, original)
    assert set(np.unique(refined)).issubset({0, 255})


@pytest.mark.parametrize(
    "config_type,kwargs",
    [
        (AdaptiveThresholdConfig, {"block_size": 4}),
        (OtsuThresholdConfig, {"channel": "hue"}),
        (MorphologyConfig, {"kernel_size": 2}),
        (RGBThresholdConfig, {"red_min": 200, "red_max": 100}),
        (HSVColorConfig, {"hue_min": 180}),
    ],
)
def test_experimental_configs_reject_invalid_parameters(
    config_type: type,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        config_type(**kwargs)
