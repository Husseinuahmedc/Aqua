"""Tests for explicit per-channel intensity histogram data."""

import numpy as np
import pytest

from core.preprocessing.histogram import compute_histogram


def test_grayscale_histogram_matches_known_pixel_counts() -> None:
    image = np.array([[0, 0, 128], [255, 128, 255]], dtype=np.uint8)

    histogram = compute_histogram(image, "GRAY", bins=256)

    assert histogram.channel == "gray"
    assert histogram.counts.shape == (256,)
    assert histogram.counts[0] == 2
    assert histogram.counts[128] == 2
    assert histogram.counts[255] == 2
    assert int(histogram.counts.sum()) == image.size


@pytest.mark.parametrize(
    ("color_space", "channel", "expected_bin"),
    [
        ("BGR", "blue", 255),
        ("RGB", "red", 255),
        ("HSV", "hue", 0),
    ],
)
def test_color_histogram_uses_explicit_channel_names(
    color_space: str,
    channel: str,
    expected_bin: int,
) -> None:
    if color_space == "HSV":
        image = np.array([[[0, 255, 255]]], dtype=np.uint8)
    else:
        image = np.array([[[255, 0, 0]]], dtype=np.uint8)

    histogram = compute_histogram(image, color_space, channel=channel)

    assert histogram.channel == channel
    assert int(histogram.counts.sum()) == 1
    assert histogram.counts[expected_bin] == 1


def test_histogram_rejects_invalid_channels_and_bins() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid for BGR"):
        compute_histogram(image, "BGR", channel="hue")
    with pytest.raises(ValueError, match="between 1 and 256"):
        compute_histogram(image, "BGR", bins=0)
