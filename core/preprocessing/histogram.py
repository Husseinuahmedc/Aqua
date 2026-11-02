"""Intensity and color-channel histograms for scientific inspection."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import ColorSpace, validate_image


HistogramChannel = Literal[
    "gray", "blue", "green", "red", "hue", "saturation", "value"
]


@dataclass(frozen=True)
class HistogramData:
    """Counts and bin edges for one explicitly selected image channel."""

    channel: str
    counts: NDArray[np.int64]
    bin_edges: NDArray[np.float64]


def compute_histogram(
    image: np.ndarray,
    color_space: ColorSpace,
    *,
    channel: HistogramChannel | None = None,
    bins: int = 256,
) -> HistogramData:
    """Calculate a histogram for a grayscale image or named color channel.

    BGR channels must be named blue/green/red; RGB channels red/green/blue;
    HSV channels hue/saturation/value. OpenCV uint8 hue bins span 0–180,
    while intensity-like channels span 0–256. Histograms discard spatial
    arrangement and do not establish semantic classes.
    """
    validate_image(image, color_space)
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("Histogram bin count must be an integer.")
    if not 1 <= bins <= 256:
        raise ValueError("Histogram bin count must be between 1 and 256.")

    if color_space == "GRAY":
        if channel not in (None, "gray"):
            raise ValueError("Grayscale input only supports the 'gray' histogram.")
        selected = image
        channel_name = "gray"
        value_range = (0, 256)
    else:
        channel_order = {
            "BGR": {"blue": 0, "green": 1, "red": 2},
            "RGB": {"red": 0, "green": 1, "blue": 2},
            "HSV": {"hue": 0, "saturation": 1, "value": 2},
        }[color_space]
        if channel is None:
            channel = next(iter(channel_order))
        if channel not in channel_order:
            raise ValueError(
                f"Channel '{channel}' is invalid for {color_space} input; "
                f"choose one of {', '.join(channel_order)}."
            )
        selected = image[:, :, channel_order[channel]]
        channel_name = channel
        value_range = (
            (0, 180)
            if color_space == "HSV" and channel == "hue"
            else (0, 256)
        )

    counts, bin_edges = np.histogram(selected, bins=bins, range=value_range)
    return HistogramData(
        channel=channel_name,
        counts=counts.astype(np.int64, copy=False),
        bin_edges=bin_edges.astype(np.float64, copy=False),
    )
