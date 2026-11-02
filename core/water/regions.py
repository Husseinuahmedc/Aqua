"""Connected-region measurements and contour extraction for binary masks."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class WaterRegion:
    """Geometry for one connected foreground component.

    ``area_pixels`` is the exact count of foreground pixels in the component.
    It is a pixel measurement, not a geographic area.
    """

    label: int
    area_pixels: int
    x: int
    y: int
    width: int
    height: int
    centroid_x: float
    centroid_y: float
    contour: NDArray[np.int32] | None = field(default=None, repr=False, compare=False)


def validate_binary_mask(mask: np.ndarray) -> NDArray[np.uint8]:
    """Require a uint8 H×W mask encoded with exactly 0 and 255."""
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array.")
    if mask.dtype != np.uint8:
        raise TypeError(f"Mask dtype must be uint8, got {mask.dtype}.")
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("Mask must have non-empty shape (height, width).")
    values = np.unique(mask)
    if not np.isin(values, (0, 255)).all():
        raise ValueError("Binary mask values must be exactly 0 and/or 255.")
    return mask


def _connected_components(
    mask: NDArray[np.uint8],
) -> tuple[int, NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]:
    return cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )


def remove_small_regions(
    mask: np.ndarray, min_area_pixels: int
) -> NDArray[np.uint8]:
    """Remove connected foreground components smaller than a pixel area."""
    validate_binary_mask(mask)
    if isinstance(min_area_pixels, bool) or not isinstance(min_area_pixels, int):
        raise TypeError("Minimum region area must be an integer number of pixels.")
    if min_area_pixels < 0:
        raise ValueError("Minimum region area cannot be negative.")

    count, labels, stats, _ = _connected_components(mask)
    keep = np.zeros(count, dtype=bool)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1:] = areas >= min_area_pixels
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def extract_regions(
    mask: np.ndarray, min_area_pixels: int = 1
) -> tuple[WaterRegion, ...]:
    """Measure connected foreground regions and extract their outer contours.

    Components use 8-connectivity. ``min_area_pixels`` filters by exact pixel
    count, not by OpenCV's contour-area estimate.
    """
    validate_binary_mask(mask)
    if isinstance(min_area_pixels, bool) or not isinstance(min_area_pixels, int):
        raise TypeError("Minimum region area must be an integer number of pixels.")
    if min_area_pixels < 0:
        raise ValueError("Minimum region area cannot be negative.")

    count, labels, stats, centroids = _connected_components(mask)
    regions: list[WaterRegion] = []
    output_label = 1
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area_pixels:
            continue

        component = np.where(
            labels[y : y + height, x : x + width] == label, 255, 0
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour: NDArray[np.int32] | None = None
        if contours:
            contour = max(contours, key=cv2.contourArea)
            contour[:, 0, 0] += x
            contour[:, 0, 1] += y

        regions.append(
            WaterRegion(
                label=output_label,
                area_pixels=area,
                x=x,
                y=y,
                width=width,
                height=height,
                centroid_x=float(centroids[label, 0]),
                centroid_y=float(centroids[label, 1]),
                contour=contour,
            )
        )
        output_label += 1
    return tuple(regions)
