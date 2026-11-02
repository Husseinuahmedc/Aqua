"""Experimental geometry-only region proposals from binary image masks."""

from dataclasses import dataclass

import cv2
import numpy as np

from core.water.regions import extract_regions, validate_binary_mask


@dataclass(frozen=True)
class CandidateRegion:
    """A geometric region for human inspection, not a semantic detection."""

    index: int
    x: int
    y: int
    width: int
    height: int
    area_pixels: int
    contour_area_pixels: float
    aspect_ratio: float
    extent: float
    centroid_x: float
    centroid_y: float


def extract_candidate_regions(
    mask: np.ndarray,
    *,
    min_area_pixels: int = 20,
    max_area_pixels: int | None = None,
    min_extent: float = 0.0,
) -> tuple[CandidateRegion, ...]:
    """Extract geometry-only proposals from a binary mask.

    The input can be a potential-water mask or an explicitly thresholded
    difference map. Area filters and 8-connected contours produce bounding
    rectangles and basic shape ratios. Results are experimental and do not
    identify people, vehicles, or other semantic objects.
    """
    validate_binary_mask(mask)
    if isinstance(min_area_pixels, bool) or not isinstance(min_area_pixels, int):
        raise TypeError("Minimum candidate area must be an integer number of pixels.")
    if min_area_pixels < 0:
        raise ValueError("Minimum candidate area cannot be negative.")
    if max_area_pixels is not None:
        if isinstance(max_area_pixels, bool) or not isinstance(max_area_pixels, int):
            raise TypeError(
                "Maximum candidate area must be an integer number of pixels."
            )
        if max_area_pixels < min_area_pixels:
            raise ValueError("Maximum candidate area must not be below the minimum.")
    if not 0.0 <= min_extent <= 1.0:
        raise ValueError("Minimum extent must be between 0 and 1.")

    candidates: list[CandidateRegion] = []
    for region in extract_regions(mask, min_area_pixels=min_area_pixels):
        if max_area_pixels is not None and region.area_pixels > max_area_pixels:
            continue
        box_area = region.width * region.height
        aspect_ratio = region.width / region.height
        extent = region.area_pixels / box_area if box_area else 0.0
        if extent < min_extent:
            continue
        contour_area = (
            float(cv2.contourArea(region.contour))
            if region.contour is not None
            else 0.0
        )
        candidates.append(
            CandidateRegion(
                index=len(candidates) + 1,
                x=region.x,
                y=region.y,
                width=region.width,
                height=region.height,
                area_pixels=region.area_pixels,
                contour_area_pixels=contour_area,
                aspect_ratio=float(aspect_ratio),
                extent=float(extent),
                centroid_x=region.centroid_x,
                centroid_y=region.centroid_y,
            )
        )
    return tuple(candidates)
