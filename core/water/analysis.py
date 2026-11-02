"""Pixel-based water coverage and frame-to-frame mask comparisons."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from core.water.regions import WaterRegion, extract_regions, validate_binary_mask


@dataclass(frozen=True)
class WaterAnalysis:
    """Pixel counts and connected-region measurements for a binary mask."""

    total_pixels: int
    water_pixels: int
    coverage_percent: float
    region_count: int
    region_areas_pixels: tuple[int, ...]


@dataclass(frozen=True)
class MaskComparison:
    """Change between two same-size binary masks; area units are pixels."""

    first_water_pixels: int
    second_water_pixels: int
    water_pixel_change: int
    first_coverage_percent: float
    second_coverage_percent: float
    coverage_change_percentage_points: float
    changed_pixels: int


def analyze_water(
    mask: np.ndarray,
    *,
    regions: Sequence[WaterRegion] | None = None,
) -> WaterAnalysis:
    """Calculate mask coverage and region areas in pixels, never geography."""
    validate_binary_mask(mask)
    total_pixels = int(mask.size)
    water_pixels = int(np.count_nonzero(mask == 255))
    coverage = 100.0 * water_pixels / total_pixels
    measured_regions = tuple(regions) if regions is not None else extract_regions(mask)
    return WaterAnalysis(
        total_pixels=total_pixels,
        water_pixels=water_pixels,
        coverage_percent=coverage,
        region_count=len(measured_regions),
        region_areas_pixels=tuple(region.area_pixels for region in measured_regions),
    )


def compare_masks(first: np.ndarray, second: np.ndarray) -> MaskComparison:
    """Compare same-size masks using water pixels and percentage points.

    ``water_pixel_change`` and ``coverage_change_percentage_points`` are
    second minus first. ``changed_pixels`` counts all pixels whose mask label
    differs. No real-world area or physical motion is inferred.
    """
    validate_binary_mask(first)
    validate_binary_mask(second)
    if first.shape != second.shape:
        raise ValueError(
            "Masks must have identical dimensions, got "
            f"{first.shape} and {second.shape}."
        )
    total_pixels = int(first.size)
    first_water_pixels = int(np.count_nonzero(first == 255))
    second_water_pixels = int(np.count_nonzero(second == 255))
    first_coverage = 100.0 * first_water_pixels / total_pixels
    second_coverage = 100.0 * second_water_pixels / total_pixels
    return MaskComparison(
        first_water_pixels=first_water_pixels,
        second_water_pixels=second_water_pixels,
        water_pixel_change=second_water_pixels - first_water_pixels,
        first_coverage_percent=first_coverage,
        second_coverage_percent=second_coverage,
        coverage_change_percentage_points=second_coverage - first_coverage,
        changed_pixels=int(np.count_nonzero(first != second)),
    )
