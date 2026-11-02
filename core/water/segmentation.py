"""Configurable HSV thresholding and binary-mask refinement."""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.transforms import BlurMethod, apply_blur
from core.preprocessing.validation import validate_image
from core.water.regions import WaterRegion, extract_regions, remove_small_regions


@dataclass(frozen=True)
class SegmentationConfig:
    """Explicit parameters for the traditional HSV water-mask baseline.

    Hue bounds use OpenCV's uint8 HSV range 0–179. When ``hue_min`` is greater
    than ``hue_max``, the hue interval wraps across zero. The selected range is
    a starting hypothesis for a particular image, not a universal water color.
    """

    hue_min: int = 80
    hue_max: int = 130
    saturation_min: int = 30
    saturation_max: int = 255
    value_min: int = 20
    value_max: int = 255
    blur_method: BlurMethod = "gaussian"
    blur_kernel_size: int = 5
    morphology_kernel_size: int = 3
    morphology_iterations: int = 1
    min_region_area: int = 40

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("hue_min", self.hue_min, 179),
            ("hue_max", self.hue_max, 179),
            ("saturation_min", self.saturation_min, 255),
            ("saturation_max", self.saturation_max, 255),
            ("value_min", self.value_min, 255),
            ("value_max", self.value_max, 255),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if not 0 <= value <= maximum:
                raise ValueError(f"{name} must be in the range 0–{maximum}.")
        if self.saturation_min > self.saturation_max:
            raise ValueError("saturation_min must not exceed saturation_max.")
        if self.value_min > self.value_max:
            raise ValueError("value_min must not exceed value_max.")
        if self.blur_method not in ("none", "gaussian", "median"):
            raise ValueError("blur_method must be none, gaussian, or median.")
        if self.blur_method != "none":
            if (
                isinstance(self.blur_kernel_size, bool)
                or not isinstance(self.blur_kernel_size, int)
                or self.blur_kernel_size < 3
                or self.blur_kernel_size % 2 == 0
            ):
                raise ValueError("Blur kernel size must be odd and at least 3.")
        if (
            isinstance(self.morphology_kernel_size, bool)
            or not isinstance(self.morphology_kernel_size, int)
            or self.morphology_kernel_size < 1
            or self.morphology_kernel_size % 2 == 0
        ):
            raise ValueError("Morphology kernel size must be a positive odd integer.")
        if (
            isinstance(self.morphology_iterations, bool)
            or not isinstance(self.morphology_iterations, int)
            or not 0 <= self.morphology_iterations <= 10
        ):
            raise ValueError("Morphology iterations must be an integer from 0 to 10.")
        if (
            isinstance(self.min_region_area, bool)
            or not isinstance(self.min_region_area, int)
            or self.min_region_area < 0
        ):
            raise ValueError("Minimum region area must be a non-negative integer.")


@dataclass(frozen=True)
class SegmentationResult:
    """Potential-water mask and connected regions for one source image."""

    mask: NDArray[np.uint8]
    regions: tuple[WaterRegion, ...]
    config: SegmentationConfig


def _hsv_threshold(
    hsv: NDArray[np.uint8], config: SegmentationConfig
) -> NDArray[np.uint8]:
    lower_sv = (config.saturation_min, config.value_min)
    upper_sv = (config.saturation_max, config.value_max)
    if config.hue_min <= config.hue_max:
        lower = (config.hue_min, *lower_sv)
        upper = (config.hue_max, *upper_sv)
        return cv2.inRange(hsv, lower, upper)

    # A wrapped hue range is the union of its high-hue and low-hue intervals.
    high_hue = cv2.inRange(
        hsv,
        (config.hue_min, *lower_sv),
        (179, *upper_sv),
    )
    low_hue = cv2.inRange(
        hsv,
        (0, *lower_sv),
        (config.hue_max, *upper_sv),
    )
    return cv2.bitwise_or(high_hue, low_hue)


def segment_water(
    image_bgr: np.ndarray,
    config: SegmentationConfig | None = None,
) -> SegmentationResult:
    """Generate a potential-water mask from an OpenCV-decoded BGR image.

    Processing order: selected Gaussian/median smoothing, BGR→HSV conversion,
    configurable HSV threshold, morphological opening and closing, then exact
    connected-component pixel-area filtering. Input data are not modified and
    output dimensions match the source. Colors alone cannot distinguish every
    floodwater condition from permanent water or water-like surfaces.
    """
    validate_image(image_bgr, "BGR")
    settings = config or SegmentationConfig()
    filtered = apply_blur(
        image_bgr,
        settings.blur_method,
        settings.blur_kernel_size,
        color_space="BGR",
    )
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    mask = _hsv_threshold(hsv, settings)

    kernel_size = settings.morphology_kernel_size
    iterations = settings.morphology_iterations
    if kernel_size > 1 and iterations > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

    mask = remove_small_regions(mask, settings.min_region_area)
    regions = extract_regions(mask, min_area_pixels=settings.min_region_area)
    if mask.shape != image_bgr.shape[:2]:
        raise RuntimeError("Segmentation unexpectedly changed image dimensions.")
    return SegmentationResult(mask=mask, regions=regions, config=settings)
