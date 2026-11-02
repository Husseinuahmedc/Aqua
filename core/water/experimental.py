"""Independent, configurable traditional-DIP water-mask experiments.

These methods are experimental comparators and are not used by the production
workspace. Inputs are OpenCV-decoded BGR ``uint8`` images; masks use ``uint8``
values 0 and 255 and preserve the source geometry.
"""

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import validate_image
from core.water.regions import validate_binary_mask


ThresholdChannel = Literal["grayscale", "saturation", "value"]


@dataclass(frozen=True)
class HSVColorConfig:
    """Raw HSV color bounds, before blur or morphology."""

    hue_min: int = 70
    hue_max: int = 145
    saturation_min: int = 0
    saturation_max: int = 255
    value_min: int = 15
    value_max: int = 255

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


@dataclass(frozen=True)
class RGBThresholdConfig:
    """Inclusive RGB channel bounds; segmentation input remains BGR."""

    red_min: int = 0
    red_max: int = 255
    green_min: int = 0
    green_max: int = 255
    blue_min: int = 70
    blue_max: int = 255

    def __post_init__(self) -> None:
        for name, value in (
            ("red_min", self.red_min),
            ("red_max", self.red_max),
            ("green_min", self.green_min),
            ("green_max", self.green_max),
            ("blue_min", self.blue_min),
            ("blue_max", self.blue_max),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be in the range 0–255.")
        for channel in ("red", "green", "blue"):
            if getattr(self, f"{channel}_min") > getattr(self, f"{channel}_max"):
                raise ValueError(f"{channel}_min must not exceed {channel}_max.")


@dataclass(frozen=True)
class AdaptiveThresholdConfig:
    """Adaptive threshold settings for grayscale or an HSV-derived channel."""

    channel: ThresholdChannel = "saturation"
    method: Literal["mean", "gaussian"] = "gaussian"
    block_size: int = 31
    constant: int = 5
    invert: bool = True

    def __post_init__(self) -> None:
        if self.channel not in ("grayscale", "saturation", "value"):
            raise ValueError("channel must be grayscale, saturation, or value.")
        if self.method not in ("mean", "gaussian"):
            raise ValueError("method must be mean or gaussian.")
        if (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size < 3
            or self.block_size % 2 == 0
        ):
            raise ValueError("block_size must be an odd integer of at least 3.")
        if isinstance(self.constant, bool) or not isinstance(self.constant, int):
            raise TypeError("constant must be an integer.")
        if not -255 <= self.constant <= 255:
            raise ValueError("constant must be in the range -255 to 255.")
        if not isinstance(self.invert, bool):
            raise TypeError("invert must be a boolean.")


@dataclass(frozen=True)
class OtsuThresholdConfig:
    """Global Otsu threshold settings for grayscale or an HSV-derived channel."""

    channel: ThresholdChannel = "saturation"
    invert: bool = True

    def __post_init__(self) -> None:
        if self.channel not in ("grayscale", "saturation", "value"):
            raise ValueError("channel must be grayscale, saturation, or value.")
        if not isinstance(self.invert, bool):
            raise TypeError("invert must be a boolean.")


@dataclass(frozen=True)
class MorphologyConfig:
    """Optional binary-mask morphology settings for color-mask experiments."""

    operation: Literal["opening", "closing", "open_close"] = "open_close"
    kernel_size: int = 3
    iterations: int = 1

    def __post_init__(self) -> None:
        if self.operation not in ("opening", "closing", "open_close"):
            raise ValueError("operation must be opening, closing, or open_close.")
        if (
            isinstance(self.kernel_size, bool)
            or not isinstance(self.kernel_size, int)
            or self.kernel_size < 1
            or self.kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer.")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or not 0 <= self.iterations <= 10
        ):
            raise ValueError("iterations must be an integer from 0 to 10.")


def segment_hsv_color_threshold(
    image_bgr: np.ndarray,
    config: HSVColorConfig | None = None,
) -> NDArray[np.uint8]:
    """Threshold an OpenCV BGR image in HSV without blur or morphology."""
    validate_image(image_bgr, "BGR")
    settings = config or HSVColorConfig()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    low_sv = (settings.saturation_min, settings.value_min)
    high_sv = (settings.saturation_max, settings.value_max)
    if settings.hue_min <= settings.hue_max:
        return cv2.inRange(
            hsv,
            (settings.hue_min, *low_sv),
            (settings.hue_max, *high_sv),
        )
    high_hue = cv2.inRange(
        hsv,
        (settings.hue_min, *low_sv),
        (179, *high_sv),
    )
    low_hue = cv2.inRange(
        hsv,
        (0, *low_sv),
        (settings.hue_max, *high_sv),
    )
    return cv2.bitwise_or(high_hue, low_hue)


def segment_rgb_threshold(
    image_bgr: np.ndarray,
    config: RGBThresholdConfig | None = None,
) -> NDArray[np.uint8]:
    """Apply inclusive per-channel RGB ranges to an OpenCV BGR image."""
    validate_image(image_bgr, "BGR")
    settings = config or RGBThresholdConfig()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    lower = (settings.red_min, settings.green_min, settings.blue_min)
    upper = (settings.red_max, settings.green_max, settings.blue_max)
    return cv2.inRange(image_rgb, lower, upper)


def _threshold_channel(
    image_bgr: np.ndarray,
    channel: ThresholdChannel,
) -> NDArray[np.uint8]:
    if channel == "grayscale":
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1 if channel == "saturation" else 2]


def segment_adaptive_threshold(
    image_bgr: np.ndarray,
    config: AdaptiveThresholdConfig | None = None,
) -> NDArray[np.uint8]:
    """Apply local mean/Gaussian thresholding to a declared image channel."""
    validate_image(image_bgr, "BGR")
    settings = config or AdaptiveThresholdConfig()
    source = _threshold_channel(image_bgr, settings.channel)
    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if settings.method == "mean"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )
    threshold_type = cv2.THRESH_BINARY_INV if settings.invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        source,
        255,
        adaptive_method,
        threshold_type,
        settings.block_size,
        settings.constant,
    )


def segment_otsu_threshold(
    image_bgr: np.ndarray,
    config: OtsuThresholdConfig | None = None,
) -> NDArray[np.uint8]:
    """Apply global Otsu thresholding to a declared image channel."""
    validate_image(image_bgr, "BGR")
    settings = config or OtsuThresholdConfig()
    source = _threshold_channel(image_bgr, settings.channel)
    threshold_type = cv2.THRESH_BINARY_INV if settings.invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(
        source,
        0,
        255,
        threshold_type | cv2.THRESH_OTSU,
    )
    return mask


def refine_mask_morphology(
    mask: np.ndarray,
    config: MorphologyConfig | None = None,
) -> NDArray[np.uint8]:
    """Apply configured opening, closing, or opening followed by closing."""
    validate_binary_mask(mask)
    settings = config or MorphologyConfig()
    result = mask.copy()
    if settings.iterations == 0 or settings.kernel_size == 1:
        return result
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (settings.kernel_size, settings.kernel_size),
    )
    operations = {
        "opening": (cv2.MORPH_OPEN,),
        "closing": (cv2.MORPH_CLOSE,),
        "open_close": (cv2.MORPH_OPEN, cv2.MORPH_CLOSE),
    }[settings.operation]
    for operation in operations:
        result = cv2.morphologyEx(
            result,
            operation,
            kernel,
            iterations=settings.iterations,
        )
    return validate_binary_mask(result)
