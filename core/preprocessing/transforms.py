"""Color conversions and smoothing operations for uint8 images."""

from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import ColorSpace, validate_image


BlurMethod = Literal["none", "gaussian", "median"]


def convert_color(
    image: np.ndarray,
    source_space: ColorSpace,
    target_space: ColorSpace,
) -> NDArray[np.uint8]:
    """Convert between BGR, RGB, HSV, and grayscale representations.

    The input space is mandatory. Conversions involving grayscale or HSV are
    lossy; converting back cannot recover information discarded earlier.
    OpenCV uint8 HSV hue values use the range 0–179.
    """
    validate_image(image, source_space)
    if source_space == target_space:
        return image.copy()

    if source_space == "BGR":
        bgr = image
    elif source_space == "RGB":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif source_space == "HSV":
        bgr = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if target_space == "BGR":
        converted = bgr
    elif target_space == "RGB":
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif target_space == "HSV":
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    else:
        converted = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    validate_image(converted, target_space)
    return converted


def apply_blur(
    image: np.ndarray,
    method: BlurMethod = "none",
    kernel_size: int = 5,
    *,
    color_space: ColorSpace = "BGR",
) -> NDArray[np.uint8]:
    """Apply no blur, a Gaussian blur, or a median blur.

    Kernel size must be a positive odd integer and at least 3 for filtering.
    A new array is returned even for ``method='none'`` so callers can safely
    preserve their original input.
    """
    validate_image(image, color_space)
    if method not in ("none", "gaussian", "median"):
        raise ValueError(f"Unsupported blur method: {method}")
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("Blur kernel size must be an integer.")
    if method == "none":
        return image.copy()
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("Blur kernel size must be an odd integer of at least 3.")

    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)
    return cv2.medianBlur(image, kernel_size)


def preprocess_image(
    image_bgr: np.ndarray,
    *,
    target_space: ColorSpace = "BGR",
    blur_method: BlurMethod = "none",
    kernel_size: int = 5,
) -> NDArray[np.uint8]:
    """Filter an original BGR image, then convert to the requested space."""
    validate_image(image_bgr, "BGR")
    filtered = apply_blur(image_bgr, blur_method, kernel_size, color_space="BGR")
    return convert_color(filtered, "BGR", target_space)


def to_display_rgb(image: np.ndarray, color_space: ColorSpace) -> NDArray[np.uint8]:
    """Convert an image from its declared space to display-ready RGB."""
    return convert_color(image, color_space, "RGB")
