"""Input validation for image arrays used by the core processing API."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


ColorSpace = Literal["BGR", "RGB", "HSV", "GRAY"]
UInt8Image = NDArray[np.uint8]


def validate_image(image: np.ndarray, color_space: ColorSpace) -> None:
    """Validate an 8-bit image array and its explicitly declared color space.

    Color images must have shape ``(height, width, 3)``. Grayscale images
    must have shape ``(height, width)``. No implicit dtype conversion occurs.
    """
    if color_space not in ("BGR", "RGB", "HSV", "GRAY"):
        raise ValueError(f"Unsupported image color space: {color_space}")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a NumPy array.")
    if image.dtype != np.uint8:
        raise TypeError(f"Image dtype must be uint8, got {image.dtype}.")
    if image.size == 0:
        raise ValueError("Image must not be empty.")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("Image must have positive height and width.")

    if color_space == "GRAY":
        if image.ndim != 2:
            raise ValueError("Grayscale images must have shape (height, width).")
        return

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"{color_space} images must have shape (height, width, 3)."
        )
    if color_space == "HSV" and int(image[:, :, 0].max()) > 179:
        raise ValueError("OpenCV uint8 HSV hue values must be in the range 0–179.")


def validate_color_image(
    image: np.ndarray, channel_order: Literal["BGR", "RGB"]
) -> UInt8Image:
    """Validate and return an ordinary uint8 BGR or RGB image array."""
    validate_image(image, channel_order)
    return image
