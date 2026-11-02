"""Full-resolution RGB image representations for rescue candidate cues."""

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import validate_image


@dataclass(frozen=True)
class ImageRepresentations:
    """Same-geometry RGB, OpenCV HSV/Lab, and grayscale views."""

    rgb: NDArray[np.uint8]
    hsv: NDArray[np.uint8]
    lab: NDArray[np.uint8]
    gray: NDArray[np.uint8]


def prepare_rgb_image(image_rgb: np.ndarray) -> ImageRepresentations:
    """Validate RGB uint8 input and create analysis views without resizing.

    HSV separates hue/saturation/value for interpreting color contrast; Lab
    supports a perceptual-ish color distance across its three channels; gray
    supports local intensity, texture, adaptive threshold, and edge operators.
    The original RGB array is returned by reference and never modified.
    """
    validate_image(image_rgb, "RGB")
    return ImageRepresentations(
        rgb=image_rgb,
        hsv=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV),
        lab=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB),
        gray=cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY),
    )
