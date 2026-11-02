"""Display overlays for reviewing potential-water masks."""

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import validate_image
from core.water.regions import validate_binary_mask


def create_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    color_bgr: tuple[int, int, int] = (32, 86, 230),
    alpha: float = 0.42,
) -> NDArray[np.uint8]:
    """Blend a colored mask indication over a BGR image without mutation."""
    validate_image(image_bgr, "BGR")
    validate_binary_mask(mask)
    if image_bgr.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match for an overlay.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Overlay alpha must be between 0 and 1.")
    if len(color_bgr) != 3 or any(not 0 <= value <= 255 for value in color_bgr):
        raise ValueError("Overlay color must contain three BGR values in 0–255.")

    overlay = image_bgr.copy()
    selected = mask == 255
    if not np.any(selected):
        return overlay
    color_layer = np.zeros_like(image_bgr)
    color_layer[:] = color_bgr
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_layer, alpha, 0)
    overlay[selected] = blended[selected]
    return overlay
