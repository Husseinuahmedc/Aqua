"""BGR previews for experimental geometry-only candidate regions."""

import cv2
import numpy as np
from numpy.typing import NDArray

from core.objects.candidates import CandidateRegion
from core.preprocessing.validation import validate_image


def draw_candidate_boxes(
    image_bgr: np.ndarray,
    candidates: tuple[CandidateRegion, ...],
    *,
    color_bgr: tuple[int, int, int] = (59, 123, 197),
) -> NDArray[np.uint8]:
    """Draw measured candidate bounding rectangles on a copy of a BGR image."""
    validate_image(image_bgr, "BGR")
    if len(color_bgr) != 3 or any(not 0 <= value <= 255 for value in color_bgr):
        raise ValueError(
            "Candidate outline color must contain three BGR values in 0–255."
        )
    height, width = image_bgr.shape[:2]
    output = image_bgr.copy()
    for candidate in candidates:
        if candidate.x < 0 or candidate.y < 0:
            raise ValueError("Candidate rectangle coordinates cannot be negative.")
        if (
            candidate.x + candidate.width > width
            or candidate.y + candidate.height > height
        ):
            raise ValueError("Candidate rectangle must remain inside the source image.")
        if candidate.width <= 0 or candidate.height <= 0:
            raise ValueError("Candidate rectangle dimensions must be positive.")
        cv2.rectangle(
            output,
            (candidate.x, candidate.y),
            (candidate.x + candidate.width - 1, candidate.y + candidate.height - 1),
            color_bgr,
            thickness=2,
        )
    return output
