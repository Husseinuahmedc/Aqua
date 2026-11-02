"""RGB candidate overlays that preserve the source image."""

import cv2
import numpy as np

from core.preprocessing.validation import validate_image
from core.rescue.candidates import Candidate, format_category_label

_CATEGORY_COLORS_RGB = {
    "person-like": (212, 91, 46),
    "floating-object-like": (24, 117, 145),
    "unknown": (151, 103, 24),
}
_SELECTED_HIGHLIGHT_RGB = (19, 122, 112)
_SELECTED_OUTER_RGB = (255, 255, 255)


def draw_candidate_boxes_rgb(
    image_rgb: np.ndarray,
    candidates: tuple[Candidate, ...],
    *,
    selected_candidate_id: str | None = None,
) -> np.ndarray:
    """Draw labeled candidate boxes on a new RGB array without mutating input.

    When ``selected_candidate_id`` matches a candidate, that region is drawn
    with a high-contrast double border and its readable category label so the
    human reviewer can locate it immediately.
    """
    validate_image(image_rgb, "RGB")
    overlay = image_rgb.copy()
    height_px, width_px = overlay.shape[:2]
    selected_entry: tuple[int, Candidate] | None = None

    for index, candidate in enumerate(candidates, start=1):
        if candidate.candidate_id == selected_candidate_id:
            selected_entry = (index, candidate)
            continue
        x, y, width, height = candidate.bbox
        color = _CATEGORY_COLORS_RGB[candidate.category]
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            color,
            2,
        )
        cv2.putText(
            overlay,
            f"C{index}",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if selected_entry is not None:
        index, candidate = selected_entry
        x, y, width, height = candidate.bbox
        x1 = max(0, x - 2)
        y1 = max(0, y - 2)
        x2 = min(width_px - 1, x + width + 1)
        y2 = min(height_px - 1, y + height + 1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _SELECTED_OUTER_RGB, 4)
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width - 1, y + height - 1),
            _SELECTED_HIGHLIGHT_RGB,
            2,
        )
        label = f"C{index} {format_category_label(candidate.category)}"
        cv2.putText(
            overlay,
            label,
            (x, max(14, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            _SELECTED_HIGHLIGHT_RGB,
            2,
            cv2.LINE_AA,
        )
    return overlay
