"""Explicit normalization and loading of manually annotated binary masks."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.preprocessing.image_io import load_grayscale_image
from core.water.regions import validate_binary_mask


def normalize_reference_mask(mask: np.ndarray) -> NDArray[np.uint8]:
    """Normalize an explicitly binary uint8 reference mask to 0/255.

    The only accepted encodings are ``{0, 255}`` and ``{0, 1}``. Other gray
    levels are rejected instead of being silently thresholded.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Reference mask must be a NumPy array.")
    if mask.dtype != np.uint8:
        raise TypeError(f"Reference mask dtype must be uint8, got {mask.dtype}.")
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("Reference mask must have non-empty shape (height, width).")
    values = np.unique(mask)
    if np.isin(values, (0, 255)).all():
        return mask.copy()
    if np.isin(values, (0, 1)).all():
        return (mask * np.uint8(255)).astype(np.uint8)
    raise ValueError(
        "Reference mask must use only 0/255 or explicitly binary 0/1 values."
    )


def load_reference_mask(path: Path | str) -> NDArray[np.uint8]:
    """Load a grayscale manual mask, accepting only explicit binary encodings."""
    mask = load_grayscale_image(path)
    normalized = normalize_reference_mask(mask)
    validate_binary_mask(normalized)
    return normalized
