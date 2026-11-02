"""Unicode-safe image loading and explicit image saving helpers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import validate_image


@dataclass(frozen=True)
class ImageInfo:
    """Basic information about a decoded image."""

    width: int
    height: int
    channels: int
    dtype: str


def _read_encoded(path: Path) -> NDArray[np.uint8]:
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise OSError(f"Could not read image file '{path}': {exc}") from exc
    if not encoded:
        raise ValueError(f"Image file is empty: {path}")
    return np.frombuffer(encoded, dtype=np.uint8)


def load_image(path: Path | str) -> NDArray[np.uint8]:
    """Load a local image as a validated uint8 BGR array.

    Reading the encoded bytes before decoding supports paths containing
    non-ASCII characters on platforms where OpenCV path handling is limited.
    The original file is never modified. Alpha is discarded by the explicit
    ``IMREAD_COLOR`` decode; grayscale files become three-channel BGR.
    """
    image_path = Path(path)
    encoded = _read_encoded(image_path)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError(f"Could not decode image '{image_path}': {exc}") from exc
    if image is None:
        raise ValueError(
            f"Unsupported or invalid image file '{image_path}'. "
            "Try a readable PNG, JPEG, BMP, or TIFF image."
        )
    validate_image(image, "BGR")
    return image


def load_grayscale_image(path: Path | str) -> NDArray[np.uint8]:
    """Load a local image as validated uint8 grayscale data."""
    image_path = Path(path)
    encoded = _read_encoded(image_path)
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    except cv2.error as exc:
        raise ValueError(f"Could not decode image '{image_path}': {exc}") from exc
    if image is None:
        raise ValueError(f"Unsupported or invalid image file '{image_path}'.")
    validate_image(image, "GRAY")
    return image


def image_info(image: np.ndarray) -> ImageInfo:
    """Return dimensions and storage dtype for a validated image array."""
    if image.ndim == 2:
        validate_image(image, "GRAY")
        channels = 1
    elif image.ndim == 3 and image.shape[2] == 3:
        validate_image(image, "BGR")
        channels = 3
    else:
        raise ValueError("Image must be grayscale or three-channel color data.")
    height, width = image.shape[:2]
    return ImageInfo(
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype),
    )


def save_image(
    path: Path | str,
    image: np.ndarray,
    *,
    color_space: Literal["BGR", "RGB", "GRAY"] = "BGR",
    overwrite: bool = False,
) -> Path:
    """Save a uint8 image, refusing to overwrite unless explicitly allowed.

    Color arrays must declare BGR, RGB, or grayscale ordering. OpenCV encoding
    is selected from the destination suffix; PNG is used when no suffix exists.
    """
    output_path = Path(path)
    if color_space not in ("BGR", "RGB", "GRAY"):
        raise ValueError(f"Unsupported image color space: {color_space}")
    validate_image(image, color_space)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    image_to_encode = image
    if color_space == "RGB":
        image_to_encode = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        success, encoded = cv2.imencode(output_path.suffix.lower(), image_to_encode)
    except cv2.error as exc:
        raise ValueError(
            f"Could not encode image as '{output_path.suffix}': {exc}"
        ) from exc
    if not success:
        raise ValueError(f"Unsupported image extension: {output_path.suffix}")

    try:
        if overwrite:
            output_path.write_bytes(encoded.tobytes())
        else:
            with output_path.open("xb") as output_file:
                output_file.write(encoded.tobytes())
    except OSError as exc:
        raise OSError(f"Could not save image to '{output_path}': {exc}") from exc
    return output_path
