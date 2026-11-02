"""Tests for validated image input and independent preprocessing operations."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.preprocessing.image_io import load_grayscale_image, load_image, save_image
from core.preprocessing.transforms import (
    apply_blur,
    convert_color,
    preprocess_image,
    to_display_rgb,
)
from core.preprocessing.validation import validate_image


def test_load_and_save_image_with_unicode_path(tmp_path: Path) -> None:
    image = np.zeros((7, 9, 3), dtype=np.uint8)
    image[2:5, 3:7] = (40, 120, 230)
    path = tmp_path / "field notes Δ" / "aerial image.png"
    path.parent.mkdir()

    saved_path = save_image(path, image)

    assert saved_path == path
    np.testing.assert_array_equal(load_image(path), image)


def test_load_image_rejects_missing_empty_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "missing.png")

    empty_path = tmp_path / "empty.png"
    empty_path.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_image(empty_path)

    invalid_path = tmp_path / "not-an-image.png"
    invalid_path.write_text("not image data", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid image"):
        load_image(invalid_path)


def test_load_grayscale_image_is_decoded_as_uint8_gray(tmp_path: Path) -> None:
    image = np.arange(35, dtype=np.uint8).reshape(5, 7)
    path = tmp_path / "gray.png"
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.write_bytes(encoded.tobytes())

    decoded = load_image(path)
    grayscale = load_grayscale_image(path)

    assert decoded.shape == (5, 7, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(grayscale, image)


def test_color_conversions_use_explicit_channel_order() -> None:
    bgr_red = np.array([[[0, 0, 255]]], dtype=np.uint8)

    rgb = convert_color(bgr_red, "BGR", "RGB")
    hsv = convert_color(bgr_red, "BGR", "HSV")
    gray = convert_color(bgr_red, "BGR", "GRAY")

    np.testing.assert_array_equal(rgb, np.array([[[255, 0, 0]]], dtype=np.uint8))
    np.testing.assert_array_equal(hsv, np.array([[[0, 255, 255]]], dtype=np.uint8))
    assert gray.shape == (1, 1)
    assert 75 <= int(gray[0, 0]) <= 77
    assert convert_color(rgb, "RGB", "BGR").tolist() == bgr_red.tolist()


def test_conversion_to_display_rgb_supports_hsv_and_gray() -> None:
    bgr = np.array([[[80, 160, 220]]], dtype=np.uint8)
    hsv = convert_color(bgr, "BGR", "HSV")
    gray = convert_color(bgr, "BGR", "GRAY")

    np.testing.assert_allclose(
        to_display_rgb(hsv, "HSV"),
        convert_color(bgr, "BGR", "RGB"),
        atol=1,
    )
    assert to_display_rgb(gray, "GRAY").shape == (1, 1, 3)


def test_blur_operations_return_new_arrays_and_preserve_dimensions() -> None:
    image = np.zeros((21, 23, 3), dtype=np.uint8)
    image[:, 11:] = 255

    unchanged = apply_blur(image, "none")
    blurred = apply_blur(image, "gaussian", 5)
    median = apply_blur(image, "median", 3)

    assert unchanged is not image
    np.testing.assert_array_equal(unchanged, image)
    assert blurred.shape == image.shape
    assert median.shape == image.shape
    assert image[:, 11:].min() == 255


def test_preprocess_image_converts_and_does_not_mutate_source() -> None:
    image = np.full((10, 12, 3), (10, 20, 30), dtype=np.uint8)
    original = image.copy()

    output = preprocess_image(
        image,
        target_space="GRAY",
        blur_method="gaussian",
        kernel_size=3,
    )

    assert output.shape == (10, 12)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_blur_rejects_invalid_filter_kernel(kernel_size: int) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        apply_blur(image, "median", kernel_size)


def test_validation_rejects_wrong_shape_dtype_and_hsv_range() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_image(np.zeros((4, 4, 4), dtype=np.uint8), "BGR")
    with pytest.raises(TypeError, match="uint8"):
        validate_image(np.zeros((4, 4, 3), dtype=np.float32), "RGB")
    invalid_hsv = np.zeros((2, 2, 3), dtype=np.uint8)
    invalid_hsv[:, :, 0] = 200
    with pytest.raises(ValueError, match="hue"):
        validate_image(invalid_hsv, "HSV")


def test_save_image_refuses_overwrite_without_explicit_permission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "image.png"
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    save_image(path, image)

    with pytest.raises(FileExistsError):
        save_image(path, image)

    save_image(path, np.full_like(image, 25), overwrite=True)
    assert int(load_image(path).max()) == 25
