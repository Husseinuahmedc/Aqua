"""Smoke tests for the supported Python runtime and approved dependencies."""

from importlib import import_module
from importlib.metadata import version
import sys

import pytest


APPROVED_DISTRIBUTIONS = (
    ("PySide6", "PySide6"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
)


def test_python_version_is_supported() -> None:
    """The project environment requires Python 3.12 or newer."""
    assert sys.version_info >= (3, 12)


@pytest.mark.parametrize(
    ("module_name", "distribution_name"), APPROVED_DISTRIBUTIONS
)
def test_approved_dependency_imports(
    module_name: str, distribution_name: str
) -> None:
    """Each approved direct dependency can be imported and has metadata."""
    module = import_module(module_name)

    assert module is not None
    assert version(distribution_name)
