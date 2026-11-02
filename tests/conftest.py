"""Headless Qt application fixture for PySide6 tests."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if os.environ.get("QT_STYLE_OVERRIDE", "").casefold() == "kvantum":
    os.environ.pop("QT_STYLE_OVERRIDE", None)

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def qt_application() -> QApplication:
    """Create one offscreen QApplication for worker and UI smoke tests."""
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application
