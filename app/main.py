"""AquaRescue desktop application entry point."""

import logging
import os
import sys

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QStyleFactory

from app.theme import apply_application_theme
from app.window import AquaRescueWindow


def main() -> int:
    """Create and run the AquaRescue Qt application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    style_override = os.environ.get("QT_STYLE_OVERRIDE")
    available_styles = {name.casefold() for name in QStyleFactory.keys()}
    if style_override and style_override.casefold() not in available_styles:
        logging.warning(
            "Qt style override %r is unavailable; AquaRescue will use Fusion.",
            style_override,
        )
        os.environ.pop("QT_STYLE_OVERRIDE", None)
    existing_app = QApplication.instance()
    application = (
        existing_app
        if isinstance(existing_app, QApplication)
        else QApplication(sys.argv)
    )
    application.setApplicationName("AquaRescue")
    application.setOrganizationName("AquaRescue")
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont)
    font.setPointSize(10)
    application.setFont(font)
    apply_application_theme(application)
    window = AquaRescueWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
