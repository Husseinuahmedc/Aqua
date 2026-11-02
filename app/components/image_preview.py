"""Aspect-preserving image preview with honest empty and loaded states."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.theme import COLORS
from core.preprocessing.transforms import to_display_rgb
from core.preprocessing.validation import ColorSpace, validate_image


class ImagePreviewWidget(QWidget):
    """A labeled preview that never changes the underlying analysis geometry."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.title_label = QLabel(title)
        self.title_label.setMinimumHeight(42)
        self.title_label.setContentsMargins(14, 0, 14, 0)
        self.title_label.setStyleSheet(
            f"background: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; font-weight: 650;"
        )
        layout.addWidget(self.title_label)

        self.image_label = QLabel()
        self.image_label.setAccessibleName(f"{title} preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(220, 180)
        self.image_label.setStyleSheet(
            f"background: {COLORS['side_surface']}; "
            f"border: 1px solid {COLORS['border']}; border-top: none;"
        )
        self.image_label.hide()
        layout.addWidget(self.image_label, 1)

        self.empty_panel = QWidget()
        empty_layout = QVBoxLayout(self.empty_panel)
        empty_layout.setContentsMargins(28, 28, 28, 28)
        empty_layout.addStretch(1)
        self.empty_heading = QLabel("No image loaded")
        self.empty_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_heading.setStyleSheet("font-size: 16px; font-weight: 650;")
        empty_layout.addWidget(self.empty_heading)
        self.empty_detail = QLabel("Open a local image to begin analysis.")
        self.empty_detail.setObjectName("mutedText")
        self.empty_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_detail.setWordWrap(True)
        empty_layout.addWidget(self.empty_detail)
        empty_layout.addStretch(1)
        self.empty_panel.setMinimumSize(220, 180)
        self.empty_panel.setStyleSheet(
            f"background: {COLORS['surface']}; "
            f"border: 1px solid {COLORS['border']}; border-top: none;"
        )
        layout.addWidget(self.empty_panel, 1)

    def set_empty(self, heading: str, detail: str) -> None:
        """Show a meaningful empty state rather than a fabricated preview."""
        self._pixmap = None
        self.empty_heading.setText(heading)
        self.empty_detail.setText(detail)
        self.image_label.clear()
        self.image_label.hide()
        self.empty_panel.show()

    def set_image(self, image: np.ndarray, color_space: ColorSpace) -> None:
        """Display an image with explicit channel ordering and aspect ratio."""
        validate_image(image, color_space)
        if color_space == "HSV":
            image = to_display_rgb(image, "HSV")
            color_space = "RGB"
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        if color_space == "BGR":
            image_format = QImage.Format.Format_BGR888
        elif color_space == "RGB":
            image_format = QImage.Format.Format_RGB888
        else:
            image_format = QImage.Format.Format_Grayscale8
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            image_format,
        ).copy()
        self.image_label.setAccessibleDescription(
            f"{color_space} image, {width} pixels wide by {height} pixels high."
        )
        self._pixmap = QPixmap.fromImage(qimage)
        self.empty_panel.hide()
        self.image_label.show()
        self._render_pixmap()

    def _render_pixmap(self) -> None:
        if self._pixmap is None or self.image_label.width() <= 0:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name.
        super().resizeEvent(event)
        self._render_pixmap()
