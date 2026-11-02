"""Supporting Water Analysis workspace combining still-image and video water segmentation."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.components.image_workspace import ImageWorkspace
from app.components.video_workspace import VideoWorkspace


class WaterWorkspace(QWidget):
    """Supporting water-segmentation tools with a single mode selector (no nested tabs)."""

    status_changed = Signal(str)
    prediction_ready = Signal(object, object, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image_workspace = ImageWorkspace()
        self.video_workspace = VideoWorkspace()
        self._build_ui()

        self.image_workspace.status_changed.connect(self.status_changed)
        self.video_workspace.status_changed.connect(self.status_changed)
        self.image_workspace.prediction_ready.connect(self.prediction_ready)
        self.video_workspace.prediction_ready.connect(self.prediction_ready)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(0)

        mode_bar = QHBoxLayout()
        mode_bar.setContentsMargins(16, 4, 16, 4)
        mode_bar.setSpacing(10)
        mode_label = QLabel("Water segmentation mode")
        mode_label.setObjectName("metricValue")
        self.mode_selector = QComboBox()
        self.mode_selector.addItem("Still Image Water Segmentation", "image")
        self.mode_selector.addItem("Video Frame Water Segmentation", "video")
        self.mode_selector.setToolTip(
            "Switch between still-image potential-water segmentation and sequential video water coverage."
        )
        supporting_note = QLabel(
            "Supporting analysis · Potential-water masks and pixel coverage (not floodwater confirmation)"
        )
        supporting_note.setObjectName("mutedText")
        mode_bar.addWidget(mode_label)
        mode_bar.addWidget(self.mode_selector)
        mode_bar.addSpacing(8)
        mode_bar.addWidget(supporting_note)
        mode_bar.addStretch(1)
        root.addLayout(mode_bar)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.image_workspace)
        self.mode_stack.addWidget(self.video_workspace)
        self.mode_selector.currentIndexChanged.connect(
            self.mode_stack.setCurrentIndex
        )
        root.addWidget(self.mode_stack, 1)

    def shutdown(self) -> None:
        """Stop active image and video water-segmentation workers."""
        self.image_workspace.shutdown()
        self.video_workspace.shutdown()
