"""Top-level PySide6 window and rescue-first workspace navigation."""

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.components.evaluation_workspace import EvaluationWorkspace
from app.components.rescue_workspace import RescueWorkspace
from app.components.water_workspace import WaterWorkspace


class AquaRescueWindow(QMainWindow):
    """Compose rescue-first analysis, supporting water segmentation, and evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AquaRescue")
        self.setMinimumSize(1000, 680)
        self.resize(1440, 900)

        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 0)
        layout.setSpacing(10)
        header = QWidget()
        header.setObjectName("applicationHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 11, 18, 11)
        header_title = QLabel("AquaRescue")
        header_title.setObjectName("applicationTitle")
        header_context = QLabel("Search & Rescue Analysis · Traditional DIP")
        header_context.setObjectName("headerSecondary")
        header_layout.addWidget(header_title)
        header_layout.addSpacing(18)
        header_layout.addWidget(header_context)
        header_layout.addStretch(1)
        review_label = QLabel("Requires Human Review")
        review_label.setObjectName("headerSecondary")
        header_layout.addWidget(review_label)
        layout.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.rescue_workspace = RescueWorkspace()
        self.water_workspace = WaterWorkspace()
        self.image_workspace = self.water_workspace.image_workspace
        self.video_workspace = self.water_workspace.video_workspace
        self.evaluation_workspace = EvaluationWorkspace()
        self.tabs.addTab(self.rescue_workspace, "Rescue Analysis")
        self.tabs.addTab(self.water_workspace, "Water Analysis")
        self.tabs.addTab(self.evaluation_workspace, "Evaluation")
        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        self.water_workspace.prediction_ready.connect(
            self.evaluation_workspace.set_prediction
        )
        for workspace in (
            self.rescue_workspace,
            self.water_workspace,
            self.evaluation_workspace,
        ):
            workspace.status_changed.connect(self.statusBar().showMessage)
        self.tabs.currentChanged.connect(self._workspace_changed)

        open_image = QAction("Open image", self)
        open_image.setShortcut(QKeySequence.StandardKey.Open)
        open_image.setStatusTip(
            "Open a local UAV image in the active analysis workspace."
        )
        open_image.triggered.connect(self._open_in_active_workspace)
        self.addAction(open_image)

        self.statusBar().showMessage(
            "Search & Rescue analysis aid · Potential Person and Floating Object candidates require human review."
        )

    def _open_in_active_workspace(self) -> None:
        if self.tabs.currentWidget() is self.water_workspace:
            if self.water_workspace.mode_stack.currentWidget() is self.video_workspace:
                self.video_workspace.open_video_dialog()
            else:
                self.image_workspace.open_image_dialog()
        else:
            self.rescue_workspace.open_image_dialog()

    def _workspace_changed(self, index: int) -> None:
        names = ("Rescue Analysis", "Water Analysis", "Evaluation")
        if 0 <= index < len(names):
            self.statusBar().showMessage(f"{names[index]} workspace")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt virtual method name.
        self.rescue_workspace.shutdown()
        self.water_workspace.shutdown()
        self.evaluation_workspace.shutdown()
        event.accept()
