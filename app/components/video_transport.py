"""Familiar playback, frame-navigation, and sequential-progress controls."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class VideoTransportPanel(QWidget):
    """Video controls that emit intent without owning decoder state."""

    open_requested = Signal()
    play_pause_requested = Signal()
    previous_requested = Signal()
    next_requested = Signal()
    process_frame_requested = Signal()
    analyze_requested = Signal()
    cancel_requested = Signal()
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        actions = QHBoxLayout()
        self.open_button = QPushButton("Open video")
        self.open_button.setObjectName("primaryAction")
        self.play_button = QPushButton("Play")
        self.previous_button = QPushButton("Previous frame")
        self.next_button = QPushButton("Next frame")
        self.process_current_button = QPushButton("Process current frame")
        self.analyze_button = QPushButton("Analyze all frames")
        self.cancel_button = QPushButton("Cancel analysis")
        self.cancel_button.setEnabled(False)
        for button in (
            self.play_button,
            self.previous_button,
            self.next_button,
            self.process_current_button,
            self.analyze_button,
        ):
            button.setEnabled(False)
        for button in (
            self.open_button,
            self.play_button,
            self.previous_button,
            self.next_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.process_current_button)
        actions.addWidget(self.analyze_button)
        actions.addWidget(self.cancel_button)
        layout.addLayout(actions)

        navigation = QHBoxLayout()
        self.frame_label = QLabel("Frame —")
        self.frame_label.setObjectName("metricValue")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setAccessibleName("Video frame")
        self.frame_slider.setAccessibleDescription(
            "Choose a frame index after the video has loaded."
        )
        self.time_label = QLabel("No video loaded")
        self.time_label.setObjectName("mutedText")
        navigation.addWidget(self.frame_label)
        navigation.addWidget(self.frame_slider, 1)
        navigation.addWidget(self.time_label)
        layout.addLayout(navigation)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("No sequential analysis running")
        self.progress_bar.setAccessibleName("Sequential video analysis progress")
        layout.addWidget(self.progress_bar)

        self.open_button.clicked.connect(self.open_requested)
        self.play_button.clicked.connect(self.play_pause_requested)
        self.previous_button.clicked.connect(self.previous_requested)
        self.next_button.clicked.connect(self.next_requested)
        self.process_current_button.clicked.connect(self.process_frame_requested)
        self.analyze_button.clicked.connect(self.analyze_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.frame_slider.sliderReleased.connect(
            lambda: self.seek_requested.emit(self.frame_slider.value())
        )
