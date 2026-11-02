"""Parameter and task controls shared inside the image workspace."""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.components.segmentation_controls import SegmentationControls
from app.components.image_export import ImageExportPanel


class ImageControlsPanel(QWidget):
    """Compose image settings without owning processing or result state."""

    histogram_requested = Signal()
    segmentation_parameters_changed = Signal()
    preprocessing_parameters_changed = Signal()
    histogram_parameters_changed = Signal()
    candidate_option_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.segmentation_controls = SegmentationControls()
        layout.addWidget(self.segmentation_controls)

        self.experimental_group = self._build_experimental_group()
        layout.addWidget(self.experimental_group)
        self.preprocessing_group = self._build_preprocessing_group()
        layout.addWidget(self.preprocessing_group)

        self.export_panel = ImageExportPanel()
        layout.addWidget(self.export_panel)
        layout.addStretch(1)

        self.segmentation_controls.parameters_changed.connect(
            self.segmentation_parameters_changed
        )
        self.extract_candidates_checkbox.toggled.connect(self.candidate_option_changed)
        self.preprocess_space.currentIndexChanged.connect(
            self._update_histogram_channels
        )
        self.preprocess_space.currentIndexChanged.connect(
            self.preprocessing_parameters_changed
        )
        self.preprocess_space.currentIndexChanged.connect(
            self.histogram_parameters_changed
        )
        for control in (self.preprocess_blur, self.preprocess_kernel):
            control.currentIndexChanged.connect(self.preprocessing_parameters_changed)
        self.histogram_channel.currentIndexChanged.connect(
            self.histogram_parameters_changed
        )
        self.histogram_button.clicked.connect(self.histogram_requested)

    def _build_experimental_group(self) -> QGroupBox:
        group = QGroupBox("Experimental · human inspection only")
        layout = QVBoxLayout(group)
        self.extract_candidates_checkbox = QCheckBox(
            "Propose geometry-only candidate regions"
        )
        self.extract_candidates_checkbox.setToolTip(
            "Uses connected regions and bounding boxes; it does not identify "
            "people or objects."
        )
        self.candidate_details = QLabel("Candidate regions have not been extracted.")
        self.candidate_details.setObjectName("mutedText")
        self.candidate_details.setWordWrap(True)
        layout.addWidget(self.extract_candidates_checkbox)
        layout.addWidget(self.candidate_details)
        return group

    def _build_preprocessing_group(self) -> QGroupBox:
        group = QGroupBox("Preprocessing and histogram")
        layout = QFormLayout(group)
        self.preprocess_space = QComboBox()
        for label, value in (
            ("BGR", "BGR"),
            ("RGB", "RGB"),
            ("HSV", "HSV"),
            ("Grayscale", "GRAY"),
        ):
            self.preprocess_space.addItem(label, value)
        self.preprocess_blur = QComboBox()
        for label, value in (
            ("None", "none"),
            ("Gaussian", "gaussian"),
            ("Median", "median"),
        ):
            self.preprocess_blur.addItem(label, value)
        self.preprocess_kernel = QComboBox()
        for value in (3, 5, 7, 9):
            self.preprocess_kernel.addItem(str(value), value)
        self.preprocess_kernel.setCurrentIndex(1)
        self.histogram_channel = QComboBox()
        self.histogram_button = QPushButton("Plot selected channel histogram")
        self.histogram_button.setEnabled(False)
        self.histogram_button.setToolTip(
            "Calculate pixel counts for the selected channel without assigning "
            "semantic classes."
        )
        layout.addRow("Output color space", self.preprocess_space)
        layout.addRow("Filter", self.preprocess_blur)
        layout.addRow("Kernel · odd", self.preprocess_kernel)
        layout.addRow("Histogram channel", self.histogram_channel)
        layout.addRow(self.histogram_button)
        self._update_histogram_channels()
        return group

    def _update_histogram_channels(self, *_args) -> None:
        channels = {
            "BGR": ("blue", "green", "red"),
            "RGB": ("red", "green", "blue"),
            "HSV": ("hue", "saturation", "value"),
            "GRAY": ("gray",),
        }[self.preprocess_space.currentData()]
        self.histogram_channel.clear()
        for channel in channels:
            self.histogram_channel.addItem(channel.title(), channel)

    def set_busy(self, busy: bool, has_source: bool) -> None:
        """Disable settings while a snapshot of those settings is processing."""
        self.segmentation_controls.setEnabled(not busy)
        self.experimental_group.setEnabled(not busy)
        self.preprocessing_group.setEnabled(not busy)
        self.histogram_button.setEnabled(not busy and has_source)
