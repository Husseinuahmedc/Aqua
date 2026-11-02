"""Consistent, visible controls for the HSV segmentation baseline."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)

from core.water.segmentation import SegmentationConfig


class SegmentationControls(QGroupBox):
    """Editable parameters shared by the image and video workspaces."""

    parameters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Potential-water segmentation", parent)
        layout = QFormLayout(self)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(7)

        self.hue_min, self.hue_max = self._range_row(179, 80, 130)
        self.saturation_min, self.saturation_max = self._range_row(255, 30, 255)
        self.value_min, self.value_max = self._range_row(255, 20, 255)
        layout.addRow("Hue", self._pair(self.hue_min, self.hue_max))
        layout.addRow(
            "Saturation",
            self._pair(self.saturation_min, self.saturation_max),
        )
        layout.addRow("Value", self._pair(self.value_min, self.value_max))

        self.blur_method = QComboBox()
        for label, method in (
            ("None", "none"),
            ("Gaussian", "gaussian"),
            ("Median", "median"),
        ):
            self.blur_method.addItem(label, method)
        self.blur_method.setCurrentIndex(1)
        layout.addRow("Noise reduction", self.blur_method)

        self.blur_kernel = self._choice((3, 5, 7, 9), 5)
        layout.addRow("Blur kernel · odd", self.blur_kernel)
        self.morphology_kernel = self._choice((1, 3, 5, 7, 9), 3)
        layout.addRow("Morphology kernel", self.morphology_kernel)

        self.morphology_iterations = QSpinBox()
        self.morphology_iterations.setRange(0, 5)
        self.morphology_iterations.setValue(1)
        self.morphology_iterations.setToolTip(
            "Opening removes small foreground noise; closing fills small gaps."
        )
        layout.addRow("Open / close iterations", self.morphology_iterations)

        self.minimum_area = QSpinBox()
        self.minimum_area.setRange(0, 100_000_000)
        self.minimum_area.setValue(40)
        self.minimum_area.setSuffix(" px")
        self.minimum_area.setToolTip(
            "Minimum connected-region pixel count; this is not real-world area."
        )
        layout.addRow("Minimum region area", self.minimum_area)

        for control in (
            self.hue_min,
            self.hue_max,
            self.saturation_min,
            self.saturation_max,
            self.value_min,
            self.value_max,
            self.blur_method,
            self.blur_kernel,
            self.morphology_kernel,
            self.morphology_iterations,
            self.minimum_area,
        ):
            signal = getattr(control, "valueChanged", None)
            if signal is None:
                signal = control.currentIndexChanged
            signal.connect(self.parameters_changed)

        note = QLabel(
            "Hue: 0–179 (wrap allowed); saturation/value: 0–255. Thresholds "
            "are starting points: shadows, reflections, roofs, and permanent "
            "water may be labeled; some floodwater may be missed."
        )
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        layout.addRow(note)

    @staticmethod
    def _choice(values: tuple[int, ...], selected: int) -> QComboBox:
        combo = QComboBox()
        for value in values:
            combo.addItem(str(value), value)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))
        return combo

    @staticmethod
    def _range_row(
        maximum: int, lower: int, upper: int
    ) -> tuple[QSpinBox, QSpinBox]:
        low = QSpinBox()
        high = QSpinBox()
        for widget in (low, high):
            widget.setRange(0, maximum)
            widget.setKeyboardTracking(False)
            widget.setFixedWidth(56)
        low.setValue(lower)
        high.setValue(upper)
        return low, high

    @staticmethod
    def _pair(low: QSpinBox, high: QSpinBox) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(low)
        separator = QLabel("to")
        separator.setFixedWidth(14)
        separator.setObjectName("mutedText")
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(separator)
        layout.addWidget(high)
        return row

    def config(self) -> SegmentationConfig:
        """Read the visible widget state into an explicit core configuration."""
        return SegmentationConfig(
            hue_min=self.hue_min.value(),
            hue_max=self.hue_max.value(),
            saturation_min=self.saturation_min.value(),
            saturation_max=self.saturation_max.value(),
            value_min=self.value_min.value(),
            value_max=self.value_max.value(),
            blur_method=self.blur_method.currentData(),
            blur_kernel_size=int(self.blur_kernel.currentData()),
            morphology_kernel_size=int(self.morphology_kernel.currentData()),
            morphology_iterations=self.morphology_iterations.value(),
            min_region_area=self.minimum_area.value(),
        )
