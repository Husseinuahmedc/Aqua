"""Small UI adapter for independently selecting candidate proposal cues."""

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.rescue.pipeline import RescueConfig


class RescueControls(QGroupBox):
    """Visible method switches and broad component-size controls."""

    parameters_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Traditional-DIP proposal methods", parent)
        layout = QVBoxLayout(self)
        self.method_checkboxes: dict[str, QCheckBox] = {}
        for method, description in (
            ("color", "Local Lab color contrast"),
            ("intensity", "Local grayscale intensity contrast"),
            ("texture", "Local texture differences"),
            ("edge", "Canny edge-region proposals"),
            ("adaptive", "Adaptive threshold · both polarities"),
        ):
            checkbox = QCheckBox(description)
            checkbox.setChecked(True)
            checkbox.toggled.connect(self.parameters_changed)
            self.method_checkboxes[method] = checkbox
            layout.addWidget(checkbox)
        fields = QFormLayout()
        self.min_area = QSpinBox()
        self.min_area.setRange(1, 100_000)
        self.min_area.setValue(9)
        self.min_area.setSuffix(" px")
        self.max_area_fraction = QDoubleSpinBox()
        self.max_area_fraction.setRange(0.001, 1.0)
        self.max_area_fraction.setDecimals(3)
        self.max_area_fraction.setSingleStep(0.01)
        self.max_area_fraction.setValue(0.2)
        self.min_area.valueChanged.connect(self.parameters_changed)
        self.max_area_fraction.valueChanged.connect(self.parameters_changed)
        fields.addRow("Minimum region area", self.min_area)
        fields.addRow("Maximum frame-area fraction", self.max_area_fraction)
        layout.addLayout(fields)

        self.reset_button = QPushButton("Restore default parameters")
        self.reset_button.clicked.connect(self.reset_defaults)
        layout.addWidget(self.reset_button)

    def reset_defaults(self) -> None:
        """Restore the default RescueConfig method switches and area bounds."""
        defaults = RescueConfig()
        for name, checkbox in self.method_checkboxes.items():
            with QSignalBlocker(checkbox):
                checkbox.setChecked(name in defaults.methods)
        with QSignalBlocker(self.min_area):
            self.min_area.setValue(defaults.min_area_pixels)
        with QSignalBlocker(self.max_area_fraction):
            self.max_area_fraction.setValue(defaults.max_area_fraction)
        self.parameters_changed.emit()

    def config(self) -> RescueConfig:
        """Build validated explicit core settings from current controls."""
        methods = tuple(
            name for name, checkbox in self.method_checkboxes.items() if checkbox.isChecked()
        )
        return RescueConfig(
            methods=methods,
            min_area_pixels=self.min_area.value(),
            max_area_fraction=self.max_area_fraction.value(),
        )

    def set_enabled(self, enabled: bool) -> None:
        """Disable control changes while a snapshot is processing."""
        self.setEnabled(enabled)
