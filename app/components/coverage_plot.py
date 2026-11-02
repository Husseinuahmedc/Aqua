"""Matplotlib coverage history for analyzed video frame records."""

from math import ceil
from typing import Mapping, Sequence

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.theme import COLORS


class CoveragePlotWidget(QWidget):
    """Plot actual per-frame potential-water coverage or a truthful empty state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(4.8, 2.35), dpi=90, tight_layout=True)
        self.figure.set_facecolor(COLORS["surface"])
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.set_empty()

    def set_empty(self) -> None:
        """Show no measurement until frame analysis has actually run."""
        self.axes.clear()
        self.axes.set_facecolor(COLORS["surface"])
        self.axes.text(
            0.5,
            0.5,
            "Coverage history appears after video analysis.",
            ha="center",
            va="center",
            color=COLORS["muted"],
            wrap=True,
            transform=self.axes.transAxes,
        )
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def set_records(self, records: Sequence[Mapping[str, object]]) -> None:
        """Draw measured coverage values; downsample only for display density."""
        if not records:
            self.set_empty()
            return
        frame_indices = np.asarray(
            [int(record["frame_index"]) for record in records], dtype=np.int64
        )
        coverage = np.asarray(
            [float(record["coverage_percent"]) for record in records],
            dtype=np.float64,
        )
        stride = max(1, ceil(len(records) / 3000))
        self.axes.clear()
        self.axes.set_axis_on()
        self.axes.set_facecolor(COLORS["surface"])
        self.axes.plot(
            frame_indices[::stride],
            coverage[::stride],
            color=COLORS["accent"],
            linewidth=1.5,
        )
        self.axes.set_ylim(0, 100)
        self.axes.set_xlabel("Frame index", color=COLORS["muted"])
        self.axes.set_ylabel("Potential-water pixels (%)", color=COLORS["muted"])
        self.axes.tick_params(colors=COLORS["muted"], labelsize=8)
        self.axes.grid(axis="y", color=COLORS["border"], linewidth=0.6)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["left"].set_color(COLORS["border"])
        self.axes.spines["bottom"].set_color(COLORS["border"])
        self.canvas.draw_idle()
