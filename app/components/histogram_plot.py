"""Matplotlib display for actual image-channel histogram counts."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from app.theme import COLORS
from core.preprocessing.histogram import HistogramData


class HistogramPlotWidget(QWidget):
    """Show a selected channel histogram or a clear no-data state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(6.2, 3.6), dpi=100, tight_layout=True)
        self.figure.set_facecolor(COLORS["surface"])
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.set_empty()

    def set_empty(self) -> None:
        self.axes.clear()
        self.axes.text(
            0.5,
            0.5,
            "Select a channel and compute its histogram.",
            ha="center",
            va="center",
            color=COLORS["muted"],
            wrap=True,
            transform=self.axes.transAxes,
        )
        self.axes.set_axis_off()
        self.canvas.draw_idle()

    def set_histogram(self, histogram: HistogramData) -> None:
        """Plot measured bin counts for the selected image channel."""
        self.axes.clear()
        self.axes.set_axis_on()
        self.axes.set_facecolor(COLORS["surface"])
        self.axes.step(
            histogram.bin_edges[:-1],
            histogram.counts,
            where="post",
            color=COLORS["accent"],
            linewidth=1.4,
        )
        self.axes.set_xlabel(
            f"{histogram.channel.title()} value",
            color=COLORS["muted"],
        )
        self.axes.set_ylabel("Pixel count", color=COLORS["muted"])
        self.axes.tick_params(colors=COLORS["muted"], labelsize=9)
        self.axes.grid(axis="y", color=COLORS["border"], linewidth=0.6)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["left"].set_color(COLORS["border"])
        self.axes.spines["bottom"].set_color(COLORS["border"])
        self.canvas.draw_idle()
