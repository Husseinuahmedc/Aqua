"""User-confirmed current-frame and sequential-metric video exports."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

from app.components.file_dialogs import choose_save_path
from app.exporting import export_records_csv, software_provenance
from core.preprocessing.image_io import save_image
from core.water.segmentation import SegmentationConfig


class VideoExportPanel(QGroupBox):
    """Export actions scoped to the current frame or analyzed video."""

    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Export")
        layout = QVBoxLayout(self)
        self.save_frame_button = QPushButton("Save current frame")
        self.save_mask_button = QPushButton("Save current mask")
        self.save_overlay_button = QPushButton("Save current overlay")
        self.export_metrics_button = QPushButton("Export frame metrics to CSV")
        for button in (
            self.save_frame_button,
            self.save_mask_button,
            self.save_overlay_button,
            self.export_metrics_button,
        ):
            button.setEnabled(False)
            layout.addWidget(button)
        self.save_frame_button.clicked.connect(self.save_current_frame)
        self.save_mask_button.clicked.connect(self.save_current_mask)
        self.save_overlay_button.clicked.connect(self.save_current_overlay)
        self.export_metrics_button.clicked.connect(self.export_metrics)
        self._video_path: Path | None = None
        self._frame_index: int | None = None
        self._frame_bgr: NDArray[np.uint8] | None = None
        self._mask: NDArray[np.uint8] | None = None
        self._overlay_bgr: NDArray[np.uint8] | None = None
        self._records: list[dict[str, object]] = []
        self._config: SegmentationConfig | None = None

    def set_snapshot(
        self,
        video_path: Path | None,
        frame_index: int | None,
        frame_bgr: NDArray[np.uint8] | None,
        mask: NDArray[np.uint8] | None,
        overlay_bgr: NDArray[np.uint8] | None,
        records: list[dict[str, object]],
        analysis_config: SegmentationConfig | None,
        *,
        frame_result_current: bool = True,
        analysis_results_current: bool = True,
    ) -> None:
        """Update available exports from the currently displayed video state."""
        self._video_path = video_path
        self._frame_index = frame_index
        self._frame_bgr = frame_bgr
        self._mask = mask
        self._overlay_bgr = overlay_bgr
        self._records = records
        self._analysis_config = analysis_config
        self.save_frame_button.setEnabled(
            video_path is not None and frame_bgr is not None
        )
        self.save_mask_button.setEnabled(
            video_path is not None and mask is not None and frame_result_current
        )
        self.save_overlay_button.setEnabled(
            video_path is not None and overlay_bgr is not None and frame_result_current
        )
        self.export_metrics_button.setEnabled(
            bool(records) and analysis_results_current
        )

    def _select_frame_path(self, suffix: str) -> tuple[Path, bool] | None:
        if self._video_path is None:
            return None
        frame_name = (
            f"_frame-{self._frame_index + 1:06d}"
            if self._frame_index is not None
            else ""
        )
        return choose_save_path(
            self,
            title=f"Save current {suffix.replace('-', ' ')}",
            suggested_path=self._video_path.parent
            / f"{self._video_path.stem}{frame_name}_{suffix}.png",
            file_filter="PNG image (*.png)",
            default_suffix=".png",
        )

    def _save_image(
        self,
        image: NDArray[np.uint8] | None,
        color_space: str,
        suffix: str,
    ) -> None:
        if image is None:
            return
        selected = self._select_frame_path(suffix)
        if selected is None:
            return
        path, overwrite = selected
        try:
            save_image(path, image, color_space=color_space, overwrite=overwrite)
            self.status_changed.emit(f"Saved {suffix.replace('-', ' ')} to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

    def save_current_frame(self) -> None:
        self._save_image(self._frame_bgr, "BGR", "frame")

    def save_current_mask(self) -> None:
        self._save_image(self._mask, "GRAY", "potential-water-mask")

    def save_current_overlay(self) -> None:
        self._save_image(self._overlay_bgr, "BGR", "overlay")

    def export_metrics(self) -> None:
        if self._video_path is None or not self._records:
            return
        selected = choose_save_path(
            self,
            title="Export video frame metrics",
            suggested_path=self._video_path.parent
            / f"{self._video_path.stem}_frame-analysis.csv",
            file_filter="CSV file (*.csv)",
            default_suffix=".csv",
        )
        if selected is None:
            return
        path, overwrite = selected
        metadata: dict[str, object] = {
            "source_video_path": str(self._video_path),
            **software_provenance(),
            "channel_order": "BGR",
            "method": "HSV threshold, optional blur, morphology, and connected regions",
            **(
                asdict(self._analysis_config)
                if self._analysis_config is not None
                else {}
            ),
        }
        rows = [{**metadata, **record} for record in self._records]
        try:
            export_records_csv(path, rows, overwrite=overwrite)
            self.status_changed.emit(f"Video metrics exported to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))
