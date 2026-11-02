"""User-confirmed image and metric exports for the image workspace."""

from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

from app.components.file_dialogs import choose_save_path
from app.exporting import export_records_csv, software_provenance
from app.workers import PreprocessOutput, SegmentationOutput
from core.preprocessing.image_io import ImageInfo, save_image


class ImageExportPanel(QGroupBox):
    """Export only results that were actually produced for the current image."""

    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Export", parent)
        layout = QVBoxLayout(self)
        self.save_processed_button = QPushButton("Save processed preview")
        self.save_mask_button = QPushButton("Save binary mask")
        self.save_overlay_button = QPushButton("Save overlay image")
        self.export_metrics_button = QPushButton("Export water metrics to CSV")
        for button in (
            self.save_processed_button,
            self.save_mask_button,
            self.save_overlay_button,
            self.export_metrics_button,
        ):
            button.setEnabled(False)
            layout.addWidget(button)
        self.save_processed_button.clicked.connect(self.save_processed_image)
        self.save_mask_button.clicked.connect(self.save_mask)
        self.save_overlay_button.clicked.connect(self.save_overlay)
        self.export_metrics_button.clicked.connect(self.export_metrics)
        self._source_path: Path | None = None
        self._source_info: ImageInfo | None = None
        self._preprocessed: PreprocessOutput | None = None
        self._segmentation: SegmentationOutput | None = None

    def set_results(
        self,
        source_path: Path | None,
        source_info: ImageInfo | None,
        preprocessed: PreprocessOutput | None,
        segmentation: SegmentationOutput | None,
        *,
        preprocessing_current: bool = True,
        segmentation_current: bool = True,
    ) -> None:
        """Update available export actions after source or analysis changes."""
        self._source_path = source_path
        self._source_info = source_info
        self._preprocessed = preprocessed
        self._segmentation = segmentation
        self.save_processed_button.setEnabled(
            preprocessed is not None and preprocessing_current
        )
        self.save_mask_button.setEnabled(
            segmentation is not None and segmentation_current
        )
        self.save_overlay_button.setEnabled(
            segmentation is not None and segmentation_current
        )
        self.export_metrics_button.setEnabled(
            segmentation is not None and segmentation_current
        )

    def _select_path(self, suffix: str, file_filter: str) -> tuple[Path, bool] | None:
        if self._source_path is None:
            return None
        return choose_save_path(
            self,
            title=f"Save {suffix.replace('_', ' ')}",
            suggested_path=self._source_path.parent
            / f"{self._source_path.stem}_{suffix}",
            file_filter=file_filter,
            default_suffix=Path(suffix).suffix or ".png",
        )

    def save_processed_image(self) -> None:
        if self._preprocessed is None:
            return
        selected = self._select_path(
            "processed.png", "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
        )
        if selected is None:
            return
        path, overwrite = selected
        try:
            save_image(
                path,
                self._preprocessed.display_rgb,
                color_space="RGB",
                overwrite=overwrite,
            )
            self.status_changed.emit(f"Processed preview saved to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

    def save_mask(self) -> None:
        if self._segmentation is None:
            return
        selected = self._select_path("potential-water-mask.png", "PNG image (*.png)")
        if selected is None:
            return
        path, overwrite = selected
        try:
            save_image(
                path,
                self._segmentation.result.mask,
                color_space="GRAY",
                overwrite=overwrite,
            )
            self.status_changed.emit(f"Binary mask saved to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

    def save_overlay(self) -> None:
        if self._segmentation is None:
            return
        selected = self._select_path(
            "overlay.png", "PNG image (*.png);;JPEG image (*.jpg *.jpeg)"
        )
        if selected is None:
            return
        path, overwrite = selected
        try:
            save_image(
                path,
                self._segmentation.overlay_bgr,
                color_space="BGR",
                overwrite=overwrite,
            )
            self.status_changed.emit(f"Overlay saved to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))

    def export_metrics(self) -> None:
        if self._segmentation is None or self._source_info is None:
            return
        selected = self._select_path("water-analysis.csv", "CSV file (*.csv)")
        if selected is None:
            return
        path, overwrite = selected
        analysis = self._segmentation.water_analysis
        config = self._segmentation.result.config
        record = {
            "source_path": str(self._source_path),
            "width_pixels": self._source_info.width,
            "height_pixels": self._source_info.height,
            **software_provenance(),
            "channel_order": "BGR",
            "method": "HSV threshold, optional blur, morphology, and connected regions",
            **asdict(config),
            "total_pixels": analysis.total_pixels,
            "potential_water_pixels": analysis.water_pixels,
            "potential_water_coverage_percent": analysis.coverage_percent,
            "region_count": analysis.region_count,
            "region_areas_pixels": ";".join(map(str, analysis.region_areas_pixels)),
        }
        try:
            export_records_csv(path, [record], overwrite=overwrite)
            self.status_changed.emit(f"Image metrics exported to {path}")
        except (OSError, ValueError) as exc:
            self.error_occurred.emit(str(exc))
