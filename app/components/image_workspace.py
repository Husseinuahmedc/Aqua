"""Image analysis workspace for loading, previewing, and processing images."""

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.components.histogram_plot import HistogramPlotWidget
from app.components.image_controls import ImageControlsPanel
from app.components.image_export import ImageExportPanel
from app.components.image_preview import ImagePreviewWidget
from app.workers import (
    ImageTaskWorker,
    HistogramOutput,
    HistogramTask,
    LoadImageTask,
    LoadedImageOutput,
    PreprocessOutput,
    PreprocessTask,
    SegmentTask,
    SegmentationOutput,
)
from core.preprocessing.histogram import HistogramData
from core.preprocessing.image_io import ImageInfo
from core.water.analysis import WaterAnalysis


class ImageWorkspace(QWidget):
    """Primary still-image workflow, with all analysis kept in core workers."""

    status_changed = Signal(str)
    prediction_ready = Signal(object, object, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_path: Path | None = None
        self.source_bgr: NDArray[np.uint8] | None = None
        self.source_info: ImageInfo | None = None
        self.preprocessed: PreprocessOutput | None = None
        self.segmentation: SegmentationOutput | None = None
        self.histogram_data: HistogramData | None = None
        self.segmentation_current = False
        self.preprocessing_current = False
        self.histogram_current = False
        self._histogram_settings: tuple[str, str] | None = None
        self._candidate_extraction_used = False
        self.candidate_options_current = True
        self._worker: ImageTaskWorker | None = None
        self._busy = False
        self._build_ui()
        self._reset_views()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        toolbar = QHBoxLayout()
        self.open_button = QPushButton("Open image")
        self.open_button.setObjectName("primaryAction")
        self.open_button.setToolTip("Load a supported local image file.")
        self.preprocess_button = QPushButton("Preview preprocessing")
        self.segment_button = QPushButton("Run segmentation")
        self.segment_button.setObjectName("primaryAction")
        self.source_label = QLabel("No source selected")
        self.source_label.setObjectName("mutedText")
        self.source_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        toolbar.addWidget(self.open_button)
        toolbar.addWidget(self.preprocess_button)
        toolbar.addWidget(self.segment_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.source_label)
        root.addLayout(toolbar)

        self.status_label = QLabel("Open a local image to begin.")
        self.status_label.setObjectName("mutedText")
        self.status_label.setMinimumHeight(22)
        root.addWidget(self.status_label)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        evidence_panel = QWidget()
        evidence_layout = QVBoxLayout(evidence_panel)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        evidence_layout.setSpacing(10)
        view_splitter = QSplitter(Qt.Orientation.Horizontal)
        view_splitter.setChildrenCollapsible(False)
        self.original_preview = ImagePreviewWidget("Original image")
        self.processed_preview = ImagePreviewWidget("Preprocessed preview")
        self.mask_preview = ImagePreviewWidget("Potential-water mask · uint8 0/255")
        self.overlay_preview = ImagePreviewWidget("Overlay · potential-water regions")
        self.candidate_preview = ImagePreviewWidget("Experimental candidate regions")
        self.histogram_plot = HistogramPlotWidget()

        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(6)
        self.result_view_selector = QComboBox()
        for label in (
            "Potential-water mask",
            "Overlay",
            "Preprocessing preview",
            "Histogram",
            "Experimental candidate regions",
        ):
            self.result_view_selector.addItem(label)
        self.result_stack = QStackedWidget()
        for preview in (
            self.mask_preview,
            self.overlay_preview,
            self.processed_preview,
            self.histogram_plot,
            self.candidate_preview,
        ):
            self.result_stack.addWidget(preview)
        self.result_view_selector.currentIndexChanged.connect(
            self.result_stack.setCurrentIndex
        )
        self.result_state_label = QLabel("No analysis result is current.")
        self.result_state_label.setObjectName("resultState")
        self.result_state_label.setWordWrap(True)
        result_layout.addWidget(self.result_view_selector)
        result_layout.addWidget(self.result_state_label)
        result_layout.addWidget(self.result_stack, 1)
        self.result_stack.currentChanged.connect(self._refresh_result_state)
        view_splitter.addWidget(self.original_preview)
        view_splitter.addWidget(result_panel)
        self.original_preview.setMinimumWidth(260)
        result_panel.setMinimumWidth(260)
        view_splitter.setStretchFactor(0, 1)
        view_splitter.setStretchFactor(1, 1)
        view_splitter.setSizes([500, 500])
        evidence_layout.addWidget(view_splitter, 1)

        self.registration_label = QLabel(
            "Registration · source, mask, and reference geometry are kept explicit."
        )
        self.registration_label.setObjectName("registrationStrip")
        self.registration_label.setWordWrap(True)
        evidence_layout.addWidget(self.registration_label)

        metrics_group = QGroupBox("Water analysis · pixel measurements")
        metrics_layout = QFormLayout(metrics_group)
        self.total_pixels_value = self._metric_value()
        self.water_pixels_value = self._metric_value()
        self.coverage_value = self._metric_value()
        self.region_count_value = self._metric_value()
        self.region_areas_value = self._metric_value()
        metrics_layout.addRow("Total image pixels", self.total_pixels_value)
        metrics_layout.addRow("Potential-water pixels", self.water_pixels_value)
        metrics_layout.addRow("Potential-water coverage", self.coverage_value)
        metrics_layout.addRow("Connected regions", self.region_count_value)
        metrics_layout.addRow("Region areas", self.region_areas_value)
        evidence_layout.addWidget(metrics_group)
        self.main_splitter.addWidget(evidence_panel)

        self.controls_panel = ImageControlsPanel()
        self.segmentation_controls = self.controls_panel.segmentation_controls
        self.extract_candidates_checkbox = (
            self.controls_panel.extract_candidates_checkbox
        )
        self.candidate_details = self.controls_panel.candidate_details
        self.preprocess_space = self.controls_panel.preprocess_space
        self.preprocess_blur = self.controls_panel.preprocess_blur
        self.preprocess_kernel = self.controls_panel.preprocess_kernel
        self.histogram_channel = self.controls_panel.histogram_channel
        self.histogram_button = self.controls_panel.histogram_button
        self.export_panel: ImageExportPanel = self.controls_panel.export_panel
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumWidth(250)
        self.controls_scroll.setMaximumWidth(420)
        self.controls_scroll.setWidget(self.controls_panel)
        self.main_splitter.addWidget(self.controls_scroll)
        self.main_splitter.setSizes([820, 450])

        self.open_button.clicked.connect(self.open_image_dialog)
        self.preprocess_button.clicked.connect(self.preview_preprocessing)
        self.segment_button.clicked.connect(self.run_segmentation)
        self.controls_panel.histogram_requested.connect(self.compute_histogram)
        self.controls_panel.segmentation_parameters_changed.connect(
            self._mark_segmentation_stale
        )
        self.controls_panel.preprocessing_parameters_changed.connect(
            self._mark_preprocessing_stale
        )
        self.controls_panel.histogram_parameters_changed.connect(
            self._mark_histogram_stale
        )
        self.controls_panel.candidate_option_changed.connect(
            self._mark_candidate_stale
        )
        self.export_panel.status_changed.connect(self._set_status)
        self.export_panel.error_occurred.connect(self._on_worker_error)

    @staticmethod
    def _metric_value() -> QLabel:
        value = QLabel("—")
        value.setObjectName("metricValue")
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return value

    def _reset_views(self) -> None:
        self.original_preview.set_empty(
            "No image loaded",
            "Open a local RGB image to inspect it and begin analysis.",
        )
        self.processed_preview.set_empty(
            "No preprocessing preview",
            "Choose a color space or filter, then preview it.",
        )
        self.mask_preview.set_empty(
            "No mask generated", "Run segmentation to view the potential-water mask."
        )
        self.overlay_preview.set_empty(
            "No overlay available",
            "The overlay is created from an actual segmentation result.",
        )
        self.histogram_plot.set_empty()
        self.candidate_preview.set_empty(
            "Candidate regions not extracted",
            "Enable the experimental option to propose areas for human inspection.",
        )
        self._refresh_result_state()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open aerial image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if path:
            self.open_image(Path(path))

    def open_image(self, path: Path) -> None:
        self._start_task(
            LoadImageTask(path),
            self._on_image_loaded,
            f"Loading {path.name}…",
        )

    def _start_task(
        self,
        task: LoadImageTask | HistogramTask | PreprocessTask | SegmentTask,
        result_handler: Callable[[object], None],
        busy_message: str,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        worker = ImageTaskWorker(task, self)
        self._worker = worker
        worker.result_ready.connect(result_handler)
        worker.error_occurred.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        self._set_busy(True)
        self._set_status(busy_message)
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.open_button.setEnabled(not busy)
        self.preprocess_button.setEnabled(not busy and self.source_bgr is not None)
        self.segment_button.setEnabled(not busy and self.source_bgr is not None)
        self.controls_panel.set_busy(busy, self.source_bgr is not None)

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            worker.deleteLater()
        self._set_busy(False)

    def _on_worker_error(self, message: str) -> None:
        self._set_status(f"Error: {message}")
        QMessageBox.warning(self, "Image operation failed", message)

    def _on_image_loaded(self, output: LoadedImageOutput) -> None:
        self.source_path = output.path
        self.source_bgr = output.image_bgr
        self.source_info = output.info
        self.preprocessed = None
        self.segmentation = None
        self.histogram_data = None
        self.segmentation_current = False
        self.preprocessing_current = False
        self.histogram_current = False
        self._histogram_settings = None
        self._candidate_extraction_used = False
        self.candidate_options_current = True
        self.export_panel.set_results(
            self.source_path,
            self.source_info,
            None,
            None,
            preprocessing_current=False,
            segmentation_current=False,
        )
        self.original_preview.set_image(self.source_bgr, "BGR")
        self.processed_preview.set_empty(
            "No preprocessing preview",
            "Choose an output space or filter and preview it.",
        )
        self.mask_preview.set_empty(
            "No mask generated", "Run segmentation to view the potential-water mask."
        )
        self.overlay_preview.set_empty(
            "No overlay available", "The overlay is derived from the mask result."
        )
        self.candidate_preview.set_empty(
            "Candidate regions not extracted",
            "Enable the experimental option to propose geometry-only regions.",
        )
        self.candidate_details.setText("Candidate regions have not been extracted.")
        self.histogram_plot.set_empty()
        self.result_view_selector.setCurrentIndex(0)
        self.source_label.setText(
            f"{self.source_path.name} · {output.info.width:,} × "
            f"{output.info.height:,} px · BGR"
        )
        self.registration_label.setText(
            f"Registration · {output.info.width:,} × {output.info.height:,} px · "
            "source preserved · OpenCV BGR"
        )
        self._set_metrics(None)
        self.prediction_ready.emit(None, None, "", None)
        self._refresh_result_state()
        self._set_status("Image loaded. Adjust parameters or preview preprocessing.")
        self._set_busy(False)

    def preview_preprocessing(self) -> None:
        if self.source_bgr is None:
            return
        self._start_task(
            PreprocessTask(
                image_bgr=self.source_bgr.copy(),
                target_space=self.preprocess_space.currentData(),
                blur_method=self.preprocess_blur.currentData(),
                kernel_size=int(self.preprocess_kernel.currentData()),
            ),
            self._on_preprocessed,
            "Creating preprocessing preview…",
        )

    def compute_histogram(self) -> None:
        if self.source_bgr is None:
            return
        self._start_task(
            HistogramTask(
                image_bgr=self.source_bgr.copy(),
                color_space=self.preprocess_space.currentData(),
                channel=self.histogram_channel.currentData(),
            ),
            self._on_histogram_ready,
            "Calculating channel histogram…",
        )

    def _on_preprocessed(self, output: PreprocessOutput) -> None:
        self.preprocessed = output
        self.preprocessing_current = True
        self.export_panel.set_results(
            self.source_path,
            self.source_info,
            self.preprocessed,
            self.segmentation,
            preprocessing_current=self.preprocessing_current,
            segmentation_current=self.segmentation_current,
        )
        self.processed_preview.set_image(output.display_rgb, "RGB")
        self.result_view_selector.setCurrentIndex(2)
        self._refresh_result_state()
        self._set_status(f"Preprocessing preview ready · {output.color_space} display.")

    def _on_histogram_ready(self, output: HistogramOutput) -> None:
        self.histogram_data = output.histogram
        self._histogram_settings = (output.color_space, output.channel)
        self.histogram_current = True
        self.histogram_plot.set_histogram(output.histogram)
        self.result_view_selector.setCurrentIndex(3)
        self._refresh_result_state()
        self._set_status(
            f"Histogram ready · {output.histogram.channel} channel · "
            f"{int(output.histogram.counts.sum()):,} pixels."
        )

    def run_segmentation(self) -> None:
        if self.source_bgr is None:
            return
        try:
            config = self.segmentation_controls.config()
        except (TypeError, ValueError) as exc:
            self._on_worker_error(str(exc))
            return
        self._start_task(
            SegmentTask(
                self.source_bgr.copy(),
                config,
                extract_candidates=self.extract_candidates_checkbox.isChecked(),
            ),
            self._on_segmented,
            "Processing image with the configured HSV threshold…",
        )

    def _on_segmented(self, output: SegmentationOutput) -> None:
        self.segmentation = output
        self.segmentation_current = True
        self._candidate_extraction_used = output.candidate_overlay_bgr is not None
        self.candidate_options_current = True
        self.export_panel.set_results(
            self.source_path,
            self.source_info,
            self.preprocessed,
            self.segmentation,
            preprocessing_current=self.preprocessing_current,
            segmentation_current=self.segmentation_current,
        )
        self.mask_preview.set_image(output.result.mask, "GRAY")
        self.overlay_preview.set_image(output.overlay_bgr, "BGR")
        self.result_view_selector.setCurrentIndex(0)
        if output.candidate_overlay_bgr is not None:
            self.candidate_preview.set_image(output.candidate_overlay_bgr, "BGR")
            self.candidate_details.setText(
                f"Experimental geometry proposals: {len(output.candidates)}. "
                "They are not people or confirmed objects."
            )
            self.result_view_selector.setCurrentIndex(4)
        else:
            self.candidate_preview.set_empty(
                "Candidate regions not extracted",
                "Enable the experimental option to propose geometry-only regions.",
            )
            self.candidate_details.setText("Candidate regions were not requested.")
        self._set_metrics(output.water_analysis)
        self.registration_label.setText(
            f"Registration · {output.water_analysis.total_pixels:,} source pixels · "
            f"{len(output.result.regions)} connected regions · "
            "source geometry unchanged"
        )
        self._emit_current_prediction()
        self._refresh_result_state()
        self._set_status(
            "Segmentation complete. Labels indicate potential water, "
            "not confirmed floodwater."
        )

    def _set_metrics(self, analysis: WaterAnalysis | None) -> None:
        if analysis is None:
            for value in (
                self.total_pixels_value,
                self.water_pixels_value,
                self.coverage_value,
                self.region_count_value,
                self.region_areas_value,
            ):
                value.setText("—")
            return
        self.total_pixels_value.setText(f"{analysis.total_pixels:,} px")
        self.water_pixels_value.setText(f"{analysis.water_pixels:,} px")
        self.coverage_value.setText(f"{analysis.coverage_percent:.2f}% of image pixels")
        self.region_count_value.setText(str(analysis.region_count))
        areas = analysis.region_areas_pixels
        rendered_areas = ", ".join(f"{area:,}" for area in areas[:8])
        if len(areas) > 8:
            rendered_areas += f", … ({len(areas)} total)"
        self.region_areas_value.setText(f"{rendered_areas or 'None'} px")

    def _mark_segmentation_stale(self, *_args) -> None:
        if self.segmentation is None:
            return
        try:
            current_config = self.segmentation_controls.config()
        except (TypeError, ValueError):
            current_config = None
        self.segmentation_current = current_config == self.segmentation.result.config
        self.candidate_options_current = (
            self.extract_candidates_checkbox.isChecked()
            == self._candidate_extraction_used
        )
        self._update_export_availability()
        self._emit_current_prediction()
        self._refresh_result_state()

    def _mark_preprocessing_stale(self, *_args) -> None:
        if self.preprocessed is None:
            return
        self.preprocessing_current = (
            self.preprocessed.color_space == self.preprocess_space.currentData()
            and self.preprocessed.blur_method == self.preprocess_blur.currentData()
            and self.preprocessed.kernel_size
            == int(self.preprocess_kernel.currentData())
        )
        self._update_export_availability()
        self._refresh_result_state()
        if not self.preprocessing_current:
            self._set_status(
                "Preprocessing settings changed. Preview again to refresh the result."
            )

    def _mark_histogram_stale(self, *_args) -> None:
        if self._histogram_settings is None:
            return
        self.histogram_current = self._histogram_settings == (
            self.preprocess_space.currentData(),
            self.histogram_channel.currentData(),
        )
        self._refresh_result_state()

    def _mark_candidate_stale(self, *_args) -> None:
        if self.segmentation is None:
            return
        self.candidate_options_current = (
            self.extract_candidates_checkbox.isChecked()
            == self._candidate_extraction_used
        )
        if not self.candidate_options_current:
            self.candidate_details.setText(
                "Candidate setting changed; these proposals are stale. "
                "Rerun segmentation to refresh the experimental view."
            )
        else:
            self.candidate_details.setText(
                f"Experimental geometry proposals: {len(self.segmentation.candidates)}. "
                "They are not people or confirmed objects."
                if self._candidate_extraction_used
                else "Candidate regions were not requested."
            )
        self._refresh_result_state()

    def _update_export_availability(self) -> None:
        self.export_panel.set_results(
            self.source_path,
            self.source_info,
            self.preprocessed,
            self.segmentation,
            preprocessing_current=self.preprocessing_current,
            segmentation_current=self.segmentation_current,
        )

    def _emit_current_prediction(self) -> None:
        if self.segmentation is None or not self.segmentation_current:
            self.prediction_ready.emit(None, None, "", None)
            return
        self.prediction_ready.emit(
            self.segmentation.original_bgr.copy(),
            self.segmentation.result.mask.copy(),
            self.source_path.name if self.source_path else "Current image",
            {
                "method": "HSV threshold, optional blur, morphology, "
                "and connected regions",
                "channel_order": "BGR",
                "parameters": asdict(self.segmentation.result.config),
            },
        )

    def _refresh_result_state(self, *_args) -> None:
        current_view = self.result_stack.currentWidget()
        message = "No analysis result is current."
        stale = False
        if current_view in (self.mask_preview, self.overlay_preview):
            if self.segmentation is None:
                message = "No potential-water mask has been generated."
            elif not self.segmentation_current:
                message = (
                    "Stale mask · parameters changed. Rerun segmentation before "
                    "exporting or evaluation."
                )
                stale = True
            else:
                message = "Mask uses the displayed segmentation parameters."
        elif current_view is self.processed_preview:
            if self.preprocessed is None:
                message = "No preprocessing preview has been generated."
            elif not self.preprocessing_current:
                message = "Stale preprocessing preview · refresh to match controls."
                stale = True
            else:
                message = "Preprocessing preview matches the displayed controls."
        elif current_view is self.histogram_plot:
            if self.histogram_data is None:
                message = "No channel histogram has been calculated."
            elif not self.histogram_current:
                message = "Stale histogram · recompute for the selected channel."
                stale = True
            else:
                message = (
                    f"Histogram shows the measured {self.histogram_data.channel} "
                    "channel counts."
                )
        elif current_view is self.candidate_preview:
            if self.segmentation is None or not self._candidate_extraction_used:
                message = "Candidate proposals were not requested; no detections are implied."
            elif not self.segmentation_current or not self.candidate_options_current:
                message = "Stale experimental proposals · rerun with current settings."
                stale = True
            else:
                message = "Experimental geometry proposals for human inspection only."
        self.result_state_label.setText(message)
        self.result_state_label.setProperty("stale", stale)
        self.result_state_label.style().unpolish(self.result_state_label)
        self.result_state_label.style().polish(self.result_state_label)

    def shutdown(self) -> None:
        """Wait for an active image task before its owning widget is destroyed."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
