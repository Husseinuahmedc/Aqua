"""Manual-reference mask selection and metric evaluation workspace."""

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.components.file_dialogs import choose_save_path
from app.components.image_preview import ImagePreviewWidget
from app.exporting import export_records_csv, software_provenance
from app.workers import (
    EvaluateTask,
    ImageTaskWorker,
    LoadReferenceMaskTask,
    LoadedMaskOutput,
)
from core.evaluation.metrics import EvaluationMetrics


class EvaluationWorkspace(QWidget):
    """Evaluate a current prediction against a manually selected mask."""

    status_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.predicted_mask = None
        self.prediction_source = ""
        self.prediction_metadata: dict[str, object] = {}
        self.reference_mask = None
        self.reference_path: Path | None = None
        self.metrics: EvaluationMetrics | None = None
        self._worker: ImageTaskWorker | None = None
        self._input_revision = 0
        self._active_evaluation_revision: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        heading = QLabel("Compare a predicted mask with a manually selected reference.")
        heading.setObjectName("mutedText")
        root.addWidget(heading)
        self.status_label = QLabel(
            "Run image or video segmentation, then select a reference mask."
        )
        self.status_label.setObjectName("mutedText")
        root.addWidget(self.status_label)

        previews = QSplitter(Qt.Orientation.Horizontal)
        previews.setChildrenCollapsible(False)
        self.prediction_preview = ImagePreviewWidget("Predicted potential-water mask")
        self.reference_preview = ImagePreviewWidget("Manual reference mask")
        self.prediction_preview.set_empty(
            "No prediction available",
            "Run segmentation in the Image or Video workspace.",
        )
        self.reference_preview.set_empty(
            "No reference mask selected",
            "Select a local binary water mask to evaluate.",
        )
        previews.addWidget(self.prediction_preview)
        previews.addWidget(self.reference_preview)
        previews.setStretchFactor(0, 1)
        previews.setStretchFactor(1, 1)
        root.addWidget(previews, 1)

        provenance = QGroupBox("Evaluation inputs")
        provenance_layout = QFormLayout(provenance)
        self.prediction_label = QLabel("—")
        self.reference_label = QLabel("—")
        self.dimension_label = QLabel("—")
        for label in (
            self.prediction_label,
            self.reference_label,
            self.dimension_label,
        ):
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setWordWrap(True)
        provenance_layout.addRow("Prediction", self.prediction_label)
        provenance_layout.addRow("Reference", self.reference_label)
        provenance_layout.addRow("Dimensions", self.dimension_label)
        root.addWidget(provenance)

        controls = QHBoxLayout()
        self.select_reference_button = QPushButton("Select reference mask")
        self.select_reference_button.setObjectName("primaryAction")
        self.evaluate_button = QPushButton("Evaluate masks")
        self.export_button = QPushButton("Export evaluation metrics to CSV")
        self.evaluate_button.setEnabled(False)
        self.export_button.setEnabled(False)
        controls.addWidget(self.select_reference_button)
        controls.addWidget(self.evaluate_button)
        controls.addStretch(1)
        controls.addWidget(self.export_button)
        root.addLayout(controls)

        metrics_group = QGroupBox("Evaluation metrics")
        metrics_layout = QFormLayout(metrics_group)
        self.iou_value = self._metric_label("Intersection over Union (IoU)")
        self.dice_value = self._metric_label("Dice coefficient")
        self.precision_value = self._metric_label("Precision")
        self.recall_value = self._metric_label("Recall")
        self.counts_value = self._metric_label("Confusion counts · TP / FP / FN / TN")
        metrics_layout.addRow("IoU · overlap / union", self.iou_value)
        metrics_layout.addRow("Dice · overlap F1", self.dice_value)
        metrics_layout.addRow(
            "Precision · positive prediction quality", self.precision_value
        )
        metrics_layout.addRow(
            "Recall · reference positives recovered", self.recall_value
        )
        metrics_layout.addRow("Pixel counts", self.counts_value)
        root.addWidget(metrics_group)

        self.select_reference_button.clicked.connect(self.select_reference_dialog)
        self.evaluate_button.clicked.connect(self.evaluate)
        self.export_button.clicked.connect(self.export_metrics)

    @staticmethod
    def _metric_label(description: str) -> QLabel:
        label = QLabel("—")
        label.setObjectName("metricValue")
        label.setToolTip(description)
        return label

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)

    def set_prediction(
        self,
        image_bgr: NDArray[np.uint8] | None,
        mask: NDArray[np.uint8] | None,
        source_label: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Accept a new prediction from an image or video workspace."""
        self._input_revision += 1
        self.metrics = None
        self.export_button.setEnabled(False)
        if image_bgr is None or mask is None:
            self.predicted_mask = None
            self.prediction_source = ""
            self.prediction_metadata = {}
            self.prediction_preview.set_empty(
                "No prediction available",
                "Run segmentation in the Image or Video workspace.",
            )
            self.prediction_label.setText("—")
            self._clear_metrics()
            self._update_evaluation_availability()
            return
        if image_bgr.shape[:2] != mask.shape:
            self._set_status(
                "Prediction dimensions do not match its source image/frame."
            )
            self.predicted_mask = None
            self.prediction_source = ""
            self.prediction_metadata = {}
            self.prediction_preview.set_empty(
                "Invalid prediction",
                "The mask geometry does not match its source frame.",
            )
            self._update_evaluation_availability()
            return
        self.predicted_mask = mask.copy()
        self.prediction_source = source_label
        self.prediction_metadata = metadata.copy() if metadata else {}
        self.prediction_preview.set_image(self.predicted_mask, "GRAY")
        self.prediction_label.setText(source_label)
        self._update_evaluation_availability()
        self._clear_metrics()

    def select_reference_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select manually annotated binary water mask",
            str(Path.home()),
            "Mask images (*.png *.tif *.tiff *.bmp);;All files (*)",
        )
        if path:
            self.load_reference_mask(Path(path))

    def load_reference_mask(self, path: Path) -> None:
        self._start_task(
            LoadReferenceMaskTask(path),
            self._on_reference_loaded,
            f"Loading reference mask {path.name}…",
        )

    def _start_task(
        self,
        task: LoadReferenceMaskTask | EvaluateTask,
        result_handler: Callable[[object], None],
        message: str,
    ) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        worker = ImageTaskWorker(task, self)
        self._worker = worker
        worker.result_ready.connect(result_handler)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._on_worker_finished)
        self.select_reference_button.setEnabled(False)
        self.evaluate_button.setEnabled(False)
        self._set_status(message)
        worker.start()

    def _on_reference_loaded(self, output: LoadedMaskOutput) -> None:
        self._input_revision += 1
        self.reference_path = output.path
        self.reference_mask = output.mask
        self.reference_preview.set_image(output.mask, "GRAY")
        self.reference_label.setText(str(output.path))
        self._clear_metrics()
        self._update_evaluation_availability()
        self._set_status(
            "Reference mask loaded. Only binary 0/255 or explicitly binary "
            "0/1 values are accepted."
        )

    def _on_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            worker.deleteLater()
        self.select_reference_button.setEnabled(True)
        self._update_evaluation_availability()

    def _on_error(self, message: str) -> None:
        self._set_status(f"Evaluation input error: {message}")
        QMessageBox.warning(self, "Evaluation operation failed", message)

    def _update_evaluation_availability(self) -> None:
        if self.predicted_mask is None or self.reference_mask is None:
            self.evaluate_button.setEnabled(False)
            self.dimension_label.setText("Load a prediction and a reference mask.")
            return
        if self.predicted_mask.shape != self.reference_mask.shape:
            self.dimension_label.setText(
                f"Dimension mismatch · prediction {self.predicted_mask.shape[1]} × "
                f"{self.predicted_mask.shape[0]} px; reference "
                f"{self.reference_mask.shape[1]} × {self.reference_mask.shape[0]} px. "
                "No resize is applied."
            )
            self.evaluate_button.setEnabled(False)
            return
        height, width = self.predicted_mask.shape
        self.dimension_label.setText(f"{width:,} × {height:,} px · exact match")
        self.evaluate_button.setEnabled(self._worker is None)

    def evaluate(self) -> None:
        if self.predicted_mask is None or self.reference_mask is None:
            return
        if self.predicted_mask.shape != self.reference_mask.shape:
            self._update_evaluation_availability()
            return
        self._active_evaluation_revision = self._input_revision
        self._start_task(
            EvaluateTask(self.predicted_mask.copy(), self.reference_mask.copy()),
            self._on_evaluated,
            "Calculating mask agreement metrics…",
        )

    def _on_evaluated(self, metrics: EvaluationMetrics) -> None:
        if self._active_evaluation_revision != self._input_revision:
            self._set_status(
                "Inputs changed while evaluation was running. Evaluate the "
                "current masks again."
            )
            return
        self.metrics = metrics
        self.iou_value.setText(f"{metrics.intersection_over_union:.4f}")
        self.dice_value.setText(f"{metrics.dice_coefficient:.4f}")
        self.precision_value.setText(self._format_optional_metric(metrics.precision))
        self.recall_value.setText(self._format_optional_metric(metrics.recall))
        self.counts_value.setText(
            f"{metrics.true_positive:,} / {metrics.false_positive:,} / "
            f"{metrics.false_negative:,} / {metrics.true_negative:,} · "
            f"{metrics.total_pixels:,} pixels"
        )
        self.export_button.setEnabled(True)
        self._set_status(
            "Evaluation complete. Metrics describe agreement with the "
            "selected mask only."
        )

    @staticmethod
    def _format_optional_metric(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A · denominator is zero"

    def _clear_metrics(self) -> None:
        self.metrics = None
        for value in (
            self.iou_value,
            self.dice_value,
            self.precision_value,
            self.recall_value,
            self.counts_value,
        ):
            value.setText("—")
        self.export_button.setEnabled(False)

    def export_metrics(self) -> None:
        if self.metrics is None or self.reference_path is None:
            return
        suggested = self.reference_path.with_name(
            f"{self.reference_path.stem}_evaluation.csv"
        )
        selected = choose_save_path(
            self,
            title="Export evaluation metrics",
            suggested_path=suggested,
            file_filter="CSV file (*.csv)",
            default_suffix=".csv",
        )
        if selected is None:
            return
        path, overwrite = selected
        record = {
            "prediction_source": self.prediction_source,
            "reference_path": str(self.reference_path),
            **software_provenance(),
            "prediction_method": self.prediction_metadata.get("method", "Unknown"),
            "channel_order": self.prediction_metadata.get("channel_order", "Unknown"),
            "prediction_parameters": str(
                self.prediction_metadata.get("parameters", {})
            ),
            **self.metrics.to_record(),
        }
        try:
            export_records_csv(path, [record], overwrite=overwrite)
            self._set_status(f"Evaluation metrics exported to {path}")
        except (OSError, ValueError) as exc:
            self._on_error(str(exc))

    def shutdown(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
