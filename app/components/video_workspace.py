"""Video playback, frame processing, and sequential water-mask analysis workspace."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.components.coverage_plot import CoveragePlotWidget
from app.components.image_preview import ImagePreviewWidget
from app.components.segmentation_controls import SegmentationControls
from app.components.video_export import VideoExportPanel
from app.components.video_transport import VideoTransportPanel
from app.workers import (
    ImageTaskWorker,
    SegmentTask,
    SegmentationOutput,
    VideoAnalysisWorker,
    VideoPlaybackWorker,
    VideoProgress,
)
from core.video.reader import VideoFrame, VideoInfo
from core.water.analysis import WaterAnalysis
from core.water.segmentation import SegmentationConfig


class VideoWorkspace(QWidget):
    """Supporting video water-segmentation UI; decoding runs in worker threads."""

    status_changed = Signal(str)
    prediction_ready = Signal(object, object, str, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.video_path: Path | None = None
        self.video_info: VideoInfo | None = None
        self.current_frame_bgr: NDArray[np.uint8] | None = None
        self.current_frame_index: int | None = None
        self.current_mask: NDArray[np.uint8] | None = None
        self.current_overlay: NDArray[np.uint8] | None = None
        self.current_segmentation_config: SegmentationConfig | None = None
        self.analysis_config: SegmentationConfig | None = None
        self.frame_result_current = False
        self.analysis_results_current = False
        self.analysis_records: list[dict[str, object]] = []
        self._frame_revision = 0
        self._processing_frame_revision: int | None = None
        self._playing = False
        self._playback_worker: VideoPlaybackWorker | None = None
        self._frame_worker: ImageTaskWorker | None = None
        self._analysis_worker: VideoAnalysisWorker | None = None
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_analysis_progress)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self.transport = VideoTransportPanel()
        self.open_button = self.transport.open_button
        self.play_button = self.transport.play_button
        self.previous_button = self.transport.previous_button
        self.next_button = self.transport.next_button
        self.process_current_button = self.transport.process_current_button
        self.analyze_button = self.transport.analyze_button
        self.cancel_button = self.transport.cancel_button
        self.frame_label = self.transport.frame_label
        self.frame_slider = self.transport.frame_slider
        self.time_label = self.transport.time_label
        self.progress_bar = self.transport.progress_bar
        self.transport.open_requested.connect(self.open_video_dialog)
        self.transport.play_pause_requested.connect(self.toggle_playback)
        self.transport.previous_requested.connect(lambda: self._seek_relative(-1))
        self.transport.next_requested.connect(lambda: self._seek_relative(1))
        self.transport.process_frame_requested.connect(self.process_current_frame)
        self.transport.analyze_requested.connect(self.analyze_video)
        self.transport.cancel_requested.connect(self.cancel_analysis)
        self.transport.seek_requested.connect(self._seek_frame)
        root.addWidget(self.transport)

        self.status_label = QLabel(
            "Open a local video. Playback begins paused on its first frame."
        )
        self.status_label.setObjectName("mutedText")
        root.addWidget(self.status_label)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        root.addWidget(self.main_splitter, 1)

        evidence = QWidget()
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        evidence_layout.setSpacing(10)
        views = QSplitter(Qt.Orientation.Horizontal)
        views.setChildrenCollapsible(False)
        self.frame_preview = ImagePreviewWidget("Original video frame")
        self.mask_preview = ImagePreviewWidget("Potential-water mask · uint8 0/255")
        views.addWidget(self.frame_preview)
        views.addWidget(self.mask_preview)
        views.setStretchFactor(0, 1)
        views.setStretchFactor(1, 1)
        evidence_layout.addWidget(views, 1)

        self.registration_label = QLabel(
            "Frame registration · Video is processed sequentially; "
            "no geospatial area is inferred."
        )
        self.registration_label.setObjectName("registrationStrip")
        self.registration_label.setWordWrap(True)
        evidence_layout.addWidget(self.registration_label)
        self.result_state_label = QLabel("No frame mask or video analysis is current.")
        self.result_state_label.setObjectName("resultState")
        self.result_state_label.setWordWrap(True)
        evidence_layout.addWidget(self.result_state_label)

        self.main_splitter.addWidget(evidence)

        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setContentsMargins(8, 0, 0, 0)
        control_layout.setSpacing(12)
        self.segmentation_controls = SegmentationControls()
        self.segmentation_controls.parameters_changed.connect(
            self._mark_video_results_stale
        )
        control_layout.addWidget(self.segmentation_controls)

        metrics_group = QGroupBox("Current frame · pixel measurements")
        metrics_layout = QFormLayout(metrics_group)
        self.total_pixels_value = self._metric_label()
        self.water_pixels_value = self._metric_label()
        self.coverage_value = self._metric_label()
        self.region_count_value = self._metric_label()
        self.change_value = self._metric_label()
        metrics_layout.addRow("Total image pixels", self.total_pixels_value)
        metrics_layout.addRow("Potential-water pixels", self.water_pixels_value)
        metrics_layout.addRow("Potential-water coverage", self.coverage_value)
        metrics_layout.addRow("Connected regions", self.region_count_value)
        metrics_layout.addRow("Changed pixels vs previous frame", self.change_value)
        control_layout.addWidget(metrics_group)

        plot_group = QGroupBox("Coverage history")
        plot_layout = QVBoxLayout(plot_group)
        self.coverage_plot = CoveragePlotWidget()
        plot_layout.addWidget(self.coverage_plot)
        control_layout.addWidget(plot_group)

        self.export_panel = VideoExportPanel()
        self.export_panel.status_changed.connect(self._set_status)
        self.export_panel.error_occurred.connect(self._on_video_error)
        control_layout.addWidget(self.export_panel)
        control_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(250)
        scroll.setMaximumWidth(420)
        scroll.setWidget(control_container)
        self.main_splitter.addWidget(scroll)
        self.main_splitter.setSizes([820, 450])

        self._set_current_metrics(None)
        self.frame_preview.set_empty(
            "No video loaded",
            "Open a local RGB video supported by the installed codec backend.",
        )
        self.mask_preview.set_empty(
            "No frame mask",
            "Process the current frame or analyze the video sequentially.",
        )

    @staticmethod
    def _metric_label() -> QLabel:
        label = QLabel("—")
        label.setObjectName("metricValue")
        return label

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)

    def _update_export_panel(self) -> None:
        self.export_panel.set_snapshot(
            self.video_path,
            self.current_frame_index,
            self.current_frame_bgr,
            self.current_mask,
            self.current_overlay,
            self.analysis_records,
            self.analysis_config,
            frame_result_current=self.frame_result_current,
            analysis_results_current=self.analysis_results_current,
        )

    def _set_result_state(self, message: str, *, stale: bool = False) -> None:
        self.result_state_label.setText(message)
        self.result_state_label.setProperty("stale", stale)
        self.result_state_label.style().unpolish(self.result_state_label)
        self.result_state_label.style().polish(self.result_state_label)

    def _refresh_video_result_state(self) -> None:
        stale_messages: list[str] = []
        if self.current_mask is not None and not self.frame_result_current:
            stale_messages.append("current-frame mask")
        if self.analysis_records and not self.analysis_results_current:
            stale_messages.append("video frame metrics")
        if stale_messages:
            self._set_result_state(
                f"Stale {' and '.join(stale_messages)} · rerun with current "
                "parameters before export or evaluation.",
                stale=True,
            )
        elif self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._set_result_state(
                "Sequential analysis is running with its captured settings."
            )
        elif self.current_mask is not None:
            self._set_result_state(
                "Current-frame mask uses the displayed segmentation settings."
            )
        elif self.analysis_records:
            self._set_result_state(
                "Frame metrics use the displayed segmentation settings."
            )
        else:
            self._set_result_state("No analysis result is current.")

    def _emit_current_prediction(self) -> None:
        if (
            self.current_frame_bgr is None
            or self.current_mask is None
            or not self.frame_result_current
        ):
            self.prediction_ready.emit(None, None, "", None)
            return
        frame_number = (
            str(self.current_frame_index + 1)
            if self.current_frame_index is not None
            else "—"
        )
        self.prediction_ready.emit(
            self.current_frame_bgr.copy(),
            self.current_mask.copy(),
            f"{self.video_path.name if self.video_path else 'Video'} · frame {frame_number}",
            {
                "method": "HSV threshold, optional blur, morphology, and connected regions",
                "channel_order": "BGR",
                "parameters": (
                    asdict(self.current_segmentation_config)
                    if self.current_segmentation_config is not None
                    else {}
                ),
            },
        )

    def _mark_video_results_stale(self, *_args) -> None:
        try:
            current_config = self.segmentation_controls.config()
        except (TypeError, ValueError):
            current_config = None
        if self.current_mask is not None and self.current_segmentation_config is not None:
            self.frame_result_current = (
                self.current_segmentation_config == current_config
            )
        else:
            self.frame_result_current = False
        if self.analysis_records and self.analysis_config is not None:
            self.analysis_results_current = self.analysis_config == current_config
        else:
            self.analysis_results_current = False
        self._update_export_panel()
        self._emit_current_prediction()
        self._refresh_video_result_state()
        if (self.current_mask is not None and not self.frame_result_current) or (
            self.analysis_records and not self.analysis_results_current
        ):
            self._set_status(
                "Segmentation settings changed. Existing results are marked stale; "
                "rerun before exporting or evaluating."
            )

    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open aerial video",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path) -> None:
        self.shutdown_video_workers()
        self._playing = False
        self.play_button.setText("Play")
        self.video_path = path
        self.video_info = None
        self.current_frame_bgr = None
        self.current_frame_index = None
        self.current_mask = None
        self.current_overlay = None
        self.current_segmentation_config = None
        self.analysis_config = None
        self.frame_result_current = False
        self.analysis_results_current = False
        self.analysis_records = []
        self.coverage_plot.set_empty()
        self._set_current_metrics(None)
        self.frame_preview.set_empty(
            "Loading video", "Reading metadata and the first frame…"
        )
        self.mask_preview.set_empty(
            "No frame mask", "Process a frame to create a mask."
        )
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("No sequential analysis running")
        self.frame_slider.setEnabled(False)
        self._set_playback_controls(False)
        self._update_export_panel()
        self.prediction_ready.emit(None, None, "", None)
        worker = VideoPlaybackWorker(path, self)
        self._playback_worker = worker
        worker.info_ready.connect(self._on_video_info)
        worker.frame_ready.connect(self._on_video_frame)
        worker.error_occurred.connect(self._on_video_error)
        worker.playback_ended.connect(self._on_playback_ended)
        worker.start()
        self._set_status(f"Opening {path.name}…")

    def _on_video_info(self, info: VideoInfo) -> None:
        self.video_info = info
        if info.frame_count:
            self.frame_slider.setRange(0, max(0, info.frame_count - 1))
            self.frame_slider.setEnabled(info.frame_count > 1)
        else:
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setEnabled(False)
        fps = f"{info.fps:.2f} fps" if info.fps is not None else "fps unavailable"
        frames = (
            f"{info.frame_count:,} frames"
            if info.frame_count
            else "frame count unknown"
        )
        self.time_label.setText(f"{info.width:,} × {info.height:,} px · {fps}")
        self.registration_label.setText(
            f"Frame registration · {info.path.name} · {info.width:,} × "
            f"{info.height:,} px · {frames} · backend: {info.backend}"
        )
        self._set_playback_controls(True)
        self._refresh_video_result_state()
        self._set_status("Video loaded and paused on the first frame.")

    def _on_video_frame(self, frame: VideoFrame) -> None:
        self._frame_revision += 1
        self.current_frame_bgr = frame.image_bgr.copy()
        self.current_frame_index = frame.index
        self.frame_preview.set_image(frame.image_bgr, "BGR")
        with QSignalBlocker(self.frame_slider):
            if self.frame_slider.maximum() >= frame.index:
                self.frame_slider.setValue(frame.index)
        total = self.video_info.frame_count if self.video_info else None
        frame_text = f"Frame {frame.index + 1:,}"
        if total:
            frame_text += f" / {total:,}"
        self.frame_label.setText(frame_text)
        if self.video_info and self.video_info.fps:
            seconds = frame.index / self.video_info.fps
            self.time_label.setText(
                f"{seconds // 60:02.0f}:{seconds % 60:05.2f} · "
                f"{self.video_info.width:,} × {self.video_info.height:,} px"
            )
        if self.current_mask is None or frame.index != getattr(
            self, "_mask_frame_index", None
        ):
            self.current_mask = None
            self.current_overlay = None
            self.current_segmentation_config = None
            self.frame_result_current = False
            self._mask_frame_index = None
            self.mask_preview.set_empty(
                "No mask for this frame",
                "Process the current frame or analyze the video.",
            )
        self._set_current_metrics(None)
        self._refresh_video_result_state()
        self._update_export_panel()
        self._emit_current_prediction()
        self._refresh_video_result_state()
        busy = any(
            worker is not None and worker.isRunning()
            for worker in (self._frame_worker, self._analysis_worker)
        )
        self._set_playback_controls(self.video_info is not None and not busy)
        if self._playback_worker is not None:
            self._playback_worker.acknowledge_frame()

    def _on_video_error(self, message: str) -> None:
        self._set_status(f"Video error: {message}")

    def _set_playback_controls(self, enabled: bool) -> None:
        for button in (self.play_button, self.previous_button, self.next_button):
            button.setEnabled(enabled)
        self.process_current_button.setEnabled(
            enabled and self.current_frame_bgr is not None
        )
        self.analyze_button.setEnabled(enabled and self.video_path is not None)
        can_seek = bool(
            enabled
            and self.video_info is not None
            and self.video_info.frame_count is not None
            and self.video_info.frame_count > 1
        )
        self.frame_slider.setEnabled(can_seek)

    def toggle_playback(self) -> None:
        if self._playback_worker is None:
            return
        if self._playing:
            self._playing = False
            self._playback_worker.pause()
            self.play_button.setText("Play")
            self._set_status("Playback paused.")
        else:
            self._playing = True
            self._playback_worker.play()
            self.play_button.setText("Pause")
            self._set_status("Playing video preview; analysis remains frame-by-frame.")

    def _on_playback_ended(self) -> None:
        self._playing = False
        self.play_button.setText("Play")
        self._set_status("End of video. Press Play to replay from the first frame.")

    def _seek_relative(self, amount: int) -> None:
        if self._playback_worker is None or self.current_frame_index is None:
            return
        maximum = self.frame_slider.maximum()
        target = max(0, min(maximum, self.current_frame_index + amount))
        self._playback_worker.seek(target)

    def _seek_frame(self, frame_index: int) -> None:
        if self._playback_worker is not None:
            self._playback_worker.seek(frame_index)

    def process_current_frame(self) -> None:
        if self.current_frame_bgr is None:
            return
        if self._frame_worker is not None and self._frame_worker.isRunning():
            return
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            return
        if self._playing and self._playback_worker is not None:
            self._playing = False
            self._playback_worker.pause()
            self.play_button.setText("Play")
        try:
            config = self.segmentation_controls.config()
        except (TypeError, ValueError) as exc:
            self._on_video_error(str(exc))
            return
        self._processing_frame_revision = self._frame_revision
        self.frame_result_current = (
            self.current_mask is not None
            and self.current_segmentation_config == config
        )
        self._set_playback_controls(False)
        self.segmentation_controls.setEnabled(False)
        self._update_export_panel()
        self._refresh_video_result_state()
        worker = ImageTaskWorker(
            SegmentTask(self.current_frame_bgr.copy(), config),
            self,
        )
        self._frame_worker = worker
        worker.result_ready.connect(self._on_current_frame_segmented)
        worker.error_occurred.connect(self._on_video_error)
        worker.finished.connect(self._on_frame_worker_finished)
        self.process_current_button.setEnabled(False)
        self._set_status("Processing current frame…")
        worker.start()

    def _on_current_frame_segmented(self, output: SegmentationOutput) -> None:
        if self._processing_frame_revision != self._frame_revision:
            self._set_status(
                "Frame changed while processing. The old result was discarded; "
                "process the current frame again."
            )
            return
        self.current_mask = output.result.mask
        self.current_overlay = output.overlay_bgr
        self.current_segmentation_config = output.result.config
        self.frame_result_current = (
            output.result.config == self.segmentation_controls.config()
        )
        self._mask_frame_index = self.current_frame_index
        self.mask_preview.set_image(self.current_mask, "GRAY")
        self._set_current_metrics(output.water_analysis)
        self._update_export_panel()
        self._emit_current_prediction()
        self._refresh_video_result_state()
        self._set_status("Current frame processed. Mask is a potential-water result.")

    def _on_frame_worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._frame_worker:
            self._frame_worker = None
            worker.deleteLater()
        self._processing_frame_revision = None
        self.segmentation_controls.setEnabled(True)
        self._mark_video_results_stale()
        self._set_playback_controls(self.video_info is not None)

    def _set_current_metrics(
        self,
        analysis: WaterAnalysis | VideoProgress | None,
    ) -> None:
        if analysis is None:
            values = (
                self.total_pixels_value,
                self.water_pixels_value,
                self.coverage_value,
                self.region_count_value,
                self.change_value,
            )
            for value in values:
                value.setText("—")
            return
        self.total_pixels_value.setText(f"{analysis.total_pixels:,} px")
        self.water_pixels_value.setText(f"{analysis.water_pixels:,} px")
        self.coverage_value.setText(f"{analysis.coverage_percent:.2f}%")
        self.region_count_value.setText(str(analysis.region_count))
        if isinstance(analysis, VideoProgress):
            self.change_value.setText(
                f"{analysis.changed_pixels:,} px"
                if analysis.changed_pixels is not None
                else "—"
            )
        else:
            self.change_value.setText("—")

    def analyze_video(self) -> None:
        if self.video_path is None or self.video_info is None:
            return
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            return
        if self._frame_worker is not None and self._frame_worker.isRunning():
            return
        if self._playing and self._playback_worker is not None:
            self._playing = False
            self._playback_worker.pause()
            self.play_button.setText("Play")
        try:
            config = self.segmentation_controls.config()
        except (TypeError, ValueError) as exc:
            self._on_video_error(str(exc))
            return
        previous_frame_config = self.current_segmentation_config
        self.analysis_config = config
        self.analysis_records = []
        self.analysis_results_current = False
        self.coverage_plot.set_empty()
        self.frame_result_current = (
            self.current_mask is not None
            and previous_frame_config == config
        )
        self._set_playback_controls(False)
        self.segmentation_controls.setEnabled(False)
        self._update_export_panel()
        self._refresh_video_result_state()
        worker = VideoAnalysisWorker(
            self.video_path,
            config,
            parent=self,
        )
        self._analysis_worker = worker
        worker.completed.connect(self._on_analysis_completed)
        worker.cancelled.connect(self._on_analysis_cancelled)
        worker.error_occurred.connect(self._on_video_error)
        worker.finished.connect(self._on_analysis_worker_finished)
        if self.video_info.frame_count:
            self.progress_bar.setRange(0, self.video_info.frame_count)
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Processing frames…")
        self.cancel_button.setEnabled(True)
        self.analyze_button.setEnabled(False)
        self._poll_timer.start()
        self._set_status("Sequential frame analysis started.")
        worker.start()

    def _poll_analysis_progress(self) -> None:
        if self._analysis_worker is None:
            return
        progress = self._analysis_worker.take_latest_progress()
        if progress is None:
            return
        self._show_video_progress(progress)

    def _show_video_progress(self, progress: VideoProgress) -> None:
        self.current_frame_bgr = progress.frame_bgr
        self.current_frame_index = progress.frame_index
        self.current_mask = progress.mask
        self.current_overlay = None
        self.current_segmentation_config = self.analysis_config
        self.frame_result_current = True
        self._mask_frame_index = progress.frame_index
        self.frame_preview.set_image(progress.frame_bgr, "BGR")
        self.mask_preview.set_image(progress.mask, "GRAY")
        total = self.video_info.frame_count if self.video_info else None
        frame_text = f"Frame {progress.frame_index + 1:,}"
        if total:
            frame_text += f" / {total:,}"
        self.frame_label.setText(frame_text)
        with QSignalBlocker(self.frame_slider):
            if progress.frame_index <= self.frame_slider.maximum():
                self.frame_slider.setValue(progress.frame_index)
        if self.video_info and self.video_info.fps:
            timestamp_ms = (
                progress.timestamp_ms
                if progress.timestamp_ms is not None
                else progress.frame_index * 1000.0 / self.video_info.fps
            )
            elapsed_seconds = timestamp_ms / 1000.0
            self.time_label.setText(
                f"{elapsed_seconds // 60:02.0f}:{elapsed_seconds % 60:05.2f} · "
                f"{self.video_info.width:,} × {self.video_info.height:,} px"
            )
        if self.video_info and self.video_info.frame_count:
            self.progress_bar.setValue(progress.frame_index + 1)
        self._set_current_metrics(progress)
        comparison = progress.changed_pixels
        self.change_value.setText(
            f"{comparison:,} px" if comparison is not None else "—"
        )
        self._update_export_panel()
        self._emit_current_prediction()
        self._refresh_video_result_state()

    def _set_analysis_rows(self, rows: list[dict[str, object]]) -> None:
        self.analysis_records = rows
        try:
            current_config = self.segmentation_controls.config()
        except (TypeError, ValueError):
            current_config = None
        self.analysis_results_current = bool(rows) and (
            self.analysis_config == current_config
        )
        self.coverage_plot.set_records(rows)
        self._update_export_panel()
        self._refresh_video_result_state()

    def _on_analysis_completed(self, rows: list[dict[str, object]]) -> None:
        self._poll_analysis_progress()
        self._set_analysis_rows(rows)
        if self.video_info and self.video_info.frame_count:
            self.progress_bar.setRange(0, self.video_info.frame_count)
            self.progress_bar.setValue(len(rows))
        self.progress_bar.setFormat("Frame analysis complete")
        self._set_status(f"Analyzed {len(rows):,} frames sequentially.")
        if self.current_frame_index is not None and self._playback_worker is not None:
            self._playback_worker.seek(self.current_frame_index)

    def _on_analysis_cancelled(self, rows: list[dict[str, object]]) -> None:
        self._poll_analysis_progress()
        self._set_analysis_rows(rows)
        self.progress_bar.setFormat("Analysis cancelled · partial metrics retained")
        self._set_status(
            f"Analysis cancelled after {len(rows):,} frames; "
            "partial metrics are available."
        )

    def _on_analysis_worker_finished(self) -> None:
        self._poll_timer.stop()
        worker = self.sender()
        if worker is self._analysis_worker:
            self._analysis_worker = None
            worker.deleteLater()
        self.cancel_button.setEnabled(False)
        self.segmentation_controls.setEnabled(True)
        self._mark_video_results_stale()
        self._set_playback_controls(self.video_info is not None)

    def cancel_analysis(self) -> None:
        if self._analysis_worker is not None and self._analysis_worker.isRunning():
            self._analysis_worker.cancel()
            self.cancel_button.setEnabled(False)
            self._set_status(
                "Cancellation requested; the active frame will finish first."
            )

    def shutdown_video_workers(self) -> None:
        """Cancel and join workers before replacing a video or closing the app."""
        self._poll_timer.stop()
        if self._analysis_worker is not None:
            worker = self._analysis_worker
            if worker.isRunning():
                worker.cancel()
                worker.wait()
            worker.deleteLater()
        if self._frame_worker is not None:
            worker = self._frame_worker
            if worker.isRunning():
                worker.wait()
            worker.deleteLater()
        if self._playback_worker is not None:
            worker = self._playback_worker
            if worker.isRunning():
                worker.stop()
                worker.wait()
            worker.deleteLater()
        self._analysis_worker = None
        self._frame_worker = None
        self._playback_worker = None

    def shutdown(self) -> None:
        self.shutdown_video_workers()
