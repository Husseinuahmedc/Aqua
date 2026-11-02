"""Primary Search & Rescue analysis workspace for UAV still images and videos."""

from dataclasses import replace
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.components.file_dialogs import choose_save_path
from app.components.image_preview import ImagePreviewWidget
from app.components.rescue_controls import RescueControls
from app.exporting import (
    export_image_candidates_csv,
    export_image_candidates_json,
    export_video_candidates_csv,
    export_video_candidates_json,
)
from app.workers import (
    ImageTaskWorker,
    LoadImageTask,
    LoadedImageOutput,
    RescueOutput,
    RescueTask,
    RescueVideoAnalysisWorker,
    RescueVideoProgress,
)
from core.rescue.candidates import (
    Candidate,
    describe_candidate_reasoning,
    format_category_label,
    format_review_status,
)
from core.rescue.pipeline import RescueAnalysis, RescueConfig
from core.rescue.video import TrackedCandidate
from core.rescue.visualization import draw_candidate_boxes_rgb
from core.video.reader import VideoCaptureReader, VideoInfo
from core.water.segmentation import SegmentationConfig


class RescueWorkspace(QWidget):
    """Primary rescue-first workspace supporting UAV image and video review."""

    status_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.input_mode: Literal["none", "image", "video"] = "none"
        self.source_path: Path | None = None
        self.source_bgr: NDArray[np.uint8] | None = None
        self.source_rgb: NDArray[np.uint8] | None = None
        self.video_info: VideoInfo | None = None
        self.current_frame_index: int | None = None
        self.analysis: RescueAnalysis | None = None
        self.water_context_config: SegmentationConfig | None = None
        self.analysis_current = False
        self.rescue_results_current = False
        self.rescue_config: RescueConfig | None = None
        self.rescue_water_config: SegmentationConfig | None = None
        self.rescue_use_water_context = False
        self.rescue_candidate_records: list[dict[str, object]] = []
        self.rescue_frame_records: list[dict[str, object]] = []
        self._tracked_by_frame: dict[int, tuple[TrackedCandidate, ...]] = {}
        self._track_lookup: dict[str, tuple[str, int]] = {}
        self._candidates: tuple[Candidate, ...] = ()
        self._worker: ImageTaskWorker | None = None
        self._video_worker: RescueVideoAnalysisWorker | None = None
        self._selected_candidate_id: str | None = None
        self._input_revision = 0
        self._active_task_revision: int | None = None
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(80)
        self._poll_timer.timeout.connect(self._poll_video_progress)
        self._build_ui()
        self._reset_results()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        # Area A & B: Input selection and primary Analysis Controls bar.
        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.open_button = QPushButton("Open image")
        self.open_button.setToolTip("Import a local UAV RGB image for candidate analysis.")
        self.open_video_button = QPushButton("Open video")
        self.open_video_button.setToolTip(
            "Import a local UAV video for sequential frame candidate analysis."
        )
        self.clear_button = QPushButton("Clear input")
        self.clear_button.setEnabled(False)
        self.clear_button.setToolTip("Remove the current input and reset analysis views.")

        self.run_button = QPushButton("Analyze")
        self.run_button.setObjectName("primaryAction")
        self.run_button.setEnabled(False)
        self.run_button.setToolTip(
            "Run traditional DIP analysis to highlight Potential Person and Floating Object regions."
        )
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip("Stop sequential video analysis after the active frame.")

        self.advanced_toggle_button = QPushButton("Advanced settings ▸")
        self.advanced_toggle_button.setCheckable(True)
        self.advanced_toggle_button.setChecked(False)
        self.advanced_toggle_button.setToolTip(
            "Show or hide optional water context and DIP proposal parameters."
        )

        self.source_label = QLabel("No image or video selected")
        self.source_label.setObjectName("mutedText")
        self.source_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        toolbar.addWidget(self.open_button)
        toolbar.addWidget(self.open_video_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addSpacing(8)
        toolbar.addWidget(self.run_button)
        toolbar.addWidget(self.cancel_button)
        toolbar.addWidget(self.advanced_toggle_button)
        toolbar.addStretch(1)
        toolbar.addWidget(self.source_label)
        root.addLayout(toolbar)

        # Progress and status row.
        status_row = QHBoxLayout()
        status_row.setSpacing(12)
        self.status_label = QLabel(
            "Import a UAV image or video, then click Analyze to propose regions for human review."
        )
        self.status_label.setObjectName("mutedText")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.setAccessibleName("Rescue analysis progress")
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

        # Collapsible Advanced Settings panel (collapsed by default).
        self.advanced_panel = QGroupBox("Advanced DIP configuration")
        advanced_layout = QHBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(10, 8, 10, 8)
        advanced_layout.setSpacing(16)
        self.rescue_controls = RescueControls()
        self.water_context_checkbox = QCheckBox(
            "Use potential-water segmentation as optional context"
        )
        self.water_context_checkbox.setToolTip(
            "Runs the potential-water baseline as a separate context cue for Floating Object "
            "scoring. Candidate generation still runs across the entire frame."
        )
        context_box = QVBoxLayout()
        context_box.addWidget(self.water_context_checkbox)
        context_note = QLabel(
            "Sensible defaults are active automatically. Expand this section only when "
            "tuning proposal methods or area bounds on development imagery."
        )
        context_note.setObjectName("mutedText")
        context_note.setWordWrap(True)
        context_box.addWidget(context_note)
        context_box.addStretch(1)
        advanced_layout.addWidget(self.rescue_controls, 2)
        advanced_layout.addLayout(context_box, 1)
        self.advanced_panel.setVisible(False)
        root.addWidget(self.advanced_panel)

        # Main workspace splitter: Area C (Results) at left, Area D (Candidate Review) at right.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        # Area C: Results (Side-by-side visual comparison + summary strip).
        evidence = QWidget()
        evidence_layout = QVBoxLayout(evidence)
        evidence_layout.setContentsMargins(0, 0, 0, 0)
        evidence_layout.setSpacing(8)
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)
        preview_splitter.setChildrenCollapsible(False)
        self.original_preview = ImagePreviewWidget("Original UAV input")
        self.candidate_preview = ImagePreviewWidget(
            "Candidate regions · Requires Human Review"
        )
        preview_splitter.addWidget(self.original_preview)
        preview_splitter.addWidget(self.candidate_preview)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 1)
        evidence_layout.addWidget(preview_splitter, 1)

        self.registration_label = QLabel(
            "Registration · Source geometry preserved · All outputs require human verification."
        )
        self.registration_label.setObjectName("registrationStrip")
        self.registration_label.setWordWrap(True)
        evidence_layout.addWidget(self.registration_label)

        self.result_state_label = QLabel("No analysis result is current.")
        self.result_state_label.setObjectName("resultState")
        self.result_state_label.setWordWrap(True)
        evidence_layout.addWidget(self.result_state_label)
        splitter.addWidget(evidence)

        # Area D: Candidate Review panel.
        review_container = QWidget()
        review_layout = QVBoxLayout(review_container)
        review_layout.setContentsMargins(4, 0, 0, 0)
        review_layout.setSpacing(10)

        filter_group = QGroupBox("Candidate list & filters")
        filter_layout = QVBoxLayout(filter_group)
        filter_form = QFormLayout()
        self.category_filter = QComboBox()
        for label, category in (
            ("All candidates", "all"),
            ("Potential Person", "person-like"),
            ("Floating Object", "floating-object-like"),
            ("Unknown Candidate", "unknown"),
        ):
            self.category_filter.addItem(label, category)
        self.video_scope_selector = QComboBox()
        for label, scope in (
            ("All analyzed frames (Frame Candidates)", "all_frames"),
            ("Current preview frame only", "current_frame"),
            ("Tracked Candidates (2+ frames · experimental)", "tracked_only"),
        ):
            self.video_scope_selector.addItem(label, scope)
        self.video_scope_selector.setEnabled(False)
        filter_form.addRow("Category", self.category_filter)
        filter_form.addRow("Video view", self.video_scope_selector)
        filter_layout.addLayout(filter_form)

        self.candidate_table = QTableWidget(0, 5)
        self.candidate_table.setHorizontalHeaderLabels(
            ("Candidate", "Category", "Context", "Area (px)", "Status")
        )
        self.candidate_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.candidate_table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self.candidate_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.setMinimumHeight(180)
        filter_layout.addWidget(self.candidate_table, 1)
        review_layout.addWidget(filter_group, 2)

        review_group = QGroupBox("Selected candidate review")
        review_form = QFormLayout(review_group)
        self.selected_candidate_label = QLabel("No candidate selected")
        self.selected_candidate_label.setWordWrap(True)
        self.review_status = QComboBox()
        for label, value in (
            ("Requires Human Review", "needs_review"),
            ("Confirmed Candidate", "confirmed"),
            ("False Alarm", "false_alarm"),
        ):
            self.review_status.addItem(label, value)
        self.review_status.setEnabled(False)
        self.reasoning_label = QLabel(
            "Select a candidate in the list to highlight its region and inspect its visual reasoning."
        )
        self.reasoning_label.setWordWrap(True)
        self.reasoning_label.setObjectName("mutedText")
        self.feature_details = QLabel("No candidate selected.")
        self.feature_details.setWordWrap(True)
        self.feature_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        review_form.addRow("Region", self.selected_candidate_label)
        review_form.addRow("Review decision", self.review_status)
        review_form.addRow("Why proposed", self.reasoning_label)
        review_form.addRow("Measurements", self.feature_details)
        review_layout.addWidget(review_group, 1)

        caution = QLabel(
            "Requires Human Review: Traditional DIP highlights Potential Person and "
            "Floating Object regions using geometry and contrast only. Results are "
            "suggestions for human verification, never confirmed survivors."
        )
        caution.setWordWrap(True)
        caution.setObjectName("mutedText")
        review_layout.addWidget(caution)

        export_row = QHBoxLayout()
        self.export_csv_button = QPushButton("Export candidates to CSV")
        self.export_json_button = QPushButton("Export candidates to JSON")
        self.export_csv_button.setEnabled(False)
        self.export_json_button.setEnabled(False)
        export_row.addWidget(self.export_csv_button)
        export_row.addWidget(self.export_json_button)
        review_layout.addLayout(export_row)

        review_scroll = QScrollArea()
        review_scroll.setWidgetResizable(True)
        review_scroll.setMinimumWidth(340)
        review_scroll.setMaximumWidth(480)
        review_scroll.setWidget(review_container)
        splitter.addWidget(review_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([920, 440])

        self.open_button.clicked.connect(self.open_image_dialog)
        self.open_video_button.clicked.connect(self.open_video_dialog)
        self.clear_button.clicked.connect(self.clear_input)
        self.run_button.clicked.connect(self.run_analysis)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        self.advanced_toggle_button.toggled.connect(self._toggle_advanced_settings)
        self.category_filter.currentIndexChanged.connect(self._refresh_candidates)
        self.video_scope_selector.currentIndexChanged.connect(self._refresh_candidates)
        self.candidate_table.itemSelectionChanged.connect(self._selection_changed)
        self.review_status.currentIndexChanged.connect(self._review_status_changed)
        self.rescue_controls.parameters_changed.connect(self._mark_analysis_stale)
        self.water_context_checkbox.toggled.connect(self._mark_analysis_stale)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_json_button.clicked.connect(self.export_json)

    def _toggle_advanced_settings(self, checked: bool) -> None:
        self.advanced_panel.setVisible(checked)
        self.advanced_toggle_button.setText(
            "Advanced settings ▾" if checked else "Advanced settings ▸"
        )

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)

    def _set_result_state(self, message: str, *, stale: bool = False) -> None:
        self.result_state_label.setText(message)
        self.result_state_label.setProperty("stale", stale)
        self.result_state_label.style().unpolish(self.result_state_label)
        self.result_state_label.style().polish(self.result_state_label)

    def _reset_results(self) -> None:
        self.original_preview.set_empty(
            "No image or video loaded",
            "Use Open image or Open video to load UAV imagery for Search & Rescue review.",
        )
        self.candidate_preview.set_empty(
            "No candidate regions generated",
            "Click Analyze to highlight Potential Person and Floating Object regions.",
        )
        self.candidate_table.setRowCount(0)
        self.selected_candidate_label.setText("No candidate selected")
        self.reasoning_label.setText(
            "Select a candidate in the list to highlight its region and inspect its visual reasoning."
        )
        self.feature_details.setText("No candidate selected.")
        self.review_status.setEnabled(False)
        self._set_result_state("No analysis result is current.")

    def clear_input(self) -> None:
        """Clear the active image or video input and reset all review state."""
        if self._is_busy():
            return
        self._input_revision += 1
        self.input_mode = "none"
        self.source_path = None
        self.source_bgr = None
        self.source_rgb = None
        self.video_info = None
        self.current_frame_index = None
        self.analysis = None
        self.water_context_config = None
        self.analysis_current = False
        self.rescue_results_current = False
        self.rescue_config = None
        self.rescue_water_config = None
        self.rescue_use_water_context = False
        self.rescue_candidate_records = []
        self.rescue_frame_records = []
        self._tracked_by_frame = {}
        self._track_lookup = {}
        self._candidates = ()
        self._selected_candidate_id = None
        self.source_label.setText("No image or video selected")
        self.registration_label.setText(
            "Registration · Source geometry preserved · All outputs require human verification."
        )
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        self._reset_results()
        self._set_busy(False)
        self._set_status("Input cleared. Open a local UAV image or video to begin.")

    def _is_busy(self) -> bool:
        return (
            self._worker is not None and self._worker.isRunning()
        ) or (
            self._video_worker is not None and self._video_worker.isRunning()
        )

    def open_image_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open UAV image for Search & Rescue analysis",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if path:
            self.open_image(Path(path))

    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open UAV video for Search & Rescue analysis",
            str(Path.home()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.wmv);;All files (*)",
        )
        if path:
            self.open_video(Path(path))

    def open_image(self, path: Path) -> None:
        if self._is_busy():
            return
        self._input_revision += 1
        self._active_task_revision = self._input_revision
        self._start_task(
            LoadImageTask(path), self._on_image_loaded, f"Loading {path.name}…"
        )

    def open_video(self, path: Path) -> None:
        """Load a local UAV video and display its first frame paused for analysis."""
        if self._is_busy():
            return
        self._input_revision += 1
        try:
            with VideoCaptureReader(path) as reader:
                info = reader.info
                first = reader.read_next()
        except Exception as exc:
            self._on_error(str(exc))
            return
        if first is None:
            self._on_error(f"Video '{path}' contains no decodable frames.")
            return
        self.input_mode = "video"
        self.source_path = path
        self.video_info = info
        self.current_frame_index = first.index
        self.source_bgr = first.image_bgr.copy()
        self.source_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
        self.analysis = None
        self.water_context_config = None
        self.analysis_current = False
        self.rescue_results_current = False
        self.rescue_config = None
        self.rescue_water_config = None
        self.rescue_use_water_context = False
        self.rescue_candidate_records = []
        self.rescue_frame_records = []
        self._tracked_by_frame = {}
        self._track_lookup = {}
        self._candidates = ()
        self._selected_candidate_id = None

        frames_text = (
            f"{info.frame_count:,} frames" if info.frame_count else "frame count unknown"
        )
        fps_text = f"{info.fps:.2f} fps" if info.fps else "fps unknown"
        self.source_label.setText(
            f"UAV Video · {path.name} · {info.width:,} × {info.height:,} px · "
            f"{frames_text} · {fps_text}"
        )
        self.registration_label.setText(
            f"Registration · {path.name} · Frame 1 · {info.width:,} × {info.height:,} px · "
            "Sequential frame-by-frame processing"
        )
        self._reset_results()
        self.original_preview.set_image(self.source_bgr, "BGR")
        self.candidate_preview.set_empty(
            "Video ready for candidate analysis",
            "Click Analyze to run frame-by-frame proposal extraction and experimental tracking.",
        )
        self.progress_bar.setRange(0, max(1, info.frame_count or 1))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self._set_busy(False)
        self._set_status(
            "Video loaded on frame 1. Click Analyze to extract frame candidates and tracks."
        )

    def _start_task(
        self,
        task: LoadImageTask | RescueTask,
        handler: Callable[[object], None],
        message: str,
    ) -> None:
        if self._is_busy():
            return
        worker = ImageTaskWorker(task, self)
        self._worker = worker
        worker.result_ready.connect(handler)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._worker_finished)
        self._set_busy(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat("Processing…")
        self._set_status(message)
        worker.start()

    def _set_busy(self, busy: bool) -> None:
        has_input = self.source_bgr is not None and self.source_path is not None
        has_results = self._has_results()
        is_current = (
            self.analysis_current
            if self.input_mode == "image"
            else self.rescue_results_current
        )
        self.open_button.setEnabled(not busy)
        self.open_video_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy and has_input)
        self.run_button.setEnabled(not busy and has_input)
        self.cancel_button.setEnabled(
            busy and self._video_worker is not None and self._video_worker.isRunning()
        )
        self.water_context_checkbox.setEnabled(not busy)
        self.rescue_controls.set_enabled(not busy)
        self.category_filter.setEnabled(not busy and has_results)
        self.video_scope_selector.setEnabled(
            not busy and self.input_mode == "video" and has_results
        )
        self.export_csv_button.setEnabled(not busy and has_results and is_current)
        self.export_json_button.setEnabled(not busy and has_results and is_current)
        self.review_status.setEnabled(
            not busy and self._selected_candidate_id is not None
        )

    def _has_results(self) -> bool:
        if self.input_mode == "video":
            return bool(self.rescue_frame_records) or self.rescue_config is not None
        return self.analysis is not None

    def _worker_finished(self) -> None:
        worker = self.sender()
        if worker is self._worker:
            self._worker = None
            worker.deleteLater()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if self._has_results() else 0)
        self.progress_bar.setFormat("Complete" if self._has_results() else "Idle")
        self._set_busy(False)

    def _on_error(self, message: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error")
        self._set_status(f"Error: {message}")
        QMessageBox.warning(self, "Rescue analysis operation failed", message)

    def _on_image_loaded(self, output: LoadedImageOutput) -> None:
        if self._active_task_revision != self._input_revision:
            return
        self.input_mode = "image"
        self.source_path = output.path
        self.video_info = None
        self.current_frame_index = None
        self.source_bgr = output.image_bgr
        self.source_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
        self.analysis = None
        self.water_context_config = None
        self.analysis_current = False
        self.rescue_results_current = False
        self.rescue_candidate_records = []
        self.rescue_frame_records = []
        self._tracked_by_frame = {}
        self._track_lookup = {}
        self._candidates = ()
        self._selected_candidate_id = None
        self.source_label.setText(
            f"Still Image · {output.path.name} · "
            f"{output.info.width:,} × {output.info.height:,} px"
        )
        self.registration_label.setText(
            f"Registration · {output.path.name} · "
            f"{output.info.width:,} × {output.info.height:,} px · "
            "Full resolution preserved"
        )
        self._reset_results()
        self.original_preview.set_image(self.source_bgr, "BGR")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Ready")
        self._set_busy(False)
        self._set_status(
            "Image loaded. Click Analyze to identify Potential Person and Floating Object candidates."
        )

    def run_analysis(self) -> None:
        """Primary Analyze action for either a loaded still image or UAV video."""
        if self.input_mode == "video":
            self.analyze_rescue_video()
        else:
            self.run_candidates()

    def run_candidates(self) -> None:
        """Run still-image candidate extraction (or dispatch video if in video mode)."""
        if self.input_mode == "video":
            self.analyze_rescue_video()
            return
        if self.source_bgr is None or self.source_path is None:
            return
        try:
            config = self.rescue_controls.config()
        except (TypeError, ValueError) as exc:
            self._on_error(str(exc))
            return
        self._active_task_revision = self._input_revision
        task = RescueTask(
            image_bgr=self.source_bgr.copy(),
            image_id=str(self.source_path),
            use_water_context=self.water_context_checkbox.isChecked(),
            config=config,
        )
        self._start_task(
            task,
            self._on_candidates_ready,
            "Analyzing image for Potential Person and Floating Object candidates…",
        )

    def _on_candidates_ready(self, output: RescueOutput) -> None:
        if self._active_task_revision != self._input_revision:
            return
        self.analysis = output.analysis
        self.water_context_config = output.water_context_config
        self.analysis_current = True
        self._candidates = output.analysis.candidates
        self._selected_candidate_id = None
        self.original_preview.set_image(self.source_bgr, "BGR")
        self._refresh_candidates()
        self._update_summary_banner()
        self._set_busy(False)
        context = (
            "with optional water context"
            if output.analysis.water_context_used
            else "without water context"
        )
        self._set_status(
            f"Analysis complete ({output.analysis.processing_time_ms:.1f} ms, {context}). "
            f"{len(self._candidates)} candidate(s) require human review."
        )

    def analyze_rescue_video(self) -> None:
        """Run sequential video candidate extraction and experimental tracking."""
        if self.source_path is None or self.video_info is None or self._is_busy():
            return
        try:
            config = self.rescue_controls.config()
        except (TypeError, ValueError) as exc:
            self._on_error(str(exc))
            return
        self._active_task_revision = self._input_revision
        self.rescue_config = config
        self.rescue_use_water_context = self.water_context_checkbox.isChecked()
        self.rescue_water_config = (
            SegmentationConfig() if self.rescue_use_water_context else None
        )
        self.rescue_results_current = False
        self.rescue_candidate_records = []
        self.rescue_frame_records = []
        self._tracked_by_frame = {}
        self._track_lookup = {}
        self._candidates = ()
        self._selected_candidate_id = None
        self.candidate_table.setRowCount(0)
        self.candidate_preview.set_empty(
            "Analyzing video frames…",
            "Extracting frame candidates and associating experimental optical-flow tracks.",
        )
        if self.video_info.frame_count:
            self.progress_bar.setRange(0, self.video_info.frame_count)
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Analyzing frames…")
        self._set_result_state("Sequential video rescue analysis is running…")

        worker = RescueVideoAnalysisWorker(
            self.source_path,
            config,
            use_water_context=self.rescue_use_water_context,
            water_config=self.rescue_water_config,
            parent=self,
        )
        self._video_worker = worker
        worker.completed.connect(self._on_video_completed)
        worker.cancelled.connect(self._on_video_cancelled)
        worker.error_occurred.connect(self._on_error)
        worker.finished.connect(self._on_video_worker_finished)
        self._set_busy(True)
        self._poll_timer.start()
        self._set_status(
            "Analyzing UAV video frame by frame; candidates and tracks require human review."
        )
        worker.start()

    def cancel_analysis(self) -> None:
        """Request cooperative cancellation of active video candidate analysis."""
        if self._video_worker is not None and self._video_worker.isRunning():
            self._video_worker.cancel()
            self.cancel_button.setEnabled(False)
            self._set_status(
                "Cancellation requested; finishing the active video frame…"
            )

    def _poll_video_progress(self) -> None:
        if self._video_worker is None:
            return
        progress = self._video_worker.take_latest_progress()
        if progress is not None:
            self._show_video_progress(progress)

    def _show_video_progress(self, progress: RescueVideoProgress) -> None:
        self.current_frame_index = progress.frame_index
        self.source_bgr = progress.frame_bgr.copy()
        self.source_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
        frame_candidates = tuple(
            item.candidate for item in progress.tracked_candidates
        )
        self.original_preview.set_image(self.source_bgr, "BGR")
        self.candidate_preview.set_image(
            draw_candidate_boxes_rgb(self.source_rgb, frame_candidates),
            "RGB",
        )
        if self.video_info and self.video_info.frame_count:
            self.progress_bar.setValue(progress.frame_index + 1)
        motion_note = (
            "camera motion compensated"
            if progress.motion_compensated
            else "raw frame coordinates"
        )
        self.registration_label.setText(
            f"Registration · {self.source_path.name if self.source_path else 'Video'} · "
            f"Frame {progress.frame_index + 1:,} · "
            f"{len(frame_candidates)} frame candidate(s) · "
            f"{progress.active_track_count} active track(s) ({motion_note})"
        )

    def _apply_video_payload(self, payload: dict[str, object]) -> None:
        if self._active_task_revision != self._input_revision:
            return
        self.rescue_candidate_records = list(payload.get("candidates", []))
        self.rescue_frame_records = list(payload.get("frames", []))
        raw_tracked = payload.get("tracked_by_frame", {})
        self._tracked_by_frame = (
            dict(raw_tracked) if isinstance(raw_tracked, dict) else {}
        )
        last_bgr = payload.get("last_frame_bgr")
        last_index = payload.get("last_frame_index")
        if isinstance(last_bgr, np.ndarray) and isinstance(last_index, int):
            self.current_frame_index = last_index
            self.source_bgr = last_bgr.copy()
            self.source_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
            self.original_preview.set_image(self.source_bgr, "BGR")

        all_candidates: list[Candidate] = []
        self._track_lookup = {}
        for frame_idx in sorted(self._tracked_by_frame):
            for tracked in self._tracked_by_frame[frame_idx]:
                all_candidates.append(tracked.candidate)
                self._track_lookup[tracked.candidate.candidate_id] = (
                    tracked.track_id,
                    tracked.persistence_frames,
                )
        self._candidates = tuple(all_candidates)
        self.rescue_results_current = True
        self.analysis_current = True
        self._refresh_candidates()
        self._update_summary_banner()

    def _on_video_completed(self, payload: dict[str, object]) -> None:
        self._poll_video_progress()
        self._apply_video_payload(payload)
        if self.video_info and self.video_info.frame_count:
            self.progress_bar.setRange(0, self.video_info.frame_count)
            self.progress_bar.setValue(len(self.rescue_frame_records))
        self.progress_bar.setFormat("Complete")
        self._set_status(
            f"Analyzed {len(self.rescue_frame_records):,} video frame(s); "
            f"{len(self._candidates):,} frame candidate instance(s) ready for human review."
        )

    def _on_video_cancelled(self, payload: dict[str, object]) -> None:
        self._poll_video_progress()
        self._apply_video_payload(payload)
        self.progress_bar.setFormat("Cancelled · partial results")
        self._set_status(
            f"Video analysis cancelled after {len(self.rescue_frame_records):,} frame(s). "
            "Partial frame candidates and tracks are retained for review."
        )

    def _on_video_worker_finished(self) -> None:
        self._poll_video_progress()
        self._poll_timer.stop()
        worker = self.sender()
        if worker is self._video_worker:
            self._video_worker = None
            worker.deleteLater()
        self._set_busy(False)

    def _update_summary_banner(self) -> None:
        if not self._has_results():
            self._set_result_state("No analysis result is current.")
            return
        is_current = (
            self.analysis_current
            if self.input_mode == "image"
            else self.rescue_results_current
        )
        person_count = sum(
            1 for item in self._candidates if item.category == "person-like"
        )
        floating_count = sum(
            1 for item in self._candidates if item.category == "floating-object-like"
        )
        unknown_count = sum(
            1 for item in self._candidates if item.category == "unknown"
        )
        breakdown = (
            f"Potential Person: {person_count} · "
            f"Floating Object: {floating_count} · "
            f"Unknown Candidate: {unknown_count}"
        )
        if self.input_mode == "video":
            unique_tracks = len({ info[0] for info in self._track_lookup.values() })
            persistent_tracks = len(
                {
                    info[0]
                    for info in self._track_lookup.values()
                    if info[1] >= 2
                }
            )
            summary = (
                f"Video results ({len(self.rescue_frame_records)} frames) · "
                f"Frame Candidates: {len(self._candidates)} ({breakdown}) · "
                f"Experimental Tracks: {unique_tracks} ({persistent_tracks} persistent across 2+ frames)."
            )
        else:
            summary = (
                f"Image results · {len(self._candidates)} candidate(s) "
                f"({breakdown}) · Requires Human Review."
            )
        if not is_current:
            self._set_result_state(
                f"Stale results · Settings changed since last run. Click Analyze to refresh. ({summary})",
                stale=True,
            )
        else:
            self._set_result_state(summary, stale=False)

    def _visible_candidates(self) -> tuple[Candidate, ...]:
        selected_category = self.category_filter.currentData()
        candidates = self._candidates
        if self.input_mode == "video":
            scope = self.video_scope_selector.currentData()
            if scope == "current_frame" and self.current_frame_index is not None:
                candidates = tuple(
                    item
                    for item in candidates
                    if item.frame_index == self.current_frame_index
                )
            elif scope == "tracked_only":
                candidates = tuple(
                    item
                    for item in candidates
                    if self._track_lookup.get(item.candidate_id, ("", 1))[1] >= 2
                )
        if selected_category == "all":
            return candidates
        return tuple(
            candidate
            for candidate in candidates
            if candidate.category == selected_category
        )

    def _frame_candidates_for_preview(self) -> tuple[Candidate, ...]:
        visible = self._visible_candidates()
        if self.input_mode == "video" and self.current_frame_index is not None:
            return tuple(
                item
                for item in visible
                if item.frame_index == self.current_frame_index
            )
        return visible

    def _render_candidate_preview(self) -> None:
        if self.source_rgb is None or not self._has_results():
            return
        preview_candidates = self._frame_candidates_for_preview()
        self.candidate_preview.set_image(
            draw_candidate_boxes_rgb(
                self.source_rgb,
                preview_candidates,
                selected_candidate_id=self._selected_candidate_id,
            ),
            "RGB",
        )

    def _refresh_candidates(self, *_args) -> None:
        candidates = self._visible_candidates()
        self.candidate_table.blockSignals(True)
        self.candidate_table.setRowCount(len(candidates))
        selected_row = None
        for row, candidate in enumerate(candidates):
            if self.input_mode == "video" and candidate.frame_index is not None:
                track_id, persistence = self._track_lookup.get(
                    candidate.candidate_id, ("—", 1)
                )
                context_text = f"F{candidate.frame_index + 1} · {track_id} ({persistence}f)"
            else:
                context_text = f"Score {candidate.rule_score:.2f}"
            values = (
                f"C{row + 1} ({candidate.candidate_id.split('#')[-1]})",
                format_category_label(candidate.category),
                context_text,
                f"{candidate.area_pixels:,}",
                format_review_status(candidate.review_status),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, candidate.candidate_id)
                self.candidate_table.setItem(row, column, item)
            if candidate.candidate_id == self._selected_candidate_id:
                selected_row = row
        if selected_row is not None:
            self.candidate_table.selectRow(selected_row)
        self.candidate_table.blockSignals(False)
        if selected_row is None:
            self._selected_candidate_id = None
            self.selected_candidate_label.setText("No candidate selected")
            self.reasoning_label.setText(
                "Select a candidate in the list to highlight its region and inspect its visual reasoning."
            )
            self.feature_details.setText("No candidate selected.")
            self.review_status.setEnabled(False)
        self._render_candidate_preview()

    def _selection_changed(self) -> None:
        rows = self.candidate_table.selectionModel().selectedRows()
        if not rows:
            self._selected_candidate_id = None
            self._render_candidate_preview()
            return
        item = self.candidate_table.item(rows[0].row(), 0)
        self._selected_candidate_id = item.data(Qt.ItemDataRole.UserRole)
        self._show_selected_candidate()

    def _selected_candidate(self) -> Candidate | None:
        return next(
            (
                candidate
                for candidate in self._candidates
                if candidate.candidate_id == self._selected_candidate_id
            ),
            None,
        )

    def _load_video_frame_for_candidate(self, frame_index: int) -> None:
        if (
            self.input_mode != "video"
            or self.source_path is None
            or self.current_frame_index == frame_index
            or self._is_busy()
        ):
            return
        try:
            with VideoCaptureReader(self.source_path) as reader:
                reader.seek_frame(frame_index)
                frame = reader.read_next()
        except Exception:
            return
        if frame is not None:
            self.current_frame_index = frame.index
            self.source_bgr = frame.image_bgr.copy()
            self.source_rgb = cv2.cvtColor(self.source_bgr, cv2.COLOR_BGR2RGB)
            self.original_preview.set_image(self.source_bgr, "BGR")

    def _show_selected_candidate(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            return
        if self.input_mode == "video" and candidate.frame_index is not None:
            self._load_video_frame_for_candidate(candidate.frame_index)
        self._render_candidate_preview()

        x, y, width, height = candidate.bbox
        category_label = format_category_label(candidate.category)
        if self.input_mode == "video" and candidate.frame_index is not None:
            track_id, persistence = self._track_lookup.get(
                candidate.candidate_id, ("—", 1)
            )
            self.selected_candidate_label.setText(
                f"{category_label} · Frame {candidate.frame_index + 1} · "
                f"Box ({x}, {y}, {width}×{height} px) · "
                f"Experimental {track_id} ({persistence} frame(s))"
            )
        else:
            self.selected_candidate_label.setText(
                f"{category_label} · {candidate.candidate_id} · "
                f"Box ({x}, {y}, {width}×{height} px)"
            )
        self.review_status.blockSignals(True)
        status_index = self.review_status.findData(candidate.review_status)
        self.review_status.setCurrentIndex(max(0, status_index))
        self.review_status.blockSignals(False)
        self.review_status.setEnabled(not self._is_busy())
        self.reasoning_label.setText(describe_candidate_reasoning(candidate))

        features = candidate.features
        water_available = bool(features.get("water_context_available", 0.0) >= 0.5)
        water_line = (
            f"Water overlap: {features.get('water_overlap_ratio', 0.0) * 100:.1f}% · "
            f"Distance to water: {features.get('distance_to_water_px', 0.0):.1f} px"
            if water_available
            else "Water context: not used"
        )
        self.feature_details.setText(
            f"Category: {category_label}\n"
            f"Status: {format_review_status(candidate.review_status)}\n"
            f"Area: {candidate.area_pixels:,} px · Center: ({candidate.center[0]:.1f}, {candidate.center[1]:.1f}) px\n"
            f"Aspect ratio: {features.get('aspect_ratio', 0.0):.2f} · "
            f"Solidity: {features.get('solidity', 0.0):.2f} · "
            f"Compactness: {features.get('compactness', 0.0):.2f}\n"
            f"Local color Δ: {features.get('local_color_contrast', 0.0):.1f} · "
            f"Intensity Δ: {features.get('local_intensity_contrast', 0.0):.1f} · "
            f"Texture Δ: {features.get('texture_difference', 0.0):.1f}\n"
            f"{water_line}\n"
            f"Rule score: {candidate.rule_score:.3f} (heuristic ranking aid, not probability)"
        )

    def _review_status_changed(self, *_args) -> None:
        candidate = self._selected_candidate()
        status = self.review_status.currentData()
        if candidate is None or status is None:
            return
        self._candidates = tuple(
            replace(item, review_status=status)
            if item.candidate_id == candidate.candidate_id
            else item
            for item in self._candidates
        )
        if self.input_mode == "video":
            for frame_idx, tracked_tuple in list(self._tracked_by_frame.items()):
                self._tracked_by_frame[frame_idx] = tuple(
                    replace(
                        tracked,
                        candidate=replace(tracked.candidate, review_status=status),
                    )
                    if tracked.candidate.candidate_id == candidate.candidate_id
                    else tracked
                    for tracked in tracked_tuple
                )
            for record in self.rescue_candidate_records:
                if record.get("candidate_id") == candidate.candidate_id:
                    record["review_status"] = status
                    record["review_status_label"] = format_review_status(status)
        self._refresh_candidates()
        self._show_selected_candidate()

    def _mark_analysis_stale(self, *_args) -> None:
        if not self._has_results():
            return
        try:
            config = self.rescue_controls.config()
        except (TypeError, ValueError):
            config = None
        if self.input_mode == "video":
            self.rescue_results_current = (
                config == self.rescue_config
                and self.water_context_checkbox.isChecked()
                == self.rescue_use_water_context
            )
            self.analysis_current = self.rescue_results_current
        else:
            assert self.analysis is not None
            self.analysis_current = (
                config == self.analysis.config
                and self.water_context_checkbox.isChecked()
                == self.analysis.water_context_used
            )
        self.export_csv_button.setEnabled(self.analysis_current)
        self.export_json_button.setEnabled(self.analysis_current)
        self._update_summary_banner()
        self._set_status(
            "Candidate settings changed; results are stale. Click Analyze before exporting."
            if not self.analysis_current
            else "Candidate settings match the displayed analysis."
        )

    def export_csv(self) -> None:
        if self.source_path is None or not self.analysis_current:
            return
        suffix = (
            "_candidate_tracks.csv"
            if self.input_mode == "video"
            else "_candidates.csv"
        )
        selected = choose_save_path(
            self,
            title="Export review candidates to CSV",
            suggested_path=self.source_path.with_name(
                f"{self.source_path.stem}{suffix}"
            ),
            file_filter="CSV file (*.csv)",
            default_suffix=".csv",
        )
        if selected is None:
            return
        path, overwrite = selected
        try:
            if self.input_mode == "video":
                export_video_candidates_csv(
                    path,
                    source_video=self.source_path,
                    config=self.rescue_config,
                    candidate_records=self.rescue_candidate_records,
                    water_context_used=self.rescue_use_water_context,
                    water_context_config=self.rescue_water_config,
                    overwrite=overwrite,
                )
            else:
                if self.analysis is None:
                    return
                export_image_candidates_csv(
                    path,
                    source_path=self.source_path,
                    analysis=self.analysis,
                    candidates=self._candidates,
                    water_context_config=self.water_context_config,
                    overwrite=overwrite,
                )
            self._set_status(f"Candidate CSV exported to {path}")
        except (OSError, ValueError) as exc:
            self._on_error(str(exc))

    def export_json(self) -> None:
        if self.source_path is None or not self.analysis_current:
            return
        suffix = (
            "_candidate_tracks.json"
            if self.input_mode == "video"
            else "_candidates.json"
        )
        selected = choose_save_path(
            self,
            title="Export review candidates to JSON",
            suggested_path=self.source_path.with_name(
                f"{self.source_path.stem}{suffix}"
            ),
            file_filter="JSON file (*.json)",
            default_suffix=".json",
        )
        if selected is None:
            return
        path, overwrite = selected
        try:
            if self.input_mode == "video":
                export_video_candidates_json(
                    path,
                    source_video=self.source_path,
                    config=self.rescue_config,
                    frame_records=self.rescue_frame_records,
                    candidate_records=self.rescue_candidate_records,
                    water_context_used=self.rescue_use_water_context,
                    water_context_config=self.rescue_water_config,
                    overwrite=overwrite,
                )
            else:
                if self.analysis is None:
                    return
                export_image_candidates_json(
                    path,
                    source_path=self.source_path,
                    analysis=self.analysis,
                    candidates=self._candidates,
                    water_context_config=self.water_context_config,
                    overwrite=overwrite,
                )
            self._set_status(f"Candidate JSON exported to {path}")
        except (OSError, ValueError) as exc:
            self._on_error(str(exc))

    def shutdown(self) -> None:
        """Cancel and wait for active image or video workers before destruction."""
        self._poll_timer.stop()
        if self._video_worker is not None:
            if self._video_worker.isRunning():
                self._video_worker.cancel()
                self._video_worker.wait()
            self._video_worker.deleteLater()
            self._video_worker = None
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()
