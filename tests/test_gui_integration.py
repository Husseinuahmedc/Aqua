"""End-to-end workspace checks using local synthetic image/video fixtures."""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer

from app.window import AquaRescueWindow
from core.preprocessing.image_io import save_image


def wait_until(predicate, timeout_ms: int = 10_000) -> bool:
    """Process the Qt event queue until a condition is true or times out."""
    if predicate():
        return True
    event_loop = QEventLoop()
    poll_timer = QTimer()
    poll_timer.setInterval(10)
    poll_timer.timeout.connect(lambda: event_loop.quit() if predicate() else None)
    timeout_timer = QTimer()
    timeout_timer.setSingleShot(True)
    timeout_timer.timeout.connect(event_loop.quit)
    poll_timer.start()
    timeout_timer.start(timeout_ms)
    event_loop.exec()
    poll_timer.stop()
    timeout_timer.stop()
    return predicate()


def _write_video(path: Path) -> bool:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    if not writer.isOpened():
        writer.release()
        return False
    first = np.full((24, 32, 3), (0, 0, 255), dtype=np.uint8)
    first[5:18, 6:20] = (255, 0, 0)
    second = first.copy()
    second[5:18, 6:20] = (0, 255, 0)
    second[12:22, 20:30] = (255, 0, 0)
    writer.write(first)
    writer.write(second)
    writer.release()
    return True


def test_image_workspace_loads_segments_and_evaluates_manual_mask(
    tmp_path: Path,
) -> None:
    source = np.full((32, 36, 3), (0, 0, 255), dtype=np.uint8)
    source[8:24, 9:25] = (255, 0, 0)
    image_path = tmp_path / "research image Δ.png"
    ok, encoded = cv2.imencode(".png", source)
    assert ok
    image_path.write_bytes(encoded.tobytes())

    window = AquaRescueWindow()
    try:
        image = window.image_workspace
        image.open_image(image_path)
        assert wait_until(
            lambda: image.source_bgr is not None and image._worker is None
        )
        assert image.source_info.width == 36
        assert image.source_info.height == 32

        image.compute_histogram()
        assert wait_until(
            lambda: image.histogram_data is not None and image._worker is None
        )
        assert int(image.histogram_data.counts.sum()) == (
            source.shape[0] * source.shape[1]
        )

        image.extract_candidates_checkbox.setChecked(True)
        image.run_segmentation()
        assert wait_until(
            lambda: image.segmentation is not None and image._worker is None
        )
        assert image.segmentation.result.mask.shape == source.shape[:2]
        assert image.segmentation.water_analysis.water_pixels > 0
        assert image.segmentation.candidates
        assert image.segmentation.candidate_overlay_bgr is not None
        assert "not people" in image.candidate_details.text().lower()
        assert window.evaluation_workspace.predicted_mask is not None

        image.segmentation_controls.saturation_min.setValue(31)
        assert not image.segmentation_current
        assert "stale" in image.result_state_label.text().lower()
        assert not image.export_panel.save_mask_button.isEnabled()
        assert window.evaluation_workspace.predicted_mask is None
        image.segmentation_controls.saturation_min.setValue(30)
        assert image.segmentation_current
        assert image.export_panel.save_mask_button.isEnabled()
        assert window.evaluation_workspace.predicted_mask is not None

        reference_path = tmp_path / "manual water mask.png"
        save_image(
            reference_path,
            image.segmentation.result.mask,
            color_space="GRAY",
        )
        evaluation = window.evaluation_workspace
        evaluation.load_reference_mask(reference_path)
        assert wait_until(
            lambda: evaluation.reference_mask is not None and evaluation._worker is None
        )
        evaluation.evaluate()
        assert wait_until(
            lambda: evaluation.metrics is not None and evaluation._worker is None
        )
        assert evaluation.metrics.intersection_over_union == 1.0
        assert evaluation.metrics.dice_coefficient == 1.0

        rescue = window.rescue_workspace
        rescue.open_image(image_path)
        assert wait_until(lambda: rescue.source_bgr is not None and rescue._worker is None)
        assert rescue.clear_button.isEnabled()
        assert rescue.advanced_panel.isHidden()
        rescue.advanced_toggle_button.setChecked(True)
        assert not rescue.advanced_panel.isHidden()
        rescue.run_analysis()
        assert wait_until(lambda: rescue.analysis is not None and rescue._worker is None)
        assert rescue.analysis.image_width == 36
        assert rescue.analysis.image_height == 32
        assert rescue.candidate_table.rowCount() == len(rescue.analysis.candidates)
        assert rescue.candidate_preview._pixmap is not None
        assert rescue.analysis.candidates
        rescue.candidate_table.selectRow(0)
        assert rescue.review_status.isEnabled()
        assert "heuristic" in rescue.reasoning_label.text().lower()
        rescue.review_status.setCurrentIndex(rescue.review_status.findData("confirmed"))
        assert any(
            candidate.review_status == "confirmed"
            for candidate in rescue._candidates
        )
        rescue.rescue_controls.min_area.setValue(10)
        assert not rescue.analysis_current
        assert not rescue.export_json_button.isEnabled()
        rescue.rescue_controls.reset_defaults()
        assert rescue.analysis_current
        assert rescue.export_json_button.isEnabled()
        rescue.clear_input()
        assert rescue.source_bgr is None
        assert not rescue.run_button.isEnabled()
    finally:
        window.close()


def test_video_workspace_loads_navigates_processes_and_analyzes(tmp_path: Path) -> None:
    video_path = tmp_path / "short research clip.avi"
    if not _write_video(video_path):
        import pytest

        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")

    window = AquaRescueWindow()
    try:
        video = window.video_workspace
        video.open_video(video_path)
        assert wait_until(
            lambda: video.video_info is not None and video.current_frame_bgr is not None
        )
        assert video.current_frame_index == 0
        assert not video._playing

        video.process_current_frame()
        assert wait_until(
            lambda: video.current_mask is not None and video._frame_worker is None
        )
        assert video.current_mask.shape == (24, 32)

        heartbeat_count = 0
        heartbeat = QTimer()
        heartbeat.setInterval(1)

        def process_gui_heartbeat() -> None:
            nonlocal heartbeat_count
            heartbeat_count += 1

        heartbeat.timeout.connect(process_gui_heartbeat)
        heartbeat.start()
        video.analyze_video()
        assert wait_until(
            lambda: bool(video.analysis_records) and video._analysis_worker is None
        )
        heartbeat.stop()
        assert heartbeat_count > 0
        assert len(video.analysis_records) == 2
        assert video.coverage_plot.axes.lines
        assert window.evaluation_workspace.predicted_mask is not None

        video.segmentation_controls.hue_min.setValue(81)
        assert not video.analysis_results_current
        assert not video.export_panel.export_metrics_button.isEnabled()
        assert "stale" in video.result_state_label.text().lower()
        video.analyze_video()
        assert video.analysis_records == []
        assert not video.export_panel.export_metrics_button.isEnabled()
        assert wait_until(
            lambda: bool(video.analysis_records) and video._analysis_worker is None
        )
        assert video.analysis_results_current
        assert video.analysis_config.hue_min == 81
        assert video.export_panel.export_metrics_button.isEnabled()

        rescue = window.rescue_workspace
        rescue.open_video(video_path)
        assert rescue.input_mode == "video"
        assert rescue.video_info is not None
        rescue.run_analysis()
        assert wait_until(
            lambda: rescue._video_worker is None
            and len(rescue.rescue_frame_records) == 2
        )
        assert rescue.rescue_results_current
        assert [row["frame_index"] for row in rescue.rescue_frame_records] == [0, 1]
        assert rescue.candidate_preview._pixmap is not None
        assert rescue.export_json_button.isEnabled()
        assert rescue.video_scope_selector.isEnabled()
    finally:
        window.close()
