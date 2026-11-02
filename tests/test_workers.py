"""Tests for GUI-safe asynchronous image and video worker wrappers."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtTest import QSignalSpy

from app.workers import (
    ImageTaskWorker,
    RescueTask,
    RescueVideoAnalysisWorker,
    RescueOutput,
    SegmentTask,
    SegmentationOutput,
    VideoAnalysisWorker,
    VideoPlaybackWorker,
)
from core.water.segmentation import SegmentationConfig
from core.rescue.pipeline import RescueConfig


def _write_video(path: Path, frame_count: int = 2) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (24, 20),
    )
    if not writer.isOpened():
        writer.release()
        return False
    for index in range(frame_count):
        frame = np.full((20, 24, 3), (20, 40, 80), dtype=np.uint8)
        frame[2 : 5 + index % 3, 2:8] = (index, 120, 200)
        writer.write(frame)
    writer.release()
    return True


def _config() -> SegmentationConfig:
    return SegmentationConfig(
        hue_min=0,
        hue_max=179,
        saturation_min=0,
        saturation_max=255,
        value_min=0,
        value_max=255,
        blur_method="none",
        morphology_kernel_size=1,
        morphology_iterations=0,
        min_region_area=0,
    )


def _wait_for_count(
    spy: QSignalSpy,
    signal,
    expected_count: int = 1,
    timeout_ms: int = 10_000,
) -> bool:
    """Pump a Qt event loop until a signal arrives or the timeout expires."""
    if spy.count() >= expected_count:
        return True
    event_loop = QEventLoop()
    signal.connect(event_loop.quit)
    QTimer.singleShot(timeout_ms, event_loop.quit)
    while spy.count() < expected_count:
        event_loop.exec()
        if spy.count() < expected_count:
            return False
    try:
        signal.disconnect(event_loop.quit)
    except RuntimeError:
        pass
    return spy.count() > 0


def test_image_worker_emits_result_off_the_calling_thread() -> None:
    image = np.full((18, 22, 3), (30, 40, 50), dtype=np.uint8)
    worker = ImageTaskWorker(SegmentTask(image.copy(), _config()))
    result_spy = QSignalSpy(worker.result_ready)
    error_spy = QSignalSpy(worker.error_occurred)

    worker.start()

    assert _wait_for_count(result_spy, worker.result_ready)
    assert worker.wait(10_000)
    assert error_spy.count() == 0
    output = result_spy.at(0)[0]
    assert isinstance(output, SegmentationOutput)
    assert output.result.mask.shape == image.shape[:2]


def test_image_worker_reports_invalid_input_as_signal() -> None:
    invalid_image = np.zeros((4, 4), dtype=np.uint8)
    worker = ImageTaskWorker(
        SegmentTask(invalid_image, _config())  # type: ignore[arg-type]
    )
    error_spy = QSignalSpy(worker.error_occurred)

    worker.start()

    assert _wait_for_count(error_spy, worker.error_occurred)
    assert worker.wait(10_000)
    assert "shape" in str(error_spy.at(0)[0]).lower()


def test_rescue_image_worker_runs_separate_candidate_pipeline() -> None:
    image = np.full((40, 48, 3), 100, dtype=np.uint8)
    image[12:25, 18:30] = (20, 30, 220)
    worker = ImageTaskWorker(
        RescueTask(
            image.copy(),
            "worker-image",
            config=RescueConfig(methods=("color", "intensity")),
        )
    )
    result_spy = QSignalSpy(worker.result_ready)
    error_spy = QSignalSpy(worker.error_occurred)

    worker.start()

    assert _wait_for_count(result_spy, worker.result_ready)
    assert worker.wait(10_000)
    assert error_spy.count() == 0
    output = result_spy.at(0)[0]
    assert isinstance(output, RescueOutput)
    assert output.analysis.candidates
    assert output.candidate_overlay_rgb.shape == image.shape


def test_rescue_worker_records_optional_water_context_settings() -> None:
    image = np.full((36, 42, 3), (180, 80, 20), dtype=np.uint8)
    worker = ImageTaskWorker(
        RescueTask(
            image,
            "context-image",
            use_water_context=True,
            config=RescueConfig(methods=("color",)),
        )
    )
    result_spy = QSignalSpy(worker.result_ready)
    error_spy = QSignalSpy(worker.error_occurred)

    worker.start()

    assert _wait_for_count(result_spy, worker.result_ready)
    assert worker.wait(10_000)
    assert error_spy.count() == 0
    output = result_spy.at(0)[0]
    assert isinstance(output, RescueOutput)
    assert output.analysis.water_context_used
    assert output.water_context_config == SegmentationConfig()


def test_rescue_video_worker_returns_frame_and_track_records(tmp_path: Path) -> None:
    path = tmp_path / "rescue_worker_clip.avi"
    if not _write_video(path):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")
    worker = RescueVideoAnalysisWorker(
        path,
        RescueConfig(methods=("color", "intensity")),
    )
    completed_spy = QSignalSpy(worker.completed)
    error_spy = QSignalSpy(worker.error_occurred)

    worker.start()

    assert _wait_for_count(completed_spy, worker.completed)
    assert worker.wait(10_000)
    assert error_spy.count() == 0
    result = completed_spy.at(0)[0]
    assert [frame["frame_index"] for frame in result["frames"]] == [0, 1]
    assert all("candidate_count" in frame for frame in result["frames"])


def test_video_playback_worker_supports_pause_seek_and_stop(tmp_path: Path) -> None:
    path = tmp_path / "worker_clip.avi"
    if not _write_video(path):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")

    worker = VideoPlaybackWorker(path)
    info_spy = QSignalSpy(worker.info_ready)
    frame_spy = QSignalSpy(worker.frame_ready)
    error_spy = QSignalSpy(worker.error_occurred)
    worker.start()

    try:
        assert _wait_for_count(info_spy, worker.info_ready)
        assert _wait_for_count(frame_spy, worker.frame_ready)
        first_frame = frame_spy.at(0)[0]
        assert first_frame.index == 0

        worker.acknowledge_frame()
        worker.seek(1)
        assert _wait_for_count(frame_spy, worker.frame_ready, expected_count=2)
        second_frame = frame_spy.at(1)[0]
        assert second_frame.index == 1
        assert error_spy.count() == 0
    finally:
        worker.acknowledge_frame()
        worker.stop()
        worker.wait(10_000)


def test_video_analysis_worker_cancels_and_returns_partial_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "cancel_worker_clip.avi"
    if not _write_video(path, frame_count=60):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")
    worker = VideoAnalysisWorker(path, _config())
    cancelled_spy = QSignalSpy(worker.cancelled)
    event_loop = QEventLoop()
    check_timer = QTimer()
    check_timer.setInterval(1)

    def request_cancel_after_frames() -> None:
        progress = worker.take_latest_progress()
        if progress is not None and progress.frame_index >= 2:
            worker.cancel()

    check_timer.timeout.connect(request_cancel_after_frames)
    worker.cancelled.connect(event_loop.quit)
    check_timer.start()
    worker.start()
    QTimer.singleShot(10_000, event_loop.quit)
    event_loop.exec()
    check_timer.stop()

    assert worker.wait(10_000)
    assert cancelled_spy.count() == 1
    rows = cancelled_spy.at(0)[0]
    assert 1 <= len(rows) < 60


def test_rescue_video_worker_cancels_cooperatively(tmp_path: Path) -> None:
    path = tmp_path / "cancel_rescue_clip.avi"
    if not _write_video(path, frame_count=40):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")
    worker = RescueVideoAnalysisWorker(
        path,
        RescueConfig(methods=("color",)),
    )
    cancelled_spy = QSignalSpy(worker.cancelled)
    event_loop = QEventLoop()
    check_timer = QTimer()
    check_timer.setInterval(1)

    def request_cancel() -> None:
        progress = worker.take_latest_progress()
        if progress is not None and progress.frame_index >= 1:
            worker.cancel()

    check_timer.timeout.connect(request_cancel)
    worker.cancelled.connect(event_loop.quit)
    check_timer.start()
    worker.start()
    QTimer.singleShot(10_000, event_loop.quit)
    event_loop.exec()
    check_timer.stop()

    assert worker.wait(10_000)
    assert cancelled_spy.count() == 1
    payload = cancelled_spy.at(0)[0]
    assert 1 <= len(payload["frames"]) < 40
