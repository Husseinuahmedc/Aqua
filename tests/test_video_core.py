"""Tests for sequential video reading and processing."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.video.analysis import iter_video_analysis
from core.video.reader import VideoCaptureReader, get_video_info, iter_video_frames
from core.water.segmentation import SegmentationConfig


def create_test_video(path: Path) -> bool:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (24, 20),
    )
    if not writer.isOpened():
        writer.release()
        return False
    first = np.full((20, 24, 3), (0, 0, 255), dtype=np.uint8)
    first[4:12, 5:13] = (255, 0, 0)
    second = first.copy()
    second[4:12, 5:13] = (0, 255, 0)
    second[10:17, 15:22] = (255, 0, 0)
    writer.write(first)
    writer.write(second)
    writer.release()
    return True


def analysis_config() -> SegmentationConfig:
    return SegmentationConfig(
        hue_min=110,
        hue_max=130,
        saturation_min=200,
        saturation_max=255,
        value_min=100,
        value_max=255,
        blur_method="none",
        morphology_kernel_size=1,
        morphology_iterations=0,
        min_region_area=0,
    )


def test_video_reader_and_frame_analysis_are_sequential(tmp_path: Path) -> None:
    video_path = tmp_path / "short_clip.avi"
    if not create_test_video(video_path):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")

    info = get_video_info(video_path)
    frames = list(iter_video_frames(video_path))
    analyses = list(iter_video_analysis(video_path, analysis_config()))
    with VideoCaptureReader(video_path) as reader:
        first = reader.read_next()
        reader.seek_frame(1)
        sought = reader.read_next()

    assert info.width == 24
    assert info.height == 20
    assert info.frame_count == 2
    assert info.fps == pytest.approx(10.0)
    assert [frame.index for frame in frames] == [0, 1]
    assert frames[0].image_bgr.shape == (20, 24, 3)
    assert frames[0].timestamp_ms == pytest.approx(0.0)
    assert frames[1].timestamp_ms == pytest.approx(100.0)
    assert [frame.frame_index for frame in analyses] == [0, 1]
    assert analyses[0].comparison_to_previous is None
    assert analyses[1].comparison_to_previous is not None
    assert analyses[1].comparison_to_previous.changed_pixels > 0
    assert first is not None and first.index == 0
    assert sought is not None and sought.index == 1


def test_video_analysis_cancels_between_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "cancel_clip.avi"
    if not create_test_video(video_path):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks > 1

    analyses = list(
        iter_video_analysis(
            video_path,
            analysis_config(),
            should_cancel=should_cancel,
        )
    )

    assert len(analyses) == 1
    assert analyses[0].frame_index == 0


def test_video_reader_reports_missing_or_invalid_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / "missing.mp4")

    invalid_video = tmp_path / "invalid.mp4"
    invalid_video.write_text("not a video", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not open video"):
        get_video_info(invalid_video)
