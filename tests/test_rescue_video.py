"""Tests for motion compensation and candidate association on video frames."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.rescue.candidates import Candidate
from core.rescue.pipeline import RescueConfig
from core.rescue.video import (
    CandidateTracker,
    estimate_camera_affine,
    iter_video_candidate_analysis,
)


def _candidate(x: int, y: int = 15) -> Candidate:
    return Candidate(
        candidate_id=f"frame-{x}",
        image_id="clip",
        frame_index=x,
        bbox=(x, y, 10, 12),
        center=(x + 5.0, y + 6.0),
        area_pixels=120,
        category="unknown",
        features={},
        rule_score=0.4,
        source_methods=("color",),
    )


def _write_video(path: Path) -> bool:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (80, 60)
    )
    if not writer.isOpened():
        writer.release()
        return False
    for x in (15, 17, 19):
        frame = np.full((60, 80, 3), (80, 90, 100), dtype=np.uint8)
        cv2.rectangle(frame, (x, 20), (x + 12, 38), (220, 40, 20), -1)
        writer.write(frame)
    writer.release()
    return True


def test_optical_flow_estimates_global_translation() -> None:
    rng = np.random.default_rng(17)
    previous = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
    previous = cv2.GaussianBlur(previous, (5, 5), 0)
    affine = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]], dtype=np.float32)
    current = cv2.warpAffine(previous, affine, (160, 120))

    estimated = estimate_camera_affine(previous, current)

    assert estimated is not None
    assert estimated[0, 2] == pytest.approx(3.0, abs=1.0)
    assert estimated[1, 2] == pytest.approx(2.0, abs=1.0)


def test_candidate_tracker_uses_camera_transform_and_tolerates_short_occlusion() -> None:
    tracker = CandidateTracker(max_centroid_distance_px=20, max_missed_frames=1)
    first = tracker.update(
        0, (_candidate(10),), image_shape=(80, 100), previous_to_current=None
    )
    camera_motion = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    second = tracker.update(
        1,
        (_candidate(15),),
        image_shape=(80, 100),
        previous_to_current=camera_motion,
    )
    tracker.update(2, (), image_shape=(80, 100), previous_to_current=None)
    third = tracker.update(
        3, (_candidate(15),), image_shape=(80, 100), previous_to_current=None
    )

    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert second[0].persistence_frames == 2
    assert third[0].persistence_frames == 3


def test_framewise_video_candidates_keep_order_and_context_optional(
    tmp_path: Path,
) -> None:
    path = tmp_path / "search video.avi"
    if not _write_video(path):
        pytest.skip("Installed OpenCV backend cannot create MJPG test videos.")

    frames = list(
        iter_video_candidate_analysis(
            path,
            config=RescueConfig(methods=("color", "intensity"), min_area_pixels=8),
        )
    )

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert all(frame.image_bgr.shape == (60, 80, 3) for frame in frames)
    assert all(frame.processing_time_ms >= 0 for frame in frames)
    assert all(not frame.motion_compensated or frame.frame_index > 0 for frame in frames)
    assert all(
        item.candidate.review_status == "needs_review"
        for frame in frames
        for item in frame.tracked_candidates
    )


def test_pipeline_handles_a_single_pixel_image() -> None:
    from core.rescue.pipeline import analyze_rescue_image

    result = analyze_rescue_image(np.zeros((1, 1, 3), dtype=np.uint8))

    assert (result.image_width, result.image_height) == (1, 1)
    assert result.candidates == ()
