"""Frame-by-frame candidate tracking with optical-flow camera compensation."""

from dataclasses import dataclass
import math
from math import hypot
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from core.rescue.candidates import Candidate, CandidateCategory, box_iou
from core.rescue.pipeline import RescueConfig, analyze_rescue_image
from core.video.reader import VideoFrame, iter_video_frames
from core.water.segmentation import SegmentationConfig, segment_water


@dataclass(frozen=True)
class TrackedCandidate:
    """One frame candidate associated with a persistent review track."""

    candidate: Candidate
    track_id: str
    persistence_frames: int


@dataclass(frozen=True)
class VideoCandidateFrame:
    """One current-frame output; consumers should not retain all frame arrays."""

    frame_index: int
    timestamp_ms: float | None
    image_bgr: NDArray[np.uint8]
    tracked_candidates: tuple[TrackedCandidate, ...]
    active_track_count: int
    motion_compensated: bool
    processing_time_ms: float


@dataclass
class _TrackState:
    track_id: str
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    category: CandidateCategory
    hits: int
    missed_frames: int
    last_frame_index: int


def estimate_camera_affine(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    max_corners: int = 400,
    min_tracked_points: int = 8,
    ransac_threshold: float = 3.0,
) -> NDArray[np.float32] | None:
    """Estimate previous-to-current camera motion with sparse Lucas–Kanade flow.

    ``None`` means there were not enough stable features for a defensible
    global affine estimate; callers should then associate using raw image
    coordinates and record that compensation was unavailable.
    """
    for image in (previous_gray, current_gray):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 2:
            raise TypeError("Optical-flow inputs must be uint8 grayscale images.")
        if image.size == 0:
            raise ValueError("Optical-flow inputs must not be empty.")
    if previous_gray.shape != current_gray.shape:
        raise ValueError("Optical-flow frames must have matching dimensions.")
    if (
        isinstance(max_corners, bool)
        or isinstance(min_tracked_points, bool)
        or not isinstance(max_corners, int)
        or not isinstance(min_tracked_points, int)
        or max_corners < min_tracked_points
        or min_tracked_points < 3
    ):
        raise ValueError("Optical-flow corner limits are invalid.")
    if not math.isfinite(ransac_threshold) or ransac_threshold <= 0:
        raise ValueError("RANSAC reprojection threshold must be positive.")

    points = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if points is None or len(points) < min_tracked_points:
        return None
    tracked, status, _errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return None
    valid = status.reshape(-1) == 1
    source_points = points.reshape(-1, 2)[valid]
    target_points = tracked.reshape(-1, 2)[valid]
    if len(source_points) < min_tracked_points:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        source_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is None or inliers is None or int(inliers.sum()) < min_tracked_points:
        return None
    return matrix.astype(np.float32)


def _transform_box(
    bbox: tuple[int, int, int, int],
    matrix: NDArray[np.float32] | None,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    if matrix is None:
        return bbox
    corners = np.array(
        [[x, y], [x + width, y], [x, y + height], [x + width, y + height]],
        dtype=np.float32,
    )
    transformed = corners @ matrix[:, :2].T + matrix[:, 2]
    image_height, image_width = image_shape
    left = max(0, int(np.floor(transformed[:, 0].min())))
    top = max(0, int(np.floor(transformed[:, 1].min())))
    right = min(image_width, int(np.ceil(transformed[:, 0].max())))
    bottom = min(image_height, int(np.ceil(transformed[:, 1].max())))
    return left, top, max(0, right - left), max(0, bottom - top)


class CandidateTracker:
    """Greedy centroid/IoU association with motion-compensated prior boxes."""

    def __init__(
        self,
        *,
        max_centroid_distance_px: float = 60.0,
        min_iou: float = 0.05,
        max_missed_frames: int = 2,
    ) -> None:
        if (
            not math.isfinite(max_centroid_distance_px)
            or max_centroid_distance_px <= 0
            or not math.isfinite(min_iou)
            or not 0.0 <= min_iou <= 1.0
        ):
            raise ValueError("Invalid candidate tracking association thresholds.")
        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, int)
            or max_missed_frames < 0
        ):
            raise ValueError("Maximum missed frames must be a non-negative integer.")
        self.max_centroid_distance_px = max_centroid_distance_px
        self.min_iou = min_iou
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[str, _TrackState] = {}
        self._next_track_number = 1

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def update(
        self,
        frame_index: int,
        candidates: tuple[Candidate, ...],
        *,
        image_shape: tuple[int, int],
        previous_to_current: NDArray[np.float32] | None,
    ) -> tuple[TrackedCandidate, ...]:
        """Associate detections once each and keep short occlusions alive."""
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative.")
        predictions = {
            track_id: _transform_box(track.bbox, previous_to_current, image_shape)
            for track_id, track in self._tracks.items()
        }
        candidate_centers = [candidate.center for candidate in candidates]
        pairs: list[tuple[float, str, int, float, float]] = []
        for track_id, predicted_box in predictions.items():
            track = self._tracks[track_id]
            x, y, width, height = predicted_box
            predicted_center = (x + width / 2.0, y + height / 2.0)
            for candidate_index, candidate in enumerate(candidates):
                if (
                    track.category != candidate.category
                    and track.category != "unknown"
                    and candidate.category != "unknown"
                ):
                    continue
                distance = hypot(
                    predicted_center[0] - candidate_centers[candidate_index][0],
                    predicted_center[1] - candidate_centers[candidate_index][1],
                )
                overlap = box_iou(predicted_box, candidate.bbox)
                if overlap < self.min_iou and distance > self.max_centroid_distance_px:
                    continue
                association_score = overlap + 0.25 * max(
                    0.0, 1.0 - distance / self.max_centroid_distance_px
                )
                pairs.append(
                    (association_score, track_id, candidate_index, distance, overlap)
                )
        pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
        matched_tracks: set[str] = set()
        matched_candidates: set[int] = set()
        assignments: dict[int, _TrackState] = {}
        for _score, track_id, candidate_index, _distance, _overlap in pairs:
            if track_id in matched_tracks or candidate_index in matched_candidates:
                continue
            track = self._tracks[track_id]
            candidate = candidates[candidate_index]
            track.bbox = candidate.bbox
            track.center = candidate.center
            track.category = candidate.category
            track.hits += 1
            track.missed_frames = 0
            track.last_frame_index = frame_index
            matched_tracks.add(track_id)
            matched_candidates.add(candidate_index)
            assignments[candidate_index] = track

        for track_id, track in tuple(self._tracks.items()):
            if track_id not in matched_tracks:
                track.bbox = predictions[track_id]
                x, y, width, height = track.bbox
                track.center = (x + width / 2.0, y + height / 2.0)
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched_candidates:
                continue
            track_id = f"track-{self._next_track_number:06d}"
            self._next_track_number += 1
            track = _TrackState(
                track_id=track_id,
                bbox=candidate.bbox,
                center=candidate.center,
                category=candidate.category,
                hits=1,
                missed_frames=0,
                last_frame_index=frame_index,
            )
            self._tracks[track_id] = track
            assignments[candidate_index] = track

        return tuple(
            TrackedCandidate(
                candidate=candidate,
                track_id=assignments[index].track_id,
                persistence_frames=assignments[index].hits,
            )
            for index, candidate in enumerate(candidates)
        )


def iter_video_candidate_analysis(
    path: Path | str,
    *,
    config: RescueConfig | None = None,
    use_water_context: bool = False,
    water_config: SegmentationConfig | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[VideoCandidateFrame]:
    """Yield compact candidate/tracking records while decoding one frame at a time."""
    frames = iter_video_frames(path)
    tracker = CandidateTracker()
    previous_gray: NDArray[np.uint8] | None = None
    settings = config or RescueConfig()
    water_settings = water_config or SegmentationConfig()
    try:
        for frame in frames:
            if should_cancel is not None and should_cancel():
                return
            start = perf_counter()
            gray = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2GRAY)
            affine = None
            if previous_gray is not None:
                if previous_gray.shape != gray.shape:
                    raise ValueError("Video frame dimensions changed during candidate tracking.")
                affine = estimate_camera_affine(previous_gray, gray)
            water_mask = (
                segment_water(frame.image_bgr, water_settings).mask
                if use_water_context
                else None
            )
            image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
            analysis = analyze_rescue_image(
                image_rgb,
                image_id=f"{Path(path).name}#frame-{frame.index:06d}",
                frame_index=frame.index,
                water_mask=water_mask,
                config=settings,
            )
            tracked = tracker.update(
                frame.index,
                analysis.candidates,
                image_shape=gray.shape,
                previous_to_current=affine,
            )
            yield VideoCandidateFrame(
                frame_index=frame.index,
                timestamp_ms=frame.timestamp_ms,
                image_bgr=frame.image_bgr,
                tracked_candidates=tracked,
                active_track_count=tracker.active_track_count,
                motion_compensated=affine is not None,
                processing_time_ms=(perf_counter() - start) * 1000.0,
            )
            previous_gray = gray.copy()
    finally:
        frames.close()
