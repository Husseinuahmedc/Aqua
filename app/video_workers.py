"""Qt workers for sequential video analysis and frame playback."""

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Event, Lock

from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal

from core.video.analysis import iter_video_analysis
from core.video.reader import VideoCaptureReader, VideoFrame
from core.rescue.pipeline import RescueConfig
from core.rescue.video import TrackedCandidate, iter_video_candidate_analysis
from core.water.segmentation import SegmentationConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoProgress:
    """Latest video preview/metrics snapshot, bounded to one frame."""

    frame_index: int
    timestamp_ms: float | None
    frame_bgr: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    total_pixels: int
    water_pixels: int
    coverage_percent: float
    region_count: int
    changed_pixels: int | None
    water_pixel_change: int | None
    coverage_change_percentage_points: float | None


@dataclass(frozen=True)
class RescueVideoProgress:
    """One replaceable video-frame/candidate overlay snapshot."""

    frame_index: int
    timestamp_ms: float | None
    frame_bgr: NDArray[np.uint8]
    tracked_candidates: tuple[TrackedCandidate, ...]
    active_track_count: int
    motion_compensated: bool


class RescueVideoAnalysisWorker(QThread):
    """Run bounded frame-wise candidate analysis and retain compact records."""

    completed = Signal(object)
    cancelled = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        path: Path,
        config: RescueConfig,
        *,
        use_water_context: bool = False,
        water_config: SegmentationConfig | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._config = config
        self._use_water_context = use_water_context
        self._water_config = water_config
        self._cancel_event = Event()
        self._state_lock = Lock()
        self._latest_progress: RescueVideoProgress | None = None

    def cancel(self) -> None:
        """Request cancellation between decoded frame operations."""
        self._cancel_event.set()

    def take_latest_progress(self) -> RescueVideoProgress | None:
        """Return the most recent frame only; stale preview frames are replaced."""
        with self._state_lock:
            progress = self._latest_progress
            self._latest_progress = None
        return progress

    def run(self) -> None:
        candidate_rows: list[dict[str, object]] = []
        frame_rows: list[dict[str, object]] = []
        tracked_by_frame: dict[int, tuple[TrackedCandidate, ...]] = {}
        last_frame_bgr: NDArray[np.uint8] | None = None
        last_frame_index: int | None = None
        try:
            for result in iter_video_candidate_analysis(
                self._path,
                config=self._config,
                use_water_context=self._use_water_context,
                water_config=self._water_config,
                should_cancel=self._cancel_event.is_set,
            ):
                tracked_by_frame[result.frame_index] = result.tracked_candidates
                last_frame_bgr = result.image_bgr
                last_frame_index = result.frame_index
                frame_rows.append(
                    {
                        "frame_index": result.frame_index,
                        "timestamp_ms": result.timestamp_ms,
                        "candidate_count": len(result.tracked_candidates),
                        "active_track_count": result.active_track_count,
                        "motion_compensated": result.motion_compensated,
                        "processing_time_ms": result.processing_time_ms,
                    }
                )
                for tracked in result.tracked_candidates:
                    candidate_rows.append(
                        {
                            **tracked.candidate.to_record(),
                            "source_video": str(self._path),
                            "timestamp_ms": result.timestamp_ms,
                            "track_id": tracked.track_id,
                            "track_persistence_frames": tracked.persistence_frames,
                            "motion_compensated": result.motion_compensated,
                            "frame_processing_time_ms": result.processing_time_ms,
                        }
                    )
                progress = RescueVideoProgress(
                    frame_index=result.frame_index,
                    timestamp_ms=result.timestamp_ms,
                    frame_bgr=result.image_bgr,
                    tracked_candidates=result.tracked_candidates,
                    active_track_count=result.active_track_count,
                    motion_compensated=result.motion_compensated,
                )
                with self._state_lock:
                    self._latest_progress = progress
            payload = {
                "candidates": candidate_rows,
                "frames": frame_rows,
                "tracked_by_frame": tracked_by_frame,
                "last_frame_bgr": last_frame_bgr,
                "last_frame_index": last_frame_index,
            }
            if self._cancel_event.is_set():
                self.cancelled.emit(payload)
            else:
                self.completed.emit(payload)
        except Exception as exc:
            LOGGER.exception("Video candidate worker failed for %s", self._path)
            self.error_occurred.emit(str(exc))


class VideoAnalysisWorker(QThread):
    """Analyze video sequentially and expose one replaceable preview snapshot."""

    completed = Signal(object)
    cancelled = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        path: Path,
        config: SegmentationConfig,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._config = config
        self._cancel_event = Event()
        self._state_lock = Lock()
        self._latest_progress: VideoProgress | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation between frames."""
        self._cancel_event.set()

    def take_latest_progress(self) -> VideoProgress | None:
        """Return and clear the latest frame snapshot without queuing signals."""
        with self._state_lock:
            progress = self._latest_progress
            self._latest_progress = None
        return progress

    def run(self) -> None:
        rows: list[dict[str, int | float | None]] = []
        try:
            for frame_analysis in iter_video_analysis(
                self._path,
                self._config,
                should_cancel=self._cancel_event.is_set,
            ):
                if self._cancel_event.is_set():
                    break
                comparison = frame_analysis.comparison_to_previous
                row: dict[str, int | float | None] = {
                    "frame_index": frame_analysis.frame_index,
                    "timestamp_ms": frame_analysis.timestamp_ms,
                    "total_pixels": frame_analysis.water.total_pixels,
                    "water_pixels": frame_analysis.water.water_pixels,
                    "coverage_percent": frame_analysis.water.coverage_percent,
                    "region_count": frame_analysis.water.region_count,
                    "changed_pixels": (
                        comparison.changed_pixels if comparison is not None else None
                    ),
                    "water_pixel_change": (
                        comparison.water_pixel_change
                        if comparison is not None
                        else None
                    ),
                    "coverage_change_percentage_points": (
                        comparison.coverage_change_percentage_points
                        if comparison is not None
                        else None
                    ),
                }
                rows.append(row)
                progress = VideoProgress(
                    frame_index=frame_analysis.frame_index,
                    timestamp_ms=frame_analysis.timestamp_ms,
                    frame_bgr=frame_analysis.image_bgr,
                    mask=frame_analysis.mask,
                    total_pixels=frame_analysis.water.total_pixels,
                    water_pixels=frame_analysis.water.water_pixels,
                    coverage_percent=frame_analysis.water.coverage_percent,
                    region_count=frame_analysis.water.region_count,
                    changed_pixels=(
                        comparison.changed_pixels if comparison is not None else None
                    ),
                    water_pixel_change=(
                        comparison.water_pixel_change
                        if comparison is not None
                        else None
                    ),
                    coverage_change_percentage_points=(
                        comparison.coverage_change_percentage_points
                        if comparison is not None
                        else None
                    ),
                )
                with self._state_lock:
                    self._latest_progress = progress
            if self._cancel_event.is_set():
                self.cancelled.emit(rows)
            else:
                self.completed.emit(rows)
        except Exception as exc:
            LOGGER.exception("Video analysis worker failed for %s", self._path)
            self.error_occurred.emit(str(exc))


class VideoPlaybackWorker(QThread):
    """Decode playback frames on a worker thread with one-frame backpressure."""

    frame_ready = Signal(object)
    info_ready = Signal(object)
    error_occurred = Signal(str)
    playback_ended = Signal()

    def __init__(self, path: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._path = path
        self._state_lock = Lock()
        self._playing = False
        self._at_end = False
        self._seek_to: int | None = None
        self._stop_event = Event()
        self._command_event = Event()
        self._frame_acknowledged = Event()
        self._frame_acknowledged.set()

    def play(self) -> None:
        with self._state_lock:
            self._playing = True
            if self._at_end:
                self._seek_to = 0
                self._at_end = False
        self._command_event.set()

    def pause(self) -> None:
        with self._state_lock:
            self._playing = False
        self._command_event.set()

    def seek(self, frame_index: int) -> None:
        with self._state_lock:
            self._seek_to = max(0, int(frame_index))
            self._at_end = False
        self._command_event.set()

    def acknowledge_frame(self) -> None:
        """Allow the decoder to enqueue at most one subsequent preview frame."""
        self._frame_acknowledged.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._frame_acknowledged.set()
        self._command_event.set()

    def _take_command(self) -> tuple[bool, int | None]:
        with self._state_lock:
            seek_to = self._seek_to
            self._seek_to = None
            return self._playing, seek_to

    def _emit_frame(
        self,
        frame: VideoFrame,
    ) -> None:
        self._frame_acknowledged.clear()
        self.frame_ready.emit(frame)

    def run(self) -> None:
        reader: VideoCaptureReader | None = None
        try:
            reader = VideoCaptureReader(self._path)
            self.info_ready.emit(reader.info)
            first_frame = reader.read_next()
            if first_frame is None:
                raise ValueError(f"Video '{self._path}' contains no decodable frames.")
            self._emit_frame(first_frame)
            frame_interval = (
                1.0 / reader.info.fps if reader.info.fps is not None else 1.0 / 24.0
            )

            while not self._stop_event.is_set():
                if not self._frame_acknowledged.wait(timeout=0.05):
                    continue
                if self._stop_event.is_set():
                    break
                playing, seek_to = self._take_command()
                if seek_to is not None:
                    try:
                        reader.seek_frame(seek_to)
                    except (TypeError, ValueError, RuntimeError) as exc:
                        self.error_occurred.emit(str(exc))
                        continue
                    frame = reader.read_next()
                    if frame is not None:
                        self._emit_frame(frame)
                    else:
                        self.error_occurred.emit(
                            f"Could not decode requested frame {seek_to}."
                        )
                    continue

                if not playing:
                    self._command_event.wait(timeout=0.1)
                    self._command_event.clear()
                    continue

                frame = reader.read_next()
                if frame is None:
                    with self._state_lock:
                        self._playing = False
                        self._at_end = True
                    self.playback_ended.emit()
                    continue
                self._emit_frame(frame)
                self._command_event.wait(timeout=frame_interval)
                self._command_event.clear()
        except Exception as exc:
            LOGGER.exception("Video playback worker failed for %s", self._path)
            self.error_occurred.emit(str(exc))
        finally:
            if reader is not None:
                reader.close()
