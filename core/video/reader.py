"""OpenCV video metadata and lazy frame decoding."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from core.preprocessing.validation import validate_image


@dataclass(frozen=True)
class VideoInfo:
    """Video stream properties reported by the active OpenCV backend."""

    path: Path
    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    backend: str


@dataclass(frozen=True)
class VideoFrame:
    """One decoded video frame. Color pixels are OpenCV BGR uint8."""

    index: int
    timestamp_ms: float | None
    image_bgr: NDArray[np.uint8]


def _validate_video_path(path: Path | str) -> Path:
    video_path = Path(path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video path is not a file: {video_path}")
    return video_path


class VideoCaptureReader:
    """Small stateful OpenCV reader with metadata, sequential read, and seek."""

    def __init__(self, path: Path | str) -> None:
        self.path = _validate_video_path(path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self._capture.release()
            raise ValueError(
                f"Could not open video '{self.path}'. Check the file and "
                "installed codecs."
            )
        self._next_index = 0
        try:
            self.info = self._read_info()
        except Exception:
            self.close()
            raise

    def _read_info(self) -> VideoInfo:
        capture = self._capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            backend = str(capture.getBackendName())
        except cv2.error:
            backend = "unknown"
        if width <= 0 or height <= 0:
            raise ValueError(f"Video '{self.path}' reports invalid dimensions.")
        fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else None
        frame_count = (
            max(0, int(raw_count))
            if math.isfinite(raw_count) and raw_count >= 0
            else None
        )
        duration = (
            frame_count / fps
            if frame_count is not None and fps is not None and fps > 0
            else None
        )
        return VideoInfo(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration,
            backend=backend,
        )

    def read_next(self) -> VideoFrame | None:
        """Decode and return the next BGR frame, or ``None`` at end of stream."""
        if self._capture is None:
            raise RuntimeError("Video reader is closed.")
        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        validate_image(frame, "BGR")
        fps = self.info.fps
        timestamp = self._next_index * 1000.0 / fps if fps is not None else None
        result = VideoFrame(
            index=self._next_index,
            timestamp_ms=timestamp,
            image_bgr=frame,
        )
        self._next_index += 1
        return result

    def seek_frame(self, frame_index: int) -> None:
        """Move to a zero-based frame index when the active backend supports it."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise TypeError("Frame index must be an integer.")
        if frame_index < 0:
            raise ValueError("Frame index cannot be negative.")
        if self._capture is None:
            raise RuntimeError("Video reader is closed.")
        if not self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise ValueError(f"Video backend could not seek to frame {frame_index}.")
        self._next_index = frame_index

    def close(self) -> None:
        """Release the OpenCV capture; repeated calls are safe."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoCaptureReader":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def get_video_info(path: Path | str) -> VideoInfo:
    """Read stream metadata and release the capture on exit."""
    with VideoCaptureReader(path) as reader:
        return reader.info


def iter_video_frames(path: Path | str) -> Iterator[VideoFrame]:
    """Yield decoded BGR frames sequentially and release capture on close.

    Consumers that stop early should close the iterator (or use ``for`` inside
    a generator/finally block) so the OpenCV capture is released promptly.
    Codec support is supplied by the installed OpenCV backend and is
    platform-dependent.
    """
    frame_count = 0
    with VideoCaptureReader(path) as reader:
        while True:
            frame = reader.read_next()
            if frame is None:
                break
            yield frame
            frame_count += 1
        if frame_count == 0:
            raise ValueError(
                f"Video '{reader.path}' opened but contained no decodable frames."
            )
