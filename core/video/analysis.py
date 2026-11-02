"""Lazy, frame-by-frame video segmentation and mask-change summaries."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

from core.water.analysis import (
    MaskComparison,
    WaterAnalysis,
    analyze_water,
    compare_masks,
)
from core.water.segmentation import SegmentationConfig, segment_water
from core.video.reader import iter_video_frames


@dataclass(frozen=True)
class VideoFrameAnalysis:
    """Analysis for one frame; consumers should not retain all frame arrays."""

    frame_index: int
    timestamp_ms: float | None
    image_bgr: NDArray[np.uint8]
    mask: NDArray[np.uint8]
    water: WaterAnalysis
    comparison_to_previous: MaskComparison | None


def iter_video_analysis(
    path: Path | str,
    config: SegmentationConfig,
    *,
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[VideoFrameAnalysis]:
    """Yield each frame's segmentation and compact coverage/change metrics.

    Only the current frame and previous binary mask are needed for processing;
    callers decide whether to retain compact metric records for export. The
    source frames are never resized, and cancellation is checked between frames.
    """
    frames = iter_video_frames(path)
    previous_mask: NDArray[np.uint8] | None = None
    try:
        for frame in frames:
            if should_cancel is not None and should_cancel():
                return
            result = segment_water(frame.image_bgr, config)
            water = analyze_water(result.mask, regions=result.regions)
            comparison = (
                compare_masks(previous_mask, result.mask)
                if previous_mask is not None
                else None
            )
            yield VideoFrameAnalysis(
                frame_index=frame.index,
                timestamp_ms=frame.timestamp_ms,
                image_bgr=frame.image_bgr,
                mask=result.mask,
                water=water,
                comparison_to_previous=comparison,
            )
            previous_mask = result.mask
    finally:
        frames.close()
