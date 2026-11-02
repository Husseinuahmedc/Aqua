"""Frame-by-frame video decoding and analysis interfaces."""

from core.video.analysis import VideoFrameAnalysis, iter_video_analysis
from core.video.reader import (
    VideoCaptureReader,
    VideoFrame,
    VideoInfo,
    get_video_info,
    iter_video_frames,
)

__all__ = [
    "VideoFrame",
    "VideoFrameAnalysis",
    "VideoInfo",
    "VideoCaptureReader",
    "get_video_info",
    "iter_video_analysis",
    "iter_video_frames",
]
