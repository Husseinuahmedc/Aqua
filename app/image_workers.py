"""Qt worker for loading and processing one image or mask."""

from dataclasses import dataclass
import cv2
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QThread, Signal

from core.evaluation.masks import load_reference_mask
from core.evaluation.metrics import evaluate_masks
from core.objects.candidates import CandidateRegion, extract_candidate_regions
from core.objects.visualization import draw_candidate_boxes
from core.preprocessing.histogram import HistogramData, compute_histogram
from core.preprocessing.image_io import ImageInfo, image_info, load_image
from core.preprocessing.transforms import (
    BlurMethod,
    ColorSpace,
    convert_color,
    preprocess_image,
    to_display_rgb,
)
from core.water.analysis import WaterAnalysis, analyze_water
from core.water.overlay import create_overlay
from core.water.segmentation import (
    SegmentationConfig,
    SegmentationResult,
    segment_water,
)
from core.rescue.pipeline import RescueAnalysis, RescueConfig, analyze_rescue_image
from core.rescue.visualization import draw_candidate_boxes_rgb

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadImageTask:
    """Data-only request to decode a local image off the GUI thread."""

    path: Path


@dataclass(frozen=True)
class LoadReferenceMaskTask:
    """Data-only request to decode and validate a manual binary mask."""

    path: Path


@dataclass(frozen=True)
class HistogramTask:
    """Data-only request for a per-channel image histogram."""

    image_bgr: NDArray[np.uint8]
    color_space: ColorSpace
    channel: str


@dataclass(frozen=True)
class PreprocessTask:
    """Data-only request for an image preprocessing operation."""

    image_bgr: NDArray[np.uint8]
    target_space: ColorSpace
    blur_method: BlurMethod
    kernel_size: int


@dataclass(frozen=True)
class SegmentTask:
    """Data-only request for a water segmentation operation."""

    image_bgr: NDArray[np.uint8]
    config: SegmentationConfig
    extract_candidates: bool = False


@dataclass(frozen=True)
class EvaluateTask:
    """Data-only request for comparing two binary masks."""

    predicted_mask: NDArray[np.uint8]
    reference_mask: NDArray[np.uint8]


@dataclass(frozen=True)
class RescueTask:
    """Data-only request for independent search-and-rescue candidates."""

    image_bgr: NDArray[np.uint8]
    image_id: str
    use_water_context: bool = False
    config: RescueConfig = RescueConfig()
    water_config: SegmentationConfig = SegmentationConfig()


@dataclass(frozen=True)
class LoadedImageOutput:
    """Decoded source pixels and basic information for the image workspace."""

    path: Path
    image_bgr: NDArray[np.uint8]
    info: ImageInfo


@dataclass(frozen=True)
class LoadedMaskOutput:
    """Normalized manual reference mask and its source path."""

    path: Path
    mask: NDArray[np.uint8]


@dataclass(frozen=True)
class HistogramOutput:
    """Computed histogram values for a selected image channel."""

    histogram: HistogramData
    color_space: ColorSpace
    channel: str


@dataclass(frozen=True)
class PreprocessOutput:
    """Preprocessed pixels and the space required to display/save them."""

    image: NDArray[np.uint8]
    color_space: ColorSpace
    display_rgb: NDArray[np.uint8]
    blur_method: BlurMethod
    kernel_size: int


@dataclass(frozen=True)
class SegmentationOutput:
    """Segmentation outputs prepared for image, metrics, and overlay views."""

    original_bgr: NDArray[np.uint8]
    result: SegmentationResult
    water_analysis: WaterAnalysis
    overlay_bgr: NDArray[np.uint8]
    candidates: tuple[CandidateRegion, ...]
    candidate_overlay_bgr: NDArray[np.uint8] | None


@dataclass(frozen=True)
class RescueOutput:
    """Candidate output and RGB box overlay for the rescue review workspace."""

    analysis: RescueAnalysis
    candidate_overlay_rgb: NDArray[np.uint8]
    water_context_config: SegmentationConfig | None


class ImageTaskWorker(QThread):
    """Run one CPU-bound image task away from the Qt GUI thread."""

    result_ready = Signal(object)
    error_occurred = Signal(str)

    def __init__(
        self,
        task: (
            LoadImageTask
            | LoadReferenceMaskTask
            | HistogramTask
            | PreprocessTask
            | SegmentTask
            | EvaluateTask
            | RescueTask
        ),
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._task = task

    def run(self) -> None:
        try:
            if isinstance(self._task, LoadImageTask):
                image = load_image(self._task.path)
                output = LoadedImageOutput(
                    path=self._task.path,
                    image_bgr=image,
                    info=image_info(image),
                )
            elif isinstance(self._task, LoadReferenceMaskTask):
                output = LoadedMaskOutput(
                    path=self._task.path,
                    mask=load_reference_mask(self._task.path),
                )
            elif isinstance(self._task, HistogramTask):
                converted = convert_color(
                    self._task.image_bgr,
                    "BGR",
                    self._task.color_space,
                )
                output = HistogramOutput(
                    histogram=compute_histogram(
                        converted,
                        self._task.color_space,
                        channel=self._task.channel,
                    ),
                    color_space=self._task.color_space,
                    channel=self._task.channel,
                )
            elif isinstance(self._task, PreprocessTask):
                result = preprocess_image(
                    self._task.image_bgr,
                    target_space=self._task.target_space,
                    blur_method=self._task.blur_method,
                    kernel_size=self._task.kernel_size,
                )
                output = PreprocessOutput(
                    image=result,
                    color_space=self._task.target_space,
                    display_rgb=to_display_rgb(result, self._task.target_space),
                    blur_method=self._task.blur_method,
                    kernel_size=self._task.kernel_size,
                )
            elif isinstance(self._task, SegmentTask):
                result = segment_water(self._task.image_bgr, self._task.config)
                water = analyze_water(result.mask, regions=result.regions)
                candidates: tuple[CandidateRegion, ...] = ()
                candidate_overlay = None
                if self._task.extract_candidates:
                    candidates = extract_candidate_regions(
                        result.mask,
                        min_area_pixels=self._task.config.min_region_area,
                    )
                    candidate_overlay = draw_candidate_boxes(
                        self._task.image_bgr,
                        candidates,
                    )
                output = SegmentationOutput(
                    original_bgr=self._task.image_bgr,
                    result=result,
                    water_analysis=water,
                    overlay_bgr=create_overlay(self._task.image_bgr, result.mask),
                    candidates=candidates,
                    candidate_overlay_bgr=candidate_overlay,
                )
            elif isinstance(self._task, RescueTask):
                water_mask = (
                    segment_water(self._task.image_bgr, self._task.water_config).mask
                    if self._task.use_water_context
                    else None
                )
                image_rgb = cv2.cvtColor(self._task.image_bgr, cv2.COLOR_BGR2RGB)
                analysis = analyze_rescue_image(
                    image_rgb,
                    image_id=self._task.image_id,
                    water_mask=water_mask,
                    config=self._task.config,
                )
                output = RescueOutput(
                    analysis=analysis,
                    candidate_overlay_rgb=draw_candidate_boxes_rgb(
                        image_rgb, analysis.candidates
                    ),
                    water_context_config=(
                        self._task.water_config
                        if self._task.use_water_context
                        else None
                    ),
                )
            else:
                output = evaluate_masks(
                    self._task.predicted_mask,
                    self._task.reference_mask,
                )
            self.result_ready.emit(output)
        except Exception as exc:  # Report an actionable error to the GUI.
            LOGGER.exception("Image processing worker failed")
            self.error_occurred.emit(str(exc))
