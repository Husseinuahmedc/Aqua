"""Image-based traditional-DIP candidate generation pipeline."""

from dataclasses import dataclass
import math
from time import perf_counter

import cv2
import numpy as np

from core.preprocessing.validation import validate_image
from core.rescue.analysis import measure_proposal
from core.rescue.candidates import Candidate, fuse_scored_proposals
from core.rescue.preparation import ImageRepresentations, prepare_rgb_image
from core.rescue.proposals import generate_proposals
from core.water.regions import validate_binary_mask


@dataclass(frozen=True)
class RescueConfig:
    """Explicit thresholds and method switches for image candidate proposals."""

    methods: tuple[str, ...] = ("color", "intensity", "texture", "edge", "adaptive")
    min_area_pixels: int = 9
    max_area_fraction: float = 0.20
    color_threshold: float = 14.0
    intensity_threshold: float = 18.0
    texture_threshold: float = 7.0
    edge_threshold_low: int = 50
    edge_threshold_high: int = 140
    adaptive_block_size: int = 31
    adaptive_constant: float = 5.0
    morphology_kernel_size: int = 3
    water_near_distance_px: float = 30.0
    fusion_iou_threshold: float = 0.45

    def __post_init__(self) -> None:
        supported = {"color", "intensity", "texture", "edge", "adaptive"}
        if (
            not isinstance(self.methods, tuple)
            or not self.methods
            or any(not isinstance(method, str) for method in self.methods)
            or len(set(self.methods)) != len(self.methods)
        ):
            raise ValueError("Select one or more unique candidate proposal methods.")
        if set(self.methods) - supported:
            raise ValueError(
                "Candidate methods must be color, intensity, texture, edge, or adaptive."
            )
        if (
            isinstance(self.min_area_pixels, bool)
            or not isinstance(self.min_area_pixels, int)
            or self.min_area_pixels < 1
        ):
            raise ValueError("Minimum candidate area must be a positive integer.")
        if (
            isinstance(self.max_area_fraction, bool)
            or not math.isfinite(self.max_area_fraction)
            or not 0.0 < self.max_area_fraction <= 1.0
        ):
            raise ValueError("Maximum candidate area fraction must be in (0, 1].")
        if (
            isinstance(self.morphology_kernel_size, bool)
            or not isinstance(self.morphology_kernel_size, int)
            or self.morphology_kernel_size < 1
            or self.morphology_kernel_size % 2 == 0
        ):
            raise ValueError("Morphology kernel size must be a positive odd integer.")
        if (
            isinstance(self.water_near_distance_px, bool)
            or not math.isfinite(self.water_near_distance_px)
            or self.water_near_distance_px < 0
        ):
            raise ValueError("Water proximity distance cannot be negative.")
        if (
            isinstance(self.fusion_iou_threshold, bool)
            or not math.isfinite(self.fusion_iou_threshold)
            or not 0.0 <= self.fusion_iou_threshold <= 1.0
        ):
            raise ValueError("Fusion IoU threshold must be between 0 and 1.")
        for name, value in (
            ("color threshold", self.color_threshold),
            ("intensity threshold", self.intensity_threshold),
            ("texture threshold", self.texture_threshold),
            ("adaptive constant", self.adaptive_constant),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name.title()} cannot be negative.")
        if (
            isinstance(self.edge_threshold_low, bool)
            or isinstance(self.edge_threshold_high, bool)
            or not isinstance(self.edge_threshold_low, int)
            or not isinstance(self.edge_threshold_high, int)
            or not 0 <= self.edge_threshold_low < self.edge_threshold_high <= 255
        ):
            raise ValueError("Edge thresholds must satisfy 0 <= low < high <= 255.")
        if (
            isinstance(self.adaptive_block_size, bool)
            or not isinstance(self.adaptive_block_size, int)
            or self.adaptive_block_size < 3
            or self.adaptive_block_size % 2 == 0
        ):
            raise ValueError("Adaptive block size must be an odd integer >= 3.")


@dataclass(frozen=True)
class RescueAnalysis:
    """Candidate list and measured runtime for one source image/frame."""

    image_id: str
    frame_index: int | None
    image_width: int
    image_height: int
    candidates: tuple[Candidate, ...]
    processing_time_ms: float
    water_context_used: bool
    config: RescueConfig


def _validate_water_mask(mask: np.ndarray, shape: tuple[int, int]) -> None:
    if not isinstance(mask, np.ndarray):
        raise TypeError("Water context must be a NumPy array.")
    if mask.dtype != np.uint8 or mask.ndim != 2:
        raise TypeError("Water context must be a uint8 grayscale binary mask.")
    if mask.shape != shape:
        raise ValueError("Water context dimensions must match the source image.")
    try:
        validate_binary_mask(mask)
    except ValueError as exc:
        raise ValueError("Water context values must be exactly 0 or 255.") from exc


def analyze_rescue_image(
    image_rgb: np.ndarray,
    *,
    image_id: str = "image",
    frame_index: int | None = None,
    water_mask: np.ndarray | None = None,
    config: RescueConfig | None = None,
) -> RescueAnalysis:
    """Generate human-review candidates from a full-resolution RGB image.

    ``water_mask`` is optional context. No candidate proposal is masked away
    based on it; color, intensity, texture, edge, and adaptive methods run
    independently of water prediction and may propose regions anywhere.
    The mask is only used to measure overlap/distance and form a floating-
    object-like rule label. The returned boxes and feature scores are not
    probabilities and do not confirm people or rescue objects.
    """
    validate_image(image_rgb, "RGB")
    if not isinstance(image_id, str) or not image_id:
        raise ValueError("Image identifier must be a non-empty string.")
    if frame_index is not None and (
        isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0
    ):
        raise ValueError("Frame index must be a non-negative integer or None.")
    settings = config or RescueConfig()
    if water_mask is not None:
        _validate_water_mask(water_mask, image_rgb.shape[:2])
    start = perf_counter()
    representations: ImageRepresentations = prepare_rgb_image(image_rgb)
    proposals = generate_proposals(
        representations,
        methods=settings.methods,
        min_area_pixels=settings.min_area_pixels,
        max_area_fraction=settings.max_area_fraction,
        color_threshold=settings.color_threshold,
        intensity_threshold=settings.intensity_threshold,
        texture_threshold=settings.texture_threshold,
        edge_threshold_low=settings.edge_threshold_low,
        edge_threshold_high=settings.edge_threshold_high,
        adaptive_block_size=settings.adaptive_block_size,
        adaptive_constant=settings.adaptive_constant,
        morphology_kernel_size=settings.morphology_kernel_size,
    )
    edge_map = cv2.Canny(
        representations.gray,
        settings.edge_threshold_low,
        settings.edge_threshold_high,
    )
    water_distance_map = (
        cv2.distanceTransform(cv2.bitwise_not(water_mask), cv2.DIST_L2, 3)
        if water_mask is not None
        else None
    )
    water_boundary_distance_map = None
    if water_mask is not None:
        water_boundary = cv2.morphologyEx(
            water_mask,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        water_boundary_distance_map = cv2.distanceTransform(
            cv2.bitwise_not(water_boundary), cv2.DIST_L2, 3
        )
    scored = tuple(
        measure_proposal(
            proposal,
            representations,
            water_mask=water_mask,
            water_distance_map=water_distance_map,
            water_boundary_distance_map=water_boundary_distance_map,
            water_near_distance_px=settings.water_near_distance_px,
            edge_map=edge_map,
        )
        for proposal in proposals
    )
    candidates = fuse_scored_proposals(
        scored,
        image_id=image_id,
        frame_index=frame_index,
        iou_threshold=settings.fusion_iou_threshold,
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    return RescueAnalysis(
        image_id=image_id,
        frame_index=frame_index,
        image_width=int(image_rgb.shape[1]),
        image_height=int(image_rgb.shape[0]),
        candidates=candidates,
        processing_time_ms=elapsed_ms,
        water_context_used=water_mask is not None,
        config=settings,
    )
