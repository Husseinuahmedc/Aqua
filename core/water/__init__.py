"""Traditional image-processing methods for potential-water masks."""

from core.water.analysis import (
    MaskComparison,
    WaterAnalysis,
    analyze_water,
    compare_masks,
)
from core.water.overlay import create_overlay
from core.water.regions import WaterRegion, extract_regions
from core.water.segmentation import (
    SegmentationConfig,
    SegmentationResult,
    segment_water,
)

__all__ = [
    "MaskComparison",
    "SegmentationConfig",
    "SegmentationResult",
    "WaterAnalysis",
    "WaterRegion",
    "analyze_water",
    "compare_masks",
    "create_overlay",
    "extract_regions",
    "segment_water",
]
