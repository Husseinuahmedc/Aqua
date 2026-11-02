"""Read-only adapters for external dataset annotations."""

from core.datasets.annotations import (
    AnnotationRecord,
    BoundingBox,
    CocoDataset,
    CocoImage,
    load_coco_annotations,
    load_nitc_annotations,
    load_seadronesssee_annotations,
)

__all__ = [
    "AnnotationRecord",
    "BoundingBox",
    "CocoDataset",
    "CocoImage",
    "load_coco_annotations",
    "load_nitc_annotations",
    "load_seadronesssee_annotations",
]
