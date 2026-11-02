"""Tests for read-only dataset annotation normalization."""

import json
from pathlib import Path

import pytest

from core.datasets.annotations import (
    load_coco_annotations,
    load_nitc_annotations,
    load_seadronesssee_annotations,
)


def _write_coco(path: Path, *, bbox: list[float] | None = None) -> Path:
    payload = {
        "images": [
            {
                "id": 12,
                "file_name": "videos/clip A/0001.jpg",
                "width": 640,
                "height": 480,
                "source": {"video": "clip A"},
            },
            {"id": 13, "file_name": "negative.jpg", "width": 20, "height": 10},
        ],
        "categories": [
            {"id": 4, "name": "swimmer"},
            {"id": 8, "name": "ignore"},
        ],
        "annotations": [
            {"id": 5, "image_id": 12, "category_id": 4, "bbox": bbox or [1, 2, 3, 4]},
            {
                "id": 6,
                "image_id": 12,
                "category_id": 8,
                "bbox": [10, 11, 5, 6],
                "iscrowd": 1,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_seadronesssee_adapter_preserves_source_metadata_and_classes(
    tmp_path: Path,
) -> None:
    dataset = load_seadronesssee_annotations(
        _write_coco(tmp_path / "annotations.json"), default_split="validation"
    )

    assert len(dataset.images) == 2
    assert dataset.images[0].image_id == "12"
    assert dataset.images[0].source_video_id == "clip A"
    assert dataset.images[0].split == "validation"
    assert dataset.images[1].image_id == "13"  # preserves annotation-free images
    swimmer, ignored = dataset.annotations
    assert swimmer.bbox.x == 1.0 and swimmer.bbox.width == 3.0
    assert swimmer.original_class == "swimmer"
    assert swimmer.source_dataset == "SeaDronesSee Object Detection v2"
    assert swimmer.annotation_status == "annotated"
    assert ignored.annotation_status == "ignore"


def test_nitc_adapter_accepts_only_validated_coco_exports(tmp_path: Path) -> None:
    payload = {
        "images": [{"id": "frame-1", "file_name": "v1/frame.jpg"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"image_id": "frame-1", "category_id": 1, "bbox": [2, 3, 7, 9]}
        ],
    }
    path = tmp_path / "authorized-format.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = load_nitc_annotations(path, default_split="train")

    assert dataset.annotations[0].original_class == "person"
    assert dataset.annotations[0].source_dataset == "NITC Person Rescue"
    assert dataset.images[0].split == "train"


@pytest.mark.parametrize(
    "bbox",
    [[1, 2, 0, 4], [-1, 2, 3, 4], [1, 2, float("nan"), 4], [1, 2, 3]],
)
def test_adapter_rejects_invalid_or_incomplete_boxes(
    tmp_path: Path, bbox: list[float]
) -> None:
    path = _write_coco(tmp_path / "invalid.json", bbox=bbox)
    with pytest.raises(ValueError):
        load_coco_annotations(path, source_dataset="fixture")


def test_adapter_reports_unknown_image_reference(tmp_path: Path) -> None:
    payload = {
        "images": [],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]}
        ],
    }
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown image"):
        load_coco_annotations(path, source_dataset="fixture")
