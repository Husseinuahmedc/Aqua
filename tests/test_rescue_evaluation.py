"""Tests for class-separated, annotation-aware candidate evaluation."""

import pytest

from core.datasets.annotations import AnnotationRecord, BoundingBox, CocoImage
from core.rescue.candidates import Candidate
from core.rescue.evaluation import (
    evaluate_candidate_proposals,
    validate_video_disjoint_partitions,
)


def _candidate(
    candidate_id: str,
    box: tuple[int, int, int, int],
    *,
    category: str = "person-like",
    score: float = 0.8,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        image_id="frame-1",
        frame_index=None,
        bbox=box,
        center=(box[0] + box[2] / 2, box[1] + box[3] / 2),
        area_pixels=box[2] * box[3],
        category=category,
        features={},
        rule_score=score,
        source_methods=("color",),
    )


def _data() -> tuple[list[CocoImage], list[AnnotationRecord]]:
    image = CocoImage("frame-1", "frame.jpg", 100, 100, "video-1", "test")
    annotations = [
        AnnotationRecord(
            "frame-1", "video-1", BoundingBox(10, 10, 20, 20), "person", "NITC"
        ),
        AnnotationRecord(
            "frame-1", "video-1", BoundingBox(60, 60, 10, 10), "buoy", "SeaDronesSee"
        ),
        AnnotationRecord(
            "frame-1", "video-1", BoundingBox(40, 40, 15, 15), "ignore", "SeaDronesSee", "ignore"
        ),
    ]
    return [image], annotations


def test_category_metrics_recall_by_iou_and_count_proposals_without_negative_claims() -> None:
    images, annotations = _data()
    candidates = {
        "frame-1": (
            _candidate("p1", (10, 10, 20, 20)),
            _candidate("p2", (12, 12, 18, 18), score=0.7),
            _candidate("f1", (60, 60, 10, 10), category="floating-object-like"),
            _candidate("ignored", (40, 40, 15, 15), score=0.2),
        )
    }

    result = evaluate_candidate_proposals(
        candidates,
        annotations,
        images,
        iou_thresholds=(0.5, 0.75),
        processing_time_ms={"frame-1": 12.5},
    )

    person = [metric for metric in result.metrics if metric.category == "person-like"]
    floating = [metric for metric in result.metrics if metric.category == "floating-object-like"]
    assert [metric.candidate_recall for metric in person] == [1.0, 1.0]
    assert [metric.precision for metric in person] == [None, None]
    assert person[0].candidate_count == 3
    assert person[0].proposals_required_for_observed_recall == 1
    assert floating[0].candidate_recall == 1.0
    assert floating[0].candidate_count == 1
    assert result.processing_time_ms_per_image["frame-1"] == 12.5


def test_complete_annotation_mode_reports_precision_fp_rate_and_excludes_ignore_boxes() -> None:
    images, annotations = _data()
    result = evaluate_candidate_proposals(
        {
            "frame-1": (
                _candidate("tp", (10, 10, 20, 20)),
                _candidate("fp", (1, 70, 8, 8)),
                _candidate("ignored", (40, 40, 15, 15)),
            )
        },
        annotations,
        images,
        categories=("person-like",),
        iou_thresholds=(0.5,),
        complete_negative_labels=True,
    )
    metric = result.metrics[0]
    assert metric.precision == pytest.approx(0.5)
    assert metric.f1 == pytest.approx(2 / 3)
    assert metric.false_positives_per_image == pytest.approx(1.0)


def test_video_partitions_reject_related_frames_split_across_partitions() -> None:
    train = [CocoImage("1", "1.jpg", 10, 10, "same-video", "train")]
    test = [CocoImage("2", "2.jpg", 10, 10, "same-video", "test")]

    with pytest.raises(ValueError, match="multiple partitions"):
        validate_video_disjoint_partitions({"train": train, "test": test})


def test_frame_evaluation_reports_false_positives_and_candidate_counts_per_frame() -> None:
    images, annotations = _data()
    result = evaluate_candidate_proposals(
        {
            "frame-1": (
                _candidate("tp", (10, 10, 20, 20)),
                _candidate("fp", (1, 70, 8, 8)),
            )
        },
        annotations,
        images,
        categories=("person-like",),
        iou_thresholds=(0.5,),
        complete_negative_labels=True,
        sample_unit="frame",
    )

    assert result.sample_unit == "frame"
    assert result.candidate_count_per_frame == {"frame-1": 2}
    assert result.candidate_count_per_image == {}
    assert result.metrics[0].false_positives_per_frame == 1.0
    assert result.metrics[0].false_positives_per_image is None
