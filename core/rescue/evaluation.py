"""Annotation-backed candidate recall and proposal-count measurements."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Mapping, Sequence

from core.datasets.annotations import AnnotationRecord, CocoImage
from core.rescue.candidates import Candidate, CandidateCategory, box_iou

PERSON_CLASSES = {
    "person",
    "swimmer",
    "floater",
    "swimmer†",
    "floater†",
}
FLOATING_CLASSES = {
    "boat",
    "jetski",
    "buoy",
    "life saving appliance",
    "life-saving appliance",
    "life jacket",
    "lifebelt",
}


@dataclass(frozen=True)
class ThresholdMetrics:
    """One category/IoU summary, with undefined negative metrics represented None."""

    category: str
    iou_threshold: float
    ground_truth_count: int
    candidate_count: int
    matched_count: int
    candidate_recall: float | None
    precision: float | None
    f1: float | None
    false_positives_per_image: float | None
    false_positives_per_frame: float | None
    proposals_required_for_observed_recall: int | None
    processing_time_ms_per_image: float | None
    processing_time_ms_per_frame: float | None


@dataclass(frozen=True)
class CandidateEvaluation:
    """Aggregate candidate metrics and per-image proposal/runtime measurements."""

    metrics: tuple[ThresholdMetrics, ...]
    candidate_count_per_image: Mapping[str, int]
    candidate_count_per_frame: Mapping[str, int]
    processing_time_ms_per_image: Mapping[str, float]
    processing_time_ms_per_frame: Mapping[str, float]
    complete_negative_labels: bool
    sample_unit: Literal["image", "frame"]

    def to_record(self) -> dict[str, object]:
        """Produce plain JSON-serializable summary records."""
        return {
            "complete_negative_labels": self.complete_negative_labels,
            "metrics": [asdict(metric) for metric in self.metrics],
            "candidate_count_per_image": dict(self.candidate_count_per_image),
            "candidate_count_per_frame": dict(self.candidate_count_per_frame),
            "processing_time_ms_per_image": dict(self.processing_time_ms_per_image),
            "processing_time_ms_per_frame": dict(self.processing_time_ms_per_frame),
            "sample_unit": self.sample_unit,
        }


def annotation_category(original_class: str) -> CandidateCategory | None:
    """Map documented class names to broad review categories without rewriting."""
    name = " ".join(original_class.strip().lower().replace("_", " ").split())
    if name in PERSON_CLASSES:
        return "person-like"
    if name in FLOATING_CLASSES:
        return "floating-object-like"
    return None


def validate_video_disjoint_partitions(
    partitions: Mapping[str, Iterable[CocoImage]],
) -> None:
    """Raise if a source video occurs in more than one supplied partition."""
    video_partitions: dict[str, set[str]] = defaultdict(set)
    for split_name, images in partitions.items():
        for image in images:
            if image.source_video_id is not None:
                video_partitions[image.source_video_id].add(split_name)
    leakage = {
        video_id: names
        for video_id, names in video_partitions.items()
        if len(names) > 1
    }
    if leakage:
        example = next(iter(sorted(leakage)))
        raise ValueError(
            f"Source video {example!r} occurs in multiple partitions; group related frames."
        )


def _candidate_category(category: CandidateCategory) -> CandidateCategory | None:
    return category if category != "unknown" else None


def evaluate_candidate_proposals(
    candidates_by_image: Mapping[str, Sequence[Candidate]],
    annotations: Sequence[AnnotationRecord],
    images: Sequence[CocoImage],
    *,
    categories: Sequence[CandidateCategory] = ("person-like", "floating-object-like"),
    iou_thresholds: Sequence[float] = (0.25, 0.50, 0.75),
    processing_time_ms: Mapping[str, float] | None = None,
    complete_negative_labels: bool = False,
    sample_unit: Literal["image", "frame"] = "image",
) -> CandidateEvaluation:
    """Evaluate class-separated proposal recall, count, and optionally precision.

    Positive ground truths are matched greedily to descending rule-score
    proposals, one-to-one, independently for each IoU threshold. Unknown
    annotations are not silently folded into either rescue category. An
    annotation marked ``ignore`` can absorb overlapping proposals. Because
    annotation completeness is dataset-policy dependent, precision, F1, and
    false-positive rate remain undefined unless the caller explicitly asserts
    complete negative labels for this evaluation subset.
    """
    if not images:
        raise ValueError("At least one image record is required for evaluation.")
    if sample_unit not in ("image", "frame"):
        raise ValueError("Evaluation sample unit must be 'image' or 'frame'.")
    if not categories or any(
        category not in ("person-like", "floating-object-like")
        for category in categories
    ):
        raise ValueError("Evaluate person-like and/or floating-object-like candidates.")
    if not iou_thresholds or any(not 0.0 <= value <= 1.0 for value in iou_thresholds):
        raise ValueError("IoU thresholds must be a non-empty sequence in [0, 1].")

    image_ids = [image.image_id for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Image inventory contains duplicate image identifiers.")
    known_ids = set(image_ids)
    if any(image_id not in known_ids for image_id in candidates_by_image):
        raise ValueError("Candidate results reference an image outside the evaluation inventory.")
    if any(record.image_id not in known_ids for record in annotations):
        raise ValueError("Annotation references an image outside the evaluation inventory.")

    by_image_category: dict[tuple[str, str], list[AnnotationRecord]] = defaultdict(list)
    by_image_ignore: dict[str, list[AnnotationRecord]] = defaultdict(list)
    for record in annotations:
        if record.annotation_status == "ignore":
            by_image_ignore[record.image_id].append(record)
            continue
        category = annotation_category(record.original_class)
        if category is not None:
            by_image_category[(record.image_id, category)].append(record)

    counts = {
        image_id: len(candidates_by_image.get(image_id, ())) for image_id in image_ids
    }
    timings = {
        image_id: float(processing_time_ms[image_id])
        for image_id in image_ids
        if processing_time_ms is not None and image_id in processing_time_ms
    }
    output: list[ThresholdMetrics] = []
    total_images = len(image_ids)
    for category in categories:
        ground_truth_total = sum(
            len(by_image_category[(image_id, category)]) for image_id in image_ids
        )
        category_candidates = {
            image_id: sorted(
                (
                    candidate
                    for candidate in candidates_by_image.get(image_id, ())
                    if _candidate_category(candidate.category) == category
                ),
                key=lambda candidate: (-candidate.rule_score, candidate.candidate_id),
            )
            for image_id in image_ids
        }
        category_candidate_total = sum(map(len, category_candidates.values()))
        for threshold in iou_thresholds:
            matched_total = 0
            ignored_prediction_total = 0
            proposals_to_observed_recall = 0
            for image_id in image_ids:
                remaining = set(range(len(by_image_category[(image_id, category)])))
                ignored_boxes = [record.bbox for record in by_image_ignore[image_id]]
                candidates = category_candidates[image_id]
                last_matched_rank = 0
                for rank, candidate in enumerate(candidates, start=1):
                    x, y, width, height = candidate.bbox
                    overlaps = [
                        (
                            box_iou(
                                (x, y, width, height),
                                (
                                    record.bbox.x,
                                    record.bbox.y,
                                    record.bbox.width,
                                    record.bbox.height,
                                ),
                            ),
                            index,
                        )
                        for index, record in enumerate(
                            by_image_category[(image_id, category)]
                        )
                        if index in remaining
                    ]
                    best_iou, best_index = max(overlaps, default=(0.0, -1))
                    if best_iou >= threshold:
                        remaining.remove(best_index)
                        matched_total += 1
                        last_matched_rank = rank
                    elif any(
                        box_iou(
                            (x, y, width, height),
                            (
                                box.x,
                                box.y,
                                box.width,
                                box.height,
                            ),
                        )
                        >= threshold
                        for box in ignored_boxes
                    ):
                        ignored_prediction_total += 1
                        continue
                proposals_to_observed_recall += last_matched_rank

            recall = (
                matched_total / ground_truth_total if ground_truth_total else None
            )
            if complete_negative_labels:
                scored_candidate_total = max(
                    0, category_candidate_total - ignored_prediction_total
                )
                precision = (
                    matched_total / scored_candidate_total
                    if scored_candidate_total
                    else None
                )
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and precision + recall
                    else (0.0 if precision is not None and recall == 0.0 else None)
                )
                false_positives = max(0, scored_candidate_total - matched_total)
                fp_per_sample = false_positives / total_images
            else:
                precision = None
                f1 = None
                fp_per_sample = None
            output.append(
                ThresholdMetrics(
                    category=category,
                    iou_threshold=float(threshold),
                    ground_truth_count=ground_truth_total,
                    candidate_count=category_candidate_total,
                    matched_count=matched_total,
                    candidate_recall=recall,
                    precision=precision,
                    f1=f1,
                    false_positives_per_image=(
                        fp_per_sample if sample_unit == "image" else None
                    ),
                    false_positives_per_frame=(
                        fp_per_sample if sample_unit == "frame" else None
                    ),
                    proposals_required_for_observed_recall=proposals_to_observed_recall,
                    processing_time_ms_per_image=(
                        sum(timings.values()) / len(timings)
                        if timings and sample_unit == "image"
                        else None
                    ),
                    processing_time_ms_per_frame=(
                        sum(timings.values()) / len(timings)
                        if timings and sample_unit == "frame"
                        else None
                    ),
                )
            )
    return CandidateEvaluation(
        metrics=tuple(output),
        candidate_count_per_image=counts if sample_unit == "image" else {},
        candidate_count_per_frame=counts if sample_unit == "frame" else {},
        processing_time_ms_per_image=timings if sample_unit == "image" else {},
        processing_time_ms_per_frame=timings if sample_unit == "frame" else {},
        complete_negative_labels=complete_negative_labels,
        sample_unit=sample_unit,
    )
