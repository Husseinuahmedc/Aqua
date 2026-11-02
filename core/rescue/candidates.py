"""Candidate records and overlap-based proposal fusion."""

from dataclasses import asdict, dataclass
from typing import Literal, Mapping

CandidateCategory = Literal["person-like", "floating-object-like", "unknown"]
CandidateReviewStatus = Literal["needs_review", "confirmed", "false_alarm"]

CATEGORY_LABELS: Mapping[CandidateCategory, str] = {
    "person-like": "Potential Person",
    "floating-object-like": "Floating Object",
    "unknown": "Unknown Candidate",
}

REVIEW_STATUS_LABELS: Mapping[CandidateReviewStatus, str] = {
    "needs_review": "Requires Human Review",
    "confirmed": "Confirmed Candidate",
    "false_alarm": "False Alarm",
}


def format_category_label(category: CandidateCategory) -> str:
    """Return the user-facing label for a heuristic candidate category."""
    return CATEGORY_LABELS.get(category, "Unknown Candidate")


def format_review_status(status: CandidateReviewStatus) -> str:
    """Return the user-facing label for a candidate human-review status."""
    return REVIEW_STATUS_LABELS.get(status, "Requires Human Review")


@dataclass(frozen=True)
class CandidateProposal:
    """One rectangle proposed by one independent image-processing cue."""

    x: int
    y: int
    width: int
    height: int
    area_pixels: int
    method: str
    contour_points: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class Candidate:
    """Measured region for human review; it is not a confirmed semantic object."""

    candidate_id: str
    image_id: str
    frame_index: int | None
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area_pixels: int
    category: CandidateCategory
    features: Mapping[str, float]
    rule_score: float
    source_methods: tuple[str, ...]
    review_status: CandidateReviewStatus = "needs_review"

    def to_record(self) -> dict[str, object]:
        """Return a flat, JSON/CSV-friendly record with feature fields included."""
        record: dict[str, object] = {
            "candidate_id": self.candidate_id,
            "image_id": self.image_id,
            "frame_index": self.frame_index,
            "x": self.bbox[0],
            "y": self.bbox[1],
            "width": self.bbox[2],
            "height": self.bbox[3],
            "center_x": self.center[0],
            "center_y": self.center[1],
            "area_pixels": self.area_pixels,
            "category": self.category,
            "category_label": format_category_label(self.category),
            "rule_score": self.rule_score,
            "source_methods": ";".join(self.source_methods),
            "review_status": self.review_status,
            "review_status_label": format_review_status(self.review_status),
        }
        record.update({f"feature_{key}": value for key, value in self.features.items()})
        return record


def describe_candidate_reasoning(candidate: Candidate) -> str:
    """Explain the transparent DIP measurements behind a candidate's category."""
    features = candidate.features
    aspect = features.get("aspect_ratio", 0.0)
    solidity = features.get("solidity", 0.0)
    compactness = features.get("compactness", 0.0)
    color_contrast = features.get("local_color_contrast", 0.0)
    intensity_contrast = features.get("local_intensity_contrast", 0.0)
    water_near = bool(features.get("water_near", 0.0) >= 0.5)
    water_available = bool(features.get("water_context_available", 0.0) >= 0.5)
    cues = ", ".join(candidate.source_methods) if candidate.source_methods else "none"

    if candidate.category == "person-like":
        summary = (
            f"Grouped as {format_category_label(candidate.category)} based on compact "
            f"proportions (aspect ratio {aspect:.2f}, solidity {solidity:.2f}, "
            f"compactness {compactness:.2f}) and local contrast "
            f"(color Δ {color_contrast:.1f}, intensity Δ {intensity_contrast:.1f})."
        )
    elif candidate.category == "floating-object-like":
        summary = (
            f"Grouped as {format_category_label(candidate.category)} because it lies "
            f"within or near the potential-water mask and exhibits local appearance "
            f"contrast (color Δ {color_contrast:.1f}, intensity Δ {intensity_contrast:.1f})."
        )
    else:
        water_note = (
            "in or near potential water"
            if water_near
            else ("outside potential water" if water_available else "without water context")
        )
        summary = (
            f"Grouped as {format_category_label(candidate.category)} ({water_note}): "
            f"region contrasts with surroundings (color Δ {color_contrast:.1f}, "
            f"intensity Δ {intensity_contrast:.1f}) but has ambiguous geometry "
            f"(aspect ratio {aspect:.2f}, solidity {solidity:.2f})."
        )
    return (
        f"{summary} Proposed by: {cues}. "
        "Heuristic DIP suggestion only — requires human verification."
    )


def box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    """Compute intersection-over-union for ``(x, y, width, height)`` boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union else 0.0


@dataclass(frozen=True)
class ScoredProposal:
    """Proposal after rule-based measurements, before duplicate suppression."""

    bbox: tuple[int, int, int, int]
    area_pixels: int
    category: CandidateCategory
    features: Mapping[str, float]
    rule_score: float
    method: str


def fuse_scored_proposals(
    proposals: tuple[ScoredProposal, ...],
    *,
    image_id: str,
    frame_index: int | None,
    iou_threshold: float,
) -> tuple[Candidate, ...]:
    """Suppress high-overlap duplicates, retaining the strongest box.

    Proposals are ranked by their transparent rule score, then by area and
    coordinates for deterministic ties. Any lower-ranked proposal with IoU at
    or above the threshold is merged as provenance into the earlier winner;
    its box/semantic hint does not replace the winner's measurements.
    """
    ordered = sorted(
        proposals,
        key=lambda item: (
            -item.rule_score,
            -item.area_pixels,
            item.bbox[1],
            item.bbox[0],
            item.method,
        ),
    )
    winners: list[tuple[ScoredProposal, set[str]]] = []
    for proposal in ordered:
        overlap_winner: tuple[ScoredProposal, set[str]] | None = None
        for winner in winners:
            if box_iou(proposal.bbox, winner[0].bbox) >= iou_threshold:
                overlap_winner = winner
                break
        if overlap_winner is None:
            winners.append((proposal, {proposal.method}))
        else:
            overlap_winner[1].add(proposal.method)

    candidates = []
    for index, (proposal, methods) in enumerate(winners, start=1):
        x, y, width, height = proposal.bbox
        candidates.append(
            Candidate(
                candidate_id=f"{image_id}#candidate-{index:04d}",
                image_id=image_id,
                frame_index=frame_index,
                bbox=proposal.bbox,
                center=(x + width / 2.0, y + height / 2.0),
                area_pixels=proposal.area_pixels,
                category=proposal.category,
                features=dict(proposal.features),
                rule_score=float(proposal.rule_score),
                source_methods=tuple(sorted(methods)),
            )
        )
    return tuple(candidates)


def candidate_to_mapping(candidate: Candidate) -> Mapping[str, object]:
    """Expose a JSON-serializable record without modifying candidate state."""
    return asdict(candidate)
