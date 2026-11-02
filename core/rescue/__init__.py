"""Traditional-DIP candidate generation for optional human review."""

from core.rescue.candidates import (
    CATEGORY_LABELS,
    REVIEW_STATUS_LABELS,
    Candidate,
    CandidateCategory,
    CandidateReviewStatus,
    describe_candidate_reasoning,
    format_category_label,
    format_review_status,
)
from core.rescue.pipeline import RescueAnalysis, RescueConfig, analyze_rescue_image

__all__ = [
    "CATEGORY_LABELS",
    "REVIEW_STATUS_LABELS",
    "Candidate",
    "CandidateCategory",
    "CandidateReviewStatus",
    "RescueAnalysis",
    "RescueConfig",
    "analyze_rescue_image",
    "describe_candidate_reasoning",
    "format_category_label",
    "format_review_status",
]
