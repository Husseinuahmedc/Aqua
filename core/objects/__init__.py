"""Experimental classical candidate-region proposals for human review."""

from core.objects.candidates import CandidateRegion, extract_candidate_regions
from core.objects.visualization import draw_candidate_boxes

__all__ = ["CandidateRegion", "draw_candidate_boxes", "extract_candidate_regions"]
