"""Transparent geometric and appearance feature measurements."""

import math

import cv2
import numpy as np

from core.rescue.candidates import CandidateProposal, CandidateCategory, ScoredProposal
from core.rescue.preparation import ImageRepresentations


def _proposal_contour(proposal: CandidateProposal) -> np.ndarray:
    """Recover the original proposal boundary in full-image coordinates."""
    if proposal.contour_points:
        return np.asarray(proposal.contour_points, dtype=np.int32).reshape(-1, 1, 2)
    x, y, width, height = (
        proposal.x,
        proposal.y,
        proposal.width,
        proposal.height,
    )
    return np.asarray(
        [
            [[x, y]],
            [[x + width - 1, y]],
            [[x + width - 1, y + height - 1]],
            [[x, y + height - 1]],
        ],
        dtype=np.int32,
    )


def _ring_bounds(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    margin = max(2, int(round(max(width, height) * 0.5)))
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(shape[1], x + width + margin),
        min(shape[0], y + height + margin),
    )


def measure_proposal(
    proposal: CandidateProposal,
    representations: ImageRepresentations,
    *,
    water_mask: np.ndarray | None,
    water_distance_map: np.ndarray | None,
    water_boundary_distance_map: np.ndarray | None,
    water_near_distance_px: float,
    edge_map: np.ndarray,
) -> ScoredProposal:
    """Measure shape, local appearance, edge, and optional water-context cues."""
    x, y, width, height = proposal.x, proposal.y, proposal.width, proposal.height
    bbox = (x, y, width, height)
    height_px, width_px = representations.gray.shape
    crop_slice = np.s_[y : y + height, x : x + width]
    component_area = max(1, proposal.area_pixels)
    box_area = max(1, width * height)
    aspect = width / max(height, 1)
    contour = _proposal_contour(proposal)
    contour_area = max(float(cv2.contourArea(contour)), float(component_area))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(1.0, contour_area / max(hull_area, float(component_area)))
    extent = min(1.0, component_area / box_area)
    compactness = min(
        1.0,
        (4.0 * math.pi * component_area) / max(perimeter * perimeter, 1.0),
    )
    contour_complexity = (perimeter * perimeter) / (4.0 * math.pi * component_area)

    ring_x1, ring_y1, ring_x2, ring_y2 = _ring_bounds(bbox, (height_px, width_px))
    ring_mask = np.ones((ring_y2 - ring_y1, ring_x2 - ring_x1), dtype=bool)
    ring_mask[y - ring_y1 : y - ring_y1 + height, x - ring_x1 : x - ring_x1 + width] = False
    ring_gray = representations.gray[ring_y1:ring_y2, ring_x1:ring_x2][ring_mask]
    ring_lab = representations.lab[ring_y1:ring_y2, ring_x1:ring_x2][ring_mask]
    roi_gray = representations.gray[crop_slice]
    roi_lab = representations.lab[crop_slice]
    intensity_contrast = (
        abs(float(roi_gray.mean()) - float(ring_gray.mean())) if ring_gray.size else 0.0
    )
    color_contrast = (
        float(np.linalg.norm(roi_lab.mean(axis=(0, 1)) - ring_lab.mean(axis=0)))
        if ring_lab.size
        else 0.0
    )
    texture_difference = (
        abs(float(roi_gray.std()) - float(ring_gray.std())) if ring_gray.size else 0.0
    )
    boundary_strength = float(edge_map[crop_slice].mean()) / 255.0

    water_overlap = 0.0
    distance_to_water = float("inf")
    distance_to_water_boundary = float("inf")
    if water_mask is not None:
        roi_water = water_mask[crop_slice] == 255
        water_overlap = float(roi_water.mean()) if roi_water.size else 0.0
        distance_to_water = (
            0.0
            if np.any(roi_water)
            else float(water_distance_map[crop_slice].min())
        )
        distance_to_water_boundary = float(
            water_boundary_distance_map[crop_slice].min()
        )
    water_near = float(
        water_mask is not None
        and (
            water_overlap > 0.0
            or distance_to_water <= water_near_distance_px
            or distance_to_water_boundary <= water_near_distance_px
        )
    )

    aspect_similarity = max(0.0, 1.0 - abs(math.log(max(aspect, 1e-6) / 0.65)) / 2.2)
    contrast_score = min(1.0, color_contrast / 45.0)
    intensity_score = min(1.0, intensity_contrast / 55.0)
    texture_score = min(1.0, texture_difference / 35.0)
    person_score = (
        0.22 * aspect_similarity
        + 0.20 * solidity
        + 0.18 * extent
        + 0.18 * compactness
        + 0.12 * max(contrast_score, intensity_score)
        + 0.10 * min(1.0, 1.0 / max(contour_complexity, 1.0))
    )
    floating_score = (
        0.25 * water_near
        + 0.23 * contrast_score
        + 0.18 * intensity_score
        + 0.18 * texture_score
        + 0.16 * boundary_strength
    )
    if water_near and floating_score >= 0.25 and floating_score > person_score:
        category: CandidateCategory = "floating-object-like"
        score = floating_score
    elif person_score >= 0.34 and person_score >= floating_score:
        category = "person-like"
        score = person_score
    else:
        category = "unknown"
        score = max(person_score, floating_score)

    features = {
        "aspect_ratio": float(aspect),
        "solidity": float(solidity),
        "extent": float(extent),
        "compactness": float(compactness),
        "contour_complexity": float(contour_complexity),
        "local_color_contrast": float(color_contrast),
        "local_intensity_contrast": float(intensity_contrast),
        "texture_difference": float(texture_difference),
        "boundary_strength": float(boundary_strength),
        "water_overlap_ratio": float(water_overlap),
        "distance_to_water_px": (
            float(distance_to_water) if water_mask is not None else -1.0
        ),
        "distance_to_water_boundary_px": (
            float(distance_to_water_boundary) if water_mask is not None else -1.0
        ),
        "water_context_available": float(water_mask is not None),
        "water_near": float(water_near),
        "person_like_rule_score": float(person_score),
        "floating_object_like_rule_score": float(floating_score),
    }
    return ScoredProposal(
        bbox=bbox,
        area_pixels=component_area,
        category=category,
        features=features,
        rule_score=float(score),
        method=proposal.method,
    )
