"""Independent local-contrast, texture, edge, and adaptive proposals."""

import cv2
import numpy as np

from core.rescue.candidates import CandidateProposal
from core.rescue.preparation import ImageRepresentations
from core.water.regions import extract_regions


def _refine(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply a small open/close cleanup while keeping the source dimensions."""
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _components(
    mask: np.ndarray,
    *,
    method: str,
    min_area_pixels: int,
    max_area_fraction: float,
) -> list[CandidateProposal]:
    """Extract connected components by reusing core.water.regions.extract_regions."""
    image_area = mask.shape[0] * mask.shape[1]
    max_area = max_area_fraction * image_area
    proposals: list[CandidateProposal] = []
    for region in extract_regions(mask, min_area_pixels=min_area_pixels):
        if region.area_pixels > max_area:
            continue
        contour_points: tuple[tuple[int, int], ...] = ()
        if region.contour is not None:
            contour_points = tuple(
                (int(point[0][0]), int(point[0][1])) for point in region.contour
            )
        proposals.append(
            CandidateProposal(
                region.x,
                region.y,
                region.width,
                region.height,
                region.area_pixels,
                method,
                contour_points,
            )
        )
    return proposals


def local_color_contrast_mask(
    representations: ImageRepresentations,
    *,
    threshold: float = 14.0,
    background_kernel_size: int = 31,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Threshold Euclidean Lab distance from a blurred local background."""
    if threshold < 0 or background_kernel_size < 3 or background_kernel_size % 2 == 0:
        raise ValueError(
            "Color contrast needs a non-negative threshold and odd blur kernel >= 3."
        )
    local_background = cv2.GaussianBlur(
        representations.lab, (background_kernel_size, background_kernel_size), 0
    )
    delta = cv2.absdiff(representations.lab, local_background).astype(np.float32)
    magnitude = np.linalg.norm(delta, axis=2)
    mask = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
    return _refine(mask, morphology_kernel_size)


def local_intensity_contrast_mask(
    representations: ImageRepresentations,
    *,
    threshold: float = 18.0,
    background_kernel_size: int = 31,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Threshold absolute grayscale deviation from a Gaussian local mean."""
    if threshold < 0 or background_kernel_size < 3 or background_kernel_size % 2 == 0:
        raise ValueError(
            "Intensity contrast needs a non-negative threshold and odd blur kernel >= 3."
        )
    background = cv2.GaussianBlur(
        representations.gray, (background_kernel_size, background_kernel_size), 0
    )
    difference = cv2.absdiff(representations.gray, background)
    mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    return _refine(mask, morphology_kernel_size)


def texture_difference_mask(
    representations: ImageRepresentations,
    *,
    threshold: float = 7.0,
    local_kernel_size: int = 7,
    context_kernel_size: int = 31,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Propose areas whose local standard deviation differs from context."""
    if (
        threshold < 0
        or local_kernel_size < 3
        or local_kernel_size % 2 == 0
        or context_kernel_size <= local_kernel_size
        or context_kernel_size % 2 == 0
    ):
        raise ValueError("Texture windows must be odd, >= 3, with context larger than local.")
    gray = representations.gray.astype(np.float32)
    local_mean = cv2.boxFilter(gray, cv2.CV_32F, (local_kernel_size, local_kernel_size))
    local_square_mean = cv2.boxFilter(
        gray * gray, cv2.CV_32F, (local_kernel_size, local_kernel_size)
    )
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    context_mean = cv2.GaussianBlur(
        local_std, (context_kernel_size, context_kernel_size), 0
    )
    difference = cv2.absdiff(local_std, context_mean)
    mask = np.where(difference >= threshold, 255, 0).astype(np.uint8)
    return _refine(mask, morphology_kernel_size)


def edge_proposal_mask(
    representations: ImageRepresentations,
    *,
    threshold_low: int = 50,
    threshold_high: int = 140,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Build a closed Canny edge map as a boundary-driven region proposal."""
    if not 0 <= threshold_low < threshold_high <= 255:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255.")
    edges = cv2.Canny(representations.gray, threshold_low, threshold_high)
    if morphology_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size)
        )
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def adaptive_threshold_mask(
    representations: ImageRepresentations,
    *,
    block_size: int = 31,
    constant: float = 5.0,
    morphology_kernel_size: int = 3,
) -> np.ndarray:
    """Propose locally darker and brighter regions using both polarities."""
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("Adaptive threshold block size must be an odd integer >= 3.")
    if constant < 0:
        raise ValueError("Adaptive threshold constant cannot be negative.")
    dark = cv2.adaptiveThreshold(
        representations.gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        constant,
    )
    bright = cv2.adaptiveThreshold(
        representations.gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constant,
    )
    combined = cv2.bitwise_or(dark, bright)
    return _refine(combined, morphology_kernel_size)


def generate_proposals(
    representations: ImageRepresentations,
    *,
    methods: tuple[str, ...],
    min_area_pixels: int,
    max_area_fraction: float,
    color_threshold: float,
    intensity_threshold: float,
    texture_threshold: float,
    edge_threshold_low: int,
    edge_threshold_high: int,
    adaptive_block_size: int,
    adaptive_constant: float,
    morphology_kernel_size: int,
) -> tuple[CandidateProposal, ...]:
    """Run only selected proposal methods and preserve each cue's identity."""
    method_masks = {
        "color": lambda: local_color_contrast_mask(
            representations,
            threshold=color_threshold,
            morphology_kernel_size=morphology_kernel_size,
        ),
        "intensity": lambda: local_intensity_contrast_mask(
            representations,
            threshold=intensity_threshold,
            morphology_kernel_size=morphology_kernel_size,
        ),
        "texture": lambda: texture_difference_mask(
            representations,
            threshold=texture_threshold,
            morphology_kernel_size=morphology_kernel_size,
        ),
        "edge": lambda: edge_proposal_mask(
            representations,
            threshold_low=edge_threshold_low,
            threshold_high=edge_threshold_high,
            morphology_kernel_size=morphology_kernel_size,
        ),
        "adaptive": lambda: adaptive_threshold_mask(
            representations,
            block_size=adaptive_block_size,
            constant=adaptive_constant,
            morphology_kernel_size=morphology_kernel_size,
        ),
    }
    unknown = set(methods) - method_masks.keys()
    if unknown:
        raise ValueError(f"Unknown candidate proposal method(s): {', '.join(sorted(unknown))}.")
    proposals: list[CandidateProposal] = []
    for method in methods:
        mask = method_masks[method]()
        if method == "edge":
            contours, _hierarchy = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            image_area = mask.shape[0] * mask.shape[1]
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = int(max(cv2.contourArea(contour), width * height * 0.1))
                if area >= min_area_pixels and area <= max_area_fraction * image_area:
                    proposals.append(
                        CandidateProposal(
                            x,
                            y,
                            width,
                            height,
                            area,
                            method,
                            tuple(
                                (int(point[0][0]), int(point[0][1]))
                                for point in contour
                            ),
                        )
                    )
        else:
            proposals.extend(
                _components(
                    mask,
                    method=method,
                    min_area_pixels=min_area_pixels,
                    max_area_fraction=max_area_fraction,
                )
            )
    return tuple(proposals)
