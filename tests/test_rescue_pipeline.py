"""Synthetic correctness tests for independent rescue candidate cues."""

import numpy as np
import pytest

from core.rescue.candidates import (
    ScoredProposal,
    box_iou,
    describe_candidate_reasoning,
    format_category_label,
    format_review_status,
    fuse_scored_proposals,
)
from core.rescue.pipeline import RescueConfig, analyze_rescue_image
from core.rescue.preparation import prepare_rgb_image
from core.rescue.proposals import (
    adaptive_threshold_mask,
    edge_proposal_mask,
    local_color_contrast_mask,
    local_intensity_contrast_mask,
    texture_difference_mask,
)
from core.rescue.visualization import draw_candidate_boxes_rgb


def _image_with_patch() -> np.ndarray:
    image = np.full((80, 100, 3), 110, dtype=np.uint8)
    image[25:45, 35:52] = (220, 35, 25)
    return image


@pytest.mark.parametrize(
    "method",
    [
        local_color_contrast_mask,
        local_intensity_contrast_mask,
        texture_difference_mask,
        edge_proposal_mask,
        adaptive_threshold_mask,
    ],
)
def test_each_enabled_proposal_method_returns_same_geometry_binary_mask(method) -> None:
    image = _image_with_patch()
    representations = prepare_rgb_image(image)
    mask = method(representations)

    assert mask.shape == image.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_pipeline_preserves_input_geometry_and_proposes_without_water_mask() -> None:
    image = _image_with_patch()
    original = image.copy()
    result = analyze_rescue_image(
        image,
        image_id="synthetic image Δ",
        config=RescueConfig(methods=("color", "intensity")),
    )

    assert image.shape == original.shape
    assert np.array_equal(image, original)
    assert (result.image_width, result.image_height) == (100, 80)
    assert not result.water_context_used
    assert result.candidates
    assert all(candidate.image_id == "synthetic image Δ" for candidate in result.candidates)
    assert all(candidate.review_status == "needs_review" for candidate in result.candidates)
    assert all(0.0 <= candidate.rule_score <= 1.0 for candidate in result.candidates)
    assert all(candidate.bbox[2] > 0 and candidate.bbox[3] > 0 for candidate in result.candidates)


def test_water_mask_is_context_only_and_does_not_suppress_outside_proposals() -> None:
    image = _image_with_patch()
    water = np.zeros(image.shape[:2], dtype=np.uint8)
    water[60:75, 65:90] = 255
    config = RescueConfig(methods=("color", "intensity"))

    without_water = analyze_rescue_image(image, config=config)
    with_water = analyze_rescue_image(image, water_mask=water, config=config)

    assert without_water.candidates
    assert with_water.water_context_used
    assert with_water.candidates
    assert any(
        x < 60 and y < 55
        for candidate in with_water.candidates
        for x, y, _width, _height in (candidate.bbox,)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"methods": ()},
        {"methods": ("color", "color")},
        {"methods": ("model",)},
        {"min_area_pixels": 1.5},
        {"max_area_fraction": 1.1},
        {"fusion_iou_threshold": -0.1},
        {"adaptive_block_size": 2},
    ],
)
def test_rescue_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        RescueConfig(**kwargs)


def test_pipeline_validates_water_mask_and_frame_metadata() -> None:
    image = _image_with_patch()
    with pytest.raises(ValueError, match="dimensions"):
        analyze_rescue_image(image, water_mask=np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="exactly 0 or 255"):
        analyze_rescue_image(
            image,
            water_mask=np.full(image.shape[:2], 1, dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="Frame index"):
        analyze_rescue_image(image, frame_index=-1)


def test_iou_fusion_suppresses_duplicate_proposals_and_keeps_provenance() -> None:
    assert box_iou((0, 0, 10, 10), (5, 5, 10, 10)) == pytest.approx(25 / 175)
    candidates = fuse_scored_proposals(
        (
            ScoredProposal((10, 10, 20, 20), 300, "unknown", {}, 0.8, "color"),
            ScoredProposal((11, 11, 20, 20), 290, "unknown", {}, 0.7, "edge"),
            ScoredProposal((60, 60, 10, 12), 120, "unknown", {}, 0.4, "texture"),
        ),
        image_id="image-1",
        frame_index=4,
        iou_threshold=0.45,
    )

    assert len(candidates) == 2
    assert candidates[0].source_methods == ("color", "edge")
    assert candidates[0].frame_index == 4
    assert candidates[0].candidate_id == "image-1#candidate-0001"
    assert candidates[1].bbox == (60, 60, 10, 12)


def test_candidate_labels_reasoning_and_selected_highlight() -> None:
    image = _image_with_patch()
    result = analyze_rescue_image(
        image,
        image_id="highlight-check",
        config=RescueConfig(methods=("color", "intensity")),
    )
    candidate = result.candidates[0]
    record = candidate.to_record()

    assert format_category_label("person-like") == "Potential Person"
    assert format_category_label("floating-object-like") == "Floating Object"
    assert format_category_label("unknown") == "Unknown Candidate"
    assert format_review_status("needs_review") == "Requires Human Review"
    assert format_review_status("confirmed") == "Confirmed Candidate"
    assert format_review_status("false_alarm") == "False Alarm"
    assert record["category_label"] in (
        "Potential Person",
        "Floating Object",
        "Unknown Candidate",
    )
    assert record["review_status_label"] == "Requires Human Review"
    assert "human verification" in describe_candidate_reasoning(candidate).lower()

    unselected = draw_candidate_boxes_rgb(image, result.candidates)
    selected = draw_candidate_boxes_rgb(
        image,
        result.candidates,
        selected_candidate_id=candidate.candidate_id,
    )
    assert unselected.shape == image.shape
    assert selected.shape == image.shape
    assert np.any(selected != unselected)
