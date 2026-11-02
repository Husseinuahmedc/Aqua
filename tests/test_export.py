"""Tests for explicit CSV export and overwrite protection."""

from pathlib import Path

import pandas as pd
import pytest

from app.exporting import (
    export_evaluation_csv,
    export_image_candidates_csv,
    export_image_candidates_json,
    export_json,
    export_records_csv,
    export_video_candidates_csv,
    export_video_candidates_json,
)
from core.evaluation.metrics import evaluate_masks
from core.rescue.pipeline import RescueConfig, analyze_rescue_image
import numpy as np


def test_export_records_to_unicode_path_and_reopen_csv(tmp_path: Path) -> None:
    path = tmp_path / "analysis Δ.csv"
    records = [
        {"frame_index": 0, "coverage_percent": 12.5},
        {"frame_index": 1, "coverage_percent": 14.0},
    ]

    exported = export_records_csv(path, records)
    loaded = pd.read_csv(exported)

    assert list(loaded.columns) == ["frame_index", "coverage_percent"]
    assert loaded["frame_index"].tolist() == [0, 1]
    assert loaded["coverage_percent"].tolist() == [12.5, 14.0]


def test_csv_export_refuses_overwrite_without_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    export_records_csv(path, [{"water_pixels": 4}])

    with pytest.raises(FileExistsError):
        export_records_csv(path, [{"water_pixels": 8}])

    export_records_csv(path, [{"water_pixels": 8}], overwrite=True)
    assert pd.read_csv(path)["water_pixels"].tolist() == [8]


def test_empty_export_can_include_explicit_headers(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"

    export_records_csv(path, [], columns=["frame_index", "coverage_percent"])

    assert path.read_text(encoding="utf-8").strip() == "frame_index,coverage_percent"


def test_evaluation_export_includes_metrics_and_counts(tmp_path: Path) -> None:
    prediction = np.array([[0, 255]], dtype=np.uint8)
    reference = np.array([[0, 255]], dtype=np.uint8)
    metrics = evaluate_masks(prediction, reference)
    path = tmp_path / "evaluation.csv"

    export_evaluation_csv(path, metrics)

    loaded = pd.read_csv(path)
    assert loaded.loc[0, "intersection_over_union"] == 1.0
    assert loaded.loc[0, "dice_coefficient"] == 1.0
    assert loaded.loc[0, "true_positive"] == 1


def test_csv_export_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_records_csv(tmp_path / "absent" / "metrics.csv", [{"a": 1}])


def test_json_export_round_trips_unicode_candidate_records_and_protects_source(
    tmp_path: Path,
) -> None:
    import json

    path = tmp_path / "candidate review Δ.json"
    payload = {
        "metadata": {"source": "roof image Δ.png", "candidate_count": 1},
        "candidates": [{"category": "person-like", "review_status": "needs_review"}],
    }

    export_json(path, payload)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        export_json(path, payload)


def test_consolidated_image_and_video_candidate_exports(tmp_path: Path) -> None:
    import json

    image = np.full((60, 80, 3), 100, dtype=np.uint8)
    image[18:36, 24:40] = (220, 30, 20)
    config = RescueConfig(methods=("color", "intensity"))
    analysis = analyze_rescue_image(image, image_id="sample-img", config=config)

    img_csv = tmp_path / "img_candidates.csv"
    img_json = tmp_path / "img_candidates.json"
    export_image_candidates_csv(
        img_csv,
        source_path=tmp_path / "sample.png",
        analysis=analysis,
        candidates=analysis.candidates,
    )
    export_image_candidates_json(
        img_json,
        source_path=tmp_path / "sample.png",
        analysis=analysis,
        candidates=analysis.candidates,
    )

    loaded_csv = pd.read_csv(img_csv)
    loaded_json = json.loads(img_json.read_text(encoding="utf-8"))
    assert "category_label" in loaded_csv.columns
    assert "review_status_label" in loaded_csv.columns
    assert loaded_json["metadata"]["image_id"] == "sample-img"
    assert len(loaded_json["candidates"]) == len(analysis.candidates)

    vid_csv = tmp_path / "vid_candidates.csv"
    vid_json = tmp_path / "vid_candidates.json"
    export_video_candidates_csv(
        vid_csv,
        source_video=tmp_path / "clip.mp4",
        config=config,
        candidate_records=[],
    )
    export_video_candidates_json(
        vid_json,
        source_video=tmp_path / "clip.mp4",
        config=config,
        frame_records=[{"frame_index": 0, "candidate_count": 0}],
        candidate_records=[],
    )
    assert "category_label" in pd.read_csv(vid_csv).columns
    assert json.loads(vid_json.read_text(encoding="utf-8"))["frames"][0]["frame_index"] == 0
