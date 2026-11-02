"""Local end-to-end test for the candidate evaluation command boundary."""

import json
from pathlib import Path

import numpy as np

from core.rescue.evaluation import CandidateEvaluation
from core.preprocessing.image_io import save_image
from tools.evaluate_rescue_candidates import _parser, run_evaluation


def test_local_coco_candidate_evaluation_records_provenance_and_metrics(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "source images"
    images_dir.mkdir()
    image = np.full((80, 90, 3), 100, dtype=np.uint8)
    image[28:47, 34:53] = (220, 30, 20)
    save_image(images_dir / "rescue image Δ.png", image, color_space="RGB")
    annotations_path = tmp_path / "validation.json"
    annotations_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 42,
                        "file_name": "rescue image Δ.png",
                        "width": 90,
                        "height": 80,
                        "split": "validation",
                    }
                ],
                "categories": [{"id": 7, "name": "swimmer"}],
                "annotations": [
                    {
                        "id": 99,
                        "image_id": 42,
                        "category_id": 7,
                        "bbox": [34, 28, 19, 19],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _parser().parse_args(
        [
            "--dataset",
            "seadronesssee",
            "--annotations",
            str(annotations_path),
            "--images-dir",
            str(images_dir),
            "--split",
            "validation",
            "--dataset-version",
            "synthetic-fixture-v1",
            "--methods",
            "color",
            "intensity",
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    report = run_evaluation(args)

    assert report["image_count"] == 1
    assert report["dataset_version"] == "synthetic-fixture-v1"
    assert report["annotation_classes"] == ["swimmer"]
    assert report["water_context_used"] is False
    assert report["image_sha256_by_id"]["42"]
    assert report["image_inventory_sha256"]
    assert report["evaluation"]["complete_negative_labels"] is False
    assert report["configuration"]["methods"] == ("color", "intensity")
    assert len(report["evaluation"]["metrics"]) == 6


def test_candidate_evaluation_record_has_explicit_unit_mapping() -> None:
    evaluation = CandidateEvaluation(
        metrics=(),
        candidate_count_per_image={},
        candidate_count_per_frame={"clip#0": 2},
        processing_time_ms_per_image={},
        processing_time_ms_per_frame={"clip#0": 4.0},
        complete_negative_labels=False,
        sample_unit="frame",
    )

    record = evaluation.to_record()

    assert record["sample_unit"] == "frame"
    assert record["candidate_count_per_frame"] == {"clip#0": 2}
    assert record["candidate_count_per_image"] == {}
