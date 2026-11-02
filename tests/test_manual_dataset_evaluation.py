"""Offline tests for the reproducible manual-set evaluation tool."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate_floodwater_manual import run_evaluation


def _write_pair(
    root: Path,
    sample_id: str,
    bgr: tuple[int, int, int],
    mask_value: int,
) -> None:
    image = np.full((16, 16, 3), bgr, dtype=np.uint8)
    mask = np.full((16, 16), mask_value, dtype=np.uint8)
    assert cv2.imwrite(str(root / "images" / f"{sample_id}.jpg"), image)
    assert cv2.imwrite(str(root / "masks" / f"{sample_id}.png"), mask)


def test_manual_evaluation_tool_runs_offline_and_writes_reproducible_records(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "manual"
    (data_root / "images").mkdir(parents=True)
    (data_root / "masks").mkdir()
    _write_pair(data_root, "sample_000001", (255, 0, 0), 255)
    _write_pair(data_root, "sample_000002", (0, 0, 255), 0)
    output_dir = tmp_path / "results"

    summary = run_evaluation(
        data_root,
        output_dir,
        expected_count=2,
        failure_preview_count=1,
    )

    assert summary["status"] == "complete"
    assert summary["n_pairs_evaluated"] == 2
    assert summary["aggregation"]["macro"]["iou"]["mean"] == 1.0
    assert summary["aggregation"]["micro"]["dice"] == 1.0
    with (output_dir / "per_image_metrics.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in rows] == [
        "sample_000001",
        "sample_000002",
    ]
    assert rows[0]["iou"] == "1.0"
    assert rows[1]["precision"] == ""
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["algorithm"]["parameters"]["hue_min"] == 80
    assert metadata["dataset"]["pairs_discovered"] == 2
    assert list((output_dir / "failure_previews").glob("*.png"))
