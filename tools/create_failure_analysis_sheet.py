"""Render development-only baseline/candidate failure comparisons.

Case selection reads saved development metrics only. The tool reruns the two
locked prediction functions solely to render masks/errors for selected
development images; it does not calculate or update benchmark metrics and
never opens holdout image or mask files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.water.experimental import (
    HSVColorConfig,
    MorphologyConfig,
    refine_mask_morphology,
    segment_hsv_color_threshold,
)
from core.water.segmentation import SegmentationConfig, segment_water


PANEL_WIDTH = 320
PANEL_HEIGHT = 180
PANEL_LABEL_HEIGHT = 26
CASE_HEADER_HEIGHT = 34
PANELS = (
    "Original",
    "Ground truth",
    "Original HSV prediction",
    "Selected HSV prediction",
    "Original errors: FP magenta / FN yellow",
    "Candidate errors: FP magenta / FN yellow",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(row["subset"] != "development" for row in rows):
        raise ValueError(f"Expected non-empty development-only results: {path}")
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate sample IDs in {path}.")
    return result


def _choose_case(
    rows: dict[str, dict[str, str]],
    role: str,
    metric: str,
    *,
    largest: bool,
    roles: dict[str, list[str]],
) -> None:
    values = [(sample_id, float(row[metric])) for sample_id, row in rows.items()]
    values.sort(key=lambda item: ((-item[1] if largest else item[1]), item[0]))
    roles.setdefault(values[0][0], []).append(role)


def _select_cases(
    baseline: dict[str, dict[str, str]],
    candidate: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    if baseline.keys() != candidate.keys():
        raise ValueError("Development metric files have different sample IDs.")
    deltas = {
        sample_id: {
            **candidate[sample_id],
            "iou_delta": str(
                float(candidate[sample_id]["iou"])
                - float(baseline[sample_id]["iou"])
            ),
        }
        for sample_id in candidate
    }
    roles: dict[str, list[str]] = {}
    _choose_case(baseline, "best original-baseline IoU", "iou", largest=True, roles=roles)
    _choose_case(candidate, "best selected-candidate IoU", "iou", largest=True, roles=roles)
    _choose_case(deltas, "largest candidate IoU gain", "iou_delta", largest=True, roles=roles)
    _choose_case(deltas, "largest candidate IoU reduction", "iou_delta", largest=False, roles=roles)
    _choose_case(
        candidate,
        "largest candidate false-positive count",
        "false_positive",
        largest=True,
        roles=roles,
    )
    _choose_case(
        candidate,
        "largest candidate false-negative count",
        "false_negative",
        largest=True,
        roles=roles,
    )
    _choose_case(candidate, "worst candidate IoU", "iou", largest=False, roles=roles)
    _choose_case(
        baseline,
        "largest baseline false-negative count",
        "false_negative",
        largest=True,
        roles=roles,
    )
    return roles


def _panel(image: np.ndarray, label: str) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.zeros(
        (PANEL_HEIGHT + PANEL_LABEL_HEIGHT, PANEL_WIDTH, 3),
        dtype=np.uint8,
    )
    panel[:PANEL_LABEL_HEIGHT] = (38, 43, 47)
    panel[PANEL_LABEL_HEIGHT:] = resized
    cv2.putText(
        panel,
        label,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def _error_map(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref_positive = reference == 255
    pred_positive = prediction == 255
    result = np.zeros((*reference.shape, 3), dtype=np.uint8)
    result[ref_positive & pred_positive] = (0, 150, 0)
    result[pred_positive & ~ref_positive] = (255, 0, 255)
    result[ref_positive & ~pred_positive] = (0, 255, 255)
    return result


def _predict_candidate(
    image_bgr: np.ndarray,
    selection: dict[str, Any],
) -> np.ndarray:
    parameters = selection["selected_parameters"]
    hsv_config = HSVColorConfig(**parameters["hsv"])
    morphology_config = MorphologyConfig(**parameters["morphology"])
    raw_mask = segment_hsv_color_threshold(image_bgr, hsv_config)
    return refine_mask_morphology(raw_mask, morphology_config)


def create_failure_sheet(
    data_root: Path,
    development_dir: Path,
    split_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Render metric-selected cases from development files only."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Failure-analysis output is not empty: {output_dir}")

    selection_path = development_dir / "candidate_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("selection_subset") != "development":
        raise ValueError("Candidate selection must be locked from development.")
    baseline_rows = _read_csv(
        development_dir / "baseline_hsv_default" / "per_image_metrics.csv"
    )
    candidate_rows = _read_csv(
        development_dir
        / selection["selected_experiment_id"]
        / "per_image_metrics.csv"
    )
    case_roles = _select_cases(baseline_rows, candidate_rows)
    split_ids: set[str] = set()
    with split_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["subset"] == "development":
                split_ids.add(row["sample_id"])
    if not set(case_roles).issubset(split_ids):
        raise ValueError("Selected visual cases are not all development IDs.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    row_images: list[np.ndarray] = []
    baseline_config = SegmentationConfig()
    for sample_id in sorted(case_roles):
        image_path = data_root / "images" / f"{sample_id}.jpg"
        mask_path = data_root / "masks" / f"{sample_id}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        reference = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or reference is None:
            raise OSError(f"Could not load development case {sample_id}.")

        # Visual-only inference on already-selected development images. Scores
        # are read from the saved benchmark CSVs, not recalculated here.
        baseline_prediction = segment_water(image.copy(), baseline_config).mask
        candidate_prediction = _predict_candidate(image.copy(), selection)
        if baseline_prediction.shape != reference.shape:
            raise ValueError(f"Baseline geometry mismatch for {sample_id}.")
        if candidate_prediction.shape != reference.shape:
            raise ValueError(f"Candidate geometry mismatch for {sample_id}.")

        base = baseline_rows[sample_id]
        candidate = candidate_rows[sample_id]
        header = (
            f"{sample_id} | {', '.join(case_roles[sample_id])} | "
            f"IoU baseline {float(base['iou']):.3f} → "
            f"candidate {float(candidate['iou']):.3f}"
        )
        panels = (
            _panel(image, "Original BGR"),
            _panel(reference, "Ground truth 0/255"),
            _panel(baseline_prediction, "Original HSV"),
            _panel(candidate_prediction, "Selected HSV"),
            _panel(
                _error_map(reference, baseline_prediction),
                "Base errors: TP green FP magenta FN yellow",
            ),
            _panel(
                _error_map(reference, candidate_prediction),
                "Candidate errors: TP green FP magenta FN yellow",
            ),
        )
        header_height = CASE_HEADER_HEIGHT
        case_height = header_height + panels[0].shape[0]
        case_canvas = np.zeros(
            (case_height, PANEL_WIDTH * len(panels), 3),
            dtype=np.uint8,
        )
        case_canvas[:header_height] = (27, 31, 34)
        cv2.putText(
            case_canvas,
            header[:210],
            (8, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        for index, panel in enumerate(panels):
            x_start = index * PANEL_WIDTH
            case_canvas[
                header_height : header_height + panel.shape[0],
                x_start : x_start + PANEL_WIDTH,
            ] = panel
        row_images.append(case_canvas)
        cases.append(
            {
                "sample_id": sample_id,
                "roles": case_roles[sample_id],
                "subset": "development",
                "baseline_iou": float(base["iou"]),
                "candidate_iou": float(candidate["iou"]),
                "iou_delta": float(candidate["iou"]) - float(base["iou"]),
                "baseline_false_positive": int(base["false_positive"]),
                "baseline_false_negative": int(base["false_negative"]),
                "candidate_false_positive": int(candidate["false_positive"]),
                "candidate_false_negative": int(candidate["false_negative"]),
            }
        )

    sheet = np.concatenate(row_images, axis=0)
    sheet_path = output_dir / "development_failure_comparison.png"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(f"Could not save comparison sheet: {sheet_path}")
    with (output_dir / "selected_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        fields = tuple(cases[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cases)

    metadata = {
        "subset": "development only",
        "development_case_count": len(cases),
        "selection_source": "Saved M6.1 development per-image metrics only.",
        "holdout_files_opened": False,
        "metrics_recomputed": False,
        "visual_inference_scope": (
            "Prediction masks regenerated only for selected development IDs "
            "to render the sheet; no new benchmark scores or tuning."
        ),
        "baseline_parameters": asdict(baseline_config),
        "candidate_selection_file": str(selection_path.resolve()),
        "candidate_experiment_id": selection["selected_experiment_id"],
        "split_file_sha256": _sha256(split_path),
        "cases": cases,
        "sheet": str(sheet_path.resolve()),
    }
    (output_dir / "visual_comparison_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    create_failure_sheet(
        args.data_root.resolve(),
        args.development_dir.resolve(),
        args.split_file.resolve(),
        args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
