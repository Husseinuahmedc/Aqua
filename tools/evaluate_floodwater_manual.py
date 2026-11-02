"""Evaluate the unchanged AquaRescue baseline on manual Floodwater pairs.

The utility reads an already-extracted manual evaluation set. It does not
download or modify source data. Each image is processed independently using
``SegmentationConfig()`` and compared with its paired 0/255 manual mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.evaluation.metrics import EvaluationMetrics, evaluate_masks
from core.water.regions import validate_binary_mask
from core.water.segmentation import SegmentationConfig, segment_water


DATASET_URL = "https://github.com/decide-ugent/floodwater-dataset"
DATASET_COMMIT = "84f8baa90a97ce2e0f8b2f946da5ec2797db5f0a"
DATASET_RELEASE = "floodwater-manual-v1.0.0"
SAMPLE_ID_PATTERN = re.compile(r"sample_[0-9]{6}")
SOURCE_FILES = (
    "core/water/segmentation.py",
    "core/water/regions.py",
    "core/preprocessing/transforms.py",
    "core/preprocessing/validation.py",
    "core/evaluation/metrics.py",
    "tools/evaluate_floodwater_manual.py",
)
PER_IMAGE_FIELDS = (
    "sample_id",
    "image_file",
    "mask_file",
    "status",
    "width_pixels",
    "height_pixels",
    "total_pixels",
    "reference_water_pixels",
    "predicted_water_pixels",
    "true_positive",
    "false_positive",
    "false_negative",
    "true_negative",
    "iou",
    "dice",
    "precision",
    "recall",
    "segmentation_seconds",
    "end_to_end_seconds",
    "error",
)


@dataclass(frozen=True)
class ManualPair:
    """Paths for one stable, paired manual evaluation sample."""

    sample_id: str
    image_path: Path
    mask_path: Path


def discover_pairs(data_root: Path, expected_count: int = 700) -> list[ManualPair]:
    """Discover strictly paired JPEG/PNG samples and verify the expected IDs."""
    image_directory = data_root / "images"
    mask_directory = data_root / "masks"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(
            f"Expected images/ and masks/ directories under {data_root}."
        )

    image_paths = sorted(path for path in image_directory.iterdir() if path.is_file())
    mask_paths = sorted(path for path in mask_directory.iterdir() if path.is_file())
    if any(path.suffix.lower() != ".jpg" for path in image_paths):
        raise ValueError("Manual image directory contains a non-JPEG file.")
    if any(path.suffix.lower() != ".png" for path in mask_paths):
        raise ValueError("Manual mask directory contains a non-PNG file.")

    images = {path.stem: path for path in image_paths}
    masks = {path.stem: path for path in mask_paths}
    if len(images) != len(image_paths) or len(masks) != len(mask_paths):
        raise ValueError("Duplicate sample IDs found across file extensions.")
    if images.keys() != masks.keys():
        missing_images = sorted(masks.keys() - images.keys())
        missing_masks = sorted(images.keys() - masks.keys())
        raise ValueError(
            "Unpaired samples: "
            f"missing images={missing_images[:10]}, "
            f"missing masks={missing_masks[:10]}"
        )

    sample_ids = sorted(images)
    if any(SAMPLE_ID_PATTERN.fullmatch(sample_id) is None for sample_id in sample_ids):
        raise ValueError("Sample IDs must use the sample_XXXXXX naming scheme.")
    expected_ids = {f"sample_{index:06d}" for index in range(1, expected_count + 1)}
    if set(sample_ids) != expected_ids:
        raise ValueError(
            f"Expected IDs sample_000001 through sample_{expected_count:06d}; "
            f"found {len(sample_ids)} paired IDs."
        )

    return [
        ManualPair(sample_id, images[sample_id], masks[sample_id])
        for sample_id in sample_ids
    ]


def _metric_record(metrics: EvaluationMetrics) -> dict[str, float | int | None]:
    return {
        "iou": metrics.intersection_over_union,
        "dice": metrics.dice_coefficient,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "true_negative": metrics.true_negative,
        "total_pixels": metrics.total_pixels,
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate macro-image and pixel/micro aggregates for successful pairs."""
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        return {
            "n_pairs_discovered": len(rows),
            "n_pairs_evaluated": 0,
            "n_pair_errors": len(rows),
            "aggregation": {
                "macro": {
                    name: {"mean": None, "n_defined": 0, "n_total": 0}
                    for name in ("iou", "dice", "precision", "recall")
                },
                "micro": {
                    "iou": None,
                    "dice": None,
                    "precision": None,
                    "recall": None,
                    "true_positive": 0,
                    "false_positive": 0,
                    "false_negative": 0,
                    "true_negative": 0,
                    "total_pixels": 0,
                },
            },
            "timing": {
                "segmentation_seconds_total": 0.0,
                "segmentation_seconds_mean": None,
                "segmentation_seconds_median": None,
                "segmentation_seconds_p95": None,
                "images_per_segmentation_second": None,
            },
        }

    tp = sum(int(row["true_positive"]) for row in successful)
    fp = sum(int(row["false_positive"]) for row in successful)
    fn = sum(int(row["false_negative"]) for row in successful)
    tn = sum(int(row["true_negative"]) for row in successful)
    pixel_count = sum(int(row["total_pixels"]) for row in successful)
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    predicted_positive = tp + fp
    reference_positive = tp + fn

    macro: dict[str, Any] = {}
    for metric_name in ("iou", "dice", "precision", "recall"):
        values = [
            row[metric_name]
            for row in successful
            if row[metric_name] is not None
        ]
        macro[metric_name] = {
            "mean": statistics.fmean(values) if values else None,
            "n_defined": len(values),
            "n_total": len(successful),
        }

    times = [float(row["segmentation_seconds"]) for row in successful]
    resolution_counts: dict[str, int] = {}
    for row in successful:
        dimensions = f"{row['width_pixels']}x{row['height_pixels']}"
        resolution_counts[dimensions] = resolution_counts.get(dimensions, 0) + 1
    empty_reference_masks = sum(
        int(row["reference_water_pixels"]) == 0 for row in successful
    )
    all_water_reference_masks = sum(
        int(row["reference_water_pixels"]) == int(row["total_pixels"])
        for row in successful
    )
    return {
        "n_pairs_discovered": len(rows),
        "n_pairs_evaluated": len(successful),
        "n_pair_errors": len(rows) - len(successful),
        "dataset_integrity": {
            "resolution_counts": resolution_counts,
            "empty_reference_masks": empty_reference_masks,
            "all_water_reference_masks": all_water_reference_masks,
        },
        "aggregation": {
            "macro": macro,
            "micro": {
            "iou": tp / union if union else 1.0,
            "dice": 2 * tp / dice_denominator if dice_denominator else 1.0,
                "precision": tp / predicted_positive if predicted_positive else None,
                "recall": tp / reference_positive if reference_positive else None,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "total_pixels": pixel_count,
            },
        },
        "timing": {
            "segmentation_seconds_total": sum(times),
            "segmentation_seconds_mean": statistics.fmean(times),
            "segmentation_seconds_median": statistics.median(times),
            "segmentation_seconds_p95": float(np.percentile(times, 95)),
            "images_per_segmentation_second": len(times) / sum(times)
            if sum(times)
            else None,
        },
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(repo_root: Path) -> dict[str, str]:
    return {
        name: _sha256_file(repo_root / name)
        for name in SOURCE_FILES
    }


def _git_state(repo_root: Path) -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "working_tree_dirty": True}
    return {"revision": revision, "working_tree_dirty": dirty}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _preview_panel(image: np.ndarray, label: str) -> np.ndarray:
    max_width, max_height = 480, 270
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panel = np.zeros((max_height + 34, max_width, 3), dtype=np.uint8)
    x_offset = (max_width - size[0]) // 2
    y_offset = 34 + (max_height - size[1]) // 2
    panel[y_offset : y_offset + size[1], x_offset : x_offset + size[0]] = resized
    cv2.putText(
        panel,
        label,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


def _save_failure_preview(
    pair: ManualPair,
    output_path: Path,
    config: SegmentationConfig,
) -> None:
    image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or reference is None:
        raise OSError(f"Cannot reload sample {pair.sample_id} for preview.")
    prediction = segment_water(image, config).mask
    validate_binary_mask(reference)

    reference_overlay = image.copy()
    reference_overlay[reference == 255] = (30, 190, 30)
    reference_overlay = cv2.addWeighted(image, 0.48, reference_overlay, 0.52, 0)
    prediction_overlay = image.copy()
    prediction_overlay[prediction == 255] = (210, 100, 20)
    prediction_overlay = cv2.addWeighted(image, 0.48, prediction_overlay, 0.52, 0)
    errors = np.zeros_like(image)
    reference_positive = reference == 255
    prediction_positive = prediction == 255
    errors[reference_positive & prediction_positive] = (0, 150, 0)
    errors[prediction_positive & ~reference_positive] = (255, 0, 255)
    errors[reference_positive & ~prediction_positive] = (0, 255, 255)

    panels = [
        _preview_panel(image, "Original"),
        _preview_panel(reference_overlay, "Manual mask · green"),
        _preview_panel(prediction_overlay, "Prediction · blue"),
        _preview_panel(errors, "TP green · FP magenta · FN yellow"),
    ]
    montage = np.concatenate(panels, axis=1)
    cv2.putText(
        montage,
        pair.sample_id,
        (montage.shape[1] - 210, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_path), montage):
        raise OSError(f"Cannot write failure preview: {output_path}")


def run_evaluation(
    data_root: Path,
    output_dir: Path,
    *,
    expected_count: int = 700,
    archive_path: Path | None = None,
    failure_preview_count: int = 12,
) -> dict[str, Any]:
    """Evaluate all discovered pairs and write machine-readable results."""
    if expected_count < 1:
        raise ValueError("Expected sample count must be at least one.")
    if failure_preview_count < 0:
        raise ValueError("Failure preview count cannot be negative.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    pairs = discover_pairs(data_root, expected_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SegmentationConfig()
    rows: list[dict[str, Any]] = []
    evaluation_started = time.perf_counter()

    for pair in pairs:
        sample_started = time.perf_counter()
        row: dict[str, Any] = {
            "sample_id": pair.sample_id,
            "image_file": pair.image_path.relative_to(data_root).as_posix(),
            "mask_file": pair.mask_path.relative_to(data_root).as_posix(),
            "status": "error",
            "error": "",
        }
        try:
            image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise OSError(f"Cannot decode JPEG image: {pair.image_path}")
            reference = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
            if reference is None:
                raise OSError(f"Cannot decode PNG mask: {pair.mask_path}")
            validate_binary_mask(reference)
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Decoded image must be uint8 BGR with three channels.")
            if image.shape[:2] != reference.shape:
                raise ValueError(
                    "Image/mask shape mismatch: "
                    f"{image.shape[:2]} vs {reference.shape}."
                )

            segmentation_started = time.perf_counter()
            prediction = segment_water(image, config).mask
            segmentation_seconds = time.perf_counter() - segmentation_started
            metrics = evaluate_masks(prediction, reference)
            row.update(_metric_record(metrics))
            row.update(
                {
                    "status": "ok",
                    "width_pixels": int(image.shape[1]),
                    "height_pixels": int(image.shape[0]),
                    "reference_water_pixels": int(np.count_nonzero(reference == 255)),
                    "predicted_water_pixels": int(np.count_nonzero(prediction == 255)),
                    "segmentation_seconds": segmentation_seconds,
                    "error": "",
                }
            )
        except (OSError, TypeError, ValueError, cv2.error) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["segmentation_seconds"] = None
        row["end_to_end_seconds"] = time.perf_counter() - sample_started
        rows.append(row)

    aggregate = aggregate_metrics(rows)
    evaluation_seconds = time.perf_counter() - evaluation_started
    successful = [row for row in rows if row["status"] == "ok"]
    lowest_iou = sorted(successful, key=lambda row: (row["iou"], row["sample_id"]))
    selected_failures = lowest_iou[:failure_preview_count]
    preview_directory = output_dir / "failure_previews"
    preview_seconds = 0.0
    if selected_failures:
        preview_directory.mkdir(exist_ok=True)
        pairs_by_id = {pair.sample_id: pair for pair in pairs}
        preview_started = time.perf_counter()
        for row in selected_failures:
            _save_failure_preview(
                pairs_by_id[row["sample_id"]],
                preview_directory / f"{row['sample_id']}.png",
                config,
            )
        preview_seconds = time.perf_counter() - preview_started

    failure_rows = [
        {
            "rank": rank,
            "sample_id": row["sample_id"],
            "iou": row["iou"],
            "dice": row["dice"],
            "precision": row["precision"],
            "recall": row["recall"],
            "false_positive": row["false_positive"],
            "false_negative": row["false_negative"],
            "preview_file": f"failure_previews/{row['sample_id']}.png",
        }
        for rank, row in enumerate(selected_failures, start=1)
    ]
    _write_csv(output_dir / "per_image_metrics.csv", rows, PER_IMAGE_FIELDS)
    _write_csv(
        output_dir / "lowest_iou_samples.csv",
        failure_rows,
        (
            "rank",
            "sample_id",
            "iou",
            "dice",
            "precision",
            "recall",
            "false_positive",
            "false_negative",
            "preview_file",
        ),
    )

    repo_root = Path(__file__).resolve().parents[1]
    expected_resolution_counts = {"1280x720": 466, "1920x1080": 234}
    observed_resolution_counts = aggregate["dataset_integrity"]["resolution_counts"]
    resolution_distribution_matches = (
        expected_count != 700
        or observed_resolution_counts == expected_resolution_counts
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Floodwater Dataset manual evaluation set",
            "release": DATASET_RELEASE,
            "repository_url": DATASET_URL,
            "repository_commit": DATASET_COMMIT,
            "license_as_stated_by_repository": "GNU GPL-3.0",
            "annotation": (
                "Directly annotated visible-water masks; "
                "0=background, 255=water."
            ),
            "data_root": str(data_root.resolve()),
            "archive_path": str(archive_path.resolve()) if archive_path else None,
            "archive_sha256": _sha256_file(archive_path) if archive_path else None,
            "pairs_discovered": len(pairs),
        },
        "protocol": {
            "selection": (
                "All manual evaluation pairs in the separately packaged "
                "700-pair set."
            ),
            "parameter_tuning": (
                "None. Default SegmentationConfig was fixed before "
                "evaluating this set."
            ),
            "reference_handling": (
                "Read as uint8 PNG; require matching geometry and exact "
                "values 0/255; no resize or relabeling."
            ),
            "input_handling": (
                "JPEG decoded by OpenCV as BGR uint8; source dimensions preserved."
            ),
            "aggregation": (
                "Macro is unweighted mean of per-image metrics; micro "
                "pools pixel confusion counts."
            ),
            "undefined_metrics": (
                "Per existing evaluate_masks convention: precision/recall "
                "are null when their denominator is zero; IoU/Dice are 1 "
                "when both masks are empty, otherwise 0 when one is empty."
            ),
            "failure_case_selection": (
                "Lowest per-image IoU; previews are regenerated with the "
                "same configuration and excluded from reported timings."
            ),
        },
        "dataset_integrity": {
            **aggregate["dataset_integrity"],
            "expected_release_resolution_counts": expected_resolution_counts,
            "matches_release_resolution_counts": resolution_distribution_matches,
        },
        "algorithm": {
            "name": "Existing traditional HSV threshold water segmentation",
            "parameters": asdict(config),
            "implementation": "core.water.segmentation.segment_water",
            "source_sha256": _source_fingerprint(repo_root),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "git": _git_state(repo_root),
        },
        "run": {
            "command": [sys.executable, *sys.argv],
            "output_directory": str(output_dir.resolve()),
            "evaluation_wall_seconds": evaluation_seconds,
            "failure_preview_wall_seconds": preview_seconds,
            "pair_errors": [
                {"sample_id": row["sample_id"], "error": row["error"]}
                for row in rows
                if row["status"] != "ok"
            ],
        },
    }
    run_complete = (
        aggregate["n_pair_errors"] == 0 and resolution_distribution_matches
    )
    summary = {
        "status": "complete" if run_complete else "incomplete",
        **aggregate,
        "lowest_iou_samples": failure_rows,
    }
    _write_json(output_dir / "run_metadata.json", metadata)
    _write_json(output_dir / "aggregate_metrics.json", summary)

    print(
        json.dumps(
            {
                "status": summary["status"],
                "evaluated": aggregate["n_pairs_evaluated"],
                "errors": aggregate["n_pair_errors"],
                "macro": aggregate["aggregation"]["macro"],
                "micro": aggregate["aggregation"]["micro"],
                "output_directory": str(output_dir.resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )
    if not run_complete:
        raise RuntimeError(
            "Evaluation wrote results with pair errors or a dataset integrity "
            "mismatch; inspect aggregate_metrics.json and run_metadata.json."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Extracted floodwater_manual directory containing images/ and masks/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty directory for CSV/JSON results and failure previews.",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional official ZIP archive to record its SHA-256 in provenance.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=700,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--failure-preview-count",
        type=int,
        default=12,
        help="Save previews for this many lowest-IoU samples (default: 12).",
    )
    args = parser.parse_args()
    run_evaluation(
        args.data_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        expected_count=args.expected_count,
        archive_path=args.archive.expanduser().resolve() if args.archive else None,
        failure_preview_count=args.failure_preview_count,
    )


if __name__ == "__main__":
    main()
