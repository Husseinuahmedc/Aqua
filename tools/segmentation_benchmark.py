"""Development/holdout benchmark for traditional-DIP water masks.

The split is based only on stable sample IDs. The ``dev`` command reads only
development images and masks and locks the highest macro-IoU candidate. The
``holdout`` command then reads holdout data once for that locked candidate and
the unchanged original baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from core.evaluation.metrics import EvaluationMetrics, evaluate_masks
from core.preprocessing.validation import validate_image
from core.water.experimental import (
    AdaptiveThresholdConfig,
    HSVColorConfig,
    MorphologyConfig,
    OtsuThresholdConfig,
    RGBThresholdConfig,
    refine_mask_morphology,
    segment_adaptive_threshold,
    segment_hsv_color_threshold,
    segment_otsu_threshold,
    segment_rgb_threshold,
)
from core.water.regions import validate_binary_mask
from core.water.segmentation import SegmentationConfig, segment_water
from tools.evaluate_floodwater_manual import (
    DATASET_COMMIT,
    DATASET_RELEASE,
    discover_pairs,
)


SPLIT_SEED = 20261101
DEVELOPMENT_FRACTION = 0.8
BASELINE_ID = "baseline_hsv_default"
PER_IMAGE_FIELDS = (
    "sample_id",
    "subset",
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
    "empty_prediction",
    "segmentation_seconds",
    "error",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One independently parameterized segmentation method/configuration."""

    experiment_id: str
    algorithm: str
    parameters: dict[str, Any]
    predict: Callable[[np.ndarray], np.ndarray]


def deterministic_split(
    sample_ids: list[str],
    *,
    seed: int = SPLIT_SEED,
    development_fraction: float = DEVELOPMENT_FRACTION,
) -> list[dict[str, Any]]:
    """Split IDs by seeded SHA-256 ranking without reading image/mask data."""
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_ids must be unique.")
    if len(sample_ids) < 2:
        raise ValueError("At least two sample IDs are required for a split.")
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    ranked = sorted(
        sample_ids,
        key=lambda sample_id: hashlib.sha256(
            f"{seed}:{sample_id}".encode("utf-8")
        ).hexdigest(),
    )
    development_count = int(len(ranked) * development_fraction)
    if development_count < 1 or development_count >= len(ranked):
        raise ValueError("Split fraction leaves an empty development or holdout set.")
    development_ids = set(ranked[:development_count])
    return [
        {
            "sample_id": sample_id,
            "subset": "development" if sample_id in development_ids else "holdout",
            "rank": rank,
        }
        for rank, sample_id in enumerate(ranked, start=1)
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...] | list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    source_files = (
        "core/water/segmentation.py",
        "core/water/experimental.py",
        "core/water/regions.py",
        "core/preprocessing/validation.py",
        "core/preprocessing/transforms.py",
        "core/evaluation/metrics.py",
        "tools/evaluate_floodwater_manual.py",
        "tools/segmentation_benchmark.py",
    )
    return {name: _sha256(root / name) for name in source_files}


def create_split(
    data_root: Path,
    output_dir: Path,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    """Write a new ID-only 80/20 split; never open image or mask contents."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Split output directory is not empty: {output_dir}")
    pairs = discover_pairs(data_root)
    assignments = deterministic_split([pair.sample_id for pair in pairs])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split.csv"
    _write_csv(split_path, assignments, ("sample_id", "subset", "rank"))
    development_ids = [
        row["sample_id"] for row in assignments if row["subset"] == "development"
    ]
    holdout_ids = [
        row["sample_id"] for row in assignments if row["subset"] == "holdout"
    ]
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_release": DATASET_RELEASE,
        "dataset_repository_commit": DATASET_COMMIT,
        "data_root": str(data_root.resolve()),
        "archive_path": str(archive_path.resolve()) if archive_path else None,
        "archive_sha256": _sha256(archive_path) if archive_path else None,
        "sample_count": len(assignments),
        "seed": SPLIT_SEED,
        "assignment_method": (
            "Sort sample IDs by SHA-256(seed + ':' + sample_id); first 80% "
            "development, remaining 20% holdout. Uses IDs only."
        ),
        "development_count": len(development_ids),
        "holdout_count": len(holdout_ids),
        "split_file": str(split_path.resolve()),
        "split_file_sha256": _sha256(split_path),
        "historical_exposure": (
            "All 700 images were part of the prior M6 baseline evaluation. "
            "This prospective split isolates new-method tuning from new "
            "holdout scoring, but is not a globally unseen test set."
        ),
        "holdout_access_policy": (
            "The benchmark dev command reads only development image/mask "
            "files. Holdout metrics are written only by the holdout command "
            "after candidate_selection.json exists."
        ),
    }
    _write_json(output_dir / "split_metadata.json", metadata)
    print(json.dumps(metadata, indent=2))
    return metadata


def _make_experiments() -> list[ExperimentSpec]:
    experiments: list[ExperimentSpec] = []
    baseline_config = SegmentationConfig()
    experiments.append(
        ExperimentSpec(
            BASELINE_ID,
            "Existing HSV threshold baseline",
            asdict(baseline_config),
            lambda image, config=baseline_config: segment_water(image, config).mask,
        )
    )

    hsv_configs = {
        "hsv_h70_145_s0_v15": SegmentationConfig(
            hue_min=70,
            hue_max=145,
            saturation_min=0,
            value_min=15,
        ),
        "hsv_h60_150_s10_v15": SegmentationConfig(
            hue_min=60,
            hue_max=150,
            saturation_min=10,
            value_min=15,
        ),
        "hsv_h70_145_s20_v20": SegmentationConfig(
            hue_min=70,
            hue_max=145,
            saturation_min=20,
            value_min=20,
        ),
        "hsv_h60_150_s0_v25_no_blur": SegmentationConfig(
            hue_min=60,
            hue_max=150,
            saturation_min=0,
            value_min=25,
            blur_method="none",
            morphology_kernel_size=3,
            morphology_iterations=1,
            min_region_area=40,
        ),
    }
    for experiment_id, config in hsv_configs.items():
        experiments.append(
            ExperimentSpec(
                experiment_id,
                "Configurable HSV threshold + existing refinement",
                asdict(config),
                lambda image, config=config: segment_water(image, config).mask,
            )
        )

    rgb_configs = {
        "rgb_blue_broad": RGBThresholdConfig(
            red_max=210,
            green_max=235,
            blue_min=50,
        ),
        "rgb_blue_restrictive": RGBThresholdConfig(
            red_max=180,
            green_max=215,
            blue_min=85,
        ),
        "rgb_bright_channels": RGBThresholdConfig(
            red_min=55,
            red_max=220,
            green_min=60,
            green_max=225,
            blue_min=65,
            blue_max=255,
        ),
    }
    for experiment_id, config in rgb_configs.items():
        experiments.append(
            ExperimentSpec(
                experiment_id,
                "RGB per-channel range threshold",
                asdict(config),
                lambda image, config=config: segment_rgb_threshold(image, config),
            )
        )

    adaptive_configs = {
        "adaptive_saturation_inv_g31_c5": AdaptiveThresholdConfig(
            channel="saturation", method="gaussian", block_size=31, constant=5
        ),
        "adaptive_saturation_inv_g51_c8": AdaptiveThresholdConfig(
            channel="saturation", method="gaussian", block_size=51, constant=8
        ),
        "adaptive_saturation_inv_mean31_c5": AdaptiveThresholdConfig(
            channel="saturation", method="mean", block_size=31, constant=5
        ),
        "adaptive_gray_inv_g31_c5": AdaptiveThresholdConfig(
            channel="grayscale", method="gaussian", block_size=31, constant=5
        ),
    }
    for experiment_id, config in adaptive_configs.items():
        experiments.append(
            ExperimentSpec(
                experiment_id,
                "Adaptive threshold",
                asdict(config),
                lambda image, config=config: segment_adaptive_threshold(
                    image, config
                ),
            )
        )

    otsu_configs = {
        "otsu_saturation_inv": OtsuThresholdConfig(
            channel="saturation", invert=True
        ),
        "otsu_value_inv": OtsuThresholdConfig(channel="value", invert=True),
        "otsu_grayscale_inv": OtsuThresholdConfig(
            channel="grayscale", invert=True
        ),
    }
    for experiment_id, config in otsu_configs.items():
        experiments.append(
            ExperimentSpec(
                experiment_id,
                "Otsu global threshold",
                asdict(config),
                lambda image, config=config: segment_otsu_threshold(image, config),
            )
        )

    hsv_color = HSVColorConfig(
        hue_min=70,
        hue_max=145,
        saturation_min=0,
        saturation_max=255,
        value_min=15,
        value_max=255,
    )
    morphology_configs = {
        "hsv_color_open_close_k3": MorphologyConfig(
            operation="open_close", kernel_size=3, iterations=1
        ),
        "hsv_color_open_close_k5": MorphologyConfig(
            operation="open_close", kernel_size=5, iterations=1
        ),
        "hsv_color_close_k3": MorphologyConfig(
            operation="closing", kernel_size=3, iterations=1
        ),
    }
    for experiment_id, morph_config in morphology_configs.items():
        parameters = {
            "hsv": asdict(hsv_color),
            "morphology": asdict(morph_config),
        }

        def predict_hsv_morph(
            image: np.ndarray,
            hsv_settings: HSVColorConfig = hsv_color,
            morphology_settings: MorphologyConfig = morph_config,
        ) -> np.ndarray:
            raw_mask = segment_hsv_color_threshold(image, hsv_settings)
            return refine_mask_morphology(raw_mask, morphology_settings)

        experiments.append(
            ExperimentSpec(
                experiment_id,
                "Raw HSV color threshold + morphology",
                parameters,
                predict_hsv_morph,
            )
        )
    return experiments


def _load_split(split_path: Path) -> dict[str, list[str]]:
    with split_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Split file has no assignments.")
    split: dict[str, list[str]] = {"development": [], "holdout": []}
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        subset = row["subset"]
        if sample_id in seen or subset not in split:
            raise ValueError("Split file contains duplicate IDs or unknown subsets.")
        seen.add(sample_id)
        split[subset].append(sample_id)
    if not split["development"] or not split["holdout"]:
        raise ValueError("Development and holdout subsets must both be non-empty.")
    return split


def _metric_row(
    metrics: EvaluationMetrics,
    *,
    pair: Any,
    subset: str,
    image: np.ndarray,
    reference: np.ndarray,
    predicted: np.ndarray,
    seconds: float,
) -> dict[str, Any]:
    return {
        "sample_id": pair.sample_id,
        "subset": subset,
        "image_file": pair.image_path.name,
        "mask_file": pair.mask_path.name,
        "status": "ok",
        "width_pixels": int(image.shape[1]),
        "height_pixels": int(image.shape[0]),
        "total_pixels": metrics.total_pixels,
        "reference_water_pixels": int(np.count_nonzero(reference == 255)),
        "predicted_water_pixels": int(np.count_nonzero(predicted == 255)),
        "true_positive": metrics.true_positive,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "true_negative": metrics.true_negative,
        "iou": metrics.intersection_over_union,
        "dice": metrics.dice_coefficient,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "empty_prediction": not bool(np.any(predicted)),
        "segmentation_seconds": seconds,
        "error": "",
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    if not successful:
        raise ValueError("Experiment produced no valid predictions.")
    tp = sum(row["true_positive"] for row in successful)
    fp = sum(row["false_positive"] for row in successful)
    fn = sum(row["false_negative"] for row in successful)
    tn = sum(row["true_negative"] for row in successful)
    union = tp + fp + fn
    dice_denominator = 2 * tp + fp + fn
    times = [row["segmentation_seconds"] for row in successful]
    macro = {}
    for name in ("iou", "dice", "precision", "recall"):
        values = [row[name] for row in successful if row[name] is not None]
        macro[name] = {
            "mean": statistics.fmean(values) if values else None,
            "n_defined": len(values),
            "n_total": len(successful),
        }
    return {
        "n_images": len(successful),
        "n_errors": len(rows) - len(successful),
        "empty_predictions": sum(row["empty_prediction"] for row in successful),
        "macro": macro,
        "micro": {
            "iou": tp / union if union else 1.0,
            "dice": 2 * tp / dice_denominator if dice_denominator else 1.0,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
        "timing": {
            "segmentation_seconds_total": sum(times),
            "segmentation_seconds_mean": statistics.fmean(times),
            "segmentation_seconds_median": statistics.median(times),
            "segmentation_seconds_p95": float(np.percentile(times, 95)),
        },
        "lowest_iou_sample_ids": [
            row["sample_id"]
            for row in sorted(
                successful,
                key=lambda row: (row["iou"], row["sample_id"]),
            )[:10]
        ],
    }


def _experiment_record(
    spec: ExperimentSpec,
    rows: list[dict[str, Any]],
    subset: str,
    output_root: Path,
) -> dict[str, Any]:
    experiment_dir = output_root / spec.experiment_id
    experiment_dir.mkdir()
    _write_csv(experiment_dir / "per_image_metrics.csv", rows, PER_IMAGE_FIELDS)
    aggregate = _aggregate(rows)
    record = {
        "experiment_id": spec.experiment_id,
        "algorithm": spec.algorithm,
        "parameters": spec.parameters,
        "subset": subset,
        **aggregate,
    }
    _write_json(experiment_dir / "aggregate_metrics.json", record)
    _write_json(
        experiment_dir / "experiment_metadata.json",
        {
            "experiment_id": spec.experiment_id,
            "algorithm": spec.algorithm,
            "parameters": spec.parameters,
            "subset": subset,
            "selection_policy": (
                "Configurations are compared on development only; holdout "
                "is not included in this experiment directory."
            ),
        },
    )
    return record


def _comparison_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        rows.append(
            {
                "experiment_id": record["experiment_id"],
                "algorithm": record["algorithm"],
                "parameters_json": json.dumps(
                    record["parameters"], sort_keys=True, separators=(",", ":")
                ),
                "subset": record["subset"],
                "n_images": record["n_images"],
                "empty_predictions": record["empty_predictions"],
                "macro_iou": record["macro"]["iou"]["mean"],
                "macro_dice": record["macro"]["dice"]["mean"],
                "macro_precision": record["macro"]["precision"]["mean"],
                "macro_recall": record["macro"]["recall"]["mean"],
                "micro_iou": record["micro"]["iou"],
                "micro_dice": record["micro"]["dice"],
                "micro_precision": record["micro"]["precision"],
                "micro_recall": record["micro"]["recall"],
                "mean_segmentation_seconds": record["timing"][
                    "segmentation_seconds_mean"
                ],
                "median_segmentation_seconds": record["timing"][
                    "segmentation_seconds_median"
                ],
                "p95_segmentation_seconds": record["timing"][
                    "segmentation_seconds_p95"
                ],
                "total_segmentation_seconds": record["timing"][
                    "segmentation_seconds_total"
                ],
            }
        )
    return sorted(rows, key=lambda row: (-row["macro_iou"], row["experiment_id"]))


def _candidate_rank(record: dict[str, Any]) -> tuple[float, float, float, str]:
    def defined(metric: str) -> float:
        value = record["macro"][metric]["mean"]
        return float(value) if value is not None else float("-inf")

    return (
        -defined("iou"),
        -defined("dice"),
        -defined("precision"),
        record["experiment_id"],
    )


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _development_report(
    records: list[dict[str, Any]],
    selected: dict[str, Any],
    split_metadata: dict[str, Any],
) -> str:
    ordered = sorted(
        records,
        key=lambda record: (
            -record["macro"]["iou"]["mean"],
            record["experiment_id"],
        ),
    )
    lines = [
        "# Milestone 6.1 — Development benchmark",
        "",
        f"- Split: {split_metadata['development_count']} development / "
        f"{split_metadata['holdout_count']} holdout images; seed {SPLIT_SEED}.",
        (
            "- The split is a new experimental image-level partition, not an "
            "official dataset split."
        ),
        (
            "- All 700 images were previously used in the historical fixed-default "
            "M6 evaluation; therefore this is not a globally unseen test set."
        ),
        (
            "- Only development images were opened for the experiments in this "
            "report. Holdout metrics were unavailable at candidate selection."
        ),
        (
            "- Selection: highest development macro IoU; ties by macro Dice, "
            "macro Precision, then lexicographically ascending ID."
        ),
        (
            "- No ML/DL, pretrained models, image resizing, or ground-truth "
            "mask modification was used."
        ),
        "",
        f"## Selected candidate: `{selected['experiment_id']}`",
        "",
        f"Algorithm: {selected['algorithm']}",
        "",
        "Parameters:",
        "",
        "```json",
        json.dumps(selected["parameters"], indent=2, sort_keys=True),
        "```",
        "",
        "## Development comparison",
        "",
        (
            "| Experiment | Macro IoU | Dice | Precision | Recall | "
            "Empty predictions | Mean ms/image |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in ordered:
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {:.3f} |".format(
                record["experiment_id"],
                _format_metric(record["macro"]["iou"]["mean"]),
                _format_metric(record["macro"]["dice"]["mean"]),
                _format_metric(record["macro"]["precision"]["mean"]),
                _format_metric(record["macro"]["recall"]["mean"]),
                record["empty_predictions"],
                record["timing"]["segmentation_seconds_mean"] * 1000,
            )
        )
    lines.extend(
        [
            "",
            (
                "Per-experiment parameter, timing, and per-image records are "
                "stored in sibling experiment directories and `comparison.csv`."
            ),
            "",
            "## Interpretation limits",
            "",
            (
                "Macro IoU is the preselected development ranking measure and "
                "penalizes both false positives and false negatives. Precision, "
                "Recall, Dice, micro metrics, empty-prediction counts, and "
                "processing time are also shown; higher Recall alone is not "
                "treated as improvement. Development results are exploratory "
                "and do not establish floodwater classification or field "
                "performance."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_pair(pair: Any) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode image {pair.image_path}.")
    validate_image(image, "BGR")
    reference = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
    if reference is None:
        raise OSError(f"Cannot decode reference mask {pair.mask_path}.")
    validate_binary_mask(reference)
    if image.shape[:2] != reference.shape:
        raise ValueError(
            f"Image/mask geometry mismatch for {pair.sample_id}: "
            f"{image.shape[:2]} vs {reference.shape}."
        )
    return image, reference


def _evaluate_subset(
    pairs: list[Any],
    specs: list[ExperimentSpec],
    subset: str,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_experiment = {spec.experiment_id: [] for spec in specs}
    for pair in pairs:
        image, reference = _load_pair(pair)
        for spec in specs:
            image_input = image.copy()
            started = time.perf_counter()
            prediction = spec.predict(image_input)
            elapsed = time.perf_counter() - started
            validate_binary_mask(prediction)
            if prediction.shape != reference.shape:
                raise ValueError(
                    f"{spec.experiment_id} changed geometry for {pair.sample_id}."
                )
            metrics = evaluate_masks(prediction, reference)
            rows_by_experiment[spec.experiment_id].append(
                _metric_row(
                    metrics,
                    pair=pair,
                    subset=subset,
                    image=image,
                    reference=reference,
                    predicted=prediction,
                    seconds=elapsed,
                )
            )
    return rows_by_experiment


def _environment_metadata() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "opencv_threads": cv2.getNumThreads(),
    }


def run_development(
    data_root: Path,
    split_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Benchmark all predeclared configurations on development IDs only."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Development output is not empty: {output_dir}")
    assignments = _load_split(split_path)
    pairs_by_id = {pair.sample_id: pair for pair in discover_pairs(data_root)}
    if set(assignments["development"] + assignments["holdout"]) != set(pairs_by_id):
        raise ValueError("Split IDs do not match the discovered dataset IDs.")
    development_pairs = [
        pairs_by_id[sample_id] for sample_id in assignments["development"]
    ]
    specs = _make_experiments()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_experiment = _evaluate_subset(
        development_pairs,
        specs,
        "development",
    )
    records = []
    for spec in specs:
        record = _experiment_record(
            spec,
            rows_by_experiment[spec.experiment_id],
            "development",
            output_dir,
        )
        records.append(record)

    baseline = next(
        record for record in records if record["experiment_id"] == BASELINE_ID
    )
    candidates = [
        record for record in records if record["experiment_id"] != BASELINE_ID
    ]
    selected = min(candidates, key=_candidate_rank)
    split_metadata = json.loads(
        split_path.with_name("split_metadata.json").read_text(encoding="utf-8")
    )
    source_fingerprints = _code_fingerprints()
    selection = {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_subset": "development",
        "selection_criterion": (
            "Highest macro IoU; ties by macro Dice, macro Precision, "
            "then lexicographically ascending experiment ID."
        ),
        "selected_experiment_id": selected["experiment_id"],
        "selected_algorithm": selected["algorithm"],
        "selected_parameters": selected["parameters"],
        "selected_development_metrics": {
            key: selected[key]
            for key in ("macro", "micro", "empty_predictions", "timing")
        },
        "development_baseline_metrics": {
            key: baseline[key]
            for key in ("macro", "micro", "empty_predictions", "timing")
        },
        "holdout_results_read": False,
        "split_file_sha256": _sha256(split_path),
        "source_sha256": source_fingerprints,
    }
    _write_json(output_dir / "candidate_selection.json", selection)
    comparison_rows = _comparison_rows(records)
    _write_csv(
        output_dir / "comparison.csv",
        comparison_rows,
        tuple(comparison_rows[0]),
    )
    _write_json(
        output_dir / "development_metadata.json",
        {
            "dataset_release": DATASET_RELEASE,
            "dataset_repository_commit": DATASET_COMMIT,
            "subset": "development",
            "split_file": str(split_path.resolve()),
            "split_file_sha256": _sha256(split_path),
            "development_count": len(development_pairs),
            "holdout_count": len(assignments["holdout"]),
            "holdout_files_opened": False,
            "historical_baseline_exposure": split_metadata["historical_exposure"],
            "environment": _environment_metadata(),
            "experiment_count": len(specs),
            "archive_sha256": split_metadata.get("archive_sha256"),
            "source_sha256": source_fingerprints,
        },
    )
    report = _development_report(records, selected, split_metadata)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "development_images": len(development_pairs),
                "holdout_images": len(assignments["holdout"]),
                "experiment_count": len(specs),
                "baseline": baseline["macro"],
                "selected_candidate": selected["experiment_id"],
                "selected_macro": selected["macro"],
                "selection_file": str(
                    (output_dir / "candidate_selection.json").resolve()
                ),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return selection


def _find_spec(experiment_id: str) -> ExperimentSpec:
    for spec in _make_experiments():
        if spec.experiment_id == experiment_id:
            return spec
    raise ValueError(f"Unknown locked experiment ID: {experiment_id}")


def _holdout_report(
    records: list[dict[str, Any]],
    selection: dict[str, Any],
) -> str:
    baseline = next(
        record for record in records if record["experiment_id"] == BASELINE_ID
    )
    candidate = next(
        record
        for record in records
        if record["experiment_id"] == selection["selected_experiment_id"]
    )
    lines = [
        "# Milestone 6.1 — Final holdout comparison",
        "",
        (
            "This report evaluates only the development-locked candidate and "
            "the unchanged baseline on the exact same holdout image IDs. No "
            "other methods/configurations were run on the holdout by this tool."
        ),
        "",
        f"- Selected before holdout scoring: `{candidate['experiment_id']}`.",
        (
            "- Candidate parameters: `"
            f"{json.dumps(candidate['parameters'], sort_keys=True)}`."
        ),
        f"- Holdout images: {candidate['n_images']}.",
        (
            "- Limitation: all 700 images were included in the earlier M6 "
            "baseline evaluation; this holdout is prospective for new-method "
            "selection but not globally unseen."
        ),
        "",
        (
            "| Method | Macro IoU | Dice | Precision | Recall | "
            "Empty predictions | Mean ms/image |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in (baseline, candidate):
        lines.append(
            "| `{}` | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} | {:.3f} |".format(
                record["experiment_id"],
                record["macro"]["iou"]["mean"],
                record["macro"]["dice"]["mean"],
                record["macro"]["precision"]["mean"],
                record["macro"]["recall"]["mean"],
                record["empty_predictions"],
                record["timing"]["segmentation_seconds_mean"] * 1000,
            )
        )
    lines.extend(
        [
            "",
            (
                "Micro metrics, per-image results, confusion totals, and timing "
                "summaries are in `comparison.csv` and each method's result "
                "directory. These are water/background agreement results, not "
                "floodwater classification or field validation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_holdout(
    data_root: Path,
    split_path: Path,
    development_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Score the dev-locked candidate and baseline once on the holdout IDs."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Holdout output is not empty: {output_dir}")
    selection_path = development_dir / "candidate_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Development candidate selection must exist first.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("holdout_results_read") is not False:
        raise ValueError("Candidate selection record is not holdout-clean.")
    if selection.get("source_sha256") != _code_fingerprints():
        raise ValueError(
            "Source fingerprints changed after development selection; "
            "rerun development before opening holdout data."
        )
    assignments = _load_split(split_path)
    pairs_by_id = {pair.sample_id: pair for pair in discover_pairs(data_root)}
    if set(assignments["development"] + assignments["holdout"]) != set(pairs_by_id):
        raise ValueError("Split IDs do not match the discovered dataset IDs.")
    holdout_pairs = [pairs_by_id[sample_id] for sample_id in assignments["holdout"]]
    candidate = _find_spec(selection["selected_experiment_id"])
    specs = [_find_spec(BASELINE_ID), candidate]
    if specs[0].experiment_id == specs[1].experiment_id:
        raise ValueError("Selected candidate cannot replace the original baseline.")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_experiment = _evaluate_subset(
        holdout_pairs,
        specs,
        "holdout",
    )
    records = []
    for spec in specs:
        record = _experiment_record(
            spec,
            rows_by_experiment[spec.experiment_id],
            "holdout",
            output_dir,
        )
        records.append(record)
    comparison_rows = _comparison_rows(records)
    _write_csv(
        output_dir / "comparison.csv",
        comparison_rows,
        tuple(comparison_rows[0]),
    )
    baseline_record = next(
        record for record in records if record["experiment_id"] == BASELINE_ID
    )
    candidate_record = next(
        record
        for record in records
        if record["experiment_id"] == candidate.experiment_id
    )
    split_metadata = json.loads(
        split_path.with_name("split_metadata.json").read_text(encoding="utf-8")
    )
    result = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subset": "holdout",
        "holdout_count": len(holdout_pairs),
        "development_candidate_selection_file": str(selection_path.resolve()),
        "development_candidate_selection_sha256": _sha256(selection_path),
        "split_file_sha256": _sha256(split_path),
        "archive_sha256": split_metadata.get("archive_sha256"),
        "source_sha256": _code_fingerprints(),
        "methods_scored": [BASELINE_ID, candidate.experiment_id],
        "holdout_files_opened_only_after_development_lock": True,
        "historical_baseline_exposure": (
            "The full dataset was previously part of the historical M6 baseline run; "
            "this is not a globally unseen holdout."
        ),
        "environment": _environment_metadata(),
        "baseline": baseline_record,
        "selected_candidate": candidate_record,
    }
    _write_json(output_dir / "holdout_result.json", result)
    (output_dir / "report.md").write_text(
        _holdout_report(records, selection),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "holdout_images": len(holdout_pairs),
                "baseline_macro": baseline_record["macro"],
                "candidate_id": candidate.experiment_id,
                "candidate_macro": candidate_record["macro"],
                "output_dir": str(output_dir.resolve()),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser("create-split")
    split_parser.add_argument("--data-root", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--archive", type=Path)

    development_parser = commands.add_parser("dev")
    development_parser.add_argument("--data-root", type=Path, required=True)
    development_parser.add_argument("--split-file", type=Path, required=True)
    development_parser.add_argument("--output-dir", type=Path, required=True)

    holdout_parser = commands.add_parser("holdout")
    holdout_parser.add_argument("--data-root", type=Path, required=True)
    holdout_parser.add_argument("--split-file", type=Path, required=True)
    holdout_parser.add_argument("--development-dir", type=Path, required=True)
    holdout_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "create-split":
        create_split(
            args.data_root.resolve(),
            args.output_dir.resolve(),
            args.archive.resolve() if args.archive else None,
        )
    elif args.command == "dev":
        run_development(
            args.data_root.resolve(),
            args.split_file.resolve(),
            args.output_dir.resolve(),
        )
    else:
        run_holdout(
            args.data_root.resolve(),
            args.split_file.resolve(),
            args.development_dir.resolve(),
            args.output_dir.resolve(),
        )


if __name__ == "__main__":
    main()
