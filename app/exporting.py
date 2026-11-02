"""Explicit user-directed CSV and JSON exports for analysis and evaluation records."""

from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pandas as pd

from core.evaluation.metrics import EvaluationMetrics
from core.rescue.candidates import Candidate
from core.rescue.pipeline import RescueAnalysis, RescueConfig
from core.water.segmentation import SegmentationConfig

IMAGE_CANDIDATE_CSV_COLUMNS = (
    "candidate_id",
    "image_id",
    "frame_index",
    "x",
    "y",
    "width",
    "height",
    "center_x",
    "center_y",
    "area_pixels",
    "category",
    "category_label",
    "rule_score",
    "source_methods",
    "review_status",
    "review_status_label",
    "source_path",
    "processing_time_ms",
    "water_context_used",
    "water_context_configuration",
    "configuration",
)

VIDEO_CANDIDATE_CSV_COLUMNS = (
    "candidate_id",
    "image_id",
    "frame_index",
    "timestamp_ms",
    "category",
    "category_label",
    "track_id",
    "track_persistence_frames",
    "review_status",
    "review_status_label",
    "source_video",
    "water_context_used",
    "water_context_configuration",
    "configuration",
)


def software_provenance() -> dict[str, str]:
    """Return runtime versions that help reproduce a local analysis export."""
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "opencv_version": version("opencv-python"),
    }


def export_records_csv(
    path: Path | str,
    records: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write structured records to CSV without implicit overwrite.

    Pass ``columns`` when an empty result still needs a header row.
    """
    output_path = Path(path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )
    if not records and not columns:
        raise ValueError("No records or column names were provided for CSV export.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")

    frame = pd.DataFrame.from_records(records, columns=columns)
    try:
        if overwrite:
            frame.to_csv(output_path, index=False, encoding="utf-8")
        else:
            with output_path.open("x", encoding="utf-8", newline="") as output_file:
                frame.to_csv(output_file, index=False)
    except OSError as exc:
        raise OSError(f"Could not export CSV to '{output_path}': {exc}") from exc
    return output_path


def export_evaluation_csv(
    path: Path | str,
    metrics: EvaluationMetrics,
    *,
    overwrite: bool = False,
) -> Path:
    """Export one evaluation result, preserving undefined metrics as blank."""
    return export_records_csv(
        path,
        [metrics.to_record()],
        overwrite=overwrite,
    )


def export_json(
    path: Path | str,
    payload: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a JSON export to an explicit path without implicit overwrite."""
    output_path = Path(path)
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"Output directory does not exist: {output_path.parent}"
        )
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}")
    try:
        if overwrite:
            with output_path.open("w", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2, allow_nan=False)
                output_file.write("\n")
        else:
            with output_path.open("x", encoding="utf-8") as output_file:
                json.dump(payload, output_file, indent=2, allow_nan=False)
                output_file.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not export JSON to '{output_path}': {exc}") from exc
    return output_path


def build_image_candidate_metadata(
    source_path: Path | str,
    analysis: RescueAnalysis,
    *,
    water_context_config: SegmentationConfig | None = None,
) -> dict[str, object]:
    """Build reproducible provenance metadata for a still-image candidate run."""
    return {
        "source_path": str(source_path),
        **software_provenance(),
        "image_id": analysis.image_id,
        "image_width": analysis.image_width,
        "image_height": analysis.image_height,
        "channel_order_at_pipeline_boundary": (
            "RGB (explicitly converted from OpenCV BGR)"
        ),
        "processing_time_ms": analysis.processing_time_ms,
        "water_context_used": analysis.water_context_used,
        "water_context_configuration": (
            asdict(water_context_config)
            if water_context_config is not None
            else None
        ),
        "configuration": asdict(analysis.config),
        "interpretation": (
            "Experimental traditional-DIP proposals requiring human review. "
            "Categories and rule scores are not probabilities; review status "
            "does not confirm a person or rescue victim."
        ),
    }


def export_image_candidates_csv(
    path: Path | str,
    *,
    source_path: Path | str,
    analysis: RescueAnalysis,
    candidates: Sequence[Candidate],
    water_context_config: SegmentationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Export still-image candidate proposals and provenance to CSV."""
    metadata = build_image_candidate_metadata(
        source_path,
        analysis,
        water_context_config=water_context_config,
    )
    configuration = json.dumps(metadata["configuration"], sort_keys=True)
    water_configuration = json.dumps(
        metadata["water_context_configuration"], sort_keys=True
    )
    records = [
        {
            **candidate.to_record(),
            "source_path": metadata["source_path"],
            "processing_time_ms": metadata["processing_time_ms"],
            "water_context_used": metadata["water_context_used"],
            "water_context_configuration": water_configuration,
            "configuration": configuration,
        }
        for candidate in candidates
    ]
    columns = (
        tuple(records[0]) if records else IMAGE_CANDIDATE_CSV_COLUMNS
    )
    return export_records_csv(
        path, records, columns=columns, overwrite=overwrite
    )


def export_image_candidates_json(
    path: Path | str,
    *,
    source_path: Path | str,
    analysis: RescueAnalysis,
    candidates: Sequence[Candidate],
    water_context_config: SegmentationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Export still-image candidate proposals and provenance to JSON."""
    payload = {
        "metadata": build_image_candidate_metadata(
            source_path,
            analysis,
            water_context_config=water_context_config,
        ),
        "candidates": [candidate.to_record() for candidate in candidates],
    }
    return export_json(path, payload, overwrite=overwrite)


def build_video_candidate_metadata(
    source_video: Path | str | None,
    config: RescueConfig | None,
    *,
    water_context_used: bool,
    water_context_config: SegmentationConfig | None = None,
) -> dict[str, object]:
    """Build reproducible provenance metadata for a video candidate run."""
    return {
        "source_video": str(source_video) if source_video else None,
        **software_provenance(),
        "water_context_used": water_context_used,
        "water_context_configuration": (
            asdict(water_context_config)
            if water_context_config is not None
            else None
        ),
        "configuration": asdict(config) if config is not None else None,
        "tracking": (
            "Sparse Lucas-Kanade optical flow estimates global affine camera "
            "motion; greedy centroid-distance/box-IoU association with short "
            "missed-frame persistence."
        ),
        "interpretation": (
            "Experimental traditional-DIP candidates requiring human review. "
            "Tracks are not confirmed objects or verified rescue victims."
        ),
    }


def export_video_candidates_csv(
    path: Path | str,
    *,
    source_video: Path | str,
    config: RescueConfig | None,
    candidate_records: Sequence[Mapping[str, object]],
    water_context_used: bool = False,
    water_context_config: SegmentationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Export frame-wise video candidate and track records to CSV."""
    metadata = build_video_candidate_metadata(
        source_video,
        config,
        water_context_used=water_context_used,
        water_context_config=water_context_config,
    )
    configuration = json.dumps(metadata["configuration"], sort_keys=True)
    water_configuration = json.dumps(
        metadata["water_context_configuration"], sort_keys=True
    )
    records = [
        {
            **row,
            "water_context_used": water_context_used,
            "water_context_configuration": water_configuration,
            "configuration": configuration,
        }
        for row in candidate_records
    ]
    columns = (
        tuple(records[0]) if records else VIDEO_CANDIDATE_CSV_COLUMNS
    )
    return export_records_csv(
        path, records, columns=columns, overwrite=overwrite
    )


def export_video_candidates_json(
    path: Path | str,
    *,
    source_video: Path | str,
    config: RescueConfig | None,
    frame_records: Sequence[Mapping[str, object]],
    candidate_records: Sequence[Mapping[str, object]],
    water_context_used: bool = False,
    water_context_config: SegmentationConfig | None = None,
    overwrite: bool = False,
) -> Path:
    """Export frame-wise video candidate and track records to JSON."""
    payload = {
        "metadata": build_video_candidate_metadata(
            source_video,
            config,
            water_context_used=water_context_used,
            water_context_config=water_context_config,
        ),
        "frames": list(frame_records),
        "candidates": list(candidate_records),
    }
    return export_json(path, payload, overwrite=overwrite)
