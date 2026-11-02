"""Evaluate image-based candidate proposals against local COCO annotations.

This utility performs no data acquisition. Run it only on a locally authorized
dataset partition and choose ``--complete-negative-labels`` only when the
dataset annotation policy supports treating unannotated regions as negatives.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from importlib.metadata import version

import cv2

from core.datasets.annotations import (
    CocoDataset,
    CocoImage,
    load_nitc_annotations,
    load_seadronesssee_annotations,
)
from core.rescue.evaluation import (
    evaluate_candidate_proposals,
    validate_video_disjoint_partitions,
)
from core.rescue.candidates import Candidate
from core.rescue.pipeline import RescueConfig, analyze_rescue_image
from core.preprocessing.image_io import load_image


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure traditional-DIP candidate recall on authorized COCO annotations."
    )
    parser.add_argument("--dataset", choices=("seadronesssee", "nitc"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        required=True,
        help="Official split or authorized video-grouped split name.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset-version",
        default="unspecified",
        help="Official release/commit identifier to preserve in the report.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("color", "intensity", "texture", "edge", "adaptive"),
        default=list(RescueConfig().methods),
    )
    parser.add_argument(
        "--complete-negative-labels",
        action="store_true",
        help=(
            "Explicitly assert that the selected partition has exhaustive "
            "relevant-object labels."
        ),
    )
    return parser


def _load_dataset(args: argparse.Namespace) -> CocoDataset:
    if args.dataset == "seadronesssee":
        return load_seadronesssee_annotations(
            args.annotations, default_split=args.split
        )
    return load_nitc_annotations(args.annotations, default_split=args.split)


def run_evaluation(args: argparse.Namespace) -> dict[str, object]:
    """Run one explicitly selected, local dataset partition."""
    if not args.images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images_dir}")
    dataset = _load_dataset(args)
    split_aliases = {"val": "validation", "valid": "validation"}
    requested_split = split_aliases.get(args.split.lower(), args.split.lower())
    all_partitions: dict[str, list[CocoImage]] = {}
    for image in dataset.images:
        if image.split is not None:
            split = split_aliases.get(image.split.lower(), image.split.lower())
            all_partitions.setdefault(split, []).append(image)
    if len(all_partitions) > 1:
        validate_video_disjoint_partitions(all_partitions)
    images = tuple(
        image
        for image in dataset.images
        if image.split is not None
        and split_aliases.get(image.split.lower(), image.split.lower()) == requested_split
    )
    if not images:
        raise ValueError(
            f"No annotation images were assigned to split {args.split!r}; "
            "check the partition file and split metadata."
        )
    config = RescueConfig(methods=tuple(args.methods))
    candidates_by_image: dict[str, tuple[Candidate, ...]] = {}
    processing_times: dict[str, float] = {}
    image_fingerprints: dict[str, str] = {}
    for image in images:
        image_path = args.images_dir / image.file_name
        try:
            with image_path.open("rb") as image_file:
                image_fingerprints[image.image_id] = hashlib.file_digest(
                    image_file, "sha256"
                ).hexdigest()
        except OSError as exc:
            raise OSError(f"Could not fingerprint source image '{image_path}': {exc}") from exc
        bgr = load_image(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        result = analyze_rescue_image(rgb, image_id=image.image_id, config=config)
        candidates_by_image[image.image_id] = result.candidates
        processing_times[image.image_id] = result.processing_time_ms

    image_ids = {image.image_id for image in images}
    annotations = tuple(
        annotation
        for annotation in dataset.annotations
        if annotation.image_id in image_ids
    )
    evaluation = evaluate_candidate_proposals(
        candidates_by_image,
        annotations,
        images,
        processing_time_ms=processing_times,
        complete_negative_labels=args.complete_negative_labels,
    )
    inventory_payload = [
        {
            "image_id": image.image_id,
            "file_name": image.file_name,
            "width": image.width,
            "height": image.height,
            "source_video_id": image.source_video_id,
            "split": image.split,
        }
        for image in images
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(inventory_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "dataset": args.dataset,
        "dataset_version": args.dataset_version,
        "split": args.split,
        "annotations_path": str(args.annotations),
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "images_root": str(args.images_dir),
        "image_inventory_sha256": inventory_sha256,
        "image_sha256_by_id": image_fingerprints,
        "annotation_classes": sorted({record.original_class for record in annotations}),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "algorithm": (
            "traditional DIP region proposals; rule-based measurements; "
            "no trained model"
        ),
        "water_context_used": False,
        "timing_scope": (
            "Rescue pipeline call only; excludes image fingerprint read, "
            "image decode, and BGR-to-RGB conversion."
        ),
        "configuration": asdict(config),
        "software": {
            "opencv": cv2.__version__,
            "numpy": version("numpy"),
        },
        "evaluation": evaluation.to_record(),
    }


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        report = run_evaluation(args)
        if not args.output.parent.exists():
            raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")
        with args.output.open("x", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Candidate evaluation saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
