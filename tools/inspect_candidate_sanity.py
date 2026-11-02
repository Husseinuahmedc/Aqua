"""Create a deterministic geometry-only review sheet for candidate proposals."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.objects.candidates import extract_candidate_regions
from core.objects.visualization import draw_candidate_boxes
from core.water.segmentation import SegmentationConfig, segment_water
from tools.evaluate_floodwater_manual import DATASET_COMMIT, discover_pairs


def _select_evenly_spaced(total: int, count: int) -> list[int]:
    """Return deterministic, unique indices spanning the sorted sample list."""
    if not 1 <= count <= total:
        raise ValueError(f"Sample count must be in the range 1–{total}.")
    if count == 1:
        return [total // 2]
    return sorted({round(index * (total - 1) / (count - 1)) for index in range(count)})


def _thumbnail(image: np.ndarray, label: str) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(480 / width, 270 / height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    tile = np.full((304, 480, 3), (35, 42, 42), dtype=np.uint8)
    x_offset = (480 - size[0]) // 2
    y_offset = 28 + (270 - size[1]) // 2
    tile[y_offset : y_offset + size[1], x_offset : x_offset + size[0]] = resized
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return tile


def create_review_sheet(
    data_root: Path,
    output_dir: Path,
    *,
    sample_count: int = 12,
) -> dict[str, Any]:
    """Save geometry overlays and a review montage, without semantic labels."""
    pairs = discover_pairs(data_root)
    indices = _select_evenly_spaced(len(pairs), sample_count)
    selected = [pairs[index] for index in indices]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SegmentationConfig()
    records: list[dict[str, Any]] = []
    tiles: list[np.ndarray] = []
    for pair in selected:
        image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode image for {pair.sample_id}.")
        mask = segment_water(image, config).mask
        candidates = extract_candidate_regions(
            mask,
            min_area_pixels=config.min_region_area,
        )
        overlay = draw_candidate_boxes(image, candidates)
        label = f"{pair.sample_id} | {len(candidates)} geometry proposal(s)"
        cv2.putText(
            overlay,
            "Geometry proposals only",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        overlay_path = output_dir / f"{pair.sample_id}_candidates.png"
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(f"Cannot write candidate overlay: {overlay_path}")
        records.append(
            {
                "sample_id": pair.sample_id,
                "candidate_count": len(candidates),
                "candidate_area_pixels": ";".join(
                    str(candidate.area_pixels) for candidate in candidates
                ),
                "candidate_boxes_xywh": ";".join(
                    f"{candidate.x}:{candidate.y}:"
                    f"{candidate.width}:{candidate.height}"
                    for candidate in candidates
                ),
                "min_area_pixels": config.min_region_area,
                "overlay_file": overlay_path.name,
                "human_review": "pending",
            }
        )
        tiles.append(_thumbnail(overlay, label))

    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    montage = np.full((rows * 304, columns * 480, 3), (22, 29, 29), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        montage[row * 304 : (row + 1) * 304, column * 480 : (column + 1) * 480] = tile
    montage_path = output_dir / "candidate_review_montage.png"
    if not cv2.imwrite(str(montage_path), montage):
        raise OSError(f"Cannot write candidate review montage: {montage_path}")

    with (output_dir / "candidate_sanity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "dataset_release": "floodwater-manual-v1.0.0",
        "dataset_repository_commit": DATASET_COMMIT,
        "candidate_input": (
            "Predicted mask from unchanged default segmentation; manual water "
            "masks are not used as object/person labels."
        ),
        "selection": "Evenly spaced sample IDs across the sorted 700-pair manual set.",
        "sample_ids": [pair.sample_id for pair in selected],
        "water_segmentation_parameters": asdict(config),
        "candidate_parameters": {
            "min_area_pixels": config.min_region_area,
            "max_area_pixels": None,
            "min_extent": 0.0,
        },
        "interpretation": (
            "Geometry proposals for optional human inspection only; no "
            "person/object correctness labels or accuracy claims."
        ),
        "human_review_status": (
            "pending; this sheet is prepared for review and is not itself "
            "a human annotation."
        ),
    }
    (output_dir / "candidate_sanity_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"samples": records, "montage": str(montage_path)}, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    args = parser.parse_args()
    create_review_sheet(
        args.data_root.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
