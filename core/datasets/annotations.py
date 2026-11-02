"""Read-only normalization of COCO-style bounding-box annotations.

The adapters preserve source labels and coordinates. They do not relabel,
resize, or modify annotation files. SeaDronesSee's official Object Detection
v2 exports use COCO-compatible image/category/annotation records. NITC's
public documentation states that bounding boxes and captions are provided,
but does not publish the downloadable schema; the NITC adapter therefore
accepts an authorized COCO JSON export and validates it rather than guessing
at the gated archive's private format.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in source-image pixel coordinates (COCO ``xywh``)."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise TypeError("Bounding-box values must be numeric, not booleans.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box values must be finite.")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("Bounding boxes need non-negative origins and positive size.")


@dataclass(frozen=True)
class CocoImage:
    """Image metadata needed to locate source imagery and group video frames."""

    image_id: str
    file_name: str
    width: int | None
    height: int | None
    source_video_id: str | None
    split: str | None


@dataclass(frozen=True)
class AnnotationRecord:
    """One normalized source annotation with its original semantic label."""

    image_id: str
    source_video_id: str | None
    bbox: BoundingBox
    original_class: str
    source_dataset: str
    annotation_status: str = "annotated"


@dataclass(frozen=True)
class CocoDataset:
    """Normalized image inventory and annotations from a COCO JSON file."""

    images: tuple[CocoImage, ...]
    annotations: tuple[AnnotationRecord, ...]


def _string_identifier(value: object, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"COCO {field} identifiers must be strings or integers.")
    identifier = str(value)
    if not identifier:
        raise ValueError(f"COCO {field} identifiers cannot be empty.")
    return identifier


def _video_identifier(image: Mapping[str, Any]) -> str | None:
    source = image.get("source")
    if isinstance(source, Mapping) and source.get("video") is not None:
        return str(source["video"])
    for key in ("source_video_id", "video_id"):
        if image.get(key) is not None:
            return str(image[key])
    return None


def _split_name(image: Mapping[str, Any], default_split: str | None) -> str | None:
    value = image.get("split", default_split)
    return None if value is None else str(value)


def load_coco_annotations(
    path: Path | str,
    *,
    source_dataset: str,
    default_split: str | None = None,
) -> CocoDataset:
    """Load COCO boxes into a common representation without changing labels.

    ``default_split`` is an explicit caller-supplied split for files whose
    annotations do not carry one (for example, one official partition file).
    Coordinates remain in source pixels and are neither clipped nor scaled.
    COCO crowd/ignore entries are preserved with status ``ignore`` so
    evaluators can exclude them from ordinary positive/negative matching.
    """
    annotation_path = Path(path)
    try:
        with annotation_path.open("r", encoding="utf-8") as source_file:
            payload = json.load(source_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read annotation JSON '{annotation_path}': {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("COCO annotation root must be a JSON object.")

    raw_images = payload.get("images")
    raw_categories = payload.get("categories")
    raw_annotations = payload.get("annotations")
    if not isinstance(raw_images, list) or not isinstance(raw_categories, list):
        raise ValueError("COCO JSON must contain 'images' and 'categories' lists.")
    if not isinstance(raw_annotations, list):
        raise ValueError("COCO JSON must contain an 'annotations' list.")

    categories: dict[str, str] = {}
    for category in raw_categories:
        if not isinstance(category, Mapping) or "id" not in category:
            raise ValueError("Each COCO category must have an id and name.")
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            raise ValueError("Each COCO category must have a non-empty name.")
        category_id = _string_identifier(category["id"], "category")
        if category_id in categories:
            raise ValueError(f"Duplicate COCO category id: {category_id}.")
        categories[category_id] = category["name"].strip()

    image_by_id: dict[str, CocoImage] = {}
    for image in raw_images:
        if not isinstance(image, Mapping) or "id" not in image:
            raise ValueError("Each COCO image must have an id and file_name.")
        image_id = _string_identifier(image["id"], "image")
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"COCO image {image_id} has no valid file_name.")
        if image_id in image_by_id:
            raise ValueError(f"Duplicate COCO image id: {image_id}.")
        dimensions: dict[str, int | None] = {}
        for field in ("width", "height"):
            value = image.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"COCO image {image_id} has invalid {field}.")
            dimensions[field] = value
        image_by_id[image_id] = CocoImage(
            image_id=image_id,
            file_name=file_name,
            width=dimensions["width"],
            height=dimensions["height"],
            source_video_id=_video_identifier(image),
            split=_split_name(image, default_split),
        )

    records: list[AnnotationRecord] = []
    for position, annotation in enumerate(raw_annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"COCO annotation at index {position} must be an object.")
        try:
            image_id = _string_identifier(annotation["image_id"], "image")
            category_id = _string_identifier(annotation["category_id"], "category")
            raw_bbox = annotation["bbox"]
        except KeyError as exc:
            raise ValueError(
                f"COCO annotation at index {position} is missing {exc.args[0]}."
            ) from exc
        if image_id not in image_by_id:
            raise ValueError(f"Annotation references unknown image id {image_id}.")
        if category_id not in categories:
            raise ValueError(f"Annotation references unknown category id {category_id}.")
        if (
            not isinstance(raw_bbox, list)
            or len(raw_bbox) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_bbox
            )
        ):
            raise ValueError(f"Annotation {position} bbox must be four numeric xywh values.")
        try:
            bbox = BoundingBox(*(float(value) for value in raw_bbox))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Annotation {position} has an invalid bbox: {exc}") from exc
        category_name = categories[category_id]
        category_key = " ".join(category_name.strip().lower().split())
        status = (
            "ignore"
            if (
                bool(annotation.get("iscrowd", 0))
                or bool(annotation.get("ignore", 0))
                or category_key in {"ignore", "ignored", "ignore region"}
            )
            else "annotated"
        )
        records.append(
            AnnotationRecord(
                image_id=image_id,
                source_video_id=image_by_id[image_id].source_video_id,
                bbox=bbox,
                original_class=category_name,
                source_dataset=source_dataset,
                annotation_status=status,
            )
        )

    return CocoDataset(tuple(image_by_id.values()), tuple(records))


def load_seadronesssee_annotations(
    path: Path | str,
    *,
    default_split: str | None = None,
) -> CocoDataset:
    """Read an official SeaDronesSee Object Detection v2 COCO partition."""
    return load_coco_annotations(
        path,
        source_dataset="SeaDronesSee Object Detection v2",
        default_split=default_split,
    )


def load_nitc_annotations(
    path: Path | str,
    *,
    default_split: str | None = None,
) -> CocoDataset:
    """Read an authorized NITC COCO-formatted annotation export.

    NITC's public repository confirms person boxes and image captions but
    does not document its downloadable file schema. This adapter intentionally
    accepts only a validated COCO export; inspect the authorized release and
    write a narrow converter if its actual schema differs.
    """
    return load_coco_annotations(
        path,
        source_dataset="NITC Person Rescue",
        default_split=default_split,
    )
