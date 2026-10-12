from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import queue
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

from ..concurrency import (
    ordered_bounded_map,
    recommended_local_workers,
    recommended_scan_workers,
)
from ..protocol_tracks import build_canonical_assistant_response

SOURCE_WIDTH = 1920
SOURCE_HEIGHT = 1080
SOURCE_AREA = SOURCE_WIDTH * SOURCE_HEIGHT
IMAGE_MAX_PIXELS = SOURCE_AREA
DEFAULT_IMAGE_SIZE = (SOURCE_WIDTH, SOURCE_HEIGHT)
MIN_BBOX_WIDTH = 24.0
MIN_BBOX_HEIGHT = 18.0
MIN_BBOX_AREA = 600.0
MAX_BBOX_AREA_RATIO = 0.20


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class GroundCuaTarget:
    instruction: str
    instruction_type: str
    category: str
    bbox: BBox
    software: str
    source_image_id: str
    source_target_index: int


@dataclass(frozen=True)
class GroundCuaImageGroup:
    encoded_bytes: bytes
    encoded_sha256: str
    pixel_sha256: str
    width: int
    height: int
    image_format: str
    image_mode: str
    software: tuple[str, ...]
    source_image_ids: tuple[str, ...]
    targets: tuple[GroundCuaTarget, ...]
    perceptual_hash: str = ""
    confirmed_near_duplicate_group_id: str | None = None
    all_encoded_sha256s: tuple[str, ...] = ()


@dataclass(frozen=True)
class GroundCuaScanResult:
    image_groups: tuple[GroundCuaImageGroup, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class SplitQuota:
    direct: int
    functional: int

    def __post_init__(self) -> None:
        if self.direct < 0 or self.functional < 0:
            raise ValueError("split quotas must be non-negative")

    @property
    def total(self) -> int:
        return self.direct + self.functional


@dataclass(frozen=True)
class SelectedTarget:
    selection_id: str
    split: str
    image_group: GroundCuaImageGroup
    target: GroundCuaTarget
    target_identity: str | None = None
    source_component_id: str | None = None
    augmentation_index: int = 0
    candidate_target_count: int | None = None
    selected_candidate_index: int | None = None


@dataclass(frozen=True)
class GroundCuaSelectionResult:
    selected_targets: tuple[SelectedTarget, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class QuantizedPoint:
    model_xy: tuple[int, int]
    execution_pixel_xy: tuple[float, float]
    quadrant: str | None = None


@dataclass(frozen=True)
class WaypointPath:
    requested_move_count: int
    actual_move_count: int
    points: tuple[QuantizedPoint, ...]
    downgrade_reason: str | None = None


@dataclass(frozen=True)
class TrajectoryBuildResult:
    record: dict[str, Any]
    selected_target: dict[str, Any]
    audit: dict[str, Any]


@dataclass
class _ImageGroupBuilder:
    encoded_bytes: bytes
    encoded_sha256: str
    pixel_sha256: str
    width: int
    height: int
    image_format: str
    image_mode: str
    perceptual_hash: str
    comparison_thumbnail: bytes
    encoded_sha256s: set[str]
    software: list[str]
    source_image_ids: list[str]
    targets: list[GroundCuaTarget]
    confirmed_near_duplicate_group_id: str | None = None


def _perceptual_hash_and_thumbnail(image: Image.Image) -> tuple[str, bytes]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("GroundCUA perceptual hashing requires numpy") from exc

    grayscale = image.convert("L")
    resized = grayscale.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.float64)
    indices = np.arange(32, dtype=np.float64)
    frequencies = np.arange(32, dtype=np.float64)[:, None]
    cosine = np.cos((math.pi / 32.0) * (indices + 0.5) * frequencies)
    cosine[0, :] *= math.sqrt(1.0 / 32.0)
    cosine[1:, :] *= math.sqrt(2.0 / 32.0)
    dct = cosine @ pixels @ cosine.T
    low = dct[:8, :8].copy()
    low[np.abs(low) < 1e-9] = 0.0
    flattened = low.flatten()
    median = float(np.median(flattened[1:]))
    bits = flattened > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    thumbnail = image.convert("RGB").resize((64, 36), Image.Resampling.BILINEAR).tobytes()
    return f"{value:016x}", thumbnail


def _decoded_image_identity(
    encoded: bytes,
) -> tuple[int, int, str, str, str | None, str | None, bytes | None]:
    with Image.open(io.BytesIO(encoded)) as image:
        width, height = image.size
        image_format = str(image.format or "unknown").lower()
        image_mode = image.mode
        image.load()
        rgba = image.convert("RGBA")
        pixel_hasher = hashlib.sha256()
        pixel_hasher.update(width.to_bytes(4, "big"))
        pixel_hasher.update(height.to_bytes(4, "big"))
        pixel_hasher.update(rgba.tobytes())
        perceptual_hash, comparison_thumbnail = _perceptual_hash_and_thumbnail(image)
    return (
        width,
        height,
        image_format,
        image_mode,
        pixel_hasher.hexdigest(),
        perceptual_hash,
        comparison_thumbnail,
    )


def _thumbnail_max_channel_difference(first: bytes, second: bytes) -> float:
    if len(first) != len(second) or not first:
        return math.inf
    import numpy as np

    first_array = np.frombuffer(first, dtype=np.uint8).astype(np.int16)
    second_array = np.frombuffer(second, dtype=np.uint8).astype(np.int16)
    return float(np.abs(first_array - second_array).max())


def _assign_confirmed_near_duplicate_groups(
    builders: dict[str, _ImageGroupBuilder],
) -> tuple[int, int, int]:
    by_perceptual_hash: dict[str, list[str]] = {}
    for pixel_sha, builder in builders.items():
        by_perceptual_hash.setdefault(builder.perceptual_hash, []).append(pixel_sha)
    collision_groups = [
        sorted(pixel_shas)
        for pixel_shas in by_perceptual_hash.values()
        if len(pixel_shas) > 1
    ]
    confirmed_components: list[list[str]] = []
    for pixel_shas in collision_groups:
        clusters: list[list[str]] = []
        for pixel_sha in pixel_shas:
            thumbnail = builders[pixel_sha].comparison_thumbnail
            matching_cluster = next(
                (
                    cluster
                    for cluster in clusters
                    if all(
                        _thumbnail_max_channel_difference(
                            thumbnail,
                            builders[other_sha].comparison_thumbnail,
                        )
                        <= 1.0
                        for other_sha in cluster
                    )
                ),
                None,
            )
            if matching_cluster is None:
                clusters.append([pixel_sha])
            else:
                matching_cluster.append(pixel_sha)
        confirmed_components.extend(
            sorted(cluster) for cluster in clusters if len(cluster) > 1
        )
    for items in confirmed_components:
        digest = hashlib.sha256("|".join(items).encode("utf-8")).hexdigest()[:16]
        group_id = f"near-{digest}"
        for pixel_sha in items:
            builders[pixel_sha].confirmed_near_duplicate_group_id = group_id
    return (
        len(collision_groups),
        len(confirmed_components),
        sum(len(items) for items in confirmed_components),
    )


def _aligned_row_arrays(row: dict[str, Any], *, source: str) -> tuple[list[Any], ...]:
    arrays = tuple(
        list(row.get(name) or ())
        for name in ("coords", "instructions", "bboxes", "inst_type")
    )
    lengths = {len(items) for items in arrays}
    expected = int(row.get("num_elements") or 0)
    if len(lengths) != 1 or (lengths and next(iter(lengths)) != expected):
        raise ValueError(
            f"GroundCUA aligned arrays mismatch for {source}: "
            f"num_elements={expected}, lengths={[len(items) for items in arrays]}"
        )
    return arrays


def _instruction_category(instruction_type: str) -> str | None:
    if instruction_type == "functional":
        return "functional"
    if instruction_type.startswith("direct_"):
        return "direct"
    return None


_EXPLICIT_CLICK_ACTION = re.compile(
    r"\b(?:click|double[- ]click|right[- ]click|tap|press)\b",
    re.IGNORECASE,
)
_EXPLICIT_SELECT_ACTION = re.compile(
    r"(?:^|\b(?:and|then)\s+)(?:then\s+)?select\b",
    re.IGNORECASE,
)
_CURSOR_ONLY_ACTION = re.compile(
    r"\b(?:aim|bring|hover|move|pass|pinpoint|place|point|position|use|zero\s+in)\b"
    r".{0,160}\b(?:(?:your|the)\s+)?(?:mouse(?:\s+(?:cursor|pointer))?|cursor|pointer)\b",
    re.IGNORECASE,
)
_HOVER_ACTION = re.compile(r"\bhover\b", re.IGNORECASE)
_CURSOR_STATE = re.compile(
    r"\b(?:cursor|pointer|mouse)\b.{0,40}\b(?:is\s+)?(?:located|placed|positioned)\b",
    re.IGNORECASE,
)


def _is_non_click_cursor_only_instruction(instruction: str) -> bool:
    if _EXPLICIT_CLICK_ACTION.search(instruction) or _EXPLICIT_SELECT_ACTION.search(
        instruction
    ):
        return False
    return bool(
        _HOVER_ACTION.search(instruction)
        or _CURSOR_ONLY_ACTION.search(instruction)
        or _CURSOR_STATE.search(instruction)
    )


def _resolve_image_size(
    image_size: tuple[int, int] | list[int] | None,
) -> tuple[int, int]:
    if image_size is None:
        return DEFAULT_IMAGE_SIZE
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain width and height: {image_size!r}")
    width, height = int(image_size[0]), int(image_size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"image_size must be positive: {image_size!r}")
    return width, height


def _bbox_or_reason(
    raw_bbox: Any,
    *,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[BBox | None, str | None]:
    image_width, image_height = _resolve_image_size(image_size)
    image_area = image_width * image_height
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None, "missing_bbox"
    try:
        values = tuple(float(value) for value in raw_bbox)
    except (TypeError, ValueError):
        return None, "bbox_non_finite"
    if not all(math.isfinite(value) for value in values):
        return None, "bbox_non_finite"
    bbox = BBox(*values)
    if bbox.x2 <= bbox.x1 or bbox.y2 <= bbox.y1:
        return None, "bbox_non_positive"
    if bbox.x1 < 0 or bbox.y1 < 0 or bbox.x2 > image_width or bbox.y2 > image_height:
        return None, "bbox_outside_canvas"
    if bbox.width < MIN_BBOX_WIDTH:
        return None, "bbox_width_below_minimum"
    if bbox.height < MIN_BBOX_HEIGHT:
        return None, "bbox_height_below_minimum"
    if bbox.area < MIN_BBOX_AREA:
        return None, "bbox_area_below_minimum"
    if bbox.area / image_area > MAX_BBOX_AREA_RATIO:
        return None, "bbox_area_ratio_above_maximum"
    return bbox, None


def _iter_parquet_rows(
    paths: Iterable[Path],
    *,
    file_workers: int = 1,
    prefetch_batches: int = 2,
) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("GroundCUA parquet scanning requires pyarrow") from exc

    columns = [
        "image_id",
        "image",
        "num_elements",
        "coords",
        "instructions",
        "bboxes",
        "inst_type",
        "software",
    ]
    path_list = list(paths)
    file_workers = int(file_workers)
    prefetch_batches = int(prefetch_batches)
    if file_workers < 1:
        raise ValueError("file_workers must be at least 1")
    if prefetch_batches < 1:
        raise ValueError("prefetch_batches must be at least 1")

    def batches_for_path(path: Path) -> Iterable[list[dict[str, Any]]]:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=128,
            columns=columns,
            use_threads=False,
        ):
            yield batch.to_pylist()

    if file_workers == 1 or len(path_list) <= 1:
        for path in path_list:
            row_index = 0
            for rows in batches_for_path(path):
                for row in rows:
                    yield path, row_index, row
                    row_index += 1
        return

    stop_event = threading.Event()
    batch_queues: list[queue.Queue[tuple[str, Any]]] = [
        queue.Queue(maxsize=prefetch_batches) for _path in path_list
    ]

    def bounded_put(
        output_queue: queue.Queue[tuple[str, Any]],
        item: tuple[str, Any],
    ) -> bool:
        while not stop_event.is_set():
            try:
                output_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def read_path(path: Path, output_queue: queue.Queue[tuple[str, Any]]) -> None:
        try:
            for rows in batches_for_path(path):
                if not bounded_put(output_queue, ("rows", rows)):
                    return
        except BaseException as exc:
            bounded_put(output_queue, ("error", exc))
        finally:
            bounded_put(output_queue, ("done", None))

    with ThreadPoolExecutor(max_workers=min(file_workers, len(path_list))) as executor:
        futures = [
            executor.submit(read_path, path, output_queue)
            for path, output_queue in zip(path_list, batch_queues, strict=True)
        ]
        try:
            for path, output_queue in zip(path_list, batch_queues, strict=True):
                row_index = 0
                while True:
                    kind, payload = output_queue.get()
                    if kind == "done":
                        break
                    if kind == "error":
                        raise payload
                    if kind != "rows":
                        raise RuntimeError(f"unexpected parquet prefetch message: {kind}")
                    for row in payload:
                        yield path, row_index, row
                        row_index += 1
        finally:
            stop_event.set()
            for future in futures:
                future.cancel()


def _decode_scan_row(
    item: tuple[Path, int, dict[str, Any]],
) -> tuple[
    Path,
    int,
    dict[str, Any],
    tuple[int, int, str, str, str | None, str | None, bytes | None],
]:
    path, row_index, row = item
    encoded = row.get("image")
    if not isinstance(encoded, bytes):
        raise ValueError(f"GroundCUA image is not encoded bytes for {path}:{row_index}")
    return path, row_index, row, _decoded_image_identity(encoded)


def scan_groundcua_dataset(
    dataset_root: Path,
    *,
    workers: int = 1,
    max_pending: int | None = None,
    parquet_file_workers: int = 1,
    parquet_prefetch_batches: int = 2,
) -> GroundCuaScanResult:
    parquet_paths = sorted(dataset_root.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No GroundCUA parquet files found under {dataset_root}")
    workers = int(workers)
    resolved_max_pending = workers * 2 if max_pending is None else int(max_pending)

    audit: dict[str, Any] = {
        "parquet_file_count": len(parquet_paths),
        "scan_workers": workers,
        "scan_max_pending": resolved_max_pending,
        "parquet_use_threads": False,
        "parquet_file_workers": int(parquet_file_workers),
        "parquet_prefetch_batches": int(parquet_prefetch_batches),
        "rows_total": 0,
        "native_fullhd_rows": 0,
        "non_native_size_rows": 0,
        "resolution_counts": {},
        "eligible_target_count": 0,
        "resize_count": 0,
        "crop_count": 0,
        "letterbox_count": 0,
        "padding_count": 0,
        "stretch_count": 0,
    }
    target_exclusions: Counter[str] = Counter()
    encoded_hash_counts: Counter[str] = Counter()
    builders: dict[str, _ImageGroupBuilder] = {}

    decoded_rows = ordered_bounded_map(
        _decode_scan_row,
        _iter_parquet_rows(
            parquet_paths,
            file_workers=parquet_file_workers,
            prefetch_batches=parquet_prefetch_batches,
        ),
        max_workers=workers,
        max_pending=resolved_max_pending,
    )
    for path, row_index, row, decoded_identity in decoded_rows:
        audit["rows_total"] += 1
        source = f"{path}:{row_index}"
        coords, instructions, bboxes, instruction_types = _aligned_row_arrays(
            row,
            source=source,
        )
        encoded = row["image"]
        assert isinstance(encoded, bytes)
        (
            width,
            height,
            image_format,
            image_mode,
            pixel_sha256,
            perceptual_hash,
            comparison_thumbnail,
        ) = decoded_identity
        assert pixel_sha256 is not None
        assert perceptual_hash is not None
        assert comparison_thumbnail is not None
        if (width, height) == (SOURCE_WIDTH, SOURCE_HEIGHT):
            audit["native_fullhd_rows"] += 1
        else:
            audit["non_native_size_rows"] += 1
        resolution_key = f"{width}x{height}"
        audit["resolution_counts"][resolution_key] = int(
            audit["resolution_counts"].get(resolution_key, 0)
        ) + 1
        encoded_sha256 = hashlib.sha256(encoded).hexdigest()
        encoded_hash_counts[encoded_sha256] += 1
        image_id = str(row.get("image_id") or "")
        software = str(row.get("software") or "unknown")
        builder = builders.get(pixel_sha256)
        if builder is None:
            builder = _ImageGroupBuilder(
                encoded_bytes=encoded,
                encoded_sha256=encoded_sha256,
                pixel_sha256=pixel_sha256,
                width=width,
                height=height,
                image_format=image_format,
                image_mode=image_mode,
                perceptual_hash=perceptual_hash,
                comparison_thumbnail=comparison_thumbnail,
                encoded_sha256s={encoded_sha256},
                software=[],
                source_image_ids=[],
                targets=[],
            )
            builders[pixel_sha256] = builder
        else:
            builder.encoded_sha256s.add(encoded_sha256)
            if encoded_sha256 < builder.encoded_sha256:
                builder.encoded_bytes = encoded
                builder.encoded_sha256 = encoded_sha256
                builder.image_format = image_format
                builder.image_mode = image_mode
        if software not in builder.software:
            builder.software.append(software)
        if image_id not in builder.source_image_ids:
            builder.source_image_ids.append(image_id)

        existing_target_indexes = {
            (
                target.instruction,
                target.instruction_type,
                tuple(target.bbox.as_list()),
            ): index
            for index, target in enumerate(builder.targets)
        }
        for target_index, (instruction, raw_bbox, instruction_type) in enumerate(
            zip(instructions, bboxes, instruction_types, strict=True)
        ):
            normalized_instruction = str(instruction or "").strip()
            normalized_type = str(instruction_type or "").strip()
            if not normalized_instruction:
                target_exclusions["empty_instruction"] += 1
                continue
            if _is_non_click_cursor_only_instruction(normalized_instruction):
                target_exclusions["non_click_cursor_only_instruction"] += 1
                continue
            category = _instruction_category(normalized_type)
            if category is None:
                target_exclusions["instruction_type_excluded"] += 1
                continue
            bbox, reason = _bbox_or_reason(raw_bbox, image_size=(width, height))
            if bbox is None:
                assert reason is not None
                target_exclusions[reason] += 1
                continue
            key = (normalized_instruction, normalized_type, tuple(bbox.as_list()))
            candidate = GroundCuaTarget(
                instruction=normalized_instruction,
                instruction_type=normalized_type,
                category=category,
                bbox=bbox,
                software=software,
                source_image_id=image_id,
                source_target_index=target_index,
            )
            existing_index = existing_target_indexes.get(key)
            if existing_index is not None:
                target_exclusions["duplicate_target_annotation"] += 1
                existing = builder.targets[existing_index]
                if (
                    candidate.software,
                    candidate.source_image_id,
                    candidate.source_target_index,
                ) < (
                    existing.software,
                    existing.source_image_id,
                    existing.source_target_index,
                ):
                    builder.targets[existing_index] = candidate
                continue
            existing_target_indexes[key] = len(builder.targets)
            builder.targets.append(candidate)
            audit["eligible_target_count"] += 1

    (
        perceptual_hash_collision_groups,
        confirmed_near_duplicate_groups,
        confirmed_near_duplicate_rows,
    ) = _assign_confirmed_near_duplicate_groups(builders)
    image_groups = tuple(
        GroundCuaImageGroup(
            encoded_bytes=builder.encoded_bytes,
            encoded_sha256=builder.encoded_sha256,
            pixel_sha256=builder.pixel_sha256,
            width=builder.width,
            height=builder.height,
            image_format=builder.image_format,
            image_mode=builder.image_mode,
            software=tuple(sorted(builder.software)),
            source_image_ids=tuple(sorted(builder.source_image_ids)),
            targets=tuple(builder.targets),
            perceptual_hash=builder.perceptual_hash,
            confirmed_near_duplicate_group_id=builder.confirmed_near_duplicate_group_id,
            all_encoded_sha256s=tuple(sorted(builder.encoded_sha256s)),
        )
        for _pixel_sha, builder in sorted(builders.items())
    )
    duplicate_counts = [count for count in encoded_hash_counts.values() if count > 1]
    audit["exact_encoded_duplicate_groups"] = len(duplicate_counts)
    audit["exact_encoded_duplicate_excess_rows"] = sum(count - 1 for count in duplicate_counts)
    different_encoding_groups = [
        builder for builder in builders.values() if len(builder.encoded_sha256s) > 1
    ]
    audit["same_pixel_different_encoding_groups"] = len(different_encoding_groups)
    audit["same_pixel_different_encoding_excess_rows"] = sum(
        len(builder.encoded_sha256s) - 1 for builder in different_encoding_groups
    )
    audit["perceptual_hash_collision_groups"] = perceptual_hash_collision_groups
    audit["confirmed_near_duplicate_groups"] = confirmed_near_duplicate_groups
    audit["confirmed_near_duplicate_rows"] = confirmed_near_duplicate_rows
    audit["native_fullhd_image_groups"] = sum(
        group.width == SOURCE_WIDTH and group.height == SOURCE_HEIGHT
        for group in image_groups
    )
    audit["image_group_count"] = len(image_groups)
    audit["target_exclusion_reasons"] = dict(sorted(target_exclusions.items()))
    return GroundCuaScanResult(image_groups=image_groups, audit=audit)


def profile_split_quotas(profile: str) -> dict[str, SplitQuota]:
    if profile == "smoke":
        return {
            "train": SplitQuota(direct=96, functional=64),
            "val": SplitQuota(direct=10, functional=6),
            "test": SplitQuota(direct=9, functional=7),
        }
    if profile == "pilot":
        return {
            "train": SplitQuota(direct=1200, functional=800),
            "val": SplitQuota(direct=120, functional=80),
            "test": SplitQuota(direct=120, functional=80),
        }
    raise ValueError(f"Unsupported GroundCUA subset profile: {profile}")


def _selection_group_id(group: GroundCuaImageGroup) -> str:
    if group.confirmed_near_duplicate_group_id:
        return f"near:{group.confirmed_near_duplicate_group_id}"
    return f"pixel:{group.pixel_sha256}"


def _target_identity(group: GroundCuaImageGroup, target: GroundCuaTarget) -> str:
    bbox = ",".join(f"{value:.6f}" for value in target.bbox.as_list())
    return "|".join(
        (
            group.pixel_sha256,
            target.source_image_id,
            str(target.source_target_index),
            target.instruction_type,
            target.instruction,
            bbox,
        )
    )


def _all_target_identity(
    group: GroundCuaImageGroup,
    target: GroundCuaTarget,
) -> str:
    """Return an inventory-order-independent identity for one exact annotation."""

    bbox = ",".join(f"{value:.6f}" for value in target.bbox.as_list())
    return "|".join(
        (
            group.pixel_sha256,
            target.instruction,
            target.instruction_type,
            target.category,
            bbox,
        )
    )


def _selected_target_identity(selected: SelectedTarget) -> str:
    return selected.target_identity or _target_identity(
        selected.image_group,
        selected.target,
    )


def _stable_selection_rank(
    *,
    seed: int,
    split: str,
    group: GroundCuaImageGroup,
    target: GroundCuaTarget,
) -> str:
    payload = f"{seed}|{split}|{_target_identity(group, target)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_id(
    *,
    split: str,
    group: GroundCuaImageGroup,
    target: GroundCuaTarget,
    seed: int,
) -> str:
    suffix = _stable_selection_rank(
        seed=seed,
        split=split,
        group=group,
        target=target,
    )[:12]
    return f"groundcua_{split}_{group.pixel_sha256[:12]}_{suffix}"


def select_groundcua_subset(
    scan_result: GroundCuaScanResult,
    *,
    split_quotas: Mapping[str, SplitQuota],
    seed: int,
    max_targets_per_image: int = 2,
    direct_icon_floor_fraction: float = 0.05,
) -> GroundCuaSelectionResult:
    if max_targets_per_image < 1:
        raise ValueError("max_targets_per_image must be >= 1")
    if not 0.0 <= direct_icon_floor_fraction <= 1.0:
        raise ValueError("direct_icon_floor_fraction must be between 0 and 1")
    quotas = dict(split_quotas)
    if not quotas:
        raise ValueError("at least one split quota is required")

    candidates = [
        (group, target, _target_identity(group, target))
        for group in scan_result.image_groups
        for target in group.targets
        if target.category in {"direct", "functional"}
    ]
    total_direct_quota = sum(quota.direct for quota in quotas.values())
    available_icon_count = sum(
        target.instruction_type == "direct_icon"
        for _group, target, _target_key in candidates
    )
    direct_icon_floor_required = min(
        available_icon_count,
        math.ceil(total_direct_quota * direct_icon_floor_fraction),
    )

    assigned_split_by_group: dict[str, str] = {}
    selected_count_by_pixel: Counter[str] = Counter()
    selected_keys: set[str] = set()
    selected: list[SelectedTarget] = []
    software_counts_by_split: dict[str, Counter[str]] = {
        split: Counter() for split in quotas
    }
    remaining: dict[str, dict[str, int]] = {
        split: {
            "direct": quota.direct,
            "functional": quota.functional,
        }
        for split, quota in quotas.items()
    }
    rank_cache: dict[tuple[str, str], str] = {}
    pool_cache: dict[
        tuple[str, str, bool],
        dict[str, list[tuple[str, GroundCuaImageGroup, GroundCuaTarget, str]]],
    ] = {}
    pool_positions: dict[tuple[str, str, bool, str], int] = {}

    def candidate_rank(
        split: str,
        group: GroundCuaImageGroup,
        target: GroundCuaTarget,
        target_key: str,
    ) -> str:
        cache_key = (split, target_key)
        rank = rank_cache.get(cache_key)
        if rank is None:
            rank = _stable_selection_rank(
                seed=seed,
                split=split,
                group=group,
                target=target,
            )
            rank_cache[cache_key] = rank
        return rank

    def candidate_pool(
        split: str,
        category: str,
        require_direct_icon: bool,
    ) -> dict[str, list[tuple[str, GroundCuaImageGroup, GroundCuaTarget, str]]]:
        pool_key = (split, category, require_direct_icon)
        cached_pool = pool_cache.get(pool_key)
        if cached_pool is not None:
            return cached_pool
        by_software: dict[
            str,
            list[tuple[str, GroundCuaImageGroup, GroundCuaTarget, str]],
        ] = {}
        for group, target, target_key in candidates:
            if target.category != category:
                continue
            if require_direct_icon and target.instruction_type != "direct_icon":
                continue
            by_software.setdefault(target.software, []).append(
                (
                    candidate_rank(split, group, target, target_key),
                    group,
                    target,
                    target_key,
                )
            )
        for items in by_software.values():
            items.sort(key=lambda item: item[0])
        pool_cache[pool_key] = by_software
        return by_software

    def peek_candidate(
        split: str,
        category: str,
        require_direct_icon: bool,
        software: str,
        items: list[tuple[str, GroundCuaImageGroup, GroundCuaTarget, str]],
    ) -> tuple[str, GroundCuaImageGroup, GroundCuaTarget, str] | None:
        position_key = (split, category, require_direct_icon, software)
        position = pool_positions.get(position_key, 0)
        while position < len(items):
            rank, group, target, target_key = items[position]
            group_id = _selection_group_id(group)
            assigned_split = assigned_split_by_group.get(group_id)
            invalid = (
                target_key in selected_keys
                or selected_count_by_pixel[group.pixel_sha256] >= max_targets_per_image
                or (assigned_split is not None and assigned_split != split)
            )
            if not invalid:
                pool_positions[position_key] = position
                return rank, group, target, target_key
            position += 1
        pool_positions[position_key] = position
        return None

    def choose_one(
        split: str,
        category: str,
        *,
        require_direct_icon: bool = False,
    ) -> bool:
        pool = candidate_pool(split, category, require_direct_icon)
        eligible: list[
            tuple[
                tuple[int, int, str],
                str,
                GroundCuaImageGroup,
                GroundCuaTarget,
                str,
            ]
        ] = []
        for software, items in pool.items():
            candidate = peek_candidate(
                split,
                category,
                require_direct_icon,
                software,
                items,
            )
            if candidate is None:
                continue
            rank, group, target, target_key = candidate
            eligible.append(
                (
                    (
                        software_counts_by_split[split][software],
                        selected_count_by_pixel[group.pixel_sha256],
                        rank,
                    ),
                    software,
                    group,
                    target,
                    target_key,
                )
            )
        if not eligible:
            return False
        _choice_key, software, group, target, target_key = min(
            eligible,
            key=lambda item: item[0],
        )
        position_key = (split, category, require_direct_icon, software)
        pool_positions[position_key] = pool_positions.get(position_key, 0) + 1
        group_id = _selection_group_id(group)
        assigned_split_by_group[group_id] = split
        selected_keys.add(target_key)
        selected_count_by_pixel[group.pixel_sha256] += 1
        software_counts_by_split[split][target.software] += 1
        remaining[split][category] -= 1
        selected.append(
            SelectedTarget(
                selection_id=(
                    f"groundcua_{split}_{group.pixel_sha256[:12]}_"
                    f"{candidate_rank(split, group, target, target_key)[:12]}"
                ),
                split=split,
                image_group=group,
                target=target,
            )
        )
        return True

    icons_needed = direct_icon_floor_required
    while icons_needed > 0:
        progressed = False
        for split in quotas:
            if icons_needed <= 0:
                break
            if remaining[split]["direct"] <= 0:
                continue
            if choose_one(split, "direct", require_direct_icon=True):
                icons_needed -= 1
                progressed = True
        if not progressed:
            break
    if icons_needed:
        raise ValueError(
            "Unable to satisfy direct_icon floor without violating grouped split constraints"
        )

    for split in quotas:
        for category in ("direct", "functional"):
            while remaining[split][category] > 0:
                if not choose_one(split, category):
                    raise ValueError(
                        f"Unable to satisfy {split} {category} quota; "
                        f"missing {remaining[split][category]} targets"
                    )

    split_counts: Counter[str] = Counter(item.split for item in selected)
    split_category_counts: dict[str, dict[str, int]] = {}
    for split in quotas:
        category_counts = Counter(
            item.target.category for item in selected if item.split == split
        )
        split_category_counts[split] = {
            "direct": category_counts["direct"],
            "functional": category_counts["functional"],
        }
    pixel_split_sets: dict[str, set[str]] = {}
    near_split_sets: dict[str, set[str]] = {}
    encoded_split_sets: dict[str, set[str]] = {}
    for item in selected:
        pixel_split_sets.setdefault(item.image_group.pixel_sha256, set()).add(item.split)
        encoded_split_sets.setdefault(item.image_group.encoded_sha256, set()).add(item.split)
        if item.image_group.confirmed_near_duplicate_group_id:
            near_split_sets.setdefault(
                item.image_group.confirmed_near_duplicate_group_id,
                set(),
            ).add(item.split)
    audit = {
        "selected_target_count": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "split_category_counts": {
            split: split_category_counts[split]
            for split in sorted(split_category_counts)
        },
        "software_counts_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(software_counts_by_split.items())
        },
        "max_targets_per_image": max_targets_per_image,
        "direct_icon_floor_required": direct_icon_floor_required,
        "direct_icon_selected": sum(
            item.target.instruction_type == "direct_icon" for item in selected
        ),
        "cross_split_encoded_sha_overlap": sum(
            len(splits) > 1 for splits in encoded_split_sets.values()
        ),
        "cross_split_pixel_sha_overlap": sum(
            len(splits) > 1 for splits in pixel_split_sets.values()
        ),
        "cross_split_confirmed_near_duplicate_overlap": sum(
            len(splits) > 1 for splits in near_split_sets.values()
        ),
    }
    return GroundCuaSelectionResult(selected_targets=tuple(selected), audit=audit)


def _all_target_source_components(
    image_groups: tuple[GroundCuaImageGroup, ...],
) -> tuple[tuple[str, tuple[GroundCuaImageGroup, ...]], ...]:
    """Join every source identity that is forbidden from crossing a split."""

    parent = list(range(len(image_groups)))
    rank = [0] * len(image_groups)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    pixel_owner: dict[str, int] = {}
    encoded_owner: dict[str, int] = {}
    source_id_owner: dict[str, int] = {}
    near_duplicate_owner: dict[str, int] = {}
    for index, group in enumerate(image_groups):
        for owner_map, values in (
            (pixel_owner, (group.pixel_sha256,)),
            (
                encoded_owner,
                group.all_encoded_sha256s or (group.encoded_sha256,),
            ),
            (
                source_id_owner,
                tuple(source_id for source_id in group.source_image_ids if source_id),
            ),
            (
                near_duplicate_owner,
                (
                    (group.confirmed_near_duplicate_group_id,)
                    if group.confirmed_near_duplicate_group_id
                    else ()
                ),
            ),
        ):
            for value in values:
                owner = owner_map.setdefault(value, index)
                union(index, owner)

    member_indexes: dict[int, list[int]] = {}
    for index in range(len(image_groups)):
        member_indexes.setdefault(find(index), []).append(index)

    components: list[tuple[str, tuple[GroundCuaImageGroup, ...]]] = []
    for indexes in member_indexes.values():
        groups = tuple(
            sorted(
                (image_groups[index] for index in indexes),
                key=lambda group: (
                    group.pixel_sha256,
                    group.encoded_sha256,
                    group.source_image_ids,
                ),
            )
        )
        pixel_shas = sorted(group.pixel_sha256 for group in groups)
        digest = hashlib.sha256(
            "|".join(pixel_shas).encode("utf-8")
        ).hexdigest()[:20]
        components.append((f"source-component-{digest}", groups))
    return tuple(sorted(components, key=lambda item: item[0]))


def _all_target_candidate_preference(
    group: GroundCuaImageGroup,
    target: GroundCuaTarget,
) -> tuple[Any, ...]:
    return (
        group.pixel_sha256,
        group.encoded_sha256,
        group.source_image_ids,
        target.source_image_id,
        target.source_target_index,
        target.software,
    )


def select_groundcua_all_targets(
    scan_result: GroundCuaScanResult,
    *,
    seed: int,
) -> GroundCuaSelectionResult:
    """Select every exact eligible annotation after group-first split assignment."""

    split_names = ("train", "val", "test")
    split_ratios = {"train": 10.0 / 12.0, "val": 1.0 / 12.0, "test": 1.0 / 12.0}
    feature_weights = {
        "total": 12.0,
        "category": 3.0,
        "instruction_type": 1.0,
        "software": 0.20,
    }
    components: list[dict[str, Any]] = []
    duplicate_identity_count = 0
    for component_id, groups in _all_target_source_components(scan_result.image_groups):
        candidates: dict[str, tuple[GroundCuaImageGroup, GroundCuaTarget]] = {}
        for group in groups:
            for target in group.targets:
                if target.category not in {"direct", "functional"}:
                    continue
                identity = _all_target_identity(group, target)
                current = candidates.get(identity)
                if current is None:
                    candidates[identity] = (group, target)
                    continue
                duplicate_identity_count += 1
                if _all_target_candidate_preference(group, target) < (
                    _all_target_candidate_preference(*current)
                ):
                    candidates[identity] = (group, target)
        if not candidates:
            continue
        ordered_targets = tuple(
            (identity, *candidates[identity]) for identity in sorted(candidates)
        )
        features: Counter[tuple[str, str]] = Counter()
        features[("total", "*")] = len(ordered_targets)
        for _identity, _group, target in ordered_targets:
            features[("category", target.category)] += 1
            features[("instruction_type", target.instruction_type)] += 1
            features[("software", target.software)] += 1
        components.append(
            {
                "component_id": component_id,
                "groups": groups,
                "targets": ordered_targets,
                "features": features,
            }
        )

    feature_totals: Counter[tuple[str, str]] = Counter()
    for component in components:
        feature_totals.update(component["features"])
    current_features = {split: Counter() for split in split_names}
    assigned_split_by_component: dict[str, str] = {}
    ordered_components = sorted(
        components,
        key=lambda component: (
            -component["features"][("total", "*")],
            hashlib.sha256(
                f"{seed}|{component['component_id']}|component_order".encode("utf-8")
            ).hexdigest(),
        ),
    )
    for component in ordered_components:
        candidates: list[tuple[float, str, str]] = []
        for split in split_names:
            delta = 0.0
            for feature, amount in component["features"].items():
                desired = feature_totals[feature] * split_ratios[split]
                denominator = max(1.0, desired)
                before = abs(current_features[split][feature] - desired) / denominator
                after = abs(
                    current_features[split][feature] + amount - desired
                ) / denominator
                delta += feature_weights[feature[0]] * (after - before)
            total_feature = ("total", "*")
            desired_total = feature_totals[total_feature] * split_ratios[split]
            after_total = (
                current_features[split][total_feature]
                + component["features"][total_feature]
            )
            overshoot = max(0.0, after_total - desired_total) / max(
                1.0,
                desired_total,
            )
            tie_break = hashlib.sha256(
                f"{seed}|{component['component_id']}|{split}|split_tie".encode("utf-8")
            ).hexdigest()
            candidates.append((delta + 30.0 * overshoot * overshoot, tie_break, split))
        _score, _tie_break, selected_split = min(candidates)
        assigned_split_by_component[component["component_id"]] = selected_split
        current_features[selected_split].update(component["features"])

    selected: list[SelectedTarget] = []
    software_counts_by_split = {split: Counter() for split in split_names}
    instruction_type_counts_by_split = {split: Counter() for split in split_names}
    category_counts_by_split = {split: Counter() for split in split_names}
    pixel_split_sets: dict[str, set[str]] = {}
    encoded_split_sets: dict[str, set[str]] = {}
    source_id_split_sets: dict[str, set[str]] = {}
    near_split_sets: dict[str, set[str]] = {}
    component_split_sets: dict[str, set[str]] = {}
    for component in ordered_components:
        component_id = str(component["component_id"])
        split = assigned_split_by_component[component_id]
        component_split_sets.setdefault(component_id, set()).add(split)
        for group in component["groups"]:
            pixel_split_sets.setdefault(group.pixel_sha256, set()).add(split)
            for encoded_sha in group.all_encoded_sha256s or (group.encoded_sha256,):
                encoded_split_sets.setdefault(encoded_sha, set()).add(split)
            for source_id in group.source_image_ids:
                if source_id:
                    source_id_split_sets.setdefault(source_id, set()).add(split)
            if group.confirmed_near_duplicate_group_id:
                near_split_sets.setdefault(
                    group.confirmed_near_duplicate_group_id,
                    set(),
                ).add(split)
        for identity, group, target in component["targets"]:
            suffix = hashlib.sha256(
                f"{seed}|all_targets|{identity}".encode("utf-8")
            ).hexdigest()[:12]
            selected.append(
                SelectedTarget(
                    selection_id=(
                        f"groundcua_{split}_{group.pixel_sha256[:12]}_{suffix}"
                    ),
                    split=split,
                    image_group=group,
                    target=target,
                    target_identity=identity,
                    source_component_id=component_id,
                )
            )
            software_counts_by_split[split][target.software] += 1
            instruction_type_counts_by_split[split][target.instruction_type] += 1
            category_counts_by_split[split][target.category] += 1

    split_counts = Counter(item.split for item in selected)
    audit = {
        "selection_mode": "all_targets",
        "selected_target_count": len(selected),
        "input_eligible_target_count": sum(
            len(group.targets) for group in scan_result.image_groups
        ),
        "exact_identity_duplicates_removed": duplicate_identity_count,
        "source_component_count": len(components),
        "split_counts": {split: split_counts[split] for split in split_names},
        "split_category_counts": {
            split: {
                "direct": category_counts_by_split[split]["direct"],
                "functional": category_counts_by_split[split]["functional"],
            }
            for split in split_names
        },
        "instruction_type_counts_by_split": {
            split: dict(sorted(instruction_type_counts_by_split[split].items()))
            for split in split_names
        },
        "software_counts_by_split": {
            split: dict(sorted(software_counts_by_split[split].items()))
            for split in split_names
        },
        "max_targets_per_image": None,
        "quota_sampling_applied": False,
        "direct_icon_floor_required": None,
        "cross_split_encoded_sha_overlap": sum(
            len(splits) > 1 for splits in encoded_split_sets.values()
        ),
        "cross_split_pixel_sha_overlap": sum(
            len(splits) > 1 for splits in pixel_split_sets.values()
        ),
        "cross_split_source_image_id_overlap": sum(
            len(splits) > 1 for splits in source_id_split_sets.values()
        ),
        "cross_split_source_component_overlap": sum(
            len(splits) > 1 for splits in component_split_sets.values()
        ),
        "cross_split_confirmed_near_duplicate_overlap": sum(
            len(splits) > 1 for splits in near_split_sets.values()
        ),
        "split_algorithm": "deterministic_connected_component_greedy_stratified_v1",
        "split_ratios": split_ratios,
    }
    return GroundCuaSelectionResult(selected_targets=tuple(selected), audit=audit)


def _one_bbox_candidate_identity(
    group: GroundCuaImageGroup,
    target: GroundCuaTarget,
) -> str:
    return _all_target_identity(group, target)


def _one_bbox_selected_candidates(
    image_groups: tuple[GroundCuaImageGroup, ...],
    *,
    seed: int,
) -> dict[
    str,
    tuple[
        GroundCuaImageGroup,
        GroundCuaTarget,
        str,
        int,
        int,
    ],
]:
    candidates_by_pixel: dict[
        str,
        dict[str, tuple[GroundCuaImageGroup, GroundCuaTarget]],
    ] = {}
    for group in image_groups:
        pixel_candidates = candidates_by_pixel.setdefault(group.pixel_sha256, {})
        for target in group.targets:
            if target.category not in {"direct", "functional"}:
                continue
            identity = _one_bbox_candidate_identity(group, target)
            current = pixel_candidates.get(identity)
            if current is None or _all_target_candidate_preference(
                group,
                target,
            ) < _all_target_candidate_preference(*current):
                pixel_candidates[identity] = (group, target)

    selected_by_pixel: dict[
        str,
        tuple[GroundCuaImageGroup, GroundCuaTarget, str, int, int],
    ] = {}
    for pixel_sha256, candidates in sorted(candidates_by_pixel.items()):
        if not candidates:
            continue
        ordered_identities = tuple(sorted(candidates))
        selected_index = (
            _stable_int(seed, pixel_sha256, "one_bbox_selection")
            % len(ordered_identities)
        )
        identity = ordered_identities[selected_index]
        group, target = candidates[identity]
        selected_by_pixel[pixel_sha256] = (
            group,
            target,
            identity,
            len(ordered_identities),
            selected_index,
        )
    return selected_by_pixel


def _one_bbox_component_split_assignment(
    components: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    split_names = ("train", "val", "test")
    split_ratios = {"train": 10.0 / 12.0, "val": 1.0 / 12.0, "test": 1.0 / 12.0}
    feature_weights = {
        "total": 12.0,
        "category": 3.0,
        "instruction_type": 1.0,
        "software": 0.20,
    }
    feature_totals: Counter[tuple[str, str]] = Counter()
    for component in components:
        feature_totals.update(component["features"])
    current_features = {split: Counter() for split in split_names}
    ordered_components = sorted(
        components,
        key=lambda component: (
            -component["features"][("total", "*")],
            hashlib.sha256(
                f"{seed}|{component['component_id']}|component_order".encode("utf-8")
            ).hexdigest(),
        ),
    )
    assigned_split_by_component: dict[str, str] = {}
    for component in ordered_components:
        split_candidates: list[tuple[float, str, str]] = []
        for split in split_names:
            delta = 0.0
            for feature, amount in component["features"].items():
                desired = feature_totals[feature] * split_ratios[split]
                denominator = max(1.0, desired)
                before = abs(current_features[split][feature] - desired) / denominator
                after = abs(
                    current_features[split][feature] + amount - desired
                ) / denominator
                delta += feature_weights[feature[0]] * (after - before)
            total_feature = ("total", "*")
            desired_total = feature_totals[total_feature] * split_ratios[split]
            after_total = (
                current_features[split][total_feature]
                + component["features"][total_feature]
            )
            overshoot = max(0.0, after_total - desired_total) / max(
                1.0,
                desired_total,
            )
            tie_break = hashlib.sha256(
                f"{seed}|{component['component_id']}|{split}|split_tie".encode("utf-8")
            ).hexdigest()
            split_candidates.append(
                (delta + 30.0 * overshoot * overshoot, tie_break, split)
            )
        _score, _tie_break, selected_split = min(split_candidates)
        component_id = str(component["component_id"])
        assigned_split_by_component[component_id] = selected_split
        current_features[selected_split].update(component["features"])
    return ordered_components, assigned_split_by_component


def select_groundcua_one_bbox_per_image(
    scan_result: GroundCuaScanResult,
    *,
    seed: int,
) -> GroundCuaSelectionResult:
    """Select one stable, uniformly weighted eligible bbox for each pixel image."""

    selected_by_pixel = _one_bbox_selected_candidates(
        scan_result.image_groups,
        seed=seed,
    )
    components: list[dict[str, Any]] = []
    component_groups_by_id: dict[str, tuple[GroundCuaImageGroup, ...]] = {}
    for component_id, groups in _all_target_source_components(scan_result.image_groups):
        component_groups_by_id[component_id] = groups
        component_targets = tuple(
            selected_by_pixel[pixel_sha256]
            for pixel_sha256 in sorted(
                {
                    group.pixel_sha256
                    for group in groups
                    if group.pixel_sha256 in selected_by_pixel
                }
            )
        )
        if not component_targets:
            continue
        features: Counter[tuple[str, str]] = Counter()
        features[("total", "*")] = len(component_targets)
        for _group, target, _identity, _count, _index in component_targets:
            features[("category", target.category)] += 1
            features[("instruction_type", target.instruction_type)] += 1
            features[("software", target.software)] += 1
        components.append(
            {
                "component_id": component_id,
                "groups": groups,
                "targets": component_targets,
                "features": features,
            }
        )

    ordered_components, assigned_split_by_component = (
        _one_bbox_component_split_assignment(components, seed=seed)
    )
    split_names = ("train", "val", "test")
    selected: list[SelectedTarget] = []
    selected_image_audit: list[dict[str, Any]] = []
    software_counts_by_split = {split: Counter() for split in split_names}
    instruction_type_counts_by_split = {split: Counter() for split in split_names}
    category_counts_by_split = {split: Counter() for split in split_names}
    pixel_split_sets: dict[str, set[str]] = {}
    encoded_split_sets: dict[str, set[str]] = {}
    source_id_split_sets: dict[str, set[str]] = {}
    near_split_sets: dict[str, set[str]] = {}
    component_split_sets: dict[str, set[str]] = {}
    for component in ordered_components:
        component_id = str(component["component_id"])
        split = assigned_split_by_component[component_id]
        component_split_sets.setdefault(component_id, set()).add(split)
        for group in component_groups_by_id[component_id]:
            pixel_split_sets.setdefault(group.pixel_sha256, set()).add(split)
            for encoded_sha in group.all_encoded_sha256s or (group.encoded_sha256,):
                encoded_split_sets.setdefault(encoded_sha, set()).add(split)
            for source_id in group.source_image_ids:
                if source_id:
                    source_id_split_sets.setdefault(source_id, set()).add(split)
            if group.confirmed_near_duplicate_group_id:
                near_split_sets.setdefault(
                    group.confirmed_near_duplicate_group_id,
                    set(),
                ).add(split)
        for group, target, identity, candidate_count, selected_index in component[
            "targets"
        ]:
            suffix = hashlib.sha256(
                f"{seed}|one_bbox_per_image|{identity}".encode("utf-8")
            ).hexdigest()[:12]
            item = SelectedTarget(
                selection_id=(
                    f"groundcua_{split}_{group.pixel_sha256[:12]}_{suffix}"
                ),
                split=split,
                image_group=group,
                target=target,
                target_identity=identity,
                source_component_id=component_id,
                augmentation_index=0,
                candidate_target_count=candidate_count,
                selected_candidate_index=selected_index,
            )
            selected.append(item)
            software_counts_by_split[split][target.software] += 1
            instruction_type_counts_by_split[split][target.instruction_type] += 1
            category_counts_by_split[split][target.category] += 1
            selected_image_audit.append(
                {
                    "source_pixel_sha256": group.pixel_sha256,
                    "candidate_target_count": candidate_count,
                    "selected_target_identity": identity,
                    "selected_candidate_index": selected_index,
                    "seed": seed,
                    "instruction_type": target.instruction_type,
                    "instruction_category": target.category,
                    "software": target.software,
                    "source_image_ids": list(group.source_image_ids),
                }
            )

    split_counts = Counter(item.split for item in selected)
    unique_pixel_count = len({group.pixel_sha256 for group in scan_result.image_groups})
    audit = {
        "selection_mode": "one_bbox_per_image",
        "selected_target_count": len(selected),
        "selected_image_count": len(selected),
        "image_with_eligible_bbox_count": len(selected_by_pixel),
        "image_without_eligible_bbox_count": unique_pixel_count - len(selected_by_pixel),
        "input_eligible_target_count": sum(
            len(group.targets) for group in scan_result.image_groups
        ),
        "source_component_count": len(components),
        "split_counts": {split: split_counts[split] for split in split_names},
        "split_category_counts": {
            split: {
                "direct": category_counts_by_split[split]["direct"],
                "functional": category_counts_by_split[split]["functional"],
            }
            for split in split_names
        },
        "instruction_type_counts_by_split": {
            split: dict(sorted(instruction_type_counts_by_split[split].items()))
            for split in split_names
        },
        "software_counts_by_split": {
            split: dict(sorted(software_counts_by_split[split].items()))
            for split in split_names
        },
        "eligible_bbox_count_distribution": dict(
            (str(key), value)
            for key, value in sorted(
                Counter(item.candidate_target_count for item in selected).items()
            )
        ),
        "max_targets_per_image": 1,
        "quota_sampling_applied": False,
        "direct_icon_floor_required": None,
        "selected_images": sorted(
            selected_image_audit,
            key=lambda row: str(row["source_pixel_sha256"]),
        ),
        "cross_split_encoded_sha_overlap": sum(
            len(splits) > 1 for splits in encoded_split_sets.values()
        ),
        "cross_split_pixel_sha_overlap": sum(
            len(splits) > 1 for splits in pixel_split_sets.values()
        ),
        "cross_split_source_image_id_overlap": sum(
            len(splits) > 1 for splits in source_id_split_sets.values()
        ),
        "cross_split_source_component_overlap": sum(
            len(splits) > 1 for splits in component_split_sets.values()
        ),
        "cross_split_confirmed_near_duplicate_overlap": sum(
            len(splits) > 1 for splits in near_split_sets.values()
        ),
        "split_algorithm": "deterministic_connected_component_greedy_stratified_v1",
        "split_ratios": {"train": 10.0 / 12.0, "val": 1.0 / 12.0, "test": 1.0 / 12.0},
    }
    return GroundCuaSelectionResult(selected_targets=tuple(selected), audit=audit)


def limit_groundcua_one_bbox_selection(
    selection: GroundCuaSelectionResult,
    *,
    seed: int,
    limit_images: int,
) -> GroundCuaSelectionResult:
    if limit_images <= 0:
        raise ValueError("limit_images must be a positive integer")
    ordered = sorted(
        selection.selected_targets,
        key=lambda item: (
            _stable_int(
                seed,
                item.image_group.pixel_sha256,
                "limit_images",
            ),
            item.image_group.pixel_sha256,
        ),
    )
    limited = tuple(ordered[:limit_images])
    split_names = ("train", "val", "test")
    split_counts = Counter(item.split for item in limited)
    category_counts = {split: Counter() for split in split_names}
    instruction_type_counts = {split: Counter() for split in split_names}
    software_counts = {split: Counter() for split in split_names}
    for item in limited:
        category_counts[item.split][item.target.category] += 1
        instruction_type_counts[item.split][item.target.instruction_type] += 1
        software_counts[item.split][item.target.software] += 1
    limited_pixels = {item.image_group.pixel_sha256 for item in limited}
    audit = dict(selection.audit)
    audit.update(
        {
            "pre_limit_selected_image_count": len(selection.selected_targets),
            "limit_images": limit_images,
            "selected_target_count": len(limited),
            "selected_image_count": len(limited),
            "source_component_count": len(
                {item.source_component_id for item in limited if item.source_component_id}
            ),
            "split_counts": {split: split_counts[split] for split in split_names},
            "split_category_counts": {
                split: {
                    "direct": category_counts[split]["direct"],
                    "functional": category_counts[split]["functional"],
                }
                for split in split_names
            },
            "instruction_type_counts_by_split": {
                split: dict(sorted(instruction_type_counts[split].items()))
                for split in split_names
            },
            "software_counts_by_split": {
                split: dict(sorted(software_counts[split].items()))
                for split in split_names
            },
            "eligible_bbox_count_distribution": dict(
                (str(key), value)
                for key, value in sorted(
                    Counter(item.candidate_target_count for item in limited).items()
                )
            ),
            "selected_images": [
                row
                for row in selection.audit.get("selected_images", [])
                if row.get("source_pixel_sha256") in limited_pixels
            ],
        }
    )
    return GroundCuaSelectionResult(selected_targets=limited, audit=audit)


def pixel_to_model(
    pixel_xy: tuple[float, float],
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[int, int]:
    image_width, image_height = _resolve_image_size(image_size)
    x, y = pixel_xy
    return (
        min(1000, max(0, int(round(float(x) / image_width * 1000)))),
        min(1000, max(0, int(round(float(y) / image_height * 1000)))),
    )


def model_to_execution_pixel(
    model_xy: tuple[float, float],
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[float, float]:
    image_width, image_height = _resolve_image_size(image_size)
    x, y = model_xy
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
        or not 0 <= float(x) <= 1000
        or not 0 <= float(y) <= 1000
    ):
        raise ValueError(f"model coordinate outside 0-1000: {model_xy}")
    return (
        float(x) / 1000.0 * image_width,
        float(y) / 1000.0 * image_height,
    )


def _quantized_point(
    pixel_xy: tuple[float, float],
    *,
    quadrant: str | None = None,
    image_size: tuple[int, int] | list[int] | None = None,
) -> QuantizedPoint:
    model_xy = pixel_to_model(pixel_xy, image_size=image_size)
    return QuantizedPoint(
        model_xy=model_xy,
        execution_pixel_xy=model_to_execution_pixel(model_xy, image_size=image_size),
        quadrant=quadrant,
    )


_QUADRANT_SPECS: tuple[tuple[str, tuple[float, float], tuple[float, float]], ...] = (
    ("top_left", (0.20, 0.42), (0.20, 0.42)),
    ("top_right", (0.58, 0.80), (0.20, 0.42)),
    ("bottom_left", (0.20, 0.42), (0.58, 0.80)),
    ("bottom_right", (0.58, 0.80), (0.58, 0.80)),
)


def _stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _bbox_contains(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def sample_target_point(
    bbox: BBox,
    *,
    seed: int,
    stable_key: str,
    augmentation_index: int,
    image_size: tuple[int, int] | list[int] | None = None,
) -> QuantizedPoint:
    quadrant_offset = _stable_int(seed, stable_key, "quadrant") % len(_QUADRANT_SPECS)
    quadrant_index = (quadrant_offset + augmentation_index) % len(_QUADRANT_SPECS)
    quadrant, u_range, v_range = _QUADRANT_SPECS[quadrant_index]
    for attempt in range(128):
        rng = random.Random(
            _stable_int(seed, stable_key, augmentation_index, quadrant, attempt)
        )
        u = rng.uniform(*u_range)
        v = rng.uniform(*v_range)
        if math.hypot(u - 0.5, v - 0.5) < 0.1:
            continue
        point = _quantized_point(
            (bbox.x1 + u * bbox.width, bbox.y1 + v * bbox.height),
            quadrant=quadrant,
            image_size=image_size,
        )
        if not _bbox_contains(point.execution_pixel_xy, bbox):
            continue
        executed_u = (point.execution_pixel_xy[0] - bbox.x1) / bbox.width
        executed_v = (point.execution_pixel_xy[1] - bbox.y1) / bbox.height
        if not 0.19 <= executed_u <= 0.81 or not 0.19 <= executed_v <= 0.81:
            continue
        if math.hypot(executed_u - 0.5, executed_v - 0.5) < 0.1:
            continue
        return point
    raise ValueError("Unable to sample a quantization-safe target point inside bbox")


def _proportional_schedule(
    count: int,
    *,
    weighted_values: tuple[tuple[Any, float], ...],
    seed: int,
    salt: str,
) -> list[Any]:
    if count < 0:
        raise ValueError("schedule count must be non-negative")
    raw_counts = [count * weight for _value, weight in weighted_values]
    integer_counts = [int(math.floor(value)) for value in raw_counts]
    remainder = count - sum(integer_counts)
    remainder_order = sorted(
        range(len(weighted_values)),
        key=lambda index: (
            -(raw_counts[index] - integer_counts[index]),
            index,
        ),
    )
    for index in remainder_order[:remainder]:
        integer_counts[index] += 1
    schedule: list[Any] = []
    for (value, _weight), value_count in zip(weighted_values, integer_counts, strict=True):
        schedule.extend([value] * value_count)
    random.Random(_stable_int(seed, salt)).shuffle(schedule)
    return schedule


def initial_cursor_mode_schedule(count: int, *, seed: int) -> list[str]:
    return _proportional_schedule(
        count,
        weighted_values=(
            ("center_near", 0.20),
            ("uniform", 0.40),
            ("opposite_side", 0.40),
        ),
        seed=seed,
        salt="initial_cursor_mode",
    )


def move_count_schedule(count: int, *, seed: int) -> list[int]:
    return _proportional_schedule(
        count,
        weighted_values=((1, 0.25), (2, 0.50), (3, 0.25)),
        seed=seed,
        salt="move_count",
    )


def all_target_trajectory_parameters(
    selected: SelectedTarget,
    *,
    seed: int,
) -> tuple[int, str]:
    """Choose trajectory parameters from the stable target identity, not list order."""

    identity = _selected_target_identity(selected)
    move_count = (1, 2, 2, 3)[_stable_int(seed, identity, "move_count") % 4]
    initial_cursor_mode = (
        "center_near",
        "uniform",
        "uniform",
        "opposite_side",
        "opposite_side",
    )[_stable_int(seed, identity, "initial_cursor_mode") % 5]
    return move_count, initial_cursor_mode


def one_bbox_trajectory_parameters(
    selected: SelectedTarget,
    *,
    seed: int,
) -> tuple[int, str]:
    pixel_sha256 = selected.image_group.pixel_sha256
    move_count = (1, 2, 2, 3, 3)[
        _stable_int(seed, pixel_sha256, "move_count") % 5
    ]
    initial_cursor_mode = (
        "center_near",
        "uniform",
        "uniform",
        "opposite_side",
        "opposite_side",
    )[_stable_int(seed, pixel_sha256, "initial_cursor_mode") % 5]
    return move_count, initial_cursor_mode


def sample_initial_cursor(
    *,
    bbox: BBox,
    target_execution_pixel_xy: tuple[float, float],
    mode: str,
    seed: int,
    stable_key: str,
    minimum_target_distance: float = 96.0,
    image_size: tuple[int, int] | list[int] | None = None,
) -> QuantizedPoint:
    image_width, image_height = _resolve_image_size(image_size)
    if mode not in {"center_near", "uniform", "opposite_side"}:
        raise ValueError(f"Unsupported initial cursor mode: {mode}")
    left_margin = 24.0
    top_margin = 24.0
    right_margin = image_width - 48.0
    bottom_margin = image_height - 48.0
    target_x, target_y = target_execution_pixel_xy
    for attempt in range(256):
        rng = random.Random(_stable_int(seed, stable_key, mode, attempt))
        if mode == "center_near":
            x = rng.uniform(image_width * 0.40, image_width * 0.60)
            y = rng.uniform(image_height * 0.40, image_height * 0.60)
        elif mode == "uniform":
            x = rng.uniform(left_margin, right_margin)
            y = rng.uniform(top_margin, bottom_margin)
        else:
            x_range = (
                (image_width * 0.05, image_width * 0.35)
                if target_x >= image_width / 2
                else (image_width * 0.65, image_width * 0.95)
            )
            y_range = (
                (image_height * 0.05, image_height * 0.35)
                if target_y >= image_height / 2
                else (image_height * 0.65, image_height * 0.95)
            )
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
        point = _quantized_point((x, y), image_size=image_size)
        exec_x, exec_y = point.execution_pixel_xy
        if not left_margin <= exec_x <= right_margin:
            continue
        if not top_margin <= exec_y <= bottom_margin:
            continue
        if _bbox_contains(point.execution_pixel_xy, bbox):
            continue
        if math.dist(point.execution_pixel_xy, target_execution_pixel_xy) < minimum_target_distance:
            continue
        return point
    raise ValueError(f"Unable to sample valid initial cursor for mode {mode}")


def distance_to_bbox(point: tuple[float, float], bbox: BBox) -> float:
    x, y = point
    dx = max(bbox.x1 - x, 0.0, x - bbox.x2)
    dy = max(bbox.y1 - y, 0.0, y - bbox.y2)
    return math.hypot(dx, dy)


def _path_is_valid(
    *,
    start: QuantizedPoint,
    target: QuantizedPoint,
    bbox: BBox,
    points: tuple[QuantizedPoint, ...],
    image_size: tuple[int, int] | list[int] | None = None,
) -> bool:
    image_width, image_height = _resolve_image_size(image_size)
    if not points or points[-1] != target:
        return False
    previous = start
    previous_target_distance = math.dist(
        previous.execution_pixel_xy,
        target.execution_pixel_xy,
    )
    previous_bbox_distance = distance_to_bbox(previous.execution_pixel_xy, bbox)
    seen_model_points = {start.model_xy}
    for index, point in enumerate(points):
        x, y = point.execution_pixel_xy
        if not 0 <= x <= image_width or not 0 <= y <= image_height:
            return False
        if point.model_xy in seen_model_points:
            return False
        seen_model_points.add(point.model_xy)
        if math.dist(previous.execution_pixel_xy, point.execution_pixel_xy) < 12.0:
            return False
        target_distance = math.dist(point.execution_pixel_xy, target.execution_pixel_xy)
        if target_distance >= previous_target_distance:
            return False
        bbox_distance = distance_to_bbox(point.execution_pixel_xy, bbox)
        if bbox_distance > previous_bbox_distance + 1e-9:
            return False
        if index < len(points) - 1 and _bbox_contains(point.execution_pixel_xy, bbox):
            return False
        previous = point
        previous_target_distance = target_distance
        previous_bbox_distance = bbox_distance
    return _bbox_contains(target.execution_pixel_xy, bbox)


def _candidate_waypoint_path(
    *,
    start: QuantizedPoint,
    target: QuantizedPoint,
    bbox: BBox,
    move_count: int,
    seed: int,
    stable_key: str,
    attempt: int,
    image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[QuantizedPoint, ...] | None:
    if move_count == 1:
        points = (target,)
        return points if _path_is_valid(start=start, target=target, bbox=bbox, points=points, image_size=image_size) else None
    rng = random.Random(_stable_int(seed, stable_key, move_count, attempt))
    if move_count == 2:
        alphas = (rng.uniform(0.55, 0.70),)
        jitter_scales = (0.08,)
    elif move_count == 3:
        alphas = (rng.uniform(0.30, 0.42), rng.uniform(0.68, 0.80))
        jitter_scales = (0.08, 0.035)
    else:
        raise ValueError("requested_move_count must be 1, 2, or 3")
    start_x, start_y = start.execution_pixel_xy
    target_x, target_y = target.execution_pixel_xy
    dx = target_x - start_x
    dy = target_y - start_y
    distance = math.hypot(dx, dy)
    if distance <= 0:
        return None
    perpendicular_x = -dy / distance
    perpendicular_y = dx / distance
    intermediate: list[QuantizedPoint] = []
    for alpha, jitter_scale in zip(alphas, jitter_scales, strict=True):
        max_jitter = min(120.0, distance * jitter_scale)
        jitter = rng.uniform(-max_jitter, max_jitter)
        pixel_xy = (
            start_x + alpha * dx + perpendicular_x * jitter,
            start_y + alpha * dy + perpendicular_y * jitter,
        )
        intermediate.append(_quantized_point(pixel_xy, image_size=image_size))
    points = (*intermediate, target)
    return points if _path_is_valid(start=start, target=target, bbox=bbox, points=points, image_size=image_size) else None


def generate_waypoint_path(
    *,
    start: QuantizedPoint,
    target: QuantizedPoint,
    bbox: BBox,
    requested_move_count: int,
    seed: int,
    stable_key: str,
    max_resamples: int = 32,
    image_size: tuple[int, int] | list[int] | None = None,
) -> WaypointPath:
    if requested_move_count not in {1, 2, 3}:
        raise ValueError("requested_move_count must be 1, 2, or 3")
    attempted_counts = range(requested_move_count, 0, -1)
    for move_count in attempted_counts:
        attempts = 1 if move_count == 1 else max_resamples
        for attempt in range(attempts):
            points = _candidate_waypoint_path(
                start=start,
                target=target,
                bbox=bbox,
                move_count=move_count,
                seed=seed,
                stable_key=stable_key,
                attempt=attempt,
                image_size=image_size,
            )
            if points is not None:
                downgrade_reason: str | None = None
                if move_count < requested_move_count:
                    if requested_move_count == 3 and move_count == 1:
                        downgrade_reason = "three_and_two_move_constraints_unsatisfied"
                    elif requested_move_count == 3:
                        downgrade_reason = "three_move_constraints_unsatisfied"
                    else:
                        downgrade_reason = "two_move_constraints_unsatisfied"
                return WaypointPath(
                    requested_move_count=requested_move_count,
                    actual_move_count=move_count,
                    points=tuple(points),
                    downgrade_reason=downgrade_reason,
                )
    raise ValueError("Unable to generate even a valid one-move target path")


def render_cursor_observation(
    source_path: Path,
    output_path: Path,
    *,
    cursor: QuantizedPoint,
) -> dict[str, Any]:
    from ..domains.ten_choice.sft_export import (
        CURSOR_OVERLAY_VERSION,
        overlay_cursor_on_image,
    )

    with Image.open(source_path) as image:
        image_size = tuple(int(value) for value in image.size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}-{threading.get_ident()}{output_path.suffix}"
    )
    try:
        overlay_cursor_on_image(
            source_path,
            temporary,
            cursor_xy=cursor.execution_pixel_xy,
        )
        with Image.open(temporary) as rendered:
            rendered.load()
            if tuple(rendered.size) != image_size:
                raise ValueError(
                    "GroundCUA cursor observation output size changed: "
                    f"expected {image_size}, got {rendered.size}"
                )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "cursor_overlay_version": CURSOR_OVERLAY_VERSION,
        "cursor_xy": [cursor.model_xy[0], cursor.model_xy[1]],
        "cursor_execution_pixel_xy": [
            cursor.execution_pixel_xy[0],
            cursor.execution_pixel_xy[1],
        ],
        "output_image_size": [image_size[0], image_size[1]],
    }


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-"
        f"{hashlib.sha256(content).hexdigest()[:12]}"
    )
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _image_extension(image_format: str) -> str:
    normalized = image_format.lower()
    if normalized in {"jpeg", "jpg"}:
        return ".jpg"
    if normalized == "webp":
        return ".webp"
    if normalized == "png":
        return ".png"
    raise ValueError(f"Unsupported embedded GroundCUA image format: {image_format}")


def extract_source_image(
    image_group: GroundCuaImageGroup,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    extension = _image_extension(image_group.image_format)
    output_path = (
        output_dir
        / "images"
        / "source"
        / f"{image_group.encoded_sha256}{extension}"
    )
    if output_path.exists():
        existing = output_path.read_bytes()
        if existing != image_group.encoded_bytes:
            raise ValueError(f"Existing extracted image bytes differ: {output_path}")
    else:
        _atomic_write_bytes(output_path, image_group.encoded_bytes)
    extracted = output_path.read_bytes()
    extracted_sha = hashlib.sha256(extracted).hexdigest()
    with Image.open(output_path) as image:
        output_size = [int(image.width), int(image.height)]
    if extracted_sha != image_group.encoded_sha256:
        raise ValueError(f"Extracted GroundCUA source SHA mismatch: {output_path}")
    expected_size = [int(image_group.width), int(image_group.height)]
    if output_size != expected_size:
        raise ValueError(
            f"Extracted GroundCUA source size mismatch: expected {expected_size}, got {output_size}"
        )
    return output_path.resolve(), {
        "source_image_sha256": image_group.encoded_sha256,
        "extracted_image_sha256": extracted_sha,
        "source_pixel_sha256": image_group.pixel_sha256,
        "source_image_size": [image_group.width, image_group.height],
        "output_image_size": output_size,
        "source_image_format": image_group.image_format,
        "extracted_path": str(output_path.resolve()),
        "resize_applied": False,
        "crop_applied": False,
        "letterbox_applied": False,
        "padding_applied": False,
        "stretch_applied": False,
    }


def _smoke_thought(*, action_kind: str, phase: str) -> str:
    if action_kind == "move_to" and phase == "final_position":
        return (
            "The white arrow with a black outline is the controllable cursor, and the requested "
            "control remains visibly identifiable in the static interface. This fixed positioning "
            "move places the cursor on that visible control before any mouse button is pressed."
        )
    if action_kind == "move_to":
        return (
            "The white arrow with a black outline shows the current controllable cursor away from "
            "the requested visible control. This fixed positioning move advances the cursor toward "
            "that control while leaving a later refinement step available if needed."
        )
    if action_kind == "mouse_down":
        return (
            "The controllable cursor is visibly positioned on the requested control, so pressing "
            "the left mouse button at the current location is the appropriate fixed next primitive "
            "action without claiming any page change."
        )
    if action_kind == "mouse_up":
        return (
            "The left mouse button has already been pressed at the current cursor location, so the "
            "appropriate fixed next primitive action is to release it and complete the click without "
            "claiming any unobserved interface response."
        )
    raise ValueError(f"Unsupported smoke thought action: {action_kind}")


def _canonical_action_step(
    action: dict[str, Any],
    *,
    phase: str,
    include_smoke_thoughts: bool = True,
) -> dict[str, Any]:
    step = {
        "type": "action",
        **action,
        "trainable": True,
    }
    if include_smoke_thoughts:
        step["assistant_response"] = build_canonical_assistant_response(
            thought=_smoke_thought(action_kind=str(action["kind"]), phase=phase),
            action=action,
        )
    return step


def _cursor_observation_path(
    *,
    output_dir: Path,
    record_id: str,
    step_index: int,
    point: QuantizedPoint,
) -> Path:
    x, y = point.model_xy
    return (
        output_dir
        / "images"
        / "cursor"
        / record_id
        / f"cursor_{step_index:02d}_{x:04d}_{y:04d}.png"
    )


def _observation_step(path: Path, point: QuantizedPoint) -> dict[str, Any]:
    return {
        "type": "observation",
        "image_path": str(path.resolve()),
        "cursor_xy": [point.model_xy[0], point.model_xy[1]],
    }


def _valid_cursor_observation(
    path: Path,
    *,
    image_size: tuple[int, int] | list[int] | None = None,
) -> bool:
    if not path.is_file():
        return False
    expected = _resolve_image_size(image_size)
    try:
        with Image.open(path) as image:
            image.load()
            return tuple(image.size) == expected
    except (OSError, ValueError):
        return False


def _valid_fullhd_cursor_observation(path: Path) -> bool:
    return _valid_cursor_observation(path)


def build_groundcua_trajectory(
    selected: SelectedTarget,
    *,
    output_dir: Path,
    seed: int,
    augmentation_index: int,
    requested_move_count: int,
    initial_cursor_mode: str,
    include_smoke_thoughts: bool = True,
) -> TrajectoryBuildResult:
    target_key = _selected_target_identity(selected)
    image_size = (int(selected.image_group.width), int(selected.image_group.height))
    target = sample_target_point(
        selected.target.bbox,
        seed=seed,
        stable_key=target_key,
        augmentation_index=augmentation_index,
        image_size=image_size,
    )
    start = sample_initial_cursor(
        bbox=selected.target.bbox,
        target_execution_pixel_xy=target.execution_pixel_xy,
        mode=initial_cursor_mode,
        seed=seed,
        stable_key=target_key,
        image_size=image_size,
    )
    path = generate_waypoint_path(
        start=start,
        target=target,
        bbox=selected.target.bbox,
        requested_move_count=requested_move_count,
        seed=seed,
        stable_key=target_key,
        image_size=image_size,
    )
    source_path, extraction_report = extract_source_image(selected.image_group, output_dir)
    record_id = f"{selected.selection_id}_a{augmentation_index:02d}"

    cursor_paths: list[Path] = []
    cursor_points = (start, *path.points)
    for cursor_index, point in enumerate(cursor_points):
        cursor_path = _cursor_observation_path(
            output_dir=output_dir,
            record_id=record_id,
            step_index=cursor_index,
            point=point,
        )
        if not _valid_cursor_observation(cursor_path, image_size=image_size):
            render_cursor_observation(source_path, cursor_path, cursor=point)
        cursor_paths.append(cursor_path.resolve())

    steps: list[dict[str, Any]] = [_observation_step(cursor_paths[0], start)]
    for point_index, point in enumerate(path.points):
        phase = "final_position" if point_index == len(path.points) - 1 else (
            "coarse_move" if point_index == 0 else "refine_move"
        )
        move_action = {
            "kind": "move_to",
            "x": point.model_xy[0],
            "y": point.model_xy[1],
        }
        steps.append(
            _canonical_action_step(
                move_action,
                phase=phase,
                include_smoke_thoughts=include_smoke_thoughts,
            )
        )
        steps.append(_observation_step(cursor_paths[point_index + 1], point))

    final_cursor_path = cursor_paths[-1]
    steps.append(
        _canonical_action_step(
            {"kind": "mouse_down"},
            phase="mouse_down",
            include_smoke_thoughts=include_smoke_thoughts,
        )
    )
    steps.append(_observation_step(final_cursor_path, target))
    steps.append(
        _canonical_action_step(
            {"kind": "mouse_up"},
            phase="mouse_up",
            include_smoke_thoughts=include_smoke_thoughts,
        )
    )
    steps.append(_observation_step(final_cursor_path, target))

    record = {
        "id": record_id,
        "source": "groundcua",
        "task_type": "grounding_multistep_static",
        "instruction": selected.target.instruction,
        "steps": steps,
        "metadata": {
            "coordinate_format": "qwen3_relative_0_1000",
            "task_action_kinds": ["move_to", "mouse_down", "mouse_up"],
            "protocol_track": "groundcua_static_progressive",
            "action_space_type": "strict_mouse_primitives",
            "protocol_version": "think_tag_json_action_v3",
            "trajectory_variant": f"progressive_{path.actual_move_count}_move",
            "requested_move_count": requested_move_count,
            "actual_move_count": path.actual_move_count,
            "augmentation_index": augmentation_index,
            "instruction_type": selected.target.instruction_type,
            "instruction_category": selected.target.category,
            "software": selected.target.software,
            "image_width": image_size[0],
            "image_height": image_size[1],
            "source_image_sha256": selected.image_group.encoded_sha256,
            "source_pixel_sha256": selected.image_group.pixel_sha256,
            "source_content_group_id": (
                selected.source_component_id
                or _selection_group_id(selected.image_group)
            ),
            "cursor_overlay_version": "white_arrow_black_outline_v1",
        "think_source": (
            "smoke_template_not_for_training"
            if include_smoke_thoughts
            else "pending_397b_teacher"
        ),
        "training_ready": False,
            "static_screenshot_semantics": "cursor_conditioned_progressive_positioning_only",
        },
    }

    positions = [start.execution_pixel_xy, *(point.execution_pixel_xy for point in path.points)]
    target_distances = [math.dist(position, target.execution_pixel_xy) for position in positions]
    bbox_distances = [distance_to_bbox(position, selected.target.bbox) for position in positions]
    audit = {
        "id": record_id,
        "split": selected.split,
        "selection_id": selected.selection_id,
        "source_image_ids": list(selected.image_group.source_image_ids),
        "source_image_sha256": selected.image_group.encoded_sha256,
        "source_pixel_sha256": selected.image_group.pixel_sha256,
        "all_source_encoded_sha256s": list(
            selected.image_group.all_encoded_sha256s
            or (selected.image_group.encoded_sha256,)
        ),
        "source_content_group_id": (
            selected.source_component_id
            or _selection_group_id(selected.image_group)
        ),
        "confirmed_near_duplicate_group_id": (
            selected.image_group.confirmed_near_duplicate_group_id
        ),
        "target_identity": target_key,
        "candidate_target_count": selected.candidate_target_count,
        "selected_candidate_index": selected.selected_candidate_index,
        "augmentation_index": augmentation_index,
        "instruction": selected.target.instruction,
        "instruction_type": selected.target.instruction_type,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "bbox": selected.target.bbox.as_list(),
        "target_pixel_xy": list(target.execution_pixel_xy),
        "target_model_xy": list(target.model_xy),
        "target_quadrant": target.quadrant,
        "initial_cursor_pixel_xy": list(start.execution_pixel_xy),
        "initial_cursor_model_xy": list(start.model_xy),
        "initial_cursor_mode": initial_cursor_mode,
        "requested_move_count": requested_move_count,
        "actual_move_count": path.actual_move_count,
        "downgrade_reason": path.downgrade_reason,
        "waypoints_pixel_xy": [list(point.execution_pixel_xy) for point in path.points],
        "waypoints_model_xy": [list(point.model_xy) for point in path.points],
        "target_distances_px": target_distances,
        "bbox_distances_px": bbox_distances,
        "distance_to_target_strictly_decreases": all(
            later < earlier for earlier, later in zip(target_distances, target_distances[1:])
        ),
        "distance_to_bbox_non_increasing": all(
            later <= earlier + 1e-9 for earlier, later in zip(bbox_distances, bbox_distances[1:])
        ),
        "final_move_hits_bbox": _bbox_contains(target.execution_pixel_xy, selected.target.bbox),
        "mouse_down_before_hit": not _bbox_contains(
            target.execution_pixel_xy,
            selected.target.bbox,
        ),
        "mouse_up_after_mouse_down": True,
        "source_extraction": extraction_report,
        "cursor_observation_paths": [str(path) for path in cursor_paths],
    }
    selected_target = {
        "selection_id": selected.selection_id,
        "split": selected.split,
        "instruction": selected.target.instruction,
        "instruction_type": selected.target.instruction_type,
        "instruction_category": selected.target.category,
        "software": selected.target.software,
        "source_image_ids": list(selected.image_group.source_image_ids),
        "source_image_sha256": selected.image_group.encoded_sha256,
        "source_pixel_sha256": selected.image_group.pixel_sha256,
        "source_content_group_id": (
            selected.source_component_id
            or _selection_group_id(selected.image_group)
        ),
        "target_identity": target_key,
        "source_target_index": selected.target.source_target_index,
        "bbox": selected.target.bbox.as_list(),
        "candidate_target_count": selected.candidate_target_count,
        "selected_candidate_index": selected.selected_candidate_index,
        "augmentation_index": augmentation_index,
    }
    return TrajectoryBuildResult(
        record=record,
        selected_target=selected_target,
        audit=audit,
    )


def _jsonl_text(rows: Iterable[dict[str, Any]]) -> str:
    materialized = list(rows)
    if not materialized:
        return ""
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in materialized
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(path, _jsonl_text(rows))


def _record_oracle_keys(value: Any) -> set[str]:
    forbidden = {
        "bbox",
        "bounding_box",
        "bbox_pixels",
        "target_point",
        "target_pixel_xy",
        "target_model_xy",
        "waypoint",
        "waypoints",
        "waypoints_pixel_xy",
        "waypoints_model_xy",
        "source_target_index",
        "ground_truth",
        "element_id",
        "dom",
        "oracle",
    }
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in forbidden:
                found.add(str(key))
            found.update(_record_oracle_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_record_oracle_keys(item))
    return found


def _split_overlap_counts(records_by_split: Mapping[str, list[dict[str, Any]]]) -> dict[str, int]:
    hashes = {
        split: {
            str(record.get("metadata", {}).get("source_image_sha256"))
            for record in records
        }
        for split, records in records_by_split.items()
    }
    result: dict[str, int] = {}
    split_names = [
        split for split in ("train", "val", "test") if split in records_by_split
    ]
    split_names.extend(
        split for split in sorted(records_by_split) if split not in split_names
    )
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            result[f"{first}_{second}"] = len(hashes[first] & hashes[second])
    return result


def _pairwise_split_set_overlaps(
    values_by_split: Mapping[str, set[str]],
) -> dict[str, int]:
    split_names = [
        split for split in ("train", "val", "test") if split in values_by_split
    ]
    split_names.extend(
        split for split in sorted(values_by_split) if split not in split_names
    )
    result: dict[str, int] = {}
    for first_index, first in enumerate(split_names):
        for second in split_names[first_index + 1 :]:
            result[f"{first}_{second}"] = len(
                values_by_split[first] & values_by_split[second]
            )
    return result


def _action_steps(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for step in record.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "action"
    ]


def _fullhd_image_validation(
    path: Path,
    expected_size: tuple[int, int] | list[int] | None = None,
) -> tuple[bool, bool]:
    if not path.is_file():
        return True, False
    try:
        with Image.open(path) as image:
            return False, tuple(image.size) != _resolve_image_size(expected_size)
    except Exception:
        return False, True


def _validate_export(
    *,
    output_dir: Path,
    records_by_split: Mapping[str, list[dict[str, Any]]],
    audits: list[dict[str, Any]],
    expected_target_count: int | None = None,
    expected_selected_image_count: int | None = None,
    require_unique_pixel_images: bool = False,
    expected_think_source: str | None = None,
    require_canonical_responses: bool = True,
    workers: int = 1,
    max_pending: int | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    errors: list[str] = []
    all_records = [record for records in records_by_split.values() for record in records]
    all_target_expected_count = (
        len(all_records) if expected_target_count is None else int(expected_target_count)
    )
    if len(all_records) != all_target_expected_count:
        errors.append(
            "record count does not match expected eligible target count: "
            f"{len(all_records)} != {all_target_expected_count}"
        )
    selected_image_count = len(audits)
    resolved_expected_selected_image_count = (
        selected_image_count
        if expected_selected_image_count is None
        else int(expected_selected_image_count)
    )
    record_count_matches_selected_image_count = (
        len(all_records) == selected_image_count
    )
    if not record_count_matches_selected_image_count:
        errors.append(
            "record count does not match selected image count: "
            f"{len(all_records)} != {selected_image_count}"
        )
    selected_image_coverage_matches = (
        selected_image_count == resolved_expected_selected_image_count
    )
    if not selected_image_coverage_matches:
        errors.append(
            "selected image count does not match eligible/limit expectation: "
            f"{selected_image_count} != {resolved_expected_selected_image_count}"
        )
    record_pixel_shas = [
        str(record.get("metadata", {}).get("source_pixel_sha256") or "")
        for record in all_records
    ]
    unique_source_pixel_sha256_count = len(set(record_pixel_shas))
    duplicate_source_pixel_sha256_count = (
        len(record_pixel_shas) - unique_source_pixel_sha256_count
    )
    if any(not value for value in record_pixel_shas):
        errors.append("one or more records are missing source_pixel_sha256")
    if require_unique_pixel_images and duplicate_source_pixel_sha256_count:
        errors.append(
            "duplicate source pixel images: "
            f"{duplicate_source_pixel_sha256_count}"
        )
    target_identities = [
        str(audit.get("target_identity") or "") for audit in audits
    ]
    unique_target_identity_count = len(set(target_identities))
    duplicate_target_identity_count = len(target_identities) - unique_target_identity_count
    if any(not identity for identity in target_identities):
        errors.append("one or more audit rows are missing target identity")
    if duplicate_target_identity_count:
        errors.append(
            f"duplicate target identities: {duplicate_target_identity_count}"
        )
    oracle_leak_count = sum(bool(_record_oracle_keys(record)) for record in all_records)
    if oracle_leak_count:
        errors.append(f"oracle leaks in {oracle_leak_count} records")
    expected_image_sizes: dict[Path, tuple[int, int]] = {}
    image_paths = {
        Path(step["image_path"])
        for record in all_records
        for step in record.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "observation" and step.get("image_path")
    }
    for record in all_records:
        metadata = record.get("metadata", {})
        if metadata.get("image_width") and metadata.get("image_height"):
            expected = _resolve_image_size((metadata["image_width"], metadata["image_height"]))
            for step in record.get("steps", []):
                if isinstance(step, dict) and step.get("type") == "observation" and step.get("image_path"):
                    expected_image_sizes[Path(step["image_path"])] = expected
    image_paths.update((output_dir / "images" / "source").glob("*"))
    for audit in audits:
        extraction = audit.get("source_extraction")
        if isinstance(extraction, dict) and extraction.get("extracted_path"):
            expected_image_sizes[Path(extraction["extracted_path"])] = _resolve_image_size(
                extraction.get("source_image_size")
            )
    resolved_max_pending = workers * 2 if max_pending is None else int(max_pending)
    image_validation = ordered_bounded_map(
        lambda path: _fullhd_image_validation(path, expected_image_sizes.get(path)),
        sorted(image_paths),
        max_workers=workers,
        max_pending=resolved_max_pending,
    )
    missing_image_count = 0
    non_fullhd_image_count = 0
    for missing, non_fullhd in image_validation:
        missing_image_count += int(missing)
        non_fullhd_image_count += int(non_fullhd)
    if missing_image_count:
        errors.append(f"missing images: {missing_image_count}")
    if non_fullhd_image_count:
        errors.append(f"non-FullHD images: {non_fullhd_image_count}")
    source_image_dimension_mismatch_count = 0
    cursor_image_dimension_mismatch_count = 0
    for audit in audits:
        expected_size = _resolve_image_size(
            (audit.get("image_width"), audit.get("image_height"))
        )
        extraction = audit.get("source_extraction")
        if isinstance(extraction, dict) and extraction.get("extracted_path"):
            missing, mismatch = _fullhd_image_validation(
                Path(extraction["extracted_path"]),
                expected_size,
            )
            source_image_dimension_mismatch_count += int(missing or mismatch)
        for cursor_path in audit.get("cursor_observation_paths") or []:
            missing, mismatch = _fullhd_image_validation(
                Path(cursor_path),
                expected_size,
            )
            cursor_image_dimension_mismatch_count += int(missing or mismatch)
    if source_image_dimension_mismatch_count:
        errors.append(
            "source image dimension mismatches: "
            f"{source_image_dimension_mismatch_count}"
        )
    if cursor_image_dimension_mismatch_count:
        errors.append(
            "cursor image dimension mismatches: "
            f"{cursor_image_dimension_mismatch_count}"
        )
    overlaps = _split_overlap_counts(records_by_split)
    if any(overlaps.values()):
        errors.append(f"split source image SHA overlap: {overlaps}")
    audit_rows_by_split = {
        split: [audit for audit in audits if audit.get("split") == split]
        for split in records_by_split
    }
    encoded_values_by_split = {
        split: {
            str(encoded_sha)
            for audit in split_audits
            for encoded_sha in (
                audit.get("all_source_encoded_sha256s")
                or [audit.get("source_image_sha256")]
            )
            if encoded_sha
        }
        for split, split_audits in audit_rows_by_split.items()
    }
    pixel_values_by_split = {
        split: {
            str(audit.get("source_pixel_sha256"))
            for audit in split_audits
            if audit.get("source_pixel_sha256")
        }
        for split, split_audits in audit_rows_by_split.items()
    }
    source_id_values_by_split = {
        split: {
            str(source_id)
            for audit in split_audits
            for source_id in (audit.get("source_image_ids") or [])
            if source_id
        }
        for split, split_audits in audit_rows_by_split.items()
    }
    content_group_values_by_split = {
        split: {
            str(audit.get("source_content_group_id"))
            for audit in split_audits
            if audit.get("source_content_group_id")
        }
        for split, split_audits in audit_rows_by_split.items()
    }
    near_group_values_by_split = {
        split: {
            str(audit.get("confirmed_near_duplicate_group_id"))
            for audit in split_audits
            if audit.get("confirmed_near_duplicate_group_id")
        }
        for split, split_audits in audit_rows_by_split.items()
    }
    encoded_overlaps = _pairwise_split_set_overlaps(encoded_values_by_split)
    pixel_overlaps = _pairwise_split_set_overlaps(pixel_values_by_split)
    source_id_overlaps = _pairwise_split_set_overlaps(source_id_values_by_split)
    content_group_overlaps = _pairwise_split_set_overlaps(
        content_group_values_by_split
    )
    near_group_overlaps = _pairwise_split_set_overlaps(near_group_values_by_split)
    for label, split_overlaps in (
        ("encoded SHA", encoded_overlaps),
        ("pixel SHA", pixel_overlaps),
        ("source image ID", source_id_overlaps),
        ("source content group", content_group_overlaps),
        ("confirmed near-duplicate group", near_group_overlaps),
    ):
        if any(split_overlaps.values()):
            errors.append(f"split {label} overlap: {split_overlaps}")
    source_extraction_sha_mismatch_count = 0
    transform_counts = Counter()
    invalid_bbox_count = 0
    target_outside_bbox_count = 0
    for audit in audits:
        extraction = audit.get("source_extraction")
        if isinstance(extraction, dict):
            if extraction.get("source_image_sha256") != extraction.get(
                "extracted_image_sha256"
            ):
                source_extraction_sha_mismatch_count += 1
            for transform_name in (
                "resize",
                "crop",
                "letterbox",
                "padding",
                "stretch",
            ):
                transform_counts[transform_name] += int(
                    bool(extraction.get(f"{transform_name}_applied"))
                )
        bbox, _reason = _bbox_or_reason(
            audit.get("bbox"),
            image_size=(audit.get("image_width"), audit.get("image_height"))
            if audit.get("image_width") and audit.get("image_height")
            else None,
        )
        if bbox is None:
            invalid_bbox_count += 1
            continue
        target_xy = audit.get("target_pixel_xy")
        if (
            not isinstance(target_xy, list)
            or len(target_xy) != 2
            or not all(isinstance(value, (int, float)) for value in target_xy)
            or not _bbox_contains((float(target_xy[0]), float(target_xy[1])), bbox)
        ):
            target_outside_bbox_count += 1
    if source_extraction_sha_mismatch_count:
        errors.append(
            "source extraction SHA mismatches: "
            f"{source_extraction_sha_mismatch_count}"
        )
    if any(transform_counts.values()):
        errors.append(f"forbidden source transforms: {dict(transform_counts)}")
    if invalid_bbox_count:
        errors.append(f"invalid bbox rows: {invalid_bbox_count}")
    if target_outside_bbox_count:
        errors.append(f"target points outside bbox: {target_outside_bbox_count}")
    final_miss = sum(not bool(audit.get("final_move_hits_bbox")) for audit in audits)
    down_before_hit = sum(bool(audit.get("mouse_down_before_hit")) for audit in audits)
    no_release = sum(
        [step.get("kind") for step in _action_steps(record)][-2:] != ["mouse_down", "mouse_up"]
        for record in all_records
    )
    non_monotone_target = sum(
        not bool(audit.get("distance_to_target_strictly_decreases")) for audit in audits
    )
    non_monotone_bbox = sum(
        not bool(audit.get("distance_to_bbox_non_increasing")) for audit in audits
    )
    assistant_response_missing = sum(
        not isinstance(step.get("assistant_response"), str)
        for record in all_records
        for step in _action_steps(record)
    )
    unexpected_assistant_response_count = sum(
        isinstance(step.get("assistant_response"), str)
        for record in all_records
        for step in _action_steps(record)
    ) if not require_canonical_responses else 0
    if unexpected_assistant_response_count:
        errors.append(
            "unexpected assistant_response before teacher labeling: "
            f"{unexpected_assistant_response_count}"
        )
    requested_move_count_error_count = 0
    actual_move_count_error_count = 0
    move_count_order_error_count = 0
    action_move_count_mismatch_count = 0
    intermediate_waypoint_inside_bbox_count = 0
    repeated_waypoint_count = 0
    audit_by_id = {str(audit.get("id")): audit for audit in audits}
    observation_action_sequence_error_count = 0
    for record in all_records:
        steps = record.get("steps")
        actions = _action_steps(record)
        metadata = record.get("metadata", {})
        requested_move_count = metadata.get("requested_move_count")
        actual_move_count = metadata.get("actual_move_count")
        if requested_move_count not in {1, 2, 3}:
            requested_move_count_error_count += 1
        if actual_move_count not in {1, 2, 3}:
            actual_move_count_error_count += 1
        if (
            requested_move_count in {1, 2, 3}
            and actual_move_count in {1, 2, 3}
            and actual_move_count > requested_move_count
        ):
            move_count_order_error_count += 1
        valid_alternation = (
            isinstance(steps, list)
            and len(steps) == len(actions) * 2 + 1
            and all(
                isinstance(step, dict)
                and step.get("type") == ("observation" if index % 2 == 0 else "action")
                for index, step in enumerate(steps)
            )
        )
        action_kinds = [step.get("kind") for step in actions]
        if actual_move_count in {1, 2, 3} and action_kinds != [
            *(["move_to"] * int(actual_move_count)),
            "mouse_down",
            "mouse_up",
        ]:
            action_move_count_mismatch_count += 1
        valid_primitives = all(
            kind in {"move_to", "mouse_down", "mouse_up"}
            for kind in action_kinds
        )
        if (
            not valid_alternation
            or not valid_primitives
            or action_kinds[-2:] != ["mouse_down", "mouse_up"]
        ):
            observation_action_sequence_error_count += 1
        audit = audit_by_id.get(str(record.get("id")), {})
        bbox, _reason = _bbox_or_reason(
            audit.get("bbox"),
            image_size=(audit.get("image_width"), audit.get("image_height"))
            if audit.get("image_width") and audit.get("image_height")
            else None,
        )
        waypoint_pixels = audit.get("waypoints_pixel_xy") or []
        if bbox is not None:
            intermediate_waypoint_inside_bbox_count += sum(
                isinstance(point, list)
                and len(point) == 2
                and _bbox_contains((float(point[0]), float(point[1])), bbox)
                for point in waypoint_pixels[:-1]
            )
        waypoint_models = [tuple(point) for point in audit.get("waypoints_model_xy") or []]
        initial_model = tuple(audit.get("initial_cursor_model_xy") or ())
        repeated_waypoint_count += len([initial_model, *waypoint_models]) - len(
            set([initial_model, *waypoint_models])
        )
    for label, count in (
        ("invalid requested move counts", requested_move_count_error_count),
        ("invalid actual move counts", actual_move_count_error_count),
        ("actual move count exceeds requested", move_count_order_error_count),
        ("action/move count mismatches", action_move_count_mismatch_count),
        ("intermediate waypoints inside bbox", intermediate_waypoint_inside_bbox_count),
        ("repeated waypoints", repeated_waypoint_count),
    ):
        if count:
            errors.append(f"{label}: {count}")
    think_source_mismatch_count = 0
    training_ready_true_count = 0
    for record in all_records:
        metadata = record.get("metadata", {})
        if expected_think_source is not None and metadata.get("think_source") != expected_think_source:
            think_source_mismatch_count += 1
        training_ready_true_count += int(metadata.get("training_ready") is not False)
    if think_source_mismatch_count:
        errors.append(f"think_source mismatches: {think_source_mismatch_count}")
    if training_ready_true_count:
        errors.append(f"records not marked training_ready=false: {training_ready_true_count}")
    for label, count in (
        ("final move outside bbox", final_miss),
        ("mouse_down before hit", down_before_hit),
        ("missing mouse release", no_release),
        ("non-monotone target path", non_monotone_target),
        ("non-monotone bbox path", non_monotone_bbox),
        (
            "missing assistant_response",
            assistant_response_missing if require_canonical_responses else 0,
        ),
        ("observation/action sequence errors", observation_action_sequence_error_count),
    ):
        if count:
            errors.append(f"{label}: {count}")
    loader_examples = []
    split_loader_counts: dict[str, int] = {}
    if require_canonical_responses:
        from ..train.qwen3_vl_sft import load_sft_examples

        for split in records_by_split:
            path = output_dir / f"{split}.jsonl"
            examples = load_sft_examples([path])
            split_loader_counts[split] = len(examples)
            loader_examples.extend(examples)
    else:
        split_loader_counts = {split: 0 for split in records_by_split}
    return {
        "status": (
            "passed"
            if not errors and require_canonical_responses
            else "passed_pending_teacher"
            if not errors
            else "failed"
        ),
        "errors": errors,
        "checked_record_count": len(all_records),
        "selected_image_count": selected_image_count,
        "expected_selected_image_count": resolved_expected_selected_image_count,
        "selected_image_coverage_matches": selected_image_coverage_matches,
        "unique_source_pixel_sha256_count": unique_source_pixel_sha256_count,
        "duplicate_source_pixel_sha256_count": duplicate_source_pixel_sha256_count,
        "record_count_matches_selected_image_count": record_count_matches_selected_image_count,
        "all_target_expected_count": all_target_expected_count,
        "unique_target_identity_count": unique_target_identity_count,
        "duplicate_target_identity_count": duplicate_target_identity_count,
        "checked_image_count": len(image_paths),
        "validation_workers": workers,
        "validation_max_pending": resolved_max_pending,
        "missing_image_count": missing_image_count,
        "non_fullhd_image_count": non_fullhd_image_count,
        "source_image_dimension_mismatch_count": source_image_dimension_mismatch_count,
        "cursor_image_dimension_mismatch_count": cursor_image_dimension_mismatch_count,
        "oracle_leak_count": oracle_leak_count,
        "split_source_image_sha256_overlap": overlaps,
        "split_encoded_sha256_overlap": encoded_overlaps,
        "split_pixel_sha256_overlap": pixel_overlaps,
        "split_source_image_id_overlap": source_id_overlaps,
        "split_source_content_group_overlap": content_group_overlaps,
        "split_confirmed_near_duplicate_group_overlap": near_group_overlaps,
        "source_extraction_sha_mismatch_count": source_extraction_sha_mismatch_count,
        "resize_applied_count": transform_counts["resize"],
        "crop_applied_count": transform_counts["crop"],
        "letterbox_applied_count": transform_counts["letterbox"],
        "padding_applied_count": transform_counts["padding"],
        "stretch_applied_count": transform_counts["stretch"],
        "invalid_bbox_count": invalid_bbox_count,
        "target_outside_bbox_count": target_outside_bbox_count,
        "final_move_outside_bbox_count": final_miss,
        "mouse_down_before_hit_count": down_before_hit,
        "no_release_count": no_release,
        "distance_to_target_non_monotone_count": non_monotone_target,
        "distance_to_bbox_increase_count": non_monotone_bbox,
        "assistant_response_missing_count": assistant_response_missing,
        "unexpected_assistant_response_count": unexpected_assistant_response_count,
        "requested_move_count_error_count": requested_move_count_error_count,
        "actual_move_count_error_count": actual_move_count_error_count,
        "move_count_order_error_count": move_count_order_error_count,
        "action_move_count_mismatch_count": action_move_count_mismatch_count,
        "intermediate_waypoint_inside_bbox_count": intermediate_waypoint_inside_bbox_count,
        "repeated_waypoint_count": repeated_waypoint_count,
        "canonical_responses_required": require_canonical_responses,
        "pending_teacher_think_action_count": (
            assistant_response_missing if not require_canonical_responses else 0
        ),
        "training_ready": False,
        "think_source": expected_think_source,
        "think_source_mismatch_count": think_source_mismatch_count,
        "training_ready_true_count": training_ready_true_count,
        "cross_split_encoded_sha_overlap": sum(encoded_overlaps.values()),
        "cross_split_pixel_sha_overlap": sum(pixel_overlaps.values()),
        "cross_split_source_image_id_overlap": sum(source_id_overlaps.values()),
        "cross_split_source_component_overlap": sum(content_group_overlaps.values()),
        "cross_split_confirmed_near_duplicate_overlap": sum(near_group_overlaps.values()),
        "observation_action_sequence_error_count": observation_action_sequence_error_count,
        "loader_example_count": len(loader_examples),
        "split_loader_example_counts": dict(sorted(split_loader_counts.items())),
    }


RESOLUTION_POLICY = {
    "resolution_policy": "preserve_source_resolution_transform_at_train_time",
    "source_image_required_size": None,
    "output_image_size": None,
    "resize_allowed": True,
    "crop_allowed": False,
    "letterbox_allowed": False,
    "image_max_pixels": IMAGE_MAX_PIXELS,
    "source_resolution_preserved": True,
}


def export_groundcua_subset(
    *,
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    split_quotas: Mapping[str, SplitQuota] | None = None,
    profile_name: str = "smoke",
    limit_images: int | None = None,
    scan_workers: int | None = None,
    scan_max_pending: int | None = None,
    parquet_file_workers: int = 2,
    parquet_prefetch_batches: int = 2,
    render_workers: int | None = None,
    render_max_pending: int | None = None,
    validation_workers: int | None = None,
    validation_max_pending: int | None = None,
) -> dict[str, Any]:
    all_targets_mode = profile_name == "all_targets"
    one_bbox_mode = profile_name == "one_bbox_per_image"
    pending_teacher_mode = all_targets_mode or one_bbox_mode
    if limit_images is not None and not one_bbox_mode:
        raise ValueError("limit_images is only supported by one_bbox_per_image")
    if limit_images is not None and limit_images <= 0:
        raise ValueError("limit_images must be a positive integer")
    if all_targets_mode or one_bbox_mode:
        if split_quotas is not None:
            raise ValueError(f"{profile_name} profile does not accept split quotas")
        quotas: dict[str, SplitQuota] = {}
        split_names = ("train", "val", "test")
    else:
        quotas = dict(split_quotas or profile_split_quotas(profile_name))
        split_names = tuple(quotas)
    default_workers = recommended_local_workers()
    scan_workers = (
        recommended_scan_workers() if scan_workers is None else int(scan_workers)
    )
    render_workers = default_workers if render_workers is None else int(render_workers)
    validation_workers = (
        default_workers if validation_workers is None else int(validation_workers)
    )
    resolved_scan_max_pending = (
        scan_workers * 2 if scan_max_pending is None else int(scan_max_pending)
    )
    resolved_render_max_pending = (
        render_workers * 2 if render_max_pending is None else int(render_max_pending)
    )
    resolved_validation_max_pending = (
        validation_workers * 2
        if validation_max_pending is None
        else int(validation_max_pending)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_result = scan_groundcua_dataset(
        dataset_root,
        workers=scan_workers,
        max_pending=resolved_scan_max_pending,
        parquet_file_workers=parquet_file_workers,
        parquet_prefetch_batches=parquet_prefetch_batches,
    )
    if all_targets_mode:
        selection = select_groundcua_all_targets(scan_result, seed=seed)
        trajectory_parameters = [
            all_target_trajectory_parameters(selected, seed=seed)
            for selected in selection.selected_targets
        ]
        move_counts = [parameters[0] for parameters in trajectory_parameters]
        cursor_modes = [parameters[1] for parameters in trajectory_parameters]
    elif one_bbox_mode:
        selection = select_groundcua_one_bbox_per_image(scan_result, seed=seed)
        if limit_images is not None:
            selection = limit_groundcua_one_bbox_selection(
                selection,
                seed=seed,
                limit_images=limit_images,
            )
        trajectory_parameters = [
            one_bbox_trajectory_parameters(selected, seed=seed)
            for selected in selection.selected_targets
        ]
        move_counts = [parameters[0] for parameters in trajectory_parameters]
        cursor_modes = [parameters[1] for parameters in trajectory_parameters]
    else:
        selection = select_groundcua_subset(
            scan_result,
            split_quotas=quotas,
            seed=seed,
        )
        move_counts = move_count_schedule(len(selection.selected_targets), seed=seed)
        cursor_modes = initial_cursor_mode_schedule(len(selection.selected_targets), seed=seed)
    unique_source_groups = list(
        {
            selected.image_group.encoded_sha256: selected.image_group
            for selected in selection.selected_targets
        }.values()
    )

    def extract_group(image_group: GroundCuaImageGroup) -> tuple[Path, dict[str, Any]]:
        return extract_source_image(image_group, output_dir)

    list(
        ordered_bounded_map(
            extract_group,
            unique_source_groups,
            max_workers=render_workers,
            max_pending=resolved_render_max_pending,
        )
    )
    trajectory_jobs = list(
        zip(
            range(len(selection.selected_targets)),
            selection.selected_targets,
            move_counts,
            cursor_modes,
            strict=True,
        )
    )

    def build_job(
        job: tuple[int, SelectedTarget, int, str],
    ) -> TrajectoryBuildResult:
        index, selected, requested_move_count, initial_cursor_mode = job
        del index
        return build_groundcua_trajectory(
            selected,
            output_dir=output_dir,
            seed=seed,
            augmentation_index=selected.augmentation_index,
            requested_move_count=requested_move_count,
            initial_cursor_mode=initial_cursor_mode,
            include_smoke_thoughts=not pending_teacher_mode,
        )

    built = list(
        ordered_bounded_map(
            build_job,
            trajectory_jobs,
            max_workers=render_workers,
            max_pending=resolved_render_max_pending,
        )
    )
    records_by_split: dict[str, list[dict[str, Any]]] = {
        split: [item.record for item in built if item.audit["split"] == split]
        for split in split_names
    }
    all_records = [item.record for item in built]
    selected_targets = [item.selected_target for item in built]
    audits = [item.audit for item in built]
    _write_jsonl(output_dir / "records.jsonl", all_records)
    for split, records in records_by_split.items():
        _write_jsonl(output_dir / f"{split}.jsonl", records)
    _write_jsonl(output_dir / "selected_targets.jsonl", selected_targets)
    _write_jsonl(output_dir / "audit.jsonl", audits)
    _write_jsonl(output_dir / "excluded.jsonl", [])
    action_counts = Counter(
        step["kind"]
        for record in all_records
        for step in _action_steps(record)
    )
    split_action_counts = {
        split: sum(len(_action_steps(record)) for record in records)
        for split, records in records_by_split.items()
    }
    requested_move_distribution = Counter(
        item.audit["requested_move_count"] for item in built
    )
    actual_move_distribution = Counter(item.audit["actual_move_count"] for item in built)
    downgrade_reasons = Counter(
        str(item.audit["downgrade_reason"])
        for item in built
        if item.audit.get("downgrade_reason")
    )
    selected_resolution_counts = Counter(
        f"{item.image_group.width}x{item.image_group.height}"
        for item in selection.selected_targets
    )
    deduplicated_resolution_counts = Counter(
        f"{group.width}x{group.height}" for group in scan_result.image_groups
    )
    selected_instruction_type_counts = Counter(
        item.target.instruction_type for item in selection.selected_targets
    )
    selected_instruction_category_counts = Counter(
        item.target.category for item in selection.selected_targets
    )
    selected_software_counts = Counter(
        item.target.software for item in selection.selected_targets
    )
    target_exclusion_reasons = dict(
        scan_result.audit.get("target_exclusion_reasons", {})
    )
    cursor_image_count = sum(
        len(item.audit.get("cursor_observation_paths", [])) for item in built
    )
    pending_teacher_action_count = (
        sum(action_counts.values()) if pending_teacher_mode else 0
    )
    selected_image_count = len(selection.selected_targets)
    image_with_eligible_bbox_count = int(
        selection.audit.get("image_with_eligible_bbox_count", selected_image_count)
    )
    summary = {
        "profile": profile_name,
        "seed": seed,
        "source": "groundcua",
        "task_type": "grounding_multistep_static",
        "record_count": len(all_records),
        "split_counts": dict(sorted((split, len(records)) for split, records in records_by_split.items())),
        "split_category_counts": selection.audit["split_category_counts"],
        "trainable_action_example_count": sum(action_counts.values()),
        "split_trainable_action_example_counts": dict(sorted(split_action_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "move_count_distribution": {
            str(key): value for key, value in sorted(actual_move_distribution.items())
        },
        "requested_move_count_distribution": {
            str(key): value for key, value in sorted(requested_move_distribution.items())
        },
        "actual_move_count_distribution": {
            str(key): value for key, value in sorted(actual_move_distribution.items())
        },
        "move_count_downgrade_count": sum(downgrade_reasons.values()),
        "move_count_downgrade_reasons": dict(sorted(downgrade_reasons.items())),
        "initial_cursor_mode_distribution": dict(
            sorted(Counter(item.audit["initial_cursor_mode"] for item in built).items())
        ),
        "training_ready": False,
        "think_source": (
            "pending_397b_teacher"
            if pending_teacher_mode
            else "smoke_template_not_for_training"
        ),
        "pending_teacher_action_count": pending_teacher_action_count,
        "source_row_count": int(scan_result.audit.get("rows_total", 0)),
        "deduplicated_image_group_count": len(scan_result.image_groups),
        "image_with_eligible_bbox_count": image_with_eligible_bbox_count,
        "image_without_eligible_bbox_count": len(scan_result.image_groups)
        - image_with_eligible_bbox_count,
        "selected_image_count": selected_image_count,
        "records_per_selected_image": (
            len(all_records) / selected_image_count if selected_image_count else 0.0
        ),
        "limit_images": limit_images,
        "resolution_counts": dict(sorted(deduplicated_resolution_counts.items())),
        "selected_resolution_counts": dict(sorted(selected_resolution_counts.items())),
        "eligible_target_count": int(scan_result.audit.get("eligible_target_count", 0)),
        "eligible_bbox_count_distribution": {
            str(key): value
            for key, value in sorted(
                selection.audit.get("eligible_bbox_count_distribution", {}).items()
            )
        },
        "selected_instruction_type_counts": dict(
            sorted(selected_instruction_type_counts.items())
        ),
        "selected_instruction_category_counts": dict(
            sorted(selected_instruction_category_counts.items())
        ),
        "selected_software_counts": dict(sorted(selected_software_counts.items())),
        "target_exclusion_reasons": target_exclusion_reasons,
        "duplicate_target_annotation_count": int(
            target_exclusion_reasons.get("duplicate_target_annotation", 0)
        ),
        "source_image_count": len(unique_source_groups),
        "cursor_image_count": cursor_image_count,
        "resize_count": 0,
        "crop_count": 0,
        "letterbox_count": 0,
        "padding_count": 0,
        "stretch_count": 0,
        "source_component_count": int(
            selection.audit.get("source_component_count", 0)
        ),
        "cross_split_encoded_sha_overlap": int(
            selection.audit.get("cross_split_encoded_sha_overlap", 0)
        ),
        "cross_split_pixel_sha_overlap": int(
            selection.audit.get("cross_split_pixel_sha_overlap", 0)
        ),
        "cross_split_source_image_id_overlap": int(
            selection.audit.get("cross_split_source_image_id_overlap", 0)
        ),
        "cross_split_source_component_overlap": int(
            selection.audit.get("cross_split_source_component_overlap", 0)
        ),
        "cross_split_confirmed_near_duplicate_overlap": int(
            selection.audit.get("cross_split_confirmed_near_duplicate_overlap", 0)
        ),
        "concurrency": {
            "scan_workers": scan_workers,
            "scan_max_pending": resolved_scan_max_pending,
            "parquet_file_workers": int(parquet_file_workers),
            "parquet_prefetch_batches": int(parquet_prefetch_batches),
            "render_workers": render_workers,
            "render_max_pending": resolved_render_max_pending,
            "validation_workers": validation_workers,
            "validation_max_pending": resolved_validation_max_pending,
        },
        "resolution_policy": {
            **RESOLUTION_POLICY,
            "resolution_policy": RESOLUTION_POLICY["resolution_policy"],
            "resolution_counts": scan_result.audit.get("resolution_counts", {}),
        },
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "scan_audit": scan_result.audit,
        "selection_audit": selection.audit,
    }
    _atomic_write_text(
        output_dir / "summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    validation = _validate_export(
        output_dir=output_dir,
        records_by_split=records_by_split,
        audits=audits,
        expected_target_count=(
            int(scan_result.audit["eligible_target_count"])
            if all_targets_mode
            else selected_image_count
            if one_bbox_mode
            else len(all_records)
        ),
        expected_selected_image_count=(
            min(limit_images, image_with_eligible_bbox_count)
            if one_bbox_mode and limit_images is not None
            else image_with_eligible_bbox_count
            if one_bbox_mode
            else None
        ),
        require_unique_pixel_images=one_bbox_mode,
        expected_think_source=summary["think_source"],
        require_canonical_responses=not pending_teacher_mode,
        workers=validation_workers,
        max_pending=resolved_validation_max_pending,
    )
    if (
        not pending_teacher_mode
        and validation["loader_example_count"]
        != summary["trainable_action_example_count"]
    ):
        validation["errors"].append(
            "loader example count does not match trainable action example count"
        )
        validation["status"] = "failed"
    _atomic_write_text(
        output_dir / "validation_report.json",
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
    )
    readme = "\n".join(
        (
            (
                "# GroundCUA All-Resolution All-Targets Multistep Dataset"
                if all_targets_mode
                else "# GroundCUA One-BBox-Per-Image All-Resolution Multistep Dataset"
                if one_bbox_mode
                else "# GroundCUA All-Resolution Multistep Subset"
            ),
            "",
            f"- Profile: `{profile_name}`",
            f"- Records: `{len(all_records)}`",
            f"- Splits: `{summary['split_counts']}`",
            "- Source images are copied byte-for-byte at their real decoded resolutions; training applies the shared pixel-range transform.",
            "- Cursor observations reuse `white_arrow_black_outline_v1` from ten-choice.",
                (
                    "- Current Think labels are pending formal Qwen3.5-397B-A17B teacher generation; `training_ready=false`."
                    if pending_teacher_mode
                    else "- Current Think labels are smoke-only placeholders; `training_ready=false`."
                ),
            "- Bbox, target points, and waypoints are confined to selected-target/audit files.",
            "",
        )
    )
    _atomic_write_text(output_dir / "README.md", readme)
    pending_teacher_ok = (
        pending_teacher_mode and validation["status"] == "passed_pending_teacher"
    )
    if validation["status"] != "passed" and not pending_teacher_ok:
        raise ValueError(
            "GroundCUA subset validation failed: " + "; ".join(validation["errors"][:5])
        )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("artifacts/datasets/groundcua/groundcua_train_hf_20260711"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("smoke", "pilot", "all_targets", "one_bbox_per_image"),
        default="smoke",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--limit-images", type=int)
    default_workers = recommended_local_workers()
    parser.add_argument("--scan-workers", type=int, default=recommended_scan_workers())
    parser.add_argument("--scan-max-pending", type=int)
    parser.add_argument("--parquet-file-workers", type=int, default=2)
    parser.add_argument("--parquet-prefetch-batches", type=int, default=2)
    parser.add_argument("--render-workers", type=int, default=default_workers)
    parser.add_argument("--render-max-pending", type=int)
    parser.add_argument("--validation-workers", type=int, default=default_workers)
    parser.add_argument("--validation-max-pending", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    export_groundcua_subset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        profile_name=args.profile,
        limit_images=args.limit_images,
        scan_workers=args.scan_workers,
        scan_max_pending=args.scan_max_pending,
        parquet_file_workers=args.parquet_file_workers,
        parquet_prefetch_batches=args.parquet_prefetch_batches,
        render_workers=args.render_workers,
        render_max_pending=args.render_max_pending,
        validation_workers=args.validation_workers,
        validation_max_pending=args.validation_max_pending,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
