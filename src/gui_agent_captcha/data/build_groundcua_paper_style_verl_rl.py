"""Build the immutable 10K official GroundCUA task release for VERL RL.

The builder selects source tasks only.  It never writes into the repository or
modifies the official annotations/images.  Track-specific VERL rows are produced
by :mod:`gui_agent_captcha.data.export_groundcua_paper_style_verl`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STORAGE_ROOT = REPOSITORY_ROOT
OFFICIAL_DATA_DIR = (
    STORAGE_ROOT
    / "artifacts/datasets/groundcua/.official-annotations-full-staging-L52aX2"
    / "instruction_tuning/groundcua_data"
)
DEFAULT_CANDIDATE_JSON = OFFICIAL_DATA_DIR / "remaining_instructions.json"
DEFAULT_SFT_DIR = OFFICIAL_DATA_DIR
DEFAULT_IMAGE_ROOT = (
    STORAGE_ROOT
    / "artifacts/datasets/groundcua/_shared_images/700k_groundcua_official/images"
)
DEFAULT_OUTPUT_DIR = (
    STORAGE_ROOT
    / "artifacts/datasets/groundcua/rl/paper_style_verl_qwen3vl8b_10k_v1"
)
SFT_SOURCE_NAMES = (
    "direct_description_instructions.json",
    "direct_general_templates_instructions.json",
    "direct_icon_instructions.json",
    "direct_miscellaneous_instructions.json",
    "direct_text_instructions.json",
    "functional_instructions.json",
    "functional_instructions_extra.json",
    "spatial_data.json",
)
DEFAULT_SEED = 20_260_831
DEFAULT_COUNT = 10_000
MAX_PLATFORM_SHARE = 0.05
SCHEMA_VERSION = "groundcua_paper_style_verl_selection_v1"


def normalize_instruction(value: str) -> str:
    """Normalize identity text without changing the stored instruction."""

    if not isinstance(value, str):
        raise TypeError("instruction must be text")
    return " ".join(value.split())


def _required_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _simple_instruction(row: Mapping[str, Any]) -> str | None:
    instructions = row.get("all_instructions")
    if not isinstance(instructions, Mapping):
        return None
    return _required_text(instructions.get("simple_instruction"))


def _sft_instruction(row: Mapping[str, Any]) -> str | None:
    return _required_text(row.get("instruction"))


def identity_key(row: Mapping[str, Any], instruction: str) -> tuple[str, str, str]:
    source_id = _required_text(row.get("id"))
    platform = _required_text(row.get("platform"))
    normalized = normalize_instruction(instruction)
    if source_id is None or platform is None or not normalized:
        raise ValueError("identity requires nonempty id, platform, and instruction")
    return source_id, platform, normalized


def _json_array_items(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError("GroundCUA release construction requires ijson") from exc
    with Path(path).open("rb") as handle:
        for line, item in enumerate(ijson.items(handle, "item"), 1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line} is not a JSON object")
            yield item


def load_sft_identities(sft_dir: Path) -> set[tuple[str, str, str]]:
    identities: set[tuple[str, str, str]] = set()
    for source_name in SFT_SOURCE_NAMES:
        source_path = Path(sft_dir) / source_name
        for line, row in enumerate(_json_array_items(source_path), 1):
            instruction = _simple_instruction(row) or _sft_instruction(row)
            if instruction is None:
                raise ValueError(f"{source_path}:{line} has no instruction")
            identities.add(identity_key(row, instruction))
    return identities


def _finite_bbox(row: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw = row.get("bbox")
    if not isinstance(raw, list | tuple) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    if x1 < 0.0 or y1 < 0.0 or x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def _category(row: Mapping[str, Any]) -> str:
    raw = row.get("category")
    if not isinstance(raw, str) or not raw.strip():
        return "__unlabeled__"
    return raw.strip()


def _safe_image_path(image_root: Path, source_id: str, platform: str) -> Path:
    relative = Path(platform) / f"{source_id}.png"
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("candidate platform/id image path is unsafe")
    return Path(image_root) / relative


def _image_size(path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    cached = cache.get(path)
    if cached is not None:
        return cached
    with Image.open(path) as image:
        width, height = image.size
    if width <= 1 or height <= 1:
        raise ValueError(f"image dimensions must exceed one pixel: {path}")
    cache[path] = width, height
    return width, height


def _inside_image(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> bool:
    x1, y1, x2, y2 = bbox
    return 0.0 <= x1 < x2 <= float(width) and 0.0 <= y1 < y2 <= float(height)


def _qwen_coordinate(value: float, extent: int) -> int:
    if extent <= 0:
        raise ValueError("image extent must be positive")
    return min(1000, max(0, int(math.floor(value * 1000.0 / extent + 0.5))))


def _model_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [
        x1 * 1000.0 / width,
        y1 * 1000.0 / height,
        x2 * 1000.0 / width,
        y2 * 1000.0 / height,
    ]


def _aspect_bucket(width: int, height: int) -> str:
    ratio = width / height
    if ratio < 0.8:
        return "portrait"
    if ratio <= 1.25:
        return "square"
    if ratio <= 1.8:
        return "landscape"
    return "wide"


def _size_bucket(width: int, height: int) -> str:
    pixels = width * height
    if pixels < 640 * 480:
        return "small"
    if pixels < 1280 * 720:
        return "medium"
    if pixels < 1920 * 1080:
        return "large"
    return "xlarge"


def _area_bucket(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> str:
    x1, y1, x2, y2 = bbox
    fraction = (x2 - x1) * (y2 - y1) / (width * height)
    if fraction < 0.001:
        return "tiny"
    if fraction < 0.01:
        return "small"
    if fraction < 0.05:
        return "medium"
    return "large"


def _category_quotas(rows: Iterable[Mapping[str, Any]], count: int) -> dict[str, int]:
    capacities = Counter(_category(row) for row in rows)
    if not capacities:
        raise ValueError("candidate pool has no eligible categories")
    total = sum(capacities.values())
    quotas = {
        category: min(capacity, int(count * capacity / total))
        for category, capacity in capacities.items()
    }
    remaining = count - sum(quotas.values())
    ordered = sorted(
        capacities,
        key=lambda category: (
            -(count * capacities[category] / total - quotas[category]),
            category,
        ),
    )
    for category in ordered:
        if remaining <= 0:
            break
        if quotas[category] < capacities[category]:
            quotas[category] += 1
            remaining -= 1
    if remaining:
        raise ValueError("candidate capacities cannot satisfy requested count")
    return quotas


def _candidate_rows(
    candidate_path: Path,
    sft_identities: set[tuple[str, str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rejects: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in _json_array_items(candidate_path):
        instruction = _simple_instruction(row)
        label = _required_text(row.get("text"))
        bbox = _finite_bbox(row)
        if instruction is None:
            rejects["missing_simple_instruction"] += 1
            continue
        if "``" in instruction:
            rejects["unresolved_instruction_placeholder"] += 1
            continue
        if label is None:
            rejects["empty_target_text"] += 1
            continue
        if bbox is None:
            rejects["malformed_bbox"] += 1
            continue
        try:
            key = identity_key(row, instruction)
        except ValueError:
            rejects["invalid_identity"] += 1
            continue
        if key in sft_identities:
            rejects["sft_identity_overlap"] += 1
            continue
        if key in seen:
            rejects["duplicate_candidate_identity"] += 1
            continue
        seen.add(key)
        candidates.append(
            {
                "identity": key,
                "id": key[0],
                "platform": key[1],
                "instruction": instruction,
                "target_text": label,
                "bbox_source_xyxy": list(bbox),
                "category": _category(row),
            }
        )
    return candidates, rejects


def _validated_pool_by_category(
    candidates: list[dict[str, Any]],
    *,
    image_root: Path,
    quotas: Mapping[str, int],
    seed: int,
    rejects: Counter[str],
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_category[str(candidate["category"])].append(candidate)
    image_cache: dict[Path, tuple[int, int]] = {}
    validated: dict[str, list[dict[str, Any]]] = {}
    for category, rows in sorted(by_category.items()):
        rng.shuffle(rows)
        required = int(quotas[category])
        target_pool_size = max(required * 4, min(1000, required))
        pool: list[dict[str, Any]] = []
        for candidate in rows:
            if len(pool) >= target_pool_size:
                break
            path = _safe_image_path(image_root, str(candidate["id"]), str(candidate["platform"]))
            if not path.is_file():
                rejects["missing_image"] += 1
                continue
            try:
                width, height = _image_size(path, image_cache)
            except (OSError, ValueError):
                rejects["unreadable_image"] += 1
                continue
            bbox = tuple(float(value) for value in candidate["bbox_source_xyxy"])
            if not _inside_image(bbox, width, height):
                rejects["bbox_outside_image"] += 1
                continue
            x1, y1, x2, y2 = bbox
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            candidate = {
                **candidate,
                "image_path": str(path),
                "image_width": width,
                "image_height": height,
                "bbox_model_xyxy": _model_bbox(bbox, width, height),
                "target_model_xy": [
                    _qwen_coordinate(center[0], width),
                    _qwen_coordinate(center[1], height),
                ],
                "aspect_bucket": _aspect_bucket(width, height),
                "image_size_bucket": _size_bucket(width, height),
                "target_area_bucket": _area_bucket(bbox, width, height),
            }
            pool.append(candidate)
        if len(pool) < required:
            raise ValueError(
                f"candidate category {category!r} has {len(pool)} valid rows, needs {required}"
            )
        validated[category] = pool
    return validated


def _select_diverse_rows(
    pools: Mapping[str, list[dict[str, Any]]],
    quotas: Mapping[str, int],
    *,
    count: int,
) -> list[dict[str, Any]]:
    platform_cap = max(1, int(math.ceil(count * MAX_PLATFORM_SHARE)))
    platform_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for category in sorted(pools):
        buckets: dict[tuple[str, str, str], deque[dict[str, Any]]] = defaultdict(deque)
        for candidate in pools[category]:
            key = (
                str(candidate["aspect_bucket"]),
                str(candidate["image_size_bucket"]),
                str(candidate["target_area_bucket"]),
            )
            buckets[key].append(candidate)
        needed = int(quotas[category])
        cycle = deque(sorted(buckets))
        category_selected = 0
        stalled = 0
        while category_selected < needed and cycle:
            bucket = cycle.popleft()
            choices = buckets[bucket]
            chosen: dict[str, Any] | None = None
            for _ in range(len(choices)):
                candidate = choices.popleft()
                if platform_counts[str(candidate["platform"])] < platform_cap:
                    chosen = candidate
                    break
                choices.append(candidate)
            if choices:
                cycle.append(bucket)
            if chosen is None:
                stalled += 1
                if stalled >= len(cycle) + 1:
                    break
                continue
            stalled = 0
            platform_counts[str(chosen["platform"])] += 1
            selected.append(chosen)
            category_selected += 1
        if category_selected != needed:
            raise ValueError(
                f"platform cap {platform_cap} prevents filling category {category!r}: "
                f"{category_selected}/{needed}"
            )
    if len(selected) != count:
        raise AssertionError(f"selected {len(selected)} rows, expected {count}")
    return selected


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _audit(selected: list[dict[str, Any]], rejects: Counter[str]) -> dict[str, Any]:
    def counts(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row[key]) for row in selected).items()))

    platform_counts = Counter(str(row["platform"]) for row in selected)
    concentration = sum((value / len(selected)) ** 2 for value in platform_counts.values())
    return {
        "selected_rows": len(selected),
        "split": "train",
        "rejected": dict(sorted(rejects.items())),
        "category_counts": counts("category"),
        "platform_counts": dict(sorted(platform_counts.items())),
        "aspect_bucket_counts": counts("aspect_bucket"),
        "image_size_bucket_counts": counts("image_size_bucket"),
        "target_area_bucket_counts": counts("target_area_bucket"),
        "unique_platforms": len(platform_counts),
        "largest_platform_share": max(platform_counts.values()) / len(selected),
        "platform_herfindahl_index": concentration,
    }


def _write_release(
    output_dir: Path,
    *,
    selected: list[dict[str, Any]],
    rejects: Counter[str],
    seed: int,
    candidate_json: Path,
    image_root: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"release directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        records = [{**row, "split": "train"} for row in selected]
        audit = _audit(records, rejects)
        manifest = {
            "schema": SCHEMA_VERSION,
            "status": "accepted",
            "seed": seed,
            "count": len(records),
            "split_counts": {"train": len(records)},
            "candidate_json": str(Path(candidate_json).resolve()),
            "image_root": str(Path(image_root).resolve()),
            "identity_key": ["id", "platform", "normalized_simple_instruction"],
            "source_class_provenance": "unavailable_in_remaining_instructions",
            "stratification": [
                "category",
                "platform",
                "aspect_bucket",
                "image_size_bucket",
                "target_area_bucket",
            ],
            "prompt_profiles": {
                "directclick": "screenspot_pro_qwen3_direct_dbe00114",
                "moveto_leftclick": "screenspot_pro_qwen3vl_moveto_leftclick_v1",
                "leftclick_terminate": "screenspot_pro_qwen3vl_leftclick_terminate_v1",
            },
            "reward_mapping": {
                "invalid_format": -0.2,
                "valid_but_wrong": 0.0,
                "valid_and_correct": 1.0,
            },
            "rollouts_per_prompt": 8,
        }
        _jsonl(temporary / "records.jsonl", records)
        _jsonl(temporary / "train.jsonl", records)
        for name, payload in {
            "manifest.json": manifest,
            "sampling_audit.json": audit,
            "summary.json": {**manifest, "sampling_audit": audit},
        }.items():
            (temporary / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "README.md").write_text(
            "# GroundCUA Paper-Style VERL RL Data\n\n"
            "This immutable release is a reconstruction from local official annotations, "
            "not the private RL data used by the paper. It contains 10,000 unique source "
            "tasks, all assigned to train, and is rendered separately for the three existing "
            "Qwen3-VL SFT action contracts. Reward is limited to strict format validity and "
            "terminal correctness with scores -0.2, 0.0, and 1.0.\n",
            encoding="utf-8",
        )
        (temporary / ".selection_complete").write_text("accepted\n", encoding="ascii")
        os.replace(temporary, output_dir)
    except BaseException:
        # Keep a failed temporary release for inspection; never delete source data.
        raise
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def build_release(
    *,
    candidate_json: Path = DEFAULT_CANDIDATE_JSON,
    sft_dir: Path = DEFAULT_SFT_DIR,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_COUNT,
) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count must be positive")
    sft_identities = load_sft_identities(Path(sft_dir))
    candidates, rejects = _candidate_rows(Path(candidate_json), sft_identities)
    quotas = _category_quotas(candidates, count)
    pools = _validated_pool_by_category(
        candidates,
        image_root=Path(image_root),
        quotas=quotas,
        seed=seed,
        rejects=rejects,
    )
    selected = _select_diverse_rows(pools, quotas, count=count)
    return _write_release(
        Path(output_dir),
        selected=selected,
        rejects=rejects,
        seed=seed,
        candidate_json=Path(candidate_json),
        image_root=Path(image_root),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable GroundCUA VERL RL source data.")
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_release(
        candidate_json=args.candidate_json,
        sft_dir=args.sft_dir,
        image_root=args.image_root,
        output_dir=args.output_dir,
        seed=args.seed,
        count=args.count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
