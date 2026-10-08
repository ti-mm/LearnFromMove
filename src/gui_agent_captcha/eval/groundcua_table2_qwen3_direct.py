"""Qwen3-VL direct-left-click contracts and data adapters for GroundCUA Table 2."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..prompts.screenspot_pro_groundcua import QWEN3_DIRECT_PROFILE
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    build_qwen3vl_official_messages,
    parse_complete_tool_call,
)


@dataclass(frozen=True)
class Qwen3DirectContract:
    prompt_profile: str
    image_factor: int
    image_min_pixels: int
    image_max_pixels: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int


QWEN3_DIRECT_CONTRACT = Qwen3DirectContract(
    prompt_profile=QWEN3_DIRECT_PROFILE,
    image_factor=32,
    image_min_pixels=1024,
    image_max_pixels=99_999_999,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    max_new_tokens=10_000,
)

EXPECTED_DATASET_COUNTS = {
    "screenspot-pro": 1581,
    "screenspot-v2": 1272,
    "mmbench-gui": 3594,
    "ui-vision": 5479,
    "osworld-g": 510,
}
SUPPORTED_BENCHMARKS = tuple(EXPECTED_DATASET_COUNTS)


@dataclass(frozen=True)
class GroundingSample:
    benchmark: str
    record_id: str
    instruction: str
    image_path: Path | None
    image_bytes: bytes | None
    image_size: tuple[int, int]
    bbox: tuple[float, float, float, float] | None
    polygon: tuple[tuple[float, float], ...] | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ScoreResult:
    correctness: str
    predicted_xy: tuple[float, float] | None


@dataclass(frozen=True)
class PreparedRequest:
    sample: GroundingSample
    image: Image.Image
    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    prompt: str
    prompt_sha256: str


def render_training_prompt(processor: Any, instruction: str, image: Image.Image) -> str:
    """Render the exact Qwen3 direct prompt used by the training collator."""

    prompt = processor.apply_chat_template(
        build_qwen3vl_official_messages(instruction, image),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.endswith("<|im_start|>assistant\n"):
        raise ValueError("Qwen3 processor did not produce an empty assistant generation turn")
    return prompt


def parse_complete_left_click(response: str) -> tuple[int, int]:
    """Accept exactly the complete integer left-click response trained by the checkpoint."""

    if not isinstance(response, str) or re.fullmatch(
        r"\s*<tool_call>\s*.*?\s*</tool_call>\s*",
        response,
        flags=re.DOTALL,
    ) is None:
        raise ValueError("response must be exactly one complete Tool Call")
    return parse_complete_tool_call(response)


def load_image(sample: GroundingSample) -> Image.Image:
    if sample.image_path is not None:
        with Image.open(sample.image_path) as image:
            return image.convert("RGB")
    if sample.image_bytes is not None:
        with Image.open(BytesIO(sample.image_bytes)) as image:
            return image.convert("RGB")
    raise ValueError(f"sample has no image source: {sample.record_id}")


@lru_cache(maxsize=1)
def _load_qwen3_resize() -> Any:
    from ..train.qwen3_vl_sft import resize_image_for_qwen3vl_official

    return resize_image_for_qwen3vl_official


def resize_for_inference(image: Image.Image) -> Image.Image:
    """Apply the Qwen3 official factor-32 smart resize frozen by training."""

    resized, _transform = _load_qwen3_resize()(
        image,
        image_min_pixels=QWEN3_DIRECT_CONTRACT.image_min_pixels,
        image_max_pixels=QWEN3_DIRECT_CONTRACT.image_max_pixels,
    )
    return resized


def _prepare_request(sample: GroundingSample, *, processor: Any) -> PreparedRequest:
    image = load_image(sample)
    if image.size != sample.image_size:
        raise ValueError(
            f"{sample.record_id} declared image size {sample.image_size} does not match {image.size}"
        )
    resized = resize_for_inference(image)
    prompt = render_training_prompt(processor, sample.instruction, resized)
    return PreparedRequest(
        sample=sample,
        image=resized,
        original_size=image.size,
        resized_size=resized.size,
        prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def prepare_requests(
    samples: Iterable[GroundingSample],
    processor: Any,
    *,
    workers: int = 1,
) -> list[PreparedRequest]:
    """Load, resize, and render requests while preserving their input order."""

    if workers < 1:
        raise ValueError("workers must be at least 1")
    sample_rows = list(samples)
    prepare_one = partial(_prepare_request, processor=processor)
    if workers == 1:
        prepared = [prepare_one(sample) for sample in sample_rows]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(prepare_one, sample_rows))
    if not prepared:
        raise ValueError("cannot prepare an empty evaluation")
    return prepared


def _point_on_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    previous = polygon[-1]
    px, py = point
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        x1, y1 = previous
        x2, y2 = current
        intersects = (y1 > py) != (y2 > py)
        if intersects:
            crossing_x = (x2 - x1) * (py - y1) / (y2 - y1) + x1
            if px < crossing_x:
                inside = not inside
        previous = current
    return inside


def score_prediction(
    sample: GroundingSample,
    coordinate: tuple[int, int] | None,
) -> ScoreResult:
    if coordinate is None:
        return ScoreResult("wrong_format", None)
    x, y = coordinate
    if not all(isinstance(value, int) and 0 <= value <= 1000 for value in coordinate):
        return ScoreResult("wrong_format", None)
    width, height = sample.image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"sample has nonpositive image size: {sample.record_id}")
    point = (x * width / 1000.0, y * height / 1000.0)
    if sample.bbox is not None:
        x1, y1, x2, y2 = sample.bbox
        correct = x1 <= point[0] <= x2 and y1 <= point[1] <= y2
    elif sample.polygon is not None:
        correct = point_in_polygon(point, sample.polygon)
    else:
        raise ValueError(f"sample has no target region: {sample.record_id}")
    return ScoreResult("correct" if correct else "wrong", point)


def evaluate_responses(
    prepared: Iterable[PreparedRequest],
    responses: Iterable[str],
) -> list[dict[str, Any]]:
    prepared_rows = list(prepared)
    response_rows = list(responses)
    if len(prepared_rows) != len(response_rows):
        raise ValueError(f"response count mismatch: {len(response_rows)} != {len(prepared_rows)}")
    rows: list[dict[str, Any]] = []
    for request, response in zip(prepared_rows, response_rows, strict=True):
        try:
            coordinate = parse_complete_left_click(response)
            parse_error = None
        except ValueError as error:
            coordinate = None
            parse_error = str(error)
        score = score_prediction(request.sample, coordinate)
        rows.append(
            {
                "benchmark": request.sample.benchmark,
                "record_id": request.sample.record_id,
                "instruction": request.sample.instruction,
                "original_size": list(request.original_size),
                "resized_size": list(request.resized_size),
                "prompt_sha256": request.prompt_sha256,
                "raw_response": response,
                "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                "coordinate_1000": list(coordinate) if coordinate is not None else None,
                "predicted_xy": list(score.predicted_xy) if score.predicted_xy is not None else None,
                "parse_error": parse_error,
                "correctness": score.correctness,
                "target_bbox": list(request.sample.bbox) if request.sample.bbox is not None else None,
                "target_polygon": (
                    [list(point) for point in request.sample.polygon]
                    if request.sample.polygon is not None
                    else None
                ),
                "metadata": request.sample.metadata,
            }
        )
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _contract_payload() -> dict[str, Any]:
    return {
        "prompt_profile": QWEN3_DIRECT_CONTRACT.prompt_profile,
        "prompt_template_sha256": "14b61d6a5b38425d329b71c7ce505392bc89e69f63d78fa42c34904e0d34913a",
        "coordinate_space": "qwen3_relative_0_1000",
        "image_factor": QWEN3_DIRECT_CONTRACT.image_factor,
        "image_min_pixels": QWEN3_DIRECT_CONTRACT.image_min_pixels,
        "image_max_pixels": QWEN3_DIRECT_CONTRACT.image_max_pixels,
        "generation_contract": "official_transformers_parity_v1",
        "do_sample": QWEN3_DIRECT_CONTRACT.do_sample,
        "temperature": QWEN3_DIRECT_CONTRACT.temperature,
        "top_p": QWEN3_DIRECT_CONTRACT.top_p,
        "top_k": QWEN3_DIRECT_CONTRACT.top_k,
        "max_new_tokens": QWEN3_DIRECT_CONTRACT.max_new_tokens,
    }


def sampling_kwargs() -> dict[str, float | int]:
    """Return the greedy vLLM equivalent of the official Transformers evaluator."""

    return {
        "temperature": QWEN3_DIRECT_CONTRACT.temperature,
        "top_p": QWEN3_DIRECT_CONTRACT.top_p,
        "top_k": QWEN3_DIRECT_CONTRACT.top_k,
        "max_tokens": QWEN3_DIRECT_CONTRACT.max_new_tokens,
    }


def _prompt_manifest(benchmark: str, prepared: list[PreparedRequest]) -> dict[str, Any]:
    return {
        "schema": "groundcua_table2_qwen3_direct_prompt_manifest_v1",
        "benchmark": benchmark,
        "sample_count": len(prepared),
        "records": [
            {
                "record_id": request.sample.record_id,
                "original_size": list(request.original_size),
                "resized_size": list(request.resized_size),
                "prompt_sha256": request.prompt_sha256,
            }
            for request in prepared
        ],
    }


def _protocol_audit(benchmark: str, checkpoint: Path) -> dict[str, Any]:
    return {
        "schema": "groundcua_table2_qwen3_direct_protocol_audit_v1",
        "benchmark": benchmark,
        "checkpoint": str(checkpoint),
        "training_contract": _contract_payload(),
        "prompt_renderer": "checkpoint_apply_chat_template",
        "assistant_prefill": False,
        "parser": "exactly_one_complete_integer_computer_use_left_click",
        "metric": "original_image_point_in_bbox_or_polygon",
    }


def write_preflight_artifacts(
    output_dir: Path,
    *,
    benchmark: str,
    checkpoint: Path,
    prepared: Iterable[PreparedRequest],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = list(prepared)
    audit = _protocol_audit(benchmark, checkpoint)
    audit["status"] = "preflight_passed"
    audit["sample_count"] = len(prepared_rows)
    _write_json(output_dir / "prompt_manifest.json", _prompt_manifest(benchmark, prepared_rows))
    _write_json(output_dir / "protocol_audit.json", audit)
    return audit


def write_evaluation_artifacts(
    output_dir: Path,
    *,
    benchmark: str,
    checkpoint: Path,
    prepared: Iterable[PreparedRequest],
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_rows = list(prepared)
    result_rows = list(rows)
    if len(prepared_rows) != len(result_rows):
        raise ValueError("prepared request and result row counts differ")
    predictions = output_dir / "predictions.jsonl"
    with predictions.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    prompt_manifest = _prompt_manifest(benchmark, prepared_rows)
    audit = _protocol_audit(benchmark, checkpoint)
    total = len(result_rows)
    correct = sum(row["correctness"] == "correct" for row in result_rows)
    parsed = sum(row["correctness"] != "wrong_format" for row in result_rows)
    summary = {
        "schema": "groundcua_table2_qwen3_direct_summary_v1",
        "benchmark": benchmark,
        "checkpoint": str(checkpoint),
        "total": total,
        "correct": correct,
        "wrong": sum(row["correctness"] == "wrong" for row in result_rows),
        "wrong_format": total - parsed,
        "accuracy": correct / total if total else 0.0,
        "parse_rate": parsed / total if total else 0.0,
        "predictions": str(predictions),
        "prompt_manifest": str(output_dir / "prompt_manifest.json"),
    }
    _write_json(output_dir / "prompt_manifest.json", prompt_manifest)
    _write_json(output_dir / "protocol_audit.json", audit)
    _write_json(output_dir / "summary.json", summary)
    return summary


def default_data_root() -> Path:
    """Resolve the governed external location containing the five downloaded datasets."""

    from ..integrations.storage import storage_path

    return storage_path("data")


def validate_checkpoint_training_contract(checkpoint: Path) -> None:
    """Reject checkpoints incompatible with the pinned Qwen3 direct prompt contract."""

    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint config missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("model_type") != "qwen3_vl":
        raise ValueError(f"checkpoint must be qwen3_vl, got {payload.get('model_type')!r}")
    generation_path = checkpoint / "generation_config.json"
    if not generation_path.is_file():
        raise FileNotFoundError(f"checkpoint generation config missing: {generation_path}")
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    if not isinstance(generation, dict):
        raise ValueError(f"checkpoint generation config must be an object: {generation_path}")


def load_checkpoint_processor(checkpoint: Path) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        local_files_only=True,
    )


def generate_vllm_responses(
    checkpoint: Path,
    prepared: Iterable[PreparedRequest],
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> list[str]:
    """Run the fixed Qwen3 direct contract through vLLM after preflight succeeds."""

    # vLLM's Engine Core touches CUDA in a child process; fork would re-init CUDA.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    requests = list(prepared)
    llm = LLM(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        disable_custom_all_reduce=True,
        trust_remote_code=True,
        seed=0,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": QWEN3_DIRECT_CONTRACT.image_min_pixels,
            "max_pixels": QWEN3_DIRECT_CONTRACT.image_max_pixels,
        },
    )
    outputs = llm.generate(
        [
            {"prompt": request.prompt, "multi_modal_data": {"image": request.image}}
            for request in requests
        ],
        sampling_params=SamplingParams(**sampling_kwargs()),
        use_tqdm=True,
    )
    if len(outputs) != len(requests):
        raise RuntimeError(f"vLLM output count mismatch: {len(outputs)} != {len(requests)}")
    responses: list[str] = []
    for index, output in enumerate(outputs):
        if not output.outputs:
            raise RuntimeError(f"vLLM returned no completion for request {index}")
        responses.append(str(output.outputs[0].text).strip())
    return responses


def run_evaluation(
    *,
    benchmark: str,
    checkpoint: Path,
    output_dir: Path,
    prompt_profile: str = QWEN3_DIRECT_CONTRACT.prompt_profile,
    data_root: Path | None = None,
    sample_limit: int | None = None,
    preflight_only: bool = False,
    tensor_parallel_size: int = 1,
    preparation_workers: int = 1,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.8,
) -> dict[str, Any]:
    if prompt_profile != QWEN3_DIRECT_CONTRACT.prompt_profile:
        raise ValueError(
            "Qwen3 direct evaluation supports only its trained direct-click prompt profile"
        )
    checkpoint = checkpoint.resolve()
    validate_checkpoint_training_contract(checkpoint)
    resolved_data_root = default_data_root() if data_root is None else data_root.resolve()
    samples = load_benchmark_samples(benchmark, resolved_data_root, sample_limit=sample_limit)
    processor = load_checkpoint_processor(checkpoint)
    prepared = prepare_requests(samples, processor, workers=preparation_workers)
    if preflight_only:
        return write_preflight_artifacts(
            output_dir,
            benchmark=benchmark,
            checkpoint=checkpoint,
            prepared=prepared,
        )
    responses = generate_vllm_responses(
        checkpoint,
        prepared,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    return write_evaluation_artifacts(
        output_dir,
        benchmark=benchmark,
        checkpoint=checkpoint,
        prepared=prepared,
        rows=evaluate_responses(prepared, responses),
    )


def run_all_evaluations(
    *,
    checkpoint: Path,
    output_root: Path,
    prompt_profile: str = QWEN3_DIRECT_CONTRACT.prompt_profile,
    data_root: Path | None = None,
    benchmarks: Iterable[str] = SUPPORTED_BENCHMARKS,
    sample_limit: int | None = None,
    preflight_only: bool = False,
    tensor_parallel_size: int = 1,
    preparation_workers: int = 1,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.8,
) -> dict[str, dict[str, Any]]:
    """Generate all selected Table 2 requests in one vLLM scheduling queue."""

    if prompt_profile != QWEN3_DIRECT_CONTRACT.prompt_profile:
        raise ValueError(
            "Qwen3 direct evaluation supports only its trained direct-click prompt profile"
        )
    selected = tuple(benchmarks)
    if not selected:
        raise ValueError("at least one benchmark is required")
    if len(set(selected)) != len(selected):
        raise ValueError("benchmarks must not contain duplicates")
    unsupported = set(selected).difference(SUPPORTED_BENCHMARKS)
    if unsupported:
        raise ValueError(f"unsupported benchmarks: {sorted(unsupported)}")

    checkpoint = checkpoint.resolve()
    validate_checkpoint_training_contract(checkpoint)
    resolved_data_root = default_data_root() if data_root is None else data_root.resolve()
    processor = load_checkpoint_processor(checkpoint)
    all_samples = [
        sample
        for benchmark in selected
        for sample in load_benchmark_samples(
            benchmark, resolved_data_root, sample_limit=sample_limit
        )
    ]
    all_prepared = prepare_requests(all_samples, processor, workers=preparation_workers)
    prepared_by_benchmark = {benchmark: [] for benchmark in selected}
    for request in all_prepared:
        prepared_by_benchmark[request.sample.benchmark].append(request)
    if preflight_only:
        return {
            benchmark: write_preflight_artifacts(
                output_root / benchmark,
                benchmark=benchmark,
                checkpoint=checkpoint,
                prepared=prepared,
            )
            for benchmark, prepared in prepared_by_benchmark.items()
        }

    responses = generate_vllm_responses(
        checkpoint,
        all_prepared,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    summaries: dict[str, dict[str, Any]] = {}
    offset = 0
    for benchmark in selected:
        prepared = prepared_by_benchmark[benchmark]
        next_offset = offset + len(prepared)
        benchmark_responses = responses[offset:next_offset]
        summaries[benchmark] = write_evaluation_artifacts(
            output_root / benchmark,
            benchmark=benchmark,
            checkpoint=checkpoint,
            prepared=prepared,
            rows=evaluate_responses(prepared, benchmark_responses),
        )
        offset = next_offset
    if offset != len(responses):
        raise RuntimeError(f"response partition mismatch: {offset} != {len(responses)}")
    return summaries


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _require_image(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _as_size(value: Any, *, context: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must contain [width, height]")
    width, height = (int(value[0]), int(value[1]))
    if width <= 0 or height <= 0:
        raise ValueError(f"{context} dimensions must be positive")
    return width, height


def _as_bbox(value: Any, *, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{context} must contain four coordinates")
    bbox = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in bbox):
        raise ValueError(f"{context} coordinates must be finite")
    return bbox  # type: ignore[return-value]


def _sample(
    benchmark: str,
    record_id: str,
    instruction: Any,
    image_path: Path,
    image_size: tuple[int, int],
    bbox: tuple[float, float, float, float] | None,
    metadata: dict[str, Any],
) -> GroundingSample:
    if not isinstance(instruction, str) or not instruction:
        raise ValueError(f"{benchmark}:{record_id} has an empty instruction")
    return GroundingSample(
        benchmark=benchmark,
        record_id=record_id,
        instruction=instruction,
        image_path=_require_image(image_path),
        image_bytes=None,
        image_size=image_size,
        bbox=bbox,
        polygon=None,
        metadata=metadata,
    )


def _load_screenspot_pro(root: Path) -> list[GroundingSample]:
    annotations = root / "ScreenSpot-Pro" / "annotations"
    images = root / "ScreenSpot-Pro" / "images"
    samples: list[GroundingSample] = []
    for annotation_path in sorted(annotations.glob("*.json")):
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"ScreenSpot-Pro annotation is not a list: {annotation_path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"ScreenSpot-Pro annotation is not an object: {annotation_path}:{index}")
            record_id = str(row.get("id") or f"{annotation_path.stem}:{index}")
            samples.append(
                _sample(
                    "screenspot-pro",
                    record_id,
                    row.get("instruction"),
                    images / str(row["img_filename"]),
                    _as_size(row["img_size"], context=record_id),
                    _as_bbox(row["bbox"], context=record_id),
                    {"annotation_file": annotation_path.name, **row},
                )
            )
    return samples


def _load_screenspot_v2(root: Path) -> list[GroundingSample]:
    dataset = root / "ScreenSpot-v2"
    images = dataset / "screenspotv2_image"
    samples: list[GroundingSample] = []
    for annotation_path in sorted(dataset.glob("screenspot_*_v2.json")):
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"ScreenSpot-v2 annotation is not a list: {annotation_path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"ScreenSpot-v2 annotation is not an object: {annotation_path}:{index}")
            record_id = f"{annotation_path.stem}:{index}"
            x, y, width, height = _as_bbox(row["bbox"], context=record_id)
            image_path = _require_image(images / str(row["img_filename"]))
            samples.append(
                _sample(
                    "screenspot-v2",
                    record_id,
                    row.get("instruction"),
                    image_path,
                    _image_size(image_path),
                    (x, y, x + width, y + height),
                    {"annotation_file": annotation_path.name, **row},
                )
            )
    return samples


def _load_mmbench_gui(root: Path) -> list[GroundingSample]:
    dataset = root / "mmbench-gui"
    rows = json.loads((dataset / "L2_annotations.json").read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("MMBench-GUI L2 annotations are not a list")
    images = dataset / "MMBench-GUI-OfflineImages" / "offline_images"
    samples: list[GroundingSample] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"MMBench-GUI annotation is not an object: {index}")
        record_id = str(row.get("index", index))
        image_size = _as_size(row["image_size"], context=record_id)
        x1, y1, x2, y2 = _as_bbox(row["bbox"], context=record_id)
        image_name = Path(str(row["image_path"]))
        candidate = images / str(row["platform"]) / image_name
        image_path = candidate if candidate.is_file() else images / image_name
        samples.append(
            _sample(
                "mmbench-gui",
                record_id,
                row.get("instruction"),
                image_path,
                image_size,
                (x1 * image_size[0], y1 * image_size[1], x2 * image_size[0], y2 * image_size[1]),
                row,
            )
        )
    return samples


def _load_ui_vision(root: Path) -> list[GroundingSample]:
    dataset = root / "ui-vision"
    annotations = dataset / "annotations" / "element_grounding"
    samples: list[GroundingSample] = []
    for annotation_path in sorted(annotations.glob("element_grounding_*.json")):
        rows = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"UI-Vision annotation is not a list: {annotation_path}")
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"UI-Vision annotation is not an object: {annotation_path}:{index}")
            record_id = f"{annotation_path.stem}:{index}"
            samples.append(
                _sample(
                    "ui-vision",
                    record_id,
                    row.get("prompt_to_evaluate"),
                    dataset / "images" / str(row["image_path"]),
                    _as_size(row["image_size"], context=record_id),
                    _as_bbox(row["bbox"], context=record_id),
                    {"annotation_file": annotation_path.name, **row},
                )
            )
    return samples


def _as_polygon(value: Any, *, context: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} polygon is not a sequence")
    if value and isinstance(value[0], (list, tuple)):
        pairs = value
    else:
        if len(value) % 2:
            raise ValueError(f"{context} flattened polygon has odd coordinate count")
        pairs = list(zip(value[::2], value[1::2]))
    polygon = tuple((float(x), float(y)) for x, y in pairs)
    if len(polygon) < 3 or not all(math.isfinite(value) for pair in polygon for value in pair):
        raise ValueError(f"{context} polygon is invalid")
    return polygon


def adapt_osworld_g_row(row: dict[str, Any]) -> GroundingSample:
    record_id = str(row.get("id") or "unknown")
    image_size = _as_size(row["image_size"], context=record_id)
    image_data = row.get("image")
    if not isinstance(image_data, dict) or not isinstance(image_data.get("bytes"), bytes):
        raise ValueError(f"{record_id} has no embedded OSWorld-G image bytes")
    box_type = str(row.get("box_type"))
    if box_type == "bbox":
        x, y, width, height = _as_bbox(row["box_coordinates"], context=record_id)
        bbox = (x, y, x + width, y + height)
        polygon = None
    elif box_type == "polygon":
        bbox = None
        coordinates = row["box_coordinates"]
        if (
            isinstance(coordinates, (list, tuple))
            and len(coordinates) == 4
            and not isinstance(coordinates[0], (list, tuple))
        ):
            x1, y1, x2, y2 = _as_bbox(coordinates, context=record_id)
            polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
        else:
            polygon = _as_polygon(coordinates, context=record_id)
    else:
        raise ValueError(f"unsupported OSWorld-G box type: {box_type!r}")
    return GroundingSample(
        benchmark="osworld-g",
        record_id=record_id,
        instruction=str(row["instruction"]),
        image_path=None,
        image_bytes=image_data["bytes"],
        image_size=image_size,
        bbox=bbox,
        polygon=polygon,
        metadata={key: value for key, value in row.items() if key != "image"},
    )


def _load_osworld_g(root: Path) -> list[GroundingSample]:
    import pyarrow.parquet as pq

    table = pq.read_table(root / "osworld-g" / "data" / "test-00000-of-00001.parquet")
    return [adapt_osworld_g_row(row) for row in table.to_pylist()]


def _validate_count(benchmark: str, samples: Iterable[GroundingSample]) -> list[GroundingSample]:
    loaded = list(samples)
    expected = EXPECTED_DATASET_COUNTS[benchmark]
    if len(loaded) != expected:
        raise RuntimeError(f"{benchmark} count mismatch: {len(loaded)} != {expected}")
    return loaded


def load_benchmark_samples(
    benchmark: str,
    data_root: Path,
    *,
    sample_limit: int | None = None,
) -> list[GroundingSample]:
    if benchmark not in EXPECTED_DATASET_COUNTS:
        raise ValueError(f"unsupported benchmark: {benchmark}; choose from {SUPPORTED_BENCHMARKS}")
    loaders = {
        "screenspot-pro": _load_screenspot_pro,
        "screenspot-v2": _load_screenspot_v2,
        "mmbench-gui": _load_mmbench_gui,
        "ui-vision": _load_ui_vision,
        "osworld-g": _load_osworld_g,
    }
    samples = loaders[benchmark](data_root)
    if sample_limit is None:
        return _validate_count(benchmark, samples)
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")
    if not samples:
        raise RuntimeError(f"{benchmark} loaded no samples")
    return samples[:sample_limit]
