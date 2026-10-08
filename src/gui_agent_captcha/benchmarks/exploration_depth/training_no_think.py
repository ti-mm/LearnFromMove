"""Export the six exploration-depth traces as one mixed no-Think VERL SFT set."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ...data.verl_sft_contract import messages_to_verl_row

VARIANTS = (
    "ten_choice_third_person",
    "ten_choice_first_person",
    "rotation_inner",
    "rotation_outer",
    "drag_third_person",
    "drag_first_person",
)
SIX_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
    "click",
)
EXPECTED_TRACE_COUNT_PER_VARIANT = 4000
EXPECTED_TOTAL_ROWS = 155_454
IMAGE_HISTORY_MAX = 3
IMAGE_MAX_PIXELS = 1280 * 720
SHUFFLE_SEED = 20260826
MIX_POLICY = "six_variant_full_concatenation_global_shuffle_v1"
MODEL_COORDINATE_CONTRACT = "qwen3_relative_integer_0_1000_half_up_v1"
PROMPT_CONTRACT = (
    "action_ablation_six_action_latest3_images_all_exact_assistant_responses_"
    "integer_coordinates_v2"
)
RESPONSE_CONTRACT = "empty_think_tag_then_exact_integer_action_v2"

_ACTION_SCHEMA_LINES = {
    "move_to": '- move_to: {"kind":"move_to","x":<number 0-1000>,"y":<number 0-1000>}',
    "mouse_down": '- mouse_down: {"kind":"mouse_down"}',
    "mouse_up": '- mouse_up: {"kind":"mouse_up"}',
    "left_click": '- left_click: {"kind":"left_click"}',
    "drag": (
        '- drag: {"kind":"drag","points":[[<number 0-1000>,<number 0-1000>],'
        '[<number 0-1000>,<number 0-1000>]]}'
    ),
    "click": '- click: {"kind":"click","points":[[<number 0-1000>,<number 0-1000>]]}',
}

_INTEGER_ACTION_SCHEMA_LINES = {
    "move_to": '- move_to: {"kind":"move_to","x":<integer 0-1000>,"y":<integer 0-1000>}',
    "mouse_down": '- mouse_down: {"kind":"mouse_down"}',
    "mouse_up": '- mouse_up: {"kind":"mouse_up"}',
    "left_click": '- left_click: {"kind":"left_click"}',
    "drag": (
        '- drag: {"kind":"drag","points":[[<integer 0-1000>,<integer 0-1000>],'
        '[<integer 0-1000>,<integer 0-1000>]]}'
    ),
    "click": '- click: {"kind":"click","points":[[<integer 0-1000>,<integer 0-1000>]]}',
}


class NoThinkDatasetError(ValueError):
    """A source trace or exported row violates the no-Think contract."""


@dataclass(frozen=True)
class PreparedTrace:
    record_id: str
    instruction: str
    images: tuple[Path, ...]
    actions: tuple[dict[str, object], ...]
    exploration_level: str


@dataclass(frozen=True)
class RowReference:
    variant: str
    trace_index: int
    action_index: int
    variant_row_index: int


def default_output_root() -> Path:
    from .training_manifest import default_training_root

    return default_training_root() / "sft_no_think_six_mixed_v1"


def default_input_root() -> Path:
    from .training_manifest import default_training_root

    return default_training_root() / "raw_traces"


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise NoThinkDatasetError(f"blank JSONL row at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise NoThinkDatasetError(f"non-object JSONL row at {path}:{line_number}")
            yield value


def _number(value: object, *, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NoThinkDatasetError(f"{field} must be numeric")
    if not math.isfinite(float(value)):
        raise NoThinkDatasetError(f"{field} must be finite")
    return value


def _model_coordinate(value: object, *, field: str) -> int:
    number = float(_number(value, field=field))
    if not 0.0 <= number <= 1000.0:
        raise NoThinkDatasetError(f"{field} must be within 0-1000")
    # Coordinates are nonnegative, so floor(x + 0.5) is ordinary half-up
    # rounding rather than Python's ties-to-even round().
    return int(math.floor(number + 0.5))


def _integer_model_action(action: Mapping[str, object]) -> dict[str, object]:
    kind = action.get("kind")
    if kind not in SIX_ACTION_KINDS:
        raise NoThinkDatasetError(f"unsupported action kind {kind!r}")
    result: dict[str, object] = {"kind": str(kind)}
    if kind == "move_to":
        result["x"] = _model_coordinate(action.get("x"), field="move_to.x")
        result["y"] = _model_coordinate(action.get("y"), field="move_to.y")
    elif kind in {"click", "drag"}:
        raw_points = action.get("points")
        expected_points = 1 if kind == "click" else 2
        if not isinstance(raw_points, list) or len(raw_points) != expected_points:
            raise NoThinkDatasetError(f"{kind}.points must contain {expected_points} point(s)")
        points: list[list[int]] = []
        for point_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise NoThinkDatasetError(f"each {kind} point must contain x and y")
            points.append(
                [
                    _model_coordinate(
                        raw_point[0], field=f"{kind}.points[{point_index}].x"
                    ),
                    _model_coordinate(
                        raw_point[1], field=f"{kind}.points[{point_index}].y"
                    ),
                ]
            )
        result["points"] = points
    return result


def _action_from_step(step: Mapping[str, object]) -> dict[str, object]:
    kind = step.get("kind")
    if kind not in SIX_ACTION_KINDS:
        raise NoThinkDatasetError(f"unsupported action kind {kind!r}")
    action: dict[str, object] = {"kind": str(kind)}
    if kind == "move_to":
        action["x"] = _number(step.get("x"), field=f"{kind}.x")
        action["y"] = _number(step.get("y"), field=f"{kind}.y")
    elif kind in {"click", "drag"}:
        raw_points = step.get("points")
        expected_points = 1 if kind == "click" else 2
        if not isinstance(raw_points, list) or len(raw_points) != expected_points:
            raise NoThinkDatasetError(f"{kind}.points must contain {expected_points} point(s)")
        points: list[list[int | float]] = []
        for point_index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise NoThinkDatasetError(f"each {kind} point must contain x and y")
            points.append(
                [
                    _number(raw_point[0], field=f"drag.points[{point_index}].x"),
                    _number(raw_point[1], field=f"drag.points[{point_index}].y"),
                ]
            )
        action["points"] = points
    return action


def _prepare_trace(record: Mapping[str, object], *, source_path: Path) -> PreparedTrace:
    record_id = record.get("id")
    instruction = record.get("instruction")
    steps = record.get("steps")
    if not isinstance(record_id, str) or not record_id:
        raise NoThinkDatasetError(f"{source_path}: trace has no id")
    if not isinstance(instruction, str) or not instruction.strip():
        raise NoThinkDatasetError(f"{record_id}: trace has no instruction")
    if not isinstance(steps, list) or len(steps) < 3:
        raise NoThinkDatasetError(f"{record_id}: trace has no causal steps")

    images: list[Path] = []
    actions: list[dict[str, object]] = []
    expect_observation = True
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise NoThinkDatasetError(f"{record_id}: step {step_index} is not an object")
        if expect_observation:
            if step.get("type") != "observation":
                raise NoThinkDatasetError(f"{record_id}: expected observation at step {step_index}")
            image_path = step.get("image_path")
            if not isinstance(image_path, str) or not image_path:
                raise NoThinkDatasetError(f"{record_id}: observation has no image_path")
            image = Path(image_path).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)
            images.append(image)
        else:
            if step.get("type") != "action":
                raise NoThinkDatasetError(f"{record_id}: expected action at step {step_index}")
            actions.append(_integer_model_action(_action_from_step(step)))
        expect_observation = not expect_observation
    if expect_observation:
        raise NoThinkDatasetError(f"{record_id}: trace must end with a post-action observation")
    if len(images) != len(actions) + 1 or not actions:
        raise NoThinkDatasetError(f"{record_id}: invalid observation/action counts")
    metadata = record.get("metadata")
    exploration_level = (
        str(metadata.get("exploration_level") or "") if isinstance(metadata, dict) else ""
    )
    if exploration_level not in {"L0", "L1", "L2"}:
        raise NoThinkDatasetError(f"{record_id}: invalid exploration level")
    return PreparedTrace(
        record_id=record_id,
        instruction=instruction.strip(),
        images=tuple(images),
        actions=tuple(actions),
        exploration_level=exploration_level,
    )


def empty_think_response(action: Mapping[str, object]) -> str:
    return "<think></think>\n" + json.dumps(
        {"action": dict(action)}, ensure_ascii=False, separators=(",", ":")
    )


def _no_think_action_prompt(instruction: str) -> str:
    display = ", ".join(SIX_ACTION_KINDS)
    return "\n".join(
        (
            f"Task requirement: {instruction}",
            "Action space for this task:",
            f"Allowed action kinds: {display}",
            "Action schemas:",
            *(_INTEGER_ACTION_SCHEMA_LINES[kind] for kind in SIX_ACTION_KINDS),
            "Choose exactly one next action from the current screenshot and causal history.",
            "Coordinates must be integers in the 0-1000 screenshot-relative system: x increases left to right and y top to bottom.",
            "move_to requires x and y; click requires one point; drag requires two points; left_click, mouse_down, and mouse_up take no coordinates.",
            'Return exactly <think></think> followed by one compact JSON object: {"action":{...}}.',
            "Do not place any text inside the think tags and do not add Markdown or other text.",
        )
    )


def _no_think_action_context(action_index: int) -> str:
    history = (
        f"{action_index} previous assistant response(s) are included in chronological order; "
        "older screenshots may be omitted, but all previous empty-think action responses are retained."
        if action_index
        else "No previous actions are present in this trajectory."
    )
    return "\n".join(
        (
            history,
            "The last attached image is the current observation.",
            f"Return <think></think> followed by the JSON action for step {action_index + 1}.",
        )
    )


def _messages(trace: PreparedTrace, *, action_index: int) -> list[dict[str, object]]:
    if not 0 <= action_index < len(trace.actions):
        raise NoThinkDatasetError("action index is out of range")
    prior_responses = [empty_think_response(action) for action in trace.actions[:action_index]]
    causal_images = trace.images[: action_index + 1]
    retained_images = causal_images[-IMAGE_HISTORY_MAX:]
    first_retained_index = len(causal_images) - len(retained_images)
    content: list[dict[str, str]] = [
        {
            "type": "text",
            "text": _no_think_action_prompt(trace.instruction),
        }
    ]

    def append_prior(index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {index + 1} assistant response (context only):\n"
                    f"{prior_responses[index]}\n"
                ),
            }
        )

    for index in range(first_retained_index):
        append_prior(index)
    for retained_offset, image_path in enumerate(retained_images):
        content.append({"type": "image", "image": str(image_path)})
        prior_index = first_retained_index + retained_offset
        if prior_index < action_index:
            append_prior(prior_index)
    content.append({"type": "text", "text": _no_think_action_context(action_index)})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": empty_think_response(trace.actions[action_index])},
    ]


class _ParquetSink:
    def __init__(self, path: Path, *, batch_size: int = 512) -> None:
        self.path = path
        self.batch_size = batch_size
        self.rows: list[dict[str, object]] = []
        self.writer: Any = None
        self.schema: Any = None

    def append(self, row: dict[str, object]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is None:
            raise NoThinkDatasetError("conversion produced no rows")
        self.writer.close()
        self.writer = None


def _load_sources(
    input_root: Path,
    *,
    enforce_v1_row_count: bool = True,
) -> tuple[dict[str, list[PreparedTrace]], list[RowReference], dict[str, Counter[str]]]:
    traces: dict[str, list[PreparedTrace]] = {}
    references: list[RowReference] = []
    action_counts: dict[str, Counter[str]] = {}
    for variant in VARIANTS:
        source_path = input_root / variant / "train.jsonl"
        prepared = [_prepare_trace(record, source_path=source_path) for record in _read_jsonl(source_path)]
        if len(prepared) != EXPECTED_TRACE_COUNT_PER_VARIANT:
            raise NoThinkDatasetError(
                f"{variant}: expected {EXPECTED_TRACE_COUNT_PER_VARIANT} traces, got {len(prepared)}"
            )
        if len({trace.record_id for trace in prepared}) != len(prepared):
            raise NoThinkDatasetError(f"{variant}: duplicate trace id")
        counts: Counter[str] = Counter()
        variant_row_index = 0
        for trace_index, trace in enumerate(prepared):
            for action_index, action in enumerate(trace.actions):
                counts[str(action["kind"])] += 1
                references.append(
                    RowReference(
                        variant=variant,
                        trace_index=trace_index,
                        action_index=action_index,
                        variant_row_index=variant_row_index,
                    )
                )
                variant_row_index += 1
        traces[variant] = prepared
        action_counts[variant] = counts
    if enforce_v1_row_count and len(references) != EXPECTED_TOTAL_ROWS:
        raise NoThinkDatasetError(
            f"expected {EXPECTED_TOTAL_ROWS} action rows, got {len(references)}"
        )
    return traces, references, action_counts


def _max_consecutive_run(values: Iterable[str]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for value in values:
        if value == previous:
            current += 1
        else:
            current = 1
            previous = value
        maximum = max(maximum, current)
    return maximum


def build_no_think_dataset(
    *,
    input_root: Path,
    output_root: Path,
    shuffle_seed: int = SHUFFLE_SEED,
    enforce_v1_row_count: bool = True,
) -> dict[str, object]:
    train_path = output_root / "train.parquet"
    summary_path = output_root / "summary.json"
    ready_path = output_root / "READY.json"
    if any(path.exists() for path in (train_path, summary_path, ready_path)):
        raise FileExistsError(f"refusing to overwrite dataset under {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    traces, references, action_counts = _load_sources(
        input_root, enforce_v1_row_count=enforce_v1_row_count
    )
    random.Random(shuffle_seed).shuffle(references)
    shuffled_variants = [reference.variant for reference in references]
    temporary_train = train_path.with_name(f".{train_path.name}.tmp-{os.getpid()}")
    sink = _ParquetSink(temporary_train)
    try:
        for mixed_row_index, reference in enumerate(references):
            trace = traces[reference.variant][reference.trace_index]
            row = messages_to_verl_row(
                _messages(trace, action_index=reference.action_index),
                image_max_pixels=IMAGE_MAX_PIXELS,
                # Every source observation was already resolved and checked
                # exactly once by _prepare_trace; avoid repeated network-FS
                # realpath/stat calls for the same historical frames.
                validate_image_files=False,
                metadata={
                    "source_index": reference.variant_row_index,
                    "source_record_id": trace.record_id,
                    "action_index": reference.action_index,
                    "task": reference.variant,
                    "variant": reference.variant,
                    "exploration_level": trace.exploration_level,
                    "mixture_source": reference.variant,
                    "mixture_source_row": reference.variant_row_index,
                    "mixed_row_index": mixed_row_index,
                    "image_history_max": IMAGE_HISTORY_MAX,
                    "image_max_pixels": IMAGE_MAX_PIXELS,
                    "history_contract": "latest_3_images_all_previous_assistant_responses",
                    "prompt_contract": PROMPT_CONTRACT,
                    "response_contract": RESPONSE_CONTRACT,
                    "coordinate_contract": MODEL_COORDINATE_CONTRACT,
                    "no_think": True,
                },
            )
            sink.append(row)
        sink.close()
        os.replace(temporary_train, train_path)
    finally:
        if sink.writer is not None:
            sink.writer.close()
        temporary_train.unlink(missing_ok=True)

    source_row_counts = Counter(shuffled_variants)
    all_action_counts = sum(action_counts.values(), Counter())
    summary: dict[str, object] = {
        "schema": "exploration_depth_six_variant_no_think_sft_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(input_root.resolve()),
        "train_parquet": str(train_path.resolve()),
        "train_rows": len(references),
        "trace_count": sum(len(value) for value in traces.values()),
        "trace_count_per_variant": {
            variant: len(traces[variant]) for variant in VARIANTS
        },
        "source_row_counts": dict(sorted(source_row_counts.items())),
        "action_counts": dict(sorted(all_action_counts.items())),
        "action_counts_per_variant": {
            variant: dict(sorted(action_counts[variant].items())) for variant in VARIANTS
        },
        "mix_policy": MIX_POLICY,
        "shuffle_seed": shuffle_seed,
        "source_transitions": sum(
            left != right for left, right in zip(shuffled_variants, shuffled_variants[1:])
        ),
        "max_consecutive_source_run": _max_consecutive_run(shuffled_variants),
        "first_32_sources": shuffled_variants[:32],
        "image_history_max": IMAGE_HISTORY_MAX,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "task_requirement_policy": "single_task_requirement_v1",
        "prompt_contract": PROMPT_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "assistant_target_prefix": "<think></think>",
        "row_count_contract": (
            f"exact_v1_{EXPECTED_TOTAL_ROWS}"
            if enforce_v1_row_count
            else "derived_from_validated_source_actions"
        ),
    }
    _atomic_json(summary_path, summary)
    ready: dict[str, object] = {
        "schema": "exploration_depth_six_variant_no_think_ready_v1",
        "status": "ready",
        "train_rows": len(references),
        "trace_count": sum(len(value) for value in traces.values()),
        "source_row_counts": dict(sorted(source_row_counts.items())),
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "validation": (
            "source_contract_and_half_up_integer_action_checked_during_full_export"
        ),
        "files": {
            "train.parquet": {"size_bytes": train_path.stat().st_size},
            "summary.json": {"size_bytes": summary_path.stat().st_size},
        },
    }
    _atomic_json(ready_path, ready)
    return {"summary": summary, "ready": ready}


def materialize_no_think_seven_suite(
    *,
    mixed_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Split one validated mixed release into six single-task sets plus mixed."""

    source_train = mixed_root / "train.parquet"
    source_summary_path = mixed_root / "summary.json"
    source_ready_path = mixed_root / "READY.json"
    coordinate_audit_path = mixed_root / "coordinate_audit.json"
    for path in (
        source_train,
        source_summary_path,
        source_ready_path,
        coordinate_audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite dataset under {output_root}")

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_ready = json.loads(source_ready_path.read_text(encoding="utf-8"))
    coordinate_audit = json.loads(coordinate_audit_path.read_text(encoding="utf-8"))
    if source_ready.get("status") != "ready":
        raise NoThinkDatasetError("source no-Think dataset is not ready")
    if coordinate_audit.get("status") != "passed":
        raise NoThinkDatasetError("source coordinate audit did not pass")
    if source_summary.get("response_contract") != RESPONSE_CONTRACT:
        raise NoThinkDatasetError("source response contract mismatch")
    if source_summary.get("coordinate_contract") != MODEL_COORDINATE_CONTRACT:
        raise NoThinkDatasetError("source coordinate contract mismatch")

    expected_counts = {
        str(key): int(value)
        for key, value in dict(source_summary.get("source_row_counts") or {}).items()
    }
    if set(expected_counts) != set(VARIANTS):
        raise NoThinkDatasetError("source variant row-count set mismatch")
    trace_counts = {
        str(key): int(value)
        for key, value in dict(source_summary.get("trace_count_per_variant") or {}).items()
    }
    if set(trace_counts) != set(VARIANTS):
        raise NoThinkDatasetError("source variant trace-count set mismatch")

    import pyarrow as pa
    import pyarrow.parquet as pq

    output_root.mkdir(parents=True)
    temporary_paths = {
        variant: output_root / variant / f".train.parquet.tmp-{os.getpid()}"
        for variant in VARIANTS
    }
    final_paths = {
        variant: output_root / variant / "train.parquet" for variant in VARIANTS
    }
    writers: dict[str, pq.ParquetWriter] = {}
    written_counts: Counter[str] = Counter()
    parquet = pq.ParquetFile(source_train)
    try:
        for batch in parquet.iter_batches(batch_size=2048):
            partitioned: dict[str, list[dict[str, object]]] = {
                variant: [] for variant in VARIANTS
            }
            for row in batch.to_pylist():
                metadata = row.get("metadata")
                variant = (
                    str(metadata.get("variant"))
                    if isinstance(metadata, dict)
                    else ""
                )
                if variant not in partitioned:
                    raise NoThinkDatasetError(f"unexpected source variant {variant!r}")
                partitioned[variant].append(row)
            for variant, rows in partitioned.items():
                if not rows:
                    continue
                path = temporary_paths[variant]
                path.parent.mkdir(parents=True, exist_ok=True)
                table = pa.Table.from_pylist(rows, schema=parquet.schema_arrow)
                writer = writers.get(variant)
                if writer is None:
                    writer = pq.ParquetWriter(
                        path,
                        parquet.schema_arrow,
                        compression="zstd",
                        use_dictionary=True,
                    )
                    writers[variant] = writer
                writer.write_table(table)
                written_counts[variant] += len(rows)
        for writer in writers.values():
            writer.close()
        writers.clear()
        if dict(written_counts) != expected_counts:
            raise NoThinkDatasetError(
                f"single-task row counts mismatch: {dict(written_counts)}"
            )
        for variant in VARIANTS:
            os.replace(temporary_paths[variant], final_paths[variant])

        mixed_dir = output_root / "mixed"
        mixed_dir.mkdir(parents=True)
        mixed_temporary = mixed_dir / f".train.parquet.tmp-{os.getpid()}"
        shutil.copy2(source_train, mixed_temporary)
        os.replace(mixed_temporary, mixed_dir / "train.parquet")
    finally:
        for writer in writers.values():
            writer.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    dataset_keys = (*VARIANTS, "mixed")
    dataset_summaries: dict[str, dict[str, object]] = {}
    for key in dataset_keys:
        is_mixed = key == "mixed"
        train_rows = sum(expected_counts.values()) if is_mixed else expected_counts[key]
        trace_count = sum(trace_counts.values()) if is_mixed else trace_counts[key]
        source_row_counts = expected_counts if is_mixed else {key: expected_counts[key]}
        dataset_dir = output_root / key
        summary = {
            **source_summary,
            "schema": "exploration_depth_no_think_sft_dataset_v2",
            "dataset_key": key,
            "derived_from": str(mixed_root.resolve()),
            "train_parquet": str((dataset_dir / "train.parquet").resolve()),
            "train_rows": train_rows,
            "trace_count": trace_count,
            "source_row_counts": source_row_counts,
            "action_counts": (
                source_summary.get("action_counts", {})
                if is_mixed
                else dict(source_summary.get("action_counts_per_variant", {})).get(
                    key, {}
                )
            ),
            "mix_policy": source_summary.get("mix_policy") if is_mixed else "single_variant_v1",
        }
        _atomic_json(dataset_dir / "summary.json", summary)
        ready = {
            "schema": "exploration_depth_no_think_sft_ready_v2",
            "status": "ready",
            "dataset_key": key,
            "train_rows": train_rows,
            "trace_count": trace_count,
            "source_row_counts": source_row_counts,
            "response_contract": RESPONSE_CONTRACT,
            "coordinate_contract": MODEL_COORDINATE_CONTRACT,
            "validation": "passed",
            "files": {
                "train.parquet": {
                    "size_bytes": (dataset_dir / "train.parquet").stat().st_size
                },
                "summary.json": {
                    "size_bytes": (dataset_dir / "summary.json").stat().st_size
                },
            },
        }
        _atomic_json(dataset_dir / "READY.json", ready)
        dataset_summaries[key] = summary

    shutil.copy2(coordinate_audit_path, output_root / "coordinate_audit.json")
    top_summary = {
        "schema": "exploration_depth_no_think_seven_dataset_suite_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_from": str(mixed_root.resolve()),
        "dataset_count": 7,
        "trace_count": sum(trace_counts.values()),
        "mixed_train_rows": sum(expected_counts.values()),
        "separate_train_rows": expected_counts,
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "datasets": {
            key: {
                "train_parquet": dataset_summaries[key]["train_parquet"],
                "train_rows": dataset_summaries[key]["train_rows"],
                "trace_count": dataset_summaries[key]["trace_count"],
            }
            for key in dataset_keys
        },
    }
    _atomic_json(output_root / "summary.json", top_summary)
    top_ready = {
        "schema": "exploration_depth_no_think_seven_dataset_ready_v2",
        "status": "ready",
        "dataset_count": 7,
        "trace_count": top_summary["trace_count"],
        "mixed_train_rows": top_summary["mixed_train_rows"],
        "separate_train_rows": expected_counts,
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "all_validations_passed": True,
        "datasets": {
            key: str(output_root / key / "READY.json") for key in dataset_keys
        },
    }
    _atomic_json(output_root / "READY.json", top_ready)
    return {"summary": top_summary, "ready": top_ready}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=default_input_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument(
        "--dynamic-row-count",
        action="store_true",
        help="derive the action-row total from source traces instead of enforcing the v1 total",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_no_think_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        shuffle_seed=args.shuffle_seed,
        enforce_v1_row_count=not args.dynamic_row_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
