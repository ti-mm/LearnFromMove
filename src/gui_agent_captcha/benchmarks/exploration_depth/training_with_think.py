"""Export six separate and one mixed exploration-depth Think SFT datasets."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator, Mapping

from ...data.verl_sft_contract import messages_to_verl_row
from ...protocol_tracks import build_canonical_assistant_response, split_think_json_response
from .training_no_think import (
    IMAGE_HISTORY_MAX,
    IMAGE_MAX_PIXELS,
    MODEL_COORDINATE_CONTRACT,
    SHUFFLE_SEED,
    SIX_ACTION_KINDS,
    VARIANTS,
    _INTEGER_ACTION_SCHEMA_LINES,
    _ParquetSink,
    _action_from_step,
    _atomic_json,
    _integer_model_action,
    _max_consecutive_run,
    _read_jsonl,
)

EXPECTED_TRACE_COUNT_PER_VARIANT = 4000
EXPECTED_ROWS_PER_VARIANT = {
    "ten_choice_third_person": 26_264,
    "ten_choice_first_person": 31_142,
    "rotation_inner": 28_798,
    "rotation_outer": 28_798,
    "drag_third_person": 16_000,
    "drag_first_person": 24_452,
}
EXPECTED_TOTAL_ROWS = 155_454
MIXED_DATASET_KEY = "mixed"
PROMPT_CONTRACT = (
    "exploration_depth_six_action_latest3_images_all_exact_think_action_responses_"
    "integer_coordinates_v2"
)
RESPONSE_CONTRACT = "concise_visible_think_tag_then_exact_integer_action_v2"
MIX_POLICY = "six_variant_full_concatenation_global_shuffle_v1"


class ThinkDatasetError(ValueError):
    """A teacher trace or exported row violates the Think SFT contract."""


@dataclass(frozen=True)
class PreparedThinkTrace:
    record_id: str
    instruction: str
    images: tuple[Path, ...]
    actions: tuple[dict[str, object], ...]
    responses: tuple[str, ...]
    response_metadata: tuple[dict[str, object], ...]
    exploration_level: str


@dataclass(frozen=True)
class RowReference:
    variant: str
    trace_index: int
    action_index: int
    variant_row_index: int


def default_teacher_root() -> Path:
    from .training_manifest import default_training_root

    return (
        default_training_root()
        / "teacher_thinks_v9_reasoning_tag_c128_full_m20260826154457_resume"
    )


def default_raw_root() -> Path:
    from .training_manifest import default_training_root

    return default_training_root() / "raw_traces"


def default_output_root() -> Path:
    from .training_manifest import default_training_root

    return default_training_root() / "sft_with_think_seven_v1"


def _normalized_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ThinkDatasetError(f"{field} must be a nonempty path")
    return Path(os.path.abspath(os.path.expanduser(value)))


def _exact_json_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        assert isinstance(right, dict)
        return set(left) == set(right) and all(
            _exact_json_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        assert isinstance(right, list)
        return len(left) == len(right) and all(
            _exact_json_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _validate_response(
    response: object,
    source_action: Mapping[str, object],
    *,
    output_action: Mapping[str, object] | None = None,
) -> str:
    if not isinstance(response, str) or not response.strip():
        raise ThinkDatasetError("teacher action has no assistant_response")
    if response != response.strip():
        raise ThinkDatasetError("assistant_response has surrounding whitespace")
    thought, remainder = split_think_json_response(response)
    if thought is None or not thought.strip():
        raise ThinkDatasetError("assistant_response has no nonempty Think block")
    try:
        payload = json.loads(remainder)
    except json.JSONDecodeError as exc:
        raise ThinkDatasetError("assistant_response action is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"action"}:
        raise ThinkDatasetError("assistant_response must contain only an action wrapper")
    if not _exact_json_equal(payload["action"], dict(source_action)):
        raise ThinkDatasetError("assistant_response action differs from source action")
    return build_canonical_assistant_response(
        thought=thought,
        action=dict(source_action if output_action is None else output_action),
    )


def _steps(record: Mapping[str, object], *, record_id: str) -> list[dict[str, object]]:
    value = record.get("steps")
    if not isinstance(value, list) or len(value) < 3:
        raise ThinkDatasetError(f"{record_id}: trace has no causal steps")
    if any(not isinstance(step, dict) for step in value):
        raise ThinkDatasetError(f"{record_id}: trace contains a non-object step")
    return value  # type: ignore[return-value]


def _prepare_trace_pair(
    teacher_record: Mapping[str, object],
    raw_record: Mapping[str, object],
    *,
    variant: str,
) -> PreparedThinkTrace:
    record_id = teacher_record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ThinkDatasetError(f"{variant}: teacher trace has no id")
    if raw_record.get("id") != record_id:
        raise ThinkDatasetError(f"{variant}: teacher/raw trace order or id differs")
    instruction = teacher_record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ThinkDatasetError(f"{record_id}: trace has no instruction")
    if raw_record.get("instruction") != instruction:
        raise ThinkDatasetError(f"{record_id}: teacher/raw instruction differs")

    teacher_steps = _steps(teacher_record, record_id=record_id)
    raw_steps = _steps(raw_record, record_id=record_id)
    if len(teacher_steps) != len(raw_steps):
        raise ThinkDatasetError(f"{record_id}: teacher/raw step counts differ")
    images: list[Path] = []
    actions: list[dict[str, object]] = []
    responses: list[str] = []
    response_metadata: list[dict[str, object]] = []
    for step_index, (teacher_step, raw_step) in enumerate(
        zip(teacher_steps, raw_steps)
    ):
        expected_type = "observation" if step_index % 2 == 0 else "action"
        if teacher_step.get("type") != expected_type or raw_step.get("type") != expected_type:
            raise ThinkDatasetError(
                f"{record_id}: expected {expected_type} at step {step_index}"
            )
        if expected_type == "observation":
            teacher_image = _normalized_path(
                teacher_step.get("image_path"), field=f"{record_id}.image_path"
            )
            raw_image = _normalized_path(
                raw_step.get("image_path"), field=f"{record_id}.raw.image_path"
            )
            if teacher_image != raw_image:
                raise ThinkDatasetError(
                    f"{record_id}: teacher/raw observation path differs at {step_index}"
                )
            images.append(teacher_image)
            continue
        teacher_source_action = _action_from_step(teacher_step)
        raw_source_action = _action_from_step(raw_step)
        if not _exact_json_equal(teacher_source_action, raw_source_action):
            raise ThinkDatasetError(
                f"{record_id}: teacher/raw action differs at step {step_index}"
            )
        model_action = _integer_model_action(teacher_source_action)
        metadata = teacher_step.get("teacher_think_metadata")
        if not isinstance(metadata, dict) or metadata.get("reasoning_present") is not True:
            raise ThinkDatasetError(
                f"{record_id}: action {len(actions)} lacks reasoning provenance"
            )
        actions.append(model_action)
        responses.append(
            _validate_response(
                teacher_step.get("assistant_response"),
                teacher_source_action,
                output_action=model_action,
            )
        )
        response_metadata.append(dict(metadata))

    if len(images) != len(actions) + 1 or not actions:
        raise ThinkDatasetError(f"{record_id}: invalid observation/action counts")
    metadata = teacher_record.get("metadata")
    exploration_level = (
        str(metadata.get("exploration_level") or "") if isinstance(metadata, dict) else ""
    )
    if exploration_level not in {"L0", "L1", "L2"}:
        raise ThinkDatasetError(f"{record_id}: invalid exploration level")
    return PreparedThinkTrace(
        record_id=record_id,
        instruction=instruction.strip(),
        images=tuple(images),
        actions=tuple(actions),
        responses=tuple(responses),
        response_metadata=tuple(response_metadata),
        exploration_level=exploration_level,
    )


def _paired_records(
    teacher_path: Path, raw_path: Path
) -> Iterator[tuple[dict[str, object], dict[str, object]]]:
    sentinel = object()
    for teacher_record, raw_record in zip_longest(
        _read_jsonl(teacher_path), _read_jsonl(raw_path), fillvalue=sentinel
    ):
        if teacher_record is sentinel or raw_record is sentinel:
            raise ThinkDatasetError(f"teacher/raw trace counts differ: {teacher_path}")
        assert isinstance(teacher_record, dict) and isinstance(raw_record, dict)
        yield teacher_record, raw_record


def _load_sources(
    teacher_root: Path,
    raw_root: Path,
    *,
    enforce_v1_row_counts: bool = True,
) -> tuple[
    dict[str, list[PreparedThinkTrace]],
    list[RowReference],
    dict[str, Counter[str]],
]:
    traces: dict[str, list[PreparedThinkTrace]] = {}
    references: list[RowReference] = []
    action_counts: dict[str, Counter[str]] = {}
    for variant in VARIANTS:
        teacher_path = teacher_root / variant / "train.jsonl"
        raw_path = raw_root / variant / "train.jsonl"
        prepared = [
            _prepare_trace_pair(teacher, raw, variant=variant)
            for teacher, raw in _paired_records(teacher_path, raw_path)
        ]
        if len(prepared) != EXPECTED_TRACE_COUNT_PER_VARIANT:
            raise ThinkDatasetError(
                f"{variant}: expected {EXPECTED_TRACE_COUNT_PER_VARIANT} traces, "
                f"got {len(prepared)}"
            )
        if len({trace.record_id for trace in prepared}) != len(prepared):
            raise ThinkDatasetError(f"{variant}: duplicate trace id")
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
        expected_rows = EXPECTED_ROWS_PER_VARIANT[variant]
        if enforce_v1_row_counts and variant_row_index != expected_rows:
            raise ThinkDatasetError(
                f"{variant}: expected {expected_rows} action rows, got {variant_row_index}"
            )
        traces[variant] = prepared
        action_counts[variant] = counts
        print(
            f"loaded variant={variant} traces={len(prepared)} rows={variant_row_index}",
            flush=True,
        )
    if enforce_v1_row_counts and len(references) != EXPECTED_TOTAL_ROWS:
        raise ThinkDatasetError(
            f"expected {EXPECTED_TOTAL_ROWS} total action rows, got {len(references)}"
        )
    return traces, references, action_counts


def _think_action_prompt(instruction: str) -> str:
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
            "Put one or two short English sentences grounded only in visible evidence and causal history inside <think>...</think>, at most 60 words.",
            'Then output one compact JSON object: {"action":{...}}.',
            "Do not mention hidden targets, sensitivities, oracle values, ground-truth actions, formatting, or validation.",
            "Do not add Markdown or any other text.",
        )
    )


def _messages(
    trace: PreparedThinkTrace, *, action_index: int
) -> list[dict[str, object]]:
    if not 0 <= action_index < len(trace.actions):
        raise ThinkDatasetError("action index is out of range")
    causal_images = trace.images[: action_index + 1]
    retained_images = causal_images[-IMAGE_HISTORY_MAX:]
    first_retained_index = len(causal_images) - len(retained_images)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _think_action_prompt(trace.instruction)}
    ]

    def append_prior(index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {index + 1} assistant response "
                    f"(context only):\n{trace.responses[index]}\n"
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
    history = (
        f"{action_index} previous assistant response(s) are included in chronological order."
        if action_index
        else "No previous actions are present in this trajectory."
    )
    content.append(
        {
            "type": "text",
            "text": "\n".join(
                (
                    history,
                    "The last attached image is the current observation.",
                    f"Return the Think block and JSON action for step {action_index + 1}.",
                )
            ),
        }
    )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": trace.responses[action_index]},
    ]


def _row(
    *,
    reference: RowReference,
    trace: PreparedThinkTrace,
    mixed_row_index: int,
) -> dict[str, object]:
    teacher_metadata = trace.response_metadata[reference.action_index]
    return messages_to_verl_row(
        _messages(trace, action_index=reference.action_index),
        image_max_pixels=IMAGE_MAX_PIXELS,
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
            "no_think": False,
            "teacher_reasoning_present": True,
            "teacher_source_model": str(teacher_metadata.get("source_model") or ""),
            "teacher_result_origin": str(teacher_metadata.get("result_origin") or ""),
        },
    )


def _validate_parquet(
    *,
    path: Path,
    dataset_key: str,
    traces: Mapping[str, list[PreparedThinkTrace]],
    expected_rows: int,
) -> dict[str, object]:
    import pyarrow.parquet as pq

    lookup = {
        (variant, trace.record_id): trace
        for variant, variant_traces in traces.items()
        for trace in variant_traces
    }
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ThinkDatasetError(
            f"{dataset_key}: parquet rows={parquet.metadata.num_rows}, expected={expected_rows}"
        )
    observed = 0
    source_counts: Counter[str] = Counter()
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            metadata = row["metadata"]
            variant = str(metadata["variant"])
            if dataset_key != MIXED_DATASET_KEY and variant != dataset_key:
                raise ThinkDatasetError(f"{dataset_key}: contains row from {variant}")
            trace = lookup[(variant, str(metadata["source_record_id"]))]
            action_index = int(metadata["action_index"])
            messages = row["messages"]
            if [message["role"] for message in messages] != ["user", "assistant"]:
                raise ThinkDatasetError(f"{dataset_key}: invalid message roles")
            if messages[1]["content"] != trace.responses[action_index]:
                raise ThinkDatasetError(f"{dataset_key}: assistant target changed")
            expected_images = trace.images[: action_index + 1][-IMAGE_HISTORY_MAX:]
            actual_images = [Path(item["image"]) for item in row["images"]]
            if actual_images != list(expected_images):
                raise ThinkDatasetError(f"{dataset_key}: causal image history changed")
            user_text = str(messages[0]["content"])
            if user_text.count("Previous step ") != action_index:
                raise ThinkDatasetError(f"{dataset_key}: prior response count changed")
            if any(response not in user_text for response in trace.responses[:action_index]):
                raise ThinkDatasetError(f"{dataset_key}: prior response text changed")
            _validate_response(messages[1]["content"], trace.actions[action_index])
            if metadata["no_think"] is not False:
                raise ThinkDatasetError(f"{dataset_key}: no_think flag is not false")
            source_counts[variant] += 1
            observed += 1
    if observed != expected_rows:
        raise ThinkDatasetError(
            f"{dataset_key}: iterated {observed} rows, expected {expected_rows}"
        )
    return {
        "status": "passed",
        "row_count": observed,
        "source_row_counts": dict(sorted(source_counts.items())),
        "checks": [
            "exact_teacher_think_with_integerized_action",
            "exact_integer_action_and_numeric_types",
            "latest_three_causal_images",
            "all_previous_assistant_responses",
            "two_message_roles",
        ],
    }


def build_with_think_datasets(
    *,
    teacher_root: Path,
    raw_root: Path,
    output_root: Path,
    shuffle_seed: int = SHUFFLE_SEED,
    enforce_v1_row_counts: bool = True,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite dataset under {output_root}")
    output_root.mkdir(parents=True)
    traces, references, action_counts = _load_sources(
        teacher_root,
        raw_root,
        enforce_v1_row_counts=enforce_v1_row_counts,
    )
    rows_per_variant = {
        variant: sum(action_counts[variant].values()) for variant in VARIANTS
    }
    total_rows = len(references)
    random.Random(shuffle_seed).shuffle(references)
    shuffled_variants = [reference.variant for reference in references]
    dataset_keys = (MIXED_DATASET_KEY, *VARIANTS)
    temporary_paths = {
        key: output_root / key / f".train.parquet.tmp-{os.getpid()}"
        for key in dataset_keys
    }
    final_paths = {key: output_root / key / "train.parquet" for key in dataset_keys}
    sinks: dict[str, _ParquetSink] = {}
    try:
        for key, path in temporary_paths.items():
            path.parent.mkdir(parents=True)
            sinks[key] = _ParquetSink(path)
        for mixed_row_index, reference in enumerate(references):
            trace = traces[reference.variant][reference.trace_index]
            row = _row(
                reference=reference,
                trace=trace,
                mixed_row_index=mixed_row_index,
            )
            sinks[MIXED_DATASET_KEY].append(row)
            sinks[reference.variant].append(row)
        for sink in sinks.values():
            sink.close()
        for key in dataset_keys:
            os.replace(temporary_paths[key], final_paths[key])
            print(f"wrote dataset={key} path={final_paths[key]}", flush=True)
    finally:
        for sink in sinks.values():
            if sink.writer is not None:
                sink.writer.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

    validations: dict[str, dict[str, object]] = {}
    dataset_summaries: dict[str, dict[str, object]] = {}
    for key in dataset_keys:
        expected_rows = (
            total_rows
            if key == MIXED_DATASET_KEY
            else rows_per_variant[key]
        )
        validation = _validate_parquet(
            path=final_paths[key],
            dataset_key=key,
            traces=traces,
            expected_rows=expected_rows,
        )
        validations[key] = validation
        print(
            f"validated dataset={key} rows={validation['row_count']}",
            flush=True,
        )
        source_row_counts = validation["source_row_counts"]
        summary: dict[str, object] = {
            "schema": "exploration_depth_with_think_sft_dataset_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_key": key,
            "teacher_root": str(teacher_root),
            "raw_root": str(raw_root),
            "train_parquet": str(final_paths[key]),
            "train_rows": expected_rows,
            "trace_count": (
                sum(len(value) for value in traces.values())
                if key == MIXED_DATASET_KEY
                else len(traces[key])
            ),
            "source_row_counts": source_row_counts,
            "action_counts": (
                dict(sorted(sum(action_counts.values(), Counter()).items()))
                if key == MIXED_DATASET_KEY
                else dict(sorted(action_counts[key].items()))
            ),
            "mix_policy": MIX_POLICY if key == MIXED_DATASET_KEY else "single_variant_v1",
            "shuffle_seed": shuffle_seed,
            "image_history_max": IMAGE_HISTORY_MAX,
            "image_max_pixels": IMAGE_MAX_PIXELS,
            "prompt_contract": PROMPT_CONTRACT,
            "response_contract": RESPONSE_CONTRACT,
            "coordinate_contract": MODEL_COORDINATE_CONTRACT,
            "assistant_target_prefix": "<think>",
            "validation": validation,
        }
        if key == MIXED_DATASET_KEY:
            summary.update(
                {
                    "source_transitions": sum(
                        left != right
                        for left, right in zip(
                            shuffled_variants, shuffled_variants[1:]
                        )
                    ),
                    "max_consecutive_source_run": _max_consecutive_run(
                        shuffled_variants
                    ),
                    "first_32_sources": shuffled_variants[:32],
                }
            )
        _atomic_json(output_root / key / "summary.json", summary)
        ready = {
            "schema": "exploration_depth_with_think_sft_ready_v1",
            "status": "ready",
            "dataset_key": key,
            "train_rows": expected_rows,
            "trace_count": summary["trace_count"],
            "source_row_counts": source_row_counts,
            "response_contract": RESPONSE_CONTRACT,
            "coordinate_contract": MODEL_COORDINATE_CONTRACT,
            "validation": "passed",
            "files": {
                "train.parquet": {"size_bytes": final_paths[key].stat().st_size},
                "summary.json": {
                    "size_bytes": (output_root / key / "summary.json").stat().st_size
                },
            },
        }
        _atomic_json(output_root / key / "READY.json", ready)
        dataset_summaries[key] = summary

    top_summary: dict[str, object] = {
        "schema": "exploration_depth_with_think_seven_dataset_suite_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_root": str(teacher_root),
        "raw_root": str(raw_root),
        "dataset_count": 7,
        "trace_count": sum(len(value) for value in traces.values()),
        "expanded_action_rows": total_rows,
        "datasets": {
            key: {
                "train_parquet": str(final_paths[key]),
                "train_rows": dataset_summaries[key]["train_rows"],
                "trace_count": dataset_summaries[key]["trace_count"],
            }
            for key in dataset_keys
        },
        "prompt_contract": PROMPT_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "shuffle_seed": shuffle_seed,
        "validation": validations,
    }
    _atomic_json(output_root / "summary.json", top_summary)
    top_ready = {
        "schema": "exploration_depth_with_think_seven_dataset_ready_v1",
        "status": "ready",
        "dataset_count": 7,
        "trace_count": top_summary["trace_count"],
        "mixed_train_rows": total_rows,
        "separate_train_rows": rows_per_variant,
        "response_contract": RESPONSE_CONTRACT,
        "coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "all_validations_passed": all(
            value["status"] == "passed" for value in validations.values()
        ),
        "datasets": {
            key: str(output_root / key / "READY.json") for key in dataset_keys
        },
        "files": {
            "summary.json": {
                "size_bytes": (output_root / "summary.json").stat().st_size
            }
        },
    }
    _atomic_json(output_root / "READY.json", top_ready)
    return {"summary": top_summary, "ready": top_ready}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-root", type=Path, default=default_teacher_root())
    parser.add_argument("--raw-root", type=Path, default=default_raw_root())
    parser.add_argument("--output-root", type=Path, default=default_output_root())
    parser.add_argument("--shuffle-seed", type=int, default=SHUFFLE_SEED)
    parser.add_argument(
        "--dynamic-row-count",
        action="store_true",
        help="derive per-variant action-row totals instead of enforcing v1 totals",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_with_think_datasets(
        teacher_root=args.teacher_root,
        raw_root=args.raw_root,
        output_root=args.output_root,
        shuffle_seed=args.shuffle_seed,
        enforce_v1_row_counts=not args.dynamic_row_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
