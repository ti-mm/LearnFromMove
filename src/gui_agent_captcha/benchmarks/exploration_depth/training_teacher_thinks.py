"""Backfill visible Think labels for the matched six-environment action ablations.

This module consumes ``training_ablations.py`` output directly.  Every
trajectory is assigned to one teacher deployment, while actions inside a
trajectory are generated sequentially so later teacher requests receive the
causal assistant history produced by this run.  Per-action caches make the
online generation safely resumable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ...data.generate_four_task_ablation_teacher_thinks import (
    GenerationSettings,
    OpenAICompatibleTeacher,
    TeacherBackend,
    TeacherResult,
    _task_quality_errors,
    _validate_teacher_result,
    build_teacher_messages,
)
from ...data.verl_sft_contract import messages_to_verl_row
from ...protocol_tracks import (
    build_canonical_assistant_response,
    split_think_json_response,
)
from .training_ablations import (
    ABLATION_VARIANTS,
    ENVIRONMENT_VARIANTS,
    HISTORY_CONTRACT,
    MATCHING_POLICY,
    _allowed_action_kinds,
)
from .training_ablations import SCHEMA as SOURCE_SCHEMA
from .training_no_think import (
    _INTEGER_ACTION_SCHEMA_LINES,
    IMAGE_HISTORY_MAX,
    IMAGE_MAX_PIXELS,
    MODEL_COORDINATE_CONTRACT,
    SHUFFLE_SEED,
    _integer_model_action,
    _ParquetSink,
)

SCHEMA = "exploration_depth_six_environment_action_ablations_teacherthink_v2"
LEGACY_TEACHER_SCHEMA = "exploration_depth_six_environment_action_ablations_teacherthink_v1"
PROMPT_VERSION = "exploration_depth_action_ablation_all_action_teacher_think_v4_integer"
RESPONSE_CONTRACT = "visible_teacher_think_then_exact_integer_action_v2"
TEACHER_HISTORY_CONTRACT = (
    "latest_3_pre_action_images_corresponding_latest_2_generated_responses"
)
REUSE_POLICY = (
    "exact_id_instruction_and_image_paths_with_legacy_actions_normalizing_to_"
    "current_integer_actions_v1"
)

ENVIRONMENT_TASK = {
    "ten_choice_third_person": "ten_choice",
    "ten_choice_first_person": "ten_choice",
    "rotation_inner": "rotation",
    "rotation_outer": "rotation",
    "drag_third_person": "third_person_drag",
    "drag_first_person": "first_person_drag",
}


class TeacherThinkDatasetError(ValueError):
    """The source, cache, teacher response, or rendered output breaks the contract."""


@dataclass(frozen=True)
class AnnotatedRecord:
    record: dict[str, object]
    audit_rows: tuple[dict[str, object], ...]
    provider_requests: int
    cache_hits: int
    reused_actions: int = 0


def default_input_root() -> Path:
    from .training_ablations import default_output_root

    return default_output_root()


def default_output_root(input_root: Path) -> Path:
    return input_root.with_name("action_ablations_teacherthink_qwen35_397b_three_v2")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, values: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeacherThinkDatasetError(f"{path}: expected a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TeacherThinkDatasetError(f"{path}:{line_number}: expected an object")
            values.append(value)
    return values


def _source_snapshot(
    record: Mapping[str, object],
    *,
    environment_variant: str,
    ablation_variant: str,
) -> dict[str, object]:
    return {
        "id": record.get("id"),
        "instruction": record.get("instruction"),
        "environment_variant": environment_variant,
        "ablation_variant": ablation_variant,
        "images": record.get("images"),
        "actions": record.get("actions"),
    }


def _validate_source_record(
    record: Mapping[str, object],
    *,
    environment_variant: str,
    ablation_variant: str,
) -> tuple[str, str, list[Path], list[dict[str, object]]]:
    record_id = record.get("id")
    instruction = record.get("instruction")
    raw_images = record.get("images")
    raw_actions = record.get("actions")
    if not isinstance(record_id, str) or not record_id:
        raise TeacherThinkDatasetError("source record is missing id")
    if not isinstance(instruction, str) or not instruction:
        raise TeacherThinkDatasetError(f"{record_id}: instruction is missing")
    if record.get("environment_variant") != environment_variant:
        raise TeacherThinkDatasetError(f"{record_id}: environment variant mismatch")
    if record.get("ablation_variant") != ablation_variant:
        raise TeacherThinkDatasetError(f"{record_id}: ablation variant mismatch")
    if not isinstance(raw_images, list) or not all(isinstance(item, str) for item in raw_images):
        raise TeacherThinkDatasetError(f"{record_id}: images must be paths")
    if not isinstance(raw_actions, list) or not raw_actions or not all(
        isinstance(item, dict) for item in raw_actions
    ):
        raise TeacherThinkDatasetError(f"{record_id}: actions are missing")
    if len(raw_images) != len(raw_actions) + 1:
        raise TeacherThinkDatasetError(
            f"{record_id}: expected one more image than actions, got "
            f"{len(raw_images)} images and {len(raw_actions)} actions"
        )
    images = [Path(item) for item in raw_images]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{record_id}: missing images: {missing[:3]}")
    allowed = set(_allowed_action_kinds(ablation_variant, environment_variant))
    actions = [dict(item) for item in raw_actions]
    invalid = [action.get("kind") for action in actions if action.get("kind") not in allowed]
    if invalid:
        raise TeacherThinkDatasetError(f"{record_id}: invalid action kinds: {invalid}")
    return record_id, instruction, images, actions


def _reuse_annotated_record(
    record: Mapping[str, object],
    reused_record: Mapping[str, object] | None,
    *,
    record_index: int,
    environment_variant: str,
    ablation_variant: str,
    reuse_root: Path,
) -> AnnotatedRecord | None:
    """Reuse only a whole legacy trajectory whose visible context is unchanged."""

    if reused_record is None:
        return None
    record_id, instruction, images, actions = _validate_source_record(
        record,
        environment_variant=environment_variant,
        ablation_variant=ablation_variant,
    )
    try:
        old_id, old_instruction, old_images, old_actions = _validate_source_record(
            reused_record,
            environment_variant=environment_variant,
            ablation_variant=ablation_variant,
        )
    except (FileNotFoundError, TeacherThinkDatasetError):
        return None
    if (
        old_id != record_id
        or old_instruction != instruction
        or old_images != images
        or len(old_actions) != len(actions)
        or [_integer_model_action(action) for action in old_actions] != actions
    ):
        return None
    old_responses = reused_record.get("assistant_responses")
    old_metadata = reused_record.get("metadata")
    if (
        not isinstance(old_responses, list)
        or len(old_responses) != len(actions)
        or not all(isinstance(item, str) for item in old_responses)
        or not isinstance(old_metadata, dict)
        or not isinstance(old_metadata.get("teacher_think"), dict)
    ):
        return None
    old_teacher = dict(old_metadata["teacher_think"])
    if old_teacher.get("enable_thinking") is not True:
        return None

    task = ENVIRONMENT_TASK[environment_variant]
    responses: list[str] = []
    audit_rows: list[dict[str, object]] = []
    button_state = "up"
    for zero_index, (old_response, old_action, action) in enumerate(
        zip(old_responses, old_actions, actions, strict=True)
    ):
        thought, raw_payload = split_think_json_response(old_response)
        if thought is None:
            return None
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or set(payload) != {"action"}:
            return None
        old_result = TeacherResult(
            think=thought,
            action=payload["action"] if isinstance(payload["action"], dict) else {},
            raw_text=json.dumps(payload, ensure_ascii=False),
            finish_reason="reused",
            usage={},
            attempt=0,
            elapsed_seconds=0.0,
        )
        try:
            _validate_teacher_result(old_result, expected_action=old_action)
        except Exception:
            return None
        action_index = zero_index + 1
        responses.append(build_canonical_assistant_response(thought=thought, action=action))
        audit_rows.append(
            {
                "record_id": record_id,
                "record_index": record_index,
                "environment_variant": environment_variant,
                "ablation_variant": ablation_variant,
                "action_index": action_index,
                "total_actions": len(actions),
                "action": action,
                "think": thought,
                "think_chars": len(thought),
                "soft_quality_warnings": _task_quality_errors(
                    thought,
                    task=task,
                    variant=ablation_variant,
                    action=action,
                    button_state=button_state,
                    instruction=instruction,
                ),
                "cache_hit": False,
                "reused_label": True,
                "reuse_policy": REUSE_POLICY,
                "reuse_source_root": str(reuse_root.resolve()),
                "endpoint_index": old_teacher.get("endpoint_index"),
                "teacher_model": old_teacher.get("model"),
                "finish_reason": "reused",
                "attempt": 0,
                "elapsed_seconds": 0.0,
                "usage": {},
                "teacher_image_count": min(action_index, IMAGE_HISTORY_MAX),
                "teacher_prior_response_count": min(
                    zero_index,
                    IMAGE_HISTORY_MAX - 1,
                ),
                "button_state_before": button_state,
            }
        )
        if action.get("kind") == "mouse_down":
            button_state = "down"
        elif action.get("kind") == "mouse_up":
            button_state = "up"

    output = copy.deepcopy(dict(record))
    metadata = dict(output.get("metadata")) if isinstance(output.get("metadata"), dict) else {}
    metadata["teacher_think"] = {
        "schema": SCHEMA,
        "prompt_version": old_teacher.get("prompt_version"),
        "teacher_history_contract": old_teacher.get(
            "teacher_history_contract", TEACHER_HISTORY_CONTRACT
        ),
        "response_contract": RESPONSE_CONTRACT,
        "enable_thinking": True,
        "endpoint_index": old_teacher.get("endpoint_index"),
        "model": old_teacher.get("model"),
        "label_origin": "strict_legacy_reuse",
        "reuse_policy": REUSE_POLICY,
        "reuse_source_root": str(reuse_root.resolve()),
        "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
    }
    output["metadata"] = metadata
    output["assistant_responses"] = responses
    return AnnotatedRecord(
        record=output,
        audit_rows=tuple(audit_rows),
        provider_requests=0,
        cache_hits=0,
        reused_actions=len(actions),
    )


def _teacher_result_payload(result: TeacherResult) -> dict[str, object]:
    return {
        "think": result.think,
        "action": result.action,
        "raw_text": result.raw_text,
        "finish_reason": result.finish_reason,
        "usage": result.usage,
        "attempt": result.attempt,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _teacher_result_from_payload(payload: Mapping[str, object]) -> TeacherResult:
    think = payload.get("think")
    action = payload.get("action")
    raw_text = payload.get("raw_text")
    usage = payload.get("usage")
    if not isinstance(think, str) or not think.strip():
        raise TeacherThinkDatasetError("cache has no nonempty think")
    if not isinstance(action, dict) or not isinstance(raw_text, str):
        raise TeacherThinkDatasetError("cache has malformed teacher output")
    if not isinstance(usage, dict):
        raise TeacherThinkDatasetError("cache has malformed usage")
    return TeacherResult(
        think=think,
        action=dict(action),
        raw_text=raw_text,
        finish_reason=(
            str(payload["finish_reason"])
            if payload.get("finish_reason") is not None
            else None
        ),
        usage=dict(usage),
        attempt=int(payload.get("attempt", 1)),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
    )


def _cache_path(
    cache_root: Path,
    *,
    ablation_variant: str,
    environment_variant: str,
    record_index: int,
    action_index: int,
) -> Path:
    return (
        cache_root
        / ablation_variant
        / environment_variant
        / f"record-{record_index:06d}"
        / f"action-{action_index:02d}.json"
    )


def _load_cached_result(
    path: Path,
    *,
    request: Mapping[str, object],
    teacher_identity: Mapping[str, object],
) -> TeacherResult | None:
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema") != SCHEMA:
        raise TeacherThinkDatasetError(f"{path}: cache schema differs")
    if payload.get("request") != dict(request):
        raise TeacherThinkDatasetError(f"{path}: cached request differs from source")
    if payload.get("teacher") != dict(teacher_identity):
        raise TeacherThinkDatasetError(f"{path}: cached teacher differs from run")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise TeacherThinkDatasetError(f"{path}: cache result is missing")
    return _teacher_result_from_payload(result)


def _annotate_record(
    record: Mapping[str, object],
    *,
    record_index: int,
    environment_variant: str,
    ablation_variant: str,
    endpoint_index: int,
    teacher: TeacherBackend,
    cache_root: Path,
) -> AnnotatedRecord:
    record_id, instruction, images, actions = _validate_source_record(
        record,
        environment_variant=environment_variant,
        ablation_variant=ablation_variant,
    )
    task = ENVIRONMENT_TASK[environment_variant]
    responses: list[str] = []
    audit_rows: list[dict[str, object]] = []
    provider_requests = 0
    cache_hits = 0
    button_state = "up"
    for zero_index, action in enumerate(actions):
        action_index = zero_index + 1
        request = {
            "source": _source_snapshot(
                record,
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
            ),
            "action_index": action_index,
            "button_state_before": button_state,
            "previous_responses": list(responses),
        }
        cache_path = _cache_path(
            cache_root,
            ablation_variant=ablation_variant,
            environment_variant=environment_variant,
            record_index=record_index,
            action_index=action_index,
        )
        result = _load_cached_result(
            cache_path,
            request=request,
            teacher_identity=teacher.identity,
        )
        cache_hit = result is not None
        if result is None:
            messages = build_teacher_messages(
                task=task,
                variant=ablation_variant,
                instruction=instruction,
                image_history=images[:action_index],
                previous_responses=responses,
                current_action=action,
                action_index=action_index,
                total_actions=len(actions),
                button_state=button_state,
                image_history_max=IMAGE_HISTORY_MAX,
            )
            result = teacher.generate(messages=messages, expected_action=action)
            _write_json_atomic(
                cache_path,
                {
                    "schema": SCHEMA,
                    "request": request,
                    "teacher": teacher.identity,
                    "result": _teacher_result_payload(result),
                },
            )
            provider_requests += 1
        else:
            cache_hits += 1
        _validate_teacher_result(result, expected_action=action)
        response = build_canonical_assistant_response(thought=result.think, action=action)
        warnings = _task_quality_errors(
            result.think,
            task=task,
            variant=ablation_variant,
            action=action,
            button_state=button_state,
            instruction=instruction,
        )
        responses.append(response)
        audit_rows.append(
            {
                "record_id": record_id,
                "record_index": record_index,
                "environment_variant": environment_variant,
                "ablation_variant": ablation_variant,
                "action_index": action_index,
                "total_actions": len(actions),
                "action": action,
                "think": result.think,
                "think_chars": len(result.think),
                "soft_quality_warnings": warnings,
                "cache_hit": cache_hit,
                "reused_label": False,
                "endpoint_index": endpoint_index,
                "teacher_model": teacher.settings.model,
                "finish_reason": result.finish_reason,
                "attempt": result.attempt,
                "elapsed_seconds": result.elapsed_seconds,
                "usage": result.usage,
                "teacher_image_count": min(action_index, IMAGE_HISTORY_MAX),
                "teacher_prior_response_count": min(
                    zero_index,
                    IMAGE_HISTORY_MAX - 1,
                ),
                "button_state_before": button_state,
            }
        )
        if action.get("kind") == "mouse_down":
            button_state = "down"
        elif action.get("kind") == "mouse_up":
            button_state = "up"

    output = copy.deepcopy(dict(record))
    metadata = dict(output.get("metadata")) if isinstance(output.get("metadata"), dict) else {}
    metadata["teacher_think"] = {
        "schema": SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "teacher_history_contract": TEACHER_HISTORY_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "enable_thinking": teacher.settings.enable_thinking,
        "endpoint_index": endpoint_index,
        "model": teacher.settings.model,
        "label_origin": "online_teacher_generation",
        "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
    }
    output["metadata"] = metadata
    output["assistant_responses"] = responses
    return AnnotatedRecord(
        record=output,
        audit_rows=tuple(audit_rows),
        provider_requests=provider_requests,
        cache_hits=cache_hits,
    )


def _action_prompt(instruction: str, action_kinds: tuple[str, ...]) -> str:
    return "\n".join(
        (
            f"Task requirement: {instruction}",
            "Action space for this task:",
            f"Allowed action kinds: {', '.join(action_kinds)}",
            "Action schemas:",
            *(_INTEGER_ACTION_SCHEMA_LINES[kind] for kind in action_kinds),
            "Choose exactly one next action from the current screenshot and causal history.",
            "Coordinates are integers in the 0-1000 screenshot-relative system: x increases left to right and y top to bottom.",
            "Return exactly <think>English visual rationale</think> followed by one compact JSON object: {\"action\":{...}}.",
            "Do not add Markdown or other text.",
        )
    )


def _sft_messages(
    record: Mapping[str, object],
    *,
    action_index: int,
    action_kinds: tuple[str, ...],
) -> list[dict[str, object]]:
    instruction = str(record["instruction"])
    images = [Path(str(path)) for path in record["images"]]
    responses = record.get("assistant_responses")
    if not isinstance(responses, list) or not all(isinstance(item, str) for item in responses):
        raise TeacherThinkDatasetError(f"{record.get('id')}: assistant responses are missing")
    causal_images = images[: action_index + 1]
    retained_images = causal_images[-IMAGE_HISTORY_MAX:]
    first_retained_index = len(causal_images) - len(retained_images)
    content: list[dict[str, str]] = [
        {"type": "text", "text": _action_prompt(instruction, action_kinds)}
    ]

    def append_prior(index: int) -> None:
        content.append(
            {
                "type": "text",
                "text": (
                    f"\nPrevious step {index + 1} assistant response (context only):\n"
                    f"{responses[index]}\n"
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
        f"{action_index} previous assistant response(s) are included in chronological order; "
        "older screenshots may be omitted, but all previous responses are retained."
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
                    f"Return the <think> rationale and JSON action for step {action_index + 1}.",
                )
            ),
        }
    )
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": responses[action_index]},
    ]


def _write_parquet(
    records: Sequence[Mapping[str, object]],
    *,
    output_path: Path,
    environment_variant: str,
    ablation_variant: str,
    shuffle_seed: int,
) -> tuple[int, dict[str, int]]:
    references = [
        (record_index, action_index)
        for record_index, record in enumerate(records)
        for action_index in range(len(record["actions"]))
    ]
    random.Random(shuffle_seed).shuffle(references)
    action_kinds = _allowed_action_kinds(ablation_variant, environment_variant)
    action_counts: Counter[str] = Counter()
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    sink = _ParquetSink(temporary)
    try:
        for row_index, (record_index, action_index) in enumerate(references):
            record = records[record_index]
            actions = record["actions"]
            action = actions[action_index]
            responses = record["assistant_responses"]
            teacher = record["metadata"]["teacher_think"]
            action_counts[str(action["kind"])] += 1
            row = messages_to_verl_row(
                _sft_messages(
                    record,
                    action_index=action_index,
                    action_kinds=action_kinds,
                ),
                image_max_pixels=IMAGE_MAX_PIXELS,
                validate_image_files=False,
                metadata={
                    "source_index": record_index,
                    "source_record_id": record["id"],
                    "action_index": action_index,
                    "row_index": row_index,
                    "task": environment_variant,
                    "variant": environment_variant,
                    "environment_variant": environment_variant,
                    "ablation_variant": ablation_variant,
                    "exploration_level": record["metadata"].get("exploration_level"),
                    "image_history_max": IMAGE_HISTORY_MAX,
                    "image_max_pixels": IMAGE_MAX_PIXELS,
                    "history_contract": HISTORY_CONTRACT,
                    "response_contract": RESPONSE_CONTRACT,
                    "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
                    "allowed_action_kinds": list(action_kinds),
                    "matched_across_arms": True,
                    "no_think": False,
                    "teacher_enable_thinking": teacher["enable_thinking"],
                    "teacher_endpoint_index": teacher["endpoint_index"],
                    "teacher_model": teacher["model"],
                    "assistant_response": responses[action_index],
                },
            )
            sink.append(row)
        sink.close()
        os.replace(temporary, output_path)
    finally:
        if sink.writer is not None:
            sink.writer.close()
        temporary.unlink(missing_ok=True)
    return len(references), dict(sorted(action_counts.items()))


def _annotate_cell(
    *,
    input_root: Path,
    output_root: Path,
    environment_variant: str,
    ablation_variant: str,
    teachers: Sequence[TeacherBackend],
    reuse_root: Path | None,
    workers_per_endpoint: int,
    resume: bool,
    progress_every: int,
    shuffle_seed: int,
) -> dict[str, object]:
    source_path = input_root / ablation_variant / environment_variant / "records.jsonl"
    records = _read_jsonl(source_path)
    reuse_records: dict[str, dict[str, object]] = {}
    if reuse_root is not None:
        reuse_path = reuse_root / ablation_variant / environment_variant / "records.jsonl"
        for reused_record in _read_jsonl(reuse_path):
            reused_id = reused_record.get("id")
            if not isinstance(reused_id, str) or reused_id in reuse_records:
                raise TeacherThinkDatasetError(f"{reuse_path}: invalid or duplicate record id")
            reuse_records[reused_id] = reused_record
    cell_root = output_root / ablation_variant / environment_variant
    ready_path = cell_root / "READY.json"
    if resume and ready_path.is_file():
        ready = _read_json(ready_path)
        if (
            ready.get("schema") != SCHEMA
            or ready.get("status") != "ready"
            or int(ready.get("trace_count", -1)) != len(records)
        ):
            raise TeacherThinkDatasetError(f"{ready_path}: completed cell differs from source")
        return _read_json(cell_root / "summary.json")
    cell_root.mkdir(parents=True, exist_ok=True)
    cache_root = output_root / "cache"
    started = time.time()
    results: list[AnnotatedRecord | None] = [None] * len(records)
    completed = 0
    provider_requests = 0
    cache_hits = 0
    reused_record_count = 0
    reused_action_count = 0
    errors: list[dict[str, object]] = []
    endpoint_trace_counts: Counter[int] = Counter()
    with ThreadPoolExecutor(max_workers=workers_per_endpoint * len(teachers)) as executor:
        futures = {}
        for record_index, record in enumerate(records):
            reused = _reuse_annotated_record(
                record,
                reuse_records.get(str(record.get("id") or "")),
                record_index=record_index,
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
                reuse_root=reuse_root,
            ) if reuse_root is not None else None
            if reused is not None:
                results[record_index] = reused
                completed += 1
                reused_record_count += 1
                reused_action_count += reused.reused_actions
                continue
            endpoint_index = record_index % len(teachers)
            endpoint_trace_counts[endpoint_index] += 1
            future = executor.submit(
                _annotate_record,
                record,
                record_index=record_index,
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
                endpoint_index=endpoint_index,
                teacher=teachers[endpoint_index],
                cache_root=cache_root,
            )
            futures[future] = (record_index, str(record.get("id") or ""))
        for future in as_completed(futures):
            record_index, record_id = futures[future]
            try:
                annotated = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "record_index": record_index,
                        "record_id": record_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            else:
                results[record_index] = annotated
                completed += 1
                provider_requests += annotated.provider_requests
                cache_hits += annotated.cache_hits
            processed = completed + len(errors)
            if processed % progress_every == 0 or processed == len(records):
                progress = {
                    "schema": SCHEMA,
                    "status": "running" if completed + len(errors) < len(records) else "processed",
                    "environment_variant": environment_variant,
                    "ablation_variant": ablation_variant,
                    "records": len(records),
                    "completed": completed,
                    "errors": len(errors),
                    "provider_requests": provider_requests,
                    "cache_hits": cache_hits,
                    "reused_records": reused_record_count,
                    "reused_actions": reused_action_count,
                    "recent_errors": errors[-5:],
                    "elapsed_seconds": round(time.time() - started, 3),
                }
                _write_json_atomic(cell_root / "progress.json", progress)
                print(json.dumps(progress, ensure_ascii=False), flush=True)
    if errors:
        _write_json_atomic(
            cell_root / "FAILED.json",
            {
                "schema": SCHEMA,
                "status": "failed",
                "error_count": len(errors),
                "errors": errors[:100],
            },
        )
        raise RuntimeError(
            f"{ablation_variant}/{environment_variant}: {len(errors)} trajectories failed"
        )
    (cell_root / "FAILED.json").unlink(missing_ok=True)
    annotated = [item for item in results if item is not None]
    if len(annotated) != len(records):
        raise TeacherThinkDatasetError("completed record count differs")
    output_records = [item.record for item in annotated]
    audit_rows = [row for item in annotated for row in item.audit_rows]
    output_records_path = cell_root / "records.jsonl"
    audit_path = cell_root / "teacher_audit.jsonl"
    parquet_path = cell_root / "train.parquet"
    _write_jsonl_atomic(output_records_path, output_records)
    _write_jsonl_atomic(audit_path, audit_rows)
    train_rows, action_counts = _write_parquet(
        output_records,
        output_path=parquet_path,
        environment_variant=environment_variant,
        ablation_variant=ablation_variant,
        shuffle_seed=shuffle_seed,
    )
    if train_rows != len(audit_rows):
        raise TeacherThinkDatasetError("Parquet and teacher audit action counts differ")
    warning_counts = Counter(
        warning
        for row in audit_rows
        for warning in row.get("soft_quality_warnings", [])
    )
    summary: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_variant": environment_variant,
        "ablation_variant": ablation_variant,
        "trace_count": len(output_records),
        "train_rows": train_rows,
        "action_counts": action_counts,
        "provider_requests": provider_requests,
        "cache_hits": cache_hits,
        "reused_records": reused_record_count,
        "reused_actions": reused_action_count,
        "reuse_policy": REUSE_POLICY if reuse_root is not None else None,
        "reuse_source_root": str(reuse_root.resolve()) if reuse_root is not None else None,
        "endpoint_trace_counts": {
            str(index): endpoint_trace_counts[index] for index in range(len(teachers))
        },
        "soft_quality_warning_counts": dict(sorted(warning_counts.items())),
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_history_contract": TEACHER_HISTORY_CONTRACT,
        "final_sft_history_contract": HISTORY_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "matching_policy": MATCHING_POLICY,
        "source_records": str(source_path.resolve()),
        "outputs": {
            "records_jsonl": str(output_records_path.resolve()),
            "teacher_audit_jsonl": str(audit_path.resolve()),
            "train_parquet": str(parquet_path.resolve()),
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    _write_json_atomic(cell_root / "summary.json", summary)
    _write_json_atomic(
        ready_path,
        {
            "schema": SCHEMA,
            "status": "ready",
            "environment_variant": environment_variant,
            "ablation_variant": ablation_variant,
            "trace_count": len(output_records),
            "train_rows": train_rows,
            "action_counts": action_counts,
            "teacher_enable_thinking": all(
                teacher.settings.enable_thinking for teacher in teachers
            ),
            "reused_records": reused_record_count,
            "reused_actions": reused_action_count,
            "files": {
                "records.jsonl": {"size_bytes": output_records_path.stat().st_size},
                "teacher_audit.jsonl": {"size_bytes": audit_path.stat().st_size},
                "train.parquet": {"size_bytes": parquet_path.stat().st_size},
            },
        },
    )
    return summary


def _run_config(
    *,
    input_root: Path,
    teachers: Sequence[TeacherBackend],
    reuse_root: Path | None,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "input_root": str(input_root.resolve()),
        "source_schema": SOURCE_SCHEMA,
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_history_contract": TEACHER_HISTORY_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "model_coordinate_contract": MODEL_COORDINATE_CONTRACT,
        "reuse_root": str(reuse_root.resolve()) if reuse_root is not None else None,
        "reuse_policy": REUSE_POLICY if reuse_root is not None else None,
        "image_history_max": IMAGE_HISTORY_MAX,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "teachers": [teacher.identity for teacher in teachers],
    }


def annotate_datasets(
    *,
    input_root: Path,
    output_root: Path,
    teachers: Sequence[TeacherBackend],
    reuse_root: Path | None = None,
    workers_per_endpoint: int,
    resume: bool,
    progress_every: int = 50,
) -> dict[str, object]:
    if not teachers:
        raise ValueError("at least one teacher endpoint is required")
    if workers_per_endpoint < 1:
        raise ValueError("workers_per_endpoint must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    source_ready = _read_json(input_root / "READY.json")
    source_summary = _read_json(input_root / "summary.json")
    if source_ready.get("schema") != SOURCE_SCHEMA or source_ready.get("status") != "ready":
        raise TeacherThinkDatasetError("input root is not the ready matched no-Think dataset")
    if int(source_ready.get("dataset_count", -1)) != 12:
        raise TeacherThinkDatasetError("input root must contain 12 datasets")
    if reuse_root is not None:
        reuse_ready = _read_json(reuse_root / "READY.json")
        if (
            reuse_ready.get("schema") != LEGACY_TEACHER_SCHEMA
            or reuse_ready.get("status") != "ready"
            or reuse_ready.get("all_actions_covered") is not True
            or int(reuse_ready.get("dataset_count", -1)) != 12
        ):
            raise TeacherThinkDatasetError("legacy reuse root is not a complete 12-cell dataset")
    config = _run_config(
        input_root=input_root,
        teachers=teachers,
        reuse_root=reuse_root,
    )
    config_path = output_root / "run_config.json"
    if output_root.exists() and any(output_root.iterdir()):
        if not resume:
            raise FileExistsError(f"refusing nonempty output without --resume: {output_root}")
        if not config_path.is_file() or _read_json(config_path) != config:
            raise TeacherThinkDatasetError("resume run config differs")
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(config_path, config)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(
        output_root / "RUNNING.json",
        {
            "schema": SCHEMA,
            "status": "running",
            "started_at_utc": started_at,
            "input_root": str(input_root.resolve()),
            "output_root": str(output_root.resolve()),
            "dataset_count": 12,
            "source_actions": sum(
                int(cell["train_rows"])
                for cells in source_summary["datasets"].values()
                for cell in cells.values()
            ),
        },
    )
    summaries: dict[str, dict[str, object]] = {variant: {} for variant in ABLATION_VARIANTS}
    for ablation_index, ablation_variant in enumerate(ABLATION_VARIANTS):
        for environment_index, environment_variant in enumerate(ENVIRONMENT_VARIANTS):
            summary = _annotate_cell(
                input_root=input_root,
                output_root=output_root,
                environment_variant=environment_variant,
                ablation_variant=ablation_variant,
                teachers=teachers,
                reuse_root=reuse_root,
                workers_per_endpoint=workers_per_endpoint,
                resume=resume,
                progress_every=progress_every,
                shuffle_seed=SHUFFLE_SEED + ablation_index * 100 + environment_index,
            )
            summaries[ablation_variant][environment_variant] = summary
    total_rows = sum(
        int(cell["train_rows"])
        for cells in summaries.values()
        for cell in cells.values()
    )
    source_rows = sum(
        int(cell["train_rows"])
        for cells in source_summary["datasets"].values()
        for cell in cells.values()
    )
    if total_rows != source_rows:
        raise TeacherThinkDatasetError(
            f"teacher rows {total_rows} differ from source rows {source_rows}"
        )
    top_summary: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ready",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "dataset_count": 12,
        "trace_counts": source_summary["matched_trace_count_per_environment"],
        "train_rows": total_rows,
        "teacher_enable_thinking": all(
            teacher.settings.enable_thinking for teacher in teachers
        ),
        "teachers": [teacher.identity for teacher in teachers],
        "provider_requests": sum(
            int(cell["provider_requests"])
            for cells in summaries.values()
            for cell in cells.values()
        ),
        "cache_hits": sum(
            int(cell["cache_hits"])
            for cells in summaries.values()
            for cell in cells.values()
        ),
        "online_teacher_actions": sum(
            int(cell["provider_requests"]) + int(cell["cache_hits"])
            for cells in summaries.values()
            for cell in cells.values()
        ),
        "reused_records": sum(
            int(cell["reused_records"])
            for cells in summaries.values()
            for cell in cells.values()
        ),
        "reused_actions": sum(
            int(cell["reused_actions"])
            for cells in summaries.values()
            for cell in cells.values()
        ),
        "reuse_policy": REUSE_POLICY if reuse_root is not None else None,
        "reuse_source_root": str(reuse_root.resolve()) if reuse_root is not None else None,
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_history_contract": TEACHER_HISTORY_CONTRACT,
        "final_sft_history_contract": HISTORY_CONTRACT,
        "response_contract": RESPONSE_CONTRACT,
        "matching_policy": MATCHING_POLICY,
        "datasets": summaries,
    }
    summary_path = output_root / "summary.json"
    _write_json_atomic(summary_path, top_summary)
    _write_json_atomic(
        output_root / "READY.json",
        {
            "schema": SCHEMA,
            "status": "ready",
            "dataset_count": 12,
            "train_rows": total_rows,
            "teacher_enable_thinking": True,
            "all_actions_covered": True,
            "provider_requests": top_summary["provider_requests"],
            "cache_hits": top_summary["cache_hits"],
            "online_teacher_actions": top_summary["online_teacher_actions"],
            "reused_records": top_summary["reused_records"],
            "reused_actions": top_summary["reused_actions"],
            "summary": str(summary_path.resolve()),
        },
    )
    (output_root / "RUNNING.json").unlink(missing_ok=True)
    return top_summary


def plan_datasets(
    input_root: Path,
    *,
    reuse_root: Path | None = None,
) -> dict[str, object]:
    summary = _read_json(input_root / "summary.json")
    if summary.get("schema") != SOURCE_SCHEMA:
        raise TeacherThinkDatasetError("source summary schema differs")
    cells: dict[str, dict[str, object]] = {}
    total_rows = 0
    for ablation_variant in ABLATION_VARIANTS:
        cells[ablation_variant] = {}
        for environment_variant in ENVIRONMENT_VARIANTS:
            source = input_root / ablation_variant / environment_variant / "records.jsonl"
            records = _read_jsonl(source)
            rows = sum(len(record.get("actions", [])) for record in records)
            total_rows += rows
            reusable_records = 0
            reusable_actions = 0
            if reuse_root is not None:
                reused_values = _read_jsonl(
                    reuse_root / ablation_variant / environment_variant / "records.jsonl"
                )
                reused_by_id = {
                    str(record.get("id") or ""): record for record in reused_values
                }
                for record_index, record in enumerate(records):
                    reused = _reuse_annotated_record(
                        record,
                        reused_by_id.get(str(record.get("id") or "")),
                        record_index=record_index,
                        environment_variant=environment_variant,
                        ablation_variant=ablation_variant,
                        reuse_root=reuse_root,
                    )
                    if reused is not None:
                        reusable_records += 1
                        reusable_actions += reused.reused_actions
            cells[ablation_variant][environment_variant] = {
                "records": len(records),
                "actions": rows,
                "reusable_records": reusable_records,
                "reusable_actions": reusable_actions,
                "online_teacher_records": len(records) - reusable_records,
                "online_teacher_actions": rows - reusable_actions,
                "source": str(source.resolve()),
            }
    return {
        "schema": SCHEMA,
        "input_root": str(input_root.resolve()),
        "dataset_count": 12,
        "train_rows": total_rows,
        "reuse_root": str(reuse_root.resolve()) if reuse_root is not None else None,
        "reuse_policy": REUSE_POLICY if reuse_root is not None else None,
        "reusable_actions": sum(
            int(cell["reusable_actions"])
            for arm in cells.values()
            for cell in arm.values()
        ),
        "online_teacher_actions": sum(
            int(cell["online_teacher_actions"])
            for arm in cells.values()
            for cell in arm.values()
        ),
        "datasets": cells,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--reuse-root", type=Path, default=None)
    parser.add_argument("--base-url", default="https://maas.yicloud.com.cn/v1")
    parser.add_argument("--endpoint-model", action="append", default=[])
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--workers-per-endpoint", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.input_root is None:
        args.input_root = default_input_root()
    if args.output_root is None:
        args.output_root = default_output_root(args.input_root)
    if not args.plan_only:
        if not args.enable_thinking:
            parser.error("formal teacher generation requires --enable-thinking")
        if not args.endpoint_model:
            parser.error("provide at least one --endpoint-model")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.plan_only:
        print(
            json.dumps(
                plan_datasets(args.input_root, reuse_root=args.reuse_root),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"API key environment variable is empty: {args.api_key_env}")
    settings = [
        GenerationSettings(
            model=model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=True,
            thinking_budget=args.thinking_budget,
        )
        for model in args.endpoint_model
    ]
    teachers = [
        OpenAICompatibleTeacher(
            base_url=args.base_url,
            api_key=api_key,
            settings=item,
            timeout=args.timeout,
            retries=args.retries,
            retry_base_delay=args.retry_base_delay,
            max_connections=args.workers_per_endpoint,
        )
        for item in settings
    ]
    try:
        summary = annotate_datasets(
            input_root=args.input_root,
            output_root=args.output_root,
            teachers=teachers,
            reuse_root=args.reuse_root,
            workers_per_endpoint=args.workers_per_endpoint,
            resume=args.resume,
            progress_every=args.progress_every,
        )
    finally:
        for teacher in teachers:
            teacher.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
