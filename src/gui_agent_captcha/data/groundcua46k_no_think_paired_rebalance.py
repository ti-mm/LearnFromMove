"""Build and audit paired no-Think GroundCUA 46K derivatives.

The source trajectory file is read-only.  This module keeps only the validated
trajectory contract, deterministically selects the requested move-count mix,
and writes two independent action views from that same selected source list.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from gui_agent_captcha.data.verl_sft_contract import messages_to_verl_row, write_verl_parquet
from gui_agent_captcha.prompts.screenspot_pro_qwen3vl_leftclick_terminate import (
    PROMPT_PROFILE as TERMINATE_PROMPT_PROFILE,
    SCREENSPOT_PRO_BASE_PROFILE as TERMINATE_BASE_PROFILE,
    SCREENSPOT_PRO_UPSTREAM_COMMIT,
    SCREENSPOT_PRO_UPSTREAM_REPOSITORY,
    SYSTEM_PROMPT_TEXT as TERMINATE_SYSTEM_PROMPT,
    format_tool_call as format_terminate_tool_call,
    parse_tool_call as parse_terminate_tool_call,
)
from gui_agent_captcha.prompts.screenspot_pro_qwen3vl_moveto_leftclick import (
    PROMPT_PROFILE as DIRECT_PROMPT_PROFILE,
    SYSTEM_PROMPT_TEXT as DIRECT_SYSTEM_PROMPT,
    format_tool_call as format_direct_tool_call,
    parse_tool_call as parse_direct_tool_call,
)
from gui_agent_captcha.prompts.screenspot_pro_qwen3vl_vllm import (
    IMAGE_MAX_PIXELS,
    IMAGE_MIN_PIXELS,
)


SOURCE_COORDINATE_FORMAT = "qwen3_relative_0_1000"
MODEL_FAMILY = "qwen3_vl"
IMAGE_HISTORY_MAX = 3
EXPECTED_SOURCE_DISTRIBUTION = {1: 30_018, 2: 8_149, 3: 8_149}
EXPECTED_SELECTED_DISTRIBUTION = {1: 30_018, 2: 1_668, 3: 1_667}
EXPECTED_SOURCE_RECORDS = sum(EXPECTED_SOURCE_DISTRIBUTION.values())
EXPECTED_SELECTED_RECORDS = sum(EXPECTED_SELECTED_DISTRIBUTION.values())
EXPECTED_ONE_MOVE_SHARE = 30_018 / 33_353
EXPECTED_PAIRED_TRAJECTORY_ROWS = 2 * EXPECTED_SELECTED_RECORDS
EXPECTED_PAIRED_ONE_MOVE_TRAJECTORY_ROWS = 2 * EXPECTED_SELECTED_DISTRIBUTION[1]
DIRECT_ACTION_CONTRACT = "screenspot_pro_moveto_leftclick_no_think_v1"
TERMINATE_ACTION_CONTRACT = "screenspot_pro_leftclick_terminate_no_think_v1"


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            yield value


def _assert_clean(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if re.search(r"(?:^|[_-])(sha(?:\d+)?|hash|digest)(?:$|[_-])", lowered):
                raise ValueError(f"published output contains forbidden field {key!r}")
            _assert_clean(item)
    elif isinstance(value, list):
        for item in value:
            _assert_clean(item)
    elif isinstance(value, str):
        lowered = value.lower()
        if "<think>" in lowered or "</think>" in lowered:
            raise ValueError("published output contains Think tags")


def _coordinate(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 0 or item > 1000 for item in value)
    ):
        raise ValueError(f"{label} must contain two integers in [0, 1000]")
    return int(value[0]), int(value[1])


def _validated_source(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("source record id is required")
    if record.get("source_id", record_id) != record_id:
        raise ValueError(f"source record {record_id!r} has mismatched source_id")
    source_index = record.get("source_index")
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise ValueError(f"source record {record_id!r} source_index must be an integer")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"source record {record_id!r} instruction is required")
    metadata = _mapping(record.get("metadata"), f"source record {record_id} metadata")
    if metadata.get("coordinate_format") != SOURCE_COORDINATE_FORMAT:
        raise ValueError(f"source record {record_id!r} has the wrong coordinate format")
    overlay = _mapping(metadata.get("initial_cursor_overlay"), f"source record {record_id} cursor")
    if overlay.get("cursor_overlay_version") != "white_arrow_black_outline_v1":
        raise ValueError(f"source record {record_id!r} is not a white-cursor record")

    raw_steps = record.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) < 3 or len(raw_steps) % 2 == 0:
        raise ValueError(f"source record {record_id!r} steps must alternate and end in observation")
    clean_steps: list[dict[str, Any]] = []
    source_actions: list[tuple[str, tuple[int, int] | None]] = []
    for index, raw_step in enumerate(raw_steps):
        step = _mapping(raw_step, f"source record {record_id} step {index}")
        if index % 2 == 0:
            if step.get("type") != "observation":
                raise ValueError(f"source record {record_id!r} step {index} must be an observation")
            image_path = step.get("image_path")
            if not isinstance(image_path, str) or not image_path or not Path(image_path).is_file():
                raise FileNotFoundError(f"source record {record_id!r} image is missing: {image_path!r}")
            clean_steps.append({"type": "observation", "image_path": image_path})
            continue
        if step.get("type") != "action":
            raise ValueError(f"source record {record_id!r} step {index} must be an action")
        kind = step.get("kind")
        if kind == "move_to":
            source_actions.append((kind, _coordinate([step.get("x"), step.get("y")], "move_to coordinate")))
        elif kind == "left_click":
            if step.get("x") is not None or step.get("y") is not None:
                raise ValueError(f"source record {record_id!r} left_click must be coordinate-free")
            source_actions.append((kind, None))
        else:
            raise ValueError(f"source record {record_id!r} has unsupported action {kind!r}")
        clean_steps.append({"type": "action", "kind": kind, "coordinate": source_actions[-1][1]})

    if not source_actions or source_actions[-1] != ("left_click", None):
        raise ValueError(f"source record {record_id!r} must end in a coordinate-free left_click")
    if any(kind == "left_click" for kind, _ in source_actions[:-1]):
        raise ValueError(f"source record {record_id!r} has a non-terminal left_click")
    move_count = sum(kind == "move_to" for kind, _ in source_actions)
    if move_count not in (1, 2, 3):
        raise ValueError(f"source record {record_id!r} must contain one to three move_to actions")
    return {
        "id": record_id,
        "source_id": record_id,
        "source_index": source_index,
        "instruction": instruction,
        "steps": clean_steps,
        "source_actions": source_actions,
        "source_move_count": move_count,
    }


def classify_source_record(record: Mapping[str, Any]) -> int:
    """Validate a source record and return its number of move actions."""

    return int(_validated_source(record)["source_move_count"])


def _trajectory_metadata(
    source: Mapping[str, Any], *, prompt_profile: str, action_contract: str
) -> dict[str, Any]:
    return {
        "source_record_id": source["id"],
        "source_index": source["source_index"],
        "source_move_count": source["source_move_count"],
        "coordinate_format": SOURCE_COORDINATE_FORMAT,
        "model_family": MODEL_FAMILY,
        "prompt_profile": prompt_profile,
        "action_contract": action_contract,
        "image_history_max": IMAGE_HISTORY_MAX,
        "cursor_overlay_version": "white_arrow_black_outline_v1",
        "assistant_generation_boundary": True,
        "assistant_prefill": False,
        "contains_think": False,
    }


def _build_trajectory(record: Mapping[str, Any], *, variant: str) -> dict[str, Any]:
    if (
        "source_actions" in record
        and "source_move_count" in record
        and "metadata" not in record
    ):
        source = dict(record)
    else:
        source = _validated_source(record)
    if variant == "direct":
        prompt_profile = DIRECT_PROMPT_PROFILE
        action_contract = DIRECT_ACTION_CONTRACT
        formatter = format_direct_tool_call
        parser = parse_direct_tool_call
    elif variant == "terminate":
        prompt_profile = TERMINATE_PROMPT_PROFILE
        action_contract = TERMINATE_ACTION_CONTRACT
        formatter = format_terminate_tool_call
        parser = parse_terminate_tool_call
    else:
        raise ValueError(f"unknown paired variant: {variant!r}")

    output_steps: list[dict[str, Any]] = []
    action_index = 0
    for step in source["steps"]:
        if step["type"] == "observation":
            output_steps.append({"type": "observation", "image_path": step["image_path"]})
            continue
        kind = step["kind"]
        coordinate = step["coordinate"]
        if variant == "direct":
            output_action = {"action": kind}
            if kind == "move_to":
                output_action["coordinate"] = list(coordinate)
            response = formatter(kind, coordinate=coordinate)
        elif kind == "move_to":
            output_action = {"action": "left_click", "coordinate": list(coordinate)}
            response = formatter("left_click", coordinate=coordinate)
        else:
            output_action = {"action": "terminate", "status": "success"}
            response = formatter("terminate", status="success")
        payload = {"name": "computer_use", "arguments": output_action}
        if parser(response) != payload:
            raise AssertionError("formatted action did not round-trip through its parser")
        output_steps.append(
            {
                "type": "action",
                "tool_call": payload,
                "assistant_response": response,
                "trainable": True,
                "action_index": action_index,
            }
        )
        action_index += 1
    output = {
        "id": source["id"],
        "source_id": source["source_id"],
        "source_index": source["source_index"],
        "instruction": source["instruction"],
        "steps": output_steps,
        "metadata": _trajectory_metadata(
            source, prompt_profile=prompt_profile, action_contract=action_contract
        ),
    }
    _assert_clean(output)
    return output


def build_direct_trajectory(record: Mapping[str, Any]) -> dict[str, Any]:
    return _build_trajectory(record, variant="direct")


def build_terminate_trajectory(record: Mapping[str, Any]) -> dict[str, Any]:
    return _build_trajectory(record, variant="terminate")


def _evenly_spread(items: Sequence[Mapping[str, Any]], target: int) -> list[Mapping[str, Any]]:
    ordered = sorted(items, key=lambda item: int(item["source_index"]))
    if target < 0 or target > len(ordered):
        raise ValueError(f"cannot select {target} records from class of size {len(ordered)}")
    if target == len(ordered):
        return list(ordered)
    if target == 0:
        return []
    positions = [math.floor((j + 0.5) * len(ordered) / target) for j in range(target)]
    if len(set(positions)) != target:
        raise AssertionError("evenly-spread selection produced duplicate positions")
    return [ordered[position] for position in positions]


def select_record_ids(
    records: Sequence[Mapping[str, Any]], *, two_count: int = 1_668, three_count: int = 1_667
) -> list[str]:
    """Select deterministic IDs, retaining all one-move records."""

    classes: dict[int, list[Mapping[str, Any]]] = {1: [], 2: [], 3: []}
    seen_ids: set[str] = set()
    for record in records:
        source = _validated_source(record)
        record_id = str(source["id"])
        if record_id in seen_ids:
            raise ValueError(f"duplicate source record id: {record_id!r}")
        seen_ids.add(record_id)
        classes[source["source_move_count"]].append(source)
    selected = classes[1] + _evenly_spread(classes[2], two_count) + _evenly_spread(classes[3], three_count)
    selected.sort(key=lambda item: int(item["source_index"]))
    return [str(item["id"]) for item in selected]


def _selected_records(
    records: Sequence[Mapping[str, Any]], *, two_count: int, three_count: int
) -> list[dict[str, Any]]:
    classes: dict[int, list[Mapping[str, Any]]] = {1: [], 2: [], 3: []}
    for record in records:
        classes[int(record["source_move_count"])].append(record)
    selected = classes[1] + _evenly_spread(classes[2], two_count) + _evenly_spread(classes[3], three_count)
    return [dict(record) for record in sorted(selected, key=lambda item: int(item["source_index"]))]


def _trajectory_actions(trajectory: Mapping[str, Any]) -> list[tuple[dict[str, Any], str]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError("trajectory steps must be a list")
    actions: list[tuple[dict[str, Any], str]] = []
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"trajectory step {index}")
        if step.get("type") != "action":
            continue
        if index == 0:
            raise ValueError("trajectory action has no preceding observation")
        previous = _mapping(steps[index - 1], "preceding observation")
        image_path = previous.get("image_path")
        if previous.get("type") != "observation" or not isinstance(image_path, str):
            raise ValueError("trajectory action must follow an observation")
        actions.append((dict(step), image_path))
    if not actions:
        raise ValueError("trajectory has no actions")
    return actions


def _window_rows(trajectory: Mapping[str, Any], *, variant: str) -> list[dict[str, Any]]:
    actions = _trajectory_actions(trajectory)
    instruction = str(trajectory["instruction"])
    if variant == "direct":
        system_prompt = DIRECT_SYSTEM_PROMPT
        profile = DIRECT_PROMPT_PROFILE
        contract = DIRECT_ACTION_CONTRACT
    else:
        system_prompt = TERMINATE_SYSTEM_PROMPT
        profile = TERMINATE_PROMPT_PROFILE
        contract = TERMINATE_ACTION_CONTRACT
    endpoints = [min(IMAGE_HISTORY_MAX, len(actions))]
    endpoints.extend(range(IMAGE_HISTORY_MAX + 1, len(actions) + 1))
    rows: list[dict[str, Any]] = []
    for window_index, endpoint in enumerate(endpoints):
        retained_start = max(0, endpoint - IMAGE_HISTORY_MAX)
        target_indices = tuple(range(endpoint)) if window_index == 0 else (endpoint - 1,)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        retained_image_count = 0
        for action_index, (action, image_path) in enumerate(actions[:endpoint]):
            content: list[dict[str, Any]] = []
            if action_index >= retained_start:
                content.append({"type": "image", "image": image_path})
                retained_image_count += 1
            content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": content})
            messages.append(
                {
                    "role": "assistant",
                    "content": action["assistant_response"],
                    "trainable": action_index in target_indices,
                }
            )
        rows.append(
            messages_to_verl_row(
                messages,
                image_max_pixels=IMAGE_MAX_PIXELS,
                image_min_pixels=IMAGE_MIN_PIXELS,
                metadata={
                    "row_schema_version": 1,
                    "source_record_id": trajectory["id"],
                    "source_index": trajectory["source_index"],
                    "window_index": window_index,
                    "target_action_indices": list(target_indices),
                    "prompt_profile": profile,
                    "action_contract": contract,
                    "coordinate_format": SOURCE_COORDINATE_FORMAT,
                    "image_history_max": IMAGE_HISTORY_MAX,
                    "retained_image_count": retained_image_count,
                    "contains_think": False,
                },
            )
        )
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            _assert_clean(row)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _variant_manifest(
    *,
    source_jsonl: Path,
    source_count: int,
    selected_count: int,
    selected_distribution: Mapping[int, int],
    trajectory_count: int,
    train_rows: int,
    train_one_move_rows: int,
    variant: str,
    action_counts: Mapping[str, int],
) -> dict[str, Any]:
    if variant == "direct":
        profile, base_profile, contract = DIRECT_PROMPT_PROFILE, TERMINATE_BASE_PROFILE, DIRECT_ACTION_CONTRACT
        mapping = {
            "source_move_to": "move_to with identical coordinate",
            "source_terminal_left_click": "coordinate-free left_click",
        }
    else:
        profile, base_profile, contract = TERMINATE_PROMPT_PROFILE, TERMINATE_BASE_PROFILE, TERMINATE_ACTION_CONTRACT
        mapping = {
            "source_move_to": "left_click with identical coordinate",
            "source_terminal_left_click": "terminate with status success",
        }
    train_distribution = {
        "one_move_rows": train_one_move_rows,
        "multi_move_rows": train_rows - train_one_move_rows,
    }
    manifest = {
        "schema_version": 1,
        "dataset_kind": "groundcua46k_no_think_paired_derivative",
        "dataset_split": "train",
        "source_jsonl": str(source_jsonl),
        "source_records": source_count,
        "selected_records": selected_count,
        "selected_record_distribution": {str(k): int(v) for k, v in selected_distribution.items()},
        "one_move_record_share": selected_distribution[1] / selected_count,
        "paired_trajectory_rows": trajectory_count * 2,
        "paired_one_move_trajectory_rows": selected_distribution[1] * 2,
        "records": selected_count,
        "actions": sum(action_counts.values()),
        "train_rows": train_rows,
        "train_row_distribution": train_distribution,
        "train_one_move_rows": train_one_move_rows,
        "prompt_profile": profile,
        "prompt_base_profile": base_profile,
        "prompt_base_upstream_repository": SCREENSPOT_PRO_UPSTREAM_REPOSITORY,
        "prompt_base_upstream_commit": SCREENSPOT_PRO_UPSTREAM_COMMIT,
        "coordinate_format": SOURCE_COORDINATE_FORMAT,
        "action_contract": contract,
        "action_mapping": mapping,
        "assistant_generation_boundary": True,
        "assistant_prefill": False,
        "contains_think": False,
        "cursor_overlay": "white_arrow_black_outline_v1",
        "action_counts": dict(action_counts),
    }
    if variant == "direct":
        manifest["move_to_actions"] = action_counts.get("move_to", 0)
        manifest["coordinate_free_left_click_actions"] = action_counts.get("left_click", 0)
    else:
        manifest["coordinate_left_click_actions"] = action_counts.get("left_click", 0)
        manifest["terminate_success_actions"] = action_counts.get("terminate", 0)
    _assert_clean(manifest)
    return manifest


def _load_source(source_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    previous_index: int | None = None
    for raw in _read_jsonl(source_jsonl):
        record = _validated_source(raw)
        if record["id"] in seen_ids:
            raise ValueError(f"duplicate source record id: {record['id']!r}")
        if previous_index is not None and record["source_index"] <= previous_index:
            raise ValueError("source_index must be strictly increasing")
        seen_ids.add(record["id"])
        previous_index = record["source_index"]
        records.append(record)
    return records


def _distribution(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = Counter(int(record["source_move_count"]) for record in records)
    return {index: int(counts.get(index, 0)) for index in (1, 2, 3)}


def _expected_counts(value: Mapping[int, int] | None) -> dict[int, int]:
    expected = EXPECTED_SOURCE_DISTRIBUTION if value is None else value
    return {index: int(expected.get(index, 0)) for index in (1, 2, 3)}


def _read_trajectory_file(path: Path) -> list[dict[str, Any]]:
    return list(_read_jsonl(path))


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _to_python(value.tolist())
    if isinstance(value, Mapping):
        return {key: _to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_python(item) for item in value]
    return value


def _audit_parquet(path: Path, *, variant: str) -> tuple[int, list[str]]:
    import pyarrow.parquet as parquet

    violations: list[str] = []
    try:
        parquet_file = parquet.ParquetFile(path)
    except Exception as exc:
        return 0, [f"parquet_read_error:{type(exc).__name__}"]
    profile = DIRECT_PROMPT_PROFILE if variant == "direct" else TERMINATE_PROMPT_PROFILE
    parser = parse_direct_tool_call if variant == "direct" else parse_terminate_tool_call
    row_count = 0
    for batch in parquet_file.iter_batches(batch_size=512):
        for raw_row in batch.to_pylist():
            row_count += 1
            row = _to_python(raw_row)
            _assert_clean(row)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
                violations.append("generation_boundary_or_messages_invalid")
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("prompt_profile") != profile:
                violations.append("parquet_prompt_profile_mismatch")
            for message in messages:
                if "assistant_prefill" in message:
                    violations.append("assistant_prefill_present")
                if message.get("role") == "assistant":
                    try:
                        parser(str(message.get("content", "")))
                    except ValueError:
                        violations.append("invalid_parquet_tool_call")
            for image in row.get("images", []):
                image_path = image.get("image") if isinstance(image, Mapping) else None
                if not isinstance(image_path, str) or not Path(image_path).is_file():
                    violations.append("missing_parquet_image")
    return row_count, sorted(set(violations))


def _audit_variant_trajectories(
    trajectories: Sequence[Mapping[str, Any]],
    source_by_id: Mapping[str, Mapping[str, Any]],
    *,
    variant: str,
) -> tuple[Counter[str], list[str], dict[str, list[str]]]:
    violations: list[str] = []
    counts: Counter[str] = Counter()
    images_by_id: dict[str, list[str]] = {}
    parser = parse_direct_tool_call if variant == "direct" else parse_terminate_tool_call
    expected_profile = DIRECT_PROMPT_PROFILE if variant == "direct" else TERMINATE_PROMPT_PROFILE
    previous_index: int | None = None
    for trajectory in trajectories:
        _assert_clean(trajectory)
        record_id = trajectory.get("id")
        source = source_by_id.get(str(record_id))
        if source is None:
            violations.append("output_id_not_in_source")
            continue
        if trajectory.get("instruction") != source["instruction"]:
            violations.append("instruction_mismatch")
        if trajectory.get("source_index") != source["source_index"]:
            violations.append("source_index_mismatch")
        if previous_index is not None and trajectory["source_index"] <= previous_index:
            violations.append("output_order_not_strict")
        previous_index = trajectory["source_index"]
        metadata = _mapping(trajectory.get("metadata"), "trajectory metadata")
        if metadata.get("prompt_profile") != expected_profile:
            violations.append("trajectory_prompt_profile_mismatch")
        if metadata.get("cursor_overlay_version") != "white_arrow_black_outline_v1":
            violations.append("red_or_unknown_cursor_profile")
        images = [
            str(step.get("image_path"))
            for step in trajectory.get("steps", [])
            if isinstance(step, Mapping) and step.get("type") == "observation"
        ]
        expected_images = [
            str(step["image_path"])
            for step in source["steps"]
            if step["type"] == "observation"
        ]
        if images != expected_images:
            violations.append("image_sequence_mismatch")
        if any(not Path(path).is_file() for path in images):
            violations.append("missing_trajectory_image")
        images_by_id[str(record_id)] = images
        output_actions: list[tuple[str, tuple[int, int] | None]] = []
        for step in trajectory.get("steps", []):
            if not isinstance(step, Mapping) or step.get("type") != "action":
                continue
            try:
                payload = parser(str(step.get("assistant_response", "")))
            except ValueError:
                violations.append("invalid_trajectory_tool_call")
                continue
            arguments = payload["arguments"]
            action = str(arguments["action"])
            coordinate = tuple(arguments["coordinate"]) if "coordinate" in arguments else None
            output_actions.append((action, coordinate))
            counts[action] += 1
        source_actions = list(source["source_actions"])
        if variant == "direct":
            expected_actions = source_actions
        else:
            expected_actions = [
                ("left_click", coordinate) if kind == "move_to" else ("terminate", None)
                for kind, coordinate in source_actions
            ]
        if output_actions != expected_actions:
            violations.append("action_mapping_or_coordinate_mismatch")
    return counts, sorted(set(violations)), images_by_id


def _audit_common(
    source_jsonl: Path,
    output_root: Path,
    *,
    expected_source_distribution: Mapping[int, int] | None,
    two_count: int,
    three_count: int,
) -> dict[str, Any]:
    source_records = _load_source(source_jsonl)
    source_distribution = _distribution(source_records)
    expected_source = _expected_counts(expected_source_distribution)
    violations: list[str] = []
    if source_distribution != expected_source:
        violations.append("source_distribution_mismatch")
    selected_expected = {1: source_distribution[1], 2: two_count, 3: three_count}
    selected_ids = [
        str(record["id"])
        for record in _selected_records(source_records, two_count=two_count, three_count=three_count)
    ]
    source_by_id = {record["id"]: record for record in source_records}
    direct_path = output_root / "moveto_leftclick" / "trajectories.jsonl"
    terminate_path = output_root / "leftclick_terminate" / "trajectories.jsonl"
    if not direct_path.is_file() or not terminate_path.is_file():
        violations.append("missing_trajectory_output")
        return {
            "source_records": len(source_records),
            "source_record_distribution": {str(k): v for k, v in source_distribution.items()},
            "violations": sorted(set(violations)),
        }
    direct = _read_trajectory_file(direct_path)
    terminate = _read_trajectory_file(terminate_path)
    direct_ids = [str(record.get("id")) for record in direct]
    terminate_ids = [str(record.get("id")) for record in terminate]
    if direct_ids != selected_ids or terminate_ids != selected_ids:
        violations.append("selected_id_or_order_mismatch")
    if direct_ids != terminate_ids:
        violations.append("paired_id_or_order_mismatch")
    direct_counts, direct_violations, direct_images = _audit_variant_trajectories(
        direct, source_by_id, variant="direct"
    )
    terminate_counts, terminate_violations, terminate_images = _audit_variant_trajectories(
        terminate, source_by_id, variant="terminate"
    )
    violations.extend(f"direct:{value}" for value in direct_violations)
    violations.extend(f"terminate:{value}" for value in terminate_violations)
    if [record.get("instruction") for record in direct] != [record.get("instruction") for record in terminate]:
        violations.append("paired_instruction_mismatch")
    if direct_images != terminate_images:
        violations.append("paired_image_sequence_mismatch")
    selected_records = [source_by_id[record_id] for record_id in selected_ids]
    selected_distribution = _distribution(selected_records)
    if selected_distribution != selected_expected:
        violations.append("selected_distribution_mismatch")
    expected_direct_counts = Counter(
        {
            "move_to": selected_distribution[1] + 2 * selected_distribution[2] + 3 * selected_distribution[3],
            "left_click": sum(selected_distribution.values()),
        }
    )
    expected_terminate_counts = Counter(
        {
            "left_click": expected_direct_counts["move_to"],
            "terminate": expected_direct_counts["left_click"],
        }
    )
    if direct_counts != expected_direct_counts:
        violations.append("direct_action_count_mismatch")
    if terminate_counts != expected_terminate_counts:
        violations.append("terminate_action_count_mismatch")
    train_rows: dict[str, int] = {}
    parquet_violations: list[str] = []
    for variant, output_name in (("direct", "moveto_leftclick"), ("terminate", "leftclick_terminate")):
        path = output_root / output_name
        rows, row_violations = _audit_parquet(path / "train.parquet", variant=variant)
        train_rows[variant] = rows
        parquet_violations.extend(f"{variant}:{value}" for value in row_violations)
    violations.extend(parquet_violations)
    selected_share = selected_distribution[1] / len(selected_ids) if selected_ids else 0.0
    return {
        "source_records": len(source_records),
        "source_record_distribution": {str(k): v for k, v in source_distribution.items()},
        "selected_records": len(selected_ids),
        "selected_record_distribution": {str(k): v for k, v in selected_distribution.items()},
        "one_move_record_share": selected_share,
        "paired_trajectory_rows": len(direct) + len(terminate),
        "paired_one_move_trajectory_rows": selected_distribution[1] * 2,
        "direct_action_counts": dict(direct_counts),
        "terminate_action_counts": dict(terminate_counts),
        "train_rows": train_rows,
        "prompt_profiles": {
            "direct": DIRECT_PROMPT_PROFILE,
            "terminate": TERMINATE_PROMPT_PROFILE,
        },
        "id_order_equal": direct_ids == terminate_ids == selected_ids,
        "instruction_equal": [record.get("instruction") for record in direct]
        == [record.get("instruction") for record in terminate],
        "image_sequence_equal": direct_images == terminate_images,
        "no_think": True,
        "no_assistant_prefill": True,
        "white_cursor_only": True,
        "violations": sorted(set(violations)),
    }


def audit_paired_dataset(
    source_jsonl: Path,
    output_root: Path,
    *,
    expected_source_distribution: Mapping[int, int] | None = None,
    two_count: int = 1_668,
    three_count: int = 1_667,
) -> dict[str, Any]:
    return _audit_common(
        Path(source_jsonl).expanduser().resolve(),
        Path(output_root).expanduser().resolve(),
        expected_source_distribution=expected_source_distribution,
        two_count=two_count,
        three_count=three_count,
    )


def publish_paired_dataset(
    source_jsonl: Path,
    output_root: Path,
    *,
    target_one_move_share: float = 0.9,
    expected_source_distribution: Mapping[int, int] | None = None,
    two_count: int = 1_668,
    three_count: int = 1_667,
) -> dict[str, Any]:
    source_jsonl = Path(source_jsonl).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_jsonl.is_file():
        raise FileNotFoundError(f"source JSONL not found: {source_jsonl}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    source_records = _load_source(source_jsonl)
    source_distribution = _distribution(source_records)
    expected_source = _expected_counts(expected_source_distribution)
    if source_distribution != expected_source:
        raise ValueError(f"source distribution must be {expected_source}, found {source_distribution}")
    selected = _selected_records(source_records, two_count=two_count, three_count=three_count)
    selected_distribution = _distribution(selected)
    expected_selected = {1: source_distribution[1], 2: two_count, 3: three_count}
    if selected_distribution != expected_selected:
        raise ValueError(f"selected distribution mismatch: {selected_distribution} != {expected_selected}")
    selected_share = selected_distribution[1] / len(selected)
    if abs(selected_share - target_one_move_share) > 1e-4:
        raise ValueError(
            f"selected one-move share {selected_share} does not match target {target_one_move_share}"
        )

    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "moveto_leftclick").mkdir()
    (output_root / "leftclick_terminate").mkdir()
    with (output_root / "selected_records.jsonl").open("x", encoding="utf-8") as handle:
        for record in selected:
            lineage = {
                "source_record_id": record["id"],
                "source_index": record["source_index"],
                "source_move_count": record["source_move_count"],
                "selection_class": f"{record['source_move_count']}_move",
            }
            _assert_clean(lineage)
            handle.write(json.dumps(lineage, ensure_ascii=False, separators=(",", ":")) + "\n")

    trajectories: dict[str, list[dict[str, Any]]] = {"direct": [], "terminate": []}
    rows: dict[str, list[dict[str, Any]]] = {"direct": [], "terminate": []}
    action_counts: dict[str, Counter[str]] = {"direct": Counter(), "terminate": Counter()}
    for variant in trajectories:
        output_dir = output_root / ("moveto_leftclick" if variant == "direct" else "leftclick_terminate")
        trajectory_path = output_dir / "trajectories.jsonl"
        with trajectory_path.open("x", encoding="utf-8") as handle:
            for source in selected:
                trajectory = build_direct_trajectory(source) if variant == "direct" else build_terminate_trajectory(source)
                trajectories[variant].append(trajectory)
                handle.write(json.dumps(trajectory, ensure_ascii=False, separators=(",", ":")) + "\n")
                for step in trajectory["steps"]:
                    if step["type"] == "action":
                        action_counts[variant][step["tool_call"]["arguments"]["action"]] += 1
                rows[variant].extend(_window_rows(trajectory, variant=variant))
        write_verl_parquet(rows[variant], output_dir / "train.parquet")

    audit = _audit_common(
        source_jsonl,
        output_root,
        expected_source_distribution=expected_source_distribution,
        two_count=two_count,
        three_count=three_count,
    )
    if audit["violations"]:
        raise ValueError("published pair failed audit: " + ", ".join(audit["violations"][:10]))

    trajectory_count = len(selected)
    for variant, output_name in (("direct", "moveto_leftclick"), ("terminate", "leftclick_terminate")):
        output_dir = output_root / output_name
        manifest = _variant_manifest(
            source_jsonl=source_jsonl,
            source_count=len(source_records),
            selected_count=trajectory_count,
            selected_distribution=selected_distribution,
            trajectory_count=trajectory_count,
            train_rows=len(rows[variant]),
            train_one_move_rows=selected_distribution[1],
            variant=variant,
            action_counts=action_counts[variant],
        )
        variant_audit = {
            "variant": variant,
            "records": trajectory_count,
            "actions": sum(action_counts[variant].values()),
            "train_rows": len(rows[variant]),
            "selected_record_distribution": {str(k): v for k, v in selected_distribution.items()},
            "one_move_record_share": selected_share,
            "prompt_profile": manifest["prompt_profile"],
            "contains_think": False,
            "assistant_prefill": False,
            "violations": [],
        }
        _assert_clean(variant_audit)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / "audit.json").write_text(json.dumps(variant_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_dir / ".complete").write_text("complete\n", encoding="utf-8")

    (output_root / "paired_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "source_records": len(source_records),
        "source_record_distribution": {str(k): v for k, v in source_distribution.items()},
        "selected_records": trajectory_count,
        "selected_record_distribution": {str(k): v for k, v in selected_distribution.items()},
        "one_move_record_share": selected_share,
        "paired_trajectory_rows": trajectory_count * 2,
        "paired_one_move_trajectory_rows": selected_distribution[1] * 2,
        "direct_train_rows": len(rows["direct"]),
        "terminate_train_rows": len(rows["terminate"]),
        "prompt_profiles": {"direct": DIRECT_PROMPT_PROFILE, "terminate": TERMINATE_PROMPT_PROFILE},
        "no_think": True,
        "paired_audit_passed": True,
    }
    _assert_clean(summary)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / ".complete").write_text("complete\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-one-move-share", type=float, default=0.9)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        result = audit_paired_dataset(args.source_jsonl, args.output_root)
    else:
        result = publish_paired_dataset(
            args.source_jsonl,
            args.output_root,
            target_one_move_share=args.target_one_move_share,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not result.get("violations") else 1


if __name__ == "__main__":
    raise SystemExit(main())
