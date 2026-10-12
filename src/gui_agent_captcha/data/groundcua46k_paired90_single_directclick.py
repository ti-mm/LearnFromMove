"""Build one-correct-left-click data from the paired 90% selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from gui_agent_captcha.data.verl_sft_contract import messages_to_verl_row, write_verl_parquet
from gui_agent_captcha.prompts.screenspot_pro_groundcua import (
    QWEN3_DIRECT_PROFILE,
    format_groundcua_tool_call,
    get_groundcua_profile,
    groundcua_system_prompt,
    parse_groundcua_tool_call,
)


EXPECTED_RECORDS = 33_353
EXPECTED_DISTRIBUTION = {1: 30_018, 2: 1_668, 3: 1_667}
DATASET_NAME = "groundcua46k_no_think_paired_90pct_screenspot_pro_directclick_single_leftclick_v1"
DEFAULT_OUTPUT_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / 'artifacts/datasets/groundcua/46k/')
    + DATASET_NAME
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} is not an object")
            yield row


def _action_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"record {record.get('id')!r} has no steps")
    actions = [step for step in steps if isinstance(step, Mapping) and step.get("type") == "action"]
    if not actions:
        raise ValueError(f"record {record.get('id')!r} has no actions")
    for position, action in enumerate(actions, 1):
        if action.get("kind") == "move_to":
            x, y = action.get("x"), action.get("y")
            if (
                isinstance(x, bool)
                or isinstance(y, bool)
                or not isinstance(x, int)
                or not isinstance(y, int)
                or not 0 <= x <= 1000
                or not 0 <= y <= 1000
            ):
                raise ValueError(f"record {record.get('id')!r} move {position} has invalid coordinates")
        elif action.get("kind") == "left_click":
            if action.get("x") is not None or action.get("y") is not None:
                raise ValueError(f"record {record.get('id')!r} terminal click has coordinates")
        else:
            raise ValueError(f"record {record.get('id')!r} has unsupported action")
    if actions[-1].get("kind") != "left_click":
        raise ValueError(f"record {record.get('id')!r} must end in left_click")
    return actions


def _initial_image(record: Mapping[str, Any]) -> str:
    steps = record.get("steps")
    if not isinstance(steps, list):
        raise ValueError("record steps must be a list")
    for step in steps:
        if isinstance(step, Mapping) and step.get("type") == "observation":
            image_path = step.get("image_path")
            if isinstance(image_path, str) and Path(image_path).is_file():
                return str(Path(image_path).resolve())
            raise FileNotFoundError(f"initial observation image is missing for {record.get('id')!r}")
    raise ValueError(f"record {record.get('id')!r} has no initial observation")


def _selected_action_coordinate(record: Mapping[str, Any]) -> tuple[int, int]:
    actions = _action_steps(record)
    moves = [action for action in actions if action.get("kind") == "move_to"]
    if not moves:
        raise ValueError(f"record {record.get('id')!r} has no move target")
    return int(moves[-1]["x"]), int(moves[-1]["y"])


def _require_white_cursor(record: Mapping[str, Any]) -> None:
    metadata = record.get("metadata")
    overlay = metadata.get("initial_cursor_overlay") if isinstance(metadata, Mapping) else None
    if not isinstance(overlay, Mapping) or overlay.get("cursor_overlay_version") != "white_arrow_black_outline_v1":
        raise ValueError(f"record {record.get('id')!r} is not an approved white-cursor source")


def build_single_click_row(record: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Render one image-conditioned, one-action direct-click row."""

    record_id = record.get("id")
    instruction = record.get("instruction")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("record id must be a nonempty string")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"record {record_id!r} has no instruction")
    _require_white_cursor(record)
    actions = _action_steps(record)
    coordinate = _selected_action_coordinate(record)
    image_path = _initial_image(record)
    profile = get_groundcua_profile(QWEN3_DIRECT_PROFILE)
    response = format_groundcua_tool_call(QWEN3_DIRECT_PROFILE, "left_click", coordinate=coordinate)
    if parse_groundcua_tool_call(QWEN3_DIRECT_PROFILE, response) != ("left_click", coordinate):
        raise ValueError(f"record {record_id!r} direct click failed parser round-trip")
    system = groundcua_system_prompt(QWEN3_DIRECT_PROFILE)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": instruction.strip()},
            ],
        },
        {"role": "assistant", "content": response, "trainable": True},
    ]
    row = messages_to_verl_row(
        messages,
        image_max_pixels=profile.image_max_pixels,
        image_min_pixels=profile.image_min_pixels,
        metadata={
            "row_schema_version": 1,
            "source_record_id": record_id,
            "source_move_count": len([action for action in actions if action.get("kind") == "move_to"]),
            "prompt_profile": QWEN3_DIRECT_PROFILE,
            "action_contract": "one_correct_coordinate_bearing_left_click",
            "coordinate_format": "qwen3_relative_0_1000",
            "target_action_count": 1,
            "contains_think": False,
            "assistant_prefill": False,
            "cursor_overlay": "white_arrow_black_outline_v1",
        },
    )
    public_record = {
        "id": record_id,
        "instruction": instruction.strip(),
        "source_move_count": len([action for action in actions if action.get("kind") == "move_to"]),
        "image_path": image_path,
        "action": {"kind": "left_click", "coordinate": list(coordinate)},
        "prompt_profile": QWEN3_DIRECT_PROFILE,
    }
    return row, public_record


def publish_single_click_dataset(
    source_jsonl: Path,
    selected_ledger: Path,
    output_root: Path,
    *,
    expected_records: int = EXPECTED_RECORDS,
    expected_distribution: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    source_jsonl = Path(source_jsonl).expanduser().resolve()
    selected_ledger = Path(selected_ledger).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not source_jsonl.is_file():
        raise FileNotFoundError(source_jsonl)
    if not selected_ledger.is_file():
        raise FileNotFoundError(selected_ledger)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    if output_root == Path(__file__).resolve().parents[3] or Path(__file__).resolve().parents[3] in output_root.parents:
        raise ValueError("directclick output must be outside the repository")

    selected = list(_read_jsonl(selected_ledger))
    if len(selected) != expected_records:
        raise ValueError(f"expected {expected_records} selected rows, found {len(selected)}")
    selected_ids = [row.get("source_record_id") for row in selected]
    if any(not isinstance(record_id, str) or not record_id for record_id in selected_ids):
        raise ValueError("selected ledger contains an invalid source_record_id")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selected ledger contains duplicate IDs")

    source_by_id: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(source_jsonl):
        record_id = record.get("id")
        if isinstance(record_id, str):
            source_by_id[record_id] = record
    rows: list[dict[str, Any]] = []
    public_records: list[dict[str, Any]] = []
    move_counts: Counter[int] = Counter()
    violations: list[str] = []
    for ledger_row in selected:
        record_id = str(ledger_row["source_record_id"])
        source = source_by_id.get(record_id)
        if source is None:
            raise ValueError(f"selected source record is missing: {record_id}")
        actual_moves = len([step for step in source.get("steps", []) if isinstance(step, Mapping) and step.get("type") == "action" and step.get("kind") == "move_to"])
        if actual_moves != ledger_row.get("source_move_count"):
            raise ValueError(f"move-count mismatch for {record_id}")
        row, public_record = build_single_click_row(source)
        rows.append(row)
        public_records.append(public_record)
        move_counts[actual_moves] += 1

    required_distribution = dict(expected_distribution or EXPECTED_DISTRIBUTION)
    if dict(move_counts) != required_distribution:
        raise ValueError(f"unexpected selected distribution: {dict(move_counts)}")

    output_root.mkdir(parents=True, exist_ok=False)
    try:
        write_verl_parquet(rows, output_root / "train.parquet")
        with (output_root / "trajectories.jsonl").open("x", encoding="utf-8") as handle:
            for record in public_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with (output_root / "selected_records.jsonl").open("x", encoding="utf-8") as handle:
            for row in selected:
                handle.write(
                    json.dumps(
                        {
                            "source_record_id": row["source_record_id"],
                            "source_index": row["source_index"],
                            "source_move_count": row["source_move_count"],
                            "selection_class": row["selection_class"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": 1,
            "dataset_kind": DATASET_NAME,
            "dataset_split": "train",
            "records": len(rows),
            "selected_records": len(rows),
            "selected_record_distribution": {str(key): value for key, value in sorted(move_counts.items())},
            "one_move_record_share": move_counts[1] / len(rows),
            "train_rows": len(rows),
            "trainable_assistant_messages": len(rows),
            "prompt_profile": QWEN3_DIRECT_PROFILE,
            "prompt_base_profile": "screenspot_pro_qwen3vl_vllm_dbe00114",
            "coordinate_format": "qwen3_relative_0_1000",
            "action_contract": "one_correct_coordinate_bearing_left_click",
            "actions_per_record": 1,
            "action_counts": {"left_click": len(rows)},
            "contains_think": False,
            "assistant_prefill": False,
            "cursor_overlay": "white_arrow_black_outline_v1",
            "image_policy": "initial_white_cursor_observation",
            "source_records": str(source_jsonl),
            "selected_ledger": str(selected_ledger),
        }
        audit = {
            "records": len(rows),
            "train_rows": len(rows),
            "selected_record_distribution": manifest["selected_record_distribution"],
            "one_move_record_share": manifest["one_move_record_share"],
            "id_order_equal_to_selected_ledger": [record["id"] for record in public_records] == selected_ids,
            "one_action_per_record": all(record["action"]["kind"] == "left_click" for record in public_records),
            "coordinate_valid": all(
                len(record["action"]["coordinate"]) == 2
                and all(0 <= value <= 1000 for value in record["action"]["coordinate"])
                for record in public_records
            ),
            "instructions_nonempty": all(bool(record["instruction"].strip()) for record in public_records),
            "images_present": all(Path(record["image_path"]).is_file() for record in public_records),
            "prompt_profile": QWEN3_DIRECT_PROFILE,
            "contains_think": False,
            "assistant_prefill": False,
            "white_cursor_only": True,
            "violations": violations,
        }
        (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_root / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_root / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        boolean_audit = (
            audit["id_order_equal_to_selected_ledger"],
            audit["one_action_per_record"],
            audit["coordinate_valid"],
            audit["instructions_nonempty"],
            audit["images_present"],
            audit["contains_think"] is False,
            audit["assistant_prefill"] is False,
            audit["white_cursor_only"] is True,
        )
        if violations or not all(boolean_audit):
            raise ValueError(f"directclick audit failed: {audit}")
        (output_root / ".complete").write_text("complete\n", encoding="utf-8")
        return manifest
    except Exception:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--selected-ledger", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(publish_single_click_dataset(args.source_jsonl, args.selected_ledger, args.output_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
