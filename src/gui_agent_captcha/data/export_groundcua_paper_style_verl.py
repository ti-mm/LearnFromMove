"""Export accepted GroundCUA source records as VERL AgentLoop task rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from gui_agent_captcha.data.build_groundcua_paper_style_verl_rl import (
    DEFAULT_OUTPUT_DIR,
    SCHEMA_VERSION,
)
from gui_agent_captcha.train.groundcua_paper_style_agent_loop import (
    PROMPT_PROFILES,
    TRACKS,
    task_from_mapping,
)


DATA_SOURCE = "groundcua_paper_style_static_verl_online_v1"
TASK_TYPE = "groundcua_static"
AGENT_NAMES = {
    "directclick": "groundcua_paper_style_directclick_v1",
    "moveto_leftclick": "groundcua_paper_style_moveto_leftclick_v1",
    "leftclick_terminate": "groundcua_paper_style_leftclick_terminate_v1",
}
ONLINE_PROMPT_PLACEHOLDER = "GroundCUA task is initialized by the selected SFT AgentLoop."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _task_config(record: Mapping[str, Any], track: str) -> dict[str, Any]:
    required = {
        "id",
        "platform",
        "instruction",
        "image_path",
        "bbox_model_xyxy",
        "identity",
        "split",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"selected record misses required fields: {missing}")
    if record["split"] != "train":
        raise ValueError("paper-style VERL release contains a non-training record")
    identity = record["identity"]
    if not isinstance(identity, list | tuple) or len(identity) != 3:
        raise ValueError("selected record has an invalid identity")
    task = {
        "track": track,
        "instruction": record["instruction"],
        "image_path": record["image_path"],
        "bbox_model_xyxy": record["bbox_model_xyxy"],
        "task_id": "|".join(str(item) for item in identity),
    }
    parsed = task_from_mapping(task)
    if not parsed.image_path.is_file():
        raise FileNotFoundError(f"selected image is missing: {parsed.image_path}")
    return task


def _row(record: Mapping[str, Any], *, index: int, track: str) -> dict[str, Any]:
    task = _task_config(record, track)
    row = {
        "data_source": DATA_SOURCE,
        # The AgentLoop owns the real policy prompt. It replaces this inert dataset
        # placeholder with the authoritative SFT renderer before every generation.
        "prompt": [{"role": "user", "content": ONLINE_PROMPT_PLACEHOLDER}],
        "images": [],
        "ability": TASK_TYPE,
        "reward_model": {
            "style": "groundcua_static_agent_loop_terminal_outcome",
            "ground_truth": "private_agent_loop_state",
        },
        "agent_name": AGENT_NAMES[track],
        "task_type": TASK_TYPE,
        "seed": index,
        "task_config": task,
        "extra_info": {
            "index": index,
            "split": "train",
            "prompt_profile": PROMPT_PROFILES[track],
            "identity": list(record["identity"]),
            "source_id": record["id"],
            "platform": record["platform"],
            "category": record["category"],
        },
    }
    assert_online_row(row, track=track)
    return row


def assert_online_row(row: Mapping[str, Any], *, track: str | None = None) -> None:
    expected = {
        "data_source",
        "prompt",
        "images",
        "ability",
        "reward_model",
        "agent_name",
        "task_type",
        "seed",
        "task_config",
        "extra_info",
    }
    if set(row) != expected:
        raise ValueError(f"VERL task row keys must be exactly {sorted(expected)!r}")
    if row["data_source"] != DATA_SOURCE or row["ability"] != TASK_TYPE or row["task_type"] != TASK_TYPE:
        raise ValueError("unexpected paper-style VERL task row contract")
    if row["prompt"] != [{"role": "user", "content": ONLINE_PROMPT_PLACEHOLDER}]:
        raise ValueError("online task row prompt must remain the inert AgentLoop placeholder")
    if row["images"] != []:
        raise ValueError("online task row must not duplicate AgentLoop image inputs")
    if not isinstance(row["seed"], int) or isinstance(row["seed"], bool):
        raise ValueError("VERL task row seed must be an integer")
    task = task_from_mapping(row["task_config"])
    if track is not None and task.track != track:
        raise ValueError("task track does not match selected export track")
    if row["agent_name"] != AGENT_NAMES[task.track]:
        raise ValueError("task row selects the wrong VERL AgentLoop name")
    if row["reward_model"] != {
        "style": "groundcua_static_agent_loop_terminal_outcome",
        "ground_truth": "private_agent_loop_state",
    }:
        raise ValueError("task row must use the terminal AgentLoop reward placeholder")
    info = row["extra_info"]
    if not isinstance(info, Mapping) or set(info) != {
        "index", "split", "prompt_profile", "identity", "source_id", "platform", "category"
    }:
        raise ValueError("task row extra_info has unexpected fields")
    if info["split"] != "train" or info["prompt_profile"] != PROMPT_PROFILES[task.track]:
        raise ValueError("task row split or SFT prompt profile mismatches the task")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("paper-style VERL export requires pyarrow") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _no_prompt_leakage(rows: Iterable[Mapping[str, Any]]) -> None:
    forbidden = ("bbox", "ground_truth", "source_identity", "target_model_xy")
    for row in rows:
        prompt = json.dumps(row["prompt"], ensure_ascii=False).lower()
        if any(value in prompt for value in forbidden):
            raise ValueError("online task-row prompt leaked private target metadata")


def export_release(release_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = Path(release_dir).resolve()
    if not (root / ".selection_complete").is_file():
        raise ValueError(f"source selection is not complete: {root}")
    if (root / ".complete").exists():
        raise FileExistsError(f"VERL release already exported: {root}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_VERSION or manifest.get("count") != 10_000:
        raise ValueError("release manifest is not the accepted 10K source selection")
    records = _read_jsonl(root / "train.jsonl")
    if len(records) != 10_000:
        raise ValueError(f"release has {len(records)} train records, expected 10000")
    identities = [tuple(record.get("identity", ())) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("source selection contains duplicate identities")

    tracks: dict[str, Any] = {}
    for track in TRACKS:
        rows = [_row(record, index=index, track=track) for index, record in enumerate(records)]
        _no_prompt_leakage(rows)
        output = root / "tracks" / track / "train.parquet"
        _write_parquet(output, rows)
        tracks[track] = {
            "rows": len(rows),
            "train_parquet": str(output),
            "agent_name": AGENT_NAMES[track],
            "prompt_profile": PROMPT_PROFILES[track],
        }

    audit = {
        "schema": "groundcua_paper_style_verl_export_audit_v1",
        "source_records": len(records),
        "split_counts": {"train": len(records)},
        "tracks": tracks,
        "rollouts_per_prompt": 8,
        "reward_mapping": {
            "invalid_format": -0.2,
            "valid_but_wrong": 0.0,
            "valid_and_correct": 1.0,
        },
        "no_validation_artifact": True,
        "prompt_is_rebuilt_by_agent_loop": True,
    }
    (root / "export_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "summary.json").write_text(
        json.dumps({**manifest, "export_audit": audit}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / ".complete").write_text("accepted\n", encoding="ascii")
    return audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-style GroundCUA VERL task Parquets.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(export_release(args.release_dir), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
