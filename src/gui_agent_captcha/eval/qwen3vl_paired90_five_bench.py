"""No-Think paired-90 Qwen3-VL evaluation for all five GUI benchmarks.

The runner keeps one checkpoint and one action protocol per Job.  It performs
the five benchmarks in order and writes only external evaluation artifacts.
No content hashes are calculated or emitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from ..domains.ten_choice.sft_export import overlay_cursor_on_image_memory
from ..eval.groundcua_table2_qwen3_direct import (
    EXPECTED_DATASET_COUNTS,
    GroundingSample,
    load_benchmark_samples,
    load_image,
    score_prediction,
)
from ..prompts.screenspot_pro_qwen3vl_leftclick_terminate import (
    PROMPT_PROFILE as TERMINATE_PROFILE,
    SYSTEM_PROMPT_TEXT as TERMINATE_SYSTEM_PROMPT,
)
from ..prompts.screenspot_pro_qwen3vl_moveto_leftclick import (
    PROMPT_PROFILE as MOVETO_PROFILE,
    SYSTEM_PROMPT_TEXT as MOVETO_SYSTEM_PROMPT,
    build_multiturn_prompt as build_moveto_prompt,
    parse_tool_call as parse_moveto_tool_call,
)
from ..prompts.screenspot_pro_qwen3vl_leftclick_terminate import build_multiturn_prompt as build_terminate_prompt
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    GUIDED_PROMPT_TEMPLATE as DIRECT_PROMPT,
    build_qwen3vl_official_messages,
    parse_complete_tool_call,
)
from .groundcua_table2_qwen3_direct import (
    QWEN3_DIRECT_CONTRACT,
    resize_for_inference,
)
from .qwen3vl_checkpoint import validate_hf_checkpoint
from .qwen3vl_leftclick_terminate_multiturn_five_bench import run_multiturn_rollout


SUPPORTED_MODES = ("directclick", "moveto_leftclick", "leftclick_terminate")
SUPPORTED_BENCHMARKS = tuple(EXPECTED_DATASET_COUNTS)
IMAGE_HISTORY_MAX = 3
DIRECT_IMAGE_HISTORY_MAX = 1
MAX_MOVE_TOS = 3
MAX_LEFT_CLICKS = 3
MAX_TOKENS = 100
INITIAL_CURSOR_RENDERER = "white_arrow_black_outline_v1"
CURSOR_MAP_SCHEMA = "paired90_training_cursor_position_map_v1"
GENERATION_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": MAX_TOKENS,
}
MODE_CONTRACTS = {
    "directclick": {
        "prompt_profile": "screenspot_pro_qwen3_direct_dbe00114",
        "action_contract": "screenspot_pro_directclick_left_click_no_think_v1",
        "allowed_actions": ["left_click"],
        "image_history_max": DIRECT_IMAGE_HISTORY_MAX,
        "max_intermediate_actions": 0,
    },
    "moveto_leftclick": {
        "prompt_profile": MOVETO_PROFILE,
        "action_contract": "screenspot_pro_moveto_leftclick_no_think_v1",
        "allowed_actions": ["move_to", "left_click"],
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_intermediate_actions": MAX_MOVE_TOS,
    },
    "leftclick_terminate": {
        "prompt_profile": TERMINATE_PROFILE,
        "action_contract": "screenspot_pro_leftclick_terminate_no_think_v1",
        "allowed_actions": ["left_click", "terminate"],
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_intermediate_actions": MAX_LEFT_CLICKS,
    },
}
COORDINATE_FORMAT = "qwen3_relative_0_1000"
TRAINING_IMAGE_MIN_PIXELS = 1024
TRAINING_IMAGE_MAX_PIXELS = 99_999_999
TRAINING_RECORD_COUNT = 33_353
PAPER_STYLE_RELEASE_SCHEMA = "groundcua_paper_style_verl_selection_v1"
PAPER_STYLE_RELEASE_VERSION = "groundcua_paper_style_static_verl_v1"
PAPER_STYLE_SOURCE_ROOT = Path(
    str(Path(__file__).resolve().parents[3] / 'artifacts/checkpoints/groundcua_paper_style_verl_qwen3vl8b')
)


def _resolve_benchmarks(benchmarks: tuple[str, ...] | None) -> tuple[str, ...]:
    selected = SUPPORTED_BENCHMARKS if benchmarks is None else tuple(benchmarks)
    if not selected:
        raise ValueError("at least one benchmark is required")
    if len(set(selected)) != len(selected):
        raise ValueError("benchmarks must not contain duplicates")
    unsupported = set(selected) - set(SUPPORTED_BENCHMARKS)
    if unsupported:
        raise ValueError(f"unsupported benchmarks: {sorted(unsupported)}")
    return selected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _fresh(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _checkpoint_contract(checkpoint: Path) -> dict[str, Any]:
    report = validate_hf_checkpoint(checkpoint.resolve())
    if report.get("model_type") != "qwen3_vl":
        raise ValueError(f"checkpoint is not Qwen3-VL: {report}")
    return report


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _processor_contract(checkpoint: Path) -> dict[str, Any]:
    processor_asset = next(
        (
            name
            for name in ("processor_config.json", "preprocessor_config.json")
            if (checkpoint / name).is_file()
        ),
        None,
    )
    if processor_asset is None:
        raise FileNotFoundError(
            "checkpoint processor config is missing: "
            f"{checkpoint / 'processor_config.json'}|{checkpoint / 'preprocessor_config.json'}"
        )
    payload = _read_json_object(
        checkpoint / processor_asset, "checkpoint processor config"
    )
    image_processor = payload.get("image_processor")
    if not isinstance(image_processor, dict):
        # Public Qwen3-VL model exports put the image processor fields at the
        # top level of preprocessor_config.json.
        image_processor = payload
    size = image_processor.get("size")
    if not isinstance(size, dict):
        raise ValueError("checkpoint processor config has no image size contract")
    minimum = size.get("shortest_edge")
    maximum = size.get("longest_edge")
    if (
        not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum <= 0
        or maximum < minimum
    ):
        raise ValueError("checkpoint processor pixel range is invalid")
    patch_size = image_processor.get("patch_size")
    merge_size = image_processor.get("merge_size")
    if (
        not isinstance(patch_size, int)
        or not isinstance(merge_size, int)
        or patch_size * merge_size != QWEN3_DIRECT_CONTRACT.image_factor
    ):
        raise ValueError("checkpoint processor image factor does not match Qwen3-VL training")
    if payload.get("processor_class") != "Qwen3VLProcessor":
        raise ValueError("checkpoint does not use Qwen3VLProcessor")
    if image_processor.get("image_processor_type") not in {
        "Qwen2VLImageProcessor",
        "Qwen2VLImageProcessorFast",
    }:
        raise ValueError("checkpoint image processor type drifted")
    return {
        "processor_asset": processor_asset,
        "processor_class": payload.get("processor_class"),
        "image_processor_type": image_processor.get("image_processor_type"),
        "image_factor": patch_size * merge_size,
        "checkpoint_saved_image_min_pixels": minimum,
        "checkpoint_saved_image_max_pixels": maximum,
        "inference_image_min_pixels": TRAINING_IMAGE_MIN_PIXELS,
        "inference_image_max_pixels": TRAINING_IMAGE_MAX_PIXELS,
        "inference_pixel_source": "training_parquet_image_entries",
    }


def _direct_system_prompt() -> str:
    prefix = DIRECT_PROMPT.split("<|im_start|>user\n", 1)[0]
    if not prefix.startswith("<|im_start|>") or not prefix.endswith("<|im_end|>\n"):
        raise ValueError("direct prompt has invalid system boundaries")
    return prefix[len("<|im_start|>") : -len("<|im_end|>\n")]


def _training_parquet_contract(
    training_manifest: Path,
    *,
    mode: str,
    expected: dict[str, Any],
    expected_rows: int,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    path = training_manifest.with_name("train.parquet")
    if not path.is_file():
        raise FileNotFoundError(f"training parquet is missing: {path}")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"training parquet row count mismatch: {parquet.metadata.num_rows} != {expected_rows}"
        )
    return {
        "path": str(path.resolve()),
        "rows_from_parquet_metadata": parquet.metadata.num_rows,
        "verification_source": "existing_dataset_audit_and_parquet_metadata",
        "image_min_pixels": TRAINING_IMAGE_MIN_PIXELS,
        "image_max_pixels": TRAINING_IMAGE_MAX_PIXELS,
        "image_history_max": expected["image_history_max"],
        "full_row_rescan_performed": False,
    }


def _legacy_training_inference_contract(
    *,
    mode: str,
    checkpoint: Path,
    training_manifest: Path,
    training_run_summary: Path,
    expected_training_records: int,
    manifest: dict[str, Any],
    manifest_audit: dict[str, Any],
    run_audit: dict[str, Any],
) -> dict[str, Any]:
    """Validate the final-checkpoint audit emitted by the legacy 46K SFT runs."""

    expected = MODE_CONTRACTS[mode]
    run_dir = Path(str(run_audit.get("run_dir", ""))).expanduser().resolve()
    final_checkpoint = Path(str(run_audit.get("final_checkpoint", ""))).expanduser().resolve()
    retained = run_audit.get("retained_checkpoint_steps")
    mismatches: dict[str, dict[str, Any]] = {}

    def compare(name: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            mismatches[name] = {"expected": wanted, "actual": actual}

    compare("run.schema", run_audit.get("schema"), "groundcua_legacy_directclick_sft_final_checkpoint_audit_v1")
    compare("run.cell", run_audit.get("cell"), f"move90/{mode}")
    compare("run.violations", run_audit.get("violations"), [])
    compare("run.expected_profile", run_audit.get("expected_profile"), expected["prompt_profile"])
    compare("run.expected_rows", run_audit.get("expected_rows"), manifest.get("train_rows"))
    compare("run.complete", (run_dir / ".complete").read_text(encoding="utf-8") if (run_dir / ".complete").is_file() else None, "complete\n")
    compare("run.final_checkpoint", final_checkpoint.is_dir(), True)
    merged_provenance_path = checkpoint / "merge_provenance.json"
    if checkpoint.resolve().parent == final_checkpoint:
        compare("run.final_checkpoint_lineage", checkpoint.resolve().parent, final_checkpoint)
    else:
        merged_provenance = _read_json_object(merged_provenance_path, "merged checkpoint provenance")
        compare("merged_provenance.schema", merged_provenance.get("schema"), "verl_fsdp_hf_merge_provenance_v1")
        compare(
            "merged_provenance.source_final_checkpoint",
            Path(str(merged_provenance.get("source_final_checkpoint", ""))).expanduser().resolve(),
            final_checkpoint,
        )
        compare(
            "merged_provenance.target_dir",
            Path(str(merged_provenance.get("target_dir", ""))).expanduser().resolve(),
            checkpoint.resolve(),
        )
    compare("run.world_size", run_audit.get("world_size"), 8)
    compare("run.nodes", run_audit.get("nodes"), 1)
    compare("run.retained_checkpoint_steps", retained, [run_audit.get("latest_checkpointed_iteration")])
    compare("run.only_final_checkpoint", run_audit.get("checkpoint_steps_before_prune"), retained)

    manifest_action_contract = (
        "one_correct_coordinate_bearing_left_click"
        if mode == "directclick"
        else expected["action_contract"]
    )
    compare("manifest.records", manifest.get("records"), expected_training_records)
    compare("manifest.train_rows", manifest.get("train_rows"), run_audit.get("expected_rows"))
    compare("manifest.prompt_profile", manifest.get("prompt_profile"), expected["prompt_profile"])
    compare("manifest.action_contract", manifest.get("action_contract"), manifest_action_contract)
    compare("manifest.coordinate_format", manifest.get("coordinate_format"), COORDINATE_FORMAT)
    compare("manifest.contains_think", manifest.get("contains_think"), False)
    compare("manifest.assistant_prefill", manifest.get("assistant_prefill"), False)
    compare("manifest.cursor_overlay", manifest.get("cursor_overlay"), INITIAL_CURSOR_RENDERER)
    compare("manifest.audit.violations", manifest_audit.get("violations"), [])

    trajectory_path = training_manifest.with_name("trajectories.jsonl")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"training trajectories are missing: {trajectory_path}")
    with trajectory_path.open(encoding="utf-8") as handle:
        first_row = json.loads(handle.readline())
    metadata = first_row.get("metadata") if isinstance(first_row, dict) else None
    if not isinstance(metadata, dict):
        # Direct-click trajectories predate the nested metadata envelope; their
        # prompt profile is recorded on the row itself.
        metadata = first_row if mode == "directclick" and isinstance(first_row, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("first training trajectory has no metadata contract")
    compare("trajectory.prompt_profile", metadata.get("prompt_profile"), expected["prompt_profile"])
    if mode != "directclick" or "action_contract" in metadata:
        compare("trajectory.action_contract", metadata.get("action_contract"), expected["action_contract"])
    if "coordinate_format" in metadata:
        compare("trajectory.coordinate_format", metadata.get("coordinate_format"), COORDINATE_FORMAT)
    if "cursor_overlay_version" in metadata:
        compare("trajectory.cursor_overlay_version", metadata.get("cursor_overlay_version"), INITIAL_CURSOR_RENDERER)
    if "contains_think" in metadata:
        compare("trajectory.contains_think", metadata.get("contains_think"), False)
    if "assistant_prefill" in metadata:
        compare("trajectory.assistant_prefill", metadata.get("assistant_prefill"), False)

    train_rows = manifest.get("train_rows")
    if not isinstance(train_rows, int) or train_rows < 1:
        raise ValueError("training manifest has an invalid train_rows value")
    parquet_contract = _training_parquet_contract(
        training_manifest,
        mode=mode,
        expected=expected,
        expected_rows=train_rows,
    )
    if mismatches:
        raise ValueError(
            "legacy training/inference contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, default=str, sort_keys=True)
        )
    return {
        "schema": "groundcua_legacy_training_inference_contract_v1",
        "status": "contract_passed",
        "mode": mode,
        "checkpoint": str(checkpoint.resolve()),
        "source_final_checkpoint": str(final_checkpoint),
        "final_checkpoint_step": run_audit.get("latest_checkpointed_iteration"),
        "training_manifest": str(training_manifest.resolve()),
        "training_run_summary": str(training_run_summary.resolve()),
        "prompt_profile": expected["prompt_profile"],
        "action_contract": expected["action_contract"],
        "allowed_actions": expected["allowed_actions"],
        "coordinate_format": COORDINATE_FORMAT,
        "cursor_overlay_version": INITIAL_CURSOR_RENDERER,
        "assistant_prefill": False,
        "contains_think": False,
        "image_history_max": expected["image_history_max"],
        "max_intermediate_actions": expected["max_intermediate_actions"],
        "processor": _processor_contract(checkpoint),
        "training_parquet": parquet_contract,
    }


def _training_inference_contract(
    *,
    mode: str,
    checkpoint: Path,
    training_manifest: Path,
    training_run_summary: Path,
    expected_training_records: int = TRAINING_RECORD_COUNT,
) -> dict[str, Any]:
    expected = MODE_CONTRACTS[mode]
    manifest = _read_json_object(training_manifest, "training manifest")
    summary = _read_json_object(training_run_summary, "training run summary")
    if manifest.get("schema") == PAPER_STYLE_RELEASE_SCHEMA:
        return _paper_style_training_inference_contract(
            mode=mode,
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            training_run_summary=training_run_summary,
            expected_training_records=expected_training_records,
            manifest=manifest,
            summary=summary,
        )
    manifest_audit = _read_json_object(training_manifest.with_name("audit.json"), "training manifest audit")
    run_dir = training_run_summary.parent.resolve()
    run_audit = _read_json_object(run_dir / "audit.json", "training run audit")
    if run_audit.get("schema") == "groundcua_legacy_directclick_sft_final_checkpoint_audit_v1":
        return _legacy_training_inference_contract(
            mode=mode,
            checkpoint=checkpoint,
            training_manifest=training_manifest,
            training_run_summary=training_run_summary,
            expected_training_records=expected_training_records,
            manifest=manifest,
            manifest_audit=manifest_audit,
            run_audit=run_audit,
        )
    trajectory_path = training_manifest.with_name("trajectories.jsonl")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"training trajectories are missing: {trajectory_path}")
    with trajectory_path.open(encoding="utf-8") as handle:
        first_row = json.loads(handle.readline())
    metadata = first_row.get("metadata") if isinstance(first_row, dict) else None
    if not isinstance(metadata, dict):
        raise ValueError("first training trajectory has no metadata contract")
    train_rows = manifest.get("train_rows")
    if not isinstance(train_rows, int) or train_rows < 1:
        raise ValueError("training manifest has an invalid train_rows value")
    parquet_contract = _training_parquet_contract(
        training_manifest,
        mode=mode,
        expected=expected,
        expected_rows=train_rows,
    )

    checkpoint = checkpoint.resolve()
    recorded_checkpoint = Path(str(summary.get("checkpoint", ""))).expanduser().resolve()
    dataset_contract = summary.get("dataset_contract")
    mismatches: dict[str, dict[str, Any]] = {}

    def compare(name: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            mismatches[name] = {"expected": wanted, "actual": actual}

    compare("manifest.schema", manifest.get("schema"), "groundcua700k_sft_variant_v1")
    compare("manifest.variant", manifest.get("variant"), mode)
    compare("manifest.records", manifest.get("records"), expected_training_records)
    compare("manifest.prompt_profile", manifest.get("prompt_profile"), expected["prompt_profile"])
    compare("manifest.action_contract", manifest.get("action_contract"), expected["action_contract"])
    compare("manifest.coordinate_format", manifest.get("coordinate_format"), COORDINATE_FORMAT)
    compare("manifest.cursor_overlay_version", manifest.get("cursor_overlay_version"), INITIAL_CURSOR_RENDERER)
    compare("manifest.contains_think", manifest.get("contains_think"), False)
    compare("manifest.assistant_prefill", manifest.get("assistant_prefill"), False)
    compare("manifest.audit.violations", manifest_audit.get("violations"), [])
    compare("trajectory.prompt_profile", metadata.get("prompt_profile"), expected["prompt_profile"])
    compare("trajectory.action_contract", metadata.get("action_contract"), expected["action_contract"])
    compare("trajectory.image_history_max", metadata.get("image_history_max"), expected["image_history_max"])
    compare("trajectory.coordinate_format", metadata.get("coordinate_format"), COORDINATE_FORMAT)
    compare("trajectory.cursor_overlay_version", metadata.get("cursor_overlay_version"), INITIAL_CURSOR_RENDERER)
    compare("run.complete", (run_dir / ".complete").read_text(encoding="utf-8") if (run_dir / ".complete").is_file() else None, "complete\n")
    compare("run.audit.violations", run_audit.get("violations"), [])
    compare("run.variant", summary.get("variant"), mode)
    compare("run.run_dir", Path(str(summary.get("run_dir", ""))).expanduser().resolve(), run_dir)
    latest_iteration = summary.get("latest_checkpointed_iteration")
    compare("run.recorded_checkpoint_exists", recorded_checkpoint.is_dir(), True)
    compare("run.recorded_checkpoint_parent", recorded_checkpoint.parent, run_dir / "checkpoint")
    if not isinstance(latest_iteration, int) or latest_iteration < 1:
        mismatches["run.latest_checkpointed_iteration"] = {
            "expected": "positive integer",
            "actual": latest_iteration,
        }
    else:
        compare("run.recorded_checkpoint_name", recorded_checkpoint.name, f"global_step_{latest_iteration}")
        checkpoint_name = checkpoint.name.lower().replace("_", "-")
        checkpoint_parent_name = checkpoint.parent.name.lower().replace("_", "-")
        compare(
            "merged_checkpoint.final_step_lineage",
            (
                f"step{latest_iteration}" in checkpoint_name
                or checkpoint_parent_name == f"global-step-{latest_iteration}"
            ),
            True,
        )
    checkpoint_dirs = sorted(
        path.name
        for path in (run_dir / "checkpoint").glob("global_step_*")
        if path.is_dir()
    )
    compare("run.only_final_checkpoint", checkpoint_dirs, [recorded_checkpoint.name])
    if not isinstance(dataset_contract, dict):
        mismatches["run.dataset_contract"] = {"expected": "object", "actual": dataset_contract}
    else:
        compare("run.dataset.variant", dataset_contract.get("variant"), mode)
        compare("run.dataset.root", Path(str(dataset_contract.get("dataset_root", ""))).expanduser().resolve(), training_manifest.parent.parent.resolve())
        compare("run.dataset.train_rows", dataset_contract.get("train_rows"), manifest.get("train_rows"))
    if mismatches:
        raise ValueError("training/inference contract mismatch: " + json.dumps(mismatches, ensure_ascii=False, default=str, sort_keys=True))

    return {
        "schema": (
            "groundcua700k_100k_training_inference_contract_v1"
            if expected_training_records == 100_000
            else f"groundcua{expected_training_records}_training_inference_contract_v1"
        ),
        "status": "contract_passed",
        "mode": mode,
        "checkpoint": str(checkpoint),
        "source_final_checkpoint": str(recorded_checkpoint),
        "final_checkpoint_step": latest_iteration,
        "training_manifest": str(training_manifest.resolve()),
        "training_run_summary": str(training_run_summary.resolve()),
        "prompt_profile": expected["prompt_profile"],
        "action_contract": expected["action_contract"],
        "allowed_actions": expected["allowed_actions"],
        "coordinate_format": COORDINATE_FORMAT,
        "cursor_overlay_version": INITIAL_CURSOR_RENDERER,
        "assistant_prefill": False,
        "contains_think": False,
        "image_history_max": expected["image_history_max"],
        "max_intermediate_actions": expected["max_intermediate_actions"],
        "processor": _processor_contract(checkpoint),
        "training_parquet": parquet_contract,
    }


def _paper_style_training_inference_contract(
    *,
    mode: str,
    checkpoint: Path,
    training_manifest: Path,
    training_run_summary: Path,
    expected_training_records: int,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Validate the static paper-style VERL release and its FSDP conversion."""

    expected = MODE_CONTRACTS[mode]
    release_dir = training_manifest.parent.resolve()
    mismatches: dict[str, dict[str, Any]] = {}

    def compare(name: str, actual: Any, wanted: Any) -> None:
        if actual != wanted:
            mismatches[name] = {"expected": wanted, "actual": actual}

    compare("manifest.schema", manifest.get("schema"), PAPER_STYLE_RELEASE_SCHEMA)
    compare("manifest.status", manifest.get("status"), "accepted")
    compare("manifest.count", manifest.get("count"), expected_training_records // 10)
    compare("manifest.split_counts", manifest.get("split_counts"), {"train": expected_training_records // 10})
    compare("manifest.release_dir", release_dir.name, "paper_style_verl_qwen3vl8b_10k_v1")
    compare("summary.schema", summary.get("schema"), PAPER_STYLE_RELEASE_SCHEMA)
    compare("summary.count", summary.get("count"), expected_training_records // 10)
    profiles = manifest.get("prompt_profiles")
    compare("manifest.prompt_profile", profiles.get(mode) if isinstance(profiles, dict) else None, expected["prompt_profile"])

    parquet_path = release_dir / "tracks" / mode / "train.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"paper-style training parquet is missing: {parquet_path}")
    try:
        import pyarrow.parquet as pq

        parquet_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    except Exception as error:
        raise ValueError(f"paper-style training parquet cannot be inspected: {parquet_path}") from error
    compare("training_parquet.rows", parquet_rows, expected_training_records // 10)

    provenance = _read_json_object(checkpoint / "merge_provenance.json", "merged checkpoint provenance")
    source = Path(str(provenance.get("source_checkpoint", ""))).expanduser().resolve()
    target = Path(str(provenance.get("target_dir", ""))).expanduser().resolve()
    compare("merge.schema", provenance.get("schema"), "verl_fsdp_hf_merge_provenance_v1")
    compare("merge.track", provenance.get("track"), mode)
    compare("merge.world_size", provenance.get("world_size"), 8)
    compare("merge.source_root", source.parent, (PAPER_STYLE_SOURCE_ROOT / mode).resolve())
    source_exists = source.is_dir()
    source_checkpoint_status = provenance.get("source_checkpoint_status")
    if source_exists:
        compare("merge.source_exists", True, True)
    else:
        # Checkpoint retention can prune the large FSDP input after a merged
        # Hugging Face export is complete.  In that case, require an explicit
        # provenance marker and verify every safetensors file referenced by the
        # merged index before accepting the derived model as reproducible.
        compare("merge.source_checkpoint_status", source_checkpoint_status, "pruned_after_merge")
        index_path = checkpoint / "model.safetensors.index.json"
        merged_weights_complete = False
        if index_path.is_file():
            try:
                index_payload = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index_payload.get("weight_map")
                files = set(weight_map.values()) if isinstance(weight_map, dict) else set()
                merged_weights_complete = bool(files) and all(
                    (checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
                    for name in files
                )
            except (OSError, json.JSONDecodeError, TypeError):
                merged_weights_complete = False
        compare("merge.merged_weights_complete", merged_weights_complete, True)
    compare("merge.target_dir", target, checkpoint.resolve())
    step = provenance.get("global_step")
    compare("merge.step_name", source.name, f"global_step_{step}" if isinstance(step, int) else None)
    if not isinstance(step, int) or step < 1:
        mismatches["merge.global_step"] = {"expected": "positive integer", "actual": step}
    if mismatches:
        raise ValueError(
            "paper-style training/inference contract mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, default=str, sort_keys=True)
        )
    return {
        "schema": "groundcua_paper_style_verl_training_inference_contract_v1",
        "status": "contract_passed",
        "lineage": PAPER_STYLE_RELEASE_VERSION,
        "mode": mode,
        "checkpoint": str(checkpoint.resolve()),
        "source_final_checkpoint": str(source),
        "source_checkpoint_exists": source_exists,
        "source_checkpoint_status": "present" if source_exists else source_checkpoint_status,
        "final_checkpoint_step": step,
        "world_size": 8,
        "training_manifest": str(training_manifest.resolve()),
        "training_run_summary": str(training_run_summary.resolve()),
        "release_dir": str(release_dir),
        "prompt_profile": expected["prompt_profile"],
        "action_contract": expected["action_contract"],
        "allowed_actions": expected["allowed_actions"],
        "coordinate_format": COORDINATE_FORMAT,
        "cursor_overlay_version": INITIAL_CURSOR_RENDERER,
        "assistant_prefill": False,
        "contains_think": False,
        "image_history_max": expected["image_history_max"],
        "max_intermediate_actions": expected["max_intermediate_actions"],
        "processor": _processor_contract(checkpoint),
        "training_parquet": {
            "path": str(parquet_path),
            "rows_from_parquet_metadata": parquet_rows,
            "verification_source": "paper_style_release_manifest_and_parquet_metadata",
            "image_min_pixels": TRAINING_IMAGE_MIN_PIXELS,
            "image_max_pixels": TRAINING_IMAGE_MAX_PIXELS,
            "image_history_max": expected["image_history_max"],
            "full_row_rescan_performed": False,
        },
        "merge_provenance": provenance,
    }


def _materialize_source(sample: GroundingSample, output: Path, index: int) -> Path:
    if sample.image_path is not None:
        return sample.image_path.resolve()
    target = output / "source_images" / f"sample_{index:06d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    load_image(sample).save(target, format="PNG")
    return target.resolve()


def _load_cursor_map(
    path: Path,
    samples_by_benchmark: dict[str, list[GroundingSample]],
) -> dict[str, dict[str, list[int]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != CURSOR_MAP_SCHEMA:
        raise ValueError(f"unsupported cursor map schema: {payload.get('schema')!r}")
    positions = payload.get("positions")
    if not isinstance(positions, dict):
        raise ValueError("cursor map positions must be an object")
    resolved: dict[str, dict[str, list[int]]] = {}
    for benchmark, samples in samples_by_benchmark.items():
        rows = positions.get(benchmark)
        if not isinstance(rows, dict):
            raise ValueError(f"cursor map has no rows for benchmark {benchmark}")
        expected_ids = {sample.record_id for sample in samples}
        if set(rows) != expected_ids:
            missing = sorted(expected_ids - set(rows))[:3]
            extra = sorted(set(rows) - expected_ids)[:3]
            raise ValueError(f"cursor map coverage mismatch for {benchmark}: missing={missing}, extra={extra}")
        checked: dict[str, list[int]] = {}
        for record_id, coordinate in rows.items():
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
                or any(value < 0 or value > 1000 for value in coordinate)
            ):
                raise ValueError(f"invalid cursor coordinate for {benchmark}:{record_id}")
            checked[record_id] = [int(coordinate[0]), int(coordinate[1])]
        resolved[benchmark] = checked
    return resolved


def _cursor_policy(
    initial_cursor_map: Path | None,
) -> str:
    if initial_cursor_map is not None:
        return "training_style_cursor_map"
    return "raw_benchmark_image"


def _cursor_metadata(
    *,
    initial_cursor_map: Path | None,
    coordinate: list[int] | None,
) -> dict[str, Any]:
    return {
        "initial_image_policy": _cursor_policy(initial_cursor_map),
        "initial_cursor_coordinate": list(coordinate) if coordinate is not None else None,
        "initial_cursor_renderer": (
            INITIAL_CURSOR_RENDERER
            if coordinate is not None or initial_cursor_map is not None
            else None
        ),
        "initial_cursor_map": str(initial_cursor_map.resolve()) if initial_cursor_map is not None else None,
    }


class VLLMGenerator:
    def __init__(
        self,
        checkpoint: Path,
        *,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
        image_history_max: int,
        processor_contract: dict[str, Any],
    ) -> None:
        from vllm import LLM, SamplingParams

        self._llm = LLM(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            seed=0,
            limit_mm_per_prompt={"image": image_history_max},
            mm_processor_kwargs={
                "min_pixels": processor_contract["inference_image_min_pixels"],
                "max_pixels": processor_contract["inference_image_max_pixels"],
            },
        )
        self._sampling = SamplingParams(**GENERATION_SETTINGS)

    def __call__(self, prompt: str, images: tuple[Image.Image, ...]) -> str:
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": list(images)}}],
            sampling_params=self._sampling,
            use_tqdm=False,
        )
        if len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion")
        return str(outputs[0].outputs[0].text).strip()


def _score(sample: GroundingSample, coordinate: list[int] | tuple[int, int] | None) -> str:
    return score_prediction(
        sample,
        tuple(coordinate) if coordinate is not None else None,
    ).correctness


def _direct_prompt(
    processor: Any,
    sample: GroundingSample,
    *,
    cursor_coordinate: tuple[int, int] | None = None,
) -> tuple[str, Image.Image]:
    image = load_image(sample)
    if cursor_coordinate is not None:
        cursor_xy = (
            cursor_coordinate[0] * image.width / 1000.0,
            cursor_coordinate[1] * image.height / 1000.0,
        )
        rendered = overlay_cursor_on_image_memory(image, cursor_xy=cursor_xy)
        image.close()
        image = rendered
    resized = resize_for_inference(image)
    if resized is not image:
        image.close()
        image = resized
    prompt = processor.apply_chat_template(
        build_qwen3vl_official_messages(sample.instruction, image),
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(prompt, str) or not prompt.endswith("<|im_start|>assistant\n"):
        raise ValueError("direct prompt does not end at an empty assistant boundary")
    return prompt, image


def _run_direct(
    samples: list[GroundingSample],
    *,
    processor: Any,
    generator: VLLMGenerator,
    output: Path,
    checkpoint: Path,
    initial_cursor_map: Path | None = None,
    cursor_positions: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    predictions = output / "predictions.jsonl"
    counts = {"correct": 0, "wrong": 0, "wrong_format": 0}
    with predictions.open("x", encoding="utf-8") as handle:
        for index, sample in enumerate(samples):
            sample_cursor = (
                tuple(cursor_positions[sample.record_id])
                if cursor_positions is not None
                else None
            )
            prompt, image = _direct_prompt(
                processor,
                sample,
                cursor_coordinate=sample_cursor,
            )
            raw = ""
            parse_error = None
            coordinate: tuple[int, int] | None = None
            try:
                raw = generator(prompt, (image,))
                coordinate = parse_complete_tool_call(raw)
            except Exception as error:  # model output is classified, not fatal
                parse_error = f"{type(error).__name__}: {error}"
            finally:
                image.close()
            correctness = _score(sample, coordinate)
            counts[correctness] += 1
            handle.write(
                json.dumps(
                    {
                        "benchmark": sample.benchmark,
                        "record_id": sample.record_id,
                        "instruction": sample.instruction,
                        "raw_response": raw,
                        "coordinate_1000": list(coordinate) if coordinate is not None else None,
                        "correctness": correctness,
                        "parse_error": parse_error,
                        **_cursor_metadata(
                            initial_cursor_map=initial_cursor_map,
                            coordinate=cursor_positions.get(sample.record_id) if cursor_positions else None,
                        ),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    total = len(samples)
    summary = {
        "schema": "paired90_qwen3vl_five_bench_summary_v1",
        "status": "evaluation_complete",
        "benchmark": sample_benchmark(samples),
        "prompt_profile": "screenspot_pro_qwen3_direct_dbe00114",
        "action_contract": MODE_CONTRACTS["directclick"]["action_contract"],
        "checkpoint": str(checkpoint.resolve()),
        "allowed_actions": ["left_click"],
        "assistant_prefill": False,
        "think_present": False,
        "image_history_max": DIRECT_IMAGE_HISTORY_MAX,
        **_cursor_metadata(
            initial_cursor_map=initial_cursor_map,
            coordinate=None,
        ),
        "total": total,
        **counts,
        "accuracy": counts["correct"] / total if total else 0.0,
        "predictions": str(predictions),
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "protocol.json",
        {
            "prompt_profile": "screenspot_pro_qwen3_direct_dbe00114",
            "prompt_source": "screenspot_pro_qwen3vl_vllm",
            "action_contract": MODE_CONTRACTS["directclick"]["action_contract"],
            "allowed_actions": ["left_click"],
            "checkpoint": str(checkpoint.resolve()),
            "assistant_prefill": False,
            "think_present": False,
            "image_history_max": DIRECT_IMAGE_HISTORY_MAX,
            "generation_settings": GENERATION_SETTINGS,
            **_cursor_metadata(
                initial_cursor_map=initial_cursor_map,
                coordinate=None,
            ),
        },
    )
    return summary


def _cursor_image(
    source: Path,
    output: Path,
    index: int,
    move_index: int,
    coordinate: list[int],
    *,
    observation_root: Path | None = None,
) -> Path:
    with Image.open(source) as image:
        rendered = overlay_cursor_on_image_memory(
            image,
            cursor_xy=(coordinate[0] * image.width / 1000.0, coordinate[1] * image.height / 1000.0),
        )
    root = observation_root if observation_root is not None else output / "observations"
    path = root / f"sample_{index:06d}" / f"turn_{move_index:02d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(path, format="PNG")
    return path.resolve()


def _run_moveto(
    samples: list[GroundingSample],
    *,
    generator: VLLMGenerator,
    output: Path,
    checkpoint: Path,
    initial_cursor_map: Path | None = None,
    cursor_positions: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    predictions = output / "predictions.jsonl"
    counts = {"correct": 0, "wrong": 0, "wrong_action": 0, "wrong_format": 0}
    # Cursor-rendered observations are transient model inputs. Keeping them on
    # the local container avoids consuming the shared evaluation disk quota.
    with TemporaryDirectory(prefix="qwen3vl-moveto-observations-") as temporary_root:
        observation_root = Path(temporary_root)
        with predictions.open("x", encoding="utf-8") as handle:
            for index, sample in enumerate(samples):
                source = _materialize_source(sample, observation_root, index)
                initial_coordinate = cursor_positions.get(sample.record_id) if cursor_positions else None
                initial_image = (
                    _cursor_image(
                        source,
                        output,
                        index,
                        0,
                        initial_coordinate,
                        observation_root=observation_root,
                    )
                    if initial_coordinate is not None
                    else source
                )
                image_paths = [initial_image]
                history: list[str] = []
                moves: list[list[int]] = []
                turns: list[dict[str, Any]] = []
                correctness = "wrong_format"
                for turn_index in range(MAX_MOVE_TOS + 1):
                    built = build_moveto_prompt(
                        sample.instruction,
                        image_paths=image_paths,
                        assistant_response_history=history,
                        images_to_keep=IMAGE_HISTORY_MAX,
                    )
                    images = tuple(Image.open(path).convert("RGB") for path in built.image_paths)
                    raw = ""
                    try:
                        raw = generator(built.prompt, images)
                        parsed = parse_moveto_tool_call(raw)
                        arguments = parsed["arguments"]
                        action = arguments["action"]
                        turns.append({"turn": turn_index, "action": action, "raw_response": raw})
                        if action == "move_to":
                            if len(moves) >= MAX_MOVE_TOS:
                                correctness = "wrong_action"
                                break
                            coordinate = list(arguments["coordinate"])
                            moves.append(coordinate)
                            history.append(raw)
                            image_paths.append(
                                _cursor_image(
                                    source,
                                    output,
                                    index,
                                    len(moves),
                                    coordinate,
                                    observation_root=observation_root,
                                )
                            )
                            continue
                        if action == "left_click" and moves:
                            correctness = _score(sample, moves[-1])
                        else:
                            correctness = "wrong_action"
                        break
                    except Exception as error:
                        turns.append({"turn": turn_index, "raw_response": raw, "parse_error": str(error)})
                        correctness = "wrong_format"
                        break
                    finally:
                        for image in images:
                            image.close()
                counts[correctness] += 1
                handle.write(
                    json.dumps(
                        {
                            "benchmark": sample.benchmark,
                            "record_id": sample.record_id,
                            "instruction": sample.instruction,
                            "moves": moves,
                            "turns": turns,
                            "correctness": correctness,
                            **_cursor_metadata(
                                initial_cursor_map=initial_cursor_map,
                                coordinate=initial_coordinate,
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    total = len(samples)
    summary = {
        "schema": "paired90_qwen3vl_five_bench_summary_v1",
        "status": "evaluation_complete",
        "benchmark": sample_benchmark(samples),
        "prompt_profile": MOVETO_PROFILE,
        "action_contract": MODE_CONTRACTS["moveto_leftclick"]["action_contract"],
        "checkpoint": str(checkpoint.resolve()),
        "allowed_actions": ["move_to", "left_click"],
        "assistant_prefill": False,
        "think_present": False,
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_move_tos": MAX_MOVE_TOS,
        **_cursor_metadata(
            initial_cursor_map=initial_cursor_map,
            coordinate=None,
        ),
        "total": total,
        **counts,
        "accuracy": counts["correct"] / total if total else 0.0,
        "predictions": str(predictions),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {"prompt_profile": MOVETO_PROFILE, "action_contract": MODE_CONTRACTS["moveto_leftclick"]["action_contract"], "checkpoint": str(checkpoint.resolve()), "allowed_actions": ["move_to", "left_click"], "assistant_prefill": False, "think_present": False, "image_history_max": IMAGE_HISTORY_MAX, "max_move_tos": MAX_MOVE_TOS, "generation_settings": GENERATION_SETTINGS, **_cursor_metadata(initial_cursor_map=initial_cursor_map, coordinate=None)})
    return summary


def _run_terminate(
    samples: list[GroundingSample],
    *,
    generator: VLLMGenerator,
    output: Path,
    checkpoint: Path,
    initial_cursor_map: Path | None = None,
    cursor_positions: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    predictions = output / "predictions.jsonl"
    counts = {"correct": 0, "wrong": 0, "wrong_action": 0, "wrong_format": 0}
    # OSWorld-G stores image bytes in the benchmark table. Keep source images
    # and cursor-only observations in the container-local temporary directory;
    # only the JSON prediction stream is a durable artifact.
    with TemporaryDirectory(prefix="qwen3vl-terminate-observations-") as temporary_root:
        transient_output = Path(temporary_root)
        with predictions.open("x", encoding="utf-8") as handle:
            for index, sample in enumerate(samples):
                initial_coordinate = cursor_positions.get(sample.record_id) if cursor_positions else None
                source = _materialize_source(sample, transient_output, index)
                initial_image = (
                    _cursor_image(source, transient_output, index, 0, initial_coordinate)
                    if initial_coordinate is not None
                    else None
                )
                row = run_multiturn_rollout(
                    sample,
                    sample_index=index,
                    output_dir=transient_output,
                    generate_fn=generator,
                    max_left_clicks=MAX_LEFT_CLICKS,
                    source_path=source,
                    initial_image_path=initial_image,
                )
                row["source_image"] = str(sample.image_path.resolve()) if sample.image_path is not None else None
                row.update(_cursor_metadata(initial_cursor_map=initial_cursor_map, coordinate=initial_coordinate))
                counts[row["correctness"]] += 1
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    total = len(samples)
    summary = {
        "schema": "paired90_qwen3vl_five_bench_summary_v1",
        "status": "evaluation_complete",
        "benchmark": sample_benchmark(samples),
        "prompt_profile": TERMINATE_PROFILE,
        "action_contract": MODE_CONTRACTS["leftclick_terminate"]["action_contract"],
        "checkpoint": str(checkpoint.resolve()),
        "allowed_actions": ["left_click", "terminate"],
        "assistant_prefill": False,
        "think_present": False,
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_left_clicks": MAX_LEFT_CLICKS,
        **_cursor_metadata(
            initial_cursor_map=initial_cursor_map,
            coordinate=None,
        ),
        "total": total,
        **counts,
        "accuracy": counts["correct"] / total if total else 0.0,
        "predictions": str(predictions),
    }
    _write_json(output / "summary.json", summary)
    _write_json(output / "protocol.json", {"prompt_profile": TERMINATE_PROFILE, "action_contract": MODE_CONTRACTS["leftclick_terminate"]["action_contract"], "checkpoint": str(checkpoint.resolve()), "allowed_actions": ["left_click", "terminate"], "assistant_prefill": False, "think_present": False, "image_history_max": IMAGE_HISTORY_MAX, "max_left_clicks": MAX_LEFT_CLICKS, "generation_settings": GENERATION_SETTINGS, **_cursor_metadata(initial_cursor_map=initial_cursor_map, coordinate=None)})
    return summary


def sample_benchmark(samples: list[GroundingSample]) -> str:
    if not samples:
        raise ValueError("benchmark has no samples")
    return samples[0].benchmark


def preflight(
    *,
    mode: str,
    checkpoint: Path,
    data_root: Path,
    output_root: Path,
    training_manifest: Path,
    training_run_summary: Path,
    expected_training_records: int = TRAINING_RECORD_COUNT,
    benchmarks: tuple[str, ...] | None = None,
    initial_cursor_map: Path | None = None,
) -> dict[str, Any]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    selected_benchmarks = _resolve_benchmarks(benchmarks)
    output = _fresh(output_root)
    checkpoint_report = _checkpoint_contract(checkpoint)
    training_inference_contract = _training_inference_contract(
        mode=mode,
        checkpoint=checkpoint,
        training_manifest=training_manifest.expanduser().resolve(),
        training_run_summary=training_run_summary.expanduser().resolve(),
        expected_training_records=expected_training_records,
    )
    counts: dict[str, int] = {}
    samples_by_benchmark: dict[str, list[GroundingSample]] = {}
    for benchmark in selected_benchmarks:
        samples = load_benchmark_samples(benchmark, data_root, sample_limit=None)
        samples_by_benchmark[benchmark] = samples
        counts[benchmark] = len(samples)
        if mode == "moveto_leftclick":
            sample = samples[0]
            with TemporaryDirectory(prefix="qwen3vl-preflight-moveto-") as temporary_root:
                source = _materialize_source(sample, Path(temporary_root), 0)
                if initial_cursor_map is not None:
                    cursor_positions = _load_cursor_map(initial_cursor_map, {benchmark: samples})
                    source = _cursor_image(source, Path(temporary_root), 0, 0, cursor_positions[benchmark][sample.record_id])
                built = build_moveto_prompt(sample.instruction, image_paths=(source,), assistant_response_history=())
            if not built.prompt.endswith("<|im_start|>assistant\n"):
                raise ValueError("moveto prompt has assistant prefill")
            if "<think>" in built.prompt.lower() or "</think>" in built.prompt.lower():
                raise ValueError("moveto prompt contains Think text")
        elif mode == "leftclick_terminate":
            sample = samples[0]
            with TemporaryDirectory(prefix="qwen3vl-preflight-terminate-") as temporary_root:
                source = _materialize_source(sample, Path(temporary_root), 0)
                if initial_cursor_map is not None:
                    cursor_positions = _load_cursor_map(initial_cursor_map, {benchmark: samples})
                    source = _cursor_image(source, Path(temporary_root), 0, 0, cursor_positions[benchmark][sample.record_id])
                built = build_terminate_prompt(sample.instruction, image_paths=(source,), assistant_response_history=())
            if not built.prompt.endswith("<|im_start|>assistant\n") or "<think>" in built.prompt.lower():
                raise ValueError("terminate prompt contract failed")
    if mode == "directclick":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True
        )
        sample = samples_by_benchmark[selected_benchmarks[0]][0]
        direct_coordinate = None
        if initial_cursor_map is not None:
            cursor_positions = _load_cursor_map(initial_cursor_map, {selected_benchmarks[0]: samples_by_benchmark[selected_benchmarks[0]]})
            direct_coordinate = tuple(cursor_positions[selected_benchmarks[0]][sample.record_id])
        prompt, image = _direct_prompt(
            processor,
            sample,
            cursor_coordinate=direct_coordinate,
        )
        try:
            if "<think>" in prompt.lower() or not prompt.endswith("<|im_start|>assistant\n"):
                raise ValueError("direct prompt contract failed")
        finally:
            image.close()
    expected = {benchmark: EXPECTED_DATASET_COUNTS[benchmark] for benchmark in selected_benchmarks}
    if counts != expected:
        raise ValueError(f"benchmark counts mismatch: {counts} != {expected}")
    report = {
        "schema": "paired90_qwen3vl_five_bench_preflight_v1",
        "status": "preflight_passed",
        "mode": mode,
        "checkpoint": checkpoint_report,
        "training_inference_contract": training_inference_contract,
        "data_root": str(data_root.resolve()),
        "counts": counts,
        "benchmarks": list(selected_benchmarks),
        "assistant_prefill": False,
        "think_present": False,
        "generation_settings": GENERATION_SETTINGS,
        **_cursor_metadata(
            initial_cursor_map=initial_cursor_map,
            coordinate=None,
        ),
        "cursor_map_coverage": {benchmark: len(samples) for benchmark, samples in samples_by_benchmark.items()} if initial_cursor_map is not None else None,
        "cursor_map_source": str(initial_cursor_map.resolve()) if initial_cursor_map is not None else None,
    }
    _write_json(output / "preflight.json", report)
    _write_json(output / "training_inference_contract.json", training_inference_contract)
    return report


def run(
    *,
    mode: str,
    checkpoint: Path,
    data_root: Path,
    output_root: Path,
    training_manifest: Path,
    training_run_summary: Path,
    expected_training_records: int = TRAINING_RECORD_COUNT,
    tensor_parallel_size: int = 1,
    max_model_len: int = 65536,
    gpu_memory_utilization: float = 0.8,
    benchmarks: tuple[str, ...] | None = None,
    initial_cursor_map: Path | None = None,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    selected_benchmarks = _resolve_benchmarks(benchmarks)
    preflight_report = preflight(
        mode=mode,
        checkpoint=checkpoint,
        data_root=data_root,
        output_root=output_root / "_preflight",
        training_manifest=training_manifest,
        training_run_summary=training_run_summary,
        expected_training_records=expected_training_records,
        benchmarks=selected_benchmarks,
        initial_cursor_map=initial_cursor_map,
    )
    from transformers import AutoProcessor

    processor = (
        AutoProcessor.from_pretrained(
            str(checkpoint), trust_remote_code=True, local_files_only=True
        )
        if mode == "directclick"
        else None
    )
    generator = VLLMGenerator(
        checkpoint.resolve(),
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        image_history_max=preflight_report["training_inference_contract"]["image_history_max"],
        processor_contract=preflight_report["training_inference_contract"]["processor"],
    )
    results: dict[str, Any] = {}
    for benchmark in selected_benchmarks:
        samples = load_benchmark_samples(benchmark, data_root, sample_limit=None)
        output = _fresh(output_root / benchmark)
        if mode == "directclick":
            assert processor is not None
            cursor_positions = (
                _load_cursor_map(initial_cursor_map, {benchmark: samples})[benchmark]
                if initial_cursor_map is not None
                else None
            )
            results[benchmark] = _run_direct(
                samples,
                processor=processor,
                generator=generator,
                output=output,
                checkpoint=checkpoint,
                initial_cursor_map=initial_cursor_map,
                cursor_positions=cursor_positions,
            )
        elif mode == "moveto_leftclick":
            cursor_positions = (
                _load_cursor_map(initial_cursor_map, {benchmark: samples})[benchmark]
                if initial_cursor_map is not None
                else None
            )
            results[benchmark] = _run_moveto(
                samples,
                generator=generator,
                output=output,
                checkpoint=checkpoint,
                initial_cursor_map=initial_cursor_map,
                cursor_positions=cursor_positions,
            )
        else:
            cursor_positions = (
                _load_cursor_map(initial_cursor_map, {benchmark: samples})[benchmark]
                if initial_cursor_map is not None
                else None
            )
            results[benchmark] = _run_terminate(
                samples,
                generator=generator,
                output=output,
                checkpoint=checkpoint,
                initial_cursor_map=initial_cursor_map,
                cursor_positions=cursor_positions,
            )
    aggregate = {
        "schema": "paired90_qwen3vl_five_bench_aggregate_v1",
        "status": "evaluation_complete",
        "mode": mode,
        "checkpoint": str(checkpoint.resolve()),
        "preflight": preflight_report,
        "benchmarks": results,
    }
    _write_json(output_root / "summary.json", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=SUPPORTED_MODES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--training-run-summary", type=Path, required=True)
    parser.add_argument(
        "--expected-training-records",
        type=int,
        default=TRAINING_RECORD_COUNT,
    )
    parser.add_argument("--benchmark", dest="benchmarks", action="append", choices=SUPPORTED_BENCHMARKS)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument(
        "--initial-cursor-map",
        type=Path,
        help="Read-only JSON map of benchmark record IDs to training-style normalized initial cursor coordinates.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected_benchmarks = tuple(args.benchmarks) if args.benchmarks is not None else None
    initial_cursor_map = args.initial_cursor_map.expanduser().resolve() if args.initial_cursor_map else None
    if args.preflight_only:
        report = preflight(
            mode=args.mode,
            checkpoint=checkpoint,
            data_root=data_root,
            output_root=output_root,
            training_manifest=args.training_manifest.expanduser().resolve(),
            training_run_summary=args.training_run_summary.expanduser().resolve(),
            expected_training_records=args.expected_training_records,
            benchmarks=selected_benchmarks,
            initial_cursor_map=initial_cursor_map,
        )
    else:
        report = run(
            mode=args.mode,
            checkpoint=checkpoint,
            data_root=data_root,
            output_root=output_root,
            training_manifest=args.training_manifest.expanduser().resolve(),
            training_run_summary=args.training_run_summary.expanduser().resolve(),
            expected_training_records=args.expected_training_records,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            benchmarks=selected_benchmarks,
            initial_cursor_map=initial_cursor_map,
        )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
