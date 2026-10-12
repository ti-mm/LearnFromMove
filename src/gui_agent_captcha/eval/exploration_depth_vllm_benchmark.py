from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..benchmarks.exploration_depth.contracts import episodes_for_variant
from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.service import RotationReplayServer
from ..models.exploration_depth_vllm import (
    Qwen35ExplorationDepthNoThinkVLLMBackend,
    Qwen35ExplorationDepthWithThinkVLLMBackend,
)
from .exploration_depth_local import (
    ERROR_TERMINAL_REASONS,
    THINKING_MODES,
    ExplorationDepthLocalEvalError,
    _read_episode_ids,
    _thinking_contract,
    _validated_formal_suite_id,
    _write_json,
    run_episode,
)


def prepare_subset(
    *,
    manifest_path: Path,
    variant: str,
    output_root: Path,
    case_count: int,
    transformers_workers: int,
    vllm_workers: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    suite_id = _validated_formal_suite_id(manifest)
    episodes = episodes_for_variant(manifest, variant)
    if case_count < 1 or case_count > len(episodes):
        raise ExplorationDepthLocalEvalError("case_count is outside the variant size")
    if transformers_workers < 1 or vllm_workers < 1:
        raise ExplorationDepthLocalEvalError("worker counts must be positive")

    if case_count == 1:
        selected = [episodes[0]]
    else:
        selected = [
            episodes[round(index * (len(episodes) - 1) / (case_count - 1))]
            for index in range(case_count)
        ]
    episode_ids = [str(episode["episode_id"]) for episode in selected]
    if len(episode_ids) != len(set(episode_ids)):
        raise ExplorationDepthLocalEvalError("benchmark selection contains duplicate ids")

    output_root.mkdir(parents=True, exist_ok=True)
    selected_path = output_root / "selected_episode_ids.txt"
    selected_path.write_text("".join(f"{item}\n" for item in episode_ids), encoding="utf-8")
    for phase, worker_count in (
        ("transformers", transformers_workers),
        ("vllm", vllm_workers),
    ):
        split_root = output_root / "splits" / phase
        split_root.mkdir(parents=True, exist_ok=True)
        for worker_index in range(worker_count):
            shard = episode_ids[worker_index::worker_count]
            (split_root / f"shard_{worker_index:03d}.txt").write_text(
                "".join(f"{item}\n" for item in shard), encoding="utf-8"
            )
    payload = {
        "benchmark": "exploration_depth_transformers_vs_sticky_vllm_v1",
        "suite_id": suite_id,
        "manifest_path": str(manifest_path),
        "variant": variant,
        "case_count": case_count,
        "transformers_workers": transformers_workers,
        "vllm_workers": vllm_workers,
        "episode_ids": episode_ids,
        "status": "prepared",
    }
    _write_json(output_root / "selection.json", payload)
    return payload


def evaluate_vllm_shard(
    *,
    manifest_path: Path,
    variant: str,
    episode_ids_path: Path,
    checkpoint_path: Path,
    checkpoint_step: int,
    base_model: Path,
    output_root: Path,
    server_base_url: str,
    served_model_name: str,
    max_turns: int,
    max_new_tokens: int,
    max_length: int,
    thinking_mode: str = "with-think",
    strict_first_release: bool = False,
) -> dict[str, Any]:
    contract = _thinking_contract(thinking_mode)
    manifest = load_manifest(manifest_path)
    episode_index = {
        str(episode["episode_id"]): episode
        for episode in episodes_for_variant(manifest, variant)
    }
    episode_ids = _read_episode_ids(episode_ids_path)
    missing = [item for item in episode_ids if item not in episode_index]
    if missing:
        raise ExplorationDepthLocalEvalError(f"unknown episode ids: {missing[:3]}")

    backend_class = (
        Qwen35ExplorationDepthWithThinkVLLMBackend
        if thinking_mode == "with-think"
        else Qwen35ExplorationDepthNoThinkVLLMBackend
    )
    backend = backend_class(
        checkpoint_path=checkpoint_path,
        processor_checkpoint_path=base_model,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        image_max_pixels=1280 * 720,
        image_history_max=3,
        enable_thinking=bool(contract["enable_thinking"]),
        server_base_url=server_base_url,
        served_model_name=served_model_name,
    )
    records: list[dict[str, Any]] = []
    replay_context = (
        RotationReplayServer(public_root=manifest_path.parent)
        if variant.startswith("rotation_")
        else nullcontext(None)
    )
    with replay_context as replay:
        for episode_id in episode_ids:
            backend.route_key = episode_id
            record = run_episode(
                backend=backend,
                episode=episode_index[episode_id],
                manifest_path=manifest_path,
                output_root=output_root,
                checkpoint_step=checkpoint_step,
                rotation_base_url=replay.base_url if replay is not None else None,
                max_turns=max_turns,
                thinking_mode=thinking_mode,
                strict_first_release=strict_first_release,
            )
            records.append(record)
            print(
                f"episode={episode_id} success={record['success']} "
                f"terminal={record['terminal_reason']} turns={record['executed_turns']}",
                flush=True,
            )
    summary = {
        "benchmark": "exploration_depth_sticky_vllm_worker_v1",
        "variant": variant,
        "checkpoint_step": checkpoint_step,
        "thinking_mode": thinking_mode,
        "selected_count": len(episode_ids),
        "completed_count": len(records),
        "success_count": sum(record["success"] is True for record in records),
        "terminal_counts": dict(
            sorted(Counter(str(record["terminal_reason"]) for record in records).items())
        ),
    }
    _write_json(output_root / "worker_summary.json", summary)
    errors = [
        record
        for record in records
        if str(record["terminal_reason"]) in ERROR_TERMINAL_REASONS
    ]
    if errors:
        raise ExplorationDepthLocalEvalError(
            f"vLLM worker produced {len(errors)} infrastructure records"
        )
    return summary


def summarize_phase(
    *,
    phase: str,
    phase_root: Path,
    selected_episode_ids_path: Path,
    wall_seconds: float,
    output_path: Path,
) -> dict[str, Any]:
    expected_ids = set(_read_episode_ids(selected_episode_ids_path))
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(phase_root.glob("shards/shard_*/records/*.json"))
    ]
    observed_ids = [str(record["episode_id"]) for record in records]
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    errors = [
        record
        for record in records
        if record.get("record_status") != "complete"
        or str(record.get("terminal_reason")) in ERROR_TERMINAL_REASONS
    ]
    if missing or unexpected or len(records) != len(expected_ids) or errors:
        raise ExplorationDepthLocalEvalError(
            f"{phase} benchmark phase incomplete: records={len(records)} "
            f"missing={missing[:3]} unexpected={unexpected[:3]} errors={len(errors)}"
        )
    vlm_calls = sum(int(record["vlm_calls"]) for record in records)
    latencies = [float(record["latency_s"]) for record in records]
    payload = {
        "benchmark": "exploration_depth_backend_speed_phase_v1",
        "phase": phase,
        "case_count": len(records),
        "success_count": sum(record["success"] is True for record in records),
        "terminal_counts": dict(
            sorted(Counter(str(record["terminal_reason"]) for record in records).items())
        ),
        "vlm_calls": vlm_calls,
        "wall_seconds": wall_seconds,
        "cases_per_hour": len(records) / wall_seconds * 3600.0,
        "vlm_calls_per_second": vlm_calls / wall_seconds,
        "mean_episode_latency_s": statistics.fmean(latencies),
        "status": "complete",
    }
    _write_json(output_path, payload)
    return payload


def compare_phases(
    *,
    transformers_summary_path: Path,
    vllm_summary_path: Path,
    router_stats_path: Path,
    vllm_startup_seconds: float,
    output_path: Path,
) -> dict[str, Any]:
    transformers_summary = json.loads(
        transformers_summary_path.read_text(encoding="utf-8")
    )
    vllm_summary = json.loads(vllm_summary_path.read_text(encoding="utf-8"))
    router_stats = json.loads(router_stats_path.read_text(encoding="utf-8"))
    for name, summary in (
        ("transformers", transformers_summary),
        ("vllm", vllm_summary),
    ):
        if summary.get("status") != "complete" or summary.get("phase") != name:
            raise ExplorationDepthLocalEvalError(f"{name} phase summary is incomplete")
    if transformers_summary["case_count"] != vllm_summary["case_count"]:
        raise ExplorationDepthLocalEvalError("phase case counts do not match")
    vllm_cold_total_seconds = (
        vllm_startup_seconds + vllm_summary["wall_seconds"]
    )
    vllm_cold_cases_per_hour = (
        vllm_summary["case_count"] / vllm_cold_total_seconds * 3600.0
    )
    vllm_cold_calls_per_second = (
        vllm_summary["vlm_calls"] / vllm_cold_total_seconds
    )
    payload = {
        "benchmark": "exploration_depth_transformers_vs_sticky_vllm_v1",
        "case_count": transformers_summary["case_count"],
        "transformers": transformers_summary,
        "sticky_vllm": vllm_summary,
        "speedup": {
            "cases_per_hour": (
                vllm_summary["cases_per_hour"]
                / transformers_summary["cases_per_hour"]
            ),
            "vlm_calls_per_second": (
                vllm_summary["vlm_calls_per_second"]
                / transformers_summary["vlm_calls_per_second"]
            ),
            "warm_phase_wall_time": (
                transformers_summary["wall_seconds"] / vllm_summary["wall_seconds"]
            ),
            "cold_cases_per_hour": (
                vllm_cold_cases_per_hour
                / transformers_summary["cases_per_hour"]
            ),
            "cold_vlm_calls_per_second": (
                vllm_cold_calls_per_second
                / transformers_summary["vlm_calls_per_second"]
            ),
        },
        "vllm_server_startup_seconds": vllm_startup_seconds,
        "vllm_cold_total_seconds": vllm_cold_total_seconds,
        "vllm_cold_cases_per_hour": vllm_cold_cases_per_hour,
        "vllm_cold_vlm_calls_per_second": vllm_cold_calls_per_second,
        "router_stats": router_stats,
        "status": "complete",
    }
    _write_json(output_path, payload)
    return payload


def _load_complete_formal_result(
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    completion = json.loads((root / "evaluation_complete.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("shards/shard_*/records/*.json"))
    ]
    record_index = {str(record["episode_id"]): record for record in records}
    expected_count = int(summary.get("expected_count", 0))
    errors = [
        record
        for record in records
        if record.get("record_status") != "complete"
        or record.get("infra_error") is not None
        or str(record.get("terminal_reason")) in ERROR_TERMINAL_REASONS
    ]
    if (
        completion.get("status") != "complete"
        or summary.get("completed_count") != expected_count
        or summary.get("missing_count") != 0
        or len(records) != expected_count
        or len(record_index) != expected_count
        or errors
    ):
        raise ExplorationDepthLocalEvalError(
            f"formal result is incomplete: root={root} records={len(records)} "
            f"expected={expected_count} errors={len(errors)}"
        )
    return summary, record_index


def compare_formal_results(
    *, baseline_root: Path, candidate_root: Path, output_path: Path
) -> dict[str, Any]:
    baseline_summary, baseline_records = _load_complete_formal_result(baseline_root)
    candidate_summary, candidate_records = _load_complete_formal_result(candidate_root)
    contract_keys = (
        "suite_id",
        "variant",
        "checkpoint_step",
        "checkpoint_path",
        "expected_count",
        "max_turns",
        "thinking_mode",
        "enable_thinking",
        "prompt_contract",
        "response_contract",
        "history_strategy",
    )
    mismatches = {
        key: {
            "baseline": baseline_summary.get(key),
            "candidate": candidate_summary.get(key),
        }
        for key in contract_keys
        if baseline_summary.get(key) != candidate_summary.get(key)
    }
    if mismatches:
        raise ExplorationDepthLocalEvalError(
            f"formal result contracts do not match: {mismatches}"
        )
    if baseline_records.keys() != candidate_records.keys():
        raise ExplorationDepthLocalEvalError("formal result episode sets do not match")

    episode_ids = sorted(baseline_records)
    baseline_success = {
        episode_id
        for episode_id in episode_ids
        if baseline_records[episode_id]["success"] is True
    }
    candidate_success = {
        episode_id
        for episode_id in episode_ids
        if candidate_records[episode_id]["success"] is True
    }
    success_agreement = [
        episode_id
        for episode_id in episode_ids
        if baseline_records[episode_id]["success"]
        == candidate_records[episode_id]["success"]
    ]
    terminal_agreement = [
        episode_id
        for episode_id in episode_ids
        if baseline_records[episode_id]["terminal_reason"]
        == candidate_records[episode_id]["terminal_reason"]
    ]
    baseline_only = sorted(baseline_success - candidate_success)
    candidate_only = sorted(candidate_success - baseline_success)
    expected_count = len(episode_ids)
    payload = {
        "comparison": "exploration_depth_transformers_vs_sticky_vllm_formal_v1",
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "contract": {key: baseline_summary.get(key) for key in contract_keys},
        "baseline": {
            "success_count": len(baseline_success),
            "success_rate": len(baseline_success) / expected_count,
            "vlm_calls": sum(
                int(record["vlm_calls"]) for record in baseline_records.values()
            ),
        },
        "candidate": {
            "success_count": len(candidate_success),
            "success_rate": len(candidate_success) / expected_count,
            "vlm_calls": sum(
                int(record["vlm_calls"]) for record in candidate_records.values()
            ),
        },
        "success_rate_delta": (
            len(candidate_success) - len(baseline_success)
        )
        / expected_count,
        "success_agreement_count": len(success_agreement),
        "success_agreement_rate": len(success_agreement) / expected_count,
        "terminal_agreement_count": len(terminal_agreement),
        "terminal_agreement_rate": len(terminal_agreement) / expected_count,
        "both_success_count": len(baseline_success & candidate_success),
        "both_failure_count": expected_count - len(baseline_success | candidate_success),
        "baseline_only_success_count": len(baseline_only),
        "baseline_only_success_episode_ids": baseline_only,
        "candidate_only_success_count": len(candidate_only),
        "candidate_only_success_episode_ids": candidate_only,
        "exact_score_reproduction": len(baseline_success) == len(candidate_success),
        "exact_episode_reproduction": baseline_success == candidate_success,
        "status": "complete",
    }
    _write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--variant", required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--case-count", type=int, required=True)
    prepare.add_argument("--transformers-workers", type=int, required=True)
    prepare.add_argument("--vllm-workers", type=int, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--episode-ids", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-step", type=int, required=True)
    evaluate.add_argument("--base-model", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--server-base-url", required=True)
    evaluate.add_argument("--served-model-name", required=True)
    evaluate.add_argument("--max-turns", type=int, default=12)
    evaluate.add_argument("--max-new-tokens", type=int, default=512)
    evaluate.add_argument("--max-length", type=int, default=16384)
    evaluate.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="with-think",
    )
    evaluate.add_argument("--strict-first-release", action="store_true")

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--phase", choices=("transformers", "vllm"), required=True)
    summarize.add_argument("--phase-root", type=Path, required=True)
    summarize.add_argument("--selected-episode-ids", type=Path, required=True)
    summarize.add_argument("--wall-seconds", type=float, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--transformers-summary", type=Path, required=True)
    compare.add_argument("--vllm-summary", type=Path, required=True)
    compare.add_argument("--router-stats", type=Path, required=True)
    compare.add_argument("--vllm-startup-seconds", type=float, required=True)
    compare.add_argument("--output", type=Path, required=True)

    compare_formal = subparsers.add_parser("compare-formal")
    compare_formal.add_argument("--baseline-root", type=Path, required=True)
    compare_formal.add_argument("--candidate-root", type=Path, required=True)
    compare_formal.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_subset(
            manifest_path=args.manifest,
            variant=args.variant,
            output_root=args.output_root,
            case_count=args.case_count,
            transformers_workers=args.transformers_workers,
            vllm_workers=args.vllm_workers,
        )
    elif args.command == "evaluate":
        result = evaluate_vllm_shard(
            manifest_path=args.manifest,
            variant=args.variant,
            episode_ids_path=args.episode_ids,
            checkpoint_path=args.checkpoint,
            checkpoint_step=args.checkpoint_step,
            base_model=args.base_model,
            output_root=args.output_root,
            server_base_url=args.server_base_url,
            served_model_name=args.served_model_name,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            thinking_mode=args.thinking_mode,
            strict_first_release=args.strict_first_release,
        )
    elif args.command == "summarize":
        result = summarize_phase(
            phase=args.phase,
            phase_root=args.phase_root,
            selected_episode_ids_path=args.selected_episode_ids,
            wall_seconds=args.wall_seconds,
            output_path=args.output,
        )
    elif args.command == "compare":
        result = compare_phases(
            transformers_summary_path=args.transformers_summary,
            vllm_summary_path=args.vllm_summary,
            router_stats_path=args.router_stats,
            vllm_startup_seconds=args.vllm_startup_seconds,
            output_path=args.output,
        )
    else:
        result = compare_formal_results(
            baseline_root=args.baseline_root,
            candidate_root=args.candidate_root,
            output_path=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
