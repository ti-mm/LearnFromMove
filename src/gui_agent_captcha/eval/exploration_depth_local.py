from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..actions import Action, AtomicAction, PrimitiveAction
from ..benchmarks.exploration_depth.contracts import FORMAL_SUITE_ID, episodes_for_variant
from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.runner import ExplorationTraceSession
from ..benchmarks.exploration_depth.service import RotationReplayServer
from ..benchmarks.exploration_depth.training_no_think import SIX_ACTION_KINDS
from ..core import StepResult
from latentguiworld.suite import build_evaluation_variant as build_benchmark_variant
from ..models.exploration_depth_no_think import (
    EXPLORATION_DEPTH_NO_THINK_EVAL_PROMPT_CONTRACT,
    Qwen35ExplorationDepthNoThinkLocalBackend,
)
from ..models.exploration_depth_with_think import (
    EXPLORATION_DEPTH_WITH_THINK_EVAL_PROMPT_CONTRACT,
    EXPLORATION_DEPTH_WITH_THINK_RESPONSE_CONTRACT,
    Qwen35ExplorationDepthWithThinkLocalBackend,
)
from ..models.qwen3_vl_sft_local import ModelActionParseError, Qwen3VLSftLocalBackend
from ..policies.move_observe_act import ClosedLoopPrimitivePolicy

DEFAULT_MAX_TURNS = 12
DEFAULT_IMAGE_MAX_PIXELS = 1280 * 720
DEFAULT_IMAGE_HISTORY_MAX = 3
ERROR_TERMINAL_REASONS = frozenset({"infra_error", "predict_error", "step_error"})
THINKING_MODES = ("no-think", "with-think")
SUPPORTED_FORMAL_SUITE_IDS = (FORMAL_SUITE_ID,)
FIRST_RELEASE_SCORING_POLICY = "first_real_release_attempt_success_only_no_correction"
TERMINAL_SUCCESS_REASONS = frozenset({"success", "first_release_success"})


def _is_release_action(action: Action) -> bool:
    if isinstance(action, AtomicAction):
        return action.kind in {"click", "drag"}
    return isinstance(action, PrimitiveAction) and action.kind in {"left_click", "mouse_up"}


def _validated_formal_suite_id(manifest: Mapping[str, Any]) -> str:
    suite_id = str(manifest.get("suite_id") or "")
    if suite_id not in SUPPORTED_FORMAL_SUITE_IDS or manifest.get("frozen") is not True:
        raise ExplorationDepthLocalEvalError(
            f"expected one of the frozen formal suites {SUPPORTED_FORMAL_SUITE_IDS}, "
            f"got {suite_id!r} frozen={manifest.get('frozen')!r}"
        )
    return suite_id


def _thinking_contract(thinking_mode: str) -> dict[str, object]:
    if thinking_mode == "no-think":
        return {
            "condition_name": "six_action_no_think",
            "prompt_contract": EXPLORATION_DEPTH_NO_THINK_EVAL_PROMPT_CONTRACT,
            "response_contract": "empty_think_tag_then_exact_action_v1",
            "enable_thinking": False,
        }
    if thinking_mode == "with-think":
        return {
            "condition_name": "six_action_with_think",
            "prompt_contract": EXPLORATION_DEPTH_WITH_THINK_EVAL_PROMPT_CONTRACT,
            "response_contract": EXPLORATION_DEPTH_WITH_THINK_RESPONSE_CONTRACT,
            "enable_thinking": True,
        }
    raise ExplorationDepthLocalEvalError(
        f"unsupported thinking mode {thinking_mode!r}; expected one of {THINKING_MODES}"
    )


class ExplorationDepthLocalEvalError(RuntimeError):
    """A local checkpoint evaluation artifact violates the formal contract."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    temporary.replace(path)


def _read_episode_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    episode_ids = [value for value in values if value]
    if len(episode_ids) != len(set(episode_ids)):
        raise ExplorationDepthLocalEvalError(f"duplicate episode ids in {path}")
    return episode_ids


def prepare_splits(
    *,
    manifest_path: Path,
    variant: str,
    split_root: Path,
    shard_count: int,
) -> dict[str, Any]:
    if shard_count < 1:
        raise ExplorationDepthLocalEvalError("shard_count must be positive")
    manifest = load_manifest(manifest_path)
    suite_id = _validated_formal_suite_id(manifest)
    episodes = episodes_for_variant(manifest, variant)
    if len(episodes) != 150:
        raise ExplorationDepthLocalEvalError(
            f"{variant}: expected 150 formal episodes, got {len(episodes)}"
        )
    episode_ids = [str(episode["episode_id"]) for episode in episodes]
    if len(set(episode_ids)) != 150:
        raise ExplorationDepthLocalEvalError(f"{variant}: episode ids are not unique")

    split_root.mkdir(parents=True, exist_ok=True)
    shard_sizes: list[int] = []
    for shard_index in range(shard_count):
        shard = episode_ids[shard_index::shard_count]
        shard_sizes.append(len(shard))
        (split_root / f"shard_{shard_index:03d}.txt").write_text(
            "".join(f"{episode_id}\n" for episode_id in shard),
            encoding="utf-8",
        )
    summary = {
        "suite_id": suite_id,
        "manifest_path": str(manifest_path),
        "variant": variant,
        "formal_episode_count": 150,
        "shard_count": shard_count,
        "shard_sizes": shard_sizes,
        "episode_ids": episode_ids,
    }
    _write_json(split_root / "split_summary.json", summary)
    return summary


def _record_path(output_root: Path, episode_id: str) -> Path:
    return output_root / "records" / f"{episode_id}.json"


def run_episode(
    *,
    backend: Qwen3VLSftLocalBackend,
    episode: dict[str, Any],
    manifest_path: Path,
    output_root: Path,
    checkpoint_step: int,
    rotation_base_url: str | None,
    max_turns: int,
    thinking_mode: str,
    strict_first_release: bool = False,
) -> dict[str, Any]:
    contract = _thinking_contract(thinking_mode)
    episode_id = str(episode["episode_id"])
    existing_path = _record_path(output_root, episode_id)
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if (
            existing.get("record_status") == "complete"
            and (existing.get("strict_first_release") is True) == strict_first_release
        ):
            return existing

    started_at = time.monotonic()
    case_dir = output_root / "cases" / episode_id
    environment: Any | None = None
    session: ExplorationTraceSession | None = None
    history: list[StepResult] = []
    terminal_reason = "max_turns"
    infra_error: dict[str, str] | None = None
    model_protocol_error: dict[str, str] | None = None
    prediction_calls_before = backend._call_index
    backend.call_log_dir = case_dir / "model_calls"

    try:
        kwargs: dict[str, Any] = {
            "manifest_path": manifest_path,
            "artifact_dir": case_dir / "frames",
        }
        if str(episode["variant"]).startswith("rotation_"):
            if not rotation_base_url:
                raise ExplorationDepthLocalEvalError("rotation replay server is unavailable")
            kwargs["base_url"] = rotation_base_url
        environment = build_benchmark_variant(str(episode["variant"]), **kwargs)
        session = ExplorationTraceSession(
            env=environment,
            episode=episode,
            output_dir=case_dir,
        )
        observation = session.reset()
        policy = ClosedLoopPrimitivePolicy(
            backend=backend,
            allowed_kinds=SIX_ACTION_KINDS,
            condition_name=str(contract["condition_name"]),
            budget=max_turns,
        )
        policy.reset()
        for _turn_index in range(max_turns):
            try:
                action = policy.predict(observation, history)
            except ModelActionParseError as error:
                terminal_reason = "model_protocol_error"
                model_protocol_error = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            except Exception as error:
                terminal_reason = "predict_error"
                infra_error = {"type": type(error).__name__, "message": str(error)}
                break
            try:
                result = session.step(action)
            except Exception as error:
                terminal_reason = "step_error"
                infra_error = {"type": type(error).__name__, "message": str(error)}
                break
            result.action = action
            if isinstance(backend.last_raw_prediction, str):
                result.info["model_raw_prediction"] = backend.last_raw_prediction
            if isinstance(backend.last_think_text, str) and backend.last_think_text:
                result.info["model_think_text"] = backend.last_think_text
            history.append(result)
            observation = result.observation
            if strict_first_release and result.done and _is_release_action(action):
                terminal_reason = (
                    "first_release_success"
                    if result.done and result.reward == 1.0
                    else "first_release_failure"
                )
                break
            if result.done:
                terminal_reason = "success" if result.reward == 1.0 else "failure"
                break
    except Exception as error:
        terminal_reason = "infra_error"
        infra_error = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if session is not None:
            trace_paths = session.write()
        else:
            trace_paths = {"trace_path": None, "audit_path": None}
        if environment is not None:
            environment.close()

    record = {
        "record_status": (
            "error" if terminal_reason in ERROR_TERMINAL_REASONS else "complete"
        ),
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "variant": episode["variant"],
        "exploration_level": episode["exploration_level"],
        "episode_id": episode_id,
        "case_seed": episode["case_seed"],
        "split": episode["split"],
        "checkpoint_path": str(backend.checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "success": terminal_reason in TERMINAL_SUCCESS_REASONS,
        "terminal_reason": terminal_reason,
        "scoring_policy": (
            FIRST_RELEASE_SCORING_POLICY
            if strict_first_release
            else "environment_terminal_with_correction_allowed"
        ),
        "strict_first_release": strict_first_release,
        "max_turns": max_turns,
        "executed_turns": len(history),
        "vlm_calls": backend._call_index - prediction_calls_before,
        "semantic_interactions": (
            session.semantic_interaction_index if session is not None else 0
        ),
        "allowed_action_kinds": list(SIX_ACTION_KINDS),
        "thinking_mode": thinking_mode,
        "prompt_contract": contract["prompt_contract"],
        "response_contract": contract["response_contract"],
        "history_strategy": (
            "latest_3_observation_images_and_all_exact_assistant_responses"
        ),
        "enable_thinking": contract["enable_thinking"],
        **trace_paths,
        "infra_error": infra_error,
        "model_protocol_error": model_protocol_error,
        "latency_s": time.monotonic() - started_at,
    }
    _write_json(existing_path, record)
    return record


def evaluate_shard(
    *,
    manifest_path: Path,
    variant: str,
    episode_ids_path: Path,
    checkpoint_path: Path,
    checkpoint_step: int,
    base_model: Path,
    output_root: Path,
    max_turns: int,
    max_new_tokens: int,
    max_length: int,
    thinking_mode: str = "no-think",
    strict_first_release: bool = False,
) -> dict[str, Any]:
    contract = _thinking_contract(thinking_mode)
    manifest = load_manifest(manifest_path)
    suite_id = _validated_formal_suite_id(manifest)
    episode_index = {
        str(episode["episode_id"]): episode
        for episode in episodes_for_variant(manifest, variant)
    }
    episode_ids = _read_episode_ids(episode_ids_path)
    missing = [episode_id for episode_id in episode_ids if episode_id not in episode_index]
    if missing:
        raise ExplorationDepthLocalEvalError(
            f"{variant}: shard contains unknown episodes: {missing[:3]}"
        )
    backend_class = (
        Qwen35ExplorationDepthWithThinkLocalBackend
        if thinking_mode == "with-think"
        else Qwen35ExplorationDepthNoThinkLocalBackend
    )
    backend = backend_class(
        checkpoint_path=checkpoint_path,
        processor_checkpoint_path=base_model,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        image_max_pixels=DEFAULT_IMAGE_MAX_PIXELS,
        image_history_max=DEFAULT_IMAGE_HISTORY_MAX,
        attn_implementation="flash_attention_2",
        enable_thinking=bool(contract["enable_thinking"]),
    )
    replay_context = (
        RotationReplayServer(public_root=manifest_path.parent)
        if variant.startswith("rotation_")
        else nullcontext(None)
    )
    records: list[dict[str, Any]] = []
    with replay_context as replay:
        rotation_base_url = replay.base_url if replay is not None else None
        for episode_id in episode_ids:
            record = run_episode(
                backend=backend,
                episode=episode_index[episode_id],
                manifest_path=manifest_path,
                output_root=output_root,
                checkpoint_step=checkpoint_step,
                rotation_base_url=rotation_base_url,
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
    error_records = [
        record
        for record in records
        if str(record["terminal_reason"]) in ERROR_TERMINAL_REASONS
    ]
    if error_records:
        first = error_records[0]
        raise ExplorationDepthLocalEvalError(
            f"{variant}: worker produced {len(error_records)} infrastructure records; "
            f"first={first['episode_id']} terminal={first['terminal_reason']} "
            f"error={first.get('infra_error')!r}"
        )
    return summary


def _load_step_records(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "shards").glob("shard_*/records/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("record_status") == "complete":
            records.append(payload)
    return records


def aggregate_step(
    *,
    manifest_path: Path,
    variant: str,
    output_root: Path,
    checkpoint_step: int,
    thinking_mode: str = "no-think",
    strict_first_release: bool = False,
) -> dict[str, Any]:
    contract = _thinking_contract(thinking_mode)
    manifest = load_manifest(manifest_path)
    suite_id = _validated_formal_suite_id(manifest)
    expected_ids = {
        str(episode["episode_id"])
        for episode in episodes_for_variant(manifest, variant)
    }
    records = _load_step_records(output_root)
    mismatched_modes = [
        record.get("thinking_mode", "no-think")
        for record in records
        if record.get("thinking_mode", "no-think") != thinking_mode
    ]
    if mismatched_modes:
        raise ExplorationDepthLocalEvalError(
            f"step {checkpoint_step} contains records for another thinking mode: "
            f"{sorted(set(str(value) for value in mismatched_modes))}"
        )
    mismatched_scoring = [
        record.get("episode_id")
        for record in records
        if (record.get("strict_first_release") is True) != strict_first_release
    ]
    if mismatched_scoring:
        raise ExplorationDepthLocalEvalError(
            f"step {checkpoint_step} contains records for another scoring policy: "
            f"{mismatched_scoring[:3]}"
        )
    error_records = [
        record
        for record in records
        if str(record.get("terminal_reason")) in ERROR_TERMINAL_REASONS
        or record.get("infra_error") is not None
    ]
    if error_records:
        first = error_records[0]
        raise ExplorationDepthLocalEvalError(
            f"step {checkpoint_step} contains {len(error_records)} infrastructure "
            f"records; first={first.get('episode_id')} "
            f"terminal={first.get('terminal_reason')}"
        )
    observed_ids = [str(record["episode_id"]) for record in records]
    duplicates = sorted(
        episode_id
        for episode_id, count in Counter(observed_ids).items()
        if count > 1
    )
    missing = sorted(expected_ids - set(observed_ids))
    unexpected = sorted(set(observed_ids) - expected_ids)
    if duplicates or missing or unexpected or len(records) != 150:
        raise ExplorationDepthLocalEvalError(
            f"step {checkpoint_step} aggregate incomplete: records={len(records)} "
            f"duplicates={duplicates[:3]} missing={missing[:3]} "
            f"unexpected={unexpected[:3]}"
        )
    success_count = sum(record["success"] is True for record in records)
    latencies = [float(record["latency_s"]) for record in records]
    summary = {
        "evaluation": "exploration_depth_single_variant_checkpoint_v1",
        "suite_id": suite_id,
        "manifest_path": str(manifest_path),
        "variant": variant,
        "checkpoint_step": checkpoint_step,
        "checkpoint_path": records[0]["checkpoint_path"],
        "expected_count": 150,
        "completed_count": 150,
        "missing_count": 0,
        "success_count": success_count,
        "success_rate": success_count / 150.0,
        "scoring_policy": (
            FIRST_RELEASE_SCORING_POLICY
            if strict_first_release
            else "environment_terminal_with_correction_allowed"
        ),
        "strict_first_release": strict_first_release,
        "first_release_attempt_count": sum(
            str(record["terminal_reason"]).startswith("first_release_")
            for record in records
        ),
        "no_release_count": sum(
            strict_first_release
            and not str(record["terminal_reason"]).startswith("first_release_")
            for record in records
        ),
        "terminal_counts": dict(
            sorted(Counter(str(record["terminal_reason"]) for record in records).items())
        ),
        "mean_executed_turns": statistics.fmean(
            int(record["executed_turns"]) for record in records
        ),
        "mean_latency_s": statistics.fmean(latencies),
        "allowed_action_kinds": list(SIX_ACTION_KINDS),
        "max_turns": DEFAULT_MAX_TURNS,
        "thinking_mode": thinking_mode,
        "enable_thinking": contract["enable_thinking"],
        "prompt_contract": contract["prompt_contract"],
        "response_contract": contract["response_contract"],
        "history_strategy": (
            "latest_3_observation_images_and_all_exact_assistant_responses"
        ),
    }
    sorted_records = sorted(records, key=lambda record: str(record["episode_id"]))
    _write_json(output_root / "summary.json", summary)
    _write_jsonl(output_root / "records.jsonl", sorted_records)
    _write_json(
        output_root / "evaluation_complete.json",
        {
            "status": "complete",
            "variant": variant,
            "checkpoint_step": checkpoint_step,
            "formal_episode_count": 150,
        },
    )
    return summary


def aggregate_sweep(
    *,
    run_root: Path,
    variant: str,
    steps: list[int],
) -> dict[str, Any]:
    if len(steps) != len(set(steps)):
        raise ExplorationDepthLocalEvalError("sweep steps must be unique")
    if steps == sorted(steps):
        checkpoint_direction = "ascending"
    elif steps == sorted(steps, reverse=True):
        checkpoint_direction = "descending"
    else:
        raise ExplorationDepthLocalEvalError(
            "sweep steps must be strictly ascending or descending"
        )
    checkpoints: list[dict[str, Any]] = []
    for step in steps:
        step_root = run_root / f"global_step_{step}"
        complete = json.loads(
            (step_root / "evaluation_complete.json").read_text(encoding="utf-8")
        )
        summary = json.loads((step_root / "summary.json").read_text(encoding="utf-8"))
        if complete.get("status") != "complete" or summary.get("completed_count") != 150:
            raise ExplorationDepthLocalEvalError(f"step {step} is not complete")
        checkpoints.append(
            {
                "checkpoint_step": step,
                "success_count": summary["success_count"],
                "success_rate": summary["success_rate"],
                "mean_executed_turns": summary["mean_executed_turns"],
            }
        )
    payload = {
        "evaluation": "exploration_depth_single_variant_checkpoint_sweep_v1",
        "variant": variant,
        "checkpoint_order": steps,
        "checkpoint_direction": checkpoint_direction,
        "checkpoint_count": len(steps),
        "formal_cases_per_checkpoint": 150,
        "checkpoints": checkpoints,
        "status": "complete",
    }
    _write_json(run_root / "checkpoint_sweep_summary.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--variant", required=True)
    prepare.add_argument("--split-root", type=Path, required=True)
    prepare.add_argument("--shard-count", type=int, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--variant", required=True)
    evaluate.add_argument("--episode-ids", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--checkpoint-step", type=int, required=True)
    evaluate.add_argument("--base-model", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    evaluate.add_argument("--max-new-tokens", type=int, default=128)
    evaluate.add_argument("--max-length", type=int, default=16384)
    evaluate.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="no-think",
    )
    evaluate.add_argument("--strict-first-release", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifest", type=Path, required=True)
    aggregate.add_argument("--variant", required=True)
    aggregate.add_argument("--output-root", type=Path, required=True)
    aggregate.add_argument("--checkpoint-step", type=int, required=True)
    aggregate.add_argument(
        "--thinking-mode",
        choices=THINKING_MODES,
        default="no-think",
    )
    aggregate.add_argument("--strict-first-release", action="store_true")

    sweep = subparsers.add_parser("aggregate-sweep")
    sweep.add_argument("--run-root", type=Path, required=True)
    sweep.add_argument("--variant", required=True)
    sweep.add_argument("--steps", type=int, nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        payload = prepare_splits(
            manifest_path=args.manifest,
            variant=args.variant,
            split_root=args.split_root,
            shard_count=args.shard_count,
        )
    elif args.command == "evaluate":
        if args.max_turns != DEFAULT_MAX_TURNS:
            raise ExplorationDepthLocalEvalError(
                f"formal evaluation requires max_turns={DEFAULT_MAX_TURNS}"
            )
        payload = evaluate_shard(
            manifest_path=args.manifest,
            variant=args.variant,
            episode_ids_path=args.episode_ids,
            checkpoint_path=args.checkpoint,
            checkpoint_step=args.checkpoint_step,
            base_model=args.base_model,
            output_root=args.output_root,
            max_turns=args.max_turns,
            max_new_tokens=args.max_new_tokens,
            max_length=args.max_length,
            thinking_mode=args.thinking_mode,
            strict_first_release=args.strict_first_release,
        )
    elif args.command == "aggregate":
        payload = aggregate_step(
            manifest_path=args.manifest,
            variant=args.variant,
            output_root=args.output_root,
            checkpoint_step=args.checkpoint_step,
            thinking_mode=args.thinking_mode,
            strict_first_release=args.strict_first_release,
        )
    else:
        payload = aggregate_sweep(
            run_root=args.run_root,
            variant=args.variant,
            steps=args.steps,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
