from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Any, Mapping

from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.contracts import PAPER_VARIANTS, runtime_variant
from ..benchmarks.exploration_depth.service import RotationReplayServer
from ..core import Observation, StepResult
from latentguiworld.suite import build_evaluation_variant as build_benchmark_variant
from ..envs.openai_computer import execute_openai_computer_action
from ..models.frontier_computer_use import FrontierModelConfig
from ..models.openai_computer_mouse import (
    COMPUTER_MOUSE_PROTOCOL,
    MOUSE_FUNCTION_NAMES,
    TASK_INSTRUCTION_CONTRACT,
    OpenAIComputerMouseBackend,
    computer_mouse_tools,
)
from ..models.openai_native_computer import (
    OPENAI_COMPUTER_ACTION_KINDS,
    OPENAI_NATIVE_HISTORY_STRATEGY,
    OpenAINativeAPIError,
    OpenAINativeComputerConfig,
    OpenAIResponseBudgetExhausted,
)
from .api_common import (
    VARIANTS,
    EvaluationTask,
    _action_dict,
    _case_dir,
    _json_error,
    _log,
    _meaningful_delta,
    _record_path,
    _validate_observation,
    _write_json,
    _write_jsonl,
    load_records,
    select_tasks,
)
from .api_common import (
    SCORING_POLICY,
    SYSTEM_PROMPT,
    _terminal_score,
)

MODEL_KEY = "gpt56_sol"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_MAX_TURNS = 12
DEFAULT_LIMIT_PER_VARIANT = 150
DEFAULT_WORKERS = 2
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_RETRIES = 6
PROTOCOL_TRACK = COMPUTER_MOUSE_PROTOCOL
HISTORY_STRATEGY = OPENAI_NATIVE_HISTORY_STRATEGY
TRANSPORT = "openai_responses_computer_and_mouse_functions"
EVALUATION_ID = "exploration_depth_gpt56_sol_computer_mouse_functions_first_release"
NATIVE_TOOL_TYPE = "computer"
NATIVE_BATCH_EXECUTION = "computer_call_actions_array_sequential"


def resolve_config(env: Mapping[str, str]) -> FrontierModelConfig:
    return FrontierModelConfig.from_preset(
        MODEL_KEY,
        env=env,
        model=str(env.get("GPT56_SOL_MODEL", "")).strip() or None,
        base_url=str(env.get("GPT56_SOL_BASE_URL", "")).strip() or None,
    )


def _fresh_observation(environment: object, *, instruction: str) -> Observation:
    refresh = getattr(environment, "refresh_observation", None)
    if not callable(refresh):
        raise RuntimeError("environment cannot capture a fresh observation")
    raw = refresh()
    if not isinstance(raw, Observation):
        raise RuntimeError("environment returned an invalid fresh observation")
    observation = Observation(
        instruction=instruction,
        screenshot_path=raw.screenshot_path,
        size_px=raw.size_px,
        cursor_xy=raw.cursor_xy,
        metadata=raw.metadata,
    )
    _validate_observation(observation.screenshot_path, observation.size_px)
    return observation


def _record_matches(record: Mapping[str, Any], config: FrontierModelConfig) -> bool:
    return (
        record.get("record_status") == "complete"
        and record.get("provider_key") == MODEL_KEY
        and record.get("model") == config.model
        and record.get("base_url") == config.base_url
        and record.get("protocol_track") == PROTOCOL_TRACK
        and record.get("max_responses") == DEFAULT_MAX_TURNS
        and record.get("system_prompt") == SYSTEM_PROMPT
        and record.get("task_instruction_contract") == TASK_INSTRUCTION_CONTRACT
        and record.get("tool_declarations") == computer_mouse_tools()
        and record.get("scoring_policy") == SCORING_POLICY
    )


def _load_selected_records(
    output_root: Path,
    *,
    config: FrontierModelConfig,
    episode_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for row in load_records(output_root)
        if _record_matches(row, config) and str(row.get("episode_id")) in episode_ids
    ]


def run_task(
    task: EvaluationTask,
    *,
    manifest_path: Path,
    rotation_base_url: str,
    output_root: Path,
    config: FrontierModelConfig,
    max_turns: int,
    timeout_s: float,
    retries: int,
    prompt_extension: str | None = None,
    harness_source: str | None = None,
    max_responses: int | None = 12,
) -> dict[str, Any]:
    started_at = time.monotonic()
    episode = task.episode
    case_dir = _case_dir(output_root, task)
    kwargs: dict[str, Any] = {
        "manifest_path": manifest_path,
        "artifact_dir": case_dir / "frames",
    }
    if task.variant.startswith("rotation_"):
        kwargs["base_url"] = rotation_base_url
    environment = build_benchmark_variant(task.variant, **kwargs)

    def observation_supplier() -> Observation:
        return _fresh_observation(
            environment,
            instruction=str(episode["instruction"]),
        )

    backend = OpenAIComputerMouseBackend(
        config=OpenAINativeComputerConfig(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
        ),
        system_prompt=SYSTEM_PROMPT,
        prompt_extension=prompt_extension,
        call_log_dir=case_dir / "model_calls",
        request_timeout_s=timeout_s,
        request_retries=retries,
        observation_supplier=observation_supplier,
        max_responses=max_responses,
    )
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    history: list[StepResult] = []
    terminal_reason = "max_turns"
    semantic_interactions = 0
    infra_error: dict[str, str] | None = None

    try:
        observation = environment.reset(task_id=task.episode_id)
        _validate_observation(observation.screenshot_path, observation.size_px)
        get_audit = getattr(environment, "get_evaluator_audit", None)
        audit_rows.append(
            {
                "frame_index": 0,
                "semantic_interaction_index": 0,
                "evaluator_only": get_audit() if callable(get_audit) else {},
            }
        )
        for turn_index in (count() if max_responses is not None else range(max_turns)):
            before_path = observation.screenshot_path
            try:
                action = backend.predict_action(
                    observation,
                    history,
                    budget=max_turns,
                    condition=task.variant,
                )
            except OpenAIResponseBudgetExhausted:
                terminal_reason = "max_responses"
                break
            except OpenAINativeAPIError as error:
                terminal_reason = "infra_error"
                infra_error = _json_error(error)
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": None,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "predict_error": _json_error(error),
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break
            except Exception as error:
                terminal_reason = "predict_error"
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": None,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "predict_error": _json_error(error),
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break

            if action is None:
                terminal_reason = "model_end_turn"
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": None,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "model_raw_prediction": backend.last_raw_prediction,
                        "model_native_action": backend.last_native_action_json,
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break

            native_action = _action_dict(action)  # type: ignore[arg-type]
            try:
                result = execute_openai_computer_action(
                    environment,
                    action,
                    observation,
                )
            except Exception as error:
                trace_rows.append(
                    {
                        "frame_index": turn_index,
                        "semantic_interaction_index": semantic_interactions,
                        "action": native_action,
                        "observation_before_path": before_path,
                        "observation_after_path": before_path,
                        "terminal_success": False,
                        "step_error": _json_error(error),
                        "model_raw_prediction": backend.last_raw_prediction,
                        "model_native_action": backend.last_native_action_json,
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                history.append(
                    StepResult(
                        observation=observation,
                        reward=None,
                        done=False,
                        info={"step_error": _json_error(error), "state_delta": {}},
                        action=action,  # type: ignore[arg-type]
                    )
                )
                if action.tool_type == "function":
                    terminal_reason = "step_error"
                    break
                continue

            state_delta = dict(result.info.get("state_delta", {}))
            if not result.done and _meaningful_delta(state_delta):
                semantic_interactions += 1
            success = bool(result.done and result.reward == 1.0)
            trace_rows.append(
                {
                    "frame_index": turn_index + 1,
                    "semantic_interaction_index": semantic_interactions,
                    "action": native_action,
                    "environment_actions": result.info.get("environment_actions"),
                    "environment_effect": result.info.get("environment_effect"),
                    "observation_before_path": before_path,
                    "observation_after_path": result.observation.screenshot_path,
                    "environment_state_delta": state_delta,
                    "terminal_success": success,
                    "interactions_to_success": semantic_interactions if success else None,
                    "corrective_interaction_count": max(semantic_interactions - 1, 0),
                    "model_raw_prediction": backend.last_raw_prediction,
                    "model_native_action": backend.last_native_action_json,
                    "model_native_action_batch": backend.last_native_action_batch_json,
                    "model_tool_type": action.tool_type,
                    "model_call_log_path": backend.last_call_log_path,
                }
            )
            history.append(result)
            observation = result.observation
            get_audit = getattr(environment, "get_evaluator_audit", None)
            audit_rows.append(
                {
                    "frame_index": turn_index + 1,
                    "semantic_interaction_index": semantic_interactions,
                    "evaluator_only": get_audit() if callable(get_audit) else {},
                }
            )
            if result.done:
                terminal_reason = "success" if success else "failure"
                break
    except Exception as error:
        terminal_reason = "infra_error"
        infra_error = _json_error(error)
        infra_error["traceback"] = traceback.format_exc()
    finally:
        model_responses = backend.response_count
        backend.close()
        environment.close()

    trace_path = case_dir / "trace.jsonl"
    audit_path = case_dir / "audit.jsonl"
    _write_jsonl(trace_path, trace_rows)
    _write_jsonl(audit_path, audit_rows)
    terminal_score = _terminal_score(
        terminal_reason,
        audit_rows[-1]["evaluator_only"] if audit_rows else {},
    )
    success = bool(terminal_score["success"])
    action_counts = Counter(
        str(row["action"].get("type", row["action"].get("kind")))
        for row in trace_rows
        if isinstance(row.get("action"), dict)
    )
    record = {
        "record_status": "complete",
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "variant": task.variant,
        "exploration_level": episode["exploration_level"],
        "episode_id": task.episode_id,
        "case_seed": episode["case_seed"],
        "split": episode["split"],
        "provider_key": MODEL_KEY,
        "model": config.model,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "system_prompt": SYSTEM_PROMPT,
        "harness_prompt_injected": bool(prompt_extension),
        "harness_source": harness_source,
        "task_instruction_contract": TASK_INSTRUCTION_CONTRACT,
        "tool_declarations": computer_mouse_tools(),
        "function_tool_names": list(MOUSE_FUNCTION_NAMES),
        "action_budget_disclosed_to_model": False,
        "initial_observation_delivery": "model_requested_screenshot_tool",
        "scoring_policy": SCORING_POLICY,
        "success": success,
        "success_source": terminal_score["success_source"],
        "committed_success": terminal_score["committed_success"],
        "cutoff_final_state_scored": terminal_score["cutoff_final_state_scored"],
        "cutoff_final_state_success": terminal_score["cutoff_final_state_success"],
        "final_state_evaluation": terminal_score["final_state_evaluation"],
        "terminal_reason": terminal_reason,
        "max_turns": max_turns if max_responses is None else None,
        "max_responses": max_responses,
        "model_responses": model_responses,
        "budget_unit": "model_responses" if max_responses is not None else "actions",
        "executed_turns": len(history),
        "api_calls": backend._call_index,
        "vlm_calls": backend._call_index,
        "semantic_interactions": semantic_interactions,
        "interactions_to_success": semantic_interactions if success else None,
        "corrective_interaction_count": max(semantic_interactions - 1, 0),
        "action_counts": dict(sorted(action_counts.items())),
        "native_action_kinds": list(OPENAI_COMPUTER_ACTION_KINDS),
        "native_tool_declaration": {"type": NATIVE_TOOL_TYPE},
        "native_batch_execution": NATIVE_BATCH_EXECUTION,
        "native_coordinate_format": "official_full_screenshot_pixels",
        "history_strategy": HISTORY_STRATEGY,
        "trace_path": str(trace_path),
        "audit_path": str(audit_path),
        "infra_error": infra_error,
        "api_key_persisted": False,
        "latency_s": time.monotonic() - started_at,
    }
    _write_json(_record_path(output_root, task), record)
    return record


def summarize(
    records: list[dict[str, Any]],
    *,
    selected_counts: Mapping[str, int],
    selected_variants: tuple[str, ...] = VARIANTS,
    output_root: Path,
    config: FrontierModelConfig,
    max_turns: int,
) -> dict[str, Any]:
    per_variant: dict[str, Any] = {}
    for variant in selected_variants:
        subset = [row for row in records if row.get("variant") == variant]
        successes = sum(row.get("success") is True for row in subset)
        turns = [int(row.get("executed_turns", 0)) for row in subset]
        calls = [int(row.get("api_calls", 0)) for row in subset]
        action_counts: Counter[str] = Counter()
        for row in subset:
            action_counts.update(row.get("action_counts", {}))
        per_variant[variant] = {
            "expected_count": int(selected_counts.get(variant, 0)),
            "completed_count": len(subset),
            "success_count": successes,
            "success_rate": successes / len(subset) if subset else 0.0,
            "committed_success_count": sum(
                row.get("committed_success") is True for row in subset
            ),
            "cutoff_final_state_success_count": sum(
                row.get("cutoff_final_state_success") is True for row in subset
            ),
            "mean_executed_turns": statistics.fmean(turns) if turns else None,
            "mean_api_calls": statistics.fmean(calls) if calls else None,
            "terminal_counts": dict(
                sorted(Counter(str(row.get("terminal_reason")) for row in subset).items())
            ),
            "action_counts": dict(sorted(action_counts.items())),
            "split_counts": dict(
                sorted(Counter(str(row.get("split")) for row in subset).items())
            ),
            "episode_ids": sorted(str(row["episode_id"]) for row in subset),
        }
    expected_total = sum(int(value) for value in selected_counts.values())
    successes = sum(row.get("success") is True for row in records)
    return {
        "evaluation": EVALUATION_ID,
        "provider_key": MODEL_KEY,
        "model": config.model,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "system_prompt": SYSTEM_PROMPT,
        "task_instruction_contract": TASK_INSTRUCTION_CONTRACT,
        "tool_declarations": computer_mouse_tools(),
        "function_tool_names": list(MOUSE_FUNCTION_NAMES),
        "action_budget_disclosed_to_model": False,
        "initial_observation_delivery": "model_requested_screenshot_tool",
        "scoring_policy": SCORING_POLICY,
        "output_root": str(output_root),
        "native_action_kinds": list(OPENAI_COMPUTER_ACTION_KINDS),
        "native_tool_type": NATIVE_TOOL_TYPE,
        "native_batch_execution": NATIVE_BATCH_EXECUTION,
        "native_coordinate_format": "official_full_screenshot_pixels",
        "max_turns": None,
        "max_responses": DEFAULT_MAX_TURNS,
        "budget_unit": "model_responses",
        "selected_variants": list(selected_variants),
        "history_strategy": HISTORY_STRATEGY,
        "expected_count": expected_total,
        "completed_count": len(records),
        "missing_count": max(0, expected_total - len(records)),
        "success_count": successes,
        "success_rate": successes / len(records) if records else 0.0,
        "committed_success_count": sum(
            row.get("committed_success") is True for row in records
        ),
        "cutoff_final_state_success_count": sum(
            row.get("cutoff_final_state_success") is True for row in records
        ),
        "api_key_persisted": False,
        "variants": per_variant,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate exploration-depth variants with GPT-5.6 Sol native computer use."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-responses", type=int, default=12, choices=[12])
    parser.add_argument("--limit-per-variant", type=int, default=DEFAULT_LIMIT_PER_VARIANT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(PAPER_VARIANTS) + VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-infra-errors", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_turns != DEFAULT_MAX_TURNS:
        raise SystemExit(f"formal evaluation requires --max-turns {DEFAULT_MAX_TURNS}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    try:
        config = resolve_config(os.environ)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(manifest_path)
    tasks, selected_pair_ids = select_tasks(
        manifest,
        limit_per_variant=args.limit_per_variant,
    )
    selected_variants = tuple(dict.fromkeys(runtime_variant(v) for v in args.variants))
    selected_variant_set = set(selected_variants)
    tasks = [task for task in tasks if task.variant in selected_variant_set]
    selected_episode_ids = {task.episode_id for task in tasks}
    selected_families = {str(task.episode["family"]) for task in tasks}
    selected_pair_ids = {
        family: pair_ids
        for family, pair_ids in selected_pair_ids.items()
        if family in selected_families
    }
    output_root.mkdir(parents=True, exist_ok=True)
    selected_counts = Counter(task.variant for task in tasks)
    run_manifest = {
        "suite_manifest": str(manifest_path),
        "suite_id": manifest["suite_id"],
        "provider_key": MODEL_KEY,
        "model": config.model,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "system_prompt": SYSTEM_PROMPT,
        "task_instruction_contract": TASK_INSTRUCTION_CONTRACT,
        "tool_declarations": computer_mouse_tools(),
        "function_tool_names": list(MOUSE_FUNCTION_NAMES),
        "action_budget_disclosed_to_model": False,
        "initial_observation_delivery": "model_requested_screenshot_tool",
        "scoring_policy": SCORING_POLICY,
        "selected_pair_ids": selected_pair_ids,
        "selected_variants": list(selected_variants),
        "selected_counts": dict(sorted(selected_counts.items())),
        "expected_count": len(tasks),
        "native_action_kinds": list(OPENAI_COMPUTER_ACTION_KINDS),
        "native_tool_declaration_template": {"type": NATIVE_TOOL_TYPE},
        "native_batch_execution": NATIVE_BATCH_EXECUTION,
        "native_coordinate_format": "official_full_screenshot_pixels",
        "history_strategy": HISTORY_STRATEGY,
        "max_turns": None,
        "max_responses": args.max_responses,
        "budget_unit": "model_responses",
        "workers": args.workers,
        "timeout_s": args.timeout_s,
        "retries": args.retries,
        "api_key_persisted": False,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _log("main", f"tasks={len(tasks)} workers={args.workers} output={output_root}")

    with RotationReplayServer(public_root=manifest_path.parent) as replay:

        def submit(task: EvaluationTask) -> dict[str, Any]:
            tag = f"{task.variant}/{task.episode_id}"
            record_path = _record_path(output_root, task)
            if record_path.is_file():
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if _record_matches(existing, config) and not (
                    args.retry_infra_errors and existing.get("infra_error")
                ):
                    _log(tag, "skip")
                    return existing
            _log(tag, "start")
            record = run_task(
                task,
                manifest_path=manifest_path,
                rotation_base_url=replay.base_url,
                output_root=output_root,
                config=config,
                max_turns=args.max_turns,
                timeout_s=args.timeout_s,
                retries=args.retries,
                max_responses=args.max_responses,
            )
            _log(
                tag,
                f"{'PASS' if record['success'] else 'FAIL'} "
                f"terminal={record['terminal_reason']} turns={record['executed_turns']} "
                f"calls={record['api_calls']} latency={record['latency_s']:.1f}s",
            )
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(submit, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                records = _load_selected_records(
                    output_root,
                    config=config,
                    episode_ids=selected_episode_ids,
                )
                summary = summarize(
                    records,
                    selected_counts=selected_counts,
                    selected_variants=selected_variants,
                    output_root=output_root,
                    config=config,
                    max_turns=args.max_turns,
                )
                _write_json(output_root / "summary.json", summary)
                _write_jsonl(
                    output_root / "records.jsonl",
                    sorted(
                        records,
                        key=lambda row: (str(row["variant"]), str(row["episode_id"])),
                    ),
                )

    records = _load_selected_records(
        output_root,
        config=config,
        episode_ids=selected_episode_ids,
    )
    summary = summarize(
        records,
        selected_counts=selected_counts,
        selected_variants=selected_variants,
        output_root=output_root,
        config=config,
        max_turns=args.max_turns,
    )
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["missing_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
