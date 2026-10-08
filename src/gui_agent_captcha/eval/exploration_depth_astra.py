"""Six-environment evaluation with OpenAI ``exec_py``."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from latentguiworld.suite import load_evaluation_manifest as load_manifest
from ..benchmarks.exploration_depth.contracts import PAPER_VARIANTS, runtime_variant
from ..benchmarks.exploration_depth.service import RotationReplayServer
from latentguiworld.suite import build_evaluation_variant as build_benchmark_variant
from ..envs.rotation_code_execution import (
    RotationCodeExecutionEnvironmentError,
    RotationExecPyRuntime,
    TaskTerminalExecPyRuntime,
)
from ..models.frontier_computer_use import FrontierModelConfig
from ..models.openai_code_execution_computer import (
    OPENAI_CODE_EXECUTION_HISTORY,
    OPENAI_CODE_EXECUTION_PROTOCOL,
    OpenAICodeExecutionAPIError,
    OpenAICodeExecutionComputerBackend,
    OpenAICodeExecutionConfig,
    OpenAICodeExecutionProtocolError,
    exec_py_tool_declaration,
)
from .api_common import (
    EvaluationTask,
    _case_dir,
    _json_error,
    _log,
    _record_path,
    _validate_observation,
    _write_json,
    _write_jsonl,
    load_records,
    select_tasks,
)
from .api_common import SYSTEM_PROMPT

DEFAULT_BASE_URL = "https://api.modelverse.cn/v1"
DEFAULT_MAX_RESPONSES = 12
DEFAULT_LIMIT_PER_VARIANT = 150
DEFAULT_WORKERS = 2
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_RETRIES = 6
DEFAULT_REASONING_EFFORT = "medium"
ROTATION_VARIANTS = ("rotation_inner", "rotation_outer")
FIRST_PERSON_VARIANTS = ("ten_choice_first_person", "drag_first_person")
THIRD_PERSON_VARIANTS = ("ten_choice_third_person", "drag_third_person")
VARIANTS = ROTATION_VARIANTS + FIRST_PERSON_VARIANTS + THIRD_PERSON_VARIANTS
SUPPORTED_VARIANT_SETS = (
    VARIANTS,
    ROTATION_VARIANTS, FIRST_PERSON_VARIANTS,
    THIRD_PERSON_VARIANTS,
    ("drag_first_person",), ("ten_choice_first_person",),
    ("drag_third_person",), ("ten_choice_third_person",),
)
PROTOCOL_TRACK = OPENAI_CODE_EXECUTION_PROTOCOL
HISTORY_STRATEGY = OPENAI_CODE_EXECUTION_HISTORY
TRANSPORT = "modelverse_openai_responses_function_exec_py"
SCORING_POLICY = "first_real_release_attempt_success_only_no_correction"
INITIAL_OBSERVATION_DELIVERY = "model_requested_exec_py_screenshot"
CODE_RUNTIME = "restricted_persistent_python_pyautogui_subset"
FIRST_PERSON_CODE_RUNTIME = "restricted_persistent_python_pyautogui_first_person_subset"


@dataclass(frozen=True)
class EvaluationProfile:
    key: str
    model: str
    base_url_env: str
    evaluation_id: str
    task_instruction_suffix: str = ""


ASTRA_PROFILE = EvaluationProfile(
    key="gpt6_astra",
    model="gpt-6-astra",
    base_url_env="GPT6_ASTRA_BASE_URL",
    evaluation_id="exploration_depth_gpt6_astra_rotation_strict_first_release_10x2",
)
ASTRA_EXPLORE_FIRST_PROFILE = EvaluationProfile(
    key="gpt6_astra_explore_first",
    model=ASTRA_PROFILE.model,
    base_url_env=ASTRA_PROFILE.base_url_env,
    evaluation_id="exploration_depth_gpt6_astra_explore_first_rotation_strict_first_release_10x2",
    task_instruction_suffix="Before attempting to solve the task, first explore the environment.",
)
ASTRA_EXPLORE_FIRST_PERSON_PROFILE = EvaluationProfile(
    key="gpt6_astra_explore_egocentric",
    model=ASTRA_PROFILE.model,
    base_url_env=ASTRA_PROFILE.base_url_env,
    evaluation_id="exploration_depth_gpt6_astra_explore_egocentric_rotation_strict_first_release_10x2",
    task_instruction_suffix=(
        ASTRA_EXPLORE_FIRST_PROFILE.task_instruction_suffix
        + "\n\nThe current environment is egocentric."
    ),
)
GPT56_PROFILE = EvaluationProfile(
    key="gpt56_sol_exec_py",
    model="gpt-5.6-sol",
    base_url_env="GPT56_SOL_BASE_URL",
    evaluation_id="exploration_depth_gpt56_sol_exec_py_rotation_strict_first_release_10x2",
)
PROFILES = {
    ASTRA_PROFILE.key: ASTRA_PROFILE,
    ASTRA_EXPLORE_FIRST_PROFILE.key: ASTRA_EXPLORE_FIRST_PROFILE,
    ASTRA_EXPLORE_FIRST_PERSON_PROFILE.key: ASTRA_EXPLORE_FIRST_PERSON_PROFILE,
    GPT56_PROFILE.key: GPT56_PROFILE,
}

# Preserve the Astra names used by existing imports and artifacts.
MODEL_KEY = ASTRA_PROFILE.key
DEFAULT_MODEL = ASTRA_PROFILE.model
EVALUATION_ID = ASTRA_PROFILE.evaluation_id


def evaluation_id_for(
    profile: EvaluationProfile,
    limit_per_variant: int = DEFAULT_LIMIT_PER_VARIANT,
    variants: tuple[str, ...] = VARIANTS,
) -> str:
    if limit_per_variant < 1:
        raise ValueError("limit_per_variant must be >= 1")
    variants = tuple(variants)
    if variants == VARIANTS:
        scope = "six_environments"
    elif variants == ROTATION_VARIANTS:
        scope = "rotation"
    elif variants == FIRST_PERSON_VARIANTS:
        scope = "first_person"
    elif variants == THIRD_PERSON_VARIANTS:
        scope = "third_person"
    elif variants == ("drag_first_person",):
        scope = "drag_first_person"
    elif variants == ("ten_choice_first_person",):
        scope = "ten_choice_first_person"
    elif variants == ("drag_third_person",):
        scope = "drag_third_person"
    elif variants == ("ten_choice_third_person",):
        scope = "ten_choice_third_person"
    else:
        raise ValueError(f"unsupported variant set: {variants!r}")
    suffix = "_rotation_strict_first_release_10x2"
    if not profile.evaluation_id.endswith(suffix):
        raise ValueError(f"evaluation_id must end with {suffix!r}")
    prefix = profile.evaluation_id.removesuffix(suffix)
    return f"{prefix}_{scope}_strict_first_release_{limit_per_variant}x{len(variants)}"


def code_runtime_for(variants: tuple[str, ...]) -> str:
    return FIRST_PERSON_CODE_RUNTIME if any(v.endswith("_first_person") for v in variants) else CODE_RUNTIME


def resolve_config(
    env: Mapping[str, str],
    *,
    profile: EvaluationProfile = ASTRA_PROFILE,
) -> FrontierModelConfig:
    api_key = str(env.get("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY is required for {profile.key}")
    base_url = (str(env.get(profile.base_url_env, "")).strip() or DEFAULT_BASE_URL).rstrip("/")
    return FrontierModelConfig(
        provider="openai",
        model=profile.model,
        base_url=base_url,
        api_key=api_key,
        api_key_env="OPENAI_API_KEY",
    )


def _record_matches(
    record: Mapping[str, Any],
    config: FrontierModelConfig,
    *,
    max_responses: int = DEFAULT_MAX_RESPONSES,
    limit_per_variant: int = DEFAULT_LIMIT_PER_VARIANT,
    variants: tuple[str, ...] = VARIANTS,
    profile: EvaluationProfile = ASTRA_PROFILE,
) -> bool:
    return (
        record.get("record_status") == "complete"
        and (
            record.get("variant") not in THIRD_PERSON_VARIANTS
            or record.get("terminal_detection") == "environment_submission"
        )
        and record.get("evaluation") == evaluation_id_for(profile, limit_per_variant, variants)
        and record.get("provider_key") == profile.key
        and record.get("model") == config.model
        and record.get("base_url") == config.base_url
        and record.get("protocol_track") == PROTOCOL_TRACK
        and record.get("scoring_policy") == SCORING_POLICY
        and record.get("max_responses") == max_responses
        and record.get("reasoning_effort") == DEFAULT_REASONING_EFFORT
        and record.get("system_prompt") == SYSTEM_PROMPT
        and record.get("task_instruction_suffix", "") == profile.task_instruction_suffix
    )


def _load_selected_records(
    output_root: Path,
    *,
    config: FrontierModelConfig,
    max_responses: int,
    limit_per_variant: int = DEFAULT_LIMIT_PER_VARIANT,
    episode_ids: set[str],
    variants: tuple[str, ...] = VARIANTS,
    profile: EvaluationProfile = ASTRA_PROFILE,
) -> list[dict[str, Any]]:
    return [
        row
        for row in load_records(output_root)
        if _record_matches(
            row,
            config,
            max_responses=max_responses,
            limit_per_variant=limit_per_variant,
            variants=variants,
            profile=profile,
        )
        and str(row.get("episode_id")) in episode_ids
    ]


def _first_release_attempt(audit: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(audit, Mapping):
        return None
    rotation_state = audit.get("rotation_state")
    if isinstance(rotation_state, Mapping):
        attempt = rotation_state.get("attempt")
        if isinstance(attempt, Mapping):
            return dict(attempt)
    terminal_success = audit.get("terminal_success")
    if isinstance(terminal_success, bool):
        return {"success": terminal_success}
    return None


def run_task(
    task: EvaluationTask,
    *,
    manifest_path: Path,
    rotation_base_url: str | None,
    output_root: Path,
    config: FrontierModelConfig,
    max_responses: int,
    limit_per_variant: int = DEFAULT_LIMIT_PER_VARIANT,
    variants: tuple[str, ...] = VARIANTS,
    timeout_s: float,
    retries: int,
    profile: EvaluationProfile = ASTRA_PROFILE,
    prompt_extension: str | None = None,
    harness_source: str | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    episode = task.episode
    case_dir = _case_dir(output_root, task)
    trace_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    terminal_reason = "max_responses_without_release"
    infra_error: dict[str, str] | None = None
    protocol_error: dict[str, str] | None = None
    first_release_audit: dict[str, Any] | None = None
    runtime_error_count = 0
    executed_code_calls = 0
    environment: object | None = None
    backend: OpenAICodeExecutionComputerBackend | None = None

    try:
        environment_kwargs: dict[str, Any] = {
            "manifest_path": manifest_path,
            "artifact_dir": case_dir / "frames",
        }
        if task.variant.startswith("rotation_"):
            environment_kwargs["base_url"] = rotation_base_url
        environment = build_benchmark_variant(task.variant, **environment_kwargs)
        observation = environment.reset(task_id=task.episode_id)  # type: ignore[attr-defined]
        _validate_observation(observation.screenshot_path, observation.size_px)
        get_audit = getattr(environment, "get_evaluator_audit", None)
        audit_rows.append(
            {
                "code_call_index": 0,
                "action_index": 0,
                "evaluator_only": get_audit() if callable(get_audit) else {},
            }
        )
        runtime = (
            RotationExecPyRuntime(environment, observation)
            if task.variant in ROTATION_VARIANTS
            else TaskTerminalExecPyRuntime(environment, observation)
        )
        backend = OpenAICodeExecutionComputerBackend(
            config=OpenAICodeExecutionConfig(
                model=config.model,
                base_url=config.base_url,
                api_key=config.api_key,
                reasoning_effort=DEFAULT_REASONING_EFFORT,
            ),
            system_prompt=SYSTEM_PROMPT,
            prompt_extension=prompt_extension,
            call_log_dir=case_dir / "model_calls",
            request_timeout_s=timeout_s,
            request_retries=retries,
        )
        instruction = str(episode["instruction"])
        if profile.task_instruction_suffix:
            instruction = f"{instruction.strip()}\n\n{profile.task_instruction_suffix}"
        turn = backend.start(instruction)
        while True:
            if turn.ended:
                terminal_reason = "model_end_without_release"
                trace_rows.append(
                    {
                        "row_type": "model_end",
                        "response_id": turn.response_id,
                        "message_text": turn.message_text,
                        "model_call_log_path": backend.last_call_log_path,
                    }
                )
                break

            assert turn.code is not None
            executed_code_calls += 1
            result = runtime.execute(turn.code, call_index=backend.call_count)
            runtime_error_count += result.runtime_error is not None
            trace_rows.append(
                {
                    "row_type": "exec_py",
                    "code_call_index": backend.call_count,
                    "response_id": turn.response_id,
                    "function_call_id": turn.call_id,
                    "generated_code": turn.code,
                    "program_logs": result.logs,
                    "runtime_error": result.runtime_error,
                    "first_release_seen": result.first_release_seen,
                    "model_call_log_path": backend.last_call_log_path,
                }
            )
            trace_rows.extend(
                {"row_type": "environment_action", **row} for row in result.action_rows
            )
            audit_rows.extend(result.audit_rows)
            if result.first_release_seen:
                first_release_audit = result.first_release_audit
                attempt = _first_release_attempt(first_release_audit)
                terminal_reason = (
                    "first_release_success"
                    if attempt is not None and attempt.get("success") is True
                    else "first_release_failure"
                )
                break
            if backend.call_count >= max_responses:
                terminal_reason = "max_responses_without_release"
                break
            turn = backend.continue_with(result.output)
    except OpenAICodeExecutionAPIError as error:
        terminal_reason = "infra_error"
        infra_error = _json_error(error)
    except OpenAICodeExecutionProtocolError as error:
        terminal_reason = "protocol_error"
        protocol_error = _json_error(error)
    except RotationCodeExecutionEnvironmentError as error:
        terminal_reason = "infra_error"
        infra_error = _json_error(error)
    except Exception as error:
        terminal_reason = "infra_error"
        infra_error = _json_error(error)
        infra_error["traceback"] = traceback.format_exc()
    finally:
        if backend is not None:
            backend.close()
        if environment is not None:
            close = getattr(environment, "close", None)
            if callable(close):
                close()

    trace_path = case_dir / "trace.jsonl"
    audit_path = case_dir / "audit.jsonl"
    _write_jsonl(trace_path, trace_rows)
    _write_jsonl(audit_path, audit_rows)
    attempt = _first_release_attempt(first_release_audit)
    success = bool(attempt is not None and attempt.get("success") is True)
    action_counts = Counter(
        str(row.get("action", {}).get("kind"))
        for row in trace_rows
        if row.get("row_type") == "environment_action" and isinstance(row.get("action"), Mapping)
    )
    record = {
        "record_status": "complete",
        "evaluation": evaluation_id_for(profile, limit_per_variant, variants),
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "variant": task.variant,
        "exploration_level": episode["exploration_level"],
        "episode_id": task.episode_id,
        "case_seed": episode["case_seed"],
        "split": episode["split"],
        "provider_key": profile.key,
        "model": config.model,
        "response_model_exact_match_required": True,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "history_strategy": HISTORY_STRATEGY,
        "system_prompt": SYSTEM_PROMPT,
        "harness_prompt_injected": bool(prompt_extension),
        "harness_source": harness_source,
        "task_instruction_suffix": profile.task_instruction_suffix,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "tool_declaration": exec_py_tool_declaration(),
        "code_runtime": code_runtime_for(variants),
        "initial_observation_delivery": INITIAL_OBSERVATION_DELIVERY,
        "image_detail": "original",
        "scoring_policy": SCORING_POLICY,
        "strict_first_release": True,
        "terminal_detection": "environment_submission",
        "trailing_code_after_first_release_executed": False,
        "success": success,
        "attempt_seen": attempt is not None,
        "first_release_attempt": attempt,
        "terminal_reason": terminal_reason,
        "max_responses": max_responses,
        "api_calls": backend.call_count if backend is not None else 0,
        "executed_code_calls": executed_code_calls,
        "runtime_error_count": runtime_error_count,
        "action_counts": dict(sorted(action_counts.items())),
        "trace_path": str(trace_path),
        "audit_path": str(audit_path),
        "infra_error": infra_error,
        "protocol_error": protocol_error,
        "api_key_persisted": False,
        "latency_s": time.monotonic() - started_at,
    }
    _write_json(_record_path(output_root, task), record)
    return record


def summarize(
    records: list[dict[str, Any]],
    *,
    selected_counts: Mapping[str, int],
    output_root: Path,
    config: FrontierModelConfig,
    max_responses: int,
    limit_per_variant: int = DEFAULT_LIMIT_PER_VARIANT,
    variants: tuple[str, ...] = VARIANTS,
    profile: EvaluationProfile = ASTRA_PROFILE,
) -> dict[str, Any]:
    expected_total = sum(int(selected_counts.get(variant, 0)) for variant in variants)
    error_count = sum(bool(row.get("infra_error") or row.get("protocol_error")) for row in records)
    complete = len(records) == expected_total
    formally_complete = complete and error_count == 0
    per_variant: dict[str, Any] = {}
    for variant in variants:
        subset = [row for row in records if row.get("variant") == variant]
        expected = int(selected_counts.get(variant, 0))
        successes = sum(row.get("success") is True for row in subset)
        errors = sum(bool(row.get("infra_error") or row.get("protocol_error")) for row in subset)
        calls = [int(row.get("api_calls", 0)) for row in subset]
        variant_complete = len(subset) == expected and errors == 0
        per_variant[variant] = {
            "expected_count": expected,
            "completed_count": len(subset),
            "missing_count": max(0, expected - len(subset)),
            "success_count": successes,
            "success_rate_observed": successes / len(subset) if subset else 0.0,
            "formal_success_rate": successes / expected if variant_complete else None,
            "first_release_attempt_count": sum(row.get("attempt_seen") is True for row in subset),
            "no_release_count": sum(row.get("attempt_seen") is not True for row in subset),
            "error_count": errors,
            "mean_api_calls": statistics.fmean(calls) if calls else None,
            "terminal_counts": dict(
                sorted(Counter(str(row.get("terminal_reason")) for row in subset).items())
            ),
            "split_counts": dict(sorted(Counter(str(row.get("split")) for row in subset).items())),
            "episode_ids": sorted(str(row["episode_id"]) for row in subset),
        }
    successes = sum(row.get("success") is True for row in records)
    return {
        "evaluation": evaluation_id_for(profile, limit_per_variant, variants),
        "suite_id": next((row.get("suite_id") for row in records), None),
        "provider_key": profile.key,
        "model": config.model,
        "response_model_exact_match_required": True,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "history_strategy": HISTORY_STRATEGY,
        "system_prompt": SYSTEM_PROMPT,
        "task_instruction_suffix": profile.task_instruction_suffix,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "tool_declaration": exec_py_tool_declaration(),
        "code_runtime": code_runtime_for(variants),
        "initial_observation_delivery": INITIAL_OBSERVATION_DELIVERY,
        "image_detail": "original",
        "scoring_policy": SCORING_POLICY,
        "strict_first_release": True,
        "terminal_detection": "environment_submission",
        "max_responses": max_responses,
        "selected_variants": list(variants),
        "expected_count": expected_total,
        "completed_count": len(records),
        "missing_count": max(0, expected_total - len(records)),
        "success_count": successes,
        "success_rate_observed": successes / len(records) if records else 0.0,
        "formal_success_rate": successes / expected_total if formally_complete else None,
        "first_release_attempt_count": sum(row.get("attempt_seen") is True for row in records),
        "no_release_count": sum(row.get("attempt_seen") is not True for row in records),
        "error_count": error_count,
        "api_key_persisted": False,
        "status": (
            "complete"
            if formally_complete
            else "complete_with_errors"
            if complete
            else "incomplete"
        ),
        "output_root": str(output_root),
        "variants": per_variant,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an OpenAI model with exec_py on the six benchmark environments."
        )
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-responses", type=int, default=DEFAULT_MAX_RESPONSES)
    parser.add_argument("--limit-per-variant", type=int, default=DEFAULT_LIMIT_PER_VARIANT)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(PAPER_VARIANTS) + ROTATION_VARIANTS + FIRST_PERSON_VARIANTS + THIRD_PERSON_VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--retry-infra-errors", action="store_true")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default=ASTRA_PROFILE.key,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_responses != DEFAULT_MAX_RESPONSES:
        raise SystemExit(f"formal evaluation requires --max-responses {DEFAULT_MAX_RESPONSES}")
    if args.limit_per_variant < 1:
        raise SystemExit("--limit-per-variant must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    selected_variants = tuple(dict.fromkeys(runtime_variant(v) for v in args.variants))
    if selected_variants not in SUPPORTED_VARIANT_SETS:
        raise SystemExit(
            "--variants must select all six environments, a supported pair, or a single variant"
        )
    profile = PROFILES[args.profile]
    try:
        config = resolve_config(os.environ, profile=profile)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    manifest_path = args.manifest.resolve()
    output_root = args.output_root.resolve()
    manifest = load_manifest(manifest_path)
    all_tasks, selected_pair_ids = select_tasks(
        manifest,
        limit_per_variant=args.limit_per_variant,
    )
    tasks = [task for task in all_tasks if task.variant in selected_variants]
    selected_episode_ids = {task.episode_id for task in tasks}
    expected_count = len(selected_variants) * args.limit_per_variant
    if len(tasks) != expected_count:
        raise SystemExit(f"selected {len(tasks)} Rotation tasks, expected {expected_count}")
    selected_counts = Counter(task.variant for task in tasks)
    output_root.mkdir(parents=True, exist_ok=True)
    completion_path = output_root / "evaluation_complete.json"
    if completion_path.is_file():
        completion_path.unlink()
    run_manifest = {
        "suite_manifest": str(manifest_path),
        "suite_id": manifest["suite_id"],
        "evaluation": evaluation_id_for(profile, args.limit_per_variant, selected_variants),
        "provider_key": profile.key,
        "model": config.model,
        "response_model_exact_match_required": True,
        "base_url": config.base_url,
        "protocol_track": PROTOCOL_TRACK,
        "transport": TRANSPORT,
        "history_strategy": HISTORY_STRATEGY,
        "system_prompt": SYSTEM_PROMPT,
        "task_instruction_suffix": profile.task_instruction_suffix,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "tool_declaration": exec_py_tool_declaration(),
        "code_runtime": code_runtime_for(selected_variants),
        "initial_observation_delivery": INITIAL_OBSERVATION_DELIVERY,
        "image_detail": "original",
        "scoring_policy": SCORING_POLICY,
        "strict_first_release": True,
        "terminal_detection": "environment_submission",
        "selected_pair_ids": {
            family: selected_pair_ids[family]
            for family in sorted({str(task.episode["family"]) for task in tasks})
        },
        "selected_variants": list(selected_variants),
        "selected_counts": dict(sorted(selected_counts.items())),
        "selected_episode_ids": [task.episode_id for task in tasks],
        "expected_count": len(tasks),
        "max_responses": args.max_responses,
        "workers": args.workers,
        "timeout_s": args.timeout_s,
        "retries": args.retries,
        "api_key_persisted": False,
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    _log("main", f"tasks={len(tasks)} workers={args.workers} output={output_root}")

    replay_context = (
        RotationReplayServer(public_root=manifest_path.parent)
        if any(variant.startswith("rotation_") for variant in selected_variants)
        else nullcontext(None)
    )
    with replay_context as replay:
        rotation_base_url = replay.base_url if replay is not None else None

        def submit(task: EvaluationTask) -> dict[str, Any]:
            tag = f"{task.variant}/{task.episode_id}"
            record_path = _record_path(output_root, task)
            if record_path.is_file():
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if _record_matches(
                    existing,
                    config,
                    max_responses=args.max_responses,
                    limit_per_variant=args.limit_per_variant,
                    variants=selected_variants,
                    profile=profile,
                ) and not (
                    args.retry_infra_errors
                    and (existing.get("infra_error") or existing.get("protocol_error"))
                ):
                    _log(tag, "skip")
                    return existing
            _log(tag, "start")
            record = run_task(
                task,
                manifest_path=manifest_path,
                rotation_base_url=rotation_base_url,
                output_root=output_root,
                config=config,
                max_responses=args.max_responses,
                limit_per_variant=args.limit_per_variant,
                variants=selected_variants,
                timeout_s=args.timeout_s,
                retries=args.retries,
                profile=profile,
            )
            _log(
                tag,
                f"{'PASS' if record['success'] else 'FAIL'} "
                f"terminal={record['terminal_reason']} calls={record['api_calls']} "
                f"latency={record['latency_s']:.1f}s",
            )
            return record

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(submit, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                records = _load_selected_records(
                    output_root,
                    config=config,
                    max_responses=args.max_responses,
                    limit_per_variant=args.limit_per_variant,
                    episode_ids=selected_episode_ids,
                    variants=selected_variants,
                    profile=profile,
                )
                summary = summarize(
                    records,
                    selected_counts=selected_counts,
                    output_root=output_root,
                    config=config,
                    max_responses=args.max_responses,
                    limit_per_variant=args.limit_per_variant,
                    variants=selected_variants,
                    profile=profile,
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
        max_responses=args.max_responses,
        limit_per_variant=args.limit_per_variant,
        episode_ids=selected_episode_ids,
        variants=selected_variants,
        profile=profile,
    )
    summary = summarize(
        records,
        selected_counts=selected_counts,
        output_root=output_root,
        config=config,
        max_responses=args.max_responses,
        limit_per_variant=args.limit_per_variant,
        variants=selected_variants,
        profile=profile,
    )
    _write_json(output_root / "summary.json", summary)
    if summary["status"] == "complete":
        _write_json(
            completion_path,
            {
                "status": "complete",
                "suite_id": manifest["suite_id"],
                "evaluation": evaluation_id_for(profile, args.limit_per_variant, selected_variants),
                "provider_key": profile.key,
                "model": config.model,
                "expected_count": summary["expected_count"],
                "completed_count": summary["completed_count"],
                "error_count": summary["error_count"],
                "summary_path": str(output_root / "summary.json"),
            },
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["missing_count"]:
        return 2
    return 0 if summary["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
