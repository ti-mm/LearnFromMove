"""Training-parity multi-turn evaluation for the no-Think Qwen3-VL checkpoint.

Each benchmark source is a single clean image. After a valid click, the
evaluator therefore renders only the cursor on that unchanged source image;
Each observation updates the cursor while preserving the underlying GUI image.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Sequence
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from ..data.export_groundcua_multistep_subset import (
    QuantizedPoint,
    model_to_execution_pixel,
    render_cursor_observation,
)
from ..eval.groundcua_table2_qwen3_direct import (
    EXPECTED_DATASET_COUNTS,
    GroundingSample,
    load_benchmark_samples,
    load_image,
    score_prediction,
)
from ..prompts.screenspot_pro_qwen3vl_leftclick_terminate import (
    ASSISTANT_GENERATION_BOUNDARY,
    GUIDED_PROMPT_TEMPLATE,
    OFFICIAL_TERMINATE_DESCRIPTION,
    OFFICIAL_TERMINATE_STATUS_DESCRIPTION,
    OFFICIAL_TERMINATE_STATUS_VALUES,
    PROMPT_PROFILE,
    QWEN3_VL_COMPUTER_USE_COMMIT,
    QWEN3_VL_COMPUTER_USE_COOKBOOK_URL,
    QWEN3_VL_COMPUTER_USE_SOURCE_URL,
    SCREENSPOT_PRO_UPSTREAM_COMMIT,
    SCREENSPOT_PRO_UPSTREAM_REPOSITORY,
    build_multiturn_prompt,
    format_tool_call,
    parse_tool_call,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS
from .qwen3vl_checkpoint import validate_hf_checkpoint


SUPPORTED_BENCHMARKS = tuple(EXPECTED_DATASET_COUNTS)
TOTAL_BENCHMARK_SAMPLES = sum(EXPECTED_DATASET_COUNTS.values())
MAX_LEFT_CLICKS = 3
IMAGE_HISTORY_MAX = 3
MAX_MODEL_LEN = 65536
MAX_TOKENS = 100
GENERATION_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": MAX_TOKENS,
}
CURSOR_OBSERVATION_POLICY = "cursor_only_synthetic_from_unchanged_clean_source_v1"
CURSOR_RENDERER = (
    "gui_agent_captcha.data.export_groundcua_multistep_subset.render_cursor_observation"
)
_COMPLETE_TOOL_CALL = re.compile(r"\A<tool_call>\s*(.*?)\s*</tool_call>\Z", re.DOTALL)


def default_data_root() -> Path:
    from ..integrations.storage import storage_path

    return storage_path("data")


def default_checkpoint() -> Path:
    from ..integrations.storage import storage_path

    return storage_path("artifacts", "evals", "_models", "grounding-sft")


def default_output_parent() -> Path:
    from ..integrations.storage import storage_path

    return storage_path("artifacts", "evals")


def require_fresh_output(path: Path) -> Path:
    """Create an output directory and refuse to reuse any non-empty one."""

    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"evaluation output is not a directory: {resolved}")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty evaluation output: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _require_external_output(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    parent = default_output_parent().resolve()
    if resolved != parent and parent not in resolved.parents:
        raise ValueError(f"evaluation output must stay under {parent}: {resolved}")
    return require_fresh_output(resolved)


def _coordinate(value: Any) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 0 or item > 1000 for item in value)
    ):
        raise ValueError("left_click coordinate must contain two integers in [0, 1000]")
    return [int(value[0]), int(value[1])]


def parse_multiturn_response(response: str) -> dict[str, Any]:
    """Parse one complete assistant response under the two-action contract."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("model response is empty")
    text = response.strip()
    if "<think>" in text.lower() or "</think>" in text.lower():
        raise ValueError("model response contains forbidden Think text")
    match = _COMPLETE_TOOL_CALL.fullmatch(text)
    if match is None:
        raise ValueError("model response must be exactly one complete tool call")
    payload = parse_tool_call(text)
    arguments = payload["arguments"]
    action = arguments["action"]
    if action == "left_click":
        if set(arguments) != {"action", "coordinate"}:
            raise ValueError("left_click arguments must contain action and coordinate only")
        return {"action": action, "coordinate": _coordinate(arguments["coordinate"]), "status": None}
    if action == "terminate":
        if set(arguments) != {"action", "status"}:
            raise ValueError("terminate arguments must contain action and status only")
        status = arguments["status"]
        if status not in OFFICIAL_TERMINATE_STATUS_VALUES:
            raise ValueError("terminate status is not an official value")
        return {"action": action, "coordinate": None, "status": status}
    raise ValueError(f"unsupported action: {action!r}")


def _invalid_score(reason: str, *, termination: str | None = None) -> dict[str, Any]:
    return {
        "correctness": "wrong_format",
        "final_click": None,
        "predicted_point": None,
        "termination": termination,
        "parse_error": reason,
    }


def score_multiturn_actions(
    sample: GroundingSample,
    actions: Sequence[dict[str, Any]],
    *,
    max_left_clicks: int = MAX_LEFT_CLICKS,
) -> dict[str, Any]:
    """Validate the whole sequence, then score its last click before success."""

    if max_left_clicks < 1:
        raise ValueError("max_left_clicks must be positive")
    if not isinstance(actions, Sequence) or not actions:
        return _invalid_score("action sequence is empty")

    clicks: list[list[int]] = []
    terminal: dict[str, Any] | None = None
    for index, item in enumerate(actions):
        if not isinstance(item, dict) or set(item) != {"action", "coordinate", "status"}:
            return _invalid_score(f"action {index} has an invalid shape")
        action = item["action"]
        if action == "left_click":
            if terminal is not None:
                return {
                    **_invalid_score("left_click appears after terminate", termination=terminal["status"]),
                    "correctness": "wrong_action",
                }
            if item["status"] is not None:
                return _invalid_score(f"left_click action {index} has a status")
            try:
                coordinate = _coordinate(item["coordinate"])
            except ValueError as error:
                return _invalid_score(str(error))
            clicks.append(coordinate)
            if len(clicks) > max_left_clicks:
                return {
                    **_invalid_score("left_click count exceeds the rollout limit"),
                    "correctness": "wrong_action",
                }
            continue
        if action == "terminate":
            if item["coordinate"] is not None:
                return _invalid_score(f"terminate action {index} has a coordinate")
            if item["status"] not in OFFICIAL_TERMINATE_STATUS_VALUES:
                return _invalid_score(f"terminate action {index} has an invalid status")
            if index != len(actions) - 1:
                return {
                    **_invalid_score("terminate is not the final action", termination=item["status"]),
                    "correctness": "wrong_action",
                }
            terminal = item
            continue
        return _invalid_score(f"unsupported action: {action!r}")

    if terminal is None:
        return _invalid_score("missing final terminate action")
    if terminal["status"] != "success":
        return {
            **_invalid_score("terminate status is not success", termination=terminal["status"]),
            "correctness": "wrong_action",
        }
    if not clicks:
        return {
            **_invalid_score("terminate(success) has no preceding left_click", termination="success"),
            "correctness": "wrong_action",
        }

    final_click = clicks[-1]
    score = score_prediction(sample, tuple(final_click))
    return {
        "correctness": score.correctness,
        "final_click": final_click,
        "predicted_point": list(score.predicted_xy) if score.predicted_xy else None,
        "termination": "success",
        "parse_error": None,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sample_contract(samples: Sequence[GroundingSample], benchmark: str) -> dict[str, Any]:
    expected = EXPECTED_DATASET_COUNTS[benchmark]
    if len(samples) != expected:
        raise ValueError(f"{benchmark} count mismatch: {len(samples)} != {expected}")
    ids = [sample.record_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{benchmark} contains duplicate record IDs")
    for sample in samples:
        if not sample.instruction.strip():
            raise ValueError(f"{benchmark}:{sample.record_id} has an empty instruction")
        image = load_image(sample)
        if image.size != sample.image_size:
            raise ValueError(f"{benchmark}:{sample.record_id} image size mismatch")
        built = build_multiturn_prompt(
            sample.instruction,
            image_paths=(Path("source.png"),),
            assistant_response_history=(),
        )
        if not built.prompt.endswith(ASSISTANT_GENERATION_BOUNDARY):
            raise ValueError("multi-turn prompt does not end at generation boundary")
    return {
        "benchmark": benchmark,
        "count": len(samples),
        "first_record_id": ids[0],
        "last_record_id": ids[-1],
        "unique_record_ids": len(set(ids)),
        "image_references_valid": True,
        "prompt_profile": PROMPT_PROFILE,
        "assistant_prefill": False,
        "think_present": False,
        "allowed_actions": ["left_click", "terminate"],
        "coordinate_space": "qwen3_relative_0_1000",
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_left_clicks": MAX_LEFT_CLICKS,
        "score_rule": "last_valid_left_click_before_terminate_success",
    }


def preflight_benchmark(
    benchmark: str,
    *,
    data_root: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    checkpoint_report = validate_hf_checkpoint(checkpoint)
    samples = load_benchmark_samples(benchmark, data_root, sample_limit=None)
    report = _sample_contract(samples, benchmark)
    report.update(
        {
            "schema": "qwen3vl_leftclick_terminate_multiturn_benchmark_preflight_v1",
            "status": "preflight_passed",
            "data_root": str(data_root.resolve()),
            "checkpoint": checkpoint_report,
            "cursor_observation_policy": CURSOR_OBSERVATION_POLICY,
            "cursor_renderer": CURSOR_RENDERER,
            "official_qwen3_source": {
                "computer_use_commit": QWEN3_VL_COMPUTER_USE_COMMIT,
                "source_url": QWEN3_VL_COMPUTER_USE_SOURCE_URL,
                "cookbook_url": QWEN3_VL_COMPUTER_USE_COOKBOOK_URL,
                "screenspot_repository": SCREENSPOT_PRO_UPSTREAM_REPOSITORY,
                "screenspot_commit": SCREENSPOT_PRO_UPSTREAM_COMMIT,
            },
            "generation_settings": GENERATION_SETTINGS,
            "terminate_contract": {
                "action": "terminate",
                "description": OFFICIAL_TERMINATE_DESCRIPTION,
                "status_description": OFFICIAL_TERMINATE_STATUS_DESCRIPTION,
                "status_values": list(OFFICIAL_TERMINATE_STATUS_VALUES),
            },
        }
    )
    return report


def run_cpu_preflight(
    *,
    checkpoint: Path,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    selected = tuple(SUPPORTED_BENCHMARKS)
    reports = {
        benchmark: preflight_benchmark(benchmark, data_root=data_root, checkpoint=checkpoint)
        for benchmark in selected
    }
    total = sum(report["count"] for report in reports.values())
    if total != TOTAL_BENCHMARK_SAMPLES:
        raise ValueError(f"five-benchmark total mismatch: {total} != {TOTAL_BENCHMARK_SAMPLES}")
    output = _require_external_output(output_root)
    aggregate = {
        "schema": "qwen3vl_leftclick_terminate_multiturn_five_benchmark_cpu_preflight_v1",
        "status": "preflight_passed",
        "benchmarks": list(selected),
        "total": total,
        "expected_total": TOTAL_BENCHMARK_SAMPLES,
        "reports": reports,
        "prompt_profile": PROMPT_PROFILE,
        "assistant_prefill": False,
        "think_present": False,
        "allowed_actions": ["left_click", "terminate"],
        "coordinate_space": "qwen3_relative_0_1000",
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_left_clicks": MAX_LEFT_CLICKS,
        "score_rule": "last_valid_left_click_before_terminate_success",
        "cursor_observation_policy": CURSOR_OBSERVATION_POLICY,
        "cursor_renderer": CURSOR_RENDERER,
        "generation_settings": GENERATION_SETTINGS,
    }
    _write_json(output / "preflight.json", aggregate)
    (output / "prompt_template.txt").write_text(GUIDED_PROMPT_TEMPLATE, encoding="utf-8")
    return aggregate


def _materialize_source_image(sample: GroundingSample, output: Path, sample_index: int) -> Path:
    if sample.image_path is not None:
        return sample.image_path.resolve()
    target = output / "source_images" / f"sample_{sample_index:06d}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    image = load_image(sample)
    image.save(target, format="PNG")
    return target.resolve()


def _cursor_observation(
    source_path: Path,
    output: Path,
    *,
    sample_index: int,
    click_index: int,
    coordinate: Sequence[int],
    image_size: tuple[int, int],
) -> Path:
    model_xy = (int(coordinate[0]), int(coordinate[1]))
    cursor = QuantizedPoint(
        model_xy=model_xy,
        execution_pixel_xy=model_to_execution_pixel(model_xy, image_size=image_size),
    )
    target = output / "observations" / f"sample_{sample_index:06d}" / f"turn_{click_index:02d}.png"
    render_cursor_observation(source_path, target, cursor=cursor)
    return target.resolve()


def _canonical_action(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": parsed["action"],
        "coordinate": list(parsed["coordinate"]) if parsed["coordinate"] is not None else None,
        "status": parsed["status"],
    }


GenerateFn = Callable[[str, tuple[Image.Image, ...]], str]


def run_multiturn_rollout(
    sample: GroundingSample,
    *,
    sample_index: int,
    output_dir: Path,
    generate_fn: GenerateFn,
    max_left_clicks: int = MAX_LEFT_CLICKS,
    source_path: Path | None = None,
    initial_image_path: Path | None = None,
) -> dict[str, Any]:
    if max_left_clicks < 1:
        raise ValueError("max_left_clicks must be positive")
    source_path = source_path or _materialize_source_image(sample, output_dir, sample_index)
    image_paths: list[Path] = [initial_image_path or source_path]
    assistant_history: list[str] = []
    actions: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    parse_error: str | None = None
    click_limit_error = False

    for turn_index in range(max_left_clicks + 1):
        built = build_multiturn_prompt(
            sample.instruction,
            image_paths=image_paths,
            assistant_response_history=assistant_history,
            images_to_keep=IMAGE_HISTORY_MAX,
        )
        images = tuple(load_image_from_path(path) for path in built.image_paths)
        raw_response = generate_fn(built.prompt, images)
        turn: dict[str, Any] = {
            "turn_index": turn_index,
            "prompt": built.prompt,
            "prompt_image_paths": [str(path) for path in built.image_paths],
            "prompt_image_count": built.image_count,
            "assistant_history_count": built.action_history_count,
            "raw_response": raw_response,
            "parsed_action": None,
            "parse_error": None,
        }
        try:
            parsed = parse_multiturn_response(raw_response)
        except ValueError as error:
            parse_error = str(error)
            turn["parse_error"] = parse_error
            turns.append(turn)
            break
        action = _canonical_action(parsed)
        turn["parsed_action"] = action
        turns.append(turn)
        actions.append(action)
        if parsed["action"] == "terminate":
            break
        assistant_history.append(
            format_tool_call("left_click", coordinate=parsed["coordinate"])
        )
        next_observation = _cursor_observation(
            source_path,
            output_dir,
            sample_index=sample_index,
            click_index=len(assistant_history),
            coordinate=parsed["coordinate"],
            image_size=sample.image_size,
        )
        image_paths.append(next_observation)
        if len(assistant_history) >= max_left_clicks and turn_index == max_left_clicks:
            click_limit_error = True
            break

    if parse_error is not None:
        score = _invalid_score(parse_error)
    elif click_limit_error:
        score = {
            **_invalid_score("rollout produced a left_click where terminate was required"),
            "correctness": "wrong_action",
        }
    else:
        score = score_multiturn_actions(sample, actions, max_left_clicks=max_left_clicks)
    return {
        "benchmark": sample.benchmark,
        "record_id": sample.record_id,
        "instruction": sample.instruction,
        "source_image": str(source_path),
        "cursor_observation_policy": CURSOR_OBSERVATION_POLICY,
        "cursor_renderer": CURSOR_RENDERER,
        "turns": turns,
        "actions": actions,
        "action_count": len(actions),
        **score,
    }


def load_image_from_path(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class VLLMGenerator:
    def __init__(
        self,
        checkpoint: Path,
        *,
        tensor_parallel_size: int,
        max_model_len: int,
        gpu_memory_utilization: float,
    ) -> None:
        import os

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        from vllm import LLM, SamplingParams

        self._sampling = SamplingParams(**GENERATION_SETTINGS)
        self._llm = LLM(
            model=str(checkpoint),
            tokenizer=str(checkpoint),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            seed=0,
            limit_mm_per_prompt={"image": IMAGE_HISTORY_MAX},
            mm_processor_kwargs={"min_pixels": IMAGE_MIN_PIXELS, "max_pixels": IMAGE_MAX_PIXELS},
        )

    def __call__(self, prompt: str, images: tuple[Image.Image, ...]) -> str:
        outputs = self._llm.generate(
            [{"prompt": prompt, "multi_modal_data": {"image": list(images)}}],
            sampling_params=self._sampling,
            use_tqdm=False,
        )
        if len(outputs) != 1 or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no completion")
        return str(outputs[0].outputs[0].text).strip()


def run_benchmark(
    *,
    benchmark: str,
    checkpoint: Path,
    data_root: Path,
    output_dir: Path,
    tensor_parallel_size: int = 1,
    max_model_len: int = MAX_MODEL_LEN,
    gpu_memory_utilization: float = 0.8,
    max_left_clicks: int = MAX_LEFT_CLICKS,
    run_preflight: bool = True,
) -> dict[str, Any]:
    output = _require_external_output(output_dir)
    preflight = (
        preflight_benchmark(benchmark, data_root=data_root, checkpoint=checkpoint)
        if run_preflight
        else {
            "schema": "qwen3vl_leftclick_terminate_multiturn_preflight_status_v1",
            "status": "skipped_by_user",
            "reason": "direct_submission_requested_without_preflight",
        }
    )
    if run_preflight:
        _write_json(output / "preflight.json", preflight)
    (output / "prompt_template.txt").write_text(GUIDED_PROMPT_TEMPLATE, encoding="utf-8")
    samples = load_benchmark_samples(benchmark, data_root, sample_limit=None)
    generator = VLLMGenerator(
        checkpoint,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    predictions_path = output / "predictions.jsonl"
    counts = {"correct": 0, "wrong": 0, "wrong_action": 0, "wrong_format": 0}
    think_count = 0
    with predictions_path.open("x", encoding="utf-8") as handle:
        for sample_index, sample in enumerate(samples):
            row = run_multiturn_rollout(
                sample,
                sample_index=sample_index,
                output_dir=output,
                generate_fn=generator,
                max_left_clicks=max_left_clicks,
            )
            counts[row["correctness"]] += 1
            think_count += sum("<think>" in str(turn["raw_response"]).lower() for turn in row["turns"])
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            handle.flush()
    total = len(samples)
    summary = {
        "schema": "qwen3vl_leftclick_terminate_multiturn_benchmark_summary_v1",
        "status": "evaluation_complete",
        "benchmark": benchmark,
        "checkpoint": str(Path(checkpoint).resolve()),
        "prompt_profile": PROMPT_PROFILE,
        "allowed_actions": ["left_click", "terminate"],
        "assistant_prefill": False,
        "think_present": bool(think_count),
        "think_response_count": think_count,
        "total": total,
        **counts,
        "accuracy": counts["correct"] / total if total else 0.0,
        "parse_rate": (counts["correct"] + counts["wrong"] + counts["wrong_action"]) / total if total else 0.0,
        "predictions": str(predictions_path),
        "preflight": preflight,
        "generation_settings": GENERATION_SETTINGS,
        "max_model_len": max_model_len,
        "image_history_max": IMAGE_HISTORY_MAX,
        "max_left_clicks": max_left_clicks,
        "score_rule": "last_valid_left_click_before_terminate_success",
        "cursor_observation_policy": CURSOR_OBSERVATION_POLICY,
        "cursor_renderer": CURSOR_RENDERER,
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "protocol.json",
        {
            "schema": "qwen3vl_leftclick_terminate_multiturn_protocol_v1",
            "prompt_profile": PROMPT_PROFILE,
            "allowed_actions": ["left_click", "terminate"],
            "coordinate_space": "qwen3_relative_0_1000",
            "terminate_status_values": list(OFFICIAL_TERMINATE_STATUS_VALUES),
            "terminate_description": OFFICIAL_TERMINATE_DESCRIPTION,
            "terminate_status_description": OFFICIAL_TERMINATE_STATUS_DESCRIPTION,
            "computer_use_commit": QWEN3_VL_COMPUTER_USE_COMMIT,
            "computer_use_source_url": QWEN3_VL_COMPUTER_USE_SOURCE_URL,
            "computer_use_cookbook_url": QWEN3_VL_COMPUTER_USE_COOKBOOK_URL,
            "assistant_generation_boundary": ASSISTANT_GENERATION_BOUNDARY,
            "assistant_prefill": False,
            "think_present": False,
            "max_model_len": max_model_len,
            "instruction_repeated_each_turn": True,
            "assistant_history": "all_accepted_actions_in_order",
            "image_history_max": IMAGE_HISTORY_MAX,
            "max_left_clicks": max_left_clicks,
            "score_rule": "last_valid_left_click_before_terminate_success",
            "cursor_observation_policy": CURSOR_OBSERVATION_POLICY,
            "cursor_renderer": CURSOR_RENDERER,
            "observation_update": "cursor_overlay_on_fixed_gui_image",
        },
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-left-clicks", type=int, default=MAX_LEFT_CLICKS)
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data_root = default_data_root() if args.data_root is None else args.data_root.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.preflight:
        report = run_cpu_preflight(
            checkpoint=checkpoint,
            data_root=data_root,
            output_root=args.output_dir,
        ) if args.benchmark == SUPPORTED_BENCHMARKS[0] else preflight_benchmark(
            args.benchmark,
            data_root=data_root,
            checkpoint=checkpoint,
        )
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
        return 0
    report = run_benchmark(
        benchmark=args.benchmark,
        checkpoint=checkpoint,
        data_root=data_root,
        output_dir=args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_left_clicks=args.max_left_clicks,
        run_preflight=not args.skip_preflight,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
