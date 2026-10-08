"""Static GroundCUA VERL AgentLoops with strict SFT-parity tool trajectories.

The official annotation pool provides screenshots and target boxes, not a live
application.  This module therefore exposes a deterministic static environment:
after a legal nonterminal action it renders only the cursor over the unchanged
source screenshot, rebuilds the selected SFT prompt, and lets VERL generate the
next assistant tool call.  The private target box never enters a policy prompt.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from gui_agent_captcha.data.export_groundcua_multistep_subset import (
    QuantizedPoint,
    model_to_execution_pixel,
    render_cursor_observation,
)
from gui_agent_captcha.prompts.screenspot_pro_qwen3vl_leftclick_terminate import (
    build_multiturn_prompt as build_terminate_prompt,
)
from gui_agent_captcha.prompts.screenspot_pro_qwen3vl_moveto_leftclick import (
    build_multiturn_prompt as build_moveto_prompt,
)
from gui_agent_captcha.train.qwen3_vl_sft import build_qwen3_direct_official_messages


TRACK_DIRECT = "directclick"
TRACK_MOVETO = "moveto_leftclick"
TRACK_TERMINATE = "leftclick_terminate"
TRACKS = (TRACK_DIRECT, TRACK_MOVETO, TRACK_TERMINATE)
MAX_MOVES = 3
MAX_LEFT_CLICKS = 3
IMAGE_HISTORY_MAX = 3
INVALID_FORMAT_REWARD = -0.2
VALID_BUT_WRONG_REWARD = 0.0
VALID_AND_CORRECT_REWARD = 1.0
PROMPT_PROFILES = {
    TRACK_DIRECT: "screenspot_pro_qwen3_direct_dbe00114",
    TRACK_MOVETO: "screenspot_pro_qwen3vl_moveto_leftclick_v1",
    TRACK_TERMINATE: "screenspot_pro_qwen3vl_leftclick_terminate_v1",
}


def _make_rollout_artifact_dir(*, prefix: str) -> Path:
    """Create rollout scratch space on node-local storage."""
    local_root = Path(
        os.environ.get("GROUNDCUA_PAPER_STYLE_RL_LOCAL_TMP", "/dev/shm")
    )
    try:
        local_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=prefix, dir=str(local_root)))
    except OSError as exc:
        raise RuntimeError(
            "GroundCUA rollout scratch space must be writable node-local storage; "
            f"failed to create it under {local_root}"
        ) from exc


class ResponseFormatError(ValueError):
    """A generated response violates a track's strict single-tool-call grammar."""


@dataclass(frozen=True)
class ParsedAction:
    action: str
    coordinate: tuple[int, int] | None = None
    status: str | None = None


@dataclass(frozen=True)
class GroundCUAStaticTask:
    track: str
    instruction: str
    image_path: Path
    bbox_model_xyxy: tuple[float, float, float, float]
    task_id: str


@dataclass(frozen=True)
class GroundCUAPrompt:
    text: str
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class GroundCUAOutcome:
    format_valid: bool
    final_result_correct: bool
    reward_score: float
    reason: str | None


def _finalize_last_policy_output(output: Any, outcome: GroundCUAOutcome) -> None:
    """Attach a completed trajectory outcome to its final generated action.

    A later prompt can exceed the configured context limit after a legal
    nonterminal action.  There is then no additional policy action to emit, so
    the last generated action must carry the invalid terminal outcome instead
    of leaving the trajectory reward unset.
    """

    output.reward_score = outcome.reward_score
    output.extra_fields.update(
        {
            "format_valid": outcome.format_valid,
            "final_result_correct": outcome.final_result_correct,
            "terminal_reason": outcome.reason,
            "reward_extra_info": {
                "format_valid": outcome.format_valid,
                "final_result_correct": outcome.final_result_correct,
            },
        }
    )


def _strict_json_object(value: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ResponseFormatError(f"duplicate JSON key: {key!r}")
            result[key] = item
        return result

    try:
        payload = json.loads(
            value,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ResponseFormatError(f"non-finite JSON constant: {token}")
            ),
        )
    except ResponseFormatError:
        raise
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResponseFormatError("tool-call payload must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ResponseFormatError("tool-call payload must be a JSON object")
    return payload


def _tool_call_payload(response: str) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        raise ResponseFormatError("response must be nonempty text")
    text = response.strip()
    if "<think>" in text.lower() or "</think>" in text.lower():
        raise ResponseFormatError("response must not contain Think tags")
    if not text.startswith("<tool_call>") or not text.endswith("</tool_call>"):
        raise ResponseFormatError("response must be exactly one complete tool_call")
    inner = text[len("<tool_call>") : -len("</tool_call>")].strip()
    if not inner or "<tool_call>" in inner or "</tool_call>" in inner:
        raise ResponseFormatError("tool_call tags must be exactly one non-nested pair")
    payload = _strict_json_object(inner)
    if set(payload) != {"name", "arguments"} or payload.get("name") != "computer_use":
        raise ResponseFormatError("tool_call must name computer_use and arguments only")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ResponseFormatError("tool_call arguments must be an object")
    return arguments


def _coordinate(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 0 or item > 1000 for item in value)
    ):
        raise ResponseFormatError("coordinate must contain two integers in [0, 1000]")
    return int(value[0]), int(value[1])


def parse_track_response(track: str, response: str) -> ParsedAction:
    """Parse one complete generated call under one exact SFT action grammar."""

    if track not in TRACKS:
        raise ValueError(f"unsupported GroundCUA track: {track!r}")
    arguments = _tool_call_payload(response)
    action = arguments.get("action")
    if track == TRACK_DIRECT:
        if set(arguments) != {"action", "coordinate"} or action != "left_click":
            raise ResponseFormatError("directclick requires left_click and coordinate only")
        return ParsedAction(action="left_click", coordinate=_coordinate(arguments["coordinate"]))
    if track == TRACK_MOVETO:
        if action == "move_to":
            if set(arguments) != {"action", "coordinate"}:
                raise ResponseFormatError("move_to requires action and coordinate only")
            return ParsedAction(action="move_to", coordinate=_coordinate(arguments["coordinate"]))
        if action == "left_click":
            if set(arguments) != {"action"}:
                raise ResponseFormatError("left_click requires action only")
            return ParsedAction(action="left_click")
        raise ResponseFormatError("moveto_leftclick allows move_to or left_click only")
    if action == "left_click":
        if set(arguments) != {"action", "coordinate"}:
            raise ResponseFormatError("left_click requires action and coordinate only")
        return ParsedAction(action="left_click", coordinate=_coordinate(arguments["coordinate"]))
    if action == "terminate":
        if set(arguments) != {"action", "status"} or arguments.get("status") != "success":
            raise ResponseFormatError("terminate requires status=success only")
        return ParsedAction(action="terminate", status="success")
    raise ResponseFormatError("leftclick_terminate allows left_click or terminate only")


def _inside_bbox(point: tuple[int, int], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def score_outcome(format_valid: bool, final_result_correct: bool) -> GroundCUAOutcome:
    """Apply the only permitted numeric mapping for a completed trajectory."""

    if not format_valid:
        return GroundCUAOutcome(False, False, INVALID_FORMAT_REWARD, "invalid_format")
    if not final_result_correct:
        return GroundCUAOutcome(True, False, VALID_BUT_WRONG_REWARD, "wrong_terminal_result")
    return GroundCUAOutcome(True, True, VALID_AND_CORRECT_REWARD, None)


class GroundCUAStaticProtocol:
    """Private trajectory state; it contains no prompt-visible target fields."""

    def __init__(self, task: GroundCUAStaticTask) -> None:
        if task.track not in TRACKS:
            raise ValueError(f"unsupported GroundCUA track: {task.track!r}")
        if not task.instruction.strip() or not task.image_path.is_file():
            raise ValueError("GroundCUA task requires instruction and source image")
        x1, y1, x2, y2 = task.bbox_model_xyxy
        if not all(math.isfinite(value) for value in task.bbox_model_xyxy) or not (
            0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0
        ):
            raise ValueError("bbox_model_xyxy must be ordered within 0..1000")
        self.task = task
        self.actions: list[ParsedAction] = []
        self.responses: list[str] = []
        self.observations: list[Path] = [task.image_path]
        self.cursor: tuple[int, int] | None = None
        self.target_hit = False
        self.terminal = False
        self.outcome: GroundCUAOutcome | None = None

    @property
    def terminal_outcome(self) -> GroundCUAOutcome | None:
        return self.outcome

    @property
    def current_image_path(self) -> Path:
        return self.observations[-1]

    def _invalidate(self, reason: str) -> GroundCUAOutcome:
        self.terminal = True
        self.outcome = GroundCUAOutcome(False, False, INVALID_FORMAT_REWARD, reason)
        return self.outcome

    def accept(self, response: str) -> ParsedAction:
        if self.terminal:
            raise ResponseFormatError("actions after a terminal outcome are forbidden")
        try:
            parsed = parse_track_response(self.task.track, response)
        except ResponseFormatError as exc:
            self._invalidate(str(exc))
            raise
        if self.task.track == TRACK_DIRECT:
            self.actions.append(parsed)
            self.responses.append(response.strip())
            self.terminal = True
            self.outcome = score_outcome(
                True, _inside_bbox(parsed.coordinate or (-1, -1), self.task.bbox_model_xyxy)
            )
            return parsed
        if self.task.track == TRACK_MOVETO:
            return self._accept_moveto(parsed, response)
        return self._accept_terminate(parsed, response)

    def _accept_moveto(self, parsed: ParsedAction, response: str) -> ParsedAction:
        if parsed.action == "move_to":
            move_count = sum(action.action == "move_to" for action in self.actions)
            if move_count >= MAX_MOVES:
                self._invalidate("move_to count exceeds the rollout limit")
                raise ResponseFormatError("move_to count exceeds the rollout limit")
            self.cursor = parsed.coordinate
            self.actions.append(parsed)
            self.responses.append(response.strip())
            return parsed
        if self.cursor is None:
            self._invalidate("left_click requires a preceding legal move_to")
            raise ResponseFormatError("left_click requires a preceding legal move_to")
        self.actions.append(parsed)
        self.responses.append(response.strip())
        self.terminal = True
        self.outcome = score_outcome(True, _inside_bbox(self.cursor, self.task.bbox_model_xyxy))
        return parsed

    def _accept_terminate(self, parsed: ParsedAction, response: str) -> ParsedAction:
        if parsed.action == "left_click":
            click_count = sum(action.action == "left_click" for action in self.actions)
            if click_count >= MAX_LEFT_CLICKS:
                self._invalidate("left_click count exceeds the rollout limit")
                raise ResponseFormatError("left_click count exceeds the rollout limit")
            self.cursor = parsed.coordinate
            self.target_hit = _inside_bbox(parsed.coordinate or (-1, -1), self.task.bbox_model_xyxy)
            self.actions.append(parsed)
            self.responses.append(response.strip())
            return parsed
        click_count = sum(action.action == "left_click" for action in self.actions)
        if click_count < 1:
            self._invalidate("terminate(success) requires a preceding left_click")
            raise ResponseFormatError("terminate(success) requires a preceding left_click")
        self.actions.append(parsed)
        self.responses.append(response.strip())
        self.terminal = True
        self.outcome = score_outcome(True, self.target_hit)
        return parsed

    def action_limit_reached(self) -> bool:
        if self.task.track == TRACK_DIRECT:
            return len(self.actions) >= 1
        if self.task.track == TRACK_MOVETO:
            return sum(action.action == "move_to" for action in self.actions) >= MAX_MOVES
        return sum(action.action == "left_click" for action in self.actions) >= MAX_LEFT_CLICKS

    def render_cursor_observation(self, artifact_dir: Path) -> Path:
        if self.cursor is None:
            raise ValueError("a cursor observation requires a coordinate-bearing action")
        action_index = len(self.actions)
        output = artifact_dir / f"cursor-{action_index:02d}.png"
        if not output.exists():
            with Image.open(self.task.image_path) as image:
                size = image.size
            render_cursor_observation(
                self.task.image_path,
                output,
                cursor=QuantizedPoint(
                    model_xy=self.cursor,
                    execution_pixel_xy=model_to_execution_pixel(self.cursor, size),
                ),
            )
        self.observations.append(output)
        return output


def build_track_prompt(
    protocol: GroundCUAStaticProtocol,
    *,
    processor: Any | None = None,
) -> GroundCUAPrompt:
    """Use the selected authoritative SFT renderer for the next policy turn."""

    task = protocol.task
    if task.track == TRACK_DIRECT:
        if processor is None:
            raise ValueError("directclick SFT prompt rendering requires the Qwen3 processor")
        with Image.open(protocol.current_image_path) as source:
            image = source.convert("RGB")
        messages = build_qwen3_direct_official_messages(task.instruction, image)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not isinstance(text, str):
            raise TypeError("Qwen3 processor must return directclick prompt text")
        return GroundCUAPrompt(text=text, image_paths=(protocol.current_image_path,))
    builder = build_moveto_prompt if task.track == TRACK_MOVETO else build_terminate_prompt
    built = builder(
        task.instruction,
        image_paths=tuple(protocol.observations),
        assistant_response_history=tuple(protocol.responses),
        images_to_keep=IMAGE_HISTORY_MAX,
    )
    return GroundCUAPrompt(text=built.prompt, image_paths=tuple(built.image_paths))


def task_from_mapping(value: Any) -> GroundCUAStaticTask:
    if hasattr(value, "item") and not isinstance(value, Mapping):
        value = value.item()
    if not isinstance(value, Mapping):
        raise ValueError("GroundCUA VERL row requires task_config")
    expected = {"track", "instruction", "image_path", "bbox_model_xyxy", "task_id"}
    if set(value) != expected:
        raise ValueError(f"task_config keys must be exactly {sorted(expected)!r}")
    track = value["track"]
    instruction = value["instruction"]
    image_path = value["image_path"]
    task_id = value["task_id"]
    raw_bbox = value["bbox_model_xyxy"]
    if (
        track not in TRACKS
        or not isinstance(instruction, str)
        or not isinstance(image_path, str)
        or not isinstance(task_id, str)
        or not isinstance(raw_bbox, list | tuple)
        or len(raw_bbox) != 4
    ):
        raise ValueError("GroundCUA task_config has invalid values")
    try:
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError) as exc:
        raise ValueError("bbox_model_xyxy must contain four numbers") from exc
    return GroundCUAStaticTask(
        track=str(track),
        instruction=instruction,
        image_path=Path(image_path),
        bbox_model_xyxy=bbox,  # type: ignore[arg-type]
        task_id=task_id,
    )


try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
        register,
    )
    from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids
    from verl.workers.rollout.replica import TokenOutput
except ModuleNotFoundError as exc:
    VERL_AVAILABLE = False
    VERL_IMPORT_ERROR = exc

    class GroundCUAPaperStyleAgentLoop:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("GroundCUA VERL AgentLoop requires the VERL environment") from VERL_IMPORT_ERROR

else:
    VERL_AVAILABLE = True

    @register("groundcua_paper_style_directclick_v1")
    @register("groundcua_paper_style_moveto_leftclick_v1")
    @register("groundcua_paper_style_leftclick_terminate_v1")
    class GroundCUAPaperStyleAgentLoop(AgentLoopBase):
        """VERL AgentLoop that emits one independent step sample per tool call."""

        def __init__(
            self,
            *args: Any,
            artifact_root: str | Path | None = None,
            max_response_tokens: int = 256,
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            if self.processor is None:
                raise ValueError("GroundCUA AgentLoop requires a multimodal processor")
            # ``artifact_root`` is retained in the config for compatibility, but
            # rollout cursor images are node-local scratch data.  Persisting one
            # PNG per turn on the shared volume eventually exhausts the project
            # quota during long GRPO runs.
            self.artifact_root = Path(
                artifact_root
                or os.environ.get(
                    "GROUNDCUA_PAPER_STYLE_RL_ARTIFACT_ROOT",
                    "artifacts/runs/groundcua_paper_style_verl_rl/rollouts",
                )
            )
            self.max_response_tokens = int(max_response_tokens)

        def _encode(self, prompt: GroundCUAPrompt) -> tuple[list[int], list[Image.Image]]:
            images: list[Image.Image] = []
            for path in prompt.image_paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB").copy())
            model_inputs = build_multimodal_processor_inputs(
                self.processor,
                text=[prompt.text],
                images=images,
                mm_processor_kwargs={**dict(self.mm_processor_kwargs or {}), "truncation": False},
            )
            input_ids = model_inputs["input_ids"] if isinstance(model_inputs, Mapping) else model_inputs.input_ids
            return list(normalize_token_ids(input_ids)), images

        async def run(
            self, sampling_params: dict[str, Any], **kwargs: Any
        ) -> list[AgentLoopOutput]:
            task = task_from_mapping(kwargs.get("task_config"))
            protocol = GroundCUAStaticProtocol(task)
            trajectory_id = uuid4().hex
            artifact_dir = _make_rollout_artifact_dir(
                prefix=f"groundcua-{task.track}-{trajectory_id}-"
            )
            try:
                outputs: list[AgentLoopOutput] = []
                started = time.perf_counter()
                max_turns = 1 if task.track == TRACK_DIRECT else MAX_MOVES + 1 if task.track == TRACK_MOVETO else MAX_LEFT_CLICKS + 1
                for turn_index in range(max_turns):
                    prompt = build_track_prompt(protocol, processor=self.processor)
                    prompt_ids, images = self._encode(prompt)
                    if len(prompt_ids) > int(self.rollout_config.prompt_length):
                        protocol._invalidate("prompt exceeds configured rollout prompt length")
                        break
                    per_turn = dict(sampling_params)
                    per_turn["max_tokens"] = min(
                        int(per_turn.get("max_tokens", self.max_response_tokens)),
                        self.max_response_tokens,
                    )
                    generated: TokenOutput = await self.server_manager.generate(
                        request_id=f"{trajectory_id}-{turn_index}",
                        prompt_ids=prompt_ids,
                        sampling_params=per_turn,
                        image_data=images,
                        video_data=None,
                        audio_data=None,
                        mm_processor_kwargs={**dict(self.mm_processor_kwargs or {}), "truncation": False},
                    )
                    token_ids = list(generated.token_ids)
                    logprobs = generated.log_probs
                    if logprobs is None:
                        raise RuntimeError("VERL rollout server did not return response logprobs")
                    response = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    parsed: ParsedAction | None = None
                    try:
                        parsed = protocol.accept(response)
                        if not protocol.terminal and parsed.coordinate is not None:
                            protocol.render_cursor_observation(artifact_dir)
                    except ResponseFormatError:
                        pass
                    terminal = protocol.terminal
                    if not terminal and turn_index == max_turns - 1:
                        protocol._invalidate("action budget reached before a legal terminal action")
                        terminal = True
                    terminal_outcome = protocol.terminal_outcome if terminal else None
                    outputs.append(
                        AgentLoopOutput(
                            prompt_ids=prompt_ids,
                            response_ids=token_ids,
                            response_mask=[1] * len(token_ids),
                            response_logprobs=[float(value) for value in logprobs],
                            multi_modal_data={"images": images},
                            reward_score=(terminal_outcome.reward_score if terminal_outcome else None),
                            num_turns=1,
                            metrics=AgentLoopMetrics(
                                generate_sequences=(time.perf_counter() - started) / (turn_index + 1),
                                tool_calls=float(turn_index + 1),
                                compute_score=0.0,
                                num_preempted=int(generated.num_preempted or 0),
                            ),
                            extra_fields={
                                "trajectory_id": trajectory_id,
                                "step_index": turn_index,
                                "step_count": turn_index + 1,
                                "track": task.track,
                                "prompt_profile": PROMPT_PROFILES[task.track],
                                "format_valid": (
                                    terminal_outcome.format_valid if terminal_outcome else None
                                ),
                                "final_result_correct": (
                                    terminal_outcome.final_result_correct if terminal_outcome else None
                                ),
                                "terminal_reason": (
                                    terminal_outcome.reason if terminal_outcome else None
                                ),
                                "reward_extra_info": (
                                    {
                                        "format_valid": terminal_outcome.format_valid,
                                        "final_result_correct": terminal_outcome.final_result_correct,
                                    }
                                    if terminal_outcome
                                    else {}
                                ),
                                "response": response,
                                "artifact_dir": str(artifact_dir),
                                "prompt_image_paths": [str(path) for path in prompt.image_paths],
                                "action": parsed.action if parsed is not None else None,
                                "trajectory_loss_weight": 1.0,
                            },
                            mm_processor_kwargs={**dict(self.mm_processor_kwargs or {}), "truncation": False},
                        )
                    )
                    if terminal:
                        break
                if not outputs:
                    raise RuntimeError("GroundCUA AgentLoop emitted no policy step")
                terminal_outcome = protocol.terminal_outcome
                if terminal_outcome is None:
                    raise RuntimeError("GroundCUA AgentLoop ended without a terminal outcome")
                _finalize_last_policy_output(outputs[-1], terminal_outcome)
                step_count = len(outputs)
                for index, output in enumerate(outputs):
                    output.extra_fields["step_count"] = step_count
                    output.extra_fields["step_index"] = index
                    output.extra_fields["trajectory_loss_weight"] = 1.0 / step_count
                return outputs
            finally:
                # Cursor observations are already copied into ``multi_modal_data``;
                # remove their local files before the rollout is handed back.
                shutil.rmtree(artifact_dir, ignore_errors=True)


__all__ = [
    "GroundCUAOutcome",
    "GroundCUAPaperStyleAgentLoop",
    "GroundCUAPrompt",
    "GroundCUAStaticProtocol",
    "GroundCUAStaticTask",
    "INVALID_FORMAT_REWARD",
    "MAX_LEFT_CLICKS",
    "MAX_MOVES",
    "PROMPT_PROFILES",
    "ParsedAction",
    "ResponseFormatError",
    "TRACK_DIRECT",
    "TRACK_MOVETO",
    "TRACK_TERMINATE",
    "TRACKS",
    "VALID_AND_CORRECT_REWARD",
    "VALID_BUT_WRONG_REWARD",
    "build_track_prompt",
    "parse_track_response",
    "score_outcome",
    "task_from_mapping",
]
