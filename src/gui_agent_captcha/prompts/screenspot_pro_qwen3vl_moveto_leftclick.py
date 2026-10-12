"""ScreenSpot-Pro Qwen3-VL prompt for native ``move_to`` trajectories.

The prompt is derived from the pinned ScreenSpot-Pro template.  Only the tool
schema is changed to expose ``move_to`` and coordinate-free ``left_click``;
the historical assistant click prefill is removed so generation starts at an
empty assistant boundary.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .screenspot_pro_qwen3vl_vllm import (
    GUIDED_PROMPT_TEMPLATE as SCREENSPOT_PRO_RAW_TEMPLATE,
    PROMPT_PROFILE as SCREENSPOT_PRO_BASE_PROFILE,
    UPSTREAM_COMMIT as SCREENSPOT_PRO_UPSTREAM_COMMIT,
)


PROMPT_PROFILE = "screenspot_pro_qwen3vl_moveto_leftclick_v1"
ASSISTANT_GENERATION_BOUNDARY = "<|im_start|>assistant\n"
SOURCE_COORDINATE_FORMAT = "qwen3_relative_0_1000"
_TOOLS_RE = re.compile(r"(<tools>\n)(\{.*?\})(\n</tools>)", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"\s*<tool_call>\s*(.*?)\s*</tool_call>\s*\Z", re.DOTALL)


def _build_prompt() -> str:
    match = _TOOLS_RE.search(SCREENSPOT_PRO_RAW_TEMPLATE)
    if match is None:
        raise RuntimeError("ScreenSpot-Pro raw Qwen3-VL prompt has no tools block")
    raw_schema = match.group(2)
    coordinate_end = '"type": "array"}, "required": ["action"]'
    if raw_schema.count(coordinate_end) != 1:
        raise RuntimeError("unexpected ScreenSpot-Pro coordinate schema spelling")
    raw_schema = raw_schema.replace(
        coordinate_end,
        '"type": "array"}}, "required": ["action"]',
    )
    payload = json.loads(raw_schema.replace("\n", "\\n"))
    function = payload["function"]
    parameters = function["parameters"]
    properties = parameters["properties"]
    action = properties["action"]
    coordinate = properties["coordinate"]
    function["description"] = function["description"].replace(
        "You can only use the left_click action to interact with the computer.",
        "You can use move_to and left_click actions to interact with the computer.",
    )
    action["description"] = action["description"].replace(
        "The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).",
        "The available actions are:\n"
        "* `move_to`: Move the cursor to coordinate (x, y).\n"
        "* `left_click`: Click at the current cursor position without a coordinate.",
    )
    action["enum"] = ["move_to", "left_click"]
    coordinate["description"] = (
        "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
        "coordinates to move the mouse to. Required only for `action=move_to`; "
        "`left_click` is coordinate-free."
    )
    updated_tools = json.dumps(payload, ensure_ascii=False).replace("\\n", "\n")
    updated = (
        SCREENSPOT_PRO_RAW_TEMPLATE[: match.start(2)]
        + updated_tools
        + SCREENSPOT_PRO_RAW_TEMPLATE[match.end(2) :]
    )
    boundary_index = updated.rfind(ASSISTANT_GENERATION_BOUNDARY)
    if boundary_index < 0:
        raise RuntimeError("ScreenSpot-Pro raw Qwen3-VL prompt has no assistant boundary")
    result = updated[: boundary_index + len(ASSISTANT_GENERATION_BOUNDARY)]
    if result.count(ASSISTANT_GENERATION_BOUNDARY) != 1:
        raise RuntimeError("derived Qwen3-VL prompt must contain one assistant boundary")
    if result.rsplit(ASSISTANT_GENERATION_BOUNDARY, 1)[1]:
        raise RuntimeError("derived Qwen3-VL prompt contains assistant prefill")
    return result


GUIDED_PROMPT_TEMPLATE = _build_prompt()
_SYSTEM_PREFIX = GUIDED_PROMPT_TEMPLATE.split("<|im_start|>user\n", 1)[0]
SYSTEM_PROMPT_TEXT = _SYSTEM_PREFIX[len("<|im_start|>") : -len("<|im_end|>\n")]


@dataclass(frozen=True)
class MultiturnPrompt:
    prompt: str
    image_paths: tuple[Path, ...]
    image_count: int
    action_history_count: int


def build_generation_messages(instruction: str) -> list[dict[str, Any]]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}]},
        {
            "role": "user",
            "content": [
                {"type": "image_placeholder"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def _coordinate(value: Sequence[int] | None) -> list[int]:
    if (
        value is None
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
        or any(item < 0 or item > 1000 for item in value)
    ):
        raise ValueError("coordinate must contain two integers in [0, 1000]")
    return [int(value[0]), int(value[1])]


def format_tool_call(action: str, *, coordinate: Sequence[int] | None = None) -> str:
    if action == "move_to":
        arguments: dict[str, Any] = {"action": action, "coordinate": _coordinate(coordinate)}
    elif action == "left_click":
        if coordinate is not None:
            raise ValueError("left_click is coordinate-free")
        arguments = {"action": action}
    else:
        raise ValueError(f"unsupported computer_use action: {action!r}")
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def parse_tool_call(response: str) -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("tool response must be nonempty text")
    if "<think>" in response.lower() or "</think>" in response.lower():
        raise ValueError("tool response must not contain Think tags")
    match = _TOOL_CALL_RE.fullmatch(response)
    if match is None:
        raise ValueError("response must contain exactly one complete tool call")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("tool call payload must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ValueError("tool call payload must contain name and arguments only")
    if payload.get("name") != "computer_use":
        raise ValueError("tool call name must be computer_use")
    arguments = payload["arguments"]
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be an object")
    action = arguments.get("action")
    if action == "move_to":
        if set(arguments) != {"action", "coordinate"}:
            raise ValueError("move_to requires action and coordinate only")
        arguments["coordinate"] = _coordinate(arguments["coordinate"])
    elif action == "left_click":
        if set(arguments) != {"action"}:
            raise ValueError("left_click requires action only")
    else:
        raise ValueError(f"unsupported computer_use action: {action!r}")
    return payload


def build_multiturn_prompt(
    instruction: str,
    *,
    image_paths: Sequence[Path],
    assistant_response_history: Sequence[str],
    images_to_keep: int = 3,
) -> MultiturnPrompt:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be positive")
    paths = tuple(Path(path) for path in image_paths)
    history = tuple(assistant_response_history)
    if len(paths) != len(history) + 1:
        raise ValueError("image_paths must contain one current observation per history turn plus one")
    for response in history:
        parse_tool_call(response)
    retained_start = max(0, len(paths) - images_to_keep)
    parts = [_SYSTEM_PREFIX]
    for turn_index in range(len(paths)):
        parts.append("<|im_start|>user\n")
        if turn_index >= retained_start:
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        parts.append(instruction)
        parts.append("<|im_end|>\n")
        if turn_index < len(history):
            parts.extend(["<|im_start|>assistant\n", history[turn_index], "<|im_end|>\n"])
    parts.append(ASSISTANT_GENERATION_BOUNDARY)
    prompt = "".join(parts)
    if "<think>" in prompt.lower() or "</think>" in prompt.lower():
        raise ValueError("multi-turn prompt contains forbidden Think text")
    return MultiturnPrompt(
        prompt=prompt,
        image_paths=paths[retained_start:],
        image_count=len(paths[retained_start:]),
        action_history_count=len(history),
    )


__all__ = [
    "ASSISTANT_GENERATION_BOUNDARY",
    "GUIDED_PROMPT_TEMPLATE",
    "MultiturnPrompt",
    "PROMPT_PROFILE",
    "SCREENSPOT_PRO_BASE_PROFILE",
    "SCREENSPOT_PRO_UPSTREAM_COMMIT",
    "SOURCE_COORDINATE_FORMAT",
    "SYSTEM_PROMPT_TEXT",
    "build_generation_messages",
    "build_multiturn_prompt",
    "format_tool_call",
    "parse_tool_call",
]
