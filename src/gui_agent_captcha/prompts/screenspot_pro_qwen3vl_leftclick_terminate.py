"""ScreenSpot-Pro Qwen3-VL prompt with the official termination subset.

The raw prompt is imported from the pinned local ScreenSpot-Pro Qwen3-VL
template. This module makes the two requested contract edits only: it changes
the advertised action enum and termination fields, then stops at the assistant
generation boundary instead of retaining the historical click prefill.
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
    UPSTREAM_COMMIT as _SCREENSPOT_PRO_UPSTREAM_COMMIT,
)


PROMPT_PROFILE = "screenspot_pro_qwen3vl_leftclick_terminate_v1"
SCREENSPOT_PRO_UPSTREAM_REPOSITORY = (
    "https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding"
)
SCREENSPOT_PRO_UPSTREAM_COMMIT = _SCREENSPOT_PRO_UPSTREAM_COMMIT
QWEN3_VL_COMPUTER_USE_SOURCE_URL = (
    "https://github.com/QwenLM/Qwen3-VL/blob/"
    "96a8f5947c3e8dd3e01dc3a9fd522cd6eafcc292/"
    "cookbooks/utils/agent_function_call.py"
)
QWEN3_VL_COMPUTER_USE_COOKBOOK_URL = (
    "https://github.com/QwenLM/Qwen3-VL/blob/"
    "96a8f5947c3e8dd3e01dc3a9fd522cd6eafcc292/"
    "cookbooks/computer_use.ipynb"
)
QWEN3_VL_COMPUTER_USE_COMMIT = "96a8f5947c3e8dd3e01dc3a9fd522cd6eafcc292"

OFFICIAL_TERMINATE_ACTION = "terminate"
OFFICIAL_TERMINATE_DESCRIPTION = "Terminate the current task and report its completion status."
OFFICIAL_TERMINATE_STATUS_DESCRIPTION = (
    "The status of the task. Required only by `action=terminate`."
)
OFFICIAL_TERMINATE_STATUS_VALUES = ("success", "failure")
ASSISTANT_GENERATION_BOUNDARY = "<|im_start|>assistant\n"

_TOOLS_RE = re.compile(r"(<tools>\n)(\{.*?\})(\n</tools>)", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _build_prompt() -> str:
    match = _TOOLS_RE.search(SCREENSPOT_PRO_RAW_TEMPLATE)
    if match is None:
        raise RuntimeError("ScreenSpot-Pro raw Qwen3-VL prompt has no tools block")
    # The pinned raw template embeds literal newlines inside JSON string
    # values. It also has one historical brace omission: ``required`` and
    # ``type`` appear before the properties object is closed. Repair exactly
    # that known source spelling before parsing; the raw file stays untouched.
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
        "You can only use the left_click and terminate actions to interact with the computer.",
    )
    action["description"] = action["description"].replace(
        "The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).",
        "The available actions are:\n"
        "* `left_click`: Click the left mouse button with coordinate (x, y).\n"
        f"* `terminate`: {OFFICIAL_TERMINATE_DESCRIPTION}",
    )
    action["enum"] = ["left_click", OFFICIAL_TERMINATE_ACTION]
    properties["status"] = {
        "description": OFFICIAL_TERMINATE_STATUS_DESCRIPTION,
        "type": "string",
        "enum": list(OFFICIAL_TERMINATE_STATUS_VALUES),
    }
    # Keep the ScreenSpot-Pro coordinate field and description byte-for-byte.
    properties["coordinate"] = coordinate

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


def _system_prompt_text() -> str:
    prefix = GUIDED_PROMPT_TEMPLATE.split("<|im_start|>user\n", 1)[0]
    if not prefix.startswith("<|im_start|>") or not prefix.endswith("<|im_end|>\n"):
        raise RuntimeError("derived Qwen3-VL prompt has an invalid system boundary")
    return prefix[len("<|im_start|>") : -len("<|im_end|>\n")]


SYSTEM_PROMPT_TEXT = _system_prompt_text()


@dataclass(frozen=True)
class MultiturnPrompt:
    prompt: str
    image_paths: tuple[Path, ...]
    image_count: int
    action_history_count: int


def build_multiturn_prompt(
    instruction: str,
    *,
    image_paths: Sequence[Path],
    assistant_response_history: Sequence[str],
    images_to_keep: int = 3,
) -> MultiturnPrompt:
    """Build a training-shaped multi-turn request at an empty assistant turn.

    ``image_paths`` contains one current observation for each user turn, so
    it must contain exactly one more item than the accepted assistant history.
    Older user turns remain text-only when the sliding image window is full.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be positive")
    paths = tuple(Path(path) for path in image_paths)
    history = tuple(assistant_response_history)
    if len(paths) != len(history) + 1:
        raise ValueError(
            "image_paths must contain one current observation per history turn plus one"
        )
    for response in history:
        try:
            parsed = parse_tool_call(response)
        except ValueError as error:
            raise ValueError("assistant history contains an invalid left_click tool call") from error
        if parsed["arguments"].get("action") != "left_click":
            raise ValueError("assistant history entries must be left_click tool calls")

    retained_start = max(0, len(paths) - images_to_keep)
    retained_paths = paths[retained_start:]
    parts = [_SYSTEM_PREFIX]
    for turn_index, _path in enumerate(paths):
        parts.append("<|im_start|>user\n")
        if turn_index >= retained_start:
            parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        parts.append(instruction)
        parts.append("<|im_end|>\n")
        if turn_index < len(history):
            parts.append("<|im_start|>assistant\n")
            parts.append(history[turn_index])
            parts.append("<|im_end|>\n")
    parts.append(ASSISTANT_GENERATION_BOUNDARY)
    prompt = "".join(parts)
    if not prompt.endswith(ASSISTANT_GENERATION_BOUNDARY):
        raise AssertionError("multi-turn prompt must end at assistant generation boundary")
    if "<think>" in prompt.lower() or "</think>" in prompt.lower():
        raise ValueError("multi-turn prompt contains forbidden Think text")
    if prompt.rsplit(ASSISTANT_GENERATION_BOUNDARY, 1)[1]:
        raise AssertionError("multi-turn prompt contains assistant prefill")
    return MultiturnPrompt(
        prompt=prompt,
        image_paths=retained_paths,
        image_count=len(retained_paths),
        action_history_count=len(history),
    )


def build_generation_messages(instruction: str) -> list[dict[str, Any]]:
    """Return system plus user context, with no assistant prefill message."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT_TEXT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image_placeholder"},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def _coordinate(coordinate: Sequence[int] | None) -> list[int]:
    if (
        coordinate is None
        or len(coordinate) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
        or any(value < 0 or value > 1000 for value in coordinate)
    ):
        raise ValueError("left_click coordinate must contain two integers in [0, 1000]")
    return [int(coordinate[0]), int(coordinate[1])]


def format_tool_call(
    action: str,
    *,
    coordinate: Sequence[int] | None = None,
    status: str | None = None,
) -> str:
    """Format one strict Qwen3-VL ``computer_use`` tool call."""

    if action == "left_click":
        if status is not None:
            raise ValueError("left_click does not accept status")
        arguments = {"action": action, "coordinate": _coordinate(coordinate)}
    elif action == OFFICIAL_TERMINATE_ACTION:
        if coordinate is not None:
            raise ValueError("terminate does not accept coordinate")
        if status not in OFFICIAL_TERMINATE_STATUS_VALUES:
            raise ValueError("terminate status must be success or failure")
        arguments = {"action": action, "status": status}
    else:
        raise ValueError(f"unsupported computer_use action: {action!r}")
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


def parse_tool_call(response: str) -> dict[str, Any]:
    """Parse exactly one complete no-reasoning Qwen3-VL tool call."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("tool response must be nonempty text")
    if "<think>" in response.lower() or "</think>" in response.lower():
        raise ValueError("tool response must not contain Think tags")
    matches = _TOOL_CALL_RE.findall(response.strip())
    if len(matches) != 1:
        raise ValueError("response must contain exactly one complete tool call")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("tool call payload must be valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ValueError("tool call payload must contain name and arguments only")
    if payload.get("name") != "computer_use":
        raise ValueError("tool call name must be computer_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be an object")
    action = arguments.get("action")
    if action == "left_click":
        if set(arguments) != {"action", "coordinate"}:
            raise ValueError("left_click requires action and coordinate only")
        arguments["coordinate"] = _coordinate(arguments.get("coordinate"))
    elif action == OFFICIAL_TERMINATE_ACTION:
        if set(arguments) != {"action", "status"}:
            raise ValueError("terminate requires action and status only")
        if arguments["status"] not in OFFICIAL_TERMINATE_STATUS_VALUES:
            raise ValueError("terminate status must be success or failure")
    else:
        raise ValueError(f"unsupported computer_use action: {action!r}")
    return payload


__all__ = [
    "ASSISTANT_GENERATION_BOUNDARY",
    "GUIDED_PROMPT_TEMPLATE",
    "OFFICIAL_TERMINATE_ACTION",
    "OFFICIAL_TERMINATE_DESCRIPTION",
    "OFFICIAL_TERMINATE_STATUS_DESCRIPTION",
    "OFFICIAL_TERMINATE_STATUS_VALUES",
    "PROMPT_PROFILE",
    "QWEN3_VL_COMPUTER_USE_COMMIT",
    "QWEN3_VL_COMPUTER_USE_COOKBOOK_URL",
    "QWEN3_VL_COMPUTER_USE_SOURCE_URL",
    "SCREENSPOT_PRO_BASE_PROFILE",
    "SCREENSPOT_PRO_UPSTREAM_COMMIT",
    "SCREENSPOT_PRO_UPSTREAM_REPOSITORY",
    "SYSTEM_PROMPT_TEXT",
    "MultiturnPrompt",
    "build_multiturn_prompt",
    "build_generation_messages",
    "format_tool_call",
    "parse_tool_call",
]
