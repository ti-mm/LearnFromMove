from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping

UNIFIED_THREE_ACTION_PROMPT_VERSION = "unified_three_action_tool_call_v1"
UNIFIED_THREE_ACTION_ENABLE_THINKING = True
UNIFIED_THREE_ACTION_SYSTEM_PROMPT = """You are a helpful assistant. The user will give you an instruction, and you MUST interact with the corresponding UI element via tool call. If you are not sure about where to interact, guess a most likely one.

# Tools

You may call one function to assist with the user query at each step.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\\n* The screen's resolution is 1000x1000.\\n* Make sure to move the cursor tip to the center of the corresponding buttons, links, icons, text, controls, etc.\\n* You can use only these actions to interact with the computer: move_to, mouse_down, mouse_up.\\n* Use move_to to move the cursor without pressing any mouse button.\\n* Use mouse_down to press and hold the left mouse button at the current cursor position.\\n* Use mouse_up to release the left mouse button at the current cursor position.\\n* To click a UI element, first use move_to to position the cursor on the element, then use mouse_down, then use mouse_up.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `move_to`: Move the mouse cursor to coordinate (x, y) without pressing any mouse button.\\n* `mouse_down`: Press and hold the left mouse button at the current cursor position.\\n* `mouse_up`: Release the left mouse button at the current cursor position.", "enum": ["move_to", "mouse_down", "mouse_up"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse cursor to. Required only by `action=move_to`. Do not provide coordinate for `action=mouse_down` or `action=mouse_up`.", "type": "array"}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Return exactly one tool call at each step. Do not return multiple tool calls in one response.
For move_to, the arguments MUST include coordinate: [x, y].
For mouse_down and mouse_up, the arguments MUST NOT include coordinate.
Coordinates must use the 1000x1000 screen coordinate system: x=0 is the left edge, x=1000 is the right edge; y=0 is the top edge, y=1000 is the bottom edge.
Do not output pixel coordinates from the original image resolution and do not output 0-1 decimal coordinates."""

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", flags=re.DOTALL)
ALLOWED_THREE_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up")


def build_unified_three_action_chat_prompt(
    instruction: str,
    *,
    enable_thinking: bool = False,
) -> str:
    del enable_thinking
    prompt = (
        f"<|im_start|>system\n{UNIFIED_THREE_ACTION_SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def format_unified_three_action_tool_call(action: Mapping[str, Any]) -> str:
    kind = action.get("kind")
    if kind == "move_to":
        x_value = action.get("x")
        y_value = action.get("y")
        if (
            isinstance(x_value, bool)
            or isinstance(y_value, bool)
            or not isinstance(x_value, (int, float))
            or not isinstance(y_value, (int, float))
            or not math.isfinite(float(x_value))
            or not math.isfinite(float(y_value))
        ):
            raise ValueError("move_to requires finite numeric x and y")
        arguments: dict[str, Any] = {
            "action": "move_to",
            "coordinate": [float(x_value), float(y_value)],
        }
    elif kind in {"mouse_down", "mouse_up"}:
        arguments = {"action": str(kind)}
    else:
        raise ValueError(f"unsupported three-action kind: {kind!r}")
    payload = {"name": "computer_use", "arguments": arguments}
    return (
        "<tool_call>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</tool_call>"
    )


def _valid_coordinate(values: Any) -> list[float] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return None
    parsed: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        parsed.append(number)
    return parsed


def _parse_result(
    response: str,
    *,
    action: dict[str, Any] | None,
    point: list[float] | None,
    parse_error: str | None,
    tool_call_count: int,
) -> dict[str, Any]:
    return {
        "result": "positive",
        "format": UNIFIED_THREE_ACTION_PROMPT_VERSION,
        "raw_response": response,
        "bbox": None,
        "point": point,
        "action": action,
        "parse_error": parse_error,
        "tool_call_count": tool_call_count,
    }


def parse_unified_three_action_response(response: str) -> dict[str, Any]:
    tool_calls = TOOL_CALL_RE.findall(response)
    tool_call_count = len(tool_calls)
    if tool_call_count == 0:
        return _parse_result(
            response,
            action=None,
            point=None,
            parse_error="missing_complete_tool_call",
            tool_call_count=tool_call_count,
        )
    if tool_call_count > 1:
        return _parse_result(
            response,
            action=None,
            point=None,
            parse_error="multiple_tool_calls",
            tool_call_count=tool_call_count,
        )
    try:
        payload = json.loads(tool_calls[0].strip())
    except json.JSONDecodeError as exc:
        return _parse_result(
            response,
            action=None,
            point=None,
            parse_error=f"JSONDecodeError:{exc}",
            tool_call_count=tool_call_count,
        )
    if not isinstance(payload, dict):
        parse_error = "tool_call_not_object"
    elif payload.get("name") != "computer_use":
        parse_error = "unexpected_tool_name"
    elif not isinstance(payload.get("arguments"), dict):
        parse_error = "arguments_not_object"
    else:
        arguments = payload["arguments"]
        action_name = arguments.get("action")
        if action_name not in ALLOWED_THREE_ACTION_KINDS:
            parse_error = "unexpected_action"
        elif action_name == "move_to":
            if "coordinate" not in arguments:
                parse_error = "move_to_requires_coordinate"
            else:
                coordinate = _valid_coordinate(arguments["coordinate"])
                if coordinate is None:
                    parse_error = "move_to_requires_coordinate"
                elif not (0 <= coordinate[0] <= 1000 and 0 <= coordinate[1] <= 1000):
                    parse_error = "move_to_coordinate_out_of_range"
                else:
                    return _parse_result(
                        response,
                        action={
                            "kind": "move_to",
                            "x": coordinate[0],
                            "y": coordinate[1],
                        },
                        point=[coordinate[0] / 1000, coordinate[1] / 1000],
                        parse_error=None,
                        tool_call_count=tool_call_count,
                    )
        elif "coordinate" in arguments:
            parse_error = f"{action_name}_coordinate_not_allowed"
        else:
            return _parse_result(
                response,
                action={"kind": action_name},
                point=None,
                parse_error=None,
                tool_call_count=tool_call_count,
            )
    return _parse_result(
        response,
        action=None,
        point=None,
        parse_error=parse_error,
        tool_call_count=tool_call_count,
    )
