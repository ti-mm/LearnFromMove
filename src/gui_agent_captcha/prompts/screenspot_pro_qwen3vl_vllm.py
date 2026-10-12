from __future__ import annotations

import json
import math
import re
from base64 import b64encode
from io import BytesIO
from decimal import Decimal, ROUND_HALF_UP
from typing import Any


UPSTREAM_COMMIT = "dbe00114bc53a32c61c1a267786da85967710da8"
PROMPT_PROFILE = "screenspot_pro_qwen3vl_vllm_dbe00114"
IMAGE_SIZE_FACTOR = 32
IMAGE_MIN_PIXELS = 32 * 32
IMAGE_MAX_PIXELS = 99_999_999
TEMPERATURE = 0.7
MAX_TOKENS = 100

GUIDED_PROMPT_TEMPLATE = """<|im_start|>You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{{instruction}}<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ["""

GUIDED_PROMPT_SHA256 = "14b61d6a5b38425d329b71c7ce505392bc89e69f63d78fa42c34904e0d34913a"
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def build_qwen3vl_official_messages(
    instruction: str,
    image: Any,
) -> list[dict[str, Any]]:
    """Build the message contract used by the upstream Transformers evaluator.

    This mirrors ``models/qwen3vl.py:get_qwen3vl_prompt_msg`` from the pinned
    ScreenSpot-Pro evaluator.  The image is represented as a data URL because
    that is the upstream message shape; the processor receives the same PIL
    image separately when it tokenizes the batch.
    """

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = b64encode(buffer.getvalue()).decode("ascii")
    system_text = GUIDED_PROMPT_TEMPLATE.split("<|im_start|>user\n", 1)[0]
    system_text = system_text[len("<|im_start|>") : -len("<|im_end|>\n")]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64," + image_data},
                },
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_guided_prompt(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    return GUIDED_PROMPT_TEMPLATE.replace("{{instruction}}", instruction)


def _half_up(value: Decimal) -> int:
    return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def pixel_to_qwen3vl_1000(
    x: int | float,
    y: int | float,
    width: int,
    height: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    values = (x, y)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("coordinates must be numeric")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("coordinates must be finite")
    qx = _half_up(Decimal(str(x)) * Decimal(1000) / Decimal(width))
    qy = _half_up(Decimal(str(y)) * Decimal(1000) / Decimal(height))
    return min(1000, max(0, qx)), min(1000, max(0, qy))


def format_complete_tool_call(x: int, y: int) -> str:
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("Qwen3-VL coordinates must be in [0, 1000]")
    return (
        '<tool_call>\n{"name":"computer_use","arguments":'
        f'{{"action":"left_click","coordinate":[{x},{y}]}}}}\n</tool_call>'
    )


def parse_complete_tool_call(response: str) -> tuple[int, int]:
    matches = _TOOL_CALL_RE.findall(response.strip())
    if len(matches) != 1:
        raise ValueError("response must contain exactly one complete tool call")
    payload: Any = json.loads(matches[0])
    if not isinstance(payload, dict) or payload.get("name") != "computer_use":
        raise ValueError("tool call name must be computer_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or arguments.get("action") != "left_click":
        raise ValueError("tool call action must be left_click")
    coordinate = arguments.get("coordinate")
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
    ):
        raise ValueError("coordinate must contain two integers")
    x, y = coordinate
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        raise ValueError("coordinate is outside [0, 1000]")
    return x, y


def parse_upstream_vllm_response(response: str) -> tuple[int | float, int | float]:
    payload: Any = json.loads(
        response.split("<tool_call>\n")[-1].split("\n</tool_call>")[-2]
    )
    coordinate = payload["arguments"]["coordinate"]
    if len(coordinate) == 2:
        point_x, point_y = coordinate
    elif len(coordinate) == 4:
        x1, y1, x2, y2 = coordinate
        point_x = (x1 + x2) / 2
        point_y = (y1 + y2) / 2
    else:
        raise ValueError("Wrong output format")
    _ = point_x / 1000, point_y / 1000
    return point_x, point_y
