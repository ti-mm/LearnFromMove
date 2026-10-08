"""Frozen ScreenSpot-Pro-compatible GroundCUA prompt contracts.

The model-visible contracts in this module deliberately never use GroundCUA's
legacy ``{"action": {"kind": ...}}`` protocol.  Internal ``kind`` values may
be converted by data builders, but only the function-call representation here
is sent to or learned by the model.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

from .screenspot_pro_qwen3vl_vllm import GUIDED_PROMPT_TEMPLATE as _QWEN3_DIRECT_TEMPLATE
from .screenspot_qwen25_official_base import (
    OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT,
)
from .screenspot_qwen25_mousemove import (
    QWEN25_MOUSEMOVE_LEFTCLICK_ACTION_CONTRACT,
    QWEN25_MOUSEMOVE_LEFTCLICK_GUIDED_PROMPT,
    QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_PROFILE,
)

UPSTREAM_COMMIT = "dbe00114bc53a32c61c1a267786da85967710da8"
QWEN25_UPSTREAM_SHA256 = (
    "741b9351bc311289f84e7141e16edd186d022ea742dbed94130a29e5910058c2"
)
QWEN3_UPSTREAM_SHA256 = (
    "9e4d325bd2e01945ede2ef1776333c4d696d0c84fa0d6c924fc81e9e23930105"
)

QWEN25_DIRECT_PROFILE = "screenspot_pro_qwen25_direct_dbe00114"
QWEN25_MOVE_PROFILE = "screenspot_pro_qwen25_move_dbe00114"
QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE = "screenspot_pro_qwen25_official_dbe00114"
QWEN25_MOUSEMOVE_LEFTCLICK_THINK_PROFILE = (
    "groundcua46k_qwen25_mousemove_leftclick_think_v1"
)
QWEN25_MOUSEMOVE_LEFTCLICK_THINK_RED_CURSOR_PROFILE = (
    "groundcua46k_qwen25_mousemove_leftclick_think_red_cursor_v2"
)
QWEN25_MOUSEMOVE_LEFTCLICK_SHARED_WHITE_PROFILE = QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_PROFILE
OFFICIAL_QWEN25_ACTION_CONTRACT = "direct_left_click_coordinate_full_tool_call"
CUSTOM_QWEN25_ACTION_CONTRACT = "mouse_move_then_coordinate_free_left_click"
QWEN3_DIRECT_PROFILE = "screenspot_pro_qwen3_direct_dbe00114"
QWEN3_MOVE_PROFILE = "screenspot_pro_qwen3_move_dbe00114"
QWEN3_MOVE_SEQUENTIAL_PROMPT_CONTRACT = (
    "qwen3_move_official_multiturn_all_history_last3_images_v2"
)
QWEN3_MOVE_IMAGE_HISTORY_MAX = 3

# GroundCUA 46k's Qwen3 training profiles are deliberately separate from the
# older ScreenSpot-Pro parity profiles.  They expose the model-visible
# post-normalization contract: a move followed by a current-cursor click,
# with the teacher's reasoning in a think block.
QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE = (
    "groundcua46k_qwen3_moveto_leftclick_think_v1"
)
GROUNDCUA_RED_CURSOR_SENTENCE = "The red mouse icon is the actual mouse cursor."
GROUNDCUA_RED_CURSOR_PROMPT_CONTRACT = "groundcua46k_red_cursor_prompt_v2"
GROUNDCUA_SHARED_THINK_ACTION_SENTENCE = (
    "think briefly, then perform the next mouse action for the corresponding UI element via tool call."
)
GROUNDCUA_FULL_DROP_PROMPT_CONTRACT = "groundcua46k_full_drop_shared_no_tag_v1"
QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROFILE = (
    "groundcua46k_qwen3_moveto_leftclick_think_red_cursor_v2"
)
QWEN3_MOVETO_LEFTCLICK_TOOLCALL_ONLY_RED_CURSOR_PROFILE = (
    "groundcua46k_qwen3_moveto_leftclick_no_think_tags_red_cursor_v2"
)
QWEN3_MOVETO_LEFTCLICK_THINK_NO_TAGS_RED_CURSOR_PROFILE = (
    "groundcua46k_qwen3_moveto_leftclick_think_no_tags_red_cursor_v3"
)
QWEN25_MOVE_ACTIONS = (
    "key", "type", "mouse_move", "left_click", "left_click_drag",
    "right_click", "middle_click", "double_click", "scroll", "wait", "terminate",
)
# These legacy action tuples remain import-compatible so historical callers
# reach the explicit profile rejection instead of a hidden prompt fallback.
QWEN25_MOVE_COMPAT_ACTIONS = QWEN25_MOVE_ACTIONS + ("mouse_down", "mouse_up")

QWEN25_LEGACY_PROFILE_NAMES = frozenset(
    {
        QWEN25_MOVE_PROFILE,
        QWEN25_MOUSEMOVE_LEFTCLICK_THINK_PROFILE,
        QWEN25_MOUSEMOVE_LEFTCLICK_THINK_RED_CURSOR_PROFILE,
    }
)
_QWEN3_MOVE_ACTION_SECTION_OLD = '''"description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. 
* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}'''
_QWEN3_MOVE_ACTION_SECTION_NEW = '''"description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to place the cursor tip in the center of the corresponding UI element.
* You can use only move_to, mouse_down, and mouse_up to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `move_to`: Move the cursor to coordinate (x, y).
* `mouse_down`: press and hold the left mouse button.
* `mouse_up`: release the left mouse button.", "enum": ["move_to", "mouse_down", "mouse_up"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=move_to`.", "type": "array"}, "required": ["action"], "type": "object"}}}'''
_QWEN3_MOVE_SYSTEM_PREFIX_OLD = (
    "You are a helpful assistant. The user will give you an instruction, and you MUST "
    "left click on the corresponding UI element via tool call. If you are not sure "
    "about where to click, guess a most likely one."
)
_QWEN3_MOVE_SYSTEM_PREFIX_NEW = (
    "You are a helpful assistant. The user will give you an instruction, and you MUST "
    "perform the next mouse primitive for the corresponding UI element via tool call. "
    "If you are not sure about where to move, guess a most likely one."
)

_QWEN3_MOVETO_LEFTCLICK_ACTION_SECTION = '''"description": "Use a mouse to interact with a computer.
* The screen's resolution is 1000x1000.
* Make sure to place the cursor tip in the center of the corresponding UI element.
* You can use only move_to and left_click to interact with the computer. A left_click uses the current cursor position and has no coordinate argument.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:
* `move_to`: Move the cursor to coordinate (x, y).
* `left_click`: Click the left mouse button at the current cursor position; do not provide coordinates.", "enum": ["move_to", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): Integer relative coordinates in the 0..1000 range. Required only by `action=move_to`.", "type": "array"}}, "required": ["action"], "type": "object"}}}'''


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    occurrences = text.count(old)
    if occurrences != 1:
        raise RuntimeError(f"expected one {label} replacement, found {occurrences}")
    return text.replace(old, new)


def _build_qwen3_move_template() -> str:
    result = _replace_once(
        _QWEN3_DIRECT_TEMPLATE,
        _QWEN3_MOVE_SYSTEM_PREFIX_OLD,
        _QWEN3_MOVE_SYSTEM_PREFIX_NEW,
        label="Qwen3 move system action instruction",
    )
    result = _replace_once(
        result,
        _QWEN3_MOVE_ACTION_SECTION_OLD,
        _QWEN3_MOVE_ACTION_SECTION_NEW,
        label="Qwen3 move tool action section",
    )
    suffix_index = result.rfind("<|im_start|>assistant\n")
    if suffix_index == -1:
        raise RuntimeError("Qwen3 guided prompt has no assistant boundary")
    return result[: suffix_index + len("<|im_start|>assistant\n")]


QWEN3_MOVE_PROMPT_TEMPLATE = _build_qwen3_move_template()


def _build_qwen3_moveto_leftclick_think_template() -> str:
    result = _replace_once(
        QWEN3_MOVE_PROMPT_TEMPLATE,
        _QWEN3_MOVE_ACTION_SECTION_NEW,
        _QWEN3_MOVETO_LEFTCLICK_ACTION_SECTION,
        label="Qwen3 GroundCUA move/click action section",
    )
    result = _replace_once(
        result,
        "perform the next mouse primitive for the corresponding UI element via tool call.",
        GROUNDCUA_SHARED_THINK_ACTION_SENTENCE
        + " Return exactly one <think>...</think> block followed by one complete <tool_call>.",
        label="Qwen3 GroundCUA think instruction",
    )
    suffix_index = result.rfind("<|im_start|>assistant\n")
    return result[: suffix_index + len("<|im_start|>assistant\n")]


QWEN3_MOVETO_LEFTCLICK_THINK_PROMPT_TEMPLATE = (
    _build_qwen3_moveto_leftclick_think_template()
)


def _with_groundcua_red_cursor_sentence(prompt_template: str) -> str:
    """Insert the shared red-cursor fact once, before tools and output rules."""

    if GROUNDCUA_RED_CURSOR_SENTENCE in prompt_template:
        raise RuntimeError("GroundCUA prompt already contains the red-cursor sentence")
    marker = "# Tools"
    if prompt_template.count(marker) != 1:
        raise RuntimeError("GroundCUA prompt must contain exactly one tools heading")
    tools_index = prompt_template.index(marker)
    user_boundary = prompt_template.index("<|im_end|>\n<|im_start|>user\n")
    if tools_index >= user_boundary:
        raise RuntimeError("GroundCUA tools heading must be inside the system message")
    return (
        prompt_template[:tools_index]
        + GROUNDCUA_RED_CURSOR_SENTENCE
        + "\n\n"
        + prompt_template[tools_index:]
    )


QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROMPT_TEMPLATE = (
    _with_groundcua_red_cursor_sentence(QWEN3_MOVETO_LEFTCLICK_THINK_PROMPT_TEMPLATE)
)

_QWEN3_TAGGED_SYSTEM_CLAUSE = (
    GROUNDCUA_SHARED_THINK_ACTION_SENTENCE
    + " "
    "Return exactly one <think>...</think> block followed by one complete <tool_call>."
)
_QWEN3_TOOLCALL_ONLY_SYSTEM_CLAUSE = (
    "perform the next mouse action for the corresponding UI element via tool call."
)
_QWEN3_SHARED_NO_TAG_SYSTEM_CLAUSE = (
    GROUNDCUA_SHARED_THINK_ACTION_SENTENCE
)
QWEN3_MOVETO_LEFTCLICK_TOOLCALL_ONLY_RED_CURSOR_PROMPT_TEMPLATE = _replace_once(
    QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROMPT_TEMPLATE,
    _QWEN3_TAGGED_SYSTEM_CLAUSE,
    _QWEN3_TOOLCALL_ONLY_SYSTEM_CLAUSE,
    label="Qwen3 GroundCUA toolcall-only instruction",
)
QWEN3_MOVETO_LEFTCLICK_THINK_NO_TAGS_RED_CURSOR_PROMPT_TEMPLATE = _replace_once(
    QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROMPT_TEMPLATE,
    _QWEN3_TAGGED_SYSTEM_CLAUSE,
    _QWEN3_SHARED_NO_TAG_SYSTEM_CLAUSE,
    label="Qwen3 GroundCUA shared no-tag Think instruction",
)

@dataclass(frozen=True)
class GroundCUAProfile:
    name: str
    model_family: str
    paradigm: str
    coordinate_format: str
    image_factor: int
    image_min_pixels: int
    image_max_pixels: int
    temperature: float
    max_tokens: int
    prompt_template: str
    assistant_prefill: str
    allowed_actions: tuple[str, ...]
    scoring_action: str
    upstream_file_sha256: str
    source_kind: str
    # Most historical profiles use the shared coordinate action set.  New
    # GroundCUA training profiles override it so current-cursor left_click is
    # represented without x/y fields.
    coordinate_actions: frozenset[str] | None = None
    response_protocol: str = "toolcall_only"

    @property
    def prompt_template_sha256(self) -> str:
        return hashlib.sha256(self.prompt_template.encode("utf-8")).hexdigest()


_PROFILES = {
    QWEN25_DIRECT_PROFILE: GroundCUAProfile(
        name=QWEN25_DIRECT_PROFILE,
        model_family="qwen2_5_vl",
        paradigm="direct_click",
        coordinate_format="resized_physical_pixels",
        image_factor=28,
        image_min_pixels=784,
        image_max_pixels=99_999_999,
        temperature=0.0,
        max_tokens=100,
        prompt_template=OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT,
        assistant_prefill="",
        allowed_actions=("left_click",),
        scoring_action="left_click",
        upstream_file_sha256=QWEN25_UPSTREAM_SHA256,
        source_kind="upstream_official",
    ),
    QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE: GroundCUAProfile(
        name=QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE,
        model_family="qwen2_5_vl",
        paradigm="direct_click",
        coordinate_format="resized_physical_pixels",
        image_factor=28,
        image_min_pixels=784,
        image_max_pixels=99_999_999,
        temperature=0.0,
        max_tokens=100,
        prompt_template=OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT,
        assistant_prefill="",
        allowed_actions=("left_click",),
        scoring_action="left_click",
        upstream_file_sha256=QWEN25_UPSTREAM_SHA256,
        source_kind="upstream_official",
    ),
    QWEN25_MOUSEMOVE_LEFTCLICK_SHARED_WHITE_PROFILE: GroundCUAProfile(
        name=QWEN25_MOUSEMOVE_LEFTCLICK_SHARED_WHITE_PROFILE,
        model_family="qwen2_5_vl",
        paradigm="mousemove_leftclick_no_tag",
        coordinate_format="resized_physical_pixels",
        image_factor=28,
        image_min_pixels=784,
        image_max_pixels=99_999_999,
        temperature=0.0,
        max_tokens=512,
        prompt_template=QWEN25_MOUSEMOVE_LEFTCLICK_GUIDED_PROMPT,
        assistant_prefill="",
        allowed_actions=("mouse_move", "left_click"),
        scoring_action="mouse_move",
        upstream_file_sha256="",
        source_kind="groundcua46k_shared_white_no_tag",
        coordinate_actions=frozenset(("mouse_move",)),
        response_protocol="no_tag_reasoning",
    ),
    QWEN3_DIRECT_PROFILE: GroundCUAProfile(
        name=QWEN3_DIRECT_PROFILE,
        model_family="qwen3_vl",
        paradigm="direct_click",
        coordinate_format="qwen3_relative_0_1000",
        image_factor=32,
        image_min_pixels=1024,
        image_max_pixels=99_999_999,
        temperature=0.7,
        max_tokens=100,
        prompt_template=_QWEN3_DIRECT_TEMPLATE,
        assistant_prefill="",
        allowed_actions=("left_click",),
        scoring_action="left_click",
        upstream_file_sha256=QWEN3_UPSTREAM_SHA256,
        source_kind="upstream_official",
    ),
    QWEN3_MOVE_PROFILE: GroundCUAProfile(
        name=QWEN3_MOVE_PROFILE,
        model_family="qwen3_vl",
        paradigm="move_to",
        coordinate_format="qwen3_relative_0_1000",
        image_factor=32,
        image_min_pixels=1024,
        image_max_pixels=99_999_999,
        temperature=0.7,
        max_tokens=100,
        prompt_template=QWEN3_MOVE_PROMPT_TEMPLATE,
        assistant_prefill="",
        allowed_actions=("move_to", "mouse_down", "mouse_up"),
        scoring_action="move_to",
        upstream_file_sha256=QWEN3_UPSTREAM_SHA256,
        source_kind="screenspot_pro_tool_use_derivation",
    ),
    QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE: GroundCUAProfile(
        name=QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE,
        model_family="qwen3_vl",
        paradigm="moveto_leftclick_think",
        coordinate_format="qwen3_relative_0_1000",
        image_factor=32,
        image_min_pixels=1024,
        image_max_pixels=99_999_999,
        temperature=0.7,
        max_tokens=512,
        prompt_template=QWEN3_MOVETO_LEFTCLICK_THINK_PROMPT_TEMPLATE,
        assistant_prefill="",
        allowed_actions=("move_to", "left_click"),
        scoring_action="move_to",
        upstream_file_sha256=QWEN3_UPSTREAM_SHA256,
        source_kind="groundcua46k_normalized",
        coordinate_actions=frozenset(("move_to",)),
        response_protocol="tagged_think",
    ),
}

_PROFILES.update(
    {
        QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROFILE: replace(
            _PROFILES[QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE],
            name=QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROFILE,
            prompt_template=QWEN3_MOVETO_LEFTCLICK_THINK_RED_CURSOR_PROMPT_TEMPLATE,
            source_kind=GROUNDCUA_RED_CURSOR_PROMPT_CONTRACT,
        ),
        QWEN3_MOVETO_LEFTCLICK_TOOLCALL_ONLY_RED_CURSOR_PROFILE: replace(
            _PROFILES[QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE],
            name=QWEN3_MOVETO_LEFTCLICK_TOOLCALL_ONLY_RED_CURSOR_PROFILE,
            paradigm="moveto_leftclick_toolcall_only",
            prompt_template=QWEN3_MOVETO_LEFTCLICK_TOOLCALL_ONLY_RED_CURSOR_PROMPT_TEMPLATE,
            source_kind=GROUNDCUA_RED_CURSOR_PROMPT_CONTRACT,
            response_protocol="toolcall_only",
        ),
        QWEN3_MOVETO_LEFTCLICK_THINK_NO_TAGS_RED_CURSOR_PROFILE: replace(
            _PROFILES[QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE],
            name=QWEN3_MOVETO_LEFTCLICK_THINK_NO_TAGS_RED_CURSOR_PROFILE,
            paradigm="moveto_leftclick_think_no_tags",
            prompt_template=QWEN3_MOVETO_LEFTCLICK_THINK_NO_TAGS_RED_CURSOR_PROMPT_TEMPLATE,
            source_kind=GROUNDCUA_RED_CURSOR_PROMPT_CONTRACT,
            response_protocol="no_tag_think",
        ),
    }
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_THINK_TOOL_CALL_RE = re.compile(
    r"\A\s*<think>(.*?)</think>\s*(<tool_call>\s*.*?\s*</tool_call>)\s*\Z",
    re.DOTALL,
)
_ACTIONS_WITH_COORDINATE = frozenset(("left_click", "mouse_move", "move_to", "left_click_drag"))


class UnsupportedQwen25ContractError(ValueError):
    """Raised when a removed Qwen2.5 variant is selected."""


def reject_qwen25_contract(
    *,
    profile_name: str,
    action_contract: str,
    reason: str,
) -> None:
    raise UnsupportedQwen25ContractError(
        "unsupported official ScreenSpot-Pro Qwen2.5 execution contract: "
        f"profile={profile_name!r}; action_contract={action_contract!r}; {reason}"
    )


def require_official_qwen25_direct_click(
    profile_name: str,
    *,
    action_contract: str = OFFICIAL_QWEN25_ACTION_CONTRACT,
    multi_turn: bool = False,
    has_think: bool = False,
    preserves_history: bool = False,
) -> None:
    """Allow only the unchanged one-call official Qwen2.5 contract."""

    if profile_name not in {
        QWEN25_DIRECT_PROFILE,
        QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE,
    }:
        reject_qwen25_contract(
            profile_name=profile_name,
            action_contract=action_contract,
            reason=(
                "only the official direct Qwen2.5 profiles are registered; "
                "derived profiles have no fallback"
            ),
        )
    if (
        action_contract != OFFICIAL_QWEN25_ACTION_CONTRACT
        or multi_turn
        or has_think
        or preserves_history
    ):
        reject_qwen25_contract(
            profile_name=profile_name,
            action_contract=action_contract,
            reason=(
                "the official path is one direct coordinate-bearing left_click; "
                "the requested Think/history or multi-turn action trajectory is "
                "not losslessly convertible"
            ),
        )


def get_groundcua_profile(name: str) -> GroundCUAProfile:
    if isinstance(name, str) and (
        name in QWEN25_LEGACY_PROFILE_NAMES
        or (
            "qwen25" in name.lower()
            and name
            not in {
                QWEN25_DIRECT_PROFILE,
                QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE,
                QWEN25_MOUSEMOVE_LEFTCLICK_SHARED_WHITE_PROFILE,
            }
        )
        or (
            "qwen2_5" in name.lower()
            and name
            not in {
                QWEN25_DIRECT_PROFILE,
                QWEN25_SCREENSPOT_PRO_OFFICIAL_PROFILE,
                QWEN25_MOUSEMOVE_LEFTCLICK_SHARED_WHITE_PROFILE,
            }
        )
    ):
        reject_qwen25_contract(
            profile_name=name,
            action_contract=CUSTOM_QWEN25_ACTION_CONTRACT,
            reason=(
                "legacy profile is removed; use an official direct profile or "
                "the shared mousemove-prefill profile"
            ),
        )
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown GroundCUA ScreenSpot-Pro profile: {name!r}") from exc


def is_screenspot_pro_groundcua_profile(name: str | None) -> bool:
    return isinstance(name, str) and name in _PROFILES


def build_groundcua_prompt(
    profile_name: str,
    instruction: str,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    profile = get_groundcua_profile(profile_name)
    prompt = profile.prompt_template.replace("{{instruction}}", instruction)
    if profile.model_family == "qwen2_5_vl":
        if not isinstance(screen_width, int) or screen_width < 1:
            raise ValueError("Qwen2.5 prompt requires a positive screen_width")
        if not isinstance(screen_height, int) or screen_height < 1:
            raise ValueError("Qwen2.5 prompt requires a positive screen_height")
        prompt = prompt.replace("{{screen_width}}", str(screen_width)).replace(
            "{{screen_height}}", str(screen_height)
        )
    unresolved = [
        placeholder
        for placeholder in ("{{instruction}}", "{{screen_width}}", "{{screen_height}}")
        if placeholder in prompt
    ]
    if unresolved:
        raise ValueError(
            f"unresolved prompt placeholder for {profile_name}: {', '.join(unresolved)}"
        )
    return prompt


def groundcua_system_prompt(
    profile_name: str,
    *,
    screen_width: int | None = None,
    screen_height: int | None = None,
) -> str:
    """Return only the system message from a frozen raw prompt template."""

    profile = get_groundcua_profile(profile_name)
    prompt = profile.prompt_template
    if profile.model_family == "qwen2_5_vl":
        if not isinstance(screen_width, int) or screen_width < 1:
            raise ValueError("Qwen2.5 system prompt requires a positive screen_width")
        if not isinstance(screen_height, int) or screen_height < 1:
            raise ValueError("Qwen2.5 system prompt requires a positive screen_height")
        prompt = prompt.replace("{{screen_width}}", str(screen_width)).replace(
            "{{screen_height}}", str(screen_height)
        )
    prefix = (
        "<|im_start|>system\n"
        if prompt.startswith("<|im_start|>system\n")
        else "<|im_start|>"
    )
    boundary = "<|im_end|>\n<|im_start|>user\n"
    if not prompt.startswith(prefix) or boundary not in prompt:
        raise RuntimeError(f"profile {profile_name!r} has an invalid raw prompt envelope")
    return prompt[len(prefix) :].split(boundary, 1)[0]


def qwen25_direct_tool_call(x: int, y: int) -> str:
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("Qwen2.5 direct coordinates must be integers")
    if x < 0 or y < 0:
        raise ValueError("Qwen2.5 direct coordinates must be nonnegative")
    return format_groundcua_tool_call(QWEN25_DIRECT_PROFILE, "left_click", coordinate=(x, y))


def format_groundcua_tool_call(
    profile_name: str,
    action: str,
    *,
    coordinate: tuple[int, int] | None = None,
) -> str:
    profile = get_groundcua_profile(profile_name)
    if action not in profile.allowed_actions:
        raise ValueError(f"action {action!r} is not allowed for {profile_name}")
    coordinate_actions = profile.coordinate_actions or _ACTIONS_WITH_COORDINATE
    needs_coordinate = action in coordinate_actions
    if needs_coordinate:
        if coordinate is None or len(coordinate) != 2:
            raise ValueError(f"action {action!r} requires one coordinate pair")
        x, y = coordinate
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
            or x < 0
            or y < 0
        ):
            raise ValueError("coordinates must be nonnegative integers")
        if profile.coordinate_format == "qwen3_relative_0_1000" and (x > 1000 or y > 1000):
            raise ValueError("Qwen3 coordinates must be within 0..1000")
        arguments: dict[str, object] = {"action": action, "coordinate": [x, y]}
    else:
        if coordinate is not None:
            raise ValueError(f"action {action!r} must not have coordinates")
        arguments = {"action": action}
    payload = {"name": "computer_use", "arguments": arguments}
    return "<tool_call>\n" + json.dumps(payload, separators=(",", ":")) + "\n</tool_call>"


def parse_groundcua_tool_call(
    profile_name: str,
    response: str,
) -> tuple[str, tuple[int, int] | None]:
    profile = get_groundcua_profile(profile_name)
    matches = _TOOL_CALL_RE.findall(response.strip())
    if len(matches) != 1:
        raise ValueError("response must contain exactly one complete tool call")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("tool call must contain JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"name", "arguments"}
        or payload.get("name") != "computer_use"
    ):
        raise ValueError("tool call name must be computer_use")
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError("tool call arguments must be an object")
    action = arguments.get("action")
    if not isinstance(action, str) or action not in profile.allowed_actions:
        raise ValueError(f"unexpected action for {profile_name}")
    coordinate_actions = profile.coordinate_actions or _ACTIONS_WITH_COORDINATE
    if action not in coordinate_actions:
        if set(arguments) != {"action"}:
            raise ValueError(f"action {action!r} must not have coordinates")
        return action, None
    if set(arguments) != {"action", "coordinate"}:
        raise ValueError(f"action {action!r} requires only action and coordinate")
    coordinate = arguments.get("coordinate")
    if (
        not isinstance(coordinate, list)
        or len(coordinate) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in coordinate)
    ):
        raise ValueError(f"action {action!r} requires two integer coordinates")
    x, y = coordinate
    if x < 0 or y < 0:
        raise ValueError("coordinates must be nonnegative")
    if profile.coordinate_format == "qwen3_relative_0_1000" and (x > 1000 or y > 1000):
        raise ValueError("Qwen3 coordinates must be within 0..1000")
    return action, (x, y)


def format_groundcua_think_tool_call(
    profile_name: str,
    thought: str,
    action: str,
    coordinate: tuple[int, int] | None = None,
) -> str:
    """Format the exact GroundCUA SFT target: think, then one tool call."""

    if not isinstance(thought, str) or not thought.strip():
        raise ValueError("thought must be nonempty")
    normalized_thought = thought.strip()
    if "<think>" in normalized_thought or "</think>" in normalized_thought:
        raise ValueError("thought must not contain think tags")
    if "<tool_call>" in normalized_thought or "</tool_call>" in normalized_thought:
        raise ValueError("thought must not contain tool-call tags")
    tool_call = format_groundcua_tool_call(
        profile_name,
        action,
        coordinate=coordinate,
    )
    return f"<think>{normalized_thought}</think>\n{tool_call}"


def parse_groundcua_think_tool_call(
    profile_name: str,
    response: str,
) -> tuple[str, str, tuple[int, int] | None]:
    """Strictly parse one nonempty think block followed by one tool call."""

    if not isinstance(response, str):
        raise ValueError("response must be text")
    if '"kind"' in response:
        raise ValueError("legacy kind actions are not model-visible")
    match = _THINK_TOOL_CALL_RE.fullmatch(response)
    if match is None:
        raise ValueError(
            "response must contain exactly one think block followed by one complete tool call"
        )
    thought = match.group(1).strip()
    if not thought:
        raise ValueError("think block must be nonempty")
    if "<think>" in thought or "</think>" in thought:
        raise ValueError("response must contain exactly one think block")
    if "<tool_call>" in thought or "</tool_call>" in thought:
        raise ValueError("think block must not contain a nested tool call")
    tool_call = match.group(2)
    action, coordinate = parse_groundcua_tool_call(profile_name, tool_call)
    return thought, action, coordinate


def parse_groundcua_response(
    profile_name: str,
    response: str,
    *,
    resized_size: tuple[int, int],
) -> tuple[str, tuple[int, int] | None]:
    profile = get_groundcua_profile(profile_name)
    if (
        not isinstance(resized_size, tuple)
        or len(resized_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in resized_size)
    ):
        raise ValueError("resized_size must contain two positive integers")
    action, coordinate = parse_groundcua_tool_call(profile_name, response.strip())
    if action != profile.scoring_action:
        raise ValueError(
            f"response action {action!r} is not the scoring action "
            f"{profile.scoring_action!r} for {profile_name}"
        )
    if coordinate is not None and profile.coordinate_format == "resized_physical_pixels":
        width, height = resized_size
        x, y = coordinate
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError("response coordinate is outside resized image bounds")
    return action, coordinate
