from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping

from .prompts.unified_three_action import UNIFIED_THREE_ACTION_SYSTEM_PROMPT

PROTOCOL_VERSION = "think_tag_json_action_v3"
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"
CANONICAL_ASSISTANT_FIELD = "assistant_response"
CANONICAL_THINK_FIELD = "think_text"
LEGACY_THINK_FIELDS = ("thought", "thinking_process")
THINK_TEXT_INPUT_FIELDS = (CANONICAL_THINK_FIELD,) + LEGACY_THINK_FIELDS
STRICT_MOUSE_PRIMITIVE_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up")
MOVE_LEFT_CLICK_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up", "left_click", "drag")
MOVE_LEFT_CLICK_CLICKXY_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up", "left_click", "drag", "click")
DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS = MOVE_LEFT_CLICK_ACTION_KINDS
OSWORLD_MOUSE_ACTION_KINDS = ("click", "drag", "move_to", "mouse_down", "mouse_up")
OSWORLD_MOUSE_ACTION_DISPLAY_KINDS = ("click", "drag", "move_to", "mouse_down", "mouse_up")
ROTATION_TASK_REQUIREMENT = "Rotate the central circular region to align it with the background."
OUTER_ROTATION_TASK_REQUIREMENT = (
    "Rotate the image outside the fixed central circular region to align it with the center."
)
DEFAULT_PRIMITIVE_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "type_text",
    "wait",
    "done",
)
DEFAULT_MACRO_ACTION_KINDS = (
    "click",
    "drag",
    "submit",
    "paste_text",
    "key_press",
    "scroll",
    "browser_back",
    "left_double",
    "right_single",
    "type",
    "hotkey",
    "finished",
)
SUPPORTED_OUTPUT_ACTION_KINDS = DEFAULT_PRIMITIVE_ACTION_KINDS + DEFAULT_MACRO_ACTION_KINDS

_TEN_CHOICE_LABEL_PATTERNS = (
    re.compile(r'one\s+labeled\s+"([^"]+)"', re.IGNORECASE),
    re.compile(r"one\s+labeled\s+'([^']+)'", re.IGNORECASE),
    re.compile(r'icon\s+that\s+displays\s+"([^"]+)"', re.IGNORECASE),
    re.compile(r"icon\s+that\s+displays\s+'([^']+)'", re.IGNORECASE),
)

GROUNDCUA_AGENT_SYSTEM_PROMPT = UNIFIED_THREE_ACTION_SYSTEM_PROMPT


@dataclass(frozen=True)
class ProtocolTrack:
    track_id: str
    action_space_type: str
    action_paradigm: str
    task_action_kinds: tuple[str, ...]
    protocol_version: str = PROTOCOL_VERSION
    thought_required: bool = True


def wrap_think_text(thought: str) -> str:
    normalized = thought.strip()
    if not normalized:
        raise ValueError("thought text must not be empty")
    return f"{THINK_OPEN_TAG}{normalized}{THINK_CLOSE_TAG}"


def render_think_json_response(thought: str, payload: dict[str, object]) -> str:
    return (
        wrap_think_text(thought)
        + "\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def render_action_assistant_response(thought: str, action: Mapping[str, object]) -> str:
    return render_think_json_response(thought, {"action": dict(action)})


def render_plan_assistant_response(thought: str, plan: Iterable[Mapping[str, object]]) -> str:
    return render_think_json_response(thought, {"plan": [dict(step) for step in plan]})


def build_ten_choice_instruction(target_label: str) -> str:
    return f'Click the icon that displays "{target_label}".'


def extract_ten_choice_target_label(instruction: str | None) -> str | None:
    text = str(instruction or "").strip()
    if not text:
        return None
    for pattern in _TEN_CHOICE_LABEL_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            label = match.group(1).strip()
            if label:
                return label
    return None


def split_think_json_response(text: str) -> tuple[str | None, str]:
    stripped = text.strip()
    if not stripped.startswith(THINK_OPEN_TAG):
        return None, stripped
    think_end = stripped.find(THINK_CLOSE_TAG)
    if think_end < 0:
        return None, stripped
    thought = stripped[len(THINK_OPEN_TAG) : think_end].strip() or None
    remainder = stripped[think_end + len(THINK_CLOSE_TAG) :].strip()
    return thought, remainder


def _coerce_think_text(value: object, *, action_index: int | None = None) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if action_index is None:
        return None
    if isinstance(value, Mapping):
        for key in (action_index, str(action_index)):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, list) and 0 <= action_index < len(value):
        item = value[action_index]
        if isinstance(item, str) and item.strip():
            return item.strip()
    return None


def _coerce_assistant_response(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_assistant_response(
    payload: Mapping[str, object] | None,
) -> str | None:
    if payload is None:
        return None
    return _coerce_assistant_response(payload.get(CANONICAL_ASSISTANT_FIELD))


def extract_think_text(
    payload: Mapping[str, object] | None,
    *,
    action_index: int | None = None,
) -> str | None:
    if payload is None:
        return None
    assistant_response = extract_assistant_response(payload)
    if assistant_response is not None:
        thought, _remainder = split_think_json_response(assistant_response)
        if thought is not None:
            return thought
    for key in THINK_TEXT_INPUT_FIELDS:
        text = _coerce_think_text(payload.get(key), action_index=action_index)
        if text is not None:
            return text
    return None


def set_canonical_assistant_response(
    payload: MutableMapping[str, Any],
    assistant_response: str | None,
) -> None:
    payload.pop(CANONICAL_ASSISTANT_FIELD, None)
    for key in THINK_TEXT_INPUT_FIELDS:
        payload.pop(key, None)
    if isinstance(assistant_response, str) and assistant_response.strip():
        payload[CANONICAL_ASSISTANT_FIELD] = assistant_response.strip()


def build_canonical_assistant_response(
    *,
    thought: str,
    action: Mapping[str, object] | None = None,
    plan: Iterable[Mapping[str, object]] | None = None,
) -> str:
    if action is not None and plan is not None:
        raise ValueError("assistant response cannot contain both action and plan")
    if action is not None:
        return render_action_assistant_response(thought, action)
    if plan is not None:
        return render_plan_assistant_response(thought, plan)
    raise ValueError("assistant response requires either action or plan payload")


def set_canonical_think_text(
    payload: MutableMapping[str, Any],
    think_text: str | None,
) -> None:
    for key in THINK_TEXT_INPUT_FIELDS:
        payload.pop(key, None)
    if isinstance(think_text, str) and think_text.strip():
        payload[CANONICAL_THINK_FIELD] = think_text.strip()


def _is_hover_reveal_episode_id(normalized_task_type: str) -> bool:
    return normalized_task_type.startswith("hr_") or normalized_task_type.startswith("real_hr_")


def is_ten_choice_captcha_task(task_type: str | None) -> bool:
    if task_type is None:
        return False
    normalized = task_type.lower()
    return "ten_choice_captcha" in normalized or _is_hover_reveal_episode_id(normalized)


def canonicalize_task_requirement(
    instruction: str,
    *,
    task_type: str | None,
    target_label: str | None = None,
) -> str:
    """Return the single agent-facing requirement used across train and eval prompts."""

    if is_rotation_captcha_task(task_type):
        return ROTATION_TASK_REQUIREMENT
    if not is_ten_choice_captcha_task(task_type):
        return instruction
    label = str(target_label or "").strip() or extract_ten_choice_target_label(instruction)
    if not label:
        return instruction
    return build_ten_choice_instruction(label)


def is_rotation_captcha_task(task_type: str | None) -> bool:
    if task_type is None:
        return False
    normalized = task_type.lower()
    return (
        "rotation_captcha" in normalized
        or "interaction_rotation" in normalized
    )


def is_slot_drag_task(task_type: str | None) -> bool:
    if task_type is None:
        return False
    normalized = task_type.lower()
    return (
        "slot_drag_game" in normalized
        or normalized.startswith("sdg_")
        or "slot_drag" in normalized
        or normalized == "slotdraggame"
    )


def is_third_person_drag_task(task_type: str | None) -> bool:
    if task_type is None:
        return False
    normalized = task_type.lower()
    return (
        "third_person_drag_captcha" in normalized
        or normalized.startswith("tpd_")
        or normalized == "thirdpersondragcaptcha"
        or normalized == "third_person_drag"
    )


def is_groundcua_progressive_task(task_type: str | None) -> bool:
    if task_type is None:
        return False
    normalized = task_type.lower()
    return normalized in {
        "grounding_multistep_static",
        "groundcua_static_progressive",
        "groundcua_multistep_static",
    }


def is_mouse_captcha_task(task_type: str | None) -> bool:
    return (
        is_ten_choice_captcha_task(task_type)
        or is_rotation_captcha_task(task_type)
        or is_slot_drag_task(task_type)
        or is_third_person_drag_task(task_type)
        or is_groundcua_progressive_task(task_type)
    )


def is_self_built_osworld_mouse_task(task_type: str | None) -> bool:
    return is_mouse_captcha_task(task_type)


def supported_action_kinds(action_kinds: Iterable[str] | None) -> tuple[str, ...]:
    if action_kinds is None:
        return ()
    return tuple(kind for kind in action_kinds if kind in SUPPORTED_OUTPUT_ACTION_KINDS)


def _macro_kinds(action_kinds: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(kind for kind in action_kinds if kind in DEFAULT_MACRO_ACTION_KINDS)


def _primitive_kinds(action_kinds: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(kind for kind in action_kinds if kind in DEFAULT_PRIMITIVE_ACTION_KINDS)


def _normalize_mouse_captcha_action_kinds(action_kinds: tuple[str, ...]) -> tuple[str, ...]:
    if not action_kinds:
        return DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS
    strict_legacy = set(STRICT_MOUSE_PRIMITIVE_ACTION_KINDS) | {"done"}
    if set(action_kinds).issubset(strict_legacy):
        return DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS
    return action_kinds


def resolve_protocol_track(
    *,
    task_type: str | None,
    action_kinds: Iterable[str] | None = None,
) -> ProtocolTrack:
    supported = supported_action_kinds(action_kinds)

    if is_groundcua_progressive_task(task_type):
        return ProtocolTrack(
            track_id="groundcua_static_progressive",
            action_space_type="strict_mouse_primitives",
            action_paradigm="primitive",
            task_action_kinds=STRICT_MOUSE_PRIMITIVE_ACTION_KINDS,
        )

    if is_mouse_captcha_task(task_type):
        supported = _normalize_mouse_captcha_action_kinds(supported)
        macros = _macro_kinds(supported)
        primitives = _primitive_kinds(supported)
        if macros and primitives:
            track_id = (
                "captcha_mixed_left_click_clickxy"
                if "click" in macros
                else "captcha_mixed_left_click"
            )
            return ProtocolTrack(
                track_id=track_id,
                action_space_type="mixed_primitive_macro",
                action_paradigm="mixed",
                task_action_kinds=supported,
            )
        if macros:
            return ProtocolTrack(
                track_id="captcha_old_legacy",
                action_space_type="legacy_macro",
                action_paradigm="macro_only",
                task_action_kinds=macros,
            )
        return ProtocolTrack(
            track_id="captcha_mixed_left_click",
            action_space_type="mixed_primitive_macro",
            action_paradigm="mixed",
            task_action_kinds=DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS,
        )

    macros = _macro_kinds(supported)
    primitives = _primitive_kinds(supported)

    if macros and primitives:
        return ProtocolTrack(
            track_id="traditional_new_mixed",
            action_space_type="mixed_primitive_macro",
            action_paradigm="mixed",
            task_action_kinds=supported,
        )

    if macros:
        return ProtocolTrack(
            track_id="traditional_old_legacy",
            action_space_type="legacy_macro",
            action_paradigm="macro_only",
            task_action_kinds=macros,
        )

    return ProtocolTrack(
        track_id="traditional_new_mixed",
        action_space_type="mixed_primitive_macro",
        action_paradigm="primitive",
        task_action_kinds=supported or DEFAULT_PRIMITIVE_ACTION_KINDS,
    )
