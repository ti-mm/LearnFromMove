from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..actions import Action, AtomicAction, PrimitiveAction, WebAction
from ..core import Observation, StepResult
from ..prompts.unified_three_action import (
    UNIFIED_THREE_ACTION_SYSTEM_PROMPT,
    parse_unified_three_action_response,
)
from ..protocol_tracks import (
    OSWORLD_MOUSE_ACTION_KINDS,
    canonicalize_task_requirement,
    is_self_built_osworld_mouse_task,
    split_think_json_response,
)
from ..train.qwen3_vl_sft import (
    IMAGE_HISTORY_MAX,
    QWEN3_RELATIVE_COORDINATE_FORMAT,
    build_prompt_messages_from_history,
    fold_image_history_items,
)

PLAN_SYSTEM_PROMPT = UNIFIED_THREE_ACTION_SYSTEM_PROMPT
DEFAULT_ACTION_SYSTEM_PROMPT = UNIFIED_THREE_ACTION_SYSTEM_PROMPT

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_TRADITIONAL_MACRO_ACTION_KINDS = (
    "click",
    "drag",
    "submit",
    "left_double",
    "right_single",
    "scroll",
    "type",
    "hotkey",
    "wait",
    "finished",
    "paste_text",
    "key_press",
    "browser_back",
)
_SELF_BUILT_MOUSE_ACTION_KINDS = (
    "click",
    "drag",
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
)


def encode_screenshot_base64(screenshot_path: str) -> tuple[str, str] | None:
    if not screenshot_path:
        return None
    path = Path(screenshot_path)
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime = MIME_TYPES.get(suffix, "image/png")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime, data


def build_action_prompt_text(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None = None,
    condition: str | None = None,
) -> str:
    task_type = _task_type_for_observation(obs)
    if is_self_built_osworld_mouse_task(task_type):
        return _build_osworld_mouse_action_prompt_text(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            task_type=task_type,
            budget=budget,
        )
    return _render_canonical_user_context(
        instruction=obs.instruction,
        history=history,
        budget=budget,
        cursor_xy=obs.cursor_xy,
        condition=condition,
    )


def _render_canonical_user_context(
    *,
    instruction: str,
    history: list[StepResult],
    budget: int | None,
    cursor_xy: tuple[float, float] | None,
    condition: str | None = None,
    task_type: str | None = None,
) -> str:
    lines = [
        f"Task: {instruction}",
        f"Attached images are ordered chronologically; at most the latest {IMAGE_HISTORY_MAX} screenshots are attached, and the last attached image is the current screenshot.",
    ]
    if task_type:
        lines.append(f"Task type: {task_type}")
    if condition:
        lines.append(f"Condition: {condition}")
    if budget is not None:
        lines.append(f"Budget (max VLM calls): {budget}")
    if cursor_xy is not None:
        lines.append(f"Current cursor position: ({cursor_xy[0]:.1f}, {cursor_xy[1]:.1f})")
    if history:
        lines.append(f"Total steps so far: {len(history)}")
        lines.append("Action history so far, in order:")
        for index, step in enumerate(history, start=1):
            lines.append(_format_history_step(index, step))
    return "\n".join(lines)


def _task_type_for_observation(obs: Observation) -> str | None:
    metadata = obs.metadata or {}
    for key in ("task_type", "benchmark", "task_id", "episode_id"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_osworld_mouse_action_prompt_text(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    task_type: str | None,
    budget: int | None,
) -> str:
    instruction = canonicalize_task_requirement(
        obs.instruction,
        task_type=task_type,
        target_label=(obs.metadata or {}).get("target_label"),
    )
    return _render_canonical_user_context(
        instruction=instruction,
        history=history,
        budget=budget,
        cursor_xy=obs.cursor_xy,
        task_type=task_type,
    )


def _format_osworld_mouse_schema_line(kind: str) -> str:
    if kind in {"click", "drag", "move_to", "mouse_down", "mouse_up", "left_click", "done", "submit"}:
        return kind
    return f"- {kind}"


def is_center_locked_cursor_task(obs: Observation) -> bool:
    metadata = obs.metadata or {}
    return (
        metadata.get("cursor_lock") == "center"
        or metadata.get("coordinate_contract") == "qwen3_relative_0_1000_center_locked"
    )


def _reasoning_action_kinds_for_prompt(allowed_kinds: Iterable[str]) -> tuple[str, ...]:
    allowed = tuple(dict.fromkeys(str(kind) for kind in allowed_kinds if str(kind)))
    macro_kinds = tuple(kind for kind in allowed if kind in _TRADITIONAL_MACRO_ACTION_KINDS)
    if macro_kinds:
        primitive_kinds = tuple(
            kind for kind in ("move_to", "mouse_down", "mouse_up", "left_click") if kind in allowed
        )
        if not primitive_kinds:
            primitive_kinds = ("move_to", "mouse_down", "mouse_up")
        return (*primitive_kinds, *macro_kinds)
    return allowed


def _format_history_step(index: int, step: StepResult) -> str:
    if step.action is None:
        action_text = "null"
    else:
        action_text = json.dumps(step.action.to_dict(), ensure_ascii=False, separators=(",", ":"))
    details = [
        f"done={str(step.done).lower()}",
    ]
    success = step.info.get("success")
    if success is not None:
        details.append(f"success={str(bool(success)).lower()}")
    executed_kind = step.info.get("executed_kind")
    if executed_kind is not None:
        details.append(f"executed_kind={executed_kind}")
    if step.observation.cursor_xy is not None:
        details.append(
            f"cursor_after=({step.observation.cursor_xy[0]:.1f},{step.observation.cursor_xy[1]:.1f})",
        )
    return f"{index}. action={action_text}; " + "; ".join(details)


def build_action_prompt_payload(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None = None,
    condition: str | None = None,
) -> dict[str, object]:
    return {
        "instruction": obs.instruction,
        "allowed_kinds": list(allowed_kinds),
        "condition": condition,
        "budget": budget,
        "cursor_xy": obs.cursor_xy,
        "history_length": len(history),
        "screenshot_path": obs.screenshot_path,
    }


def build_multimodal_user_content(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None = None,
    condition: str | None = None,
) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    image_entries = _image_history_entries_for_sft(
        obs,
        history,
        images_to_keep=IMAGE_HISTORY_MAX,
    )
    for image_path, _cursor_xy in image_entries:
        encoded = encode_screenshot_base64(str(image_path))
        if encoded is not None:
            mime, data = encoded
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            })
    text = build_action_prompt_text(
        obs, history,
        allowed_kinds=allowed_kinds,
        budget=budget,
        condition=condition,
    )
    parts.append({"type": "text", "text": text})
    return parts


SftImageEntry = tuple[Path, tuple[float, float] | None]
SftImageEncoder = Callable[[Path, tuple[float, float] | None], tuple[str, str] | None]


def _history_action_dicts_for_sft(history: list[StepResult]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    for step in history:
        if step.action is None:
            continue
        action_dict = step.action.to_dict()
        if action_dict.get("kind") != "move_to":
            action_dict.pop("x", None)
            action_dict.pop("y", None)
        actions.append(action_dict)
    return actions


def _history_thoughts_for_sft(history: list[StepResult]) -> list[str | None]:
    thoughts: list[str | None] = []
    for step in history:
        if step.action is None:
            continue
        thoughts.append(_extract_history_thought_for_sft(step.info))
    return thoughts


def _extract_history_thought_for_sft(info: dict[str, object]) -> str | None:
    for key in ("model_think_text", "think_text", "assistant_think", "thought"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("model_raw_prediction", "raw_prediction", "assistant_response"):
        value = info.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        thought, _json_blob = split_think_json_response(value)
        if thought is not None:
            return thought
        recovered = _recover_malformed_history_thought_for_sft(value)
        if recovered is not None:
            return recovered
    return None


def _recover_malformed_history_thought_for_sft(value: str) -> str | None:
    close_index = value.find("</think>")
    if close_index < 0:
        return None
    thought = value[:close_index].strip()
    if thought.startswith("<think>"):
        thought = thought[len("<think>") :].strip()
    return thought or None


def _cursor_xy_for_sft_prompt(
    cursor_xy: tuple[float, float] | None,
    image_size_px: tuple[int, int],
) -> tuple[int, int] | None:
    if cursor_xy is None:
        return None
    width, height = image_size_px
    if width <= 0 or height <= 0:
        return None
    x, y = cursor_xy
    return (
        min(1000, max(0, int(round(float(x) / float(width) * 1000)))),
        min(1000, max(0, int(round(float(y) / float(height) * 1000)))),
    )


def _button_state_after_history_for_sft(history: list[StepResult]) -> str:
    button_state = "up"
    for step in history:
        action = step.action
        if action is None:
            continue
        if action.kind == "mouse_down":
            button_state = "down"
        elif action.kind in {"mouse_up", "left_click", "click", "drag"}:
            button_state = "up"
    return button_state


def _image_history_entries_for_sft(
    obs: Observation,
    history: list[StepResult],
    *,
    images_to_keep: int,
) -> list[SftImageEntry]:
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be >= 1")
    entries: list[SftImageEntry] = []
    for step in history:
        path = step.observation.screenshot_path
        if path:
            entries.append((Path(path), step.observation.cursor_xy))
    current = (Path(obs.screenshot_path), obs.cursor_xy)
    if entries and entries[-1] == current:
        entries[-1] = current
    else:
        entries.append(current)
    return list(fold_image_history_items(tuple(entries), images_to_keep=images_to_keep))


@dataclass
class OnlineSftObservationHistory:
    """Track observations seen before each online action prediction.

    The runner stores each StepResult observation after an action. On the next
    policy call that same observation is also the current obs, so rebuilding
    image history as ``history observations + current obs`` duplicates the last
    frame and can evict the wrong early frame when history folding is applied.
    This cache instead records the observations at the time they are actually
    fed to the model, then applies the shared latest-N image policy.
    """

    _entries: list[SftImageEntry] = field(default_factory=list)
    _history_length: int = -1

    def entries_for(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        images_to_keep: int,
    ) -> list[SftImageEntry]:
        if images_to_keep < 1:
            raise ValueError("images_to_keep must be >= 1")
        history_length = len(history)
        current = (Path(obs.screenshot_path), obs.cursor_xy)
        if history_length == 0:
            self._entries = [current]
        elif not self._entries or history_length < self._history_length:
            self._entries = _image_history_entries_for_sft(
                obs,
                history,
                images_to_keep=max(images_to_keep, history_length + 1),
            )
        elif history_length == self._history_length:
            self._entries[-1:] = [current]
        elif history_length == self._history_length + 1:
            self._entries.append(current)
        else:
            self._entries = _image_history_entries_for_sft(
                obs,
                history,
                images_to_keep=max(images_to_keep, history_length + 1),
            )
        self._history_length = history_length
        return list(fold_image_history_items(tuple(self._entries), images_to_keep=images_to_keep))


def _default_sft_image_encoder(
    image_path: Path,
    _cursor_xy: tuple[float, float] | None,
) -> tuple[str, str] | None:
    return encode_screenshot_base64(str(image_path))


def _sft_messages_to_openai_user_content(
    messages: list[dict[str, object]],
    *,
    retained_entries: list[SftImageEntry],
    image_encoder: SftImageEncoder,
) -> list[dict[str, object]]:
    if not messages:
        return []
    user_message = next((message for message in messages if message.get("role") == "user"), messages[0])
    content = user_message.get("content")
    if not isinstance(content, list):
        return []
    parts: list[dict[str, object]] = []
    image_index = 0
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append({"type": "text", "text": text})
            continue
        if item_type != "image":
            continue
        image_path = Path(str(item.get("image") or ""))
        cursor_xy: tuple[float, float] | None = None
        if image_index < len(retained_entries):
            entry_path, cursor_xy = retained_entries[image_index]
            image_path = entry_path
        image_index += 1
        encoded = image_encoder(image_path, cursor_xy)
        if encoded is None:
            continue
        mime, data = encoded
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })
    return parts


def build_sft_compatible_multimodal_user_content(
    obs: Observation,
    history: list[StepResult],
    *,
    allowed_kinds: Iterable[str],
    budget: int | None = None,
    images_to_keep: int | None = None,
    image_encoder: SftImageEncoder | None = None,
    observation_entries: list[SftImageEntry] | None = None,
) -> list[dict[str, object]]:
    """Build OpenAI chat user content from the same prompt builder used by SFT eval."""

    keep = IMAGE_HISTORY_MAX if images_to_keep is None else images_to_keep
    entries = (
        list(observation_entries)
        if observation_entries is not None
        else _image_history_entries_for_sft(
            obs,
            history,
            images_to_keep=max(keep, len(history) + 1),
        )
    )
    image_paths = tuple(path for path, _cursor_xy in entries)
    prompt_build = build_prompt_messages_from_history(
        instruction=obs.instruction,
        image_path=Path(obs.screenshot_path),
        image_paths=image_paths,
        action_history=_history_action_dicts_for_sft(history),
        thought_history=tuple(_history_thoughts_for_sft(history)),
        total_actions=budget,
        cursor_xy=_cursor_xy_for_sft_prompt(obs.cursor_xy, obs.size_px),
        button_state=_button_state_after_history_for_sft(history),
        task_type=_task_type_for_observation(obs),
        allowed_kinds=allowed_kinds,
        budget_remaining=max(0, budget - len(history)) if budget is not None else None,
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
        image_size_px=obs.size_px,
    )
    retained_count = len(prompt_build.image_paths)
    if retained_count <= len(entries):
        retained_entries = list(fold_image_history_items(tuple(entries), images_to_keep=retained_count))
    else:
        retained_entries = [(path, None) for path in prompt_build.image_paths]
    return _sft_messages_to_openai_user_content(
        prompt_build.messages,
        retained_entries=retained_entries,
        image_encoder=image_encoder or _default_sft_image_encoder,
    )


def _extract_chat_response_text(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    if message is None:
        return None
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return str(content)


def _extract_move_to_xy(payload: dict[str, object]) -> tuple[object, object] | None:
    sources: list[object] = [payload]
    for key in ("arguments", "args", "location"):
        if key in payload:
            sources.append(payload[key])
    for source in sources:
        if isinstance(source, dict):
            x_val = source.get("x")
            y_val = source.get("y")
            if x_val is not None and y_val is not None:
                return x_val, y_val
            point = source.get("point")
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                return point[0], point[1]
        elif isinstance(source, (list, tuple)) and len(source) >= 2:
            return source[0], source[1]
    return None


def _extract_points(payload: dict[str, object]) -> list[tuple[object, object]]:
    points = payload.get("points")
    if isinstance(points, (list, tuple)):
        if len(points) >= 2 and all(isinstance(v, (int, float)) for v in points[:2]):
            return [(points[0], points[1])]
        parsed: list[tuple[object, object]] = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                parsed.append((point[0], point[1]))
        if parsed:
            return parsed
    point = payload.get("point")
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return [(point[0], point[1])]
    xy = _extract_move_to_xy(payload)
    if xy is not None:
        return [xy]
    return []


def _normalize_action_kind_name(kind: object) -> object:
    if not isinstance(kind, str):
        return kind
    normalized = kind.strip()
    aliases = {
        "leftclick": "left_click",
        "left-click": "left_click",
        "leftClick": "left_click",
    }
    return aliases.get(normalized, normalized)


def _primitive_from_unified_tool_call(response: str) -> PrimitiveAction | None:
    parsed = parse_unified_three_action_response(response)
    if parsed.get("parse_error") is not None:
        return None
    action = parsed.get("action")
    if not isinstance(action, dict):
        return None
    kind = action.get("kind")
    if kind == "move_to":
        return PrimitiveAction(kind="move_to", x=action.get("x"), y=action.get("y"))
    if kind in {"mouse_down", "mouse_up"}:
        return PrimitiveAction(kind=kind)
    return None


def _responses_input_from_chat_content(user_content: list[dict[str, object]]) -> list[dict[str, object]]:
    content: list[dict[str, object]] = []
    for item in user_content:
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str):
                content.append({"type": "input_text", "text": text})
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str):
                    content.append({"type": "input_image", "image_url": url})
    return [{"role": "user", "content": content}]


def parse_action_blob(blob: str) -> Action:
    tool_action = _primitive_from_unified_tool_call(blob)
    if tool_action is not None:
        return tool_action
    _thought, json_blob = split_think_json_response(blob)
    try:
        payload = _parse_first_json_value(json_blob, dict, "action payload")
    except ValueError:
        if json_blob == blob:
            raise
        payload = _parse_first_json_value(blob, dict, "action payload")
    if "kind" not in payload:
        action_keys = {
            "move_to", "mouse_down", "mouse_up", "left_click", "done",
            "click", "drag", "submit",
            "left_double", "right_single", "scroll", "type",
            "hotkey", "wait", "finished",
        }
        payload_type = _normalize_action_kind_name(payload.get("type"))
        if isinstance(payload_type, str) and payload_type in action_keys:
            payload = {"kind": payload_type, **{k: v for k, v in payload.items() if k != "type"}}
        for action_key, action_value in payload.items():
            normalized_action_key = _normalize_action_kind_name(action_key)
            if normalized_action_key not in action_keys:
                continue
            merged: dict[str, object] = {"kind": normalized_action_key}
            if isinstance(action_value, dict):
                merged.update(action_value)
            elif normalized_action_key == "move_to" and isinstance(action_value, list) and len(action_value) >= 2:
                merged.update({"x": action_value[0], "y": action_value[1]})
            elif normalized_action_key in {"click", "drag"} and isinstance(action_value, list):
                merged["points"] = action_value
            elif normalized_action_key in {"left_double", "right_single", "scroll"} and isinstance(action_value, list):
                merged["points"] = [action_value]
            elif normalized_action_key in {"type", "finished"} and isinstance(action_value, str):
                merged["content"] = action_value
            elif normalized_action_key == "hotkey" and isinstance(action_value, str):
                merged["key"] = action_value
            payload = merged
            break
        for key in ("action", "next_action", "response", "result"):
            nested = payload.get(key)
            if isinstance(nested, dict) and "kind" in nested:
                payload = nested
                break
            if isinstance(nested, dict):
                inner_action = nested.get("action")
                if isinstance(inner_action, dict):
                    inner_kind = (
                        inner_action.get("kind")
                        or inner_action.get("type")
                        or inner_action.get("action")
                    )
                    inner_kind = _normalize_action_kind_name(inner_kind)
                    if isinstance(inner_kind, str) and inner_kind in action_keys:
                        payload = {
                            "kind": inner_kind,
                            **{
                                nested_key: nested_value
                                for nested_key, nested_value in nested.items()
                                if nested_key not in {"kind", "type", "action"}
                            },
                            **{
                                inner_key: inner_value
                                for inner_key, inner_value in inner_action.items()
                                if inner_key not in {"kind", "type", "action"}
                            },
                        }
                        break
                for action_key, action_value in nested.items():
                    normalized_action_key = _normalize_action_kind_name(action_key)
                    if normalized_action_key not in action_keys:
                        continue
                    merged: dict[str, object] = {"kind": normalized_action_key}
                    if isinstance(action_value, dict):
                        merged.update(action_value)
                    elif normalized_action_key == "move_to" and isinstance(action_value, list) and len(action_value) >= 2:
                        merged.update({"x": action_value[0], "y": action_value[1]})
                    elif normalized_action_key == "move_to" and "y" in nested:
                        merged.update({"x": action_value, "y": nested["y"]})
                    elif normalized_action_key in {"click", "drag"} and isinstance(action_value, list):
                        merged["points"] = action_value
                    elif normalized_action_key in {"left_double", "right_single", "scroll"} and isinstance(action_value, list):
                        merged["points"] = [action_value]
                    elif normalized_action_key in {"type", "finished"} and isinstance(action_value, str):
                        merged["content"] = action_value
                    elif normalized_action_key == "hotkey" and isinstance(action_value, str):
                        merged["key"] = action_value
                    payload = merged
                    break
                if "kind" in payload:
                    break
                nested_kind = _normalize_action_kind_name(nested.get("type") or nested.get("action"))
                if isinstance(nested_kind, str) and nested_kind in action_keys:
                    payload = {
                        "kind": nested_kind,
                        **{
                            nested_key: nested_value
                            for nested_key, nested_value in nested.items()
                            if nested_key not in {"kind", "type", "action"}
                        },
                    }
                    break
            if isinstance(nested, dict) and ("key" in nested or "keys" in nested):
                payload = {"kind": "hotkey", **nested}
                break
            # Handle {"action": "drag", "x1": ..., "y1": ..., "x2": ..., "y2": ...}
            normalized_nested = _normalize_action_kind_name(nested)
            if isinstance(normalized_nested, str) and normalized_nested in {
                "move_to", "mouse_down", "mouse_up", "left_click", "done",
                "click", "drag", "submit",
                "left_double", "right_single", "scroll", "type",
                "hotkey", "wait", "finished",
            }:
                merged: dict[str, object] = {"kind": normalized_nested}
                # Absorb x1/y1/x2/y2 → points for drag
                if normalized_nested == "drag" and "x1" in payload and "y1" in payload:
                    merged["points"] = [
                        [payload["x1"], payload["y1"]],
                        [payload.get("x2", payload["x1"]), payload.get("y2", payload["y1"])],
                    ]
                elif normalized_nested == "click" and "x" in payload and "y" in payload:
                    merged["points"] = [[payload["x"], payload["y"]]]
                elif normalized_nested == "move_to":
                    xy = _extract_move_to_xy(payload)
                    if xy is not None:
                        merged.update({"x": xy[0], "y": xy[1]})
                    else:
                        merged.update({k: v for k, v in payload.items() if k != key})
                else:
                    merged.update({k: v for k, v in payload.items() if k != key})
                payload = merged
                break
    kind = _normalize_action_kind_name(payload.get("kind"))
    if isinstance(kind, str):
        payload["kind"] = kind
    if kind is None:
        # {"answer": <value>} with no "kind" → treat as submit with answer
        if "answer" in payload:
            payload = {"kind": "submit", "answer": payload["answer"]}
            kind = "submit"
        elif "key" in payload or "keys" in payload:
            payload = {"kind": "hotkey", **payload}
            kind = "hotkey"
        else:
            raise ValueError(f"Model response JSON has no 'kind' field: {payload!r}")
    current_position_gesture = (
        kind in {"left_double", "right_single"}
        and not _extract_points(payload)
    )
    if kind in {"move_to", "mouse_down", "mouse_up", "left_click", "done"} or current_position_gesture:
        # Only move_to accepts coordinates; strip x/y for other primitive actions.
        if kind == "move_to":
            xy = _extract_move_to_xy(payload)
            x_val = xy[0] if xy is not None else payload.get("x")
            y_val = xy[1] if xy is not None else payload.get("y")
            return PrimitiveAction(kind=kind, x=x_val, y=y_val)
        # `done` may carry a structured answer for non-coordinate tasks (e.g. OCW).
        if kind == "done":
            return PrimitiveAction(kind=kind, answer=payload.get("answer"))
        return PrimitiveAction(kind=kind)
    if kind in {"click", "drag", "submit"}:
        points = payload.get("points", [])
        if not points:
            x_val = payload.get("x")
            y_val = payload.get("y")
            if isinstance(x_val, list):
                # Model put points array in the x field: {"x": [[x, y], ...], "y": null}
                points = [p for p in x_val if isinstance(p, (list, tuple)) and len(p) >= 2]
                if not points and len(x_val) >= 2 and isinstance(x_val[0], (int, float)):
                    points = [x_val]  # x_val is a flat [x, y] pair
            elif x_val is not None and y_val is not None:
                points = [[x_val, y_val]]
        elif isinstance(points, (list, tuple)):
            if len(points) >= 4 and kind == "drag" and all(isinstance(v, (int, float)) for v in points[:4]):
                points = [[points[0], points[1]], [points[2], points[3]]]
            elif len(points) >= 2 and all(isinstance(v, (int, float)) for v in points[:2]):
                points = [[points[0], points[1]]]
        parsed = [tuple(p) for p in points]
        # click accepts exactly 1 point; take the first if the model returned many
        if kind == "click" and len(parsed) > 1:
            parsed = parsed[:1]
        # Guard: if click/drag still has wrong number of points, fall back to submit
        # so the episode terminates gracefully rather than crashing the runner.
        expected = {"click": 1, "drag": 2, "submit": 0}[kind]
        if expected != 0 and len(parsed) != expected:
            return AtomicAction(kind="submit", points=[])
        # For submit, extract optional structured answer (used by OCW non-coord tasks).
        answer = payload.get("answer") if kind == "submit" else None
        return AtomicAction(kind=kind, points=parsed, answer=answer)
    if kind in {"left_double", "right_single", "scroll", "type", "hotkey", "wait", "finished"}:
        points = _extract_points(payload)
        if kind in {"type", "hotkey", "wait", "finished"}:
            points = []
        content = payload.get("content", payload.get("text"))
        if content is None and kind == "finished":
            content = payload.get("answer")
        if content is not None:
            content = str(content)
        key = payload.get("key", payload.get("keys"))
        if isinstance(key, (list, tuple)):
            key = " ".join(str(part) for part in key)
        if key is not None:
            key = str(key)
        direction = payload.get("direction")
        if direction is not None:
            direction = str(direction)
        return WebAction(
            kind=kind,
            points=[(float(x), float(y)) for x, y in points],
            direction=direction,
            content=content,
            key=key,
        )
    raise ValueError(f"Unsupported action kind: {kind}")


def parse_plan_blob(blob: str, *, append_done: bool = True) -> list[PrimitiveAction]:
    tool_action = _primitive_from_unified_tool_call(blob)
    if tool_action is not None:
        actions = [tool_action]
        if append_done and actions[-1].kind != "done":
            actions.append(PrimitiveAction(kind="done"))
        return actions
    _thought, json_blob = split_think_json_response(blob)
    try:
        plan_payload = _parse_first_json_value(json_blob, dict, "plan payload")
        items = plan_payload.get("plan")
        if not isinstance(items, list):
            raise ValueError("Model response plan payload did not include a plan array")
    except ValueError:
        items = _parse_first_json_value(json_blob, list, "plan payload")
    actions: list[PrimitiveAction] = []
    for element in items:
        kind = _normalize_action_kind_name(element["kind"])
        if kind not in {"move_to", "mouse_down", "mouse_up", "left_click", "done"}:
            raise ValueError(f"Unsupported primitive action kind in plan: {kind}")
        try:
            if kind == "done":
                actions.append(PrimitiveAction(kind=kind, answer=element.get("answer")))
            elif kind == "left_click":
                actions.append(PrimitiveAction(kind=kind))
            else:
                actions.append(PrimitiveAction(kind=kind, x=element.get("x"), y=element.get("y")))
        except ValueError:
            # skip malformed steps (e.g. move_to with null coordinates)
            continue
    # Ensure every plan ends with `done` when the caller allows done and needs
    # a terminal primitive emitted by the model.
    if append_done and (not actions or actions[-1].kind != "done"):
        actions.append(PrimitiveAction(kind="done"))
    return actions


def _parse_first_json_value(
    blob: str,
    expected_type: type[dict] | type[list],
    description: str,
) -> dict | list:
    decoder = json.JSONDecoder()
    opener = "{" if expected_type is dict else "["
    for idx, char in enumerate(blob):
        if char != opener:
            continue
        try:
            payload, _ = decoder.raw_decode(blob[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, expected_type):
            return payload
    type_label = "object" if expected_type is dict else "array"
    raise ValueError(f"Model response did not include a JSON {type_label} {description}")


@dataclass
class OpenAIChatBackend:
    model: str = "gpt-5"
    system_prompt: str = DEFAULT_ACTION_SYSTEM_PROMPT
    plan_system_prompt: str = PLAN_SYSTEM_PROMPT
    base_url: str | None = None
    client: object | None = None
    _sft_observation_history: OnlineSftObservationHistory = field(
        default_factory=OnlineSftObservationHistory,
        init=False,
        repr=False,
    )

    def build_sft_user_content(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        images_to_keep: int | None = None,
        image_encoder: SftImageEncoder | None = None,
    ) -> list[dict[str, object]]:
        keep = IMAGE_HISTORY_MAX if images_to_keep is None else images_to_keep
        entries = self._sft_observation_history.entries_for(
            obs,
            history,
            images_to_keep=max(keep, len(history) + 1),
        )
        return build_sft_compatible_multimodal_user_content(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            images_to_keep=keep,
            image_encoder=image_encoder,
            observation_entries=entries,
        )

    def _ensure_client(self) -> object:
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required to use OpenAIChatBackend",
            ) from exc
        kwargs: dict[str, object] = {}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)
        return self.client

    def _generate_text(
        self,
        system_prompt: str,
        user_content: list[dict[str, object]],
        *,
        _max_retries: int = 10,
        _base_delay: float = 5.0,
    ) -> str:
        import time

        client = self._ensure_client()
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        last_exc: Exception | None = None
        for attempt in range(_max_retries):
            try:
                # Use stream=True: some OpenAI-compatible gateways only return
                # content via streaming chunks and return empty choices for
                # non-streaming requests.
                stream = client.chat.completions.create(  # type: ignore[attr-defined]
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                chunks: list[str] = []
                for chunk in stream:  # type: ignore[attr-defined]
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        content = getattr(delta, "content", None) if delta is not None else None
                        if isinstance(content, str) and content:
                            chunks.append(content)
                text = "".join(chunks).strip()
                if text:
                    return text
                raise RuntimeError("Streaming API returned no text content")
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "concurrency limit" in msg or "rate limit" in msg or "429" in msg:
                    delay = _base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
        raise RuntimeError(f"API call failed after {_max_retries} retries") from last_exc

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action:
        del condition
        user_content = self.build_sft_user_content(
            obs, history,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        raw = self._generate_text(self.system_prompt, user_content)
        return parse_action_blob(raw)

    def predict_plan(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> list[PrimitiveAction]:
        user_content = build_multimodal_user_content(
            obs, history,
            allowed_kinds=allowed_kinds,
            budget=budget,
            condition=condition,
        )
        raw = self._generate_text(self.plan_system_prompt, user_content)
        return parse_plan_blob(raw, append_done="done" in set(allowed_kinds))
