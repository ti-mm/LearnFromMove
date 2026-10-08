from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..actions import Action, AtomicAction, PrimitiveAction
from ..core import Observation, StepResult
from ..protocol_tracks import canonicalize_task_requirement
from .openai_native_computer import OpenAINativeAPIError

OPENAI_RESPONSES_SIX_ACTION_PROTOCOL = "openai_responses_custom_six_action_v1"
OPENAI_RESPONSES_SIX_ACTION_HISTORY = (
    "previous_response_id_full_response_chain_current_screenshot"
)
OPENAI_RESPONSES_SIX_ACTION_PROMPT_VERSION = (
    "training_aligned_openai_computer_use_v1"
)
OPENAI_RESPONSES_SIX_ACTION_KINDS = (
    "move_to",
    "mouse_down",
    "mouse_up",
    "left_click",
    "drag",
    "click",
)

_TOOL_NAME = "computer_use"
SIX_ACTION_INSTRUCTIONS = "\n\n".join(
    (
        "You are a helpful assistant. The user will give you an instruction, and you "
        "MUST interact with the corresponding UI element via the computer_use tool. If "
        "you are not sure about where to interact, guess a most likely one.",
        "At each step, inspect the current screenshot and call computer_use exactly once. "
        "The harness will execute that action and provide an updated screenshot for the "
        "next step. Do not return multiple actions or actions outside the provided tool "
        "schema.",
        "The screen uses a 1000x1000 screenshot-relative coordinate system: x=0 is the "
        "left edge, x=1000 is the right edge; y=0 is the top edge, y=1000 is the bottom "
        "edge. Coordinates must be integers from 0 to 1000. Do not use original-image "
        "pixel coordinates or 0-1 decimal coordinates.",
        "Use move_to with x and y to move without pressing. mouse_down presses and holds "
        "at the current cursor position. mouse_up releases at the current cursor position. "
        "left_click clicks at the current cursor or reticle position and takes no "
        "coordinates. click uses exactly one screenshot-relative point. drag performs one "
        "continuous press-drag-release gesture and uses exactly two screenshot-relative "
        "points ordered as start and end.",
        "Return the next action only through the computer_use function call. Do not wrap "
        "it in prose, XML, or another JSON object.",
    )
)

JsonRequester = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]


class OpenAIResponsesSixActionError(RuntimeError):
    """Base error for the custom six-action Responses adapter."""


class OpenAIResponsesSixActionAPIError(OpenAINativeAPIError):
    """The Responses request failed after infrastructure retries."""


class OpenAIResponsesSixActionProtocolError(OpenAIResponsesSixActionError, ValueError):
    """The model response did not satisfy the six-action contract."""


@dataclass(frozen=True)
class OpenAIResponsesSixActionConfig:
    model: str = "gpt-5.6-sol"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")


def _responses_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid OpenAI base URL: {base_url!r}")
    path = f"{parsed.path.rstrip('/')}/responses"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _safe_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    netloc = parsed.netloc.rsplit("@", 1)[-1]
    return urllib.parse.urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _default_json_requester(
    url: str,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_s: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(dict(body), ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {_safe_base_url(url)}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OpenAI Responses API returned a non-object JSON payload")
    return payload


def _retryable_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "temporarily",
            "connection",
            "502",
            "503",
            "504",
        )
    )


def _sanitize_log_value(value: Any, *, api_key: str, field_name: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_log_value(item, api_key=api_key, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_log_value(item, api_key=api_key) for item in value]
    if not isinstance(value, str):
        return value
    redacted = value.replace(api_key, "<redacted-api-key>") if api_key else value
    if field_name == "image_url" and redacted.startswith("data:"):
        return f"<redacted-data-url length={len(redacted)}>"
    return redacted


def six_action_parameters() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
            },
            "x": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Required only for move_to.",
            },
            "y": {
                "type": "integer",
                "minimum": 0,
                "maximum": 1000,
                "description": "Required only for move_to.",
            },
            "points": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 1000,
                    },
                    "minItems": 2,
                    "maxItems": 2,
                },
                "minItems": 1,
                "maxItems": 2,
                "description": (
                    "Required only for click or drag. click has one point; drag has "
                    "exactly two points ordered as start and end."
                ),
            },
        },
        "required": ["kind"],
        "additionalProperties": False,
    }


def six_action_chat_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Execute exactly one action from the six-action GUI training contract. "
                "move_to uses x/y; mouse_down, mouse_up, and left_click use the current "
                "cursor; click uses one point; drag uses start and end points."
            ),
            "parameters": six_action_parameters(),
            "strict": False,
        },
    }


def _computer_use_tool() -> dict[str, Any]:
    chat_function = six_action_chat_tool()["function"]
    return {
        "type": "function",
        "name": chat_function["name"],
        "description": chat_function["description"],
        "parameters": chat_function["parameters"],
        "strict": False,
    }


def _coordinate(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAIResponsesSixActionProtocolError(f"{label} must be an integer")
    if not 0 <= value <= 1000:
        raise OpenAIResponsesSixActionProtocolError(f"{label} must be in [0, 1000]")
    return value


def _point(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise OpenAIResponsesSixActionProtocolError(f"{label} must be [x, y]")
    return (
        float(_coordinate(value[0], label=f"{label}.x")),
        float(_coordinate(value[1], label=f"{label}.y")),
    )


def six_action_arguments_to_action(
    arguments: Mapping[str, Any],
    *,
    allowed_kinds: Iterable[str],
) -> Action:
    kind = arguments.get("kind")
    if kind not in OPENAI_RESPONSES_SIX_ACTION_KINDS:
        raise OpenAIResponsesSixActionProtocolError(f"unsupported action kind: {kind!r}")
    if kind not in set(str(value) for value in allowed_kinds):
        raise OpenAIResponsesSixActionProtocolError(f"action kind {kind!r} is disabled")

    actual = set(arguments)
    expected = {
        "move_to": {"kind", "x", "y"},
        "mouse_down": {"kind"},
        "mouse_up": {"kind"},
        "left_click": {"kind"},
        "click": {"kind", "points"},
        "drag": {"kind", "points"},
    }[str(kind)]
    if actual != expected:
        raise OpenAIResponsesSixActionProtocolError(
            f"{kind} requires fields {sorted(expected)!r}, got {sorted(actual)!r}"
        )

    if kind == "move_to":
        return PrimitiveAction(
            kind="move_to",
            x=float(_coordinate(arguments.get("x"), label="move_to.x")),
            y=float(_coordinate(arguments.get("y"), label="move_to.y")),
        )
    if kind in {"mouse_down", "mouse_up", "left_click"}:
        return PrimitiveAction(kind=kind)

    raw_points = arguments.get("points")
    expected_count = 1 if kind == "click" else 2
    if not isinstance(raw_points, list) or len(raw_points) != expected_count:
        raise OpenAIResponsesSixActionProtocolError(
            f"{kind} requires exactly {expected_count} point(s)"
        )
    points = [_point(value, label=f"{kind}.points[{index}]") for index, value in enumerate(raw_points)]
    return AtomicAction(kind=kind, points=points)


@dataclass(frozen=True)
class _ParsedFunctionCall:
    response_id: str
    call_id: str
    arguments: dict[str, Any]
    action: Action


class OpenAIResponsesSixActionBackend:
    """Stateful multimodal Responses adapter for the six-action training contract."""

    def __init__(
        self,
        *,
        config: OpenAIResponsesSixActionConfig,
        call_log_dir: Path | None = None,
        request_timeout_s: float = 180.0,
        request_retries: int = 6,
        requester: JsonRequester | None = None,
    ) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        self.config = config
        self.model = config.model
        self.call_log_dir = call_log_dir
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.requester = requester or _default_json_requester
        self.protocol_track = OPENAI_RESPONSES_SIX_ACTION_PROTOCOL
        self.history_strategy = OPENAI_RESPONSES_SIX_ACTION_HISTORY
        self.prompt_version = OPENAI_RESPONSES_SIX_ACTION_PROMPT_VERSION
        self.sampling_controls_unsupported = True
        self._url = _responses_url(config.base_url)
        self._call_index = 0
        self._episode_index = 0
        self._episode_started = False
        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None
        self._last_action_json: dict[str, Any] | None = None
        self._last_history_length: int | None = None
        self.last_prediction_index: int | None = None
        self.last_raw_prediction: str | None = None
        self.last_native_action_json: dict[str, Any] | None = None
        self.last_native_action_batch_json: list[dict[str, Any]] = []
        self.last_call_log_path: str | None = None
        self.last_error: str | None = None

    @property
    def previous_response_id(self) -> str | None:
        return self._previous_response_id

    def reset_episode(self) -> None:
        self._episode_started = False
        self._previous_response_id = None
        self._pending_call_id = None
        self._last_action_json = None
        self._last_history_length = None
        self.last_prediction_index = None
        self.last_raw_prediction = None
        self.last_native_action_json = None
        self.last_native_action_batch_json = []
        self.last_call_log_path = None
        self.last_error = None

    def reset(self) -> None:
        self.reset_episode()

    def close(self) -> None:
        self.reset_episode()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _screenshot_data_url(observation: Observation) -> str:
        path = Path(observation.screenshot_path)
        if not path.is_file():
            raise FileNotFoundError(f"screenshot is missing: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _task_type(observation: Observation) -> str | None:
        metadata = observation.metadata or {}
        for key in ("task_type", "benchmark", "task_id", "episode_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _screen_message(
        self,
        observation: Observation,
        *,
        step_index: int,
    ) -> dict[str, Any]:
        task = canonicalize_task_requirement(
            observation.instruction,
            task_type=self._task_type(observation),
            target_label=(observation.metadata or {}).get("target_label"),
        ).strip()
        if not task:
            raise OpenAIResponsesSixActionProtocolError("observation instruction must not be empty")
        if step_index == 1:
            text = f"Task: {task}\nUse the computer_use tool for UI interaction."
        else:
            text = (
                "The attached image is the updated state after executing the previous "
                "action and is the current observation. Use previous computer_use "
                "function calls as context only. Return exactly one computer_use "
                f"function call for step {step_index}."
            )
        return {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {
                    "type": "input_image",
                    "image_url": self._screenshot_data_url(observation),
                    "detail": "original",
                },
            ],
        }

    def _request_body(
        self,
        observation: Observation,
        *,
        step_index: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": SIX_ACTION_INSTRUCTIONS,
            "tools": [_computer_use_tool()],
            "tool_choice": {"type": "function", "name": _TOOL_NAME},
            "parallel_tool_calls": False,
        }
        if self._previous_response_id is None:
            body["input"] = [self._screen_message(observation, step_index=1)]
            return body
        if self._pending_call_id is None or self._last_action_json is None:
            raise OpenAIResponsesSixActionProtocolError(
                "cannot continue a response chain without the previous function call"
            )
        body["previous_response_id"] = self._previous_response_id
        body["input"] = [
            {
                "type": "function_call_output",
                "call_id": self._pending_call_id,
                "output": json.dumps(
                    {"executed_action": self._last_action_json},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            self._screen_message(observation, step_index=step_index),
        ]
        return body

    def _parse_response(
        self,
        response: Mapping[str, Any],
        *,
        allowed_kinds: tuple[str, ...],
    ) -> _ParsedFunctionCall:
        if response.get("status") != "completed":
            raise OpenAIResponsesSixActionProtocolError(
                f"OpenAI response status is {response.get('status')!r}"
            )
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise OpenAIResponsesSixActionProtocolError("OpenAI response is missing id")
        output = response.get("output")
        if not isinstance(output, list):
            raise OpenAIResponsesSixActionProtocolError("OpenAI response output must be an array")
        calls = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        if len(calls) != 1:
            raise OpenAIResponsesSixActionProtocolError(
                f"expected exactly one function_call, got {len(calls)}"
            )
        call = calls[0]
        if call.get("name") != _TOOL_NAME:
            raise OpenAIResponsesSixActionProtocolError(
                f"unexpected function name: {call.get('name')!r}"
            )
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise OpenAIResponsesSixActionProtocolError("function_call is missing call_id")
        raw_arguments = call.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise OpenAIResponsesSixActionProtocolError(
                    f"function arguments are invalid JSON: {exc}"
                ) from exc
        else:
            arguments = raw_arguments
        if not isinstance(arguments, dict):
            raise OpenAIResponsesSixActionProtocolError("function arguments must be an object")
        action = six_action_arguments_to_action(arguments, allowed_kinds=allowed_kinds)
        return _ParsedFunctionCall(
            response_id=response_id,
            call_id=call_id,
            arguments=dict(arguments),
            action=action,
        )

    @staticmethod
    def _safe_error_text(exc: BaseException) -> str:
        return str(exc)

    def _write_call_log(self, call_index: int, payload: Mapping[str, Any]) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.call_log_dir / f"call_{call_index:03d}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        sanitized = _sanitize_log_value(dict(payload), api_key=self.config.api_key)
        temporary.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.last_call_log_path = str(path)

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action:
        del budget, condition
        allowed = tuple(dict.fromkeys(str(kind) for kind in allowed_kinds))
        if allowed != OPENAI_RESPONSES_SIX_ACTION_KINDS:
            raise ValueError(
                "gpt-5.6-sol Responses evaluation requires the exact six-action "
                f"training contract {OPENAI_RESPONSES_SIX_ACTION_KINDS!r}, got {allowed!r}"
            )
        history_length = len(history)
        if (
            self._episode_started
            and self._last_history_length is not None
            and history_length <= self._last_history_length
        ):
            self.reset_episode()
        if not self._episode_started:
            self._episode_index += 1
            self._episode_started = True

        body = self._request_body(obs, step_index=history_length + 1)
        self._call_index += 1
        call_index = self._call_index
        started_at = time.time()
        attempts: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        last_error: BaseException | None = None
        for attempt in range(1, self.request_retries + 1):
            attempt_started = time.time()
            try:
                response = self.requester(
                    self._url,
                    self._headers(),
                    body,
                    self.request_timeout_s,
                )
                attempts.append(
                    {"attempt": attempt, "latency_s": time.time() - attempt_started, "error": None}
                )
                break
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "latency_s": time.time() - attempt_started,
                        "error": {"type": type(exc).__name__, "message": self._safe_error_text(exc)},
                    }
                )
                if attempt >= self.request_retries or not _retryable_error(exc):
                    break
                time.sleep(min(2.0 * (2 ** (attempt - 1)), 30.0))

        common_log = {
            "call_index": call_index,
            "episode_index": self._episode_index,
            "provider": "openai",
            "model": self.model,
            "protocol_track": self.protocol_track,
            "history_strategy": self.history_strategy,
            "prompt_version": self.prompt_version,
            "action_kinds": list(OPENAI_RESPONSES_SIX_ACTION_KINDS),
            "base_url": _safe_base_url(self.config.base_url),
            "request": dict(body),
            "request_attempts": attempts,
            "api_key_persisted": False,
        }
        if response is None:
            cause = last_error or RuntimeError("OpenAI returned no response")
            self.last_error = f"{type(cause).__name__}: {self._safe_error_text(cause)}"
            self._write_call_log(
                call_index,
                {
                    **common_log,
                    "response": None,
                    "error": self.last_error,
                    "latency_s": time.time() - started_at,
                },
            )
            raise OpenAIResponsesSixActionAPIError(
                f"OpenAI returned no response after {len(attempts)} attempt(s)"
            ) from last_error

        try:
            parsed = self._parse_response(response, allowed_kinds=allowed)
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, OpenAIResponsesSixActionError)
                else OpenAIResponsesSixActionProtocolError(str(exc))
            )
            self.last_error = f"{type(error).__name__}: {self._safe_error_text(error)}"
            self._write_call_log(
                call_index,
                {
                    **common_log,
                    "response": response,
                    "error": self.last_error,
                    "latency_s": time.time() - started_at,
                },
            )
            raise error from exc

        self._previous_response_id = parsed.response_id
        self._pending_call_id = parsed.call_id
        self._last_action_json = parsed.arguments
        self._last_history_length = history_length
        self.last_prediction_index = history_length + 1
        self.last_raw_prediction = json.dumps(
            parsed.arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.last_native_action_json = dict(parsed.arguments)
        self.last_native_action_batch_json = [dict(parsed.arguments)]
        self.last_error = None
        self._write_call_log(
            call_index,
            {
                **common_log,
                "response": response,
                "response_id": parsed.response_id,
                "function_call_id": parsed.call_id,
                "action": parsed.arguments,
                "canonical_action": parsed.action.to_dict(),
                "error": None,
                "latency_s": time.time() - started_at,
            },
        )
        return parsed.action


__all__ = [
    "OPENAI_RESPONSES_SIX_ACTION_HISTORY",
    "OPENAI_RESPONSES_SIX_ACTION_KINDS",
    "OPENAI_RESPONSES_SIX_ACTION_PROMPT_VERSION",
    "OPENAI_RESPONSES_SIX_ACTION_PROTOCOL",
    "SIX_ACTION_INSTRUCTIONS",
    "OpenAIResponsesSixActionAPIError",
    "OpenAIResponsesSixActionBackend",
    "OpenAIResponsesSixActionConfig",
    "OpenAIResponsesSixActionError",
    "OpenAIResponsesSixActionProtocolError",
    "six_action_arguments_to_action",
    "six_action_chat_tool",
    "six_action_parameters",
]
