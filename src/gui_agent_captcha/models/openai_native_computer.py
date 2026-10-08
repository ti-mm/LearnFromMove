"""OpenAI Responses API backend for the native GA ``computer`` tool.

The wire contract follows the official Computer use loop:
https://developers.openai.com/api/docs/guides/tools-computer-use
"""

from __future__ import annotations

import base64
import copy
import json
import re
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import httpx

from ..core import Observation, StepResult

JsonRequester = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float],
    Mapping[str, Any],
]
ObservationSupplier = Callable[[], Observation]
Waiter = Callable[[float], None]

OPENAI_NATIVE_COMPUTER_PROTOCOL = "openai_responses_native_computer_ga"
OPENAI_NATIVE_HISTORY_STRATEGY = "previous_response_id_full_response_chain"
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_REQUEST_RETRIES = 6
DEFAULT_MAX_SCREENSHOT_ROUND_TRIPS = 8
_TASK_SUFFIX = "Use the computer tool for UI interaction."
_DATA_URL_RE = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+")
OPENAI_COMPUTER_ACTION_KINDS = (
    "click",
    "double_click",
    "drag",
    "keypress",
    "move",
    "screenshot",
    "scroll",
    "type",
    "wait",
)
_OPENAI_COMPUTER_ACTION_KIND_SET = frozenset(OPENAI_COMPUTER_ACTION_KINDS)
_MOUSE_BUTTONS = frozenset({"left", "right", "wheel", "back", "forward"})


class OpenAINativeComputerError(RuntimeError):
    """Base error for the native OpenAI computer-use backend."""


class OpenAIResponseBudgetExhausted(OpenAINativeComputerError):
    """The episode has consumed its model-response budget."""


class OpenAINativeAPIError(OpenAINativeComputerError):
    """The Responses API request failed after infrastructure retries."""


class OpenAINativeProtocolError(OpenAINativeComputerError):
    """The provider response violated the GA computer-tool contract."""


class OpenAINativeSafetyCheckError(OpenAINativeProtocolError):
    """A computer call requires confirmation that the benchmark cannot provide."""


@dataclass(frozen=True)
class OpenAINativeComputerConfig:
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


@dataclass(frozen=True)
class OpenAIComputerAction:
    """One unmodified action from an OpenAI ``computer_call.actions[]`` batch."""

    action: dict[str, Any]
    tool_type: str = "computer"

    @property
    def kind(self) -> str:
        return str(self.action["type"])

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.action)


@dataclass(frozen=True)
class _ParsedComputerCall:
    response_id: str
    call_id: str | None
    native_actions: list[dict[str, Any]]
    queued_actions: list[OpenAIComputerAction]
    end_turn: bool = False


def _responses_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid OpenAI base URL: {base_url!r}")
    base_path = parsed.path.rstrip("/") or "/v1"
    path = f"{base_path}/responses"
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
    try:
        with httpx.Client(timeout=timeout_s, trust_env=False) as client:
            response = client.post(url, headers=dict(headers), json=dict(body))
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:4096]
        raise RuntimeError(
            f"HTTP {exc.response.status_code} from {_safe_base_url(url)}: {detail}"
        ) from exc
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


def _redact_blob(value: str, *, label: str) -> dict[str, object]:
    return {
        f"{label}_redacted": True,
        f"{label}_length": len(value),
    }


def _sanitize_log_value(
    value: Any,
    *,
    api_key: str,
    field_name: str | None = None,
) -> Any:
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
    if field_name in {"data", "image_url"} and (
        redacted.startswith("data:") or len(redacted) > 256
    ):
        return _redact_blob(redacted, label="image_payload")
    if redacted.startswith("data:") and ";base64," in redacted:
        return _redact_blob(redacted, label="data_url")
    return _DATA_URL_RE.sub("<redacted-data-url>", redacted)


def _require_fields(
    action: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    actual = set(action)
    missing = required - actual
    unknown = actual - required - optional
    if missing or unknown:
        raise OpenAINativeProtocolError(
            f"native {action.get('type')!r} action has missing fields {sorted(missing)!r} "
            f"and unknown fields {sorted(unknown)!r}"
        )


def _validate_keys(value: Any, *, label: str, optional: bool = True) -> None:
    if value is None and optional:
        return
    if not isinstance(value, list) or not all(isinstance(key, str) for key in value):
        raise OpenAINativeProtocolError(f"{label} must be an array of strings")


def _validate_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpenAINativeProtocolError(f"{label} must be an integer")
    return value


def _validate_pixel_xy(
    x_value: Any,
    y_value: Any,
    *,
    size_px: tuple[int, int],
    label: str,
) -> None:
    width, height = size_px
    if width <= 0 or height <= 0:
        raise OpenAINativeProtocolError(f"invalid screenshot size {size_px!r}")
    x = _validate_integer(x_value, label=f"{label}.x")
    y = _validate_integer(y_value, label=f"{label}.y")
    if not 0 <= x < width or not 0 <= y < height:
        raise OpenAINativeProtocolError(
            f"{label} pixel coordinate {(x, y)!r} is outside screenshot "
            f"bounds {size_px!r}"
        )


def validate_native_action(
    native_action: Mapping[str, Any],
    *,
    size_px: tuple[int, int],
) -> OpenAIComputerAction:
    """Validate and preserve one official GA computer action without canonicalizing it."""

    kind = native_action.get("type")
    if not isinstance(kind, str):
        raise OpenAINativeProtocolError("native computer action is missing string type")
    if kind not in _OPENAI_COMPUTER_ACTION_KIND_SET:
        raise OpenAINativeProtocolError(f"unknown native OpenAI computer action {kind!r}")

    if kind in {"screenshot", "wait"}:
        _require_fields(native_action, required={"type"})
    elif kind == "move":
        _require_fields(
            native_action,
            required={"type", "x", "y"},
            optional={"keys"},
        )
        _validate_keys(native_action.get("keys"), label="move.keys")
        _validate_pixel_xy(
            native_action.get("x"),
            native_action.get("y"),
            size_px=size_px,
            label="move",
        )
    elif kind == "click":
        _require_fields(
            native_action,
            required={"type", "button", "x", "y"},
            optional={"keys"},
        )
        button = native_action.get("button")
        if button not in _MOUSE_BUTTONS:
            raise OpenAINativeProtocolError(
                f"click.button must be one of {sorted(_MOUSE_BUTTONS)!r}"
            )
        _validate_keys(native_action.get("keys"), label="click.keys")
        _validate_pixel_xy(
            native_action.get("x"),
            native_action.get("y"),
            size_px=size_px,
            label="click",
        )
    elif kind == "double_click":
        _require_fields(native_action, required={"type", "keys", "x", "y"})
        _validate_keys(native_action.get("keys"), label="double_click.keys")
        _validate_pixel_xy(
            native_action.get("x"),
            native_action.get("y"),
            size_px=size_px,
            label="double_click",
        )
    elif kind == "drag":
        _require_fields(native_action, required={"type", "path"}, optional={"keys"})
        _validate_keys(native_action.get("keys"), label="drag.keys")
        path = native_action.get("path")
        if not isinstance(path, list) or len(path) < 2:
            raise OpenAINativeProtocolError(
                "native drag path must contain at least two points"
            )
        for index, raw_point in enumerate(path):
            if not isinstance(raw_point, Mapping):
                raise OpenAINativeProtocolError(
                    f"drag.path[{index}] must be an x/y object"
                )
            _require_fields(raw_point, required={"x", "y"})
            _validate_pixel_xy(
                raw_point.get("x"),
                raw_point.get("y"),
                size_px=size_px,
                label=f"drag.path[{index}]",
            )
    elif kind == "scroll":
        _require_fields(
            native_action,
            required={"type", "scroll_x", "scroll_y", "x", "y"},
            optional={"keys"},
        )
        _validate_keys(native_action.get("keys"), label="scroll.keys")
        _validate_integer(native_action.get("scroll_x"), label="scroll.scroll_x")
        _validate_integer(native_action.get("scroll_y"), label="scroll.scroll_y")
        _validate_pixel_xy(
            native_action.get("x"),
            native_action.get("y"),
            size_px=size_px,
            label="scroll",
        )
    elif kind == "keypress":
        _require_fields(native_action, required={"type", "keys"})
        _validate_keys(native_action.get("keys"), label="keypress.keys", optional=False)
    elif kind == "type":
        _require_fields(native_action, required={"type", "text"})
        if not isinstance(native_action.get("text"), str):
            raise OpenAINativeProtocolError("type.text must be a string")

    return OpenAIComputerAction(action=copy.deepcopy(dict(native_action)))


class OpenAINativeComputerBackend:
    """Stateful, one-episode adapter for OpenAI's GA ``computer`` tool."""

    def __init__(
        self,
        *,
        config: OpenAINativeComputerConfig,
        system_prompt: str | None = None,
        prompt_extension: str | None = None,
        call_log_dir: Path | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        max_screenshot_round_trips: int = DEFAULT_MAX_SCREENSHOT_ROUND_TRIPS,
        max_responses: int | None = None,
        observation_supplier: ObservationSupplier | None = None,
        requester: JsonRequester | None = None,
        waiter: Waiter | None = None,
    ) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        if max_screenshot_round_trips < 1:
            raise ValueError("max_screenshot_round_trips must be >= 1")
        if max_responses is not None and max_responses < 1:
            raise ValueError("max_responses must be >= 1")
        self.max_responses = max_responses
        self.response_count = 0
        self.config = config
        self.model = config.model
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.prompt_extension = prompt_extension.strip() if prompt_extension else None
        self.call_log_dir = call_log_dir
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.max_screenshot_round_trips = max_screenshot_round_trips
        self.observation_supplier = observation_supplier
        self.requester = requester or _default_json_requester
        self.waiter = waiter or time.sleep
        self.protocol_track = OPENAI_NATIVE_COMPUTER_PROTOCOL
        self.history_strategy = OPENAI_NATIVE_HISTORY_STRATEGY
        self.sampling_controls_unsupported = True
        self._url = _responses_url(config.base_url)
        self._call_index = 0
        self._episode_index = 0
        self._episode_started = False
        self._initial_request_sent = False
        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None
        self._end_turn_pending = False
        self._pending_actions: deque[OpenAIComputerAction] = deque()
        self._last_history_length: int | None = None
        self._prediction_index = 0
        self.last_prediction_index: int | None = None
        self.last_raw_prediction: str | None = None
        self.last_native_action_json: dict[str, Any] | None = None
        self.last_native_action_batch_json: list[dict[str, Any]] = []
        self.last_call_log_path: str | None = None
        self.last_error: str | None = None

    @property
    def previous_response_id(self) -> str | None:
        return self._previous_response_id

    @property
    def pending_call_id(self) -> str | None:
        return self._pending_call_id

    @property
    def pending_native_action_count(self) -> int:
        return len(self._pending_actions)

    def reset_episode(self) -> None:
        """Drop every server-chain and queued-action reference for the current episode."""

        self.response_count = 0
        self._episode_started = False
        self._initial_request_sent = False
        self._previous_response_id = None
        self._pending_call_id = None
        self._end_turn_pending = False
        self._pending_actions.clear()
        self._last_history_length = None
        self._prediction_index = 0
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

    def _initial_request(self, observation: Observation) -> dict[str, Any]:
        instruction = observation.instruction.strip()
        if not instruction:
            raise OpenAINativeProtocolError("observation instruction must not be empty")
        body: dict[str, Any] = {
            "model": self.model,
            "tools": [{"type": "computer"}],
            "input": f"{instruction}\n\n{_TASK_SUFFIX}",
        }
        instructions = "\n\n".join(p for p in (self.system_prompt, self.prompt_extension) if p)
        if instructions:
            body["instructions"] = instructions
        return body

    @staticmethod
    def _screenshot_data_url(observation: Observation) -> str:
        path = Path(observation.screenshot_path)
        if not path.is_file():
            raise FileNotFoundError(f"screenshot is missing: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _screenshot_output_request(self, observation: Observation) -> dict[str, Any]:
        if self._previous_response_id is None or self._pending_call_id is None:
            raise OpenAINativeProtocolError(
                "cannot return a screenshot without a preceding computer_call"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "tools": [{"type": "computer"}],
            "previous_response_id": self._previous_response_id,
            "input": [
                {
                    "type": "computer_call_output",
                    "call_id": self._pending_call_id,
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": self._screenshot_data_url(observation),
                        "detail": "original",
                    },
                }
            ],
        }
        instructions = "\n\n".join(p for p in (self.system_prompt, self.prompt_extension) if p)
        if instructions:
            body["instructions"] = instructions
        return body

    def _fresh_observation_for_screenshot(self, prior: Observation) -> Observation:
        if self.observation_supplier is None:
            raise OpenAINativeProtocolError(
                "native screenshot requires an observation_supplier that captures a fresh screenshot"
            )
        fresh = self.observation_supplier()
        if not isinstance(fresh, Observation):
            raise OpenAINativeProtocolError(
                "observation_supplier must return an Observation after native screenshot"
            )
        if fresh.instruction != prior.instruction:
            raise OpenAINativeProtocolError(
                "observation_supplier returned an observation from a different task"
            )
        return fresh

    def _safe_error_text(self, exc: BaseException) -> str:
        text = str(exc).replace(self.config.api_key, "<redacted-api-key>")
        return _DATA_URL_RE.sub("<redacted-data-url>", text)

    def _write_call_log(self, call_index: int, payload: Mapping[str, Any]) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.call_log_dir / f"call_{call_index:03d}.json"
        temporary = path.with_suffix(path.suffix + ".tmp")
        sanitized = _sanitize_log_value(
            dict(payload),
            api_key=self.config.api_key,
        )
        temporary.write_text(
            json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        self.last_call_log_path = str(path)

    def _parse_computer_call(
        self,
        response: Mapping[str, Any],
        *,
        observation: Observation,
    ) -> _ParsedComputerCall:
        status = response.get("status")
        if status != "completed":
            raise OpenAINativeProtocolError(
                f"OpenAI response status is {status!r}; "
                f"incomplete_details={response.get('incomplete_details')!r}"
            )
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise OpenAINativeProtocolError("OpenAI response is missing id")
        output = response.get("output")
        if not isinstance(output, list):
            raise OpenAINativeProtocolError("OpenAI response output must be an array")
        calls = [
            item
            for item in output
            if isinstance(item, dict) and item.get("type") == "computer_call"
        ]
        if len(calls) > 1:
            raise OpenAINativeProtocolError(
                f"expected at most one computer_call, got {len(calls)}"
            )
        if not calls:
            return _ParsedComputerCall(
                response_id=response_id,
                call_id=None,
                native_actions=[],
                queued_actions=[],
                end_turn=True,
            )
        call = calls[0]
        if call.get("status") != "completed":
            raise OpenAINativeProtocolError(
                f"computer_call status is {call.get('status')!r}, expected 'completed'"
            )
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise OpenAINativeProtocolError("computer_call is missing call_id")
        safety_checks = call.get("pending_safety_checks", [])
        if not isinstance(safety_checks, list):
            raise OpenAINativeProtocolError("pending_safety_checks must be an array")
        if safety_checks:
            identifiers = [
                str(item.get("id")) if isinstance(item, dict) else repr(item)
                for item in safety_checks
            ]
            raise OpenAINativeSafetyCheckError(
                f"computer_call requires pending safety checks {identifiers!r}; "
                "the benchmark cannot acknowledge them"
            )
        raw_actions = call.get("actions")
        if not isinstance(raw_actions, list) or not raw_actions:
            raise OpenAINativeProtocolError(
                "GA computer_call must contain a non-empty actions[] array"
            )

        native_actions: list[dict[str, Any]] = []
        queued_actions: list[OpenAIComputerAction] = []
        for index, raw_action in enumerate(raw_actions):
            if not isinstance(raw_action, dict):
                raise OpenAINativeProtocolError(
                    f"computer_call.actions[{index}] must be an object"
                )
            native = copy.deepcopy(raw_action)
            native_actions.append(native)
            queued_actions.append(
                validate_native_action(
                    native,
                    size_px=observation.size_px,
                )
            )
        return _ParsedComputerCall(
            response_id=response_id,
            call_id=call_id,
            native_actions=native_actions,
            queued_actions=queued_actions,
        )

    def _request_and_consume(
        self,
        body: Mapping[str, Any],
        *,
        stage: str,
        observation: Observation,
    ) -> None:
        if self.max_responses is not None and self.response_count >= self.max_responses:
            raise OpenAIResponseBudgetExhausted("Model-response budget exhausted")
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
                    {
                        "attempt": attempt,
                        "latency_s": time.time() - attempt_started,
                        "error": None,
                    }
                )
                break
            except Exception as exc:
                last_error = exc
                attempts.append(
                    {
                        "attempt": attempt,
                        "latency_s": time.time() - attempt_started,
                        "error": {
                            "type": type(exc).__name__,
                            "message": self._safe_error_text(exc),
                        },
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
            "reasoning_context": "model_default",
            "stage": stage,
            "base_url": _safe_base_url(self.config.base_url),
            "request": dict(body),
            "request_attempts": attempts,
            "api_key_persisted": False,
        }
        if response is None:
            error = OpenAINativeAPIError(
                f"OpenAI returned no response after {len(attempts)} attempt(s)"
            )
            cause = last_error or error
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
            raise error from last_error

        self.response_count += 1
        try:
            parsed = self._parse_computer_call(
                response,
                observation=observation,
            )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, OpenAINativeComputerError)
                else OpenAINativeProtocolError(str(exc))
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
        self._end_turn_pending = parsed.end_turn
        self._pending_actions.extend(parsed.queued_actions)
        self.last_native_action_batch_json = copy.deepcopy(parsed.native_actions)
        self.last_error = None
        self._write_call_log(
            call_index,
            {
                **common_log,
                "response": response,
                "response_id": parsed.response_id,
                "computer_call_id": parsed.call_id,
                "native_actions": parsed.native_actions,
                "returned_actions": [
                    action.to_dict()
                    for action in parsed.queued_actions
                    if action.kind not in {"screenshot", "wait"}
                ],
                "error": None,
                "latency_s": time.time() - started_at,
            },
        )

    def _set_last_native_action(self, action: OpenAIComputerAction) -> None:
        native = action.to_dict()
        self.last_native_action_json = native
        self.last_raw_prediction = json.dumps(
            native,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _drain_pending_actions(
        self,
        observation: Observation,
        *,
        history_length: int,
    ) -> tuple[OpenAIComputerAction | None, Observation]:
        current = observation
        while self._pending_actions:
            action = self._pending_actions.popleft()
            self._set_last_native_action(action)
            if action.kind == "screenshot":
                current = self._fresh_observation_for_screenshot(current)
                continue
            if action.kind == "wait":
                self.waiter(2.0)
                continue
            self._prediction_index += 1
            self.last_prediction_index = self._prediction_index
            self._last_history_length = history_length
            return action, current
        return None, current

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str] | None = None,
        budget: int | None = None,
        condition: str | None = None,
    ) -> OpenAIComputerAction | None:
        """Return the next native environment action while preserving batch order."""

        del allowed_kinds, budget, condition
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

        screenshot_round_trips = 0
        current_obs = obs
        while True:
            action, current_obs = self._drain_pending_actions(
                current_obs,
                history_length=history_length,
            )
            if action is not None:
                return action
            if self._end_turn_pending:
                native_end = {"type": "end_turn"}
                self.last_native_action_json = native_end
                self.last_raw_prediction = json.dumps(native_end, separators=(",", ":"))
                self._last_history_length = history_length
                return None

            if not self._initial_request_sent:
                body = self._initial_request(current_obs)
                stage = "initial_task"
            else:
                screenshot_round_trips += 1
                if screenshot_round_trips > self.max_screenshot_round_trips:
                    raise OpenAINativeProtocolError(
                        "computer tool exceeded the screenshot-only round-trip limit"
                    )
                body = self._screenshot_output_request(current_obs)
                stage = "computer_call_output"
            self._request_and_consume(
                body,
                stage=stage,
                observation=current_obs,
            )
            if stage == "initial_task":
                self._initial_request_sent = True


__all__ = [
    "OPENAI_NATIVE_COMPUTER_PROTOCOL",
    "OPENAI_NATIVE_HISTORY_STRATEGY",
    "OPENAI_COMPUTER_ACTION_KINDS",
    "OpenAIComputerAction",
    "OpenAINativeAPIError",
    "OpenAINativeComputerBackend",
    "OpenAINativeComputerConfig",
    "OpenAINativeComputerError",
    "OpenAINativeProtocolError",
    "OpenAINativeSafetyCheckError",
    "validate_native_action",
]
