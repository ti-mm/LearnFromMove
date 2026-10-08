"""Responses API backend for code-execution computer use."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .openai_native_computer import (
    JsonRequester,
    _default_json_requester,
    _responses_url,
    _retryable_error,
    _safe_base_url,
    _sanitize_log_value,
)

OPENAI_CODE_EXECUTION_PROTOCOL = "openai_responses_function_exec_py"
OPENAI_CODE_EXECUTION_HISTORY = "previous_response_id_full_response_chain"
OPENAI_CODE_EXECUTION_TOOL_NAME = "exec_py"
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_REQUEST_RETRIES = 6

EXEC_PY_TOOL_DESCRIPTION = (
    "Run Python in a persistent 1280x720 benchmark desktop. Available globals: "
    "pyautogui, time, display(image), and log(value). Inspect the screen with "
    "display(pyautogui.screenshot()) before acting. Supported UI calls are "
    "pyautogui.moveTo(x, y), moveRel(dx, dy), mouseDown(button='left'), "
    "mouseUp(button='left'), dragTo(x, y, duration=..., button='left'), "
    "dragRel(dx, dy, duration=..., button='left'), click(x, y), screenshot(), "
    "position(), size(), and sleep(seconds). Do not import modules or access the "
    "filesystem, shell, network, or browser DOM. The runtime returns a current "
    "full-resolution screenshot after every script."
)


class OpenAICodeExecutionError(RuntimeError):
    """Base error for the Astra code-execution backend."""


class OpenAICodeExecutionAPIError(OpenAICodeExecutionError):
    """The Responses API request failed after infrastructure retries."""


class OpenAICodeExecutionProtocolError(OpenAICodeExecutionError):
    """The provider response violated the function-call contract."""


@dataclass(frozen=True)
class OpenAICodeExecutionConfig:
    model: str = "gpt-6-astra"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default="", repr=False)
    reasoning_effort: str = "medium"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("unsupported reasoning effort")


@dataclass(frozen=True)
class OpenAICodeExecutionTurn:
    response_id: str
    call_id: str | None
    code: str | None
    message_text: str | None

    @property
    def ended(self) -> bool:
        return self.call_id is None


def exec_py_tool_declaration() -> dict[str, Any]:
    return {
        "type": "function",
        "name": OPENAI_CODE_EXECUTION_TOOL_NAME,
        "description": EXEC_PY_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": True,
    }


def _message_text(output: list[Any]) -> str | None:
    texts: list[str] = []
    for item in output:
        if not isinstance(item, Mapping) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
    return "\n".join(texts) or None


class OpenAICodeExecutionComputerBackend:
    """Stateful Responses client for the official ``exec_py`` tool pattern."""

    def __init__(
        self,
        *,
        config: OpenAICodeExecutionConfig,
        system_prompt: str | None = None,
        prompt_extension: str | None = None,
        call_log_dir: Path | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        request_retries: int = DEFAULT_REQUEST_RETRIES,
        requester: JsonRequester | None = None,
    ) -> None:
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if request_retries < 1:
            raise ValueError("request_retries must be >= 1")
        self.config = config
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.prompt_extension = prompt_extension.strip() if prompt_extension else None
        self.call_log_dir = call_log_dir
        self.request_timeout_s = request_timeout_s
        self.request_retries = request_retries
        self.requester = requester or _default_json_requester
        self.protocol_track = OPENAI_CODE_EXECUTION_PROTOCOL
        self.history_strategy = OPENAI_CODE_EXECUTION_HISTORY
        self._url = _responses_url(config.base_url)
        self._previous_response_id: str | None = None
        self._pending_call_id: str | None = None
        self._call_index = 0
        self.last_call_log_path: str | None = None
        self.last_error: str | None = None
        self.last_response_model: str | None = None

    @property
    def call_count(self) -> int:
        return self._call_index

    def close(self) -> None:
        self._previous_response_id = None
        self._pending_call_id = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _retryable_error(error: BaseException) -> bool:
        text = str(error).lower()
        return _retryable_error(error) or any(
            marker in text for marker in ("http 500", "server_error")
        )

    def _body(self, next_input: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "reasoning": {"effort": self.config.reasoning_effort},
            "parallel_tool_calls": False,
            "tools": [exec_py_tool_declaration()],
            "input": next_input,
        }
        if self._previous_response_id is not None:
            body["previous_response_id"] = self._previous_response_id
        instructions = "\n\n".join(p for p in (self.system_prompt, self.prompt_extension) if p)
        if instructions:
            body["instructions"] = instructions
        return body

    def _write_call_log(self, payload: Mapping[str, Any]) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        path = self.call_log_dir / f"call_{self._call_index:03d}.json"
        temporary = path.with_suffix(".json.tmp")
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

    def _parse(self, response: Mapping[str, Any]) -> OpenAICodeExecutionTurn:
        if response.get("status") != "completed":
            raise OpenAICodeExecutionProtocolError(
                f"OpenAI response status is {response.get('status')!r}; "
                f"incomplete_details={response.get('incomplete_details')!r}"
            )
        response_id = response.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise OpenAICodeExecutionProtocolError("response is missing id")
        response_model = response.get("model")
        if response_model != self.config.model:
            raise OpenAICodeExecutionProtocolError(
                f"response model is {response_model!r}, expected {self.config.model!r}"
            )
        self.last_response_model = str(response_model)
        output = response.get("output")
        if not isinstance(output, list):
            raise OpenAICodeExecutionProtocolError("response output must be an array")
        calls = [
            item
            for item in output
            if isinstance(item, Mapping) and item.get("type") == "function_call"
        ]
        if not calls:
            return OpenAICodeExecutionTurn(
                response_id=response_id,
                call_id=None,
                code=None,
                message_text=_message_text(output),
            )
        if len(calls) != 1:
            raise OpenAICodeExecutionProtocolError(f"expected one function_call, got {len(calls)}")
        call = calls[0]
        if call.get("name") != OPENAI_CODE_EXECUTION_TOOL_NAME:
            raise OpenAICodeExecutionProtocolError(f"unexpected function {call.get('name')!r}")
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise OpenAICodeExecutionProtocolError("function_call is missing call_id")
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            raise OpenAICodeExecutionProtocolError("function_call arguments must be JSON text")
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise OpenAICodeExecutionProtocolError(
                "function_call arguments are not valid JSON"
            ) from exc
        if not isinstance(decoded, dict) or set(decoded) != {"code"}:
            raise OpenAICodeExecutionProtocolError("exec_py arguments must contain only code")
        code = decoded.get("code")
        if not isinstance(code, str) or not code.strip():
            raise OpenAICodeExecutionProtocolError("exec_py code must be non-empty")
        return OpenAICodeExecutionTurn(
            response_id=response_id,
            call_id=call_id,
            code=code,
            message_text=None,
        )

    def _request(self, next_input: Any, *, stage: str) -> OpenAICodeExecutionTurn:
        self._call_index += 1
        body = self._body(next_input)
        started_at = time.monotonic()
        attempts: list[dict[str, Any]] = []
        response: Mapping[str, Any] | None = None
        last_error: BaseException | None = None
        for attempt in range(1, self.request_retries + 1):
            try:
                response = self.requester(
                    self._url,
                    self._headers(),
                    body,
                    self.request_timeout_s,
                )
                attempts.append({"attempt": attempt, "ok": True})
                break
            except Exception as exc:
                last_error = exc
                attempts.append({"attempt": attempt, "ok": False, "error": type(exc).__name__})
                if attempt == self.request_retries or not self._retryable_error(exc):
                    break
                time.sleep(min(30.0, float(2 ** (attempt - 1))))

        common = {
            "call_index": self._call_index,
            "provider": "openai",
            "model": self.config.model,
            "base_url": _safe_base_url(self.config.base_url),
            "protocol_track": self.protocol_track,
            "history_strategy": self.history_strategy,
            "stage": stage,
            "request": body,
            "request_attempts": attempts,
            "api_key_persisted": False,
        }
        if response is None:
            error = OpenAICodeExecutionAPIError(
                f"OpenAI returned no response after {len(attempts)} attempt(s)"
            )
            self.last_error = f"{type(last_error or error).__name__}: {last_error or error}"
            self._write_call_log(
                {
                    **common,
                    "response": None,
                    "error": self.last_error,
                    "latency_s": time.monotonic() - started_at,
                }
            )
            raise error from last_error
        try:
            turn = self._parse(response)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._write_call_log(
                {
                    **common,
                    "response": response,
                    "error": self.last_error,
                    "latency_s": time.monotonic() - started_at,
                }
            )
            raise
        self._previous_response_id = turn.response_id
        self._pending_call_id = turn.call_id
        self.last_error = None
        self._write_call_log(
            {
                **common,
                "response": response,
                "response_id": turn.response_id,
                "function_call_id": turn.call_id,
                "generated_code": turn.code,
                "message_text": turn.message_text,
                "error": None,
                "latency_s": time.monotonic() - started_at,
            }
        )
        return turn

    def start(self, instruction: str) -> OpenAICodeExecutionTurn:
        if self._previous_response_id is not None:
            raise OpenAICodeExecutionProtocolError("episode has already started")
        if not instruction.strip():
            raise OpenAICodeExecutionProtocolError("instruction must not be empty")
        task = (
            f"{instruction.strip()}\n\nUse exec_py for all UI interaction. "
            "Inspect the current screen before acting and finish the task efficiently."
        )
        return self._request(task, stage="initial_task")

    def continue_with(
        self,
        output: list[dict[str, Any]],
    ) -> OpenAICodeExecutionTurn:
        if self._pending_call_id is None:
            raise OpenAICodeExecutionProtocolError(
                "cannot return output without a pending function_call"
            )
        if not output:
            raise OpenAICodeExecutionProtocolError("function output must not be empty")
        next_input = [
            {
                "type": "function_call_output",
                "call_id": self._pending_call_id,
                "output": output,
            }
        ]
        return self._request(next_input, stage="function_call_output")


__all__ = [
    "EXEC_PY_TOOL_DESCRIPTION",
    "OPENAI_CODE_EXECUTION_HISTORY",
    "OPENAI_CODE_EXECUTION_PROTOCOL",
    "OPENAI_CODE_EXECUTION_TOOL_NAME",
    "OpenAICodeExecutionAPIError",
    "OpenAICodeExecutionComputerBackend",
    "OpenAICodeExecutionConfig",
    "OpenAICodeExecutionError",
    "OpenAICodeExecutionProtocolError",
    "OpenAICodeExecutionTurn",
    "exec_py_tool_declaration",
]
