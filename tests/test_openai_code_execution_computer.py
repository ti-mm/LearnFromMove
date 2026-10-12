from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gui_agent_captcha.models.openai_code_execution_computer import (
    OPENAI_CODE_EXECUTION_PROTOCOL,
    OpenAICodeExecutionAPIError,
    OpenAICodeExecutionComputerBackend,
    OpenAICodeExecutionConfig,
    OpenAICodeExecutionProtocolError,
    exec_py_tool_declaration,
)


def _response(
    response_id: str,
    *,
    call_id: str | None = "call-1",
    code: str = "display(pyautogui.screenshot())",
    model: str = "gpt-6-astra",
) -> dict[str, Any]:
    output: list[dict[str, Any]]
    if call_id is None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ]
    else:
        output = [
            {
                "type": "function_call",
                "name": "exec_py",
                "call_id": call_id,
                "arguments": json.dumps({"code": code}),
            }
        ]
    return {
        "id": response_id,
        "status": "completed",
        "model": model,
        "output": output,
    }


def test_exec_py_uses_function_call_output_and_previous_response_id() -> None:
    requests: list[dict[str, Any]] = []
    responses = iter(
        [
            _response("resp-1"),
            _response("resp-2", call_id=None),
        ]
    )

    def requester(
        url: str,
        headers: dict[str, str],
        body: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        requests.append({"url": url, "headers": headers, "body": body, "timeout_s": timeout_s})
        return next(responses)

    backend = OpenAICodeExecutionComputerBackend(
        config=OpenAICodeExecutionConfig(
            model="gpt-6-astra",
            base_url="https://api.modelverse.cn/v1/",
            api_key="secret",
        ),
        requester=requester,
    )

    first = backend.start("Rotate the image")
    second = backend.continue_with(
        [
            {"type": "input_text", "text": "ok"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,AAAA",
                "detail": "original",
            },
        ]
    )

    assert backend.protocol_track == OPENAI_CODE_EXECUTION_PROTOCOL
    assert first.code == "display(pyautogui.screenshot())"
    assert second.ended is True
    assert requests[0]["url"] == "https://api.modelverse.cn/v1/responses"
    assert requests[0]["body"]["tools"] == [exec_py_tool_declaration()]
    assert requests[0]["body"]["parallel_tool_calls"] is False
    assert requests[0]["body"]["reasoning"] == {"effort": "medium"}
    assert "previous_response_id" not in requests[0]["body"]
    assert requests[1]["body"]["previous_response_id"] == "resp-1"
    assert requests[1]["body"]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": [
                {"type": "input_text", "text": "ok"},
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original",
                },
            ],
        }
    ]


def test_response_model_must_match_exactly() -> None:
    backend = OpenAICodeExecutionComputerBackend(
        config=OpenAICodeExecutionConfig(api_key="secret"),
        requester=lambda *_args: _response("resp-1", model="gpt-5.6-sol"),
    )

    with pytest.raises(OpenAICodeExecutionProtocolError, match="response model"):
        backend.start("task")


def test_backend_preserves_explicit_gpt56_model() -> None:
    backend = OpenAICodeExecutionComputerBackend(
        config=OpenAICodeExecutionConfig(
            model="gpt-5.6-sol",
            api_key="secret",
        ),
        requester=lambda *_args: _response("resp-1", model="gpt-5.6-sol"),
    )

    turn = backend.start("task")

    assert turn.code == "display(pyautogui.screenshot())"
    assert backend.last_response_model == "gpt-5.6-sol"


def test_http_500_server_error_is_retried() -> None:
    calls = 0

    def requester(*_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError('HTTP 500: {"code":"server_error"}')
        return _response("resp-1")

    backend = OpenAICodeExecutionComputerBackend(
        config=OpenAICodeExecutionConfig(api_key="secret"),
        request_retries=2,
        requester=requester,
    )

    turn = backend.start("task")

    assert turn.code == "display(pyautogui.screenshot())"
    assert calls == 2


def test_api_key_is_redacted_from_failure_log(tmp_path: Path) -> None:
    api_key = "super-secret-key"

    def requester(*_args: Any) -> dict[str, Any]:
        raise RuntimeError(f"upstream rejected {api_key}")

    backend = OpenAICodeExecutionComputerBackend(
        config=OpenAICodeExecutionConfig(api_key=api_key),
        call_log_dir=tmp_path,
        request_retries=1,
        requester=requester,
    )

    with pytest.raises(OpenAICodeExecutionAPIError):
        backend.start("task")

    log = (tmp_path / "call_001.json").read_text(encoding="utf-8")
    assert api_key not in log
    assert "<redacted-api-key>" in log
