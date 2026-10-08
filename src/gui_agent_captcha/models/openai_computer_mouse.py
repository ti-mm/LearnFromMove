"""Native OpenAI computer use with independent left-button function tools."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..core import Observation
from .openai_native_computer import (
    OpenAIComputerAction,
    OpenAINativeComputerBackend,
    OpenAINativeProtocolError,
    _ParsedComputerCall,
)

COMPUTER_MOUSE_PROTOCOL = "openai_responses_computer_mouse_functions_task_instruction"
TASK_INSTRUCTION_CONTRACT = "episode_instruction_only"
MOUSE_FUNCTION_NAMES = ("mouse_down", "mouse_up")


def computer_mouse_tools() -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [{"type": "computer"}]
    for name, description in (
        ("mouse_down", "Press and hold the left mouse button at the current cursor position. "
         "The button remains held across tool calls, movements, and screenshots until mouse_up."),
        ("mouse_up", "Release the left mouse button at the current cursor position."),
    ):
        tools.append({
            "type": "function", "name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": {}, "required": [],
                           "additionalProperties": False},
        })
    return tools


class OpenAIComputerMouseBackend(OpenAINativeComputerBackend):
    """Keep native batching/history and dispatch one serial function call at a time."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.protocol_track = COMPUTER_MOUSE_PROTOCOL
        self._pending_function = False

    def reset_episode(self) -> None:
        super().reset_episode()
        self._pending_function = False

    def _initial_request(self, observation: Observation) -> dict[str, Any]:
        if not observation.instruction.strip():
            raise OpenAINativeProtocolError("observation instruction must not be empty")
        body = super()._initial_request(observation)
        body.update(tools=computer_mouse_tools(), input=observation.instruction,
                    parallel_tool_calls=False)
        return body

    def _screenshot_output_request(self, observation: Observation) -> dict[str, Any]:
        body = super()._screenshot_output_request(observation)
        body["tools"] = computer_mouse_tools()
        body["parallel_tool_calls"] = False
        if self._pending_function:
            body["input"] = [{
                "type": "function_call_output", "call_id": self._pending_call_id,
                "output": "Completed.",
            }]
        return body

    def _write_call_log(self, call_index: int, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        request = payload.get("request", {})
        items = request.get("input")
        if isinstance(items, list) and items and items[0].get("type") == "function_call_output":
            payload["stage"] = "function_call_output"
        response = payload.get("response") or {}
        output = response.get("output")
        if isinstance(output, list) and any(
            item.get("type") == "function_call" for item in output if isinstance(item, dict)
        ):
            payload["function_call_id"] = payload.pop("computer_call_id", None)
        super()._write_call_log(call_index, payload)

    def _parse_computer_call(
        self, response: Mapping[str, Any], *, observation: Observation,
    ) -> _ParsedComputerCall:
        output = response.get("output")
        if not isinstance(output, list):
            return super()._parse_computer_call(response, observation=observation)
        calls = [item for item in output if isinstance(item, dict)
                 and item.get("type") in {"computer_call", "function_call"}]
        if len(calls) > 1:
            raise OpenAINativeProtocolError("serial computer mouse protocol expects at most one tool call")
        self._pending_function = bool(calls and calls[0]["type"] == "function_call")
        if not self._pending_function:
            return super()._parse_computer_call(response, observation=observation)
        call = calls[0]
        if response.get("status") != "completed":
            raise OpenAINativeProtocolError("function response is not completed")
        if call.get("status") not in {None, "completed"}:
            raise OpenAINativeProtocolError("function call is not completed")
        for value, label in ((response.get("id"), "response id"),
                             (call.get("call_id"), "function call_id")):
            if not isinstance(value, str) or not value:
                raise OpenAINativeProtocolError(f"missing {label}")
        if call.get("name") not in MOUSE_FUNCTION_NAMES:
            raise OpenAINativeProtocolError(f"unknown mouse function: {call.get('name')!r}")
        try:
            arguments = json.loads(call["arguments"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenAINativeProtocolError("mouse function arguments must be JSON {}") from exc
        if arguments != {}:
            raise OpenAINativeProtocolError("mouse functions take no arguments")
        action = OpenAIComputerAction(action={"type": call["name"]}, tool_type="function")
        return _ParsedComputerCall(
            response_id=response["id"], call_id=call["call_id"],
            native_actions=[], queued_actions=[action],
        )
