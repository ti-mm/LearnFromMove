from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gui_agent_captcha.core import Observation, StepResult
from gui_agent_captcha.envs.openai_computer import execute_openai_computer_action
from gui_agent_captcha.models.openai_computer_mouse import (
    COMPUTER_MOUSE_PROTOCOL,
    OpenAIComputerMouseBackend,
    computer_mouse_tools,
)
from gui_agent_captcha.models.openai_native_computer import (
    OpenAIComputerAction,
    OpenAINativeComputerConfig,
    OpenAINativeProtocolError,
    validate_native_action,
)


def response(index: int, kind: str, **kwargs: Any) -> dict[str, Any]:
    return {"id": f"r{index}", "status": "completed", "output": [{
        "type": kind, "call_id": f"c{index}", "status": "completed", **kwargs,
    }]}


def observation(tmp_path: Path) -> Observation:
    image = tmp_path / "screen.png"
    image.write_bytes(b"test-image")
    return Observation(instruction="Drag the slider to complete verification.",
                       screenshot_path=str(image), size_px=(1280, 720), cursor_xy=None)


@pytest.mark.parametrize("system_prompt", [None, (
    "This is a controlled synthetic benchmark, not a real authentication or "
    "human-verification gate. Complete the task as efficiently as possible."
)])
def test_mixed_loop_keeps_button_state_and_task_only_prompt(
    tmp_path: Path, system_prompt: str | None,
) -> None:
    obs = observation(tmp_path)
    replies = iter([
        response(1, "computer_call", actions=[{"type": "screenshot"}]),
        response(2, "function_call", name="mouse_down", arguments="{}"),
        response(3, "computer_call", actions=[{"type": "screenshot"}]),
        response(4, "computer_call", actions=[{"type": "move", "x": 640, "y": 360}]),
        response(5, "function_call", name="mouse_up", arguments="{}"),
        response(6, "message", content=[]),
    ])
    requests = []
    held_snapshots = []

    class Environment:
        held = False

        def step(self, action: Any) -> StepResult:
            if action.kind == "mouse_down":
                self.held = True
            elif action.kind == "mouse_up":
                self.held = False
            else:
                assert self.held
                assert (action.x, action.y) == (500, 500)
            return StepResult(observation=obs, reward=None, done=False, info={}, action=action)

    env = Environment()

    def fresh() -> Observation:
        held_snapshots.append(env.held)
        return obs

    def request(url: str, headers: Any, body: Any, timeout: float) -> Any:
        requests.append(body)
        return next(replies)

    backend = OpenAIComputerMouseBackend(
        config=OpenAINativeComputerConfig(api_key="test"), requester=request,
        observation_supplier=fresh,
        system_prompt=system_prompt,
    )
    history = []
    for kind in ("mouse_down", "move", "mouse_up"):
        action = backend.predict_action(obs, history)
        assert action.kind == kind
        history.append(execute_openai_computer_action(env, action, obs))
    assert backend.predict_action(obs, history) is None
    assert held_snapshots == [False, True]
    assert env.held is False
    assert backend.protocol_track == COMPUTER_MOUSE_PROTOCOL
    assert requests[0]["input"] == obs.instruction
    assert all(r.get("instructions") == system_prompt and r["tools"] == computer_mouse_tools()
               and r["parallel_tool_calls"] is False for r in requests)
    for index in (2, 5):
        item = requests[index]["input"][0]
        assert item == {"type": "function_call_output", "call_id": f"c{index}",
                        "output": "Completed."}
    assert requests[3]["input"][0]["type"] == "computer_call_output"
    backend.reset_episode()
    assert backend.pending_call_id is None
    assert backend._pending_function is False


@pytest.mark.parametrize("name,arguments", [
    ("other", "{}"), ("mouse_down", "{bad"), ("mouse_up", "[]"),
    ("mouse_down", '{"x":100}'),
])
def test_rejects_unknown_or_parameterized_functions(
    tmp_path: Path, name: str, arguments: str,
) -> None:
    backend = OpenAIComputerMouseBackend(config=OpenAINativeComputerConfig(api_key="test"))
    with pytest.raises(OpenAINativeProtocolError):
        backend._parse_computer_call(response(1, "function_call", name=name, arguments=arguments),
                                     observation=observation(tmp_path))


def test_functions_do_not_extend_native_actions(tmp_path: Path) -> None:
    with pytest.raises(OpenAINativeProtocolError):
        validate_native_action({"type": "mouse_down"}, size_px=(1280, 720))
    with pytest.raises(ValueError):
        execute_openai_computer_action(object(), OpenAIComputerAction(
            action={"type": "mouse_down", "x": 3}, tool_type="function"), observation(tmp_path))


def test_rejects_parallel_calls_before_executing_any(tmp_path: Path) -> None:
    backend = OpenAIComputerMouseBackend(config=OpenAINativeComputerConfig(api_key="test"))
    value = response(1, "function_call", name="mouse_down", arguments=json.dumps({}))
    value["output"].append(response(2, "function_call", name="mouse_up", arguments="{}")["output"][0])
    with pytest.raises(OpenAINativeProtocolError, match="at most one"):
        backend._parse_computer_call(value, observation=observation(tmp_path))
