"""Restricted ``exec_py`` runtime for the paired Rotation benchmark."""

from __future__ import annotations

import ast
import base64
import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..actions import PrimitiveAction
from ..core import Observation, StepResult


class RotationCodeExecutionError(ValueError):
    """Generated code is outside the benchmark's supported Python surface."""


class RotationCodeExecutionEnvironmentError(RuntimeError):
    """The Rotation environment failed while executing a valid UI action."""


class _FirstReleaseStop(Exception):
    pass


@dataclass(frozen=True)
class _Point:
    x: float
    y: float

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.x
        yield self.y


@dataclass(frozen=True)
class _Size:
    width: int
    height: int

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.width
        yield self.height


@dataclass(frozen=True)
class _Screenshot:
    path: str
    size: tuple[int, int]


@dataclass
class RotationExecPyResult:
    output: list[dict[str, Any]]
    observation: Observation
    action_rows: list[dict[str, Any]] = field(default_factory=list)
    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    runtime_error: dict[str, str] | None = None
    first_release_audit: dict[str, Any] | None = None

    @property
    def first_release_seen(self) -> bool:
        return self.first_release_audit is not None


class RotationExecPyRuntime:
    """Interpret the small PyAutoGUI subset needed by the Rotation task.

    This deliberately does not call Python's ``exec``. Generated programs can
    inspect screenshots and issue mouse actions, but cannot access the host
    filesystem, network, shell, browser DOM, imports, or Python object model.
    """

    _MAX_SOURCE_CHARS = 20_000
    _MAX_STATEMENTS = 200
    _MAX_LOOP_ITEMS = 100
    _SAFE_CALLS = {
        "abs": abs,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "round": round,
        "str": str,
    }

    def __init__(self, environment: object, observation: Observation) -> None:
        self.environment = environment
        self.observation = observation
        self._variables: dict[str, Any] = {
            "pyautogui": "pyautogui",
            "time": "time",
        }
        self._logs: list[str] = []
        self._action_rows: list[dict[str, Any]] = []
        self._audit_rows: list[dict[str, Any]] = []
        self._first_release_audit: dict[str, Any] | None = None
        self._call_index = 0
        self._statement_count = 0

    @property
    def first_release_audit(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._first_release_audit)

    def execute(self, code: str, *, call_index: int) -> RotationExecPyResult:
        self._call_index = call_index
        self._statement_count = 0
        action_start = len(self._action_rows)
        audit_start = len(self._audit_rows)
        log_start = len(self._logs)
        runtime_error: dict[str, str] | None = None
        if len(code) > self._MAX_SOURCE_CHARS:
            runtime_error = {
                "type": "RotationCodeExecutionError",
                "message": f"code exceeds {self._MAX_SOURCE_CHARS} characters",
            }
        else:
            try:
                module = ast.parse(code, mode="exec")
                self._execute_statements(module.body)
            except _FirstReleaseStop:
                pass
            except RotationCodeExecutionError as error:
                runtime_error = {"type": type(error).__name__, "message": str(error)}

        self.observation = self._fresh_observation()
        text = "Script completed. A current screenshot is attached."
        if runtime_error is not None:
            text = (
                f"Script error: {runtime_error['message']}. "
                "A current screenshot is attached; correct the code and try again."
            )
        new_logs = self._logs[log_start:]
        if new_logs:
            text += "\nProgram output:\n" + "\n".join(new_logs)
        output = [
            {"type": "input_text", "text": text},
            self._image_output(self.observation.screenshot_path),
        ]
        return RotationExecPyResult(
            output=output,
            observation=self.observation,
            action_rows=copy.deepcopy(self._action_rows[action_start:]),
            audit_rows=copy.deepcopy(self._audit_rows[audit_start:]),
            logs=list(new_logs),
            runtime_error=runtime_error,
            first_release_audit=self.first_release_audit,
        )

    def _execute_statements(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self._statement_count += 1
            if self._statement_count > self._MAX_STATEMENTS:
                raise RotationCodeExecutionError(
                    f"program exceeds {self._MAX_STATEMENTS} executed statements"
                )
            self._execute_statement(statement)

    def _execute_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Expr):
            self._evaluate(statement.value)
            return
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            value_node = statement.value
            if value_node is None:
                raise RotationCodeExecutionError("annotation-only assignment is unsupported")
            value = self._evaluate(value_node)
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                self._assign(target, value)
            return
        if isinstance(statement, ast.AugAssign):
            if not isinstance(statement.target, ast.Name):
                raise RotationCodeExecutionError("augmented assignment requires a name")
            current = self._lookup(statement.target.id)
            value = self._binary(statement.op, current, self._evaluate(statement.value))
            self._variables[statement.target.id] = value
            return
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name not in {"pyautogui", "time"}:
                    raise RotationCodeExecutionError(f"import {alias.name!r} is not available")
                self._variables[alias.asname or alias.name] = alias.name
            return
        if isinstance(statement, ast.If):
            branch = (
                statement.body if self._truth(self._evaluate(statement.test)) else statement.orelse
            )
            self._execute_statements(branch)
            return
        if isinstance(statement, ast.For):
            values = list(self._evaluate(statement.iter))
            if len(values) > self._MAX_LOOP_ITEMS:
                raise RotationCodeExecutionError(f"loop exceeds {self._MAX_LOOP_ITEMS} items")
            for value in values:
                self._assign(statement.target, value)
                self._execute_statements(statement.body)
            if statement.orelse:
                self._execute_statements(statement.orelse)
            return
        if isinstance(statement, ast.Pass):
            return
        raise RotationCodeExecutionError(f"unsupported statement: {type(statement).__name__}")

    def _evaluate(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._lookup(node.id)
        if isinstance(node, ast.List):
            return [self._evaluate(item) for item in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._evaluate(item) for item in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._evaluate(key): self._evaluate(value)
                for key, value in zip(node.keys, node.values, strict=True)
                if key is not None
            }
        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.Not):
                return not self._truth(value)
            raise RotationCodeExecutionError(
                f"unsupported unary operator: {type(node.op).__name__}"
            )
        if isinstance(node, ast.BinOp):
            return self._binary(node.op, self._evaluate(node.left), self._evaluate(node.right))
        if isinstance(node, ast.BoolOp):
            values = [self._truth(self._evaluate(value)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Subscript):
            return self._evaluate(node.value)[self._evaluate(node.slice)]
        if isinstance(node, ast.Attribute):
            return self._attribute(self._evaluate(node.value), node.attr)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.JoinedStr):
            return "".join(str(self._evaluate(value)) for value in node.values)
        if isinstance(node, ast.FormattedValue):
            return self._evaluate(node.value)
        raise RotationCodeExecutionError(f"unsupported expression: {type(node).__name__}")

    def _call(self, node: ast.Call) -> Any:
        args = [self._evaluate(arg) for arg in node.args]
        kwargs = {
            keyword.arg: self._evaluate(keyword.value)
            for keyword in node.keywords
            if keyword.arg is not None
        }
        if len(kwargs) != len(node.keywords):
            raise RotationCodeExecutionError("expanded keyword arguments are unsupported")
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in {"display", "log", "print"}:
                return self._display_or_log(name, args, kwargs)
            if name == "range":
                values = list(range(*[int(value) for value in args]))
                if len(values) > self._MAX_LOOP_ITEMS:
                    raise RotationCodeExecutionError(f"range exceeds {self._MAX_LOOP_ITEMS} items")
                return values
            function = self._SAFE_CALLS.get(name)
            if function is None:
                raise RotationCodeExecutionError(f"call {name!r} is not available")
            return function(*args, **kwargs)
        if isinstance(node.func, ast.Attribute):
            owner = self._evaluate(node.func.value)
            if owner == "pyautogui":
                return self._pyautogui(node.func.attr, args, kwargs)
            if owner == "time" and node.func.attr == "sleep":
                return self._sleep(args, kwargs)
        raise RotationCodeExecutionError("only approved function calls are available")

    def _pyautogui(self, name: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        aliases = {
            "move_to": "moveTo",
            "move_rel": "moveRel",
            "mouse_down": "mouseDown",
            "mouse_up": "mouseUp",
            "drag_to": "dragTo",
            "drag_rel": "dragRel",
        }
        name = aliases.get(name, name)
        if name == "screenshot":
            self._require_call_shape(name, args, kwargs, max_args=0)
            self.observation = self._fresh_observation()
            return _Screenshot(self.observation.screenshot_path, self.observation.size_px)
        if name == "position":
            self._require_call_shape(name, args, kwargs, max_args=0)
            x, y = self.observation.cursor_xy or (0.0, 0.0)
            return _Point(float(x), float(y))
        if name == "size":
            self._require_call_shape(name, args, kwargs, max_args=0)
            width, height = self.observation.size_px
            return _Size(width, height)
        if name in {"sleep", "pause"}:
            return self._sleep(args, kwargs)
        if name == "moveTo":
            x, y = self._xy_arguments(name, args, kwargs)
            self._step(
                PrimitiveAction(kind="move_to", x=self._normalized_x(x), y=self._normalized_y(y))
            )
            return None
        if name == "moveRel":
            dx, dy = self._xy_arguments(name, args, kwargs, x_key="xOffset", y_key="yOffset")
            current_x, current_y = self.observation.cursor_xy or (0.0, 0.0)
            self._step(
                PrimitiveAction(
                    kind="move_to",
                    x=self._normalized_x(current_x + dx),
                    y=self._normalized_y(current_y + dy),
                )
            )
            return None
        if name in {"mouseDown", "mouseUp"}:
            button = self._button(args, kwargs)
            if button != "left":
                raise RotationCodeExecutionError("only the left mouse button is supported")
            self._step(PrimitiveAction(kind="mouse_down" if name == "mouseDown" else "mouse_up"))
            return None
        if name in {"dragTo", "dragRel"}:
            button = str(kwargs.get("button", "left"))
            if button != "left":
                raise RotationCodeExecutionError("only the left mouse button is supported")
            x, y = self._xy_arguments(
                name,
                args,
                kwargs,
                x_key="xOffset" if name == "dragRel" else "x",
                y_key="yOffset" if name == "dragRel" else "y",
            )
            if name == "dragRel":
                current_x, current_y = self.observation.cursor_xy or (0.0, 0.0)
                x += current_x
                y += current_y
            self._step(PrimitiveAction(kind="mouse_down"))
            self._step(
                PrimitiveAction(kind="move_to", x=self._normalized_x(x), y=self._normalized_y(y))
            )
            self._step(PrimitiveAction(kind="mouse_up"))
            return None
        if name == "click":
            button = str(kwargs.pop("button", "left"))
            clicks = int(kwargs.pop("clicks", 1))
            kwargs.pop("interval", None)
            if button != "left" or clicks != 1:
                raise RotationCodeExecutionError("only one left click is supported")
            if args or "x" in kwargs or "y" in kwargs:
                x, y = self._xy_arguments(name, args, kwargs)
                self._step(
                    PrimitiveAction(
                        kind="move_to", x=self._normalized_x(x), y=self._normalized_y(y)
                    )
                )
            elif kwargs:
                raise RotationCodeExecutionError(f"unsupported click arguments: {sorted(kwargs)}")
            self._step(PrimitiveAction(kind="mouse_down"))
            self._step(PrimitiveAction(kind="mouse_up"))
            return None
        raise RotationCodeExecutionError(f"pyautogui.{name} is not available")

    def _step(self, action: PrimitiveAction) -> None:
        before = self.observation
        step = getattr(self.environment, "step", None)
        if not callable(step):
            raise RotationCodeExecutionEnvironmentError("environment has no step()")
        try:
            result = step(action)
        except Exception as error:
            raise RotationCodeExecutionEnvironmentError(
                f"environment step failed for {action.kind}: {error}"
            ) from error
        if not isinstance(result, StepResult):
            raise RotationCodeExecutionEnvironmentError(
                "environment step() returned an invalid value"
            )
        self.observation = result.observation
        row = {
            "code_call_index": self._call_index,
            "action_index": len(self._action_rows) + 1,
            "action": action.to_dict(),
            "observation_before_path": before.screenshot_path,
            "observation_after_path": result.observation.screenshot_path,
            "reward": result.reward,
            "environment_done": result.done,
            "environment_info": copy.deepcopy(result.info),
        }
        self._action_rows.append(row)
        audit = self._evaluator_audit()
        self._audit_rows.append(
            {
                "code_call_index": self._call_index,
                "action_index": len(self._action_rows),
                "evaluator_only": audit,
            }
        )
        if self._is_first_release_audit(audit):
            self._first_release_audit = copy.deepcopy(audit)
            raise _FirstReleaseStop

    def _is_first_release_audit(self, audit: Mapping[str, Any]) -> bool:
        rotation_state = audit.get("rotation_state")
        attempt = rotation_state.get("attempt") if isinstance(rotation_state, Mapping) else None
        return isinstance(attempt, Mapping)

    def _fresh_observation(self) -> Observation:
        refresh = getattr(self.environment, "refresh_observation", None)
        if not callable(refresh):
            return self.observation
        try:
            observation = refresh()
        except Exception as error:
            raise RotationCodeExecutionEnvironmentError(
                f"environment screenshot failed: {error}"
            ) from error
        if not isinstance(observation, Observation):
            raise RotationCodeExecutionEnvironmentError(
                "environment returned an invalid observation"
            )
        return observation

    def _evaluator_audit(self) -> dict[str, Any]:
        get_audit = getattr(self.environment, "get_evaluator_audit", None)
        if not callable(get_audit):
            return {}
        audit = get_audit()
        return copy.deepcopy(audit) if isinstance(audit, dict) else {}

    def _image_output(self, screenshot_path: str) -> dict[str, Any]:
        path = Path(screenshot_path)
        try:
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as error:
            raise RotationCodeExecutionEnvironmentError(
                f"could not read screenshot {path}: {error}"
            ) from error
        return {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{encoded}",
            "detail": "original",
        }

    def _display_or_log(
        self,
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            raise RotationCodeExecutionError(f"{name} keyword arguments are unsupported")
        if name == "display":
            if len(args) != 1 or not isinstance(args[0], _Screenshot):
                raise RotationCodeExecutionError("display expects one screenshot")
            return
        text = " ".join(str(value) for value in args)
        self._logs.append(text[:2_000])

    def _sleep(self, args: list[Any], kwargs: dict[str, Any]) -> None:
        seconds = kwargs.pop("seconds", args[0] if args else None)
        if len(args) > 1 or kwargs or seconds is None:
            raise RotationCodeExecutionError("sleep expects one duration")
        value = self._number(seconds, "seconds")
        if value < 0 or value > 30:
            raise RotationCodeExecutionError("sleep duration must be between 0 and 30 seconds")
        time.sleep(min(value, 2.0))

    def _xy_arguments(
        self,
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        x_key: str = "x",
        y_key: str = "y",
    ) -> tuple[float, float]:
        local = dict(kwargs)
        local.pop("duration", None)
        local.pop("tween", None)
        local.pop("button", None)
        if len(args) > 2:
            raise RotationCodeExecutionError(f"{name} accepts at most two coordinates")
        x = args[0] if args else local.pop(x_key, None)
        y = args[1] if len(args) > 1 else local.pop(y_key, None)
        if x is None or y is None or local:
            raise RotationCodeExecutionError(f"invalid {name} arguments")
        return self._number(x, x_key), self._number(y, y_key)

    def _button(self, args: list[Any], kwargs: dict[str, Any]) -> str:
        local = dict(kwargs)
        local.pop("duration", None)
        local.pop("tween", None)
        button = args[0] if args else local.pop("button", "left")
        if len(args) > 1 or local:
            raise RotationCodeExecutionError("invalid mouse button arguments")
        return str(button)

    @staticmethod
    def _require_call_shape(
        name: str,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        max_args: int,
    ) -> None:
        if len(args) > max_args or kwargs:
            raise RotationCodeExecutionError(f"invalid {name} arguments")

    def _normalized_x(self, value: Any) -> float:
        width, _ = self.observation.size_px
        x = self._number(value, "x")
        if x < 0 or x >= width:
            raise RotationCodeExecutionError(f"x coordinate {x} is outside 0..{width - 1}")
        return x * 1000.0 / width

    def _normalized_y(self, value: Any) -> float:
        _, height = self.observation.size_px
        y = self._number(value, "y")
        if y < 0 or y >= height:
            raise RotationCodeExecutionError(f"y coordinate {y} is outside 0..{height - 1}")
        return y * 1000.0 / height

    @staticmethod
    def _number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RotationCodeExecutionError(f"{name} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise RotationCodeExecutionError(f"{name} must be finite")
        return number

    def _assign(self, target: ast.expr, value: Any) -> None:
        if isinstance(target, ast.Name):
            if target.id.startswith("_"):
                raise RotationCodeExecutionError("private names are unavailable")
            self._variables[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = list(value)
            if len(values) != len(target.elts):
                raise RotationCodeExecutionError("assignment unpacking length mismatch")
            for item, assigned in zip(target.elts, values, strict=True):
                self._assign(item, assigned)
            return
        raise RotationCodeExecutionError("assignment target must be a name")

    def _lookup(self, name: str) -> Any:
        if name.startswith("_") or name not in self._variables:
            raise RotationCodeExecutionError(f"name {name!r} is not available")
        return self._variables[name]

    @staticmethod
    def _attribute(owner: Any, name: str) -> Any:
        if name.startswith("_"):
            raise RotationCodeExecutionError("private attributes are unavailable")
        if isinstance(owner, _Point) and name in {"x", "y"}:
            return getattr(owner, name)
        if isinstance(owner, _Size) and name in {"width", "height"}:
            return getattr(owner, name)
        if isinstance(owner, _Screenshot) and name == "size":
            return owner.size
        raise RotationCodeExecutionError(f"attribute {name!r} is not available")

    @staticmethod
    def _binary(operator: ast.operator, left: Any, right: Any) -> Any:
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
        }
        operation = operations.get(type(operator))
        if operation is None:
            raise RotationCodeExecutionError(
                f"unsupported binary operator: {type(operator).__name__}"
            )
        return operation()

    def _compare(self, node: ast.Compare) -> bool:
        left = self._evaluate(node.left)
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = self._evaluate(comparator)
            if isinstance(operator, ast.Eq):
                passed = left == right
            elif isinstance(operator, ast.NotEq):
                passed = left != right
            elif isinstance(operator, ast.Lt):
                passed = left < right
            elif isinstance(operator, ast.LtE):
                passed = left <= right
            elif isinstance(operator, ast.Gt):
                passed = left > right
            elif isinstance(operator, ast.GtE):
                passed = left >= right
            else:
                raise RotationCodeExecutionError(
                    f"unsupported comparison: {type(operator).__name__}"
                )
            if not passed:
                return False
            left = right
        return True

    @staticmethod
    def _truth(value: Any) -> bool:
        return bool(value)


class TaskTerminalExecPyRuntime(RotationExecPyRuntime):
    """Stop Drag and Ten-Choice at submission in either perspective."""

    def _is_first_release_audit(self, audit: Mapping[str, Any]) -> bool:
        return isinstance(audit.get("terminal_success"), bool)


class FirstPersonExecPyRuntime(TaskTerminalExecPyRuntime):
    """Egocentric interface using the shared task-terminal execution contract."""


__all__ = [
    "FirstPersonExecPyRuntime",
    "TaskTerminalExecPyRuntime",
    "RotationCodeExecutionEnvironmentError",
    "RotationCodeExecutionError",
    "RotationExecPyResult",
    "RotationExecPyRuntime",
]
