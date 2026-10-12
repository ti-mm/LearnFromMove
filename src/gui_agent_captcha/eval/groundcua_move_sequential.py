"""ScreenSpot-Pro-style sequential evaluation for the GroundCUA move contract.

The evaluator keeps the frozen Qwen3 move tool schema, but adds a closed-loop
conversation around it: every model call receives the complete previous action
history and at most the latest three causal observations.  The visible action
contract is always ``computer_use`` with ``move_to``, ``mouse_down``, or
``mouse_up``; the legacy ``action/kind`` representation is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..data.export_groundcua_multistep_subset import model_to_execution_pixel
from ..prompts.groundcua_qwen25_sequential import (
    Qwen25SequentialRender,
    render_qwen25_sequential,
)
from ..prompts.screenspot_pro_groundcua import (
    QWEN3_MOVE_IMAGE_HISTORY_MAX,
    QWEN3_MOVE_PROFILE,
    QWEN3_MOVE_PROMPT_TEMPLATE,
    QWEN3_MOVE_SEQUENTIAL_PROMPT_CONTRACT,
    QWEN25_MOVE_PROFILE,
    parse_groundcua_tool_call,
)

IMAGE_HISTORY_MAX = QWEN3_MOVE_IMAGE_HISTORY_MAX
ALLOWED_ACTIONS = ("move_to", "mouse_down", "mouse_up")
EXPECTED_SCREENSPOT_PRO_TOTAL = 1581


def prompt_template_sha256() -> str:
    return hashlib.sha256(QWEN3_MOVE_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_screenspot_pro_samples(
    *,
    annotations_dir: Path,
    images_dir: Path,
    sample_limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the official ScreenSpot-Pro annotation contract.

    A full run is deliberately count-locked to all 1,581 official samples;
    ``sample_limit`` is only for explicit CPU smoke tests.
    """

    annotations_dir = Path(annotations_dir).expanduser().resolve()
    images_dir = Path(images_dir).expanduser().resolve()
    annotation_files = sorted(annotations_dir.glob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(f"no ScreenSpot-Pro annotations under {annotations_dir}")
    if sample_limit is not None and sample_limit < 1:
        raise ValueError("sample_limit must be positive")

    samples: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    for annotation_path in annotation_files:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"annotation file is not a JSON list: {annotation_path}")
        input_files.append(
            {
                "path": str(annotation_path),
                "bytes": annotation_path.stat().st_size,
                "sha256": _sha256_file(annotation_path),
                "count": len(payload),
            }
        )
        for row_index, raw in enumerate(payload):
            if not isinstance(raw, dict):
                raise ValueError(f"annotation row is not an object: {annotation_path}:{row_index}")
            sample = dict(raw)
            required = ("id", "img_filename", "instruction", "bbox", "img_size")
            missing = [name for name in required if name not in sample]
            if missing:
                raise ValueError(f"annotation row missing {missing}: {annotation_path}:{row_index}")
            image_path = (images_dir / str(sample["img_filename"])).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
            sample.update(
                {
                    "task_filename": annotation_path.stem,
                    "gt_type": "positive",
                    "instruction_style": "instruction",
                    "language": "en",
                    "prompt_to_evaluate": str(sample["instruction"]),
                    "_annotation_row": row_index,
                    "_image_path": str(image_path),
                    "_image_sha256": _sha256_file(image_path),
                }
            )
            samples.append(sample)
            if sample_limit is not None and len(samples) >= sample_limit:
                break
        if sample_limit is not None and len(samples) >= sample_limit:
            break

    limited = sample_limit is not None
    if not limited and len(samples) != EXPECTED_SCREENSPOT_PRO_TOTAL:
        raise RuntimeError(
            "official ScreenSpot-Pro count mismatch: "
            f"{len(samples)} != {EXPECTED_SCREENSPOT_PRO_TOTAL}"
        )
    if limited and not samples:
        raise RuntimeError(f"ScreenSpot-Pro sample limit loaded no samples: {sample_limit}")
    return samples, {
        "schema": "screenspot_pro_official_annotations_manifest_v1",
        "name": "ScreenSpot-Pro",
        "split": "official annotations",
        "expected_total": EXPECTED_SCREENSPOT_PRO_TOTAL,
        "loaded_total": len(samples),
        "sample_limit": sample_limit,
        "limited": limited,
        "annotations_dir": str(annotations_dir),
        "images_dir": str(images_dir),
        "input_files": input_files,
        "task": "all",
        "inst_style": "instruction",
        "language": "en",
        "gt_type": "positive",
    }


def canonical_qwen3_move_prompt_template() -> str:
    """Render the frozen move template with Qwen3's explicit system role.

    ``QWEN3_MOVE_PROMPT_TEMPLATE`` is kept as the pinned ScreenSpot-Pro raw
    source contract (and therefore keeps its manifest hash).  The model-facing
    renderer adds the role marker that the Qwen3 chat template emits before
    that source content.  Training and evaluation both call this renderer.
    """

    marker = "<|im_start|>"
    if QWEN3_MOVE_PROMPT_TEMPLATE.startswith("<|im_start|>system\n"):
        return QWEN3_MOVE_PROMPT_TEMPLATE
    if not QWEN3_MOVE_PROMPT_TEMPLATE.startswith(marker):
        raise ValueError("QWEN3_MOVE_PROMPT_TEMPLATE must start with <|im_start|>")
    return marker + "system\n" + QWEN3_MOVE_PROMPT_TEMPLATE[len(marker) :]


@dataclass(frozen=True)
class SequentialPrompt:
    prompt: str
    image_paths: tuple[Path, ...]
    image_count: int
    action_history_count: int
    retained_image_start: int
    prompt_template_sha256: str
    prompt_contract: str = QWEN3_MOVE_SEQUENTIAL_PROMPT_CONTRACT


def _validate_instruction(instruction: str) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be nonempty")
    return instruction.strip()


def _validate_response_history(
    assistant_response_history: Sequence[str],
) -> tuple[str, ...]:
    responses: list[str] = []
    for index, response in enumerate(assistant_response_history):
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"assistant_response_history[{index}] must be nonempty")
        parse_single_tool_call(response)
        responses.append(response.strip())
    return tuple(responses)


def _history_block(
    *,
    observation_index: int,
    instruction: str,
    assistant_response: str,
    include_image: bool,
) -> str:
    observation = ""
    if include_image:
        observation = (
            f"Previous observation {observation_index + 1}:\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
        )
    return (
        "<|im_start|>user\n"
        f"{observation}"
        f"Task: {instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_response}<|im_end|>\n"
    )


def _insert_current_context(
    current_template_suffix: str,
    *,
    instruction: str,
) -> str:
    current = current_template_suffix.replace("{{instruction}}", instruction)
    current = current.replace(
        "<|im_start|>user\n<|vision_start|>",
        "<|im_start|>user\nCurrent observation:\n<|vision_start|>",
        1,
    )
    context_lines = [
        "At this step, return exactly one complete tool call for the next mouse primitive.",
        "Do not return multiple tool calls or hidden reasoning.",
        "The last attached image is the current observation.",
        "Coordinates for move_to must be integer values in the 0-1000 relative coordinate system.",
        "mouse_down and mouse_up must not include a coordinate.",
    ]
    marker = "<|im_end|>\n<|im_start|>assistant\n"
    if marker not in current:
        raise ValueError("QWEN3_MOVE_PROMPT_TEMPLATE assistant boundary is missing")
    return current.replace(
        marker,
        "\n" + "\n".join(context_lines) + "\n" + marker,
        1,
    )


def build_sequential_prompt(
    *,
    instruction: str,
    image_paths: Sequence[Path],
    assistant_response_history: Sequence[str],
    images_to_keep: int = IMAGE_HISTORY_MAX,
) -> SequentialPrompt:
    """Build one sequential prompt and return the images in marker order.

    ``image_paths[i]`` is the observation immediately before action ``i``;
    therefore a valid current-turn prefix has exactly one more image than
    historical assistant responses. Every historical action remains an
    independent assistant turn; only user turns whose observations fall
    outside the image window omit the image content.
    """

    normalized_instruction = _validate_instruction(instruction)
    paths = tuple(Path(path).expanduser().resolve() for path in image_paths)
    if not paths:
        raise ValueError("image_paths must contain the current observation")
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be >= 1")
    responses = _validate_response_history(assistant_response_history)
    if len(paths) != len(responses) + 1:
        raise ValueError(
            "image_paths must contain exactly one current observation in addition "
            "to one observation per historical action"
        )

    retained_start = max(0, len(paths) - images_to_keep)
    retained_paths = paths[retained_start:]
    template_prefix, template_suffix = canonical_qwen3_move_prompt_template().split(
        "<|im_start|>user", 1
    )
    prompt_parts = [template_prefix]
    for observation_index in range(len(paths) - 1):
        prompt_parts.append(
            _history_block(
                observation_index=observation_index,
                instruction=normalized_instruction,
                assistant_response=responses[observation_index],
                include_image=observation_index >= retained_start,
            )
        )
    prompt_parts.append(
        _insert_current_context(
            "<|im_start|>user" + template_suffix,
            instruction=normalized_instruction,
        )
    )
    return SequentialPrompt(
        prompt="".join(prompt_parts),
        image_paths=retained_paths,
        image_count=len(retained_paths),
        action_history_count=len(responses),
        retained_image_start=retained_start,
        prompt_template_sha256=prompt_template_sha256(),
    )


def parse_single_tool_call(response: str) -> dict[str, Any]:
    """Parse exactly one visible Qwen2.5 or legacy Qwen3 move tool call."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a nonempty tool call")
    stripped = response.strip()
    if (
        '"kind"' in stripped
        or "action/kind" in stripped
        or "<think>" in stripped.lower()
    ):
        raise ValueError(
            "legacy action/kind syntax or hidden reasoning in tool call is forbidden"
        )
    if not stripped.startswith("<tool_call>") or not stripped.endswith("</tool_call>"):
        raise ValueError("response must be exactly one complete tool call")
    try:
        action, coordinate = parse_groundcua_tool_call(QWEN25_MOVE_PROFILE, stripped)
    except ValueError:
        action, coordinate = parse_groundcua_tool_call(QWEN3_MOVE_PROFILE, stripped)
    if coordinate is not None:
        assert coordinate is not None
        return {"kind": action, "x": coordinate[0], "y": coordinate[1]}
    return {"kind": action}


@dataclass(frozen=True)
class ActionSequenceScore:
    actions: tuple[dict[str, Any], ...]
    last_mouse_move_before_mouse_down: dict[str, Any] | None
    last_mouse_move_before_mouse_down_index: int | None
    last_mouse_move_before_mouse_down_hit: bool
    mouse_down_seen: bool
    mouse_up_seen: bool
    mouse_down_followed_by_mouse_up: bool
    last_mouse_move_immediately_before_mouse_down: bool
    protocol_errors: tuple[str, ...]
    primary_success: bool

    @property
    def last_move_to_before_mouse_down(self) -> dict[str, Any] | None:
        return self.last_mouse_move_before_mouse_down

    @property
    def last_move_to_before_mouse_down_index(self) -> int | None:
        return self.last_mouse_move_before_mouse_down_index

    @property
    def last_move_to_before_mouse_down_hit(self) -> bool:
        return self.last_mouse_move_before_mouse_down_hit

    @property
    def last_move_to_immediately_before_mouse_down(self) -> bool:
        return self.last_mouse_move_immediately_before_mouse_down


def _physical_point_inside_bbox(
    resized_xy: tuple[float, float],
    *,
    bbox: tuple[float, float, float, float],
    resized_size: tuple[int, int],
    execution_size: tuple[int, int],
) -> bool:
    x = resized_xy[0] * execution_size[0] / resized_size[0]
    y = resized_xy[1] * execution_size[1] / resized_size[1]
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def score_action_sequence(
    *,
    actions: Iterable[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    resized_size: tuple[int, int] | None = None,
    execution_size: tuple[int, int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> ActionSequenceScore:
    """Score the last move before press and require a later release.

    ``resized_size`` and ``execution_size`` select the Qwen2.5 physical-pixel
    path. ``image_size`` remains as a compatibility input for the Qwen3
    0-1000 draft and is not used by the Qwen2.5 evaluator.
    """

    legacy_qwen3 = resized_size is None and execution_size is None and image_size is not None
    if legacy_qwen3:
        resized_size = (1000, 1000)
        execution_size = image_size
        move_kind = "move_to"
    else:
        move_kind = "mouse_move"
    if resized_size is None or execution_size is None:
        raise ValueError("resized_size and execution_size are required")
    for name, size in (("resized_size", resized_size), ("execution_size", execution_size)):
        if len(size) != 2 or any(int(value) < 1 for value in size):
            raise ValueError(f"{name} must contain two positive integers")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if not (x1 <= x2 and y1 <= y2):
        raise ValueError("bbox must be ordered as x1,y1,x2,y2")

    normalized_actions = tuple(dict(action) for action in actions)
    protocol_errors: list[str] = []
    last_move: dict[str, Any] | None = None
    last_move_index: int | None = None
    selected_move: dict[str, Any] | None = None
    selected_move_index: int | None = None
    mouse_down_index: int | None = None
    mouse_up_index: int | None = None
    button_down = False
    for index, action in enumerate(normalized_actions):
        kind = action.get("kind")
        if kind == move_kind:
            x_value, y_value = action.get("x"), action.get("y")
            if (
                isinstance(x_value, bool)
                or isinstance(y_value, bool)
                or not isinstance(x_value, (int, float))
                or not isinstance(y_value, (int, float))
                or not math.isfinite(float(x_value))
                or not math.isfinite(float(y_value))
                or not 0 <= float(x_value) <= resized_size[0]
                or not 0 <= float(y_value) <= resized_size[1]
            ):
                protocol_errors.append(f"invalid {move_kind} at action {index}")
                continue
            last_move = {"kind": move_kind, "x": x_value, "y": y_value}
            last_move_index = index
            continue
        if kind == "mouse_down":
            if button_down:
                protocol_errors.append(f"duplicate mouse_down at action {index}")
            elif mouse_down_index is None:
                mouse_down_index = index
                selected_move = last_move
                selected_move_index = last_move_index
                button_down = True
                if selected_move is None:
                    protocol_errors.append("mouse_down has no preceding mouse move")
            continue
        if kind == "mouse_up":
            if not button_down:
                protocol_errors.append(f"mouse_up before mouse_down at action {index}")
            elif mouse_up_index is None:
                mouse_up_index = index
                button_down = False
            else:
                protocol_errors.append(f"duplicate mouse_up at action {index}")
            continue
        protocol_errors.append(f"unsupported action {kind!r} at action {index}")

    if legacy_qwen3 and selected_move is not None:
        execution_xy = model_to_execution_pixel(
            (float(selected_move["x"]), float(selected_move["y"])),
            image_size=execution_size,
        )
        last_move_hit = x1 <= execution_xy[0] <= x2 and y1 <= execution_xy[1] <= y2
    else:
        last_move_hit = bool(
            selected_move is not None
            and _physical_point_inside_bbox(
                (float(selected_move["x"]), float(selected_move["y"])),
                bbox=(x1, y1, x2, y2),
                resized_size=resized_size,
                execution_size=execution_size,
            )
        )
    down_seen = mouse_down_index is not None
    up_seen = mouse_up_index is not None
    down_followed_by_up = bool(
        down_seen and up_seen and mouse_up_index is not None and mouse_up_index > mouse_down_index
    )
    move_immediately_before_down = (
        last_move_index is not None
        and mouse_down_index is not None
        and last_move_index + 1 == mouse_down_index
    )
    primary_success = bool(
        last_move_hit
        and down_followed_by_up
        and not protocol_errors
    )
    return ActionSequenceScore(
        actions=normalized_actions,
        last_mouse_move_before_mouse_down=selected_move if down_seen else None,
        last_mouse_move_before_mouse_down_index=selected_move_index if down_seen else None,
        last_mouse_move_before_mouse_down_hit=last_move_hit if down_seen else False,
        mouse_down_seen=down_seen,
        mouse_up_seen=up_seen,
        mouse_down_followed_by_mouse_up=down_followed_by_up,
        last_mouse_move_immediately_before_mouse_down=move_immediately_before_down,
        protocol_errors=tuple(protocol_errors),
        primary_success=primary_success,
    )


def build_official_sequential_prediction_row(
    *,
    index: int,
    sample: dict[str, Any],
    instruction: str,
    raw_responses: Sequence[str],
    actions: Sequence[dict[str, Any]],
    observation_paths: Sequence[str | Path],
    terminal_reason: str | None,
    prompt_audits: Sequence[dict[str, Any]],
    image_size: tuple[int, int],
    parse_errors: Sequence[str | None] = (),
) -> dict[str, Any]:
    """Build one official-compatible row with sequential protocol details."""

    bbox = tuple(float(value) for value in sample["bbox"])
    score = score_action_sequence(actions=actions, bbox=bbox, image_size=image_size)
    selected = score.last_move_to_before_mouse_down
    point = (
        [float(selected["x"]) / 1000.0, float(selected["y"]) / 1000.0]
        if selected is not None
        else None
    )
    img_size = sample["img_size"]
    pred = [point[0] * float(img_size[0]), point[1] * float(img_size[1])] if point else None
    parse_error_values = [value for value in parse_errors if value]
    correctness = (
        "wrong_format"
        if parse_error_values
        else "correct"
        if score.primary_success
        else "wrong"
    )
    annotation_row = sample.get("_annotation_row", index)
    record_id = (
        f"screenspot-pro:{sample['task_filename']}.json:"
        f"{annotation_row}:{sample['img_filename']}"
    )
    return {
        "index": index,
        "id": sample["id"],
        "record_id": record_id,
        "img_filename": sample["img_filename"],
        "img_path": sample.get("_image_path"),
        "image_sha256": sample.get("_image_sha256"),
        "group": sample.get("group"),
        "platform": sample.get("platform"),
        "application": sample.get("application"),
        "lang": sample.get("language", "en"),
        "instruction_style": sample.get("instruction_style", "instruction"),
        "prompt_to_evaluate": instruction,
        "gt_type": sample.get("gt_type", "positive"),
        "ui_type": sample.get("ui_type"),
        "task_filename": sample["task_filename"],
        "bbox": sample["bbox"],
        "img_size": sample["img_size"],
        "point": point,
        "pred": pred,
        "raw_response": raw_responses[-1] if raw_responses else None,
        "raw_responses": list(raw_responses),
        "parse_error": parse_error_values[-1] if parse_error_values else None,
        "parse_errors": parse_error_values,
        "tool_call_count": len(raw_responses),
        "correctness": correctness,
        "action_history": [dict(action) for action in actions],
        "observation_paths": [str(Path(path)) for path in observation_paths],
        "prompt_audits": [dict(audit) for audit in prompt_audits],
        "terminal_reason": terminal_reason,
        "last_move_to_before_mouse_down": score.last_move_to_before_mouse_down,
        "last_move_to_before_mouse_down_index": score.last_move_to_before_mouse_down_index,
        "last_move_to_before_mouse_down_hit": score.last_move_to_before_mouse_down_hit,
        "last_move_to_immediately_before_mouse_down": score.last_move_to_immediately_before_mouse_down,
        "mouse_down_seen": score.mouse_down_seen,
        "mouse_up_seen": score.mouse_up_seen,
        "mouse_down_followed_by_mouse_up": score.mouse_down_followed_by_mouse_up,
        "protocol_errors": list(score.protocol_errors),
        "primary_success": score.primary_success,
    }


@dataclass(frozen=True)
class Qwen25SequentialLoopResult:
    responses: tuple[str, ...]
    actions: tuple[dict[str, Any], ...]
    observation_paths: tuple[Path, ...]
    renders: tuple[Qwen25SequentialRender, ...]
    score: ActionSequenceScore


def run_qwen25_sequential_loop(
    *,
    processor: Any,
    instruction: str,
    initial_image_path: Path,
    current_screen_size: tuple[int, int],
    execution_size: tuple[int, int],
    bbox: tuple[float, float, float, float],
    predict_fn: Callable[[Qwen25SequentialRender], str],
    observe_fn: Callable[[dict[str, Any], int], Path],
    max_turns: int,
) -> Qwen25SequentialLoopResult:
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    observations = [Path(initial_image_path)]
    responses: list[str] = []
    actions: list[dict[str, Any]] = []
    renders: list[Qwen25SequentialRender] = []

    for turn_index in range(max_turns):
        rendered = render_qwen25_sequential(
            processor,
            instruction=instruction,
            image_paths=tuple(observations),
            assistant_response_history=tuple(responses),
            current_screen_size=current_screen_size,
        )
        renders.append(rendered)
        response = str(predict_fn(rendered)).strip()
        action = parse_single_tool_call(response)
        if action.get("kind") not in {"mouse_move", "mouse_down", "mouse_up"}:
            raise ValueError("Qwen2.5 sequential loop received a non-Qwen2.5 action")
        responses.append(response)
        actions.append(action)
        if action["kind"] == "mouse_up":
            break
        observations.append(Path(observe_fn(action, turn_index)))

    score = score_action_sequence(
        actions=actions,
        bbox=bbox,
        resized_size=current_screen_size,
        execution_size=execution_size,
    )
    return Qwen25SequentialLoopResult(
        responses=tuple(responses),
        actions=tuple(actions),
        observation_paths=tuple(observations),
        renders=tuple(renders),
        score=score,
    )
