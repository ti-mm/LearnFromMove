from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeVar

from ..integrations.storage import checkpoint_root, storage_path
from ..prompts.groundcua_qwen25_sequential import (
    QWEN25_SEQUENTIAL_IMAGE_HISTORY_MAX,
    QWEN25_SEQUENTIAL_PROMPT_CONTRACT,
    render_qwen25_sequential,
)
from ..prompts.screenspot_pro_groundcua import (
    QWEN3_DIRECT_PROFILE,
    QWEN3_MOVE_PROFILE,
    QWEN3_MOVE_SEQUENTIAL_PROMPT_CONTRACT,
    QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE,
    QWEN25_DIRECT_PROFILE,
    QWEN25_MOUSEMOVE_LEFTCLICK_THINK_PROFILE,
    QWEN25_MOVE_COMPAT_ACTIONS,
    QWEN25_MOVE_PROFILE,
    GroundCUAProfile,
    build_groundcua_prompt,
    get_groundcua_profile,
    is_screenspot_pro_groundcua_profile,
    parse_groundcua_tool_call,
    qwen25_direct_tool_call,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    IMAGE_MAX_PIXELS as SCREENSPOT_PRO_QWEN3VL_IMAGE_MAX_PIXELS,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    IMAGE_MIN_PIXELS as SCREENSPOT_PRO_QWEN3VL_IMAGE_MIN_PIXELS,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    IMAGE_SIZE_FACTOR as SCREENSPOT_PRO_QWEN3VL_IMAGE_SIZE_FACTOR,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    PROMPT_PROFILE as SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    build_guided_prompt as build_screenspot_pro_qwen3vl_vllm_prompt,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    build_qwen3vl_official_messages,
)
from ..prompts.screenspot_pro_qwen3vl_vllm import (
    parse_complete_tool_call as parse_screenspot_pro_qwen3vl_tool_call,
)
from ..prompts.screenspot_qwen25_official import (
    QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE,
    QWEN25_MOUSE_PRIMITIVE_SYSTEM_PROMPT,
)
from ..prompts.unified_three_action import (
    format_unified_three_action_tool_call,
    parse_unified_three_action_response,
)
from ..protocol_tracks import (
    CANONICAL_ASSISTANT_FIELD,
    CANONICAL_THINK_FIELD,
    DEFAULT_MACRO_ACTION_KINDS,
    DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS,
    DEFAULT_PRIMITIVE_ACTION_KINDS,
    GROUNDCUA_AGENT_SYSTEM_PROMPT,
    OUTER_ROTATION_TASK_REQUIREMENT,
    PROTOCOL_VERSION,
    STRICT_MOUSE_PRIMITIVE_ACTION_KINDS,
    build_canonical_assistant_response,
    extract_assistant_response,
    extract_ten_choice_target_label,
    extract_think_text,
    is_groundcua_progressive_task,
    is_mouse_captcha_task,
    is_rotation_captcha_task,
    is_self_built_osworld_mouse_task,
    is_slot_drag_task,
    is_ten_choice_captcha_task,
    is_third_person_drag_task,
    resolve_protocol_track,
    split_think_json_response,
)

_DEFAULT_SFT_MANIFEST_DIRECTORY = (
    "interaction_rotation_synthetic_sft_20260511_v3_thinking_235b_repaired_20260513_coord1000"
)


def default_model_path() -> Path:
    """Return the supported Qwen3-VL base-model location in external storage."""

    return storage_path("models", "qwen3-vl-8b-instruct")


def default_train_manifest() -> Path:
    """Return the supported external training-manifest location."""

    return storage_path(
        "data",
        "sft",
        "manifests",
        _DEFAULT_SFT_MANIFEST_DIRECTORY,
        "train.jsonl",
    )


def default_eval_manifest() -> Path:
    """Return the supported external validation-manifest location."""

    return storage_path(
        "data",
        "sft",
        "manifests",
        _DEFAULT_SFT_MANIFEST_DIRECTORY,
        "val.jsonl",
    )


def default_checkpoint_output_dir() -> Path:
    """Return the external default namespace for generated SFT checkpoints."""

    return checkpoint_root() / "qwen3-vl-8b-sft"


def normalize_groundcua46k_trajectory(record: Mapping[str, Any], *, model_family: str):
    """Lazy compatibility export for the GroundCUA 46k window normalizer."""

    from ..data.groundcua46k_windows import normalize_groundcua46k_trajectory as _normalize

    return _normalize(record, model_family=model_family)


def build_groundcua46k_windows(trajectory: Any, *, model_family: str, images_to_keep: int = 3):
    """Lazy compatibility export for complete GroundCUA multi-turn windows."""

    from ..data.groundcua46k_windows import build_groundcua46k_windows as _build

    return _build(trajectory, model_family=model_family, images_to_keep=images_to_keep)


def format_groundcua_think_tool_call(
    profile_name: str,
    thought: str,
    action: str,
    coordinate: tuple[int, int] | None = None,
) -> str:
    """Lazy compatibility export for strict GroundCUA assistant targets."""

    from ..prompts.screenspot_pro_groundcua import (
        format_groundcua_think_tool_call as _format,
    )

    return _format(profile_name, thought, action, coordinate)


def parse_groundcua_think_tool_call(
    profile_name: str,
    response: str,
) -> tuple[str, str, tuple[int, int] | None]:
    """Lazy compatibility export for strict GroundCUA assistant targets."""

    from ..prompts.screenspot_pro_groundcua import (
        parse_groundcua_think_tool_call as _parse,
    )

    return _parse(profile_name, response)
HD720_IMAGE_MAX_PIXELS = 1280 * 720
FULL_HD_IMAGE_MAX_PIXELS = 1920 * 1080
QWEN_IMAGE_SIZE_FACTOR = 28
DEFAULT_IMAGE_MIN_PIXELS = HD720_IMAGE_MAX_PIXELS
PROMPT_FAMILY = "qwen3_vl_sft_closed_loop_context_v5_interleaved_history"
DEFAULT_PROMPT_PROFILE = PROMPT_FAMILY
FROZEN_FINAL_SFT_RUNTIME_SELF_HISTORY_CONTRACT = (
    "frozen_final_sft_runtime_self_history_v1"
)
FROZEN_FINAL_SFT_ROTATION_REQUIREMENT = (
    "Rotate the central circular region to align it with the background. Use only "
    "the screenshot to locate the slider and decide where to drag. Do not assume "
    "the slider is at a fixed position; it may appear anywhere on the page."
)
FROZEN_FINAL_SFT_OUTER_ROTATION_REQUIREMENT = (
    "Rotate the image outside the fixed central circular region to align it with "
    "the center. Use only the screenshot to locate the slider and decide where to "
    "drag. Do not assume the slider is at a fixed position; it may appear anywhere "
    "on the page."
)
FROZEN_FINAL_SFT_ROTATION_TASK_DESCRIPTION = (
    "Task description: This is a rotation CAPTCHA. The screen contains an image "
    "puzzle with a central rotated region and an on-screen control that changes "
    "the puzzle state. Complete the CAPTCHA according to the task requirement."
)
TASK_REQUIREMENT_POLICY = "single_task_requirement_v1"
COORDINATE_CONTRACT = "qwen3_relative_0_1000"
PIXEL_COORDINATE_FORMAT = "pixel"
NORMALIZED_COORDINATE_FORMAT = "normalized_0_1"
QWEN3_RELATIVE_COORDINATE_FORMAT = "qwen3_relative_0_1000"
SUPPORTED_SFT_ACTION_KINDS = DEFAULT_PRIMITIVE_ACTION_KINDS
SUPPORTED_SFT_MACRO_ACTION_KINDS = DEFAULT_MACRO_ACTION_KINDS
SUPPORTED_SFT_OUTPUT_ACTION_KINDS = SUPPORTED_SFT_ACTION_KINDS + SUPPORTED_SFT_MACRO_ACTION_KINDS
WEBSTAR_ACTION_MAPPING_ACTION_SPACE_TYPE = "webstar_action_mapping_v2"
MOUSE_CAPTCHA_ACTION_KINDS = DEFAULT_MOUSE_CAPTCHA_ACTION_KINDS

QWEN25_DIRECT_TRAINING_CONTRACT_FILENAME = (
    "screenspot_pro_qwen25_official_training_contract.json"
)
QWEN25_DIRECT_TRAINING_CONTRACT_SCHEMA = (
    "screenspot_pro_qwen25_official_leftclick_training_contract_v1"
)
QWEN25_DIRECT_ASSISTANT_CONTRACT = (
    "qwen25_official_full_tool_call_no_think"
)

DEFAULT_IMAGE_HISTORY_MAX = 3
SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX = 3
_HistoryItem = TypeVar("_HistoryItem")


def _resolve_image_history_max() -> int:
    raw = (
        os.environ.get("QWEN3_VL_IMAGE_HISTORY_MAX")
        or os.environ.get("IMAGE_HISTORY_MAX")
        or str(DEFAULT_IMAGE_HISTORY_MAX)
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"image history max must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"image history max must be >= 1, got {value}")
    return value


IMAGE_HISTORY_MAX = _resolve_image_history_max()
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def qwen25_direct_training_contract(output_dir: Path) -> dict[str, Any]:
    profile = get_groundcua_profile(QWEN25_DIRECT_PROFILE)
    return {
        "schema": QWEN25_DIRECT_TRAINING_CONTRACT_SCHEMA,
        "status": "passed",
        "output_dir": str(output_dir.resolve()),
        "prompt_profile": profile.name,
        "coordinate_contract": profile.coordinate_format,
        "assistant_contract": QWEN25_DIRECT_ASSISTANT_CONTRACT,
        "image_min_pixels": profile.image_min_pixels,
        "image_max_pixels": profile.image_max_pixels,
        "assistant_generation_boundary": "<|im_start|>assistant\n",
        "assistant_prefill": False,
    }


def write_qwen25_direct_training_contract(output_dir: Path) -> Path:
    receipt_path = output_dir / QWEN25_DIRECT_TRAINING_CONTRACT_FILENAME
    receipt_path.write_text(
        json.dumps(qwen25_direct_training_contract(output_dir), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return receipt_path


def qwen3_vl_multimodal_processor_kwargs(
    *,
    padding: bool = True,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> dict[str, Any]:
    """Build processor kwargs without multimodal pre-truncation.

    Qwen3-VL expands each image into many visual tokens. Truncating during
    processor tokenization can drop those visual tokens while leaving the
    corresponding image markers in text, which breaks 1920x1080 no-scale
    train/eval/infer consistency. Pad batches here and let the model consume
    the full multimodal prompt instead of pre-truncating it.
    """

    kwargs: dict[str, Any] = {"return_tensors": "pt", "truncation": False}
    if padding:
        kwargs["padding"] = True
    # Qwen VL processors route these top-level names to image_processor kwargs.
    # Supplying the same bounds used by the explicit PIL transform prevents a
    # second resize under the processor's own defaults.
    if image_min_pixels is not None:
        kwargs["min_pixels"] = image_min_pixels
    if image_max_pixels is not None:
        kwargs["max_pixels"] = image_max_pixels
    return kwargs


def append_image_history(
    image_history: list[Path],
    new_image: Path,
) -> list[Path]:
    """追加新图片，保留完整历史，后续在 prompt 构造阶段统一裁剪历史图片。"""
    image_history.append(new_image)
    return image_history


@dataclass(frozen=True)
class ImageCoordinateTransform:
    original_size: tuple[int, int]
    model_size: tuple[int, int]

    @property
    def scale_x(self) -> float:
        if self.original_size[0] <= 0:
            return 1.0
        return self.model_size[0] / float(self.original_size[0])

    @property
    def scale_y(self) -> float:
        if self.original_size[1] <= 0:
            return 1.0
        return self.model_size[1] / float(self.original_size[1])

    def to_model_xy(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale_x, y * self.scale_y

    def to_original_xy(self, x: float, y: float) -> tuple[float, float]:
        scale_x = self.scale_x
        scale_y = self.scale_y
        return (
            x / scale_x if scale_x else x,
            y / scale_y if scale_y else y,
        )


@dataclass(frozen=True)
class SftActionContext:
    action_position: int = 0
    total_actions: int | None = None
    cursor_xy: tuple[float, float] | None = None
    button_state: str = "up"
    task_type: str | None = None
    phase: str = "start"
    allowed_kinds: tuple[str, ...] = SUPPORTED_SFT_ACTION_KINDS
    task_action_kinds: tuple[str, ...] = SUPPORTED_SFT_ACTION_KINDS
    budget_remaining: int | None = None
    action_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    image_size_px: tuple[int, int] | None = None
    coordinate_format: str = PIXEL_COORDINATE_FORMAT
    action_paradigm: str = "primitive"
    protocol_track: str = "traditional_new_mixed"
    action_space_type: str = "mixed_primitive_macro"
    protocol_version: str = PROTOCOL_VERSION
    thought_required: bool = True


@dataclass(frozen=True)
class SftExample:
    instruction: str
    image_path: Path
    action: dict[str, Any]
    context: SftActionContext = field(default_factory=SftActionContext)
    assistant_response: str | None = None
    thought: str | None = None
    coordinate_format: str = PIXEL_COORDINATE_FORMAT
    image_paths: tuple[Path, ...] = ()  # 完整历史图片（含当前）；prompt 阶段再做 folding
    thought_history: tuple[str | None, ...] = ()  # 历史 action 的 thought seed，和 context.action_history 对齐
    assistant_response_history: tuple[str, ...] = ()


@dataclass(frozen=True)
class SftPromptBuildResult:
    messages: list[dict[str, Any]]
    image_paths: tuple[Path, ...]
    context: SftActionContext
    prompt_contract: str | None = None


def with_context_total_actions(
    example: SftExample,
    total_actions: int | None,
) -> SftExample:
    if total_actions is None:
        return example
    if total_actions < 1:
        raise ValueError("context total actions must be >= 1")
    context = replace(
        example.context,
        total_actions=total_actions,
        budget_remaining=max(0, total_actions - example.context.action_position),
    )
    return replace(example, context=context)


def with_context_total_actions_for_examples(
    examples: Iterable[SftExample],
    total_actions: int | None,
) -> list[SftExample]:
    return [with_context_total_actions(example, total_actions) for example in examples]


def deterministic_generation_kwargs(max_new_tokens: int) -> dict[str, Any]:
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _is_long_thinking_text(text: str) -> bool:
    normalized = text.strip()
    if len(normalized) < 60:
        return False
    markers = (
        "观察",
        "分析",
        "判断",
        "下一步",
        "observe",
        "analyze",
        "judge",
        "finally",
    )
    return sum(1 for marker in markers if marker in normalized.lower()) >= 2


def _seed_thought_clause(thought: str | None) -> str:
    if not isinstance(thought, str):
        return ""
    normalized = thought.strip()
    if not normalized:
        return ""
    return f"当前已有的直接线索是：{normalized}。"


def _is_captcha_context(context: SftActionContext | None) -> bool:
    if context is None:
        return False
    return (
        context.protocol_track.startswith("captcha")
        or is_ten_choice_captcha_task(context.task_type)
        or is_rotation_captcha_task(context.task_type)
    )


def _is_rotation_captcha_context(
    action: dict[str, Any],
    context: SftActionContext | None,
) -> bool:
    if context is None:
        return action.get("kind") in {"move_to", "mouse_down", "mouse_up", "drag"}
    if is_rotation_captcha_task(context.task_type):
        return True
    return context.protocol_track == "captcha_old_legacy" and action.get("kind") == "drag"


def _rotation_captcha_long_thought(action: dict[str, Any], seed_thought: str | None) -> str:
    kind = str(action.get("kind") or "").strip()
    seed_clause = _seed_thought_clause(seed_thought)
    if kind == "mouse_down":
        return (
            "先观察截图里滑块手柄、中心圆形边界、背景纹理连续性和当前光标位置，确认这一步是否已经对准可按下的位置。"
            f"{seed_clause}"
            "再分析当前还没有开始拖动，按下鼠标后才能在连续截图里继续观察旋转变化。"
            "然后判断现在按下是否能稳定进入拖动闭环，而不会在错误位置白白消耗一次提交机会。"
            "最后决定执行 mouse_down，先按住滑块，为后续按对齐程度继续 move_to 做准备。"
        )
    if kind == "mouse_up":
        return (
            "先观察截图里圆形区域与背景交界处的线条、阴影和纹理是否已经基本连续，同时确认当前滑块位置和拖动轨迹。"
            f"{seed_clause}"
            "再分析如果继续拖动，结果更可能过冲而不是继续改善，或者当前已经足够接近正确对齐。"
            "然后判断现在释放是否比继续试探更稳妥，避免把已经接近连续的边界再次拉偏。"
            "最后决定执行 mouse_up，在当前对齐程度下提交这一步。"
        )
    return (
        "先观察截图里中心圆形边界、背景纹理、阴影走向、滑块位置和当前光标位置，找出哪里仍然不连续。"
        f"{seed_clause}"
        "再分析当前旋转更像是还没有转够，还是已经略微过冲，并结合只能输出一个合法 move_to 的约束来确定下一次试探。"
        "然后判断把目标位置调到更接近连续边界的 x 坐标，是否能更清楚地区分欠冲和过冲，并为下一步释放创造更好的窗口。"
        "最后决定执行 move_to，把滑块移动到更接近对齐的位置，继续观察新的对齐程度。"
    )


def _captcha_click_long_thought(action: dict[str, Any], seed_thought: str | None) -> str:
    seed_clause = _seed_thought_clause(seed_thought)
    return (
        "先观察截图里的题目要求、候选区域、目标外观差异和当前光标位置，确认真正需要命中的视觉目标。"
        f"{seed_clause}"
        "再分析哪些候选和任务描述最一致，哪些只是相似但不满足当前题意。"
        "然后判断现在直接点击目标是否已经比继续移动或等待更合理，并确认这一击应当落在最稳定的命中位置。"
        "最后决定执行 click，把动作落在当前最可信的目标上。"
    )


def _traditional_long_thought(action: dict[str, Any], seed_thought: str | None) -> str:
    kind = str(action.get("kind") or "").strip()
    seed_clause = _seed_thought_clause(seed_thought)
    defaults = {
        "move_to": "最后决定先执行 move_to，把光标移动到更合适的位置，为下一步交互创造条件。",
        "mouse_down": "最后决定执行 mouse_down，在当前位置开始这次操作。",
        "mouse_up": "最后决定执行 mouse_up，在当前位置结束这次拖动或按压步骤。",
        "click": "最后决定执行 click，直接触发当前最明确的目标控件。",
        "drag": "最后决定执行 drag，把对象从当前起点拖到更合理的目标位置。",
        "type_text": "最后决定执行 type_text，在已经就绪的输入焦点里填入所需文本。",
        "paste_text": "最后决定执行 paste_text，直接插入当前需要的内容。",
        "key_press": "最后决定执行 key_press，用单次按键推进当前流程。",
        "type": "最后决定执行 type，在已经就绪的输入焦点里填入所需文本。",
        "hotkey": "最后决定执行 hotkey，用组合键推进当前流程。",
        "scroll": "最后决定执行 scroll，沿合适方向滚动以揭示或定位缺失内容。",
        "left_double": "最后决定执行 left_double，在目标位置进行双击。",
        "right_single": "最后决定执行 right_single，在目标位置进行右键点击。",
        "browser_back": "最后决定执行 browser_back，回到上一页状态继续任务。",
        "wait": "最后决定执行 wait，先保持当前状态并等待界面变化。",
        "submit": "最后决定执行 submit，在当前状态下提交结果。",
        "finished": "最后决定执行 finished，给出最终答案并结束任务。",
        "done": "最后决定执行 done，在当前状态下结束这一步。",
    }
    return (
        "先观察当前界面中的目标控件、文本提示、输入焦点、已完成步骤和当前光标位置，确认这一步真正受哪些可见状态约束。"
        f"{seed_clause}"
        "再分析当前任务距离完成还缺什么，以及有哪些动作虽然合法但会偏离目标或破坏当前状态。"
        "然后判断候选动作里哪一个最能直接推进任务，同时风险最小、与当前界面状态最一致。"
        + defaults.get(kind, "最后决定执行下一步合法动作，继续把任务往前推进。")
    )


def _default_react_thought(
    action: dict[str, Any],
    context: SftActionContext | None,
    thought: str | None = None,
) -> str:
    kind = str(action.get("kind") or "").strip()
    if _is_rotation_captcha_context(action, context):
        return _rotation_captcha_long_thought(action, thought)
    if _is_captcha_context(context) and kind == "click":
        return _captcha_click_long_thought(action, thought)
    return _traditional_long_thought(action, thought)


def _react_thought_text(
    thought: str | None,
    action: dict[str, Any],
    context: SftActionContext | None = None,
) -> str:
    if isinstance(thought, str) and thought.strip() and _is_long_thinking_text(thought):
        return thought.strip()
    return _default_react_thought(action, context, thought)


def _react_action_text(
    *,
    action: dict[str, Any],
    thought: str | None,
    context: SftActionContext | None = None,
) -> str:
    return build_canonical_assistant_response(
        thought=_react_thought_text(thought, action, context),
        action=action,
    )


def fold_image_history_items(
    items: tuple[_HistoryItem, ...],
    *,
    images_to_keep: int,
    images_to_drop: int | None = None,
) -> tuple[_HistoryItem, ...]:
    """Retain only the most recent image-history items."""

    del images_to_drop
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be >= 1")
    if len(items) <= images_to_keep:
        return items
    return items[-images_to_keep:]


def fold_image_history_paths(
    image_paths: tuple[Path, ...],
    *,
    images_to_keep: int,
    images_to_drop: int | None = None,
) -> tuple[Path, ...]:
    return fold_image_history_items(
        image_paths,
        images_to_keep=images_to_keep,
        images_to_drop=images_to_drop,
    )


def retain_recent_image_paths(image_paths: tuple[Path, ...], *, images_to_keep: int) -> tuple[Path, ...]:
    return fold_image_history_paths(image_paths, images_to_keep=images_to_keep)


def build_sft_multimodal_user_content(
    *,
    image_path: Path,
    image_paths: tuple[Path, ...] = (),
    action_history: tuple[dict[str, Any], ...] = (),
    thought_history: tuple[str | None, ...] = (),
    assistant_response_history: tuple[str, ...] = (),
    context: SftActionContext | None = None,
    images_to_keep: int = IMAGE_HISTORY_MAX,
) -> tuple[list[dict[str, str]], tuple[Path, ...]]:
    history_paths = image_paths or (image_path,)
    if images_to_keep < 1:
        raise ValueError("images_to_keep must be >= 1")

    retained_paths = fold_image_history_paths(history_paths, images_to_keep=images_to_keep)
    first_retained_image_index = max(0, len(history_paths) - len(retained_paths))
    first_image_backed_action_index = first_retained_image_index
    exact_response_history = _validate_assistant_response_history(
        action_history=action_history,
        assistant_response_history=assistant_response_history,
        context=context,
    )
    thought_offset = len(action_history) - len(thought_history)

    def thought_for_action_index(action_index: int) -> str | None:
        thought_index = action_index - thought_offset
        if 0 <= thought_index < len(thought_history):
            return thought_history[thought_index]
        return None

    def previous_response_content(action_index: int) -> dict[str, str]:
        if exact_response_history:
            response = exact_response_history[action_index]
        else:
            action = action_history[action_index]
            thought = thought_for_action_index(action_index)
            response = _history_action_text(action=action, thought=thought, context=context)
        return {
            "type": "text",
            "text": (
                f"\nPrevious step {action_index + 1} assistant response (context only):\n"
                f"{response}\n"
            ),
        }

    content: list[dict[str, str]] = []
    for action_index in range(min(first_image_backed_action_index, len(action_history))):
        content.append(previous_response_content(action_index))

    for index, path in enumerate(retained_paths):
        content.append({"type": "image", "image": str(path)})
        action_index = first_retained_image_index + index
        if action_index >= len(action_history):
            continue
        content.append(previous_response_content(action_index))
    return content, retained_paths


def _aligned_dimension(value: float, *, factor: int, mode: str) -> int:
    if factor < 1:
        raise ValueError("image size factor must be positive")
    if mode == "floor":
        units = int(value // factor)
    elif mode == "ceil":
        units = int(-(-value // factor))
    elif mode == "round":
        units = int(round(value / factor))
    else:
        raise ValueError(f"unsupported alignment mode: {mode}")
    return max(factor, units * factor)


def resized_size_for_pixel_range(
    size: tuple[int, int],
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
    size_factor: int = QWEN_IMAGE_SIZE_FACTOR,
) -> tuple[int, int]:
    """Apply Qwen-VL's smart-resize contract without a second processor resize."""
    width, height = (int(size[0]), int(size[1]))
    if width < 1 or height < 1:
        raise ValueError(f"image dimensions must be positive, got {size}")
    if image_min_pixels is not None and image_min_pixels < 1:
        raise ValueError("image_min_pixels must be positive")
    if image_max_pixels is not None and image_max_pixels < 1:
        raise ValueError("image_max_pixels must be positive")
    if (
        image_min_pixels is not None
        and image_max_pixels is not None
        and image_min_pixels > image_max_pixels
    ):
        raise ValueError("image_min_pixels must not exceed image_max_pixels")
    if max(width, height) / min(width, height) > 200:
        raise ValueError("image aspect ratio exceeds Qwen-VL's limit")

    area = width * height
    rounded_width = _aligned_dimension(width, factor=size_factor, mode="round")
    rounded_height = _aligned_dimension(height, factor=size_factor, mode="round")
    if (
        (image_min_pixels is None or image_min_pixels <= rounded_width * rounded_height)
        and (image_max_pixels is None or rounded_width * rounded_height <= image_max_pixels)
    ):
        return rounded_width, rounded_height

    if image_max_pixels is not None and rounded_width * rounded_height > image_max_pixels:
        beta = (area / float(image_max_pixels)) ** 0.5
        model_width = _aligned_dimension(width / beta, factor=size_factor, mode="floor")
        model_height = _aligned_dimension(height / beta, factor=size_factor, mode="floor")
    elif image_min_pixels is not None and rounded_width * rounded_height < image_min_pixels:
        beta = (image_min_pixels / float(area)) ** 0.5
        model_width = _aligned_dimension(width * beta, factor=size_factor, mode="ceil")
        model_height = _aligned_dimension(height * beta, factor=size_factor, mode="ceil")
    else:
        model_width, model_height = rounded_width, rounded_height

    if image_max_pixels is not None:
        while model_width * model_height > image_max_pixels:
            if model_width >= model_height and model_width > size_factor:
                model_width -= size_factor
            elif model_height > size_factor:
                model_height -= size_factor
            else:
                break
    if image_min_pixels is not None and model_width * model_height < image_min_pixels:
        if image_max_pixels is not None and model_width * model_height == 0:
            raise ValueError("cannot satisfy configured image pixel range")
        while model_width * model_height < image_min_pixels:
            if model_width / width <= model_height / height:
                model_width += size_factor
            else:
                model_height += size_factor
            if image_max_pixels is not None and model_width * model_height > image_max_pixels:
                raise ValueError("no Qwen-aligned size fits configured image pixel range")
    return model_width, model_height


def resize_image_for_pixel_range(
    image: Any,
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
    size_factor: int = QWEN_IMAGE_SIZE_FACTOR,
) -> tuple[Any, ImageCoordinateTransform]:
    original_size = (int(image.width), int(image.height))
    model_size = resized_size_for_pixel_range(
        original_size,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        size_factor=size_factor,
    )
    if model_size != original_size:
        image = image.resize(model_size)
    return image, ImageCoordinateTransform(
        original_size=original_size,
        model_size=model_size,
    )


@lru_cache(maxsize=1)
def _load_official_smart_resize() -> Any:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

    return smart_resize


def resize_image_for_qwen3vl_official(
    image: Any,
    *,
    image_min_pixels: int,
    image_max_pixels: int,
) -> tuple[Any, ImageCoordinateTransform]:
    """Apply the exact ``smart_resize`` used by the official Qwen3-VL eval."""

    original_size = (int(image.width), int(image.height))
    resized_height, resized_width = _load_official_smart_resize()(
        original_size[1],
        original_size[0],
        factor=32,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    model_size = (int(resized_width), int(resized_height))
    if model_size != original_size:
        image = image.resize(model_size)
    return image, ImageCoordinateTransform(
        original_size=original_size,
        model_size=model_size,
    )


def resize_image_for_qwen25vl_official(
    image: Any,
    *,
    image_min_pixels: int,
    image_max_pixels: int,
) -> tuple[Any, ImageCoordinateTransform]:
    """Apply the exact factor-28 smart_resize used by official Qwen2.5-VL."""

    original_size = (int(image.width), int(image.height))
    resized_height, resized_width = _load_official_smart_resize()(
        original_size[1],
        original_size[0],
        factor=28,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    model_size = (int(resized_width), int(resized_height))
    if model_size != original_size:
        image = image.resize(model_size)
    return image, ImageCoordinateTransform(
        original_size=original_size,
        model_size=model_size,
    )


def resized_size_for_max_pixels(
    size: tuple[int, int],
    image_max_pixels: int | None,
) -> tuple[int, int]:
    width, height = size
    if image_max_pixels is None or width * height <= image_max_pixels:
        return width, height
    scale = (image_max_pixels / float(width * height)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def resize_image_for_max_pixels(image: Any, image_max_pixels: int | None) -> tuple[Any, ImageCoordinateTransform]:
    original_size = (int(image.width), int(image.height))
    model_size = resized_size_for_max_pixels(original_size, image_max_pixels)
    if model_size != original_size:
        image = image.resize(model_size)
    return image, ImageCoordinateTransform(original_size=original_size, model_size=model_size)


def resized_size_for_prompt_profile(
    size: tuple[int, int],
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
    prompt_profile: str,
) -> tuple[int, int]:
    if is_screenspot_pro_groundcua_profile(prompt_profile):
        factor = get_groundcua_profile(prompt_profile).image_factor
    else:
        factor = (
            SCREENSPOT_PRO_QWEN3VL_IMAGE_SIZE_FACTOR
            if prompt_profile == SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE
            else QWEN_IMAGE_SIZE_FACTOR
        )
    return resized_size_for_pixel_range(
        size,
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        size_factor=factor,
    )


def _numeric_xy(values: Any) -> tuple[float, float] | None:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    x_value, y_value = values[0], values[1]
    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
        return None
    return float(x_value), float(y_value)


def _clean_action_for_history(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    schema_keys_by_kind = {
        "move_to": ("kind", "x", "y"),
        "mouse_move": ("kind", "x", "y"),
        "mouse_down": ("kind",),
        "mouse_up": ("kind",),
        "left_click": ("kind",),
        "click": ("kind", "points"),
        "drag": ("kind", "points"),
        "type_text": ("kind", "text"),
        "paste_text": ("kind", "text"),
        "type": ("kind", "content", "text"),
        "key_press": ("kind", "key"),
        "hotkey": ("kind", "key"),
        "scroll": ("kind", "direction"),
        "left_double": ("kind", "points"),
        "right_single": ("kind", "points"),
        "wait": ("kind", "duration"),
        "submit": ("kind",),
        "browser_back": ("kind",),
        "done": ("kind", "answer"),
        "finished": ("kind", "content", "answer"),
    }
    allowed_keys = schema_keys_by_kind.get(str(kind))
    if allowed_keys is not None:
        return {
            key: action[key]
            for key in allowed_keys
            if key in action and not (key in {"x", "y"} and action[key] is None)
        }

    provenance_keys = {
        "teacher_think_metadata",
        "metadata",
        "source",
        "model",
        "base_url",
        "cache_hit",
        "finish_reason",
        "attempt",
        "enable_thinking",
        "thinking_budget",
    }
    return {
        key: value
        for key, value in action.items()
        if key not in provenance_keys and not (key in {"x", "y"} and value is None)
    }


def _history_action_text(
    *,
    action: dict[str, Any],
    thought: str | None,
    context: SftActionContext | None = None,
) -> str:
    cleaned_action = _clean_action_for_history(action)
    if cleaned_action.get("kind") in {"move_to", "mouse_down", "mouse_up"}:
        return format_unified_three_action_tool_call(cleaned_action)
    if isinstance(thought, str) and thought.strip():
        return build_canonical_assistant_response(
            thought=thought.strip(),
            action=cleaned_action,
        )
    return _react_action_text(action=cleaned_action, thought=thought, context=context)


def _validate_assistant_response_history(
    *,
    action_history: tuple[dict[str, Any], ...],
    assistant_response_history: tuple[str, ...],
    context: SftActionContext | None = None,
) -> tuple[str, ...]:
    """Validate exact runtime responses without rewriting any response text."""

    if not assistant_response_history:
        return ()
    if len(assistant_response_history) != len(action_history):
        raise ValueError(
            "assistant_response_history requires one exact assistant response per action"
        )

    if context is not None and context.protocol_track == "groundcua_static_progressive":
        for index, (response, action) in enumerate(
            zip(assistant_response_history, action_history, strict=True)
        ):
            parsed_action = _parse_groundcua_assistant_response(
                response,
                context=f"assistant_response_history[{index}]",
            )
            if parsed_action != _clean_action_for_history(action):
                raise ValueError(
                    f"assistant_response_history[{index}] action does not match action_history[{index}]"
                )
        return assistant_response_history

    if context is not None and is_slot_drag_task(context.task_type):
        from gui_agent_captcha.domains.slot_drag.action_contract import (
            canonicalize_slot_drag_data_action as canonicalize_data_action,
        )
        from gui_agent_captcha.domains.slot_drag.action_contract import (
            parse_slot_drag_response as parse_response,
        )
        from gui_agent_captcha.domains.slot_drag.action_contract import (
            slot_drag_actions_equal as actions_equal,
        )

        response_contract_name = "SlotDrag"
    else:
        from gui_agent_captcha.domains.rotation.action_contract import (
            actions_equal,
            canonicalize_data_action,
        )
        from gui_agent_captcha.domains.rotation.action_contract import (
            parse_rotation_response as parse_response,
        )

        response_contract_name = "rotation"
    from gui_agent_captcha.domains.rotation.action_contract import ResponseFormatError

    for index, (response, action) in enumerate(
        zip(assistant_response_history, action_history, strict=True)
    ):
        try:
            parsed = parse_response(response)
        except ResponseFormatError as exc:
            raise ValueError(
                f"assistant_response_history[{index}] must be a valid "
                f"{response_contract_name} response: {exc}"
            ) from exc
        try:
            expected = canonicalize_data_action(action)
        except ValueError as exc:
            raise ValueError(f"action_history[{index}] is invalid: {exc}") from exc
        if not actions_equal(parsed.action, expected):
            raise ValueError(
                f"assistant_response_history[{index}] action does not match action_history[{index}]"
            )
    return assistant_response_history


def _normalized_to_qwen3_relative_coord(value: float) -> int:
    return min(1000, max(0, int(round(float(value) * 1000))))


def _convert_normalized_action_to_qwen3_relative(action: dict[str, Any]) -> dict[str, Any]:
    kind = action.get("kind")
    cleaned = {"kind": kind}
    if kind in {"click", "drag"}:
        points = action.get("points")
        if isinstance(points, list):
            converted_points = []
            for point in points:
                xy = _numeric_xy(point)
                if xy is None:
                    return dict(action)
                converted_points.append([
                    _normalized_to_qwen3_relative_coord(xy[0]),
                    _normalized_to_qwen3_relative_coord(xy[1]),
                ])
            cleaned["points"] = converted_points
        else:
            return dict(action)
    elif kind == "move_to":
        x_value = action.get("x")
        y_value = action.get("y")
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            cleaned["x"] = _normalized_to_qwen3_relative_coord(float(x_value))
            cleaned["y"] = _normalized_to_qwen3_relative_coord(float(y_value))
        else:
            cleaned.update({k: v for k, v in action.items() if k != "kind"})
    elif kind == "type_text":
        if "text" in action:
            cleaned["text"] = action["text"]
    elif kind == "wait":
        if "duration" in action:
            cleaned["duration"] = action["duration"]
    elif kind == "done" and "answer" in action:
        cleaned["answer"] = action["answer"]
    return cleaned


def _convert_normalized_xy_to_qwen3_relative(
    xy: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if xy is None:
        return None
    return (
        float(_normalized_to_qwen3_relative_coord(xy[0])),
        float(_normalized_to_qwen3_relative_coord(xy[1])),
    )


def _pixel_xy_to_qwen3_relative(
    xy: tuple[float, float] | None,
    image_size: tuple[int, int],
) -> tuple[float, float] | None:
    if xy is None:
        return None
    width, height = image_size
    if width <= 0 or height <= 0:
        return xy
    return (
        float(_normalized_to_qwen3_relative_coord(xy[0] / width)),
        float(_normalized_to_qwen3_relative_coord(xy[1] / height)),
    )


def _pixel_action_to_qwen3_relative(
    action: dict[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    cleaned = _clean_action_for_history(action)
    if cleaned.get("kind") in {"click", "drag"}:
        points = cleaned.get("points")
        if not isinstance(points, list):
            return cleaned
        converted_points = []
        for point in points:
            xy = _numeric_xy(point)
            if xy is None:
                return cleaned
            converted = _pixel_xy_to_qwen3_relative(xy, image_size)
            if converted is None:
                return cleaned
            converted_points.append([int(converted[0]), int(converted[1])])
        return {**cleaned, "points": converted_points}
    if cleaned.get("kind") != "move_to":
        return cleaned
    xy = _numeric_xy((cleaned.get("x"), cleaned.get("y")))
    converted = _pixel_xy_to_qwen3_relative(xy, image_size)
    if converted is None:
        return cleaned
    qx, qy = converted
    return {"kind": "move_to", "x": int(qx), "y": int(qy)}


def _finalize_action_for_coordinate_format(
    action: dict[str, Any],
    coordinate_format: str,
) -> dict[str, Any]:
    if coordinate_format == NORMALIZED_COORDINATE_FORMAT:
        return _convert_normalized_action_to_qwen3_relative(action)
    return _clean_action_for_history(action)


def _finalize_xy_for_coordinate_format(
    xy: tuple[float, float] | None,
    coordinate_format: str,
) -> tuple[float, float] | None:
    if coordinate_format == NORMALIZED_COORDINATE_FORMAT:
        return _convert_normalized_xy_to_qwen3_relative(xy)
    return xy


def _model_coordinate_format(coordinate_format: str) -> str:
    if coordinate_format == NORMALIZED_COORDINATE_FORMAT:
        return QWEN3_RELATIVE_COORDINATE_FORMAT
    return coordinate_format


def scale_action_coordinates(
    action: dict[str, Any],
    transform: ImageCoordinateTransform,
    *,
    to: str = "model",
) -> dict[str, Any]:
    if action.get("kind") in {"click", "drag"}:
        points = action.get("points")
        if not isinstance(points, list):
            return dict(action)
        scaled_points = []
        for point in points:
            xy = _numeric_xy(point)
            if xy is None:
                return dict(action)
            if to == "model":
                scaled_points.append(list(transform.to_model_xy(xy[0], xy[1])))
            elif to == "original":
                scaled_points.append(list(transform.to_original_xy(xy[0], xy[1])))
            else:
                raise ValueError(f"Unsupported coordinate transform direction: {to}")
        return {**action, "points": scaled_points}
    if action.get("kind") != "move_to":
        return dict(action)
    x_value = action.get("x")
    y_value = action.get("y")
    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
        return dict(action)
    if to == "model":
        x_scaled, y_scaled = transform.to_model_xy(float(x_value), float(y_value))
    elif to == "original":
        x_scaled, y_scaled = transform.to_original_xy(float(x_value), float(y_value))
    else:
        raise ValueError(f"Unsupported coordinate transform direction: {to}")
    scaled = dict(action)
    scaled["x"] = x_scaled
    scaled["y"] = y_scaled
    return scaled


def _scale_xy(
    xy: tuple[float, float] | None,
    transform: ImageCoordinateTransform,
    *,
    to: str,
) -> tuple[float, float] | None:
    if xy is None:
        return None
    if to == "model":
        return transform.to_model_xy(xy[0], xy[1])
    if to == "original":
        return transform.to_original_xy(xy[0], xy[1])
    raise ValueError(f"Unsupported coordinate transform direction: {to}")


def scale_action_context(
    context: SftActionContext,
    transform: ImageCoordinateTransform,
    *,
    to: str = "model",
) -> SftActionContext:
    image_size = transform.model_size if to == "model" else transform.original_size
    return replace(
        context,
        cursor_xy=_scale_xy(context.cursor_xy, transform, to=to),
        action_history=tuple(
            scale_action_coordinates(action, transform, to=to)
            for action in context.action_history
        ),
        image_size_px=image_size,
    )


def convert_pixel_example_to_qwen3_relative(
    example: SftExample,
    image_size: tuple[int, int],
) -> SftExample:
    context = example.context
    qwen_context = replace(
        context,
        cursor_xy=_pixel_xy_to_qwen3_relative(context.cursor_xy, image_size),
        action_history=tuple(
            _pixel_action_to_qwen3_relative(action, image_size)
            for action in context.action_history
        ),
        image_size_px=image_size,
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
    )
    return replace(
        example,
        action=_pixel_action_to_qwen3_relative(example.action, image_size),
        context=qwen_context,
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
    )


def convert_pixel_context_to_qwen3_relative(
    context: SftActionContext,
    image_size: tuple[int, int],
) -> SftActionContext:
    return replace(
        context,
        cursor_xy=_pixel_xy_to_qwen3_relative(context.cursor_xy, image_size),
        action_history=tuple(
            _pixel_action_to_qwen3_relative(action, image_size)
            for action in context.action_history
        ),
        image_size_px=image_size,
        coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
    )


def transform_sft_example_for_model_image(
    example: SftExample,
    transform: ImageCoordinateTransform,
) -> SftExample:
    if example.coordinate_format in {NORMALIZED_COORDINATE_FORMAT, QWEN3_RELATIVE_COORDINATE_FORMAT}:
        return replace(
            example,
            context=replace(example.context, image_size_px=transform.model_size),
        )
    return convert_pixel_example_to_qwen3_relative(example, transform.original_size)


def _format_xy(xy: tuple[float, float]) -> str:
    return f"({xy[0]:.1f}, {xy[1]:.1f})"


def _format_context_xy(xy: tuple[float, float], coordinate_format: str) -> str:
    if coordinate_format == QWEN3_RELATIVE_COORDINATE_FORMAT:
        return f"({int(round(xy[0]))}, {int(round(xy[1]))})"
    return _format_xy(xy)


def _phase_from_state(
    history: tuple[dict[str, Any], ...],
    *,
    button_state: str,
    task_type: str | None,
) -> str:
    if not history:
        return "start"
    if button_state == "down":
        return "button_down"
    last_kind = history[-1].get("kind")
    if last_kind == "move_to":
        return "cursor_positioned"
    if last_kind == "mouse_up":
        if task_type == "TYPE":
            return "input_focused_or_released"
        return "released"
    if last_kind == "type_text":
        return "text_entered"
    if last_kind == "wait":
        return "waiting"
    return "in_progress"


def _allowed_kinds_from_state(
    history: tuple[dict[str, Any], ...],
    *,
    button_state: str,
    task_type: str | None,
) -> tuple[str, ...]:
    mouse_captcha_task = _is_mouse_captcha_task(task_type)
    ten_choice_task = _is_ten_choice_captcha_task(task_type)
    if not history:
        return ("move_to",)
    if button_state == "down":
        if ten_choice_task:
            return ("mouse_up",)
        if mouse_captcha_task:
            return ("move_to", "mouse_up")
        return ("move_to", "mouse_up", "wait")
    last_kind = history[-1].get("kind")
    if last_kind == "move_to":
        if mouse_captcha_task:
            return ("mouse_down", "move_to")
        return ("mouse_down", "move_to", "done")
    if last_kind == "mouse_up":
        if mouse_captcha_task:
            return ("move_to",)
        if task_type == "TYPE":
            return ("type_text", "move_to", "done")
        return ("move_to", "done")
    if last_kind == "type_text":
        return ("move_to", "done")
    if last_kind == "wait":
        if mouse_captcha_task:
            return ("mouse_up", "move_to")
        return ("mouse_up", "wait", "move_to")
    if mouse_captcha_task:
        return MOUSE_CAPTCHA_ACTION_KINDS
    return SUPPORTED_SFT_ACTION_KINDS


def _task_action_kinds_for_task(task_type: str | None) -> tuple[str, ...]:
    return resolve_protocol_track(task_type=task_type).task_action_kinds


def _macro_task_action_kinds(action_kind: str) -> tuple[str, ...]:
    if action_kind == "click":
        return ("click",)
    if action_kind == "drag":
        return ("drag",)
    if action_kind == "submit":
        return ("submit",)
    return SUPPORTED_SFT_MACRO_ACTION_KINDS


def _is_ten_choice_captcha_task(task_type: str | None) -> bool:
    return is_ten_choice_captcha_task(task_type)


def _is_mouse_captcha_task(task_type: str | None) -> bool:
    return is_mouse_captcha_task(task_type)


def _is_groundcua_progressive_task(task_type: str | None) -> bool:
    return is_groundcua_progressive_task(task_type)


def _is_qwen25_mouse_primitive_prompt_profile(prompt_profile: str | None) -> bool:
    return prompt_profile == QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE


def _system_prompt_for_profile(prompt_profile: str | None) -> str:
    if _is_qwen25_mouse_primitive_prompt_profile(prompt_profile):
        return QWEN25_MOUSE_PRIMITIVE_SYSTEM_PROMPT
    return GROUNDCUA_AGENT_SYSTEM_PROMPT


def _context_from_state(
    *,
    action_history: list[dict[str, Any]],
    total_actions: int | None,
    cursor_xy: tuple[float, float] | None,
    button_state: str,
    task_type: str | None,
    allowed_kinds: Iterable[str] | None = None,
    budget_remaining: int | None = None,
) -> SftActionContext:
    history = tuple(_clean_action_for_history(action) for action in action_history)
    track = resolve_protocol_track(task_type=task_type, action_kinds=allowed_kinds)
    explicit_allowed = tuple(kind for kind in (allowed_kinds or ()) if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS)
    if _is_groundcua_progressive_task(task_type):
        task_action_kinds = track.task_action_kinds
    elif explicit_allowed:
        task_action_kinds = explicit_allowed
    else:
        task_action_kinds = tuple(
            kind for kind in track.task_action_kinds if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS
        )
    derived_allowed = _allowed_kinds_from_state(
        history,
        button_state=button_state,
        task_type=task_type,
    )
    if track.action_paradigm == "macro_only":
        allowed = tuple(kind for kind in task_action_kinds if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS)
    elif track.action_paradigm == "mixed":
        allowed = tuple(kind for kind in task_action_kinds if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS)
    elif task_action_kinds is not None:
        allowed_set = set(task_action_kinds)
        filtered = tuple(kind for kind in derived_allowed if kind in allowed_set)
        allowed = filtered or tuple(kind for kind in task_action_kinds if kind in SUPPORTED_SFT_ACTION_KINDS)
    else:
        allowed = derived_allowed
    return SftActionContext(
        action_position=len(history),
        total_actions=total_actions,
        cursor_xy=cursor_xy,
        button_state=button_state,
        task_type=task_type,
        phase=_phase_from_state(history, button_state=button_state, task_type=task_type),
        allowed_kinds=allowed,
        task_action_kinds=tuple(kind for kind in task_action_kinds if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS),
        budget_remaining=budget_remaining,
        action_history=history,
        action_paradigm=track.action_paradigm,
        protocol_track=track.track_id,
        action_space_type=track.action_space_type,
        protocol_version=track.protocol_version,
        thought_required=track.thought_required,
    )


def build_action_context_from_history(
    *,
    action_history: Iterable[dict[str, Any]],
    total_actions: int | None,
    cursor_xy: tuple[float, float] | None,
    button_state: str,
    task_type: str | None = None,
    allowed_kinds: Iterable[str] | None = None,
    budget_remaining: int | None = None,
) -> SftActionContext:
    return _context_from_state(
        action_history=list(action_history),
        total_actions=total_actions,
        cursor_xy=cursor_xy,
        button_state=button_state,
        task_type=task_type,
        allowed_kinds=allowed_kinds,
        budget_remaining=budget_remaining,
    )


def _update_button_state(action: dict[str, Any], current: str) -> str:
    kind = action.get("kind")
    if kind == "mouse_down":
        return "down"
    if kind == "mouse_up":
        return "up"
    return current


def _update_cursor_from_action(
    action: dict[str, Any],
    current: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if action.get("kind") != "move_to":
        points = action.get("points")
        if isinstance(points, list) and points:
            point = points[-1] if action.get("kind") == "drag" and len(points) >= 2 else points[0]
            xy = _numeric_xy(point)
            if xy is not None:
                return xy
        return current
    x_value = action.get("x")
    y_value = action.get("y")
    if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
        return float(x_value), float(y_value)
    return current


def _format_action_schema(kind: str) -> str | None:
    if kind == "click":
        return '{"kind":"click","points":[[<int 0-1000>,<int 0-1000>]]}'
    if kind == "drag":
        return '{"kind":"drag","points":[[<int 0-1000>,<int 0-1000>],[<int 0-1000>,<int 0-1000>]]}'
    if kind == "submit":
        return '{"kind":"submit"}'
    if kind == "move_to":
        return '{"kind":"move_to","x":<int 0-1000>,"y":<int 0-1000>}'
    if kind in {"mouse_down", "mouse_up", "left_click", "done"}:
        return f'{{"kind":"{kind}"}}'
    if kind == "type_text":
        return '{"kind":"type_text","text":"<text>"}'
    if kind == "paste_text":
        return '{"kind":"paste_text","text":"<text>"}'
    if kind == "key_press":
        return '{"kind":"key_press","key":"<key>"}'
    if kind == "scroll":
        return '{"kind":"scroll","direction":"up|down"}'
    if kind == "left_double":
        return '{"kind":"left_double","points":[[<int 0-1000>,<int 0-1000>]]}'
    if kind == "right_single":
        return '{"kind":"right_single","points":[[<int 0-1000>,<int 0-1000>]]}'
    if kind == "type":
        return '{"kind":"type","content":"<text>"}'
    if kind == "hotkey":
        return '{"kind":"hotkey","key":"<key>"}'
    if kind == "finished":
        return '{"kind":"finished","content":"<answer>"}'
    if kind == "browser_back":
        return '{"kind":"browser_back"}'
    if kind == "wait":
        return '{"kind":"wait"}'
    return None


def _is_old_paradigm_context(context: SftActionContext) -> bool:
    return context.action_paradigm == "macro_only"


def format_old_paradigm_action_prompt(instruction: str, action_kind: str) -> str:
    del action_kind
    return f"Task: {instruction}"


def _format_osworld_mouse_action_prompt(
    instruction: str,
    task_action_kinds: Iterable[str],
    *,
    task_type: str | None = None,
    frozen_final_sft_runtime: bool = False,
) -> str:
    if is_ten_choice_captcha_task(task_type):
        from ..protocol_tracks import canonicalize_task_requirement

        instruction = canonicalize_task_requirement(
            instruction,
            task_type=task_type,
        )
    description_lines = _self_built_mouse_task_description_lines(
        task_type,
        target_label=extract_ten_choice_target_label(instruction),
    )
    if frozen_final_sft_runtime and is_rotation_captcha_task(task_type):
        instruction = (
            FROZEN_FINAL_SFT_OUTER_ROTATION_REQUIREMENT
            if instruction == OUTER_ROTATION_TASK_REQUIREMENT
            else FROZEN_FINAL_SFT_ROTATION_REQUIREMENT
        )
        del description_lines
    elif frozen_final_sft_runtime and is_slot_drag_task(task_type):
        pass
    del task_action_kinds
    return f"Task: {instruction}"


def _self_built_mouse_task_description_lines(
    task_type: str | None,
    *,
    target_label: str | None = None,
) -> tuple[str, ...]:
    if is_ten_choice_captcha_task(task_type):
        return ()
    if is_rotation_captcha_task(task_type):
        return (
            "Task description: This is a rotation CAPTCHA. The screen contains an image puzzle with a central rotated region and an on-screen control that changes the puzzle state. Complete the CAPTCHA according to the task requirement.",
        )
    if is_slot_drag_task(task_type):
        return (
            "Task description: Place the solid colored shape into the matching gray outline using the center reticle.",
        )
    if is_third_person_drag_task(task_type):
        return (
            "Task description: Drag the solid colored shape into the matching gray outline.",
        )
    return ()


def _format_osworld_mouse_schema_line(kind: str) -> str:
    if kind == "click":
        return '- click: one immediate left click at a screenshot-relative point; schema {"kind":"click","points":[[<int 0-1000>,<int 0-1000>]]}'
    if kind == "drag":
        return '- drag: one continuous press-drag-release gesture between two screenshot-relative points; schema {"kind":"drag","points":[[<int 0-1000>,<int 0-1000>],[<int 0-1000>,<int 0-1000>]]}'
    if kind == "move_to":
        return '- move_to: move the cursor without pressing; schema {"kind":"move_to","x":<int 0-1000>,"y":<int 0-1000>}'
    if kind == "mouse_down":
        return '- mouse_down: press and hold at the current cursor position; schema {"kind":"mouse_down"}'
    if kind == "mouse_up":
        return '- mouse_up: release at the current cursor position; schema {"kind":"mouse_up"}'
    if kind == "left_click":
        return '- left_click: one immediate left click at the current cursor or reticle position; no coordinates; schema {"kind":"left_click"}'
    schema = _format_action_schema(kind)
    return f"- {kind}: schema {schema}" if schema is not None else f"- {kind}"


def format_action_prompt(
    instruction: str,
    task_action_kinds: Iterable[str],
    *,
    task_type: str | None = None,
    action_space_type: str | None = None,
    frozen_final_sft_runtime: bool = False,
    prompt_profile: str = PROMPT_FAMILY,
) -> str:
    del action_space_type
    if _is_qwen25_mouse_primitive_prompt_profile(prompt_profile):
        return instruction
    if is_self_built_osworld_mouse_task(task_type):
        return _format_osworld_mouse_action_prompt(
            instruction,
            task_action_kinds,
            task_type=task_type,
            frozen_final_sft_runtime=frozen_final_sft_runtime,
        )
    del task_action_kinds, frozen_final_sft_runtime
    return f"Task: {instruction}"


def _reasoning_action_kinds_for_prompt(allowed_kinds: Iterable[str]) -> tuple[str, ...]:
    allowed = tuple(kind for kind in allowed_kinds if kind in SUPPORTED_SFT_OUTPUT_ACTION_KINDS)
    macro_kinds = tuple(kind for kind in allowed if kind in SUPPORTED_SFT_MACRO_ACTION_KINDS)
    if macro_kinds:
        primitive_kinds = tuple(
            kind for kind in ("move_to", "mouse_down", "mouse_up", "left_click") if kind in allowed
        )
        if not primitive_kinds:
            primitive_kinds = ("move_to", "mouse_down", "mouse_up")
        return (*primitive_kinds, *macro_kinds)
    return allowed


def format_sft_action_context(context: SftActionContext) -> str:
    if _is_old_paradigm_context(context):
        return ""
    next_step = context.action_position + 1
    if context.action_position == 1:
        history_intro = "Current prediction context: 1 previous assistant response is included"
    else:
        history_intro = (
            f"Current prediction context: {context.action_position} previous assistant responses are included"
        )
    lines = [
        f"{history_intro} in this prompt in chronological order; older screenshots may be omitted, but all previous "
        "assistant tool calls are retained. The last attached image is the current observation.",
    ]
    if context.action_history:
        lines.append("Use the previous assistant tool calls as context only.")
    else:
        lines.append("No previous actions in this trajectory.")
    lines.append(f"Return the canonical tool call for step {next_step}.")
    return "\n".join(lines)


def build_user_prompt_text(
    instruction: str,
    context: SftActionContext,
    *,
    extra_context_lines: Iterable[str] = (),
) -> str:
    leading_text, trailing_text = build_user_prompt_text_sections(
        instruction,
        context,
        extra_context_lines=extra_context_lines,
    )
    sections = [section for section in (leading_text, trailing_text) if section]
    return "\n\n".join(sections)


def build_user_prompt_text_sections(
    instruction: str,
    context: SftActionContext,
    *,
    extra_context_lines: Iterable[str] = (),
    frozen_final_sft_runtime: bool = False,
    prompt_profile: str = PROMPT_FAMILY,
) -> tuple[str, str]:
    if context.protocol_track == "groundcua_static_progressive":
        extra_lines = [line for line in extra_context_lines if line]
        leading_text = format_action_prompt(
            instruction,
            context.task_action_kinds,
            task_type=context.task_type,
            action_space_type=context.action_space_type,
            frozen_final_sft_runtime=frozen_final_sft_runtime,
            prompt_profile=prompt_profile,
        )
        return leading_text, "\n".join(extra_lines)
    leading_sections = [
        format_action_prompt(
            instruction,
            context.task_action_kinds,
            task_type=context.task_type,
            action_space_type=context.action_space_type,
            frozen_final_sft_runtime=frozen_final_sft_runtime,
            prompt_profile=prompt_profile,
        ),
    ]
    trailing_sections = [
        format_sft_action_context(context),
    ]
    leading_sections = [section for section in leading_sections if section]
    trailing_sections = [section for section in trailing_sections if section]
    extra_lines = [line for line in extra_context_lines if line]
    if extra_lines:
        trailing_sections.append("\n".join(extra_lines))
    return "\n\n".join(leading_sections), "\n\n".join(trailing_sections)

def build_prompt_messages_with_images(
    *,
    instruction: str,
    image_path: Path,
    context: SftActionContext,
    action: dict[str, Any] | None = None,
    assistant_response: str | None = None,
    thought: str | None = None,
    extra_context_lines: Iterable[str] = (),
    image_paths: tuple[Path, ...] = (),
    thought_history: tuple[str | None, ...] = (),
    assistant_response_history: tuple[str, ...] = (),
    images_to_keep: int = IMAGE_HISTORY_MAX,
    frozen_final_sft_runtime: bool = False,
    prompt_profile: str = PROMPT_FAMILY,
) -> SftPromptBuildResult:
    multimodal_content, retained_paths = build_sft_multimodal_user_content(
        image_path=image_path,
        image_paths=image_paths,
        action_history=context.action_history,
        thought_history=thought_history,
        assistant_response_history=assistant_response_history,
        context=context,
        images_to_keep=images_to_keep,
    )

    system_prompt = _system_prompt_for_profile(prompt_profile)
    leading_text, trailing_text = build_user_prompt_text_sections(
        instruction,
        context,
        extra_context_lines=extra_context_lines,
        frozen_final_sft_runtime=frozen_final_sft_runtime,
        prompt_profile=prompt_profile,
    )
    user_content: list[dict[str, Any]] = []
    if leading_text:
        user_content.append({"type": "text", "text": leading_text})
    user_content.extend(multimodal_content)
    if trailing_text:
        user_content.append({"type": "text", "text": trailing_text})

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    if action is not None:
        assistant_text = (
            assistant_response.strip()
            if isinstance(assistant_response, str) and assistant_response.strip()
            else _history_action_text(action=action, thought=thought, context=context)
        )
        messages.append({"role": "assistant", "content": assistant_text})
    return SftPromptBuildResult(
        messages=messages,
        image_paths=retained_paths,
        context=context,
        prompt_contract=(
            FROZEN_FINAL_SFT_RUNTIME_SELF_HISTORY_CONTRACT
            if frozen_final_sft_runtime
            else None
        ),
    )


def build_prompt_messages(
    *,
    instruction: str,
    image_path: Path,
    context: SftActionContext,
    action: dict[str, Any] | None = None,
    assistant_response: str | None = None,
    thought: str | None = None,
    extra_context_lines: Iterable[str] = (),
    image_paths: tuple[Path, ...] = (),
    thought_history: tuple[str | None, ...] = (),
    assistant_response_history: tuple[str, ...] = (),
    images_to_keep: int = IMAGE_HISTORY_MAX,
    frozen_final_sft_runtime: bool = False,
    prompt_profile: str = PROMPT_FAMILY,
) -> list[dict[str, Any]]:
    return build_prompt_messages_with_images(
        instruction=instruction,
        image_path=image_path,
        context=context,
        action=action,
        assistant_response=assistant_response,
        thought=thought,
        extra_context_lines=extra_context_lines,
        image_paths=image_paths,
        thought_history=thought_history,
        assistant_response_history=assistant_response_history,
        images_to_keep=images_to_keep,
        frozen_final_sft_runtime=frozen_final_sft_runtime,
        prompt_profile=prompt_profile,
    ).messages


def build_prompt_messages_from_history(
    *,
    instruction: str,
    image_path: Path,
    image_paths: tuple[Path, ...],
    action_history: Iterable[dict[str, Any]],
    total_actions: int | None,
    cursor_xy: tuple[float, float] | None,
    button_state: str,
    task_type: str | None = None,
    allowed_kinds: Iterable[str] | None = None,
    budget_remaining: int | None = None,
    coordinate_format: str = QWEN3_RELATIVE_COORDINATE_FORMAT,
    image_size_px: tuple[int, int] | None = None,
    action: dict[str, Any] | None = None,
    assistant_response: str | None = None,
    thought: str | None = None,
    thought_history: tuple[str | None, ...] = (),
    assistant_response_history: tuple[str, ...] = (),
    extra_context_lines: Iterable[str] = (),
    images_to_keep: int = IMAGE_HISTORY_MAX,
    frozen_final_sft_runtime: bool = False,
    prompt_profile: str = PROMPT_FAMILY,
) -> SftPromptBuildResult:
    finalized_history = tuple(
        _finalize_action_for_coordinate_format(dict(action_item), coordinate_format)
        for action_item in action_history
    )
    context = build_action_context_from_history(
        action_history=finalized_history,
        total_actions=total_actions,
        cursor_xy=_finalize_xy_for_coordinate_format(cursor_xy, coordinate_format),
        button_state=button_state,
        task_type=task_type,
        allowed_kinds=allowed_kinds,
        budget_remaining=budget_remaining,
    )
    context = replace(
        context,
        coordinate_format=_model_coordinate_format(coordinate_format),
    )
    if coordinate_format == PIXEL_COORDINATE_FORMAT and image_size_px is not None:
        context = convert_pixel_context_to_qwen3_relative(context, image_size_px)
    elif image_size_px is not None:
        context = replace(context, image_size_px=image_size_px)
    return build_prompt_messages_with_images(
        instruction=instruction,
        image_path=image_path,
        context=context,
        action=action,
        assistant_response=assistant_response,
        thought=thought,
        extra_context_lines=extra_context_lines,
        image_paths=image_paths,
        thought_history=thought_history,
        assistant_response_history=assistant_response_history,
        images_to_keep=images_to_keep,
        frozen_final_sft_runtime=frozen_final_sft_runtime,
        prompt_profile=prompt_profile,
    )


def _normalize_action(step: dict[str, Any]) -> dict[str, Any]:
    if "action" in step and isinstance(step["action"], dict):
        return _clean_action_for_history(dict(step["action"]))
    action = {
        key: value
        for key, value in step.items()
        if key not in {
            "type",
            "index",
            "observation_image",
            "image_path",
            "source_frame_index",
            "trainable",
            "loss_weight",
            CANONICAL_ASSISTANT_FIELD,
            CANONICAL_THINK_FIELD,
            "thinking_process",
            "thought",
            "teacher_think_metadata",
            "metadata",
        }
    }
    return action


def _is_trainable_step(step: dict[str, Any]) -> bool:
    trainable = step.get("trainable")
    if trainable is False:
        return False
    loss_weight = step.get("loss_weight")
    if isinstance(loss_weight, (int, float)) and float(loss_weight) <= 0.0:
        return False
    return True


def _extract_step_thought(step: dict[str, Any]) -> str | None:
    value = extract_think_text(step)
    if value is not None:
        return value
    metadata = step.get("metadata")
    return extract_think_text(metadata if isinstance(metadata, dict) else None)


def _assistant_response_from_step(
    step: dict[str, Any],
    *,
    action: dict[str, Any],
    context: SftActionContext | None = None,
) -> str | None:
    assistant_response = extract_assistant_response(step)
    if assistant_response is not None:
        return assistant_response
    metadata = step.get("metadata")
    if isinstance(metadata, dict):
        assistant_response = extract_assistant_response(metadata)
        if assistant_response is not None:
            return assistant_response
    thought = _extract_step_thought(step)
    if thought is None:
        return None
    return _react_action_text(action=action, thought=thought, context=context)


def _extract_record_metadata_thought(
    metadata: dict[str, Any],
    action_index: int,
) -> str | None:
    return extract_think_text(metadata, action_index=action_index)


def _assistant_response_from_record_metadata(
    metadata: dict[str, Any],
    *,
    action: dict[str, Any],
    action_index: int,
    context: SftActionContext | None = None,
) -> str | None:
    assistant_response = extract_assistant_response(metadata)
    if assistant_response is not None:
        return assistant_response
    thought = _extract_record_metadata_thought(metadata, action_index)
    if thought is None:
        return None
    return _react_action_text(action=action, thought=thought, context=context)


def _parse_think_json_action_response(response: str, *, context: str) -> dict[str, Any]:
    thought, action_text = split_think_json_response(response.strip())
    if not isinstance(thought, str) or not thought.strip():
        raise ValueError(f"{context} must contain a nonempty <think> block")
    try:
        payload = json.loads(action_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} action JSON is invalid: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("action"), dict):
        raise ValueError(f"{context} must contain JSON object with an action object")
    return _clean_action_for_history(dict(payload["action"]))


def _parse_groundcua_assistant_response(
    response: str,
    *,
    context: str,
    prompt_profile: str | None = None,
) -> dict[str, Any]:
    stripped = response.strip()
    if prompt_profile in {
        QWEN3_MOVETO_LEFTCLICK_THINK_PROFILE,
        QWEN25_MOUSEMOVE_LEFTCLICK_THINK_PROFILE,
    }:
        _thought, action_name, coordinate = parse_groundcua_think_tool_call(
            prompt_profile,
            stripped,
        )
        action: dict[str, Any] = {"kind": action_name}
        if coordinate is not None:
            action.update({"x": coordinate[0], "y": coordinate[1]})
        return _clean_action_for_history(action)
    if "<tool_call>" in stripped:
        parsed = parse_unified_three_action_response(stripped)
        if parsed["parse_error"] is not None or not isinstance(parsed["action"], dict):
            raise ValueError(f"{context} tool_call is invalid: {parsed['parse_error']}")
        return _clean_action_for_history(dict(parsed["action"]))
    return _parse_think_json_action_response(stripped, context=context)


def _validate_groundcua_assistant_response(
    assistant_response: str | None,
    *,
    fixed_action: dict[str, Any],
    prompt_profile: str | None = None,
) -> str:
    if not isinstance(assistant_response, str) or not assistant_response.strip():
        raise ValueError(
            "GroundCUA action step requires canonical assistant_response; "
            "legacy think fields and generated fallback thoughts are not accepted"
        )
    parsed_action = _parse_groundcua_assistant_response(
        assistant_response,
        context="GroundCUA canonical assistant_response",
        prompt_profile=prompt_profile,
    )
    if parsed_action != _clean_action_for_history(fixed_action):
        raise ValueError(
            "GroundCUA canonical assistant_response action does not match fixed action"
        )
    return assistant_response.strip()


def _examples_from_flat_record(record: dict[str, Any]) -> Iterable[SftExample]:
    instruction = record.get("instruction", "")
    record_metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    coordinate_format = str(record_metadata.get("coordinate_format") or PIXEL_COORDINATE_FORMAT)
    task_type = str(record.get("task_type") or record_metadata.get("task_type") or "")
    track = resolve_protocol_track(task_type=task_type or None)
    action_steps = [
        step for step in record.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("action"), dict)
    ]
    action_history: list[dict[str, Any]] = []
    cursor_xy: tuple[float, float] | None = None
    button_state = "up"
    image_history: list[Path] = []
    thought_history: list[str | None] = []
    assistant_response_history: list[str] = []
    for step in record.get("steps", []):
        image_path = step.get("observation_image")
        action = step.get("action")
        if not image_path or not isinstance(action, dict):
            continue
        current_image = Path(image_path)
        append_image_history(image_history, current_image)
        finalized_cursor_xy = _finalize_xy_for_coordinate_format(cursor_xy, coordinate_format)
        finalized_history = [
            _finalize_action_for_coordinate_format(item, coordinate_format)
            for item in action_history
        ]
        context = SftActionContext(
            action_position=len(action_history),
            total_actions=len(action_steps),
            cursor_xy=finalized_cursor_xy,
            button_state=button_state,
            task_type=task_type,
            phase="flat_record",
            allowed_kinds=track.task_action_kinds,
            task_action_kinds=track.task_action_kinds,
            budget_remaining=len(action_steps) - len(action_history),
            action_history=tuple(finalized_history),
            coordinate_format=_model_coordinate_format(coordinate_format),
            action_paradigm=track.action_paradigm,
            protocol_track=track.track_id,
            action_space_type=track.action_space_type,
            protocol_version=track.protocol_version,
            thought_required=track.thought_required,
        )
        normalized = _finalize_action_for_coordinate_format(dict(action), coordinate_format)
        step_thought = (
            _extract_step_thought(step)
            or _extract_record_metadata_thought(record_metadata, len(action_history))
        )
        assistant_response = _assistant_response_from_step(step, action=normalized, context=context) or _assistant_response_from_record_metadata(
            record_metadata,
            action=normalized,
            action_index=len(action_history),
            context=context,
        )
        yield SftExample(
            instruction=instruction,
            image_path=current_image,
            action=normalized,
            context=context,
            assistant_response=assistant_response,
            thought=step_thought,
            coordinate_format=_model_coordinate_format(coordinate_format),
            image_paths=tuple(image_history),
            thought_history=tuple(thought_history),
            assistant_response_history=(
                tuple(assistant_response_history)
                if len(assistant_response_history) == len(action_history)
                else ()
            ),
        )
        if assistant_response is not None and len(assistant_response_history) == len(action_history):
            assistant_response_history.append(assistant_response)
        else:
            assistant_response_history.clear()
        action_history.append(_clean_action_for_history(dict(action)))
        thought_history.append(step_thought)
        cursor_xy = _update_cursor_from_action(dict(action), cursor_xy)
        button_state = _update_button_state(normalized, button_state)


def _examples_from_new_paradigm_record(record: dict[str, Any]) -> Iterable[SftExample]:
    instruction = record.get("instruction", "")
    current_image: Path | None = None
    current_cursor_xy: tuple[float, float] | None = None
    action_history: list[dict[str, Any]] = []
    button_state = "up"
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    metadata_action_space_type = metadata.get("action_space_type")
    metadata_protocol_track = metadata.get("protocol_track")
    metadata_protocol_version = metadata.get("protocol_version")
    coordinate_format = str(metadata.get("coordinate_format") or PIXEL_COORDINATE_FORMAT)
    metadata_task_action_kinds = metadata.get("task_action_kinds")
    if isinstance(metadata_task_action_kinds, list) and all(
        isinstance(kind, str) for kind in metadata_task_action_kinds
    ):
        record_task_action_kinds: tuple[str, ...] | None = tuple(metadata_task_action_kinds)
    else:
        record_task_action_kinds = None
    task_type = record.get("task_type") or metadata.get("task_type")
    normalized_task_type = str(task_type) if task_type is not None else None
    if _is_groundcua_progressive_task(normalized_task_type):
        record_task_action_kinds = STRICT_MOUSE_PRIMITIVE_ACTION_KINDS
    elif _is_mouse_captcha_task(normalized_task_type):
        record_task_action_kinds = MOUSE_CAPTCHA_ACTION_KINDS
    total_actions = sum(
        1
        for step in record.get("steps", [])
        if isinstance(step, dict) and step.get("type") == "action"
    )
    image_history: list[Path] = []
    thought_history: list[str | None] = []
    assistant_response_history: list[str] = []
    for step in record.get("steps", []):
        if step.get("type") == "observation":
            image_path = step.get("image_path") or step.get("observation_image")
            current_image = Path(image_path) if image_path else None
            if current_image is not None:
                append_image_history(image_history, current_image)
            cursor_xy = _numeric_xy(step.get("cursor_xy"))
            if cursor_xy is not None:
                current_cursor_xy = cursor_xy
            continue
        if step.get("type") == "action" and current_image is not None:
            action = _normalize_action(step)
            context = _context_from_state(
                action_history=action_history,
                total_actions=total_actions,
                cursor_xy=current_cursor_xy,
                button_state=button_state,
                task_type=str(task_type) if task_type is not None else None,
                allowed_kinds=record_task_action_kinds,
                budget_remaining=total_actions - len(action_history),
            )
            context = replace(
                context,
                cursor_xy=_finalize_xy_for_coordinate_format(context.cursor_xy, coordinate_format),
                action_history=tuple(
                    _finalize_action_for_coordinate_format(item, coordinate_format)
                    for item in context.action_history
                ),
                coordinate_format=_model_coordinate_format(coordinate_format),
                action_space_type=(
                    str(metadata_action_space_type)
                    if isinstance(metadata_action_space_type, str) and metadata_action_space_type
                    else context.action_space_type
                ),
                protocol_track=(
                    str(metadata_protocol_track)
                    if isinstance(metadata_protocol_track, str) and metadata_protocol_track
                    else context.protocol_track
                ),
                protocol_version=(
                    str(metadata_protocol_version)
                    if isinstance(metadata_protocol_version, str) and metadata_protocol_version
                    else context.protocol_version
                ),
            )
            model_action = _finalize_action_for_coordinate_format(action, coordinate_format)
            step_thought = (
                _extract_step_thought(step)
                or _extract_record_metadata_thought(metadata, len(action_history))
            )
            if _is_groundcua_progressive_task(normalized_task_type):
                assistant_response = _validate_groundcua_assistant_response(
                    extract_assistant_response(step),
                    fixed_action=model_action,
                    prompt_profile=(
                        metadata.get("prompt_profile")
                        if isinstance(metadata.get("prompt_profile"), str)
                        else None
                    ),
                )
            else:
                assistant_response = _assistant_response_from_step(step, action=model_action, context=context) or _assistant_response_from_record_metadata(
                    metadata,
                    action=model_action,
                    action_index=len(action_history),
                    context=context,
                )
            if _is_trainable_step(step):
                yield SftExample(
                    instruction=instruction,
                    image_path=current_image,
                    action=model_action,
                    context=context,
                    assistant_response=assistant_response,
                    thought=step_thought,
                    coordinate_format=_model_coordinate_format(coordinate_format),
                    image_paths=tuple(image_history),
                    thought_history=tuple(thought_history),
                    assistant_response_history=(
                        tuple(assistant_response_history)
                        if len(assistant_response_history) == len(action_history)
                        else ()
                    ),
                )
            if assistant_response is not None and len(assistant_response_history) == len(action_history):
                assistant_response_history.append(assistant_response)
            else:
                assistant_response_history.clear()
            action_history.append(_clean_action_for_history(action))
            thought_history.append(step_thought)
            current_cursor_xy = _update_cursor_from_action(action, current_cursor_xy)
            button_state = _update_button_state(model_action, button_state)


def _old_paradigm_action_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    action_kind = record.get("action")
    if action_kind == "click":
        x_value = record.get("x")
        y_value = record.get("y")
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            return {"kind": "click", "points": [[x_value, y_value]]}
        return None
    if action_kind == "drag":
        values = [
            record.get("start_x"),
            record.get("start_y"),
            record.get("end_x"),
            record.get("end_y"),
        ]
        if all(isinstance(value, (int, float)) for value in values):
            return {
                "kind": "drag",
                "points": [
                    [values[0], values[1]],
                    [values[2], values[3]],
                ],
            }
        return None
    if action_kind == "submit":
        action = {"kind": "submit"}
        if "answer" in record:
            action["answer"] = record["answer"]
        return action
    return None


def _examples_from_old_paradigm_record(record: dict[str, Any]) -> Iterable[SftExample]:
    image_path = record.get("image_path") or record.get("observation_image")
    if not image_path:
        return
    action = _old_paradigm_action_from_record(record)
    if action is None:
        return
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    coordinate_format = str(metadata.get("coordinate_format") or QWEN3_RELATIVE_COORDINATE_FORMAT)
    model_action = _finalize_action_for_coordinate_format(action, coordinate_format)
    action_kind = str(model_action.get("kind"))
    task_type = str(record.get("task_type") or metadata.get("task_type") or "")
    track = resolve_protocol_track(task_type=task_type or None, action_kinds=(action_kind,))
    context = SftActionContext(
        action_position=0,
        total_actions=1,
        cursor_xy=None,
        button_state="up",
        task_type=task_type or None,
        phase="macro_action",
        allowed_kinds=(action_kind,),
        task_action_kinds=track.task_action_kinds or _macro_task_action_kinds(action_kind),
        budget_remaining=1,
        action_history=(),
        coordinate_format=_model_coordinate_format(coordinate_format),
        action_paradigm=track.action_paradigm,
        protocol_track=track.track_id,
        action_space_type=track.action_space_type,
        protocol_version=track.protocol_version,
        thought_required=track.thought_required,
    )
    yield SftExample(
        instruction=str(record.get("instruction", "")),
        image_path=Path(image_path),
        action=model_action,
        context=context,
        assistant_response=(
            extract_assistant_response(record)
            or (
                _react_action_text(action=model_action, thought=extract_think_text(record), context=context)
                if extract_think_text(record) is not None
                else None
            )
        ),
        thought=extract_think_text(record),
        coordinate_format=_model_coordinate_format(coordinate_format),
        image_paths=(Path(image_path),),
    )


def _examples_from_screenspot_pro_groundcua_record(
    record: dict[str, Any],
) -> Iterable[SftExample]:
    """Load a single, model-visible ScreenSpot-Pro function-call target.

    The manifest stores only the model action spelling and the exact response.  A
    temporary internal ``kind`` object is reconstructed solely for existing
    ``SftExample`` bookkeeping and is never inserted into prompt text or labels.
    """

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ScreenSpot-Pro GroundCUA manifest row requires metadata")
    profile_name = metadata.get("prompt_profile")
    if not is_screenspot_pro_groundcua_profile(profile_name):
        raise ValueError(f"invalid ScreenSpot-Pro GroundCUA prompt_profile: {profile_name!r}")
    profile = get_groundcua_profile(profile_name)
    instruction = record.get("instruction")
    image_path = record.get("image_path")
    model_action = record.get("model_action")
    assistant_response = record.get("assistant_response")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("ScreenSpot-Pro GroundCUA instruction must be nonempty")
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("ScreenSpot-Pro GroundCUA image_path is required")
    if not isinstance(model_action, str) or model_action not in profile.allowed_actions:
        raise ValueError("ScreenSpot-Pro GroundCUA model_action is invalid for its profile")
    if not isinstance(assistant_response, str) or not assistant_response.strip():
        raise ValueError("ScreenSpot-Pro GroundCUA assistant_response is required")
    if '"kind"' in assistant_response or "<think>" in assistant_response.lower():
        raise ValueError("legacy GroundCUA action/kind assistant_response is forbidden")
    coordinate = record.get("coordinate")
    action_requires_coordinate = model_action in {
        "left_click",
        "mouse_move",
        "move_to",
        "left_click_drag",
    }
    if action_requires_coordinate:
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in coordinate)
        ):
            raise ValueError("ScreenSpot-Pro GroundCUA coordinate must contain two integers")
        x, y = coordinate
        if x < 0 or y < 0:
            raise ValueError("ScreenSpot-Pro GroundCUA coordinates must be nonnegative")
        if profile.coordinate_format == QWEN3_RELATIVE_COORDINATE_FORMAT and (x > 1000 or y > 1000):
            raise ValueError("Qwen3 ScreenSpot-Pro coordinates must be within 0..1000")
        action = {"kind": model_action, "x": x, "y": y}
    else:
        if coordinate is not None:
            raise ValueError("ScreenSpot-Pro GroundCUA coordinate is forbidden for this action")
        action = {"kind": model_action}
    if profile_name == QWEN25_DIRECT_PROFILE:
        source_size = metadata.get("source_image_size")
        model_size = metadata.get("model_image_size")
        source_coordinate = metadata.get("source_target_pixel_xy")
        if (
            not isinstance(source_size, list)
            or len(source_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source_size)
            or not isinstance(model_size, list)
            or len(model_size) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in model_size)
            or not isinstance(source_coordinate, list)
            or len(source_coordinate) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in source_coordinate)
        ):
            raise ValueError("Qwen2.5 direct manifest geometry metadata is required")
        from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize

        expected_height, expected_width = smart_resize(
            source_size[1],
            source_size[0],
            factor=profile.image_factor,
            min_pixels=profile.image_min_pixels,
            max_pixels=profile.image_max_pixels,
        )
        expected_model_size = [int(expected_width), int(expected_height)]
        if model_size != expected_model_size:
            raise ValueError(
                "Qwen2.5 direct manifest model_image_size does not match official smart_resize"
            )

        def _half_up(value: Decimal) -> int:
            return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))

        expected_coordinate = [
            _half_up(Decimal(source_coordinate[0]) * model_size[0] / source_size[0]),
            _half_up(Decimal(source_coordinate[1]) * model_size[1] / source_size[1]),
        ]
        if coordinate != expected_coordinate:
            raise ValueError(
                "Qwen2.5 direct manifest resized model coordinate does not match source coordinate"
            )
    parse_groundcua_tool_call(profile_name, assistant_response.strip())
    context = SftActionContext(
        action_position=0,
        total_actions=1,
        task_type="screenspot_pro_groundcua",
        phase="screenspot_pro_function_call",
        allowed_kinds=(model_action,),
        task_action_kinds=(model_action,),
        budget_remaining=1,
        coordinate_format=profile.coordinate_format,
        action_paradigm=profile.paradigm,
        protocol_track="screenspot_pro_groundcua",
        action_space_type="screenspot_pro_function_call",
        protocol_version=profile.name,
        thought_required=False,
    )
    yield SftExample(
        instruction=instruction,
        image_path=Path(image_path),
        action=action,
        context=context,
        assistant_response=assistant_response.strip(),
        thought=None,
        coordinate_format=profile.coordinate_format,
        image_paths=(Path(image_path),),
    )


_QWEN3_MOVE_ACTION_KINDS = ("move_to", "mouse_down", "mouse_up")
_SEQUENTIAL_MOVE_PROFILES = (QWEN25_MOVE_PROFILE, QWEN3_MOVE_PROFILE)


def _is_sequential_move_manifest_record(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    return (
        record.get("task_type") == "screenspot_pro_groundcua"
        and isinstance(metadata, dict)
        and metadata.get("prompt_profile") in _SEQUENTIAL_MOVE_PROFILES
    )


def _source_record_id(record: dict[str, Any]) -> str:
    value = record.get("source_record_id")
    if not isinstance(value, str) or not value.strip():
        value = record.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Qwen3 move manifest row requires source_record_id or id")
    return value.strip()


def _ordered_sequential_move_trajectory_records(
    records: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for record in records:
        if not _is_sequential_move_manifest_record(record):
            continue
        record_id = _source_record_id(record)
        if record_id not in groups:
            groups[record_id] = []
            order.append(record_id)
        groups[record_id].append(record)

    trajectories: list[list[dict[str, Any]]] = []
    for record_id in order:
        group = groups[record_id]
        step_values = [
            record.get("metadata", {}).get("source_step_index")
            for record in group
        ]
        if all(isinstance(value, int) and not isinstance(value, bool) for value in step_values):
            if len(set(step_values)) != len(step_values):
                raise ValueError(
                    f"duplicate source_step_index in sequential move trajectory {record_id}"
                )
            group = sorted(group, key=lambda record: int(record["metadata"]["source_step_index"]))
        trajectories.append(group)
    return trajectories


def _examples_from_sequential_move_trajectory(
    records: Sequence[dict[str, Any]],
) -> Iterable[SftExample]:
    trajectories = _ordered_sequential_move_trajectory_records(records)
    for trajectory in trajectories:
        base_examples: list[SftExample] = []
        for record in trajectory:
            parsed = list(_examples_from_screenspot_pro_groundcua_record(record))
            if len(parsed) != 1:
                raise ValueError("sequential move manifest row must produce exactly one example")
            base_examples.append(parsed[0])
        if not base_examples:
            continue
        instruction = base_examples[0].instruction
        if any(example.instruction != instruction for example in base_examples):
            raise ValueError("sequential move trajectory has inconsistent instructions")
        profile_name = base_examples[0].context.protocol_version
        if any(example.context.protocol_version != profile_name for example in base_examples):
            raise ValueError("sequential move trajectory mixes prompt profiles")
        action_kinds = get_groundcua_profile(profile_name).allowed_actions

        action_history: list[dict[str, Any]] = []
        response_history: list[str] = []
        image_history: list[Path] = []
        button_state = "up"
        total_actions = len(base_examples)
        for action_position, example in enumerate(base_examples):
            if not isinstance(example.assistant_response, str):
                raise ValueError("sequential move trajectory requires exact assistant responses")
            image_history.append(example.image_path)
            context = replace(
                example.context,
                action_position=action_position,
                total_actions=total_actions,
                button_state=button_state,
                allowed_kinds=action_kinds,
                task_action_kinds=action_kinds,
                budget_remaining=total_actions - action_position,
                action_history=tuple(action_history),
            )
            yield replace(
                example,
                context=context,
                image_paths=tuple(image_history),
                assistant_response_history=tuple(response_history),
                thought_history=(),
            )
            action_history.append(dict(example.action))
            response_history.append(example.assistant_response)
            button_state = _update_button_state(example.action, button_state)


def _examples_from_record(record: dict[str, Any]) -> list[SftExample]:
    if record.get("task_type") == "screenspot_pro_groundcua":
        return list(_examples_from_screenspot_pro_groundcua_record(record))
    if isinstance(record.get("action"), str) and (record.get("image_path") or record.get("observation_image")):
        return list(_examples_from_old_paradigm_record(record))
    steps = record.get("steps", [])
    if any(isinstance(step, dict) and step.get("type") for step in steps):
        return list(_examples_from_new_paradigm_record(record))
    return list(_examples_from_flat_record(record))


def load_sft_examples(
    manifest_paths: list[Path],
    *,
    max_examples: int | None = None,
    skip_missing_images: bool = False,
) -> list[SftExample]:
    examples: list[SftExample] = []

    def append_examples(
        generated: Iterable[SftExample],
        manifest_path: Path,
    ) -> bool:
        for example in generated:
            if skip_missing_images and not example.image_path.is_file():
                continue
            if not skip_missing_images and not example.image_path.exists():
                raise FileNotFoundError(
                    f"Missing image for {manifest_path}: {example.image_path}"
                )
            examples.append(example)
            if max_examples is not None and len(examples) >= max_examples:
                return True
        return False

    for manifest_path in manifest_paths:
        pending_trajectory: list[dict[str, Any]] = []
        pending_record_id: str | None = None
        closed_record_ids: set[str] = set()

        def flush_pending() -> bool:
            nonlocal pending_record_id, pending_trajectory
            if not pending_trajectory:
                return False
            stopped = append_examples(
                _examples_from_sequential_move_trajectory(pending_trajectory),
                manifest_path,
            )
            if pending_record_id is not None:
                closed_record_ids.add(pending_record_id)
            pending_record_id = None
            pending_trajectory = []
            return stopped

        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"manifest row must be an object: {manifest_path}:{line_number}")
                if _is_sequential_move_manifest_record(record):
                    record_id = _source_record_id(record)
                    if record_id in closed_record_ids:
                        raise ValueError(
                            "sequential move trajectory rows must be contiguous: "
                            f"{record_id} reappeared at {manifest_path}:{line_number}"
                        )
                    if pending_record_id is None:
                        pending_record_id = record_id
                    elif pending_record_id != record_id:
                        if flush_pending():
                            return examples
                        pending_record_id = record_id
                    pending_trajectory.append(record)
                    continue

                if flush_pending():
                    return examples
                if append_examples(_examples_from_record(record), manifest_path):
                    return examples
        if flush_pending():
            return examples
    return examples


def build_messages(
    example: SftExample,
    *,
    extra_context_lines: Iterable[str] = (),
    images_to_keep: int = IMAGE_HISTORY_MAX,
    prompt_profile: str = PROMPT_FAMILY,
) -> list[dict[str, Any]]:
    if is_screenspot_pro_groundcua_profile(prompt_profile):
        raise ValueError(
            "ScreenSpot-Pro GroundCUA profiles require the raw ScreenSpot-Pro "
            "GroundCUA prompt path, not generic GroundCUA messages"
        )
    return build_prompt_messages(
        instruction=example.instruction,
        image_path=example.image_path,
        context=example.context,
        action=example.action,
        assistant_response=example.assistant_response,
        thought=example.thought,
        extra_context_lines=extra_context_lines,
        image_paths=example.image_paths,
        thought_history=example.thought_history,
        assistant_response_history=example.assistant_response_history,
        images_to_keep=images_to_keep,
        prompt_profile=prompt_profile,
    )


def sft_input_key(
    example: SftExample,
    *,
    key_mode: str = "prompt_context",
) -> tuple[str, ...]:
    if key_mode == "instruction_image":
        return (example.instruction, str(example.image_path))
    if key_mode == "prompt_context":
        return (
            example.instruction,
            str(example.image_path),
            format_sft_action_context(example.context),
        )
    raise ValueError(f"Unsupported SFT input key mode: {key_mode}")


def compute_prompt_conflict_stats(
    examples: Iterable[SftExample],
    *,
    key_mode: str = "prompt_context",
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    total = 0
    for example in examples:
        total += 1
        groups[sft_input_key(example, key_mode=key_mode)][
            json.dumps(example.action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ] += 1
    conflict_groups = sum(1 for counts in groups.values() if len(counts) > 1)
    majority = sum(max(counts.values()) for counts in groups.values()) if groups else 0
    return {
        "examples": total,
        "unique_input_groups": len(groups),
        "conflict_groups": conflict_groups,
        "majority_ceiling": majority / total if total else 0.0,
        "key_mode": key_mode,
    }


class Qwen3VLSftDataset:
    def __init__(self, examples: list[SftExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SftExample:
        return self.examples[index]


class Qwen3VLCollator:
    def __init__(
        self,
        processor: Any,
        *,
        max_length: int = 4096,
        image_min_pixels: int | None = None,
        image_max_pixels: int | None = None,
        prompt_profile: str = PROMPT_FAMILY,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.prompt_profile = prompt_profile
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "right"

    def _load_image_with_transform(self, image_path: Path) -> tuple[Any, ImageCoordinateTransform]:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        if self.prompt_profile in {QWEN25_DIRECT_PROFILE, QWEN25_MOVE_PROFILE}:
            profile = get_groundcua_profile(self.prompt_profile)
            return resize_image_for_qwen25vl_official(
                image,
                image_min_pixels=(
                    self.image_min_pixels
                    if self.image_min_pixels is not None
                    else profile.image_min_pixels
                ),
                image_max_pixels=(
                    self.image_max_pixels
                    if self.image_max_pixels is not None
                    else profile.image_max_pixels
                ),
            )
        if self.prompt_profile == QWEN3_DIRECT_PROFILE:
            profile = get_groundcua_profile(self.prompt_profile)
            return resize_image_for_qwen3vl_official(
                image,
                image_min_pixels=(
                    self.image_min_pixels
                    if self.image_min_pixels is not None
                    else profile.image_min_pixels
                ),
                image_max_pixels=(
                    self.image_max_pixels
                    if self.image_max_pixels is not None
                    else profile.image_max_pixels
                ),
            )
        if self.image_min_pixels is None:
            return resize_image_for_max_pixels(image, self.image_max_pixels)
        if is_screenspot_pro_groundcua_profile(self.prompt_profile):
            size_factor = get_groundcua_profile(self.prompt_profile).image_factor
        else:
            size_factor = (
                SCREENSPOT_PRO_QWEN3VL_IMAGE_SIZE_FACTOR
                if self.prompt_profile == SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE
                else QWEN_IMAGE_SIZE_FACTOR
            )
        return resize_image_for_pixel_range(
            image,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
            size_factor=size_factor,
        )

    def _load_image(self, image_path: Path) -> Any:
        image, _transform = self._load_image_with_transform(image_path)
        return image

    def __call__(self, examples: list[SftExample]) -> dict[str, Any]:
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        images: list[list[Any]] = []

        for example in examples:
            if is_screenspot_pro_groundcua_profile(self.prompt_profile):
                if self.prompt_profile == QWEN25_MOVE_PROFILE:
                    current_image, current_transform = self._load_image_with_transform(
                        example.image_path
                    )
                    label = example.assistant_response
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError(
                            "Qwen2.5 sequential move examples require an exact assistant response"
                        )
                    rendered = render_qwen25_sequential(
                        self.processor,
                        instruction=example.instruction,
                        image_paths=example.image_paths or (example.image_path,),
                        assistant_response_history=example.assistant_response_history,
                        current_screen_size=current_transform.model_size,
                        target_response=label,
                        images_to_keep=QWEN25_SEQUENTIAL_IMAGE_HISTORY_MAX,
                    )
                    prompt_text = rendered.prompt_text
                    if rendered.full_text is None:
                        raise ValueError("Qwen2.5 sequential renderer did not return training text")
                    full_text = rendered.full_text
                    example_images = []
                    for image_path in rendered.image_paths:
                        if image_path == example.image_path:
                            image = current_image
                        else:
                            image, _transform = self._load_image_with_transform(image_path)
                        example_images.append(image)
                elif self.prompt_profile == QWEN3_MOVE_PROFILE:
                    prompt_text, prompt_image_paths = (
                        build_screenspot_pro_groundcua_training_prompt(
                            example,
                            self.prompt_profile,
                            screen_size=(1, 1),
                            images_to_keep=SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX,
                        )
                    )
                    example_images = []
                    for image_path in prompt_image_paths:
                        image, _transform = self._load_image_with_transform(image_path)
                        example_images.append(image)
                    label = example.assistant_response
                    if not isinstance(label, str) or not label.strip():
                        raise ValueError("Qwen3 move examples require an exact assistant response")
                    full_text = prompt_text + label.strip() + "<|im_end|>\n"
                else:
                    image, transform = self._load_image_with_transform(example.image_path)
                    if self.prompt_profile == QWEN3_DIRECT_PROFILE:
                        prompt_text, full_text = build_qwen3_direct_official_training_texts(
                            example,
                            self.processor,
                            image=image,
                            screen_size=transform.model_size,
                        )
                    else:
                        prompt_text, full_text = build_screenspot_pro_groundcua_training_texts(
                            example,
                            self.prompt_profile,
                            screen_size=transform.model_size,
                        )
                    example_images = [image]
                prompt_texts.append(prompt_text)
                full_texts.append(full_text)
                images.append(example_images)
                continue
            model_example = example
            if example.image_paths:
                _multimodal_content, prompt_image_paths = build_sft_multimodal_user_content(
                    image_path=example.image_path,
                    image_paths=example.image_paths,
                    action_history=example.context.action_history,
                    thought_history=example.thought_history,
                    context=example.context,
                )
                example_images = []
                last_transform = None
                for img_path in prompt_image_paths:
                    img, transform = self._load_image_with_transform(img_path)
                    example_images.append(img)
                    last_transform = transform
                transform = last_transform
            else:
                image, transform = self._load_image_with_transform(example.image_path)
                example_images = [image]

            model_example = transform_sft_example_for_model_image(example, transform)
            if self.prompt_profile == SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE:
                prompt_text, full_text = build_screenspot_pro_vllm_training_texts(model_example)
                prompt_texts.append(prompt_text)
                full_texts.append(full_text)
            else:
                messages = build_messages(
                    model_example,
                    prompt_profile=self.prompt_profile,
                )
                prompt_messages = messages[:-1]
                full_texts.append(
                    self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                )
            images.append(example_images)

        if self.prompt_profile == QWEN3_DIRECT_PROFILE:
            # The upstream Transformers evaluator passes its smart-resized PIL
            # image without processor pixel-bound overrides.
            common_kwargs = qwen3_vl_multimodal_processor_kwargs(padding=True)
        else:
            common_kwargs = qwen3_vl_multimodal_processor_kwargs(
                padding=True,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
            )
        batch = self.processor(text=full_texts, images=images, **common_kwargs)
        prompt_batch = self.processor(text=prompt_texts, images=images, **common_kwargs)

        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100

        for row, prompt_mask in enumerate(prompt_batch["attention_mask"]):
            prompt_len = int(prompt_mask.sum().item())
            labels[row, :prompt_len] = -100

        batch["labels"] = labels
        return batch


def build_screenspot_pro_vllm_training_texts(example: SftExample) -> tuple[str, str]:
    if not isinstance(example.assistant_response, str) or not example.assistant_response.strip():
        raise ValueError("ScreenSpot-Pro VLLM examples require an exact assistant_response")
    parse_screenspot_pro_qwen3vl_tool_call(example.assistant_response)
    prompt = build_screenspot_pro_qwen3vl_vllm_prompt(example.instruction)
    return prompt, prompt + example.assistant_response.strip()


def build_qwen3_direct_official_messages(
    instruction: str,
    image: Any,
) -> list[dict[str, Any]]:
    """Expose the pinned upstream Qwen3-VL Transformers message contract."""

    return build_qwen3vl_official_messages(instruction, image)


def build_qwen3_direct_official_training_texts(
    example: SftExample,
    processor: Any,
    *,
    image: Any,
    screen_size: tuple[int, int],
) -> tuple[str, str]:
    """Render Qwen3 directclick exactly as the official Transformers evaluator."""

    del screen_size  # The pinned upstream Qwen3 prompt keeps a 1000x1000 tool schema.
    if not isinstance(example.assistant_response, str) or not example.assistant_response.strip():
        raise ValueError("Qwen3 direct examples require an exact assistant_response")
    label = example.assistant_response.strip()
    parse_groundcua_tool_call(QWEN3_DIRECT_PROFILE, label)
    messages = build_qwen3_direct_official_messages(example.instruction, image)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        [*messages, {"role": "assistant", "content": label}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(prompt, str) or not isinstance(full_text, str):
        raise TypeError("Qwen3 direct processor chat template must return text")
    return prompt, full_text


def build_screenspot_pro_groundcua_training_texts(
    example: SftExample,
    profile_name: str,
    *,
    screen_size: tuple[int, int],
) -> tuple[str, str]:
    """Build raw ScreenSpot-Pro text without the legacy GroundCUA action/kind path."""

    prompt, _retained_paths = build_screenspot_pro_groundcua_training_prompt(
        example,
        profile_name,
        screen_size=screen_size,
    )
    label = example.assistant_response.strip() if isinstance(example.assistant_response, str) else ""
    if profile_name == QWEN3_MOVE_PROFILE:
        return prompt, prompt + label + "<|im_end|>\n"
    return prompt, prompt + label


def build_screenspot_pro_groundcua_training_prompt(
    example: SftExample,
    profile_name: str,
    *,
    screen_size: tuple[int, int],
    images_to_keep: int = SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX,
) -> tuple[str, tuple[Path, ...]]:
    """Build the exact model-visible prompt and image window for SFT."""

    from ..eval.groundcua_move_sequential import build_sequential_prompt

    if not is_screenspot_pro_groundcua_profile(profile_name):
        raise ValueError(f"not a ScreenSpot-Pro GroundCUA profile: {profile_name!r}")
    if not isinstance(example.assistant_response, str) or not example.assistant_response.strip():
        raise ValueError("ScreenSpot-Pro GroundCUA examples require an exact assistant_response")
    label = example.assistant_response.strip()
    if '"kind"' in label or '"action":{"kind"' in label or "<think>" in label.lower():
        raise ValueError("legacy GroundCUA action/kind or hidden reasoning is forbidden")
    profile = get_groundcua_profile(profile_name)
    parse_groundcua_tool_call(profile_name, label)
    if profile_name == QWEN3_MOVE_PROFILE:
        if images_to_keep != SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX:
            raise ValueError(
                "Qwen3 move SFT requires exactly three recent causal images"
            )
        history_paths = example.image_paths or (example.image_path,)
        built = build_sequential_prompt(
            instruction=example.instruction,
            image_paths=history_paths,
            assistant_response_history=example.assistant_response_history,
            images_to_keep=images_to_keep,
        )
        return built.prompt, built.image_paths
    if profile.model_family == "qwen2_5_vl":
        prompt = build_groundcua_prompt(
            profile_name,
            example.instruction,
            screen_width=int(screen_size[0]),
            screen_height=int(screen_size[1]),
        )
    else:
        prompt = build_groundcua_prompt(profile_name, example.instruction)
    return prompt, (example.image_path,)


def validate_groundcua_profile_example(
    example: SftExample,
    profile: GroundCUAProfile,
) -> None:
    """Validate a ScreenSpot-Pro GroundCUA SFT example without rewriting it."""

    if not is_screenspot_pro_groundcua_profile(profile.name):
        raise ValueError(f"not a ScreenSpot-Pro GroundCUA profile: {profile.name!r}")
    if example.coordinate_format != profile.coordinate_format:
        raise ValueError(
            "ScreenSpot-Pro GroundCUA coordinate_format does not match profile"
        )
    if not isinstance(example.assistant_response, str) or not example.assistant_response.strip():
        raise ValueError("ScreenSpot-Pro GroundCUA examples require an exact assistant_response")
    label = example.assistant_response.strip()
    if '"kind"' in label or '"action":{"kind"' in label:
        raise ValueError("legacy GroundCUA action/kind assistant_response is forbidden")
    action, coordinate = parse_groundcua_tool_call(profile.name, label)
    example_action = example.action.get("kind") if isinstance(example.action, dict) else None
    if example_action != action:
        raise ValueError("ScreenSpot-Pro GroundCUA model action does not match label")
    if coordinate is not None:
        expected_coordinate = (
            example.action.get("x") if isinstance(example.action, dict) else None,
            example.action.get("y") if isinstance(example.action, dict) else None,
        )
        if expected_coordinate != coordinate:
            raise ValueError("ScreenSpot-Pro GroundCUA coordinate does not match label")


def validate_screenspot_pro_groundcua_sequential_training_contract(
    examples: Sequence[SftExample],
) -> None:
    """Validate the Qwen3 move SFT examples against sequential inference."""

    from ..eval.groundcua_move_sequential import (
        build_sequential_prompt,
        parse_single_tool_call,
    )

    for index, example in enumerate(examples):
        if example.coordinate_format != QWEN3_RELATIVE_COORDINATE_FORMAT:
            raise ValueError(f"Qwen3 move example {index} has the wrong coordinate format")
        if example.context.coordinate_format != QWEN3_RELATIVE_COORDINATE_FORMAT:
            raise ValueError(f"Qwen3 move context {index} has the wrong coordinate format")
        if example.context.task_action_kinds != _QWEN3_MOVE_ACTION_KINDS:
            raise ValueError(f"Qwen3 move example {index} has the wrong action space")
        if example.thought is not None or example.thought_history:
            raise ValueError(f"Qwen3 move example {index} contains hidden reasoning")
        if not example.image_paths or example.image_paths[-1] != example.image_path:
            raise ValueError(f"Qwen3 move example {index} does not end with current image")
        if len(example.image_paths) != len(example.assistant_response_history) + 1:
            raise ValueError(
                f"Qwen3 move example {index} must have one current image per action history"
            )
        if len(example.context.action_history) != len(example.assistant_response_history):
            raise ValueError(f"Qwen3 move example {index} has misaligned action history")
        for history_index, (response, expected_action) in enumerate(
            zip(example.assistant_response_history, example.context.action_history, strict=True)
        ):
            parsed_action = parse_single_tool_call(response)
            if parsed_action != expected_action:
                raise ValueError(
                    f"Qwen3 move example {index} history action {history_index} mismatches label"
                )
        if not isinstance(example.assistant_response, str):
            raise ValueError(f"Qwen3 move example {index} has no current assistant response")
        if parse_single_tool_call(example.assistant_response) != example.action:
            raise ValueError(f"Qwen3 move example {index} current action mismatches label")
        if index in {0, len(examples) - 1}:
            prompt = build_sequential_prompt(
                instruction=example.instruction,
                image_paths=example.image_paths,
                assistant_response_history=example.assistant_response_history,
                images_to_keep=SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX,
            )
            if len(prompt.image_paths) > SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX:
                raise ValueError(f"Qwen3 move example {index} exceeds three prompt images")
            if '"kind"' in prompt.prompt or "<think>" in prompt.prompt.lower():
                raise ValueError(f"Qwen3 move example {index} leaks forbidden prompt syntax")


def _parsed_groundcua_history_action(
    profile_name: str,
    response: str,
) -> dict[str, Any]:
    action, coordinate = parse_groundcua_tool_call(profile_name, response)
    if coordinate is None:
        return {"kind": action}
    return {"kind": action, "x": coordinate[0], "y": coordinate[1]}


def validate_screenspot_pro_qwen25_sequential_training_contract(
    examples: Sequence[SftExample],
) -> None:
    profile = get_groundcua_profile(QWEN25_MOVE_PROFILE)
    if (
        profile.allowed_actions != QWEN25_MOVE_COMPAT_ACTIONS
        or profile.coordinate_format != "resized_physical_pixels"
        or profile.image_factor != 28
        or profile.image_min_pixels != 784
        or profile.image_max_pixels != 99_999_999
    ):
        raise ValueError("Qwen2.5 sequential profile contract drifted")

    for index, example in enumerate(examples):
        if example.coordinate_format != profile.coordinate_format:
            raise ValueError(f"Qwen2.5 sequential example {index} has wrong coordinates")
        if example.context.coordinate_format != profile.coordinate_format:
            raise ValueError(f"Qwen2.5 sequential context {index} has wrong coordinates")
        if example.context.task_action_kinds != profile.allowed_actions:
            raise ValueError(f"Qwen2.5 sequential example {index} has wrong action space")
        if example.thought is not None or example.thought_history:
            raise ValueError(f"Qwen2.5 sequential example {index} contains hidden reasoning")
        if not example.image_paths or example.image_paths[-1] != example.image_path:
            raise ValueError(f"Qwen2.5 sequential example {index} lacks its current image")
        if len(example.image_paths) != len(example.assistant_response_history) + 1:
            raise ValueError(
                f"Qwen2.5 sequential example {index} has misaligned image and action history"
            )
        if len(example.context.action_history) != len(example.assistant_response_history):
            raise ValueError(f"Qwen2.5 sequential example {index} has misaligned actions")
        if example.context.action_position != len(example.assistant_response_history):
            raise ValueError(f"Qwen2.5 sequential example {index} has wrong action position")
        for history_index, (response, expected_action) in enumerate(
            zip(example.assistant_response_history, example.context.action_history, strict=True)
        ):
            parsed_action = _parsed_groundcua_history_action(QWEN25_MOVE_PROFILE, response)
            if parsed_action != expected_action:
                raise ValueError(
                    f"Qwen2.5 sequential example {index} history {history_index} mismatches"
                )
        if not isinstance(example.assistant_response, str):
            raise ValueError(f"Qwen2.5 sequential example {index} has no current label")
        if (
            _parsed_groundcua_history_action(
                QWEN25_MOVE_PROFILE,
                example.assistant_response,
            )
            != example.action
        ):
            raise ValueError(f"Qwen2.5 sequential example {index} current label mismatches")


def validate_screenspot_pro_qwen25_direct_training_contract(
    examples: Iterable[SftExample],
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> None:
    """Reject any direct-click training input that drifts from upstream Qwen2.5."""

    profile = get_groundcua_profile(QWEN25_DIRECT_PROFILE)
    if image_min_pixels != profile.image_min_pixels:
        raise ValueError(
            "ScreenSpot-Pro Qwen2.5 direct image_min_pixels must be "
            f"{profile.image_min_pixels}"
        )
    if image_max_pixels != profile.image_max_pixels:
        raise ValueError(
            "ScreenSpot-Pro Qwen2.5 direct image_max_pixels must be "
            f"{profile.image_max_pixels}"
        )
    for index, example in enumerate(examples):
        validate_groundcua_profile_example(example, profile)
        if example.context.coordinate_format != profile.coordinate_format:
            raise ValueError(f"Qwen2.5 direct example {index} has the wrong context coordinate format")
        if example.context.thought_required or example.thought is not None or example.thought_history:
            raise ValueError(f"Qwen2.5 direct example {index} contains hidden reasoning")
        if example.context.allowed_kinds != ("left_click",):
            raise ValueError(f"Qwen2.5 direct example {index} has the wrong action space")
        x = example.action.get("x")
        y = example.action.get("y")
        if (
            example.action.get("kind") != "left_click"
            or isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            raise ValueError(f"Qwen2.5 direct example {index} is not one integer left_click")
        expected_label = qwen25_direct_tool_call(x, y)
        if example.assistant_response != expected_label:
            raise ValueError(
                f"Qwen2.5 direct example {index} must train one complete tool call"
            )


def validate_screenspot_pro_vllm_training_contract(
    examples: Iterable[SftExample],
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> None:
    if image_min_pixels != SCREENSPOT_PRO_QWEN3VL_IMAGE_MIN_PIXELS:
        raise ValueError(
            "ScreenSpot-Pro Qwen3-VL VLLM image_min_pixels must be "
            f"{SCREENSPOT_PRO_QWEN3VL_IMAGE_MIN_PIXELS}"
        )
    if image_max_pixels != SCREENSPOT_PRO_QWEN3VL_IMAGE_MAX_PIXELS:
        raise ValueError(
            "ScreenSpot-Pro Qwen3-VL VLLM image_max_pixels must be "
            f"{SCREENSPOT_PRO_QWEN3VL_IMAGE_MAX_PIXELS}"
        )
    for index, example in enumerate(examples):
        if (
            example.coordinate_format != QWEN3_RELATIVE_COORDINATE_FORMAT
            or example.context.coordinate_format != QWEN3_RELATIVE_COORDINATE_FORMAT
        ):
            raise ValueError(f"ScreenSpot-Pro example {index} has the wrong coordinate format")
        points = example.action.get("points")
        if (
            example.action.get("kind") != "click"
            or not isinstance(points, list)
            or len(points) != 1
            or not isinstance(points[0], list)
            or len(points[0]) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in points[0])
            or any(value < 0 or value > 1000 for value in points[0])
        ):
            raise ValueError(f"ScreenSpot-Pro example {index} is not one [0,1000] click")
        response_point = parse_screenspot_pro_qwen3vl_tool_call(
            example.assistant_response or ""
        )
        if response_point != tuple(points[0]):
            raise ValueError(
                f"ScreenSpot-Pro example {index} assistant response does not match its click"
            )
        build_screenspot_pro_qwen3vl_vllm_prompt(example.instruction)


def split_examples(
    examples: list[SftExample],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[SftExample], list[SftExample]]:
    if eval_fraction <= 0:
        return examples, []
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[eval_count:], shuffled[:eval_count]


def _install_legacy_rng_state_loader(trainer: Any) -> None:
    # Transformers 5.9 uses weights_only=True for RNG restoration. The trusted
    # legacy checkpoint contains Python/NumPy objects, so use full pickle only
    # for this explicit resume path.
    if not hasattr(trainer, "_load_rng_state"):
        return

    import numpy as np
    import torch
    from transformers.trainer import ParallelMode, set_rng_state_for_device

    original_load_rng_state = trainer._load_rng_state

    def load_rng_state(checkpoint: str | None) -> None:
        if checkpoint is None:
            return
        if trainer.args.world_size > 1:
            process_index = trainer.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        if not os.path.isfile(rng_file):
            original_load_rng_state(checkpoint)
            return

        checkpoint_rng_state = torch.load(
            rng_file,
            map_location="cpu",
            weights_only=False,
        )
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        is_distributed = trainer.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device(
                "CUDA",
                torch.cuda,
                checkpoint_rng_state,
                is_distributed,
            )

    trainer._load_rng_state = load_rng_state


def _train_with_optional_resume(trainer: Any, resume_from_checkpoint: Path | None) -> None:
    if resume_from_checkpoint is None:
        trainer.train()
        return
    _install_legacy_rng_state_loader(trainer)
    trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))


def train(
    *,
    train_manifests: list[Path],
    output_dir: Path,
    resume_from_checkpoint: Path | None = None,
    model_name_or_path: Path | str | None = None,
    eval_manifests: list[Path] | None = None,
    eval_fraction: float = 0.0,
    seed: int = 42,
    max_examples: int | None = None,
    skip_missing_images: bool = True,
    num_train_epochs: float = 1.5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 7e-6,
    warmup_ratio: float = 0.08,
    weight_decay: float = 0.05,
    lr_scheduler_type: str = "cosine",
    max_length: int = 4096,
    image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS,
    image_max_pixels: int | None = FULL_HD_IMAGE_MAX_PIXELS,
    deepspeed_config: Path | None = None,
    gradient_checkpointing: bool = True,
    bf16: bool = True,
    device_map: str | None = None,
    attn_implementation: str | None = "flash_attention_2",
    save_total_limit: int | None = 2,
    lora_r: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    context_total_actions: int | None = None,
    max_steps: int | None = None,
    save_steps: int | None = None,
    eval_steps: int | None = None,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int | None = None,
    dataloader_persistent_workers: bool = False,
    dataloader_pin_memory: bool = True,
    prompt_profile: str = DEFAULT_PROMPT_PROFILE,
    dry_run: bool = False,
) -> dict[str, Any]:
    if (
        prompt_profile in _SEQUENTIAL_MOVE_PROFILES
        and IMAGE_HISTORY_MAX != SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX
    ):
        raise ValueError(
            "sequential move training requires QWEN3_VL_IMAGE_HISTORY_MAX=3"
        )
    if dataloader_num_workers < 0:
        raise ValueError("dataloader_num_workers must be >= 0")
    if dataloader_num_workers == 0 and dataloader_persistent_workers:
        raise ValueError("dataloader_persistent_workers requires dataloader_num_workers > 0")
    if dataloader_num_workers == 0 and dataloader_prefetch_factor is not None:
        raise ValueError("dataloader_prefetch_factor requires dataloader_num_workers > 0")
    if model_name_or_path is None:
        model_name_or_path = default_model_path()

    examples = with_context_total_actions_for_examples(
        load_sft_examples(
            train_manifests,
            max_examples=max_examples,
            skip_missing_images=skip_missing_images,
        ),
        context_total_actions,
    )
    if eval_manifests:
        eval_examples = with_context_total_actions_for_examples(
            load_sft_examples(eval_manifests, skip_missing_images=skip_missing_images),
            context_total_actions,
        )
        train_examples = examples
    else:
        train_examples, eval_examples = split_examples(examples, eval_fraction=eval_fraction, seed=seed)

    if prompt_profile == SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE:
        validate_screenspot_pro_vllm_training_contract(
            (*train_examples, *eval_examples),
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
    if prompt_profile == QWEN25_DIRECT_PROFILE:
        validate_screenspot_pro_qwen25_direct_training_contract(
            (*train_examples, *eval_examples),
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )
    if prompt_profile == QWEN3_MOVE_PROFILE:
        validate_screenspot_pro_groundcua_sequential_training_contract(
            (*train_examples, *eval_examples)
        )
    if prompt_profile == QWEN25_MOVE_PROFILE:
        validate_screenspot_pro_qwen25_sequential_training_contract(
            (*train_examples, *eval_examples)
        )

    coordinate_contract = (
        get_groundcua_profile(prompt_profile).coordinate_format
        if is_screenspot_pro_groundcua_profile(prompt_profile)
        else COORDINATE_CONTRACT
    )

    summary = {
        "model_name_or_path": str(model_name_or_path),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "output_dir": str(output_dir),
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
        "finetune_mode": "lora" if lora_r > 0 else "full",
        "deepspeed_config": str(deepspeed_config) if deepspeed_config else None,
        "dry_run": dry_run,
        "prompt_family": prompt_profile,
        "prompt_profile": prompt_profile,
        "coordinate_contract": coordinate_contract,
        "image_max_pixels": image_max_pixels,
        "image_min_pixels": image_min_pixels,
        "image_history_max": IMAGE_HISTORY_MAX,
        "sequential_move_image_history_max": (
            SEQUENTIAL_MOVE_IMAGE_HISTORY_MAX
            if prompt_profile in _SEQUENTIAL_MOVE_PROFILES
            else None
        ),
        "sequential_move_prompt_contract": (
            QWEN3_MOVE_SEQUENTIAL_PROMPT_CONTRACT
            if prompt_profile == QWEN3_MOVE_PROFILE
            else (
                QWEN25_SEQUENTIAL_PROMPT_CONTRACT
                if prompt_profile == QWEN25_MOVE_PROFILE
                else None
            )
        ),
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "lr_scheduler_type": lr_scheduler_type,
        "save_total_limit": save_total_limit,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": list(lora_target_modules),
        "context_total_actions": context_total_actions,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "eval_steps": eval_steps,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_prefetch_factor": dataloader_prefetch_factor,
        "dataloader_persistent_workers": dataloader_persistent_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
    }
    if prompt_profile == QWEN25_DIRECT_PROFILE:
        summary.update(
            {
                "training_contract_schema": QWEN25_DIRECT_TRAINING_CONTRACT_SCHEMA,
                "assistant_contract": QWEN25_DIRECT_ASSISTANT_CONTRACT,
                "assistant_generation_boundary": "<|im_start|>assistant\n",
                "assistant_prefill": False,
            }
        )
    if dry_run:
        return summary

    if not train_examples:
        raise ValueError("No training examples were loaded.")

    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Real Qwen3-VL full SFT requires torch, transformers, and pillow."
        ) from exc

    processor = AutoProcessor.from_pretrained(str(model_name_or_path), trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(str(model_name_or_path), **model_kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    use_lora = lora_r > 0
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "LoRA SFT requires peft. Install it in the training environment or set LORA_R=0."
            ) from exc
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(lora_target_modules),
        )
        model = get_peft_model(model, lora_config)
        print("LoRA SFT enabled.")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        print("Full-parameter SFT enabled: all trainable model parameters will be updated.")

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        bf16=bf16,
        logging_steps=10,
        save_strategy="steps" if save_steps is not None else "epoch",
        save_steps=save_steps or 500,
        eval_strategy=(
            "steps"
            if eval_examples and eval_steps is not None
            else "epoch"
            if eval_examples
            else "no"
        ),
        eval_steps=eval_steps or 500,
        deepspeed=str(deepspeed_config) if deepspeed_config else None,
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
        save_total_limit=save_total_limit,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_pin_memory=dataloader_pin_memory,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Qwen3VLSftDataset(train_examples),
        eval_dataset=Qwen3VLSftDataset(eval_examples) if eval_examples else None,
        data_collator=Qwen3VLCollator(
            processor,
            max_length=max_length,
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
            prompt_profile=prompt_profile,
        ),
    )
    _train_with_optional_resume(trainer, resume_from_checkpoint)
    if use_lora and hasattr(trainer.model, "merge_and_unload"):
        trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(output_dir))
        if prompt_profile == QWEN25_DIRECT_PROFILE:
            write_qwen25_direct_training_contract(output_dir)
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT for Qwen3-VL on converted GUI-agent data.")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, action="append", default=None)
    parser.add_argument("--eval-manifest", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--epochs", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--eval-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--image-max-pixels", type=int, default=FULL_HD_IMAGE_MAX_PIXELS)
    parser.add_argument("--image-min-pixels", type=int, default=DEFAULT_IMAGE_MIN_PIXELS)
    parser.add_argument(
        "--no-image-min-pixels",
        action="store_const",
        const=None,
        dest="image_min_pixels",
        help="Disable the Qwen smart-resize lower pixel bound; keep only image_max_pixels.",
    )
    parser.add_argument("--deepspeed", type=Path, dest="deepspeed_config")
    parser.add_argument("--device-map")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--eval-steps", type=int)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-prefetch-factor", type=int)
    parser.add_argument("--dataloader-persistent-workers", action="store_true")
    parser.add_argument(
        "--prompt-profile",
        default=DEFAULT_PROMPT_PROFILE,
        choices=sorted(
            {
                DEFAULT_PROMPT_PROFILE,
                QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE,
                SCREENSPOT_PRO_QWEN3VL_VLLM_PROMPT_PROFILE,
                *(
                    get_groundcua_profile(name).name
                    for name in (
                        "screenspot_pro_qwen25_direct_dbe00114",
                        "screenspot_pro_qwen25_move_dbe00114",
                        "screenspot_pro_qwen3_direct_dbe00114",
                        "screenspot_pro_qwen3_move_dbe00114",
                    )
                ),
            }
        ),
        help="Select the GroundCUA prompt contract used to build the training messages.",
    )
    parser.add_argument(
        "--no-dataloader-pin-memory",
        action="store_false",
        dest="dataloader_pin_memory",
        default=True,
    )
    parser.add_argument(
        "--context-total-actions",
        type=int,
        help="Override total action budget shown in SFT closed-loop context.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16 instead of bf16.")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--fail-on-missing-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = train(
        train_manifests=args.train_manifest or [default_train_manifest()],
        eval_manifests=args.eval_manifest or [default_eval_manifest()],
        output_dir=args.output_dir or default_checkpoint_output_dir(),
        resume_from_checkpoint=args.resume_from_checkpoint,
        model_name_or_path=args.model or default_model_path(),
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        max_examples=args.max_examples,
        skip_missing_images=not args.fail_on_missing_images,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_length=args.max_length,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        deepspeed_config=args.deepspeed_config,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        bf16=not args.fp16,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        save_total_limit=args.save_total_limit,
        context_total_actions=args.context_total_actions,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        prompt_profile=args.prompt_profile,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
