from __future__ import annotations

import copy
import json
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..actions import Action, AtomicAction, PrimitiveAction
from ..core import Observation, StepResult
from ..protocol_tracks import split_think_json_response
from ..train.qwen3_vl_sft import (
    COORDINATE_CONTRACT,
    DEFAULT_IMAGE_MIN_PIXELS,
    FULL_HD_IMAGE_MAX_PIXELS,
    IMAGE_HISTORY_MAX,
    PROMPT_FAMILY,
    QWEN3_RELATIVE_COORDINATE_FORMAT,
    ImageCoordinateTransform,
    SftPromptBuildResult,
    build_prompt_messages_from_history,
    deterministic_generation_kwargs,
    qwen3_vl_multimodal_processor_kwargs,
    resize_image_for_pixel_range,
)
from .openai_chat import (
    OnlineSftObservationHistory,
    is_center_locked_cursor_task,
    parse_action_blob,
)


class ModelActionParseError(ValueError):
    """The model response does not satisfy the strict action protocol."""


@dataclass
class Qwen3VLSftLocalBackend:
    """Policy backend for a local Qwen3-VL SFT checkpoint.

    This uses the same SFT prompt builder as training, with closed-loop
    context for the current cursor, action history, allowed actions, and
    remaining budget.
    """

    checkpoint_path: Path
    processor_checkpoint_path: Path | None = None
    processor: Any | None = None
    model: Any | None = None
    max_new_tokens: int = 128
    max_length: int = 4096
    image_min_pixels: int | None = DEFAULT_IMAGE_MIN_PIXELS
    image_max_pixels: int | None = FULL_HD_IMAGE_MAX_PIXELS
    image_history_max: int = IMAGE_HISTORY_MAX
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str | None = None
    local_files_only: bool = True
    cuda_dtype_name: str = "bfloat16"
    call_log_dir: Path | None = None
    enable_thinking: bool = True
    _call_index: int = field(default=0, init=False)
    _sft_observation_history: OnlineSftObservationHistory = field(
        default_factory=OnlineSftObservationHistory,
        init=False,
    )
    last_prediction_index: int | None = field(default=None, init=False)
    last_raw_prediction: str | None = field(default=None, init=False)
    last_think_text: str | None = field(default=None, init=False)
    last_prompt_messages: list[dict[str, Any]] | None = field(default=None, init=False)
    last_prompt_text: str | None = field(default=None, init=False)
    last_prompt_image_paths: tuple[str, ...] = field(default_factory=tuple, init=False)
    last_image_transforms: tuple[dict[str, Any], ...] = field(
        default_factory=tuple,
        init=False,
    )
    last_input_token_count: int | None = field(default=None, init=False)
    generation_token_ids: dict[str, int | None] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # Preserve the historical max-only API: callers that pass a 720p or
        # smaller image_max_pixels without an explicit image_min_pixels must
        # not be forced into an exact 921600-pixel lower bound. 1280x720 does
        # not have a Qwen-28-aligned representation at exactly that area, so an
        # equal min/max pair would make every 720p inference fail before model
        # generation. The production default remains the shared
        # 921600..2073600 range.
        if (
            self.image_min_pixels == DEFAULT_IMAGE_MIN_PIXELS
            and self.image_max_pixels is not None
            and self.image_max_pixels <= DEFAULT_IMAGE_MIN_PIXELS
        ):
            self.image_min_pixels = None

    def _ensure_loaded(self) -> None:
        already_loaded = self.processor is not None and self.model is not None
        if already_loaded:
            if hasattr(self.model, "eval"):
                self.model.eval()
        else:
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Local Qwen3-VL SFT evaluation requires torch and transformers."
                ) from exc

            if self.processor is None:
                processor_source = self.processor_checkpoint_path or self.checkpoint_path
                self.processor = AutoProcessor.from_pretrained(
                    str(processor_source),
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=self.local_files_only,
                )
            if self.model is None:
                dtype = (
                    getattr(torch, self.cuda_dtype_name)
                    if torch.cuda.is_available()
                    else torch.float32
                )
                model_kwargs: dict[str, Any] = {
                    "dtype": dtype,
                    "device_map": self.device_map,
                    "trust_remote_code": self.trust_remote_code,
                    "local_files_only": self.local_files_only,
                }
                if self.attn_implementation is not None:
                    model_kwargs["attn_implementation"] = self.attn_implementation
                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(self.checkpoint_path),
                    **model_kwargs,
                )
        if hasattr(self.model, "eval"):
            self.model.eval()
        tokenizer = getattr(self.processor, "tokenizer", None)
        generation_config = getattr(self.model, "generation_config", None)
        eos_token_id = (
            getattr(tokenizer, "eos_token_id", None)
            or getattr(self.processor, "eos_token_id", None)
            or getattr(generation_config, "eos_token_id", None)
        )
        pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(self.processor, "pad_token_id", None)
            or getattr(generation_config, "pad_token_id", None)
        )
        if pad_token_id is None:
            pad_token_id = eos_token_id
        generation_token_ids: dict[str, int] = {}
        if eos_token_id is not None:
            generation_token_ids["eos_token_id"] = int(eos_token_id)
        if pad_token_id is not None:
            generation_token_ids["pad_token_id"] = int(pad_token_id)
        # Some lightweight/legacy processors delegate token IDs entirely to the
        # model generation config. Do not fail before prompt/history checks in
        # that compatible mode; real Qwen processors provide IDs above.
        self.generation_token_ids = generation_token_ids

    def _load_image_with_transform(self, image_path: Path) -> tuple[Image.Image, ImageCoordinateTransform]:
        image = Image.open(image_path).convert("RGB")
        if self.image_min_pixels is None:
            from ..train.qwen3_vl_sft import resize_image_for_max_pixels
            return resize_image_for_max_pixels(image, self.image_max_pixels)
        return resize_image_for_pixel_range(
            image,
            image_min_pixels=self.image_min_pixels,
            image_max_pixels=self.image_max_pixels,
        )

    def _load_image(self, image_path: Path) -> Image.Image:
        image, _transform = self._load_image_with_transform(image_path)
        return image

    def _history_action_dicts(self, history: list[StepResult]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for step in history:
            if step.action is None:
                continue
            action_dict = step.action.to_dict()
            if action_dict.get("kind") != "move_to":
                action_dict.pop("x", None)
                action_dict.pop("y", None)
            actions.append(action_dict)
        return actions

    def _history_observation_snapshots(self, history: list[StepResult]) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for step in history:
            if step.action is None:
                continue
            snapshots.append(
                {
                    "screenshot_path": step.observation.screenshot_path,
                    "cursor_xy": list(step.observation.cursor_xy)
                    if step.observation.cursor_xy is not None
                    else None,
                    "metadata_task_id": (step.observation.metadata or {}).get("task_id"),
                }
            )
        return snapshots

    def _history_thoughts(self, history: list[StepResult]) -> list[str | None]:
        thoughts: list[str | None] = []
        for step in history:
            if step.action is None:
                continue
            thoughts.append(_extract_history_thought(step.info))
        return thoughts

    def _cursor_xy_for_prompt(
        self,
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

    def _button_state_after_history(self, history: list[StepResult]) -> str:
        button_state = "up"
        for step in history:
            if step.action is None:
                continue
            if step.action.kind == "mouse_down":
                button_state = "down"
            elif step.action.kind == "mouse_up":
                button_state = "up"
        return button_state

    def _build_image_history(self, obs: Observation, history: list[StepResult]) -> list[Path]:
        entries = self._sft_observation_history.entries_for(
            obs,
            history,
            images_to_keep=max(self.image_history_max, len(history) + 1),
        )
        return [path for path, _cursor_xy in entries if path.is_file()]

    def _task_type_for_observation(self, obs: Observation) -> str | None:
        metadata = obs.metadata or {}
        for key in ("task_type", "benchmark", "task_id", "episode_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _build_prompt(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        image_history_paths: list[Path],
        allowed_kinds: Iterable[str],
        budget: int | None,
    ) -> SftPromptBuildResult:
        """Build one runtime prompt.

        Subclasses may override this hook for a frozen training-data prompt
        contract while reusing the model loading, generation, parsing, and
        logging implementation below.
        """
        return build_prompt_messages_from_history(
            instruction=obs.instruction,
            image_path=Path(obs.screenshot_path),
            image_paths=tuple(image_history_paths),
            action_history=self._history_action_dicts(history),
            thought_history=tuple(self._history_thoughts(history)),
            total_actions=budget,
            cursor_xy=self._cursor_xy_for_prompt(obs.cursor_xy, obs.size_px),
            button_state=self._button_state_after_history(history),
            task_type=self._task_type_for_observation(obs),
            allowed_kinds=allowed_kinds,
            budget_remaining=max(0, budget - len(history)) if budget is not None else None,
            coordinate_format=QWEN3_RELATIVE_COORDINATE_FORMAT,
            image_size_px=obs.size_px,
            images_to_keep=self.image_history_max,
        )

    def _generate_raw(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
    ) -> tuple[str, ImageCoordinateTransform]:
        assert self.processor is not None
        assert self.model is not None

        image_path = Path(obs.screenshot_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"SFT backend screenshot is missing: {image_path}")

        image_history_paths = self._build_image_history(obs, history)
        prompt_build = self._build_prompt(
            obs,
            history,
            image_history_paths=image_history_paths,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        prompt_image_paths = prompt_build.image_paths
        self.last_prompt_messages = prompt_build.messages
        self.last_prompt_image_paths = tuple(str(path) for path in prompt_image_paths)
        all_images: list[Image.Image] = []
        transforms: list[ImageCoordinateTransform] = []
        for img_path in prompt_image_paths:
            img, t = self._load_image_with_transform(img_path)
            all_images.append(img)
            transforms.append(t)
        if not all_images:
            image, transform = self._load_image_with_transform(image_path)
            all_images = [image]
            transforms = [transform]
        self.last_image_transforms = tuple(
            {
                "original_size": list(item.original_size),
                "model_size": list(item.model_size),
                "resized": item.original_size != item.model_size,
            }
            for item in transforms
        )
        transform = transforms[-1]

        prompt_text = self.processor.apply_chat_template(
            prompt_build.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        self.last_prompt_text = prompt_text
        batch = self.processor(
            text=[prompt_text],
            images=[all_images],
            **qwen3_vl_multimodal_processor_kwargs(
                padding=True,
                image_min_pixels=self.image_min_pixels,
                image_max_pixels=self.image_max_pixels,
            ),
        )
        input_ids = batch["input_ids"]
        input_len = int(input_ids.shape[-1])
        self.last_input_token_count = input_len
        if input_len > self.max_length:
            raise ValueError(
                "Qwen3-VL multimodal prompt exceeds max_length without truncation: "
                f"input tokens {input_len}, max_length {self.max_length}"
            )
        device = getattr(self.model, "device", None)
        if device is not None:
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }

        try:
            import torch
            no_grad = torch.no_grad()
        except ImportError:
            no_grad = nullcontext()
        with no_grad:
            generation_kwargs = deterministic_generation_kwargs(self.max_new_tokens)
            generation_kwargs.update(self.generation_token_ids)
            output_ids = self.model.generate(
                **batch,
                **generation_kwargs,
            )
        generated_ids = output_ids[0][input_len:]
        raw = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()
        return raw, transform

    def _normalize_model_action(
        self,
        action: Action,
        transform: ImageCoordinateTransform,
        obs_size: tuple[int, int],
    ) -> Action:
        del transform, obs_size
        if isinstance(action, PrimitiveAction) and action.kind == "move_to":
            assert action.x is not None and action.y is not None
            return PrimitiveAction(
                kind="move_to",
                x=min(1000, max(0, int(round(float(action.x))))),
                y=min(1000, max(0, int(round(float(action.y))))),
            )
        return action

    def _repair_center_locked_action(
        self,
        action: Action,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
    ) -> Action:
        allowed = set(allowed_kinds)
        if not is_center_locked_cursor_task(obs):
            return action
        if not isinstance(action, AtomicAction) or action.kind != "click" or "click" in allowed:
            return action
        if not action.points:
            return action
        point_x, point_y = action.points[0]
        if "left_click" in allowed and (
            _last_center_locked_state_is_centered(history)
            or _point_is_near_center(point_x, point_y)
        ):
            return PrimitiveAction(kind="left_click")
        if "mouse_down" in allowed and (
            _last_center_locked_state_is_centered(history)
            or _point_is_near_center(point_x, point_y)
        ):
            return PrimitiveAction(kind="mouse_down")
        if "move_to" in allowed:
            return PrimitiveAction(kind="move_to", x=point_x, y=point_y)
        if "mouse_down" in allowed:
            return PrimitiveAction(kind="mouse_down")
        return action

    def _write_call_log(
        self,
        *,
        obs: Observation,
        history: list[StepResult],
        allowed_kinds: Iterable[str],
        raw_prediction: str,
        parsed_action: Action | None,
        parse_error: str | None,
        budget: int | None,
        condition: str | None,
    ) -> None:
        if self.call_log_dir is None:
            return
        self.call_log_dir.mkdir(parents=True, exist_ok=True)
        history_action_snapshot = self._history_action_dicts(history)
        payload = {
            "checkpoint": str(self.checkpoint_path),
            "processor_checkpoint": str(
                self.processor_checkpoint_path or self.checkpoint_path
            ),
            "attn_implementation": self.attn_implementation,
            "cuda_dtype_name": self.cuda_dtype_name,
            "local_files_only": self.local_files_only,
            "instruction": obs.instruction,
            "screenshot_path": obs.screenshot_path,
            "observation_metadata": copy.deepcopy(obs.metadata or {}),
            "history_length": len(history),
            "history_action_snapshot": history_action_snapshot,
            "history_observation_snapshot": self._history_observation_snapshots(history),
            "allowed_kinds": list(allowed_kinds),
            "budget": budget,
            "condition": condition,
            "prompt_family": PROMPT_FAMILY,
            "coordinate_contract": COORDINATE_CONTRACT,
            "prompt_messages": copy.deepcopy(self.last_prompt_messages),
            "prompt_text": self.last_prompt_text,
            "prompt_user_text": _prompt_user_text(self.last_prompt_messages),
            "prompt_image_paths": list(self.last_prompt_image_paths),
            "image_transforms": list(self.last_image_transforms),
            "input_token_count": self.last_input_token_count,
            "generation_config": {
                "max_new_tokens": self.max_new_tokens,
                "max_length": self.max_length,
                "do_sample": False,
                "image_min_pixels": self.image_min_pixels,
                "image_max_pixels": self.image_max_pixels,
                "image_history_max": self.image_history_max,
                "enable_thinking": self.enable_thinking,
                **self.generation_token_ids,
            },
            "raw_prediction": raw_prediction,
            "think_text": self.last_think_text,
            "parsed_action": parsed_action.to_dict() if parsed_action is not None else None,
            "parse_error": parse_error,
        }
        path = self.call_log_dir / f"call_{self._call_index:06d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    def _begin_prediction(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None,
    ) -> tuple[str, ImageCoordinateTransform]:
        self._ensure_loaded()
        self._call_index += 1
        self.last_prediction_index = self._call_index
        self.last_raw_prediction = None
        self.last_think_text = None
        self.last_prompt_messages = None
        self.last_prompt_text = None
        self.last_prompt_image_paths = ()
        self.last_image_transforms = ()
        self.last_input_token_count = None
        raw_prediction, transform = self._generate_raw(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        self.last_raw_prediction = raw_prediction
        think_text, _json_blob = split_think_json_response(raw_prediction)
        if think_text is None:
            think_text = _recover_malformed_history_thought(raw_prediction)
        self.last_think_text = think_text
        return raw_prediction, transform

    def predict_raw(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> str:
        raw_prediction, _transform = self._begin_prediction(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        self._write_call_log(
            obs=obs,
            history=history,
            allowed_kinds=allowed_kinds,
            raw_prediction=raw_prediction,
            parsed_action=None,
            parse_error=None,
            budget=budget,
            condition=condition,
        )
        return raw_prediction

    def predict_action(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        allowed_kinds: Iterable[str],
        budget: int | None = None,
        condition: str | None = None,
    ) -> Action:
        raw_prediction, transform = self._begin_prediction(
            obs,
            history,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        parsed_action: Action | None = None
        parse_error: str | None = None
        try:
            try:
                raw_action = _parse_local_sft_action(
                    raw_prediction,
                    allowed_kinds=allowed_kinds,
                )
            except Exception as exc:
                raise ModelActionParseError(str(exc)) from exc
            repaired_action = self._repair_center_locked_action(
                raw_action,
                obs,
                history,
                allowed_kinds=allowed_kinds,
            )
            parsed_action = self._normalize_model_action(
                repaired_action,
                transform,
                obs.size_px,
            )
            return parsed_action
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._write_call_log(
                obs=obs,
                history=history,
                allowed_kinds=allowed_kinds,
                raw_prediction=raw_prediction,
                parsed_action=parsed_action,
                parse_error=parse_error,
                budget=budget,
                condition=condition,
            )


def _extract_history_thought(info: dict[str, Any]) -> str | None:
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
        recovered_thought = _recover_malformed_history_thought(value)
        if recovered_thought is not None:
            return recovered_thought
    return None


def _recover_malformed_history_thought(value: str) -> str | None:
    close_index = value.find("</think>")
    if close_index < 0:
        return None
    thought = value[:close_index].strip()
    if thought.startswith("<think>"):
        thought = thought[len("<think>") :].strip()
    return thought or None


def _prompt_user_text(messages: list[dict[str, Any]] | None) -> str | None:
    if not messages:
        return None
    parts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(str(item["text"]))
            elif item.get("type") == "image" and isinstance(item.get("image"), str):
                parts.append("<image>")
    return "\n".join(parts) if parts else None


def _last_center_locked_state_is_centered(history: list[StepResult]) -> bool:
    if not history:
        return False
    info = history[-1].info
    try:
        distance = float(info["target_distance_px"])
        hit_radius = float(info["hit_radius_px"])
    except (KeyError, TypeError, ValueError):
        return False
    return distance <= hit_radius


def _point_is_near_center(x: float, y: float, *, tolerance_bins: float = 28.0) -> bool:
    try:
        return ((float(x) - 500.0) ** 2 + (float(y) - 500.0) ** 2) ** 0.5 <= tolerance_bins
    except (TypeError, ValueError):
        return False


def _parse_local_sft_action(raw_prediction: str, *, allowed_kinds: Iterable[str]) -> Action:
    try:
        return parse_action_blob(raw_prediction)
    except Exception:
        legacy_move_to = _parse_legacy_move_to_points(
            raw_prediction,
            allowed_kinds=allowed_kinds,
        )
        if legacy_move_to is not None:
            return legacy_move_to
        raise


_LEGACY_MOVE_TO_POINT_RE = re.compile(
    r'"points"\s*:\s*\[+\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)',
    re.DOTALL,
)


def _parse_legacy_move_to_points(
    raw_prediction: str,
    *,
    allowed_kinds: Iterable[str],
) -> PrimitiveAction | None:
    allowed = set(allowed_kinds)
    if "move_to" not in allowed:
        return None
    if not re.search(r'"kind"\s*:\s*"move_to"', raw_prediction):
        return None
    match = _LEGACY_MOVE_TO_POINT_RE.search(raw_prediction)
    if match is None:
        return None
    try:
        return PrimitiveAction(
            kind="move_to",
            x=float(match.group(1)),
            y=float(match.group(2)),
        )
    except ValueError:
        return None
