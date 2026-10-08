from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from ..core import Observation, StepResult
from ..benchmarks.exploration_depth.training_no_think import SIX_ACTION_KINDS
from .exploration_depth_no_think import build_exploration_depth_no_think_eval_prompt
from .exploration_depth_with_think import Qwen35ExplorationDepthWithThinkLocalBackend
from .qwen3_vl_sft_local import ImageCoordinateTransform, SftPromptBuildResult


ROUTE_HEADER = "X-GUI-Captcha-Route-Key"
BACKEND_HEADER = "X-GUI-Captcha-Backend"


def _chat_completions_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"invalid vLLM base URL: {base_url!r}")
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path += "/chat/completions"
    else:
        path += "/v1/chat/completions"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _media_uuid(route_key: str, path: Path, transform: ImageCoordinateTransform) -> str:
    if not route_key:
        raise ValueError("route_key must be set before vLLM inference")
    width, height = transform.model_size
    return f"gui-captcha/{route_key}/{path.resolve().as_posix()}/{width}x{height}"


@dataclass
class Qwen35ExplorationDepthWithThinkVLLMBackend(
    Qwen35ExplorationDepthWithThinkLocalBackend
):
    server_base_url: str = "http://127.0.0.1:18100/v1"
    served_model_name: str = "gui-captcha-qwen35"
    request_timeout_s: float = 600.0
    route_key: str = ""
    last_vllm_usage: dict[str, int] = field(default_factory=dict, init=False)
    last_vllm_backend: str | None = field(default=None, init=False)
    last_vllm_request_latency_s: float | None = field(default=None, init=False)

    def _ensure_loaded(self) -> None:
        if self.processor is None:
            from transformers import AutoProcessor

            processor_source = self.processor_checkpoint_path or self.checkpoint_path
            self.processor = AutoProcessor.from_pretrained(
                str(processor_source),
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
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
        if not self.route_key:
            raise ValueError("route_key must be set before vLLM inference")

        image_history_paths = self._build_image_history(obs, history)
        prompt_build = self._build_prompt(
            obs,
            history,
            image_history_paths=image_history_paths,
            allowed_kinds=allowed_kinds,
            budget=budget,
        )
        self.last_prompt_messages = prompt_build.messages
        self.last_prompt_image_paths = tuple(str(path) for path in prompt_build.image_paths)
        self.last_prompt_text = self.processor.apply_chat_template(
            prompt_build.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

        transforms: list[ImageCoordinateTransform] = []
        server_messages: list[dict[str, Any]] = []
        for message in prompt_build.messages:
            server_content: list[dict[str, Any]] = []
            for item in message["content"]:
                if item["type"] == "text":
                    server_content.append({"type": "text", "text": item["text"]})
                    continue
                if item["type"] != "image":
                    raise ValueError(f"unsupported prompt content type: {item['type']!r}")
                path = Path(item["image"])
                image, transform = self._load_image_with_transform(path)
                transforms.append(transform)
                server_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                        "uuid": _media_uuid(self.route_key, path, transform),
                    }
                )
            server_messages.append({"role": message["role"], "content": server_content})
        if not transforms:
            raise ValueError("vLLM request has no image")
        self.last_image_transforms = tuple(
            {
                "original_size": list(item.original_size),
                "model_size": list(item.model_size),
                "resized": item.original_size != item.model_size,
            }
            for item in transforms
        )

        payload = {
            "model": self.served_model_name,
            "messages": server_messages,
            "max_tokens": self.max_new_tokens,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        request = urllib.request.Request(
            _chat_completions_url(self.server_base_url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer EMPTY",
                ROUTE_HEADER: self.route_key,
            },
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout_s) as response:
                response_payload = json.loads(response.read())
                self.last_vllm_backend = response.headers.get(BACKEND_HEADER)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise RuntimeError(f"vLLM HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"vLLM request failed: {error.reason}") from error
        finally:
            self.last_vllm_request_latency_s = time.monotonic() - started

        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM response has no choices")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("vLLM response has no raw assistant content")
        usage = response_payload.get("usage")
        self.last_vllm_usage = {
            key: int(value)
            for key, value in (usage.items() if isinstance(usage, dict) else ())
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and isinstance(value, (int, float))
        }
        self.last_input_token_count = self.last_vllm_usage.get("prompt_tokens")
        return content.strip(), transforms[-1]

    def _write_call_log(self, **kwargs: Any) -> None:
        super()._write_call_log(**kwargs)
        if self.call_log_dir is None:
            return
        path = self.call_log_dir / f"call_{self._call_index:06d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["serving_backend"] = {
            "type": "sticky_local_vllm",
            "served_model_name": self.served_model_name,
            "route_key": self.route_key,
            "backend_index": self.last_vllm_backend,
            "request_latency_s": self.last_vllm_request_latency_s,
            "usage": self.last_vllm_usage,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


@dataclass
class Qwen35ExplorationDepthNoThinkVLLMBackend(
    Qwen35ExplorationDepthWithThinkVLLMBackend
):
    """vLLM backend matching the six-variant empty-Think training prompt."""

    enable_thinking: bool = False

    def _build_prompt(
        self,
        obs: Observation,
        history: list[StepResult],
        *,
        image_history_paths: list[Path],
        allowed_kinds: Iterable[str],
        budget: int | None,
    ) -> SftPromptBuildResult:
        allowed = tuple(allowed_kinds)
        if allowed != SIX_ACTION_KINDS:
            raise ValueError(
                "exploration-depth evaluation requires the common six-action space; "
                f"got {allowed}"
            )
        return build_exploration_depth_no_think_eval_prompt(
            obs=obs,
            history=history,
            image_history_paths=image_history_paths,
            images_to_keep=self.image_history_max,
            budget=budget,
        )
