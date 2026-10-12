from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pyarrow.parquet as pq
from verl.utils.dataset.rl_dataset import RLHFDataset


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_builtin(item) for item in value]
    return value


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")


def _drop_none_values(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _sanitize_structured_content(content: list[Any]) -> list[Any]:
    sanitized = []
    for item in content:
        if not isinstance(item, dict):
            sanitized.append(item)
            continue

        item = _drop_none_values(item)
        kind = item.get("type")
        if kind == "image":
            image = item.get("image") or item.get("image_url")
            if image is None and "bytes" not in item:
                continue
            payload = {"type": "image"}
            if image is not None:
                payload["image"] = image
            if "bytes" in item:
                payload["bytes"] = item["bytes"]
            for key, value in item.items():
                if key not in {"type", "image", "image_url", "bytes"}:
                    payload[key] = value
            sanitized.append(payload)
        elif kind == "text":
            text = item.get("text")
            if text is not None:
                sanitized.append({"type": "text", "text": str(text)})
        else:
            sanitized.append(item)
    return sanitized


def _sanitize_structured_messages(messages: list[Any]) -> list[Any]:
    sanitized = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        message = _drop_none_values(message)
        content = message.get("content")
        if isinstance(content, list):
            message = dict(message)
            message["content"] = _sanitize_structured_content(content)
        sanitized.append(message)
    return sanitized


class LocalFileRLHFDataset(RLHFDataset):
    """RLHFDataset variant that reads local files without datasets.load_dataset.

    The current verl smoke environment has datasets 2.14.4 with fsspec 2026.4.0,
    which raises a LocalFileSystem cache error inside datasets.load_dataset().
    Reading the local parquet/json files directly keeps the rest of RLHFDataset
    behavior unchanged.
    """

    def _build_messages(self, example: dict[str, Any], key: str):
        messages = _to_builtin(example[key])
        if not isinstance(messages, list):
            raise TypeError(f"{key} must be a list of chat messages")

        # The CAPTCHA RL export already stores Qwen-style multimodal content
        # lists, e.g. {"content": [{"type": "image", ...}, {"type": "text", ...}]}.
        # verl's default implementation is for string prompts with <image>
        # placeholders plus a separate images column; applying it here would
        # try to consume images twice.
        if any(isinstance(message, dict) and isinstance(message.get("content"), list) for message in messages):
            return _sanitize_structured_messages(messages)

        return super()._build_messages(_to_builtin(example), key=key)

    @classmethod
    def process_vision_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[list[Any] | None, list[Any] | None]:
        from qwen_vl_utils import process_vision_info

        outputs = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_metadata=True,
        )
        if len(outputs) == 3:
            images, videos, _video_kwargs = outputs
        else:
            images, videos = outputs
        return images, videos

    @classmethod
    def _extract_audio_info(cls, messages: list[dict[str, Any]]) -> list[Any] | None:
        audios: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audios.append(item["audio"])
                elif "audio_url" in item:
                    audios.append(item["audio_url"])
                else:
                    audios.append({key: value for key, value in item.items() if key != "type"})
        return audios or None

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(
                isinstance(item, dict) and item.get("type") in {"image", "video"}
                for item in message["content"]
            )
            for message in messages
        )
        if has_visual:
            images, videos = cls.process_vision_info(
                messages,
                image_patch_size=image_patch_size,
                config=config,
            )
        else:
            images, videos = None, None
        audios = cls._extract_audio_info(messages)
        return images, videos, audios

    @classmethod
    async def process_multi_modal_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        return cls._process_multi_modal_info(
            messages,
            image_patch_size=image_patch_size,
            config=config,
        )

    def _read_files_and_tokenize(self) -> None:
        dataframes = []
        for data_file in self.data_files:
            path = Path(data_file)
            if path.suffix == ".parquet":
                records = pq.read_table(path).to_pylist()
            elif path.suffix in {".json", ".jsonl"}:
                records = _read_json_records(path)
            else:
                raise ValueError(f"Unsupported file format: {data_file}")

            records = [_to_builtin(record) for record in records]
            dataframes.append(datasets.Dataset.from_list(records))

        if not dataframes:
            raise ValueError("No data files were provided")
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rng_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rng_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
