from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

import pandas as pd


def content_to_verl_text_and_images(
    content: Any,
    *,
    image_max_pixels: int,
    image_min_pixels: int | None = None,
    validate_image_files: bool = True,
) -> tuple[str, list[dict[str, Any]]]:
    """Flatten a chat content value into VERL text and ordered image records."""

    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    text_parts: list[str] = []
    images: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, Mapping):
            text_parts.append(str(item))
            continue

        kind = item.get("type")
        if kind == "text":
            text_parts.append(str(item.get("text", "")))
            continue
        if kind == "image":
            raw_path = item.get("image") or item.get("image_url")
            if raw_path is None:
                raise ValueError("image content item is missing image or image_url")
            if isinstance(raw_path, Mapping):
                raw_path = raw_path.get("url")
            if raw_path is None:
                raise ValueError("image content item has an empty image URL")
            if validate_image_files:
                path = Path(str(raw_path)).expanduser().resolve()
            else:
                path = Path(os.path.abspath(os.path.expanduser(str(raw_path))))
            if validate_image_files and not path.is_file():
                raise FileNotFoundError(f"image file not found: {path}")
            text_parts.append("<image>")
            image_record: dict[str, Any] = {
                "image": str(path),
                "max_pixels": image_max_pixels,
            }
            if image_min_pixels is not None:
                image_record["min_pixels"] = image_min_pixels
            images.append(image_record)
            continue
        text_parts.append(str(item))

    return "".join(text_parts), images


def messages_to_verl_row(
    messages: Sequence[Mapping[str, Any]],
    *,
    image_max_pixels: int,
    image_min_pixels: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    validate_image_files: bool = True,
) -> dict[str, Any]:
    """Convert canonical multimodal messages into one VERL SFT row."""

    if image_max_pixels < 1:
        raise ValueError("image_max_pixels must be positive")

    # ``trainable`` is an internal dataset annotation consumed by the
    # GroundCUA window dataset.  Keep it on assistant messages only; VERL's
    # chat-template path ignores unknown message keys while the custom loader
    # uses it to mask historical assistant turns.
    verl_messages: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for message in messages:
        text, message_images = content_to_verl_text_and_images(
            message.get("content", ""),
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
            validate_image_files=validate_image_files,
        )
        serialized_message: dict[str, Any] = {
            "role": str(message.get("role", "user")),
            "content": text,
        }
        if "trainable" in message:
            value = message["trainable"]
            if not isinstance(value, bool):
                raise TypeError("message trainable flag must be a boolean")
            if serialized_message["role"] != "assistant":
                raise ValueError("trainable flag is only valid on assistant messages")
            serialized_message["trainable"] = value
        if "assistant_prefill" in message:
            prefill = message["assistant_prefill"]
            if not isinstance(prefill, str) or not prefill:
                raise TypeError("assistant_prefill must be a nonempty string")
            if serialized_message["role"] != "assistant":
                raise ValueError("assistant_prefill is only valid on assistant messages")
            serialized_message["assistant_prefill"] = prefill
        verl_messages.append(serialized_message)
        images.extend(message_images)

    if not images:
        raise ValueError("VERL SFT row must contain at least one image")
    row_metadata = dict(metadata or {"row_schema_version": 1})
    return {
        "messages": verl_messages,
        "images": images,
        "tools": [],
        "metadata": row_metadata,
    }


def write_verl_parquet(rows: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    """Write VERL SFT rows to a Parquet file."""

    if not rows:
        raise ValueError("cannot write an empty VERL SFT dataset")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(output_path, index=False)
