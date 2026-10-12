"""Shared validation for external Qwen3-VL Hugging Face checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_hf_checkpoint(checkpoint: Path) -> dict[str, Any]:
    """Validate a complete external HF checkpoint without hashing files."""

    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory missing: {root}")
    required = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    processor_assets = ("processor_config.json", "preprocessor_config.json")
    if not any((root / name).is_file() for name in processor_assets):
        missing.append("processor_config.json|preprocessor_config.json")
    chat_template_assets = ("chat_template.jinja", "chat_template.json")
    if not any((root / name).is_file() for name in chat_template_assets):
        missing.append("chat_template.jinja|chat_template.json")
    if missing:
        raise FileNotFoundError("checkpoint missing: " + ", ".join(missing))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be an object")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("checkpoint model_type must be qwen3_vl")
    if "Qwen3VLForConditionalGeneration" not in config.get("architectures", []):
        raise ValueError("checkpoint architecture must be Qwen3VLForConditionalGeneration")
    chat_template_file = next(
        name for name in chat_template_assets if (root / name).is_file()
    )
    if chat_template_file.endswith(".json"):
        chat_template_config = json.loads(
            (root / chat_template_file).read_text(encoding="utf-8")
        )
        if (
            not isinstance(chat_template_config, dict)
            or not isinstance(chat_template_config.get("chat_template"), str)
            or not chat_template_config["chat_template"].strip()
        ):
            raise ValueError("chat_template.json must contain a non-empty chat_template string")
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        shards = sorted({str(name) for name in weight_map.values()})
    else:
        shards = ["model.safetensors"]
    missing_shards = [name for name in shards if not (root / name).is_file()]
    empty_shards = [
        name for name in shards if (root / name).is_file() and (root / name).stat().st_size == 0
    ]
    if missing_shards or empty_shards:
        raise FileNotFoundError(
            f"checkpoint shard validation failed; missing={missing_shards}, empty={empty_shards}"
        )
    return {
        "status": "valid",
        "path": str(root),
        "model_type": config["model_type"],
        "architecture": "Qwen3VLForConditionalGeneration",
        "processor_config": next(
            name for name in processor_assets if (root / name).is_file()
        ),
        "chat_template": chat_template_file,
        "weight_shards": len(shards),
        "weight_map_entries": len(weight_map) if weight_map is not None else None,
    }
