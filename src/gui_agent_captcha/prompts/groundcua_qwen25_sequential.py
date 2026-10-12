from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .screenspot_pro_groundcua import (
    CUSTOM_QWEN25_ACTION_CONTRACT,
    QWEN25_MOVE_ACTIONS,
    QWEN25_MOVE_PROFILE,
    reject_qwen25_contract,
)

QWEN25_SEQUENTIAL_ACTIONS = QWEN25_MOVE_ACTIONS
QWEN25_SEQUENTIAL_IMAGE_HISTORY_MAX = 3
QWEN25_SEQUENTIAL_PROMPT_CONTRACT = "qwen25_official_messages_sequential_v1"


@dataclass(frozen=True)
class Qwen25SequentialRender:
    messages: list[dict[str, Any]]
    image_paths: tuple[Path, ...]
    prompt_text: str
    full_text: str | None
    target_text: str | None


def render_qwen25_sequential(
    processor: Any,
    *,
    instruction: str,
    image_paths: Sequence[Path],
    assistant_response_history: Sequence[str],
    current_screen_size: tuple[int, int],
    target_response: str | None = None,
    images_to_keep: int = QWEN25_SEQUENTIAL_IMAGE_HISTORY_MAX,
) -> Qwen25SequentialRender:
    del (
        processor,
        instruction,
        image_paths,
        assistant_response_history,
        current_screen_size,
        target_response,
        images_to_keep,
    )
    reject_qwen25_contract(
        profile_name=QWEN25_MOVE_PROFILE,
        action_contract=CUSTOM_QWEN25_ACTION_CONTRACT,
        reason=(
            "the multi-turn Think renderer is custom; the official entrypoint "
            "requires one coordinate-bearing direct left_click"
        ),
    )
