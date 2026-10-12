from __future__ import annotations

from .screenspot_qwen25_official_base import (
    OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT,
    OFFICIAL_QWEN25_LEFTCLICK_PROMPT_PROFILE,
    OFFICIAL_QWEN25_SOURCE,
)
from .screenspot_pro_groundcua import (
    CUSTOM_QWEN25_ACTION_CONTRACT,
    QWEN25_MOVE_PROFILE,
    reject_qwen25_contract,
)


def build_official_qwen25_leftclick_prompt(
    instruction: str,
    *,
    screen_width: int,
    screen_height: int,
) -> str:
    return (
        OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT.replace(
            "{{screen_width}}",
            str(screen_width),
        )
        .replace("{{screen_height}}", str(screen_height))
        .replace("{{instruction}}", instruction)
    )


QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE = "screenspot_pro_qwen25_move_dbe00114"
if QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE != QWEN25_MOVE_PROFILE:
    raise RuntimeError("Qwen2.5 primitive profile drifted from the shared contract")


class _RemovedQwen25Prompt:
    def _raise(self) -> None:
        reject_qwen25_contract(
            profile_name=QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE,
            action_contract=CUSTOM_QWEN25_ACTION_CONTRACT,
            reason="the derived prompt template was removed and has no fallback",
        )

    def __getattr__(self, name: str):
        del name
        self._raise()

    def __str__(self) -> str:
        self._raise()
        return ""


# Compatibility attributes keep historical imports deterministic without
# retaining a usable derived prompt string.
QWEN25_MOUSE_PRIMITIVE_PROMPT = _RemovedQwen25Prompt()
QWEN25_MOUSE_PRIMITIVE_SYSTEM_PROMPT = QWEN25_MOUSE_PRIMITIVE_PROMPT


def build_qwen25_mouse_primitive_prompt(
    instruction: str,
    *,
    screen_width: int,
    screen_height: int,
) -> str:
    del instruction, screen_width, screen_height
    reject_qwen25_contract(
        profile_name=QWEN25_MOUSE_PRIMITIVE_PROMPT_PROFILE,
        action_contract=CUSTOM_QWEN25_ACTION_CONTRACT,
        reason="the derived prompt template was removed and has no fallback",
    )
