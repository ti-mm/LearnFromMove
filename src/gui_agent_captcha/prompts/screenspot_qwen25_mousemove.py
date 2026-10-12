"""Qwen2.5 mouse-move/current-cursor-click training contract.

This is a project-specific training contract sharing the official Qwen2.5
generation boundary without an assistant response prefill.
"""

from __future__ import annotations

from .screenspot_qwen25_official_base import (
    OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT,
)


QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_PROFILE = (
    "groundcua46k_qwen25_mousemove_leftclick_shared_white_v1"
)
QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_CONTRACT = (
    "groundcua46k_qwen25_mousemove_current_cursor_click_no_tag_v1"
)
QWEN25_MOUSEMOVE_LEFTCLICK_ACTION_CONTRACT = (
    "mouse_move_resized_physical_pixels_then_coordinate_free_left_click"
)
QWEN25_MOUSEMOVE_LEFTCLICK_GUIDED_PROMPT = OFFICIAL_QWEN25_LEFTCLICK_GUIDED_PROMPT


__all__ = [
    "QWEN25_MOUSEMOVE_LEFTCLICK_ACTION_CONTRACT",
    "QWEN25_MOUSEMOVE_LEFTCLICK_GUIDED_PROMPT",
    "QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_CONTRACT",
    "QWEN25_MOUSEMOVE_LEFTCLICK_PROMPT_PROFILE",
]
