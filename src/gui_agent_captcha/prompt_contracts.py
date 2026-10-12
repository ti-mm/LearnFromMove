from __future__ import annotations

from typing import Iterable

from .prompts.unified_three_action import UNIFIED_THREE_ACTION_SYSTEM_PROMPT

RAW_UNIFIED_ACTION_REASONING_CONTRACT = UNIFIED_THREE_ACTION_SYSTEM_PROMPT


def render_raw_unified_contract_block(*, action_kinds: Iterable[str] | None = None) -> str:
    """Compatibility wrapper for retired prompt-contract imports.

    ``action_kinds`` is intentionally ignored. The repository has one runtime
    prompt contract: the three-action ``computer_use`` tool-call prompt.
    """

    del action_kinds
    return UNIFIED_THREE_ACTION_SYSTEM_PROMPT
