"""VERL dataset for GroundCUA sliding-window conversations.

GroundCUA windows retain historical assistant messages so the model sees the
complete dialogue, but only the current target assistant turn(s) should
contribute to SFT. The exporter annotates those messages with
``trainable: true`` and may mark an assistant-start prefill with
``assistant_prefill``; this class converts both annotations into VERL's token
``loss_mask`` and leaves all user/system/history/padding tokens masked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def apply_target_only_loss_mask(
    assistant_loss_mask: Sequence[int | float],
    attention_mask: Sequence[int | float],
    *,
    trainable: bool,
) -> list[int | float]:
    """Apply target selection and real-token masking to an assistant mask.

    This small dependency-free helper mirrors the tensor operation used by
    :class:`GroundCUAWindowSFTDataset`, making the contract testable without
    importing torch/VERL in the repository's lightweight environment.
    """

    if len(assistant_loss_mask) != len(attention_mask):
        raise ValueError("assistant and attention masks must have the same length")
    if not trainable:
        return [0 for _ in assistant_loss_mask]
    return [
        (loss_value * attention_value)
        for loss_value, attention_value in zip(assistant_loss_mask, attention_mask)
    ]


try:  # VERL is intentionally optional for repository-only tests.
    import torch
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
except ImportError:  # pragma: no cover - exercised only outside VERL envs.
    torch = None  # type: ignore[assignment]

    class MultiTurnSFTDataset:  # type: ignore[no-redef]
        """Import-time stub with a useful error when instantiated without VERL."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError(
                "GroundCUAWindowSFTDataset requires a repository-local VERL environment"
            )


class GroundCUAWindowSFTDataset(MultiTurnSFTDataset):
    """MultiTurnSFTDataset with per-assistant target loss selection."""

    @staticmethod
    def _is_trainable_assistant(message: dict[str, Any]) -> bool:
        return message.get("role") == "assistant" and message.get("trainable") is True

    def _prefill_loss_end(
        self,
        *,
        index: int,
        input_ids,
        prefill: str,
        tools: list[dict[str, Any]] | None,
        enable_thinking: bool | None,
        content_blocks: bool,
    ) -> int:
        """Return the exclusive token index after an assistant prefill.

        The prefix and an empty assistant message are rendered with the active
        tokenizer so wrapper and end-of-message tokens are excluded from the
        count. The rendered prefix must also match the beginning of the actual
        assistant content; otherwise the dataset fails instead of applying an
        incorrect loss boundary.
        """

        processor = self.processor if self.processor is not None else self.tokenizer
        template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking

        def render(content: str):
            message_content = (
                [{"type": "text", "text": content}] if content_blocks else content
            )
            rendered = processor.apply_chat_template(
                [{"role": "assistant", "content": message_content}],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            )
            ids = rendered["input_ids"][0]
            if index != 0:
                ids = ids[len(self.system_prompt) :]
            return ids

        prefix_ids = render(prefill)
        empty_ids = render("")
        content_token_count = int(prefix_ids.shape[0] - empty_ids.shape[0])
        generation_prompt_length = len(self.generation_prompt)
        if content_token_count <= 0:
            raise ValueError("assistant_prefill produced no content tokens")
        prefix_content_ids = prefix_ids[
            generation_prompt_length : generation_prompt_length + content_token_count
        ]
        actual_prefix_ids = input_ids[
            generation_prompt_length : generation_prompt_length + content_token_count
        ]
        if not torch.equal(prefix_content_ids, actual_prefix_ids):
            raise ValueError(
                "assistant_prefill is not the token prefix of the rendered assistant content"
            )
        return generation_prompt_length + content_token_count

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        full_message: list,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
    ):
        """Tokenize one turn, masking history and non-assistant turns."""

        # Keep internal flags out of the chat template.  Some tokenizers reject
        # unknown message keys, while others silently ignore them.
        template_message = {
            key: value
            for key, value in message.items()
            if key not in {"trainable", "assistant_prefill"}
        }
        input_ids, loss_mask, attention_mask, inputs = super()._process_single_message(
            index=index,
            message=template_message,
            full_message=full_message,
            tools=tools,
            enable_thinking=enable_thinking,
        )

        # Base VERL marks every assistant turn as trainable.  Restrict it to
        # explicit targets, and always multiply by attention so padding can
        # never produce loss even if a tokenizer returned nonzero values.
        trainable = self._is_trainable_assistant(message)
        if torch is None:  # pragma: no cover - the base call already failed.
            return input_ids, loss_mask, attention_mask, inputs
        prefill = message.get("assistant_prefill")
        if prefill is not None:
            if message.get("role") != "assistant":
                raise ValueError("assistant_prefill is only valid on assistant messages")
            if not isinstance(prefill, str) or not prefill:
                raise ValueError("assistant_prefill must be a nonempty string")
            prefill_end = self._prefill_loss_end(
                index=index,
                input_ids=input_ids,
                prefill=prefill,
                tools=tools,
                enable_thinking=enable_thinking,
                content_blocks=isinstance(template_message.get("content"), list),
            )
            loss_mask[:prefill_end] = 0
        if not trainable:
            loss_mask = torch.zeros_like(loss_mask)
        loss_mask = loss_mask * attention_mask
        return input_ids, loss_mask, attention_mask, inputs


__all__ = ["GroundCUAWindowSFTDataset", "apply_target_only_loss_mask"]
