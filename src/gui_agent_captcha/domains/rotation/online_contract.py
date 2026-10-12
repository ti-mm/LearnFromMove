from __future__ import annotations

INTERACTION_ONLINE_VIEWPORT: tuple[int, int] = (1280, 720)
INTERACTION_ONLINE_IMAGE_MAX_PIXELS = 1280 * 720
INTERACTION_ONLINE_SFT_CHECKPOINT = (
    "checkpoints/migrated-20260903/source-artifacts-checkpoints/"
    "qwen35-9b-verl-sft-rotation-teacherthink-english-h200x8-hd720-"
    "hist3-fullhistory-nosystem-ctx16k-mbs1-bs64-ep3-save100-"
    "selected-model"
)
INTERACTION_ONLINE_MM_PROCESSOR_KWARGS: dict[str, int] = {
    "max_pixels": INTERACTION_ONLINE_IMAGE_MAX_PIXELS,
}
