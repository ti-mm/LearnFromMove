from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ...integrations.storage import storage_path

ExplorationVariant = Literal[
    "ten_choice_third_person",
    "ten_choice_first_person",
    "rotation_inner",
    "rotation_outer",
    "drag_third_person",
    "drag_first_person",
]


@dataclass(frozen=True)
class ExplorationVariantContract:
    key: ExplorationVariant
    family: Literal["ten_choice", "rotation", "drag"]
    exploration_level: Literal["L0", "L1", "L2"]


EXPLORATION_BENCHMARK_VARIANTS: tuple[ExplorationVariantContract, ...] = (
    ExplorationVariantContract("ten_choice_third_person", "ten_choice", "L1"),
    ExplorationVariantContract("ten_choice_first_person", "ten_choice", "L2"),
    ExplorationVariantContract("rotation_inner", "rotation", "L2"),
    ExplorationVariantContract("rotation_outer", "rotation", "L2"),
    ExplorationVariantContract("drag_third_person", "drag", "L0"),
    ExplorationVariantContract("drag_first_person", "drag", "L2"),
)

_CONTRACT_BY_KEY = {contract.key: contract for contract in EXPLORATION_BENCHMARK_VARIANTS}
FORMAL_SUITE_ID = "learn_from_move"
SMOKE_SUITE_ID = "exploration_depth_smoke_v6"


PAPER_SUITE_ID = "learn_from_move"
PAPER_VARIANTS = {
    "ten_choice_exocentric": "ten_choice_third_person",
    "ten_choice_egocentric": "ten_choice_first_person",
    "drag_exocentric": "drag_third_person",
    "drag_egocentric": "drag_first_person",
    "rotation_inner": "rotation_inner",
    "rotation_outer": "rotation_outer",
}


def runtime_variant(key: str) -> str:
    return PAPER_VARIANTS.get(key, key)


def paper_variant(key: str) -> str:
    return {value: name for name, value in PAPER_VARIANTS.items()}.get(key, key)


def default_formal_suite_root() -> Path:
    return storage_path("data", "formal_benchmarks", FORMAL_SUITE_ID)


def default_formal_manifest_path() -> Path:
    return default_formal_suite_root() / "manifest.json"


def default_smoke_suite_root() -> Path:
    return storage_path("data", "formal_benchmarks", SMOKE_SUITE_ID)


def default_smoke_manifest_path() -> Path:
    return default_smoke_suite_root() / "manifest.json"


def variant_contract(key: str) -> ExplorationVariantContract:
    key = runtime_variant(key)
    try:
        return _CONTRACT_BY_KEY[key]  # type: ignore[index]
    except KeyError as exc:
        valid = ", ".join(contract.key for contract in EXPLORATION_BENCHMARK_VARIANTS)
        raise KeyError(f"unknown exploration benchmark variant {key!r}; valid keys: {valid}") from exc


def load_manifest(path: Path | str | None = None) -> dict[str, Any]:
    manifest_path = default_formal_manifest_path() if path is None else Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "gui_captcha_exploration_depth_manifest_v1":
        raise ValueError(f"unsupported exploration-depth manifest: {manifest_path}")
    if payload.get("suite_id") == PAPER_SUITE_ID:
        for episode in payload.get("episodes", []):
            if episode.get("suite_id") != PAPER_SUITE_ID:
                raise ValueError("episode suite IDs must match the benchmark manifest")
            episode["scene_variant"] = episode["variant"]
            episode["variant"] = runtime_variant(episode["variant"])
            episode["suite_id"] = FORMAL_SUITE_ID
        payload["suite_id"] = FORMAL_SUITE_ID
    return payload


def episodes_for_variant(
    manifest: dict[str, Any],
    variant: str,
) -> list[dict[str, Any]]:
    variant = runtime_variant(variant)
    variant_contract(variant)
    episodes = [
        dict(episode)
        for episode in manifest.get("episodes", [])
        if episode.get("variant") == variant
    ]
    return sorted(episodes, key=lambda episode: str(episode["episode_id"]))


def load_episode(
    manifest: dict[str, Any],
    *,
    variant: str,
    episode_or_pair_id: str,
) -> dict[str, Any]:
    candidates = [
        episode
        for episode in episodes_for_variant(manifest, variant)
        if episode.get("episode_id") == episode_or_pair_id
        or episode.get("pair_id") == episode_or_pair_id
    ]
    if len(candidates) != 1:
        raise KeyError(
            f"expected one {variant} episode for {episode_or_pair_id!r}, "
            f"found {len(candidates)}"
        )
    return dict(candidates[0])


def policy_observation_metadata(episode: dict[str, Any]) -> dict[str, Any]:
    """Return the complete allow-list for model-visible structured metadata."""

    viewport = episode.get("viewport")
    return {
        "suite_id": episode["suite_id"],
        "pair_id": episode["pair_id"],
        "family": episode["family"],
        "viewport": list(viewport) if isinstance(viewport, (list, tuple)) else viewport,
    }


def resolve_suite_path(manifest_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def assert_policy_metadata_safe(metadata: dict[str, Any]) -> None:
    forbidden_fragments = (
        "variant",
        "sensitivity",
        "target_idx",
        "target_index",
        "target_xy",
        "target_coord",
        "target_angle",
        "rotation_direction",
        "oracle",
        "success_geometry",
    )
    lowered_keys = {str(key).casefold() for key in metadata}
    leaked = sorted(
        key
        for key in lowered_keys
        if any(fragment in key for fragment in forbidden_fragments)
    )
    if leaked:
        raise ValueError(f"policy-facing observation metadata leaks evaluator fields: {leaked}")
