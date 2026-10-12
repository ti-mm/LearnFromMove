from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

FrontierProvider = Literal["openai"]

DEFAULT_NATIVE_MAX_STEPS = 12


@dataclass(frozen=True)
class FrontierModelPreset:
    provider: FrontierProvider
    model: str
    base_url: str
    api_key_env: str


FRONTIER_MODEL_PRESETS: dict[str, FrontierModelPreset] = {
    "gpt56_sol": FrontierModelPreset(
        provider="openai",
        model="gpt-5.6-sol",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    ),
}


@dataclass(frozen=True)
class FrontierModelConfig:
    provider: FrontierProvider
    model: str
    base_url: str
    api_key: str = field(repr=False)
    api_key_env: str = field(default="", repr=False)

    @classmethod
    def from_preset(
        cls,
        preset_key: str,
        *,
        env: Mapping[str, str],
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> FrontierModelConfig:
        try:
            preset = FRONTIER_MODEL_PRESETS[preset_key]
        except KeyError as exc:
            raise ValueError(
                f"unknown frontier preset {preset_key!r}; choose from "
                f"{sorted(FRONTIER_MODEL_PRESETS)!r}"
            ) from exc
        resolved_key_env = api_key_env or preset.api_key_env
        api_key = str(env.get(resolved_key_env, "")).strip()
        if not api_key:
            raise ValueError(f"{resolved_key_env} is required for {preset_key}")
        return cls(
            provider=preset.provider,
            model=model or preset.model,
            base_url=(base_url or preset.base_url).rstrip("/"),
            api_key=api_key,
            api_key_env=resolved_key_env,
        )


__all__ = [
    "DEFAULT_NATIVE_MAX_STEPS",
    "FRONTIER_MODEL_PRESETS",
    "FrontierModelConfig",
    "FrontierModelPreset",
    "FrontierProvider",
]
