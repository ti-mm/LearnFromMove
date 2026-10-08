from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

CustomEnvironmentKey = Literal[
    "ten_choice_captcha",
    "rotation_captcha",
    "slot_drag_game",
    "third_person_drag_captcha",
]


@dataclass(frozen=True)
class CustomEnvironmentSpec:
    key: CustomEnvironmentKey
    display_name: str
    environment_class: str
    task_family: str
    aliases: tuple[str, ...]
    canonical_dataset_roots: tuple[str, ...]
    source_files: tuple[str, ...]
    data_files: tuple[str, ...]
    eval_files: tuple[str, ...]
    run_files: tuple[str, ...]
    display_files: tuple[str, ...]
    script_files: tuple[str, ...]
    docs: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def code_paths(self) -> tuple[Path, ...]:
        path_groups = (
            self.source_files,
            self.data_files,
            self.eval_files,
            self.run_files,
            self.display_files,
            self.script_files,
            self.docs,
        )
        return tuple(Path(path) for group in path_groups for path in group)


CUSTOM_ENVIRONMENTS: tuple[CustomEnvironmentSpec, ...] = (
    CustomEnvironmentSpec(
        key="ten_choice_captcha",
        display_name="Ten-Choice CAPTCHA",
        environment_class=(
            "gui_agent_captcha.domains.ten_choice.environment.HoverRevealEnv"
        ),
        task_family="ten_choice_captcha",
        aliases=("hover_real", "hover_reveal"),
        canonical_dataset_roots=(
            "artifacts/datasets/hover_real/train",
            "artifacts/datasets/hover_real/val",
            "artifacts/datasets/hover_real/test",
        ),
        source_files=('src/gui_agent_captcha/domains/ten_choice/environment.py', 'src/gui_agent_captcha/envs/browser_base.py'),
        data_files=('src/gui_agent_captcha/domains/ten_choice/dataset.py', 'src/gui_agent_captcha/domains/ten_choice/sft_export.py'),
        eval_files=(),
        run_files=(),
        display_files=(),
        script_files=(),
        docs=(),
        notes=(
            "Ten-choice CAPTCHA is implemented through HoverReveal code paths.",
        ),
    ),
    CustomEnvironmentSpec(
        key="rotation_captcha",
        display_name="Rotation CAPTCHA",
        environment_class="gui_agent_captcha.domains.rotation.environment.InteractionCaptchaEnv",
        task_family="rotation_captcha",
        aliases=("interaction_rotation", "interaction_rotation_captcha"),
        canonical_dataset_roots=(
            "artifacts/datasets/lujiahao_rotation_test_150_openimages_bgfix_20260604",
            "artifacts/datasets/lujiahao_rotation_sft_4k_train_only_openimages1k_bgfix_20260604",
        ),
        source_files=('src/gui_agent_captcha/domains/rotation/environment.py', 'src/gui_agent_captcha/envs/browser_base.py'),
        data_files=(),
        eval_files=(),
        run_files=(),
        display_files=(),
        script_files=('scripts/interaction/serve_captcha_static_replay.py',),
        docs=(),
        notes=(
            "Current rotation eval is static_replay-only; serve_captcha_local.sh is a compatibility wrapper.",
        ),
    ),
    CustomEnvironmentSpec(
        key="slot_drag_game",
        display_name="SlotDragGame",
        environment_class="gui_agent_captcha.domains.slot_drag.environment.SlotDragGameEnv",
        task_family="slot_drag_game",
        aliases=("slot_drag", "sdg"),
        canonical_dataset_roots=(
            "data/slot_drag/benchmark_150_v1/test",
        ),
        source_files=('src/gui_agent_captcha/domains/slot_drag/environment.py',),
        data_files=('src/gui_agent_captcha/domains/slot_drag/dataset.py', 'src/gui_agent_captcha/domains/slot_drag/dataset_v2.py'),
        eval_files=(),
        run_files=(),
        display_files=(),
        script_files=(),
        docs=(),
        notes=(
            "SlotDragGame is the first-person drag task in the current four self-built task set.",
            "The fixed center reticle acts as the first-person interaction point; move_to is interpreted as a one-step view displacement.",
        ),
    ),
    CustomEnvironmentSpec(
        key="third_person_drag_captcha",
        display_name="Third-Person Drag CAPTCHA",
        environment_class=(
            "gui_agent_captcha.domains.third_person_drag.environment."
            "ThirdPersonDragCaptchaEnv"
        ),
        task_family="third_person_drag_captcha",
        aliases=("third_person_drag", "tpd"),
        canonical_dataset_roots=(
            "data/third_person_drag/benchmark_150_v1/test",
        ),
        source_files=('src/gui_agent_captcha/domains/third_person_drag/environment.py',),
        data_files=('src/gui_agent_captcha/domains/third_person_drag/dataset.py', 'src/gui_agent_captcha/domains/third_person_drag/dataset_v2.py', 'src/gui_agent_captcha/domains/third_person_drag/sft.py'),
        eval_files=(),
        run_files=(),
        display_files=(),
        script_files=(),
        docs=(),
        notes=(
            "The camera is fixed and the whole task surface is visible in third person.",
            "The cursor is free; move_to directly sets its screen position and a grabbed piece follows screen displacement one-to-one without sensitivity or view-zoom conversion.",
        ),
    ),
)


def get_custom_environment(key: str) -> CustomEnvironmentSpec:
    for spec in CUSTOM_ENVIRONMENTS:
        if spec.key == key or key in spec.aliases:
            return spec
    valid = ", ".join(spec.key for spec in CUSTOM_ENVIRONMENTS)
    raise KeyError(f"unknown custom environment {key!r}; valid keys: {valid}")


@dataclass(frozen=True)
class BenchmarkVariantSpec:
    key: str
    family: str
    exploration_level: str
    environment_class: str


EXPLORATION_DEPTH_VARIANTS: tuple[BenchmarkVariantSpec, ...] = (
    BenchmarkVariantSpec(
        "ten_choice_third_person",
        "ten_choice",
        "L1",
        "gui_agent_captcha.domains.ten_choice.first_person.PairedHoverRevealEnv",
    ),
    BenchmarkVariantSpec(
        "ten_choice_first_person",
        "ten_choice",
        "L2",
        "gui_agent_captcha.domains.ten_choice.first_person.FirstPersonTenChoiceEnv",
    ),
    BenchmarkVariantSpec(
        "rotation_inner",
        "rotation",
        "L2",
        "gui_agent_captcha.domains.rotation.paired.PairedInnerRotationEnv",
    ),
    BenchmarkVariantSpec(
        "rotation_outer",
        "rotation",
        "L2",
        "gui_agent_captcha.domains.rotation.paired.PairedOuterRotationEnv",
    ),
    BenchmarkVariantSpec(
        "drag_third_person",
        "drag",
        "L0",
        "gui_agent_captcha.benchmarks.exploration_depth.drag.PairedThirdPersonDragEnv",
    ),
    BenchmarkVariantSpec(
        "drag_first_person",
        "drag",
        "L2",
        "gui_agent_captcha.benchmarks.exploration_depth.drag.PairedFirstPersonDragEnv",
    ),
)


def get_benchmark_variant(key: str) -> BenchmarkVariantSpec:
    from ..benchmarks.exploration_depth.contracts import runtime_variant

    key = runtime_variant(key)
    for spec in EXPLORATION_DEPTH_VARIANTS:
        if spec.key == key:
            return spec
    valid = ", ".join(spec.key for spec in EXPLORATION_DEPTH_VARIANTS)
    raise KeyError(f"unknown benchmark variant {key!r}; valid keys: {valid}")


def build_benchmark_variant(key: str, **kwargs: Any) -> object:
    spec = get_benchmark_variant(key)
    module_name, class_name = spec.environment_class.rsplit(".", 1)
    environment_class = getattr(importlib.import_module(module_name), class_name)
    return environment_class(**kwargs)
