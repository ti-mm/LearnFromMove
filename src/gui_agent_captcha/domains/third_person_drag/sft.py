from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from ...actions import PrimitiveAction
from ...protocol_tracks import (
    CANONICAL_ASSISTANT_FIELD,
    CANONICAL_THINK_FIELD,
    PROTOCOL_VERSION,
    render_action_assistant_response,
)
from .dataset import TASK_DESCRIPTION
from .dataset_v2 import (
    PAPER_V2_OOD_SHAPES,
    PAPER_V2_PROFILE,
    PAPER_V2_TRAIN_SEED,
    PAPER_V2_TRAIN_SHAPES,
    PAPER_V2_VIEWPORT,
    PAPER_V3_TRAIN_HELD_MOVE_ASSIGNMENT,
    PAPER_V3_TRAIN_HELD_MOVE_COUNTS,
    PAPER_V3_TRAIN_PROFILE,
    PAPER_V3_TRAIN_TRAJECTORY_PROFILE,
    generate_training_episode_dataset,
)
from .paths import paper_v3_sft_root

DEFAULT_SFT_COUNT = 4000
DEFAULT_SFT_MAX_STEPS = 6
SFT_SOURCE = "third_person_drag_captcha_v3_synthetic_closed_loop"


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )


def _clean_output_dir(output_dir: Path) -> None:
    for name in (
        "episodes",
        "screenshots",
        "traces",
        "train.jsonl",
        "audit.jsonl",
        "summary.json",
    ):
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _clean_action(action: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {"kind": str(action["kind"])}
    if cleaned["kind"] == "move_to":
        cleaned["x"] = float(action["x"])
        cleaned["y"] = float(action["y"])
    return cleaned


def _primitive(action: dict[str, Any]) -> PrimitiveAction:
    cleaned = _clean_action(action)
    return PrimitiveAction(
        kind=str(cleaned["kind"]),
        x=cleaned.get("x"),
        y=cleaned.get("y"),
    )


def _validated_training_action_plan(
    source_plan: list[dict[str, Any]],
    *,
    expected_held_move_count: int,
) -> tuple[list[dict[str, Any]], int]:
    cleaned_source = [_clean_action(action) for action in source_plan]
    action_kinds = [action["kind"] for action in cleaned_source]
    held_move_count = len(action_kinds) - 3
    if not (
        action_kinds[:2] == ["move_to", "mouse_down"]
        and action_kinds[-1:] == ["mouse_up"]
        and set(action_kinds[2:-1]) == {"move_to"}
        and held_move_count in PAPER_V3_TRAIN_HELD_MOVE_COUNTS
        and held_move_count == expected_held_move_count
    ):
        raise RuntimeError("source episode has an invalid paper-v3 drag oracle")
    return cleaned_source, held_move_count


def _observation_step(observation: Any) -> dict[str, Any]:
    step: dict[str, Any] = {
        "type": "observation",
        "image_path": str(Path(observation.screenshot_path).resolve()),
    }
    if observation.cursor_xy is not None and observation.size_px is not None:
        width, height = observation.size_px
        step["cursor_xy"] = [
            round(float(observation.cursor_xy[0]) / width * 1000.0, 3),
            round(float(observation.cursor_xy[1]) / height * 1000.0, 3),
        ]
    return step


def _seed_think(
    action: dict[str, Any],
    *,
    button_is_down: bool,
    held_move_index: int | None = None,
    held_move_count: int | None = None,
) -> str:
    kind = str(action["kind"])
    if kind == "move_to" and not button_is_down:
        return (
            "I move the free cursor onto the visible solid piece before pressing "
            "the mouse button."
        )
    if kind == "mouse_down":
        return "The cursor is on the solid piece, so I press and hold to grab it."
    if kind == "move_to" and held_move_index != held_move_count:
        return (
            "I keep the mouse button held and move the piece partway toward the "
            "matching gray outline."
        )
    if kind == "move_to":
        return (
            "I keep the mouse button held and move the piece onto the matching "
            "gray outline."
        )
    return (
        "The held piece is aligned with the matching gray outline, so I release "
        "to complete the placement."
    )


def _action_step(
    action: dict[str, Any],
    *,
    thought: str,
) -> dict[str, Any]:
    cleaned = _clean_action(action)
    step: dict[str, Any] = {
        "type": "action",
        "kind": cleaned["kind"],
        CANONICAL_THINK_FIELD: thought,
        CANONICAL_ASSISTANT_FIELD: render_action_assistant_response(
            thought,
            cleaned,
        ),
    }
    if cleaned["kind"] == "move_to":
        step["x"] = cleaned["x"]
        step["y"] = cleaned["y"]
    return step


def _trace_row(
    *,
    episode_id: str,
    action_index: int,
    action: dict[str, Any],
    before_observation: Any,
    result: Any,
    thought: str,
) -> dict[str, Any]:
    return {
        "record_id": episode_id,
        "action_index": action_index,
        "before_observation_image": str(
            Path(before_observation.screenshot_path).resolve()
        ),
        "after_observation_image": str(
            Path(result.observation.screenshot_path).resolve()
        ),
        "action": _clean_action(action),
        "think_text": thought,
        "reward": float(result.reward),
        "done": bool(result.done),
        "success": bool(result.info.get("success")),
        "env_info": result.info,
    }


def _assert_image_is_720p(path: str) -> None:
    with Image.open(path) as image:
        if image.size != PAPER_V2_VIEWPORT:
            raise RuntimeError(
                f"paper-bound third-person SFT image is {image.size}, "
                f"expected {PAPER_V2_VIEWPORT}: {path}"
            )


def generate_sft_dataset(
    *,
    output_dir: Path | None = None,
    count: int = DEFAULT_SFT_COUNT,
    seed: int = PAPER_V2_TRAIN_SEED,
    clean: bool = True,
) -> dict[str, Any]:
    """Render success-only v3 third-person trajectories for SFT."""

    if output_dir is None:
        output_dir = paper_v3_sft_root()
    if count < 1:
        raise ValueError("count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean:
        _clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_root = output_dir / "episodes"
    episode_manifest = generate_training_episode_dataset(
        output_dir=episode_root,
        count=count,
        seed=seed,
        clean=True,
    )
    validation = episode_manifest.get("validation", {})
    if validation.get("status") != "passed":
        raise RuntimeError(
            "third-person v3 episode validation failed before SFT rendering: "
            f"{validation.get('issues', [])!r}"
        )

    from .environment import ThirdPersonDragCaptchaEnv

    dataset_root = episode_root / "train"
    env = ThirdPersonDragCaptchaEnv(
        dataset_root=dataset_root,
        artifact_dir=output_dir / "screenshots",
    )
    records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    layout_counts: Counter[str] = Counter()
    candidate_count_counts: Counter[str] = Counter()
    distance_band_counts: Counter[str] = Counter()
    action_step_counts: Counter[str] = Counter()
    held_move_step_counts: Counter[str] = Counter()

    try:
        for episode_id in episode_manifest["splits"]["train"]:
            meta_path = dataset_root / episode_id / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("distribution") != "iid":
                raise RuntimeError(f"{episode_id}: non-IID episode entered training")
            shape = str(meta.get("piece", {}).get("shape"))
            if shape not in PAPER_V2_TRAIN_SHAPES or shape in PAPER_V2_OOD_SHAPES:
                raise RuntimeError(f"{episode_id}: held-out shape entered training: {shape}")

            source_action_plan = [
                _clean_action(action)
                for action in meta["oracle_primitive_sequence"]
            ]
            action_plan, held_move_step_count = _validated_training_action_plan(
                source_action_plan,
                expected_held_move_count=int(meta["held_move_step_count"]),
            )

            observation = env.reset(task_type=episode_id, task_id=episode_id)
            _assert_image_is_720p(observation.screenshot_path)
            steps: list[dict[str, Any]] = [_observation_step(observation)]
            trace_rows: list[dict[str, Any]] = []
            button_is_down = False

            for action_index, action in enumerate(action_plan, start=1):
                held_move_index = (
                    action_index - 2
                    if action["kind"] == "move_to" and button_is_down
                    else None
                )
                thought = _seed_think(
                    action,
                    button_is_down=button_is_down,
                    held_move_index=held_move_index,
                    held_move_count=held_move_step_count,
                )
                steps.append(_action_step(action, thought=thought))
                before_observation = observation
                result = env.step(_primitive(action))
                _assert_image_is_720p(result.observation.screenshot_path)
                action_counts[str(action["kind"])] += 1
                trace_rows.append(
                    _trace_row(
                        episode_id=episode_id,
                        action_index=action_index,
                        action=action,
                        before_observation=before_observation,
                        result=result,
                        thought=thought,
                    )
                )
                observation = result.observation
                steps.append(_observation_step(observation))
                if action["kind"] == "mouse_down":
                    button_is_down = True
                elif action["kind"] == "mouse_up":
                    button_is_down = False
                if result.done and action_index != len(action_plan):
                    raise RuntimeError(
                        f"{episode_id}: episode ended before the planned release"
                    )

            if not trace_rows or trace_rows[-1]["success"] is not True:
                raise RuntimeError(
                    f"{episode_id}: generated paper-v3 drag trace did not succeed"
                )

            trace_path = output_dir / "traces" / episode_id / "trace.jsonl"
            _write_jsonl(trace_path, trace_rows)
            metadata = {
                "task_type": "third_person_drag_captcha",
                "benchmark": "ThirdPersonDragCaptcha",
                "dataset_profile": PAPER_V3_TRAIN_PROFILE,
                "geometry_profile": PAPER_V2_PROFILE,
                "trajectory_profile": PAPER_V3_TRAIN_TRAJECTORY_PROFILE,
                "perspective": "third_person",
                "source_mode": "synthetic_oracle",
                "success": True,
                "action_step_count": len(action_plan),
                "held_move_step_count": held_move_step_count,
                "max_steps": DEFAULT_SFT_MAX_STEPS,
                "viewport": list(PAPER_V2_VIEWPORT),
                "image_max_pixels": PAPER_V2_VIEWPORT[0] * PAPER_V2_VIEWPORT[1],
                "coordinate_format": "qwen3_relative_0_1000",
                "cursor_lock": "none",
                "camera_motion": False,
                "movement_mapping": "absolute_screen_position_direct",
                "drag_screen_displacement_scale": 1.0,
                "action_paradigm": "primitive_closed_loop",
                "protocol_version": PROTOCOL_VERSION,
            }
            records.append(
                {
                    "id": episode_id,
                    "source": SFT_SOURCE,
                    "instruction": TASK_DESCRIPTION,
                    "metadata": metadata,
                    "steps": steps,
                }
            )
            audit_records.append(
                {
                    "id": episode_id,
                    "source": SFT_SOURCE,
                    "metadata": {
                        **metadata,
                        "piece_shape": shape,
                        "layout_family": meta["layout_family"],
                        "slot_candidate_count": meta["slot_candidate_count"],
                        "drag_distance_band": meta["drag_distance_band"],
                        "drag_distance_px": meta["drag_distance_px"],
                        "source_oracle_action_plan": source_action_plan,
                        "oracle_action_plan": action_plan,
                    },
                    "episode_meta_path": str(meta_path),
                    "trace_path": str(trace_path),
                }
            )
            shape_counts[shape] += 1
            layout_counts[str(meta["layout_family"])] += 1
            candidate_count_counts[str(meta["slot_candidate_count"])] += 1
            distance_band_counts[str(meta["drag_distance_band"])] += 1
            action_step_counts[str(len(action_plan))] += 1
            held_move_step_counts[str(held_move_step_count)] += 1
    finally:
        env.close()

    _write_jsonl(output_dir / "train.jsonl", records)
    _write_jsonl(output_dir / "audit.jsonl", audit_records)
    summary = {
        "dataset": "third_person_drag_captcha_sft_v3",
        "source": SFT_SOURCE,
        "dataset_profile": PAPER_V3_TRAIN_PROFILE,
        "geometry_profile": PAPER_V2_PROFILE,
        "trajectory_profile": PAPER_V3_TRAIN_TRAJECTORY_PROFILE,
        "records": len(records),
        "actions": sum(action_counts.values()),
        "seed": seed,
        "viewport": list(PAPER_V2_VIEWPORT),
        "image_max_pixels": PAPER_V2_VIEWPORT[0] * PAPER_V2_VIEWPORT[1],
        "action_counts": dict(sorted(action_counts.items())),
        "action_step_count_distribution": dict(sorted(action_step_counts.items())),
        "held_move_step_count_distribution": dict(
            sorted(held_move_step_counts.items())
        ),
        "held_move_step_counts": list(PAPER_V3_TRAIN_HELD_MOVE_COUNTS),
        "held_move_assignment": PAPER_V3_TRAIN_HELD_MOVE_ASSIGNMENT,
        "shape_distribution": dict(sorted(shape_counts.items())),
        "layout_distribution": dict(sorted(layout_counts.items())),
        "candidate_count_distribution": dict(
            sorted(candidate_count_counts.items())
        ),
        "drag_distance_band_distribution": dict(
            sorted(distance_band_counts.items())
        ),
        "train_shapes": list(PAPER_V2_TRAIN_SHAPES),
        "ood_shapes": list(PAPER_V2_OOD_SHAPES),
        "ood_shapes_present": False,
        "image_history_max": 3,
        "task_requirement_policy": "single_task_requirement_v1",
        "validation": validation,
        "outputs": {
            "train_jsonl": str(output_dir / "train.jsonl"),
            "audit_jsonl": str(output_dir / "audit.jsonl"),
            "episode_manifest": str(episode_root / "manifest.json"),
            "screenshots": str(output_dir / "screenshots"),
            "traces": str(output_dir / "traces"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ThirdPersonDragCaptcha paper-v3 SFT trajectories."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--count", type=int, default=DEFAULT_SFT_COUNT)
    parser.add_argument("--seed", type=int, default=PAPER_V2_TRAIN_SEED)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = paper_v3_sft_root()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate_sft_dataset(
        output_dir=args.output_dir or paper_v3_sft_root(),
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("validation", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
