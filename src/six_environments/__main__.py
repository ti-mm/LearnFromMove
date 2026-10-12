"""Run the benchmark environments from Learn from Move."""
from __future__ import annotations

import argparse
import json
import shutil
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

from gui_agent_captcha.actions import PrimitiveAction
from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
    EXPLORATION_BENCHMARK_VARIANTS, default_formal_manifest_path,
    load_manifest, PAPER_VARIANTS, runtime_variant, paper_variant,
)
from gui_agent_captcha.benchmarks.exploration_depth.server import make_handler, main as serve
from gui_agent_captcha.custom_envs import build_benchmark_variant

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "smoke", "serve"))
    parser.add_argument("--variant", choices=[*PAPER_VARIANTS, *(v.key for v in EXPLORATION_BENCHMARK_VARIANTS)])
    parser.add_argument("--manifest", type=Path, default=default_formal_manifest_path())
    parser.add_argument("--output", type=Path, default=ROOT / "runs/smoke")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if args.variant:
        args.variant = runtime_variant(args.variant)
    if manifest["suite_id"] != "learn_from_move":
        raise ValueError("This package requires the paper benchmark manifest")
    variants = [v.key for v in EXPLORATION_BENCHMARK_VARIANTS]
    if args.command == "list":
        for variant in variants:
            print(paper_variant(variant), sum(e["variant"] == variant for e in manifest["episodes"]))
        return
    if args.command == "serve":
        return serve(["--manifest", str(args.manifest), "--port", str(args.port)])
    args.output.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(args.manifest))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    results = []
    try:
        for variant in ([args.variant] if args.variant else variants):
            kwargs = dict(manifest_path=args.manifest, artifact_dir=args.output / variant)
            if variant.startswith("rotation_"):
                kwargs["base_url"] = f"http://127.0.0.1:{server.server_port}/rotation-replay"
            env = build_benchmark_variant(variant, **kwargs)
            try:
                task_ids = env.list_task_ids()
                observation = env.reset(task_id=task_ids[0])
                assert Path(observation.screenshot_path).is_file()
                family = next(e["family"] for e in manifest["episodes"] if e["variant"] == variant)
                preview = args.output / "paired_reset_examples" / family / f"{variant}.png"
                preview.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(observation.screenshot_path, preview)
                result = env.step(PrimitiveAction(kind="move_to", x=500, y=500))
                assert Path(result.observation.screenshot_path).is_file()
                if variant.startswith("rotation_"):
                    episode = next(e for e in manifest["episodes"] if e["episode_id"] == task_ids[0])
                    box = episode["shared_scene_config"]["slider_geometry"]
                    width, height = episode["viewport"]
                    env.step(PrimitiveAction(kind="move_to",
                        x=(box["x"] + box["width"] - 8) / (width - 1) * 1000,
                        y=(box["y"] + box["height"] / 2) / (height - 1) * 1000))
                # Raw rotation environments may continue after an unsuccessful release.
                env.step(PrimitiveAction(kind="mouse_down"))
                terminal = env.step(PrimitiveAction(kind="mouse_up"))
                assert Path(terminal.observation.screenshot_path).is_file()
                results.append(dict(variant=variant, task_count=len(task_ids),
                                    reset=True, move=True, release=True, done=bool(terminal.done)))
                print(variant, "passed", flush=True)
            finally:
                close = getattr(env, "close", None)
                if close is not None:
                    close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    (args.output / "verification.json").write_text(
        json.dumps(dict(suite_id=manifest["suite_id"], status="passed", environments=results), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
