"""Shared evaluation contract for benchmark and ablation runners."""
from collections import Counter

from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
    EXPLORATION_BENCHMARK_VARIANTS,
    FORMAL_SUITE_ID,
    load_manifest,
)


def load_evaluation_manifest(path=None):
    manifest = load_manifest(path)
    if manifest.get("suite_id") != FORMAL_SUITE_ID or manifest.get("frozen") is not True:
        raise ValueError("evaluation requires the fixed paper benchmark suite")
    episodes = manifest.get("episodes", [])
    expected = {variant.key: 150 for variant in EXPLORATION_BENCHMARK_VARIANTS}
    if Counter(episode["variant"] for episode in episodes) != expected:
        raise ValueError("benchmark evaluation requires 150 episodes per variant")
    if len({episode["episode_id"] for episode in episodes}) != 900:
        raise ValueError("benchmark evaluation requires 900 unique episode IDs")
    if any(episode.get("suite_id") != FORMAL_SUITE_ID for episode in episodes):
        raise ValueError("episode suite IDs must match the benchmark manifest")
    return manifest


def build_evaluation_variant(variant, **kwargs):
    """Build a paper environment with first-release episode termination."""
    from gui_agent_captcha.custom_envs.catalog import build_benchmark_variant
    from gui_agent_captcha.benchmarks.exploration_depth.contracts import runtime_variant

    environment = build_benchmark_variant(variant, **kwargs)
    if not runtime_variant(variant).startswith("rotation_"):
        return environment
    original_step = environment.step
    original_reset = environment.reset
    finished = False

    def reset(*args, **options):
        nonlocal finished
        observation = original_reset(*args, **options)
        finished = False
        return observation

    def step(action):
        nonlocal finished
        if finished:
            raise RuntimeError("episode ended at the first rotation release")
        result = original_step(action)
        audit = environment.get_evaluator_audit()
        attempt = audit.get("rotation_state", {}).get("attempt")
        if isinstance(attempt, dict):
            result.done = True
            result.reward = float(attempt.get("success") is True)
            result.info["strict_first_release"] = True
            finished = True
        return result

    environment.step = step
    environment.reset = reset
    return environment


def terminal_score(terminal_reason, evaluator_audit):
    """Score committed outcomes, including the first actual rotation attempt."""
    attempt = evaluator_audit.get("rotation_state", {}).get("attempt")
    committed = (attempt.get("success") is True if isinstance(attempt, dict)
                 else terminal_reason == "success")
    return {
        "success": committed,
        "success_source": "environment_terminal" if committed else None,
        "committed_success": committed,
        "cutoff_final_state_scored": False,
        "cutoff_final_state_success": False,
        "final_state_evaluation": None,
    }
