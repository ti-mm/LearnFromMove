"""Exercise release adapters against one environment/scoring contract."""
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentguiworld import suite
from latentguiworld.evaluate import VARIANTS


@pytest.mark.parametrize("runner", ["local", "astra", "gpt56"])
def test_shared_environment_factory(runner):
    module = importlib.import_module(f"gui_agent_captcha.eval.exploration_depth_{runner}")
    assert module.build_benchmark_variant is suite.build_evaluation_variant


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("success", [False, True])
def test_first_release_ends_rotation_and_reset_starts_new_episode(monkeypatch, variant, success):
    from gui_agent_captcha.custom_envs import catalog
    class Environment:
        attempt = None
        def reset(self, **kwargs):
            self.attempt = None
            return "observation"
        def step(self, action):
            if action == "release":
                self.attempt = {"success": success}
            return SimpleNamespace(done=False, reward=0.0, info={})
        def get_evaluator_audit(self):
            return {"rotation_state": {"attempt": self.attempt}}
    monkeypatch.setattr(catalog, "build_benchmark_variant", lambda *a, **k: Environment())
    environment = suite.build_evaluation_variant(variant)
    environment.reset()
    assert not environment.step("move").done
    result = environment.step("release")
    assert result.done is variant.startswith("rotation_")
    if variant.startswith("rotation_"):
        assert result.reward == float(success)
        with pytest.raises(RuntimeError, match="first rotation release"):
            environment.step("move")
        environment.reset()
        assert not environment.step("move").done


@pytest.mark.parametrize("runner", ["gpt56"])
def test_aligned_cutoff_is_not_success_and_first_attempt_controls_score(runner):
    module = importlib.import_module(f"gui_agent_captcha.eval.exploration_depth_{runner}")
    audit = {"rotation_state": {"sliderValue": 0}, "hidden_mapping": {
        "target_slider_value": 0, "degrees_per_slider_unit": 1,
        "sensitivity": 1, "rotation_direction": 1}, "tolerance_deg": 10}
    assert not module._terminal_score("max_turns", audit)["success"]
    assert not module._terminal_score("max_responses", audit)["success"]
    audit["rotation_state"]["attempt"] = {"success": False}
    assert not module._terminal_score("success", audit)["success"]
    audit["rotation_state"]["attempt"] = {"success": True}
    assert module._terminal_score("success", audit)["success"]




@pytest.mark.parametrize("runner", ["gpt56"])
def test_full_suite_response_budget_defaults(runner):
    module = importlib.import_module(f"gui_agent_captcha.eval.exploration_depth_{runner}")
    args = module._parser().parse_args(["manifest.json", "--output-root", "results"])
    assert args.max_responses == 12
    assert args.limit_per_variant == 150


def test_astra_default_and_prompt_ablation_use_one_six_environment_runner():
    from gui_agent_captcha.eval import exploration_depth_astra as runner
    args = runner._parser().parse_args(["manifest.json", "--output-root", "results"])
    assert set(args.variants) == set(VARIANTS)
    assert args.limit_per_variant == 150
    assert runner.evaluation_id_for(runner.ASTRA_PROFILE).endswith("150x6")
    assert runner.evaluation_id_for(runner.ASTRA_EXPLORE_FIRST_PERSON_PROFILE,
        50, runner.FIRST_PERSON_VARIANTS).endswith("50x2")
    assert "egocentric" in runner.ASTRA_EXPLORE_FIRST_PERSON_PROFILE.task_instruction_suffix




@pytest.mark.browser
@pytest.mark.parametrize("runner", ["local"])
def test_release_without_a_grab_does_not_end_rotation(runner, tmp_path):
    from gui_agent_captcha.actions import PrimitiveAction
    from gui_agent_captcha.benchmarks.exploration_depth.contracts import episodes_for_variant
    from gui_agent_captcha.benchmarks.exploration_depth.service import RotationReplayServer
    from latentguiworld.evaluate import SUITE
    class Backend:
        _call_index = 0
        checkpoint_path = Path("probe")
        last_raw_prediction = None
        last_think_text = None
        last_reasoning_content = None
        last_response_model = None
        last_call_log_path = None
        call_log_dir = None
        def contract(self):
            return {"model": "probe"}
        def predict_action(self, *args, **kwargs):
            self._call_index += 1
            return PrimitiveAction(kind="mouse_up")
    backend = Backend()
    episode = episodes_for_variant(suite.load_evaluation_manifest(SUITE), "rotation_inner")[0]
    module = importlib.import_module(f"gui_agent_captcha.eval.exploration_depth_{runner}")
    options = {"checkpoint_step": 0, "thinking_mode": "with-think"} if runner == "local" else {}
    with RotationReplayServer(public_root=SUITE.parent) as replay:
        record = module.run_episode(backend=backend, episode=episode, manifest_path=SUITE,
            output_root=tmp_path, rotation_base_url=replay.base_url, max_turns=2,
            strict_first_release=True, **options)
    assert record["terminal_reason"] == "max_turns"
    assert not record["success"]
    assert backend._call_index == 2


@pytest.mark.parametrize("module", ["exploration_depth_astra", "exploration_depth_gpt56"])
def test_api_cli_accepts_paper_environment_names(module):
    import importlib
    from gui_agent_captcha.benchmarks.exploration_depth.contracts import runtime_variant

    runner = importlib.import_module("gui_agent_captcha.eval." + module)
    args = runner._parser().parse_args([
        "manifest.json", "--output-root", "results/test",
        "--variants", "ten_choice_egocentric", "drag_egocentric",
        "--limit-per-variant", "50",
    ])
    assert tuple(runtime_variant(v) for v in args.variants) == ("ten_choice_first_person", "drag_first_person")
    assert args.limit_per_variant == 50
