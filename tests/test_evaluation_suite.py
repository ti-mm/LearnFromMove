"""Exercise the shared evaluation contract across model and ablation runners."""
import importlib
import json

import pytest

from latentguiworld.evaluate import SUITE
from latentguiworld.suite import load_evaluation_manifest


@pytest.mark.parametrize("runner", [
    "local", "vllm_benchmark", "astra", "gpt56",
])
def test_model_runner_uses_benchmark_loader(runner, tmp_path):
    module = importlib.import_module(f"gui_agent_captcha.eval.exploration_depth_{runner}")
    assert module.load_manifest is load_evaluation_manifest
    manifest = load_evaluation_manifest(SUITE)
    assert len(manifest["episodes"]) == 900
    manifest["suite_id"] = "unsupported-suite"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fixed paper benchmark"):
        module.load_manifest(path)


@pytest.mark.parametrize("mutation", ["unfrozen", "missing", "duplicate", "mixed_suite"])
def test_invalid_manifest_rejected_before_evaluation(mutation, tmp_path):
    manifest = json.loads(SUITE.read_text())
    if mutation == "unfrozen":
        manifest["frozen"] = False
    elif mutation == "missing":
        manifest["episodes"].pop()
    elif mutation == "duplicate":
        manifest["episodes"][1] = manifest["episodes"][0]
    else:
        manifest["episodes"][0]["suite_id"] = "unsupported-suite"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_evaluation_manifest(path)
