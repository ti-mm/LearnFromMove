"""Public benchmark manifests retain the environment and evaluation contracts."""
import json
from collections import Counter

import pytest

from gui_agent_captcha.benchmarks.exploration_depth.contracts import (
    PAPER_SUITE_ID, PAPER_VARIANTS, episodes_for_variant,
)
from gui_agent_captcha.custom_envs import build_benchmark_variant
from latentguiworld.evaluate import SUITE
from latentguiworld.suite import load_evaluation_manifest


@pytest.fixture
def paper_manifest(tmp_path):
    source = json.loads(SUITE.read_text())
    source["suite_id"] = PAPER_SUITE_ID
    for episode in source["episodes"]:
        episode["suite_id"] = PAPER_SUITE_ID
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(source))
    return path


@pytest.mark.parametrize("name", PAPER_VARIANTS)
def test_public_names_load_and_build_environments(paper_manifest, name):
    original = paper_manifest.read_bytes()
    raw = json.loads(original)
    manifest = load_evaluation_manifest(paper_manifest)
    episodes = episodes_for_variant(manifest, name)
    expected = [e for e in raw["episodes"] if e["variant"] == name]
    assert len(episodes) == 150
    assert [e["shared_scene_config"] for e in episodes] == [e["shared_scene_config"] for e in expected]
    assert all(e["scene_variant"] == name for e in episodes)
    env = build_benchmark_variant(name, manifest_path=paper_manifest)
    assert len(env.list_task_ids()) == 150
    assert paper_manifest.read_bytes() == original


def test_mixed_public_suite_is_rejected(paper_manifest):
    manifest = json.loads(paper_manifest.read_text())
    manifest["episodes"][0]["suite_id"] = "another-benchmark"
    paper_manifest.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="suite IDs"):
        load_evaluation_manifest(paper_manifest)


def test_distribution_labels_and_manifest_validation():
    from gui_agent_captcha.benchmarks.exploration_depth.manifest import validate_manifest_structure

    manifest = load_evaluation_manifest(SUITE)
    for name in PAPER_VARIANTS:
        episodes = episodes_for_variant(manifest, name)
        counts = Counter(e["split"] for e in episodes)
        assert counts == ({"iid": 75, "ood": 75} if name.startswith("drag_") else {None: 150})
    validate_manifest_structure(manifest, expected_count=150)
    for episode in manifest["episodes"]:
        if episode["family"] == "rotation":
            episode["split"] = "ood"
    with pytest.raises(ValueError, match="null distribution"):
        validate_manifest_structure(manifest, expected_count=150)


def test_rotation_training_holdout_role(tmp_path):
    from gui_agent_captcha.domains.rotation.paired_online_dataset import _build_heldout_episodes

    original = SUITE.read_bytes()
    episodes = _build_heldout_episodes(root=tmp_path, formal_suite_root=SUITE.parent)
    assert len(episodes) == 300
    assert {e["split"] for e in episodes} == {"heldout"}
    assert all((tmp_path / e["canonical_case_path"]).is_file() for e in episodes)
    assert SUITE.read_bytes() == original


def test_ten_choice_scene_matches_manifest_names():
    manifest = json.loads(SUITE.read_text())
    for episode in manifest["episodes"]:
        if episode["family"] != "ten_choice":
            continue
        html = (SUITE.parent / episode["shared_scene_config"]["html_path"]).read_text()
        assert "variant === 'ten_choice_egocentric'" in html
