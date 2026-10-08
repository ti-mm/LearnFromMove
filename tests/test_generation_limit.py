"""CPU regression checks against GUI_CAPTCHA_BUDGET_SNAPSHOT (no GPU imports).

Execute the snapshot's register and generate_turn implementations with a real
Hydra lazy import and a fake generation server. No model or browser is started.
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import hydra
import pytest
from omegaconf import OmegaConf

from gui_agent_captcha.integrations.browser_trajectory import (
    DynamicGenerationBudgetV4,
    GeneratedTurnV4,
    OnlineInfrastructureErrorV4,
)


@pytest.fixture
def snapshot():
    return Path(__file__).resolve().parents[1]


def compile_nodes(nodes, namespace):
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
              *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "<snapshot source>", "exec"), namespace)


@pytest.mark.parametrize("first_view", ["first", "third"])
def test_yaml_budget_survives_lazy_import_and_repeated_instances(
    snapshot, tmp_path, monkeypatch, first_view
):
    source = snapshot / "third_party/verl/verl/experimental/agent_loop/agent_loop.py"
    tree = ast.parse(source.read_text())
    register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register")
    registry = {}
    namespace = {"_agent_loop_registry": registry}
    compile_nodes([register], namespace)
    bridge = ModuleType("_budget_test_bridge")
    bridge.register = namespace["register"]
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Mirrors the real dependency: importing third-person also imports first-person.
    for view in ("first", "third"):
        name = f"_budget_lazy_{view}"
        monkeypatch.delitem(sys.modules, name, raising=False)
        (tmp_path / f"{name}.py").write_text(
            ("import _budget_lazy_first\n" if view == "third" else "")
            + "from _budget_test_bridge import register\n"
            + f"@register('{view}')\n"
            + "class Agent:\n"
            + "    def __init__(self, max_generation_tokens=None, browser_concurrency=4):\n"
            + "        self.max_generation_tokens = max_generation_tokens\n"
            + "        self.browser_concurrency = browser_concurrency\n"
        )
        registry[view] = OmegaConf.create({
            "_target_": f"{name}.Agent", "max_generation_tokens": 384,
            "browser_concurrency": 7,
        })
    try:
        order = [first_view, "third" if first_view == "first" else "first"] * 3
        for view in order:
            agent = hydra.utils.instantiate(registry[view])
            assert agent.max_generation_tokens == 384
            assert agent.browser_concurrency == 7
            budget = DynamicGenerationBudgetV4(max_generation_tokens=agent.max_generation_tokens)
            assert budget.sampling_params(
                {"max_tokens": 8192, "max_new_tokens": 8192},
                prompt_token_count=3000, accumulated_response_token_count=100,
            ) == {"max_tokens": 384}
        # Explicit YAML still overrides a class that was registered first.
        namespace["register"]("plain")(type("Plain", (), {}))
        assert "_target_" in registry["plain"]
        registry["plain"] = OmegaConf.create({"_target_": "configured.Plain", "limit": 384})
        namespace["register"]("plain")(type("Plain", (), {}))
        assert registry["plain"].limit == 384
    finally:
        for view in ("first", "third"):
            sys.modules.pop(f"_budget_lazy_{view}", None)


@pytest.fixture
def generate_turn(snapshot):
    source = snapshot / "src/gui_agent_captcha/integrations/verl_browser.py"
    tree = ast.parse(source.read_text())
    function = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "generate_turn"
    )
    wrapper = ast.parse(
        "async def invoke(self, request, base_sampling):\n"
        "    generate_seconds = 0.0\n"
        "    num_preempted = 0\n"
        "    pass\n"
        "    return await generate_turn(request, base_sampling)\n"
    ).body[0]
    wrapper.body[2] = function
    namespace = {
        "asyncio": asyncio, "time": time, "Any": Any,
        "GeneratedTurnV4": GeneratedTurnV4,
        "OnlineInfrastructureErrorV4": OnlineInfrastructureErrorV4,
        "ONLINE_V4_IMAGE_MAX_PIXELS": 921600,
    }
    compile_nodes([wrapper], namespace)
    return namespace["invoke"]


@pytest.mark.parametrize("returned_tokens", [383, 384, 385])
@pytest.mark.parametrize("consume_params", [False, True])
def test_actual_adapter_sends_cap_and_rejects_backend_overrun(
    generate_turn, returned_tokens, consume_params
):
    received = []

    async def generate(**kwargs):
        received.append(dict(kwargs["sampling_params"]))
        # Some server implementations consume this dictionary with pop().
        if consume_params:
            kwargs["sampling_params"].pop("max_tokens")
        return SimpleNamespace(
            token_ids=[1] * returned_tokens, log_probs=[-0.1] * returned_tokens,
            num_preempted=0, extra_fields={},
        )

    adapter = SimpleNamespace(
        _encode_runtime_prompt=lambda _: SimpleNamespace(prompt_ids=[1] * 3000, images=[], messages=[]),
        prompt_length=8192, response_length=8192,
        generation_budget=DynamicGenerationBudgetV4(max_generation_tokens=384),
        server_manager=SimpleNamespace(generate=generate),
        tokenizer=SimpleNamespace(decode=lambda *a, **kw: "text"),
        prompt_owned_think_opening=True,
    )
    request = SimpleNamespace(
        prompt=None, request_id="test", accumulated_response_token_count=100,
        observations=[],
    )
    call = generate_turn(adapter, request, {"max_tokens": 8192, "max_new_tokens": 8192})
    if returned_tokens > 384:
        with pytest.raises(OnlineInfrastructureErrorV4, match="385 > 384"):
            asyncio.run(call)
    else:
        result = asyncio.run(call)
        assert len(result.token_ids) == returned_tokens
        assert result.extra_fields["generation_budget"] == 384
    assert received == [{"max_tokens": 384}]
