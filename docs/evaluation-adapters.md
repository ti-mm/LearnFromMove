# Evaluation interfaces

All benchmark runners use the same 900-episode manifest and environment factory.
Drag and Ten-Choice stop at submission in both perspectives. Rotation ends at
the first release attempt. Each episode allows 12 model
responses; the adapter executes actions until the environment terminates.

| Model | Entry point | Interface |
|---|---|---|
| LatentLearner SFT, RL, and ablation checkpoints | `latentguiworld-eval` | Think/action format, local checkpoint or vLLM endpoint, three screenshots |
| GPT-5.6-Sol | `gui_agent_captcha.eval.exploration_depth_gpt56` | Native `computer` actions and independent mouse-button functions |
| GPT-6 Astra | `gui_agent_captcha.eval.exploration_depth_astra` | Persistent `exec_py` with mouse and screenshot operations |

Shared task selection and result I/O are in `eval/api_common.py`; environment
construction and scoring are in `latentguiworld/suite.py`. Paths under `eval/`
are relative to `src/gui_agent_captcha/`.

## API evaluation

Run from the repository root. Set `OPENAI_API_KEY` in your shell. The runners
accept `GPT56_SOL_BASE_URL` and `GPT6_ASTRA_BASE_URL` for deployment endpoints.

```bash
BENCHMARK_MANIFEST=data/formal_benchmarks/learn_from_move/manifest.json
python -m gui_agent_captcha.eval.exploration_depth_gpt56 "$BENCHMARK_MANIFEST" \
  --output-root results/sol --max-responses 12 --limit-per-variant 150
python -m gui_agent_captcha.eval.exploration_depth_astra "$BENCHMARK_MANIFEST" \
  --output-root results/astra --max-responses 12 --limit-per-variant 150
```

Sol exposes native computer actions plus `mouse_down` and `mouse_up` functions.
The button functions support held-button control in egocentric Drag and Rotation.
Native actions return screenshots; button functions return completion messages.
Astra requests screenshots and executes mouse operations through
`exec_py`. Each adapter preserves its model's conversation history.

## Checkpoint evaluation

```bash
latentguiworld-eval --model models/latentlearner-sft --output results/sft
latentguiworld-eval --model models/latentlearner-rl --output results/rl
latentguiworld-eval --model models/paradigm-ablation --output results/paradigm-ablation
latentguiworld-eval --model models/data-ablation --output results/data-ablation
```

For a vLLM deployment of the same checkpoint:

```bash
latentguiworld-eval --model models/latentlearner \
  --server-url http://localhost:8000/v1 --served-model latentlearner \
  --output results/latentlearner
```

`--variant` selects an environment. `--limit` selects the first requested number
of episodes from its fixed order. The launcher saves success metrics and
`summary.json`. `--gain-analysis` additionally saves numeric measurements for
[gain compensation analysis](analysis.md).

## Ablation evaluation

SFT, RL, paradigm-ablation, and data-ablation checkpoints use the same launcher.
For prompt ablation, use the same 50 episodes in the two egocentric environments.
The prompted condition adds both exploration and egocentric instructions:

```bash
python -m gui_agent_captcha.eval.exploration_depth_astra "$BENCHMARK_MANIFEST" \
  --profile gpt6_astra --variants ten_choice_egocentric drag_egocentric \
  --limit-per-variant 50 --output-root results/prompt-baseline
python -m gui_agent_captcha.eval.exploration_depth_astra "$BENCHMARK_MANIFEST" \
  --profile gpt6_astra_explore_egocentric --variants ten_choice_egocentric drag_egocentric \
  --limit-per-variant 50 --output-root results/prompt-explore
```

Grounding evaluation uses `latentguiworld.grounding_eval` with the dataset
layouts described in [data interfaces](data.md).
