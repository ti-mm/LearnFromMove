# Learn from Move: the Next Step for GUI Agents

This repository provides the environments, evaluation code, and training code
for **Learn from Move: the Next Step for GUI Agents**. It includes the
LatentGUIWorld benchmark, LatentLearner, and the MOVE approach to GUI grounding.

## Install

Run these commands from this directory. Editable installation keeps the scene
assets, configurations, and source code under the same root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m playwright install --with-deps chromium
```

GPU evaluation additionally uses `pip install -e '.[eval]'`. GRPO uses Python
3.12 and a CUDA 12.9 runtime:

```bash
pip install -r requirements/rl.txt
pip install -e third_party/verl
pip install -e '.[eval]'
```

The bundled VERL supports action-level policy optimization for GUI interaction.

## Environments

```bash
python -m six_environments list
python -m six_environments smoke --output results/environment-smoke
python -m six_environments serve --port 8765
```

Each environment has 150 test episodes at 1280 × 720 resolution, for 900 episodes.
Rotation uses a horizontal slider to rotate the inner or outer image region. Evaluation scores the first release attempt.

| Family | Environment identifiers |
|---|---|
| Ten-Choice | `ten_choice_egocentric`, `ten_choice_exocentric` |
| Rotation | `rotation_inner`, `rotation_outer` |
| Drag | `drag_egocentric`, `drag_exocentric` |

The benchmark scenes are under `data/formal_benchmarks/`.
Per-environment descriptors are under `environments/`. The benchmark dataset is
available on [Hugging Face](https://huggingface.co/datasets/OpenMOSS-Team/LearnFromMove).

## Evaluation

The evaluation launcher runs models on the paper's six environments and reports
per-environment success rates.

Use a local Hugging Face LatentLearner checkpoint:

```bash
latentguiworld-eval --model models/latentlearner --output results/latentlearner
```

For a vLLM server that exposes the same checkpoint:

```bash
latentguiworld-eval --model models/latentlearner \
  --server-url http://localhost:8000/v1 --served-model latentlearner \
  --output results/latentlearner
```

`--variant` selects one environment and `--limit` selects a smaller episode count.
The evaluator uses greedy decoding, 12 turns, three screenshots, a 16,384-token
context, 128 generated tokens per turn, and the Think/action response format.
Results are saved as per-environment metrics and `summary.json`.

External-model evaluation includes GPT-5.6-Sol and GPT-6 Astra.
See [evaluation adapters](docs/evaluation-adapters.md) for model interfaces,
environment action mappings, and runtime endpoint configuration.

Evaluate the paradigm-ablation and data-ablation checkpoints with the same
launcher, passing their checkpoint directories to `--model`. Prompt ablation
uses the Astra runner's `--profile` and `--limit-per-variant` options with the
same benchmark manifest. See [ablation evaluation](docs/evaluation-adapters.md#ablation-evaluation).

Grounding evaluation supports ScreenSpot-Pro, ScreenSpot-v2, MMBench-GUI,
UI-Vision, and OSWorld-G. See [data formats](docs/data.md) for their directory layout.

```bash
python -m latentguiworld.grounding_eval --model models/grounding-move \
  --data-root data/grounding-benchmarks --mode moveto_leftclick \
  --output results/grounding-move
```

Use `--mode directclick` for the direct-click baseline. `--initial-cursor-map`
accepts the shared per-example cursor positions used for a matched comparison.

## Training

The launcher resolves the paper settings into the actual trainer command and
writes `resolved_config.json`. Run the same command without `--dry-run` to train.

```bash
latentguiworld-train --recipe benchmark_sft --model models/Qwen3.5-9B \
  --train-files data/training/benchmark/train.parquet \
  --output runs/benchmark-sft --dry-run

latentguiworld-train --recipe drag_rl --model models/latentlearner \
  --train-files data/training/drag/train.parquet \
  --val-files data/training/drag/validation.parquet \
  --output runs/drag-rl --dry-run
```

Recipes: `benchmark_sft`, `drag_rl`, `ten_choice_rl`, `rotation_rl`,
`grounding_directclick_sft`, `grounding_moveto_leftclick_sft`,
`grounding_directclick_rl`, and `grounding_moveto_leftclick_rl`.
SFT supports `--gpus`, `--nodes`, `--node-rank`, `--master-addr`, and `--master-port`.
Grounding direct-click and MOVE SFT default to four and two nodes of eight GPUs,
respectively. Start the same SFT command on every node with its corresponding rank.
For distributed RL, connect the worker nodes to a shared Ray cluster and launch
once on the head node with `RAY_ADDRESS=auto` and the desired `--nodes`.
The dataset and output directories must be available at the same path on each node.

Paired RL defaults to eight GPUs. Its 16 prompts × 5 rollouts give 80 rollout
samples; the step-sample manager also uses 80 action samples per optimizer
minibatch. `ppo_mini_batch_size=16` is multiplied by the five-rollout factor
inside VERL. All paired recipes enforce the 384-token generation cap.

See [training settings](docs/training.md) for the complete parameter table,
checkpoint selection, and training-data preparation modules.

## Analysis

The paired RL recipes share one GRPO training entry point. Pearson analysis
computes episode-median gain compensation for SFT and RL checkpoints. See
[RL and gain compensation analysis](docs/analysis.md) for commands and formulas.

## Verification

```bash
pytest tests -q
python -m six_environments smoke --output results/environment-smoke
```

## License and attribution

Original code is released under Apache-2.0. VERL retains its license
and upstream notices in `third_party/`. See [attribution](NOTICE.md)
for the environment assets and external datasets.
