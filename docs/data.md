# Data interfaces

## Benchmark scenes

The benchmark under `data/formal_benchmarks/` contains the paper's six
environments. Its `manifest.json` enumerates 900 episodes and their scene files,
with 150 episodes per environment. The `cases/` and `assets/` subdirectories
contain the rendering resources.

Use the runtime to locate the default manifest:

```python
from gui_agent_captcha.benchmarks.exploration_depth.contracts import default_formal_manifest_path

manifest_path = default_formal_manifest_path()
```

## Hugging Face benchmark

Download [OpenMOSS-Team/LearnFromMove](https://huggingface.co/datasets/OpenMOSS-Team/LearnFromMove)
with `snapshot_download`, and pass its `manifest.json` to the runtime. The dataset
uses `drag_egocentric`, `drag_exocentric`, `ten_choice_egocentric`, and
`ten_choice_exocentric`, alongside `rotation_inner` and `rotation_outer`.

```bash
python -m six_environments list --manifest benchmark/manifest.json
python -m six_environments smoke --manifest benchmark/manifest.json --variant ten_choice_egocentric
latentguiworld-eval --manifest benchmark/manifest.json --variant drag_egocentric \
  --model models/latentlearner --output results/drag-egocentric
```

## Test-set composition

The GitHub and Hugging Face benchmarks use the same test-set labels. Each Drag
variant contains 75 IID episodes using training shape categories and 75 OOD
episodes using held-out shape categories. Rotation and Ten-Choice use `null`
distribution values, with no IID/OOD subdivision. In the runtime manifest this
label is stored in `split`; Hugging Face exposes every subset as a `test` split.

All Ten-Choice test scenes draw from 240 messages disjoint from the 80 training
messages. Rotation uses 150 test background images disjoint from the 1,000
training backgrounds. Both tasks evaluate held-out content across their full
test sets.

## SFT Parquet

Benchmark SFT rows contain `messages` and `images`. The messages are a user
instruction/history followed by the target assistant Think/action response.
Images are local image entries in chronological order, capped at the latest three.
Grounding MOVE windows additionally mark supervised assistant messages with
`trainable: true`; earlier assistant responses supply history. The custom
`GroundCUAWindowSFTDataset` translates these annotations into token loss masks.

`training_with_think` serializes the benchmark format from successful trajectories
and teacher responses. The GroundCUA preparation modules select the same records
for direct-click and MOVE, preserving source IDs and final click coordinates.

## GRPO Parquet

Benchmark rows contain a prompt, `agent_name`, `data_source`, and `extra_info`
with task configuration. The configuration references a training manifest and
its environment instance. `six_env_rl_dataset` produces these task-only rows;
combine 2,560 rows from each of the family's two variants and shuffle for one
epoch. Use training scenes distinct from validation and test scenes.

Grounding rows contain the instruction, local screenshot, target geometry, and
track metadata. `build_groundcua_paper_style_verl_rl` selects 10,000 instructions
excluded from the supplied SFT annotations. `export_groundcua_paper_style_verl`
converts the same selection to direct-click and MOVE tracks.

## Grounding evaluation layout

Place the benchmark's original annotation/image files in these directories:

```text
data/grounding-benchmarks/
  ScreenSpot-Pro/annotations/
  ScreenSpot-Pro/images/
  ScreenSpot-v2/
  mmbench-gui/
  ui-vision/
  osworld-g/data/test-00000-of-00001.parquet
```

The loaders in `eval/groundcua_table2_qwen3_direct.py` define each source schema.
Full evaluation expects 1,581 ScreenSpot-Pro, 1,272 ScreenSpot-v2, 3,594
MMBench-GUI, 5,479 UI-Vision, and 510 OSWorld-G examples. A shared initial-cursor
map uses the `paired90_training_cursor_position_map_v1` schema consumed by
`_load_cursor_map` in `eval/qwen3vl_paired90_five_bench.py`.
