# Training settings

The JSON files in `configs/` supply the launcher's effective trainer parameters.

| Setting | Benchmark SFT | Paired benchmark RL | Grounding SFT | Grounding RL |
|---|---:|---:|---:|---:|
| Epochs | 3 | 1 | 2 | 1 |
| Learning rate | 3e-6 | 1e-6 | 3e-6 | 5e-7 |
| Schedule | cosine | cosine | cosine | cosine |
| Warmup ratio | 0.08 | 0.08 | 0.05 | 0.05 |
| Weight decay | 0.05 | 0.05 | 0.1 | 0.01 |
| Gradient clipping | 1 | 1 | 1 | 1 |
| Batch | 64 examples | 16 prompts | 128 examples | 8 prompts |
| Microbatch/GPU | 1 | 1 | 1 | 1 |
| Sequence length | 16384 | 16384 | 32768 | 16384 |
| Rollouts/prompt | — | 5 | — | 8 |
| Temperature / top-p | — | 1 / 0.95 | — | 1 / 0.95 |
| KL loss coefficient | — | 0.005 | — | 0.01 |

Benchmark SFT uses 24,000 trajectories, 4,000 per environment, yielding 154,713
supervised examples. Targets contain a Think rationale and one primitive action;
input retains the latest three screenshots and preceding actions. Grounding SFT
uses 33,353 shared records: 30,018 one-move, 1,668 two-move, and 1,667 three-move
trajectories. Direct-click uses 33,353 final-click examples. MOVE uses 35,020
windows and masks actions already supervised in earlier windows. Both grounding
tracks use action-only targets, BF16, FlashAttention 2, seed 1, and FSDP2.

Each paired RL dataset contains 5,120 prompts, 2,560 per variant, shuffled for one
epoch. AdamW uses betas (0.9, 0.95), minimum LR ratio 0.1, PPO clip 0.2, one PPO
epoch, and entropy coefficient 0. Task success contributes 1 and valid trajectory
format contributes 0.1. Generation/context exhaustion in Drag and Ten-Choice
receives zero reward. Context is 16,384 tokens, with an 8,192-token trajectory
response budget, a 384-token per-turn cap, a 32-token reserve, 12 turns, and three
screenshots. KL regularization is a separate loss.

Grounding RL uses 10,000 instructions excluded from the SFT instruction set.
Rewards are -0.2 for invalid format, 0 for a valid wrong answer, and 1 for a valid
correct answer. MOVE allows four actions and three screenshots; direct-click
allows one coordinate-bearing click.

## Validation and model selection

Use the separate validation data for checkpoint selection. Benchmark SFT selects
by mean success across the six environments. Single-environment specialists
select on their own variant. Paired Drag and Ten-Choice select by the mean of both
views, and paired Rotation selects on inner-ring validation. Ties select the
later checkpoint. Each validation environment contains 150 episodes.

`--save-freq` controls checkpoint retention frequency; every saved checkpoint is
retained by default. After evaluation on the validation data, use the selected
checkpoint for the test run.

## Data preparation modules

The training inputs are local Parquet files and their referenced images. The
following modules implement the paper's transformations:

- `benchmarks.exploration_depth.training_manifest`: scene construction.
- `benchmarks.exploration_depth.training_oracle`: successful primitive trajectories.
- `benchmarks.exploration_depth.training_teacher_thinks`: teacher rationales.
- `benchmarks.exploration_depth.training_with_think`: SFT message serialization.
- `benchmarks.exploration_depth.training_ablations`: action-space variants.
- `benchmarks.exploration_depth.six_env_rl_dataset`: task-only GRPO prompts.
- `data.groundcua46k_no_think_paired_rebalance`: shared 90:5:5 record selection.
- `data.groundcua46k_paired90_single_directclick`: final-click serialization.
- `data.groundcua46k_windows`: MOVE sliding windows and supervision masks.
- `data.build_groundcua_paper_style_verl_rl`: SFT-excluded grounding RL selection.
- `data.export_groundcua_paper_style_verl`: track-specific GRPO serialization.

All module names above are relative to `gui_agent_captcha`. Their CLI help and
function signatures expose source/output paths. Set `GUI_CAPTCHA_STORAGE_ROOT`
to organize preparation assets under a custom root; the default is this checkout.
