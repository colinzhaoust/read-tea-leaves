# The data: where every run comes from

Everything is in the Hugging Face dataset
[`xinranz3/read-tea-leaves`](https://huggingface.co/datasets/xinranz3/read-tea-leaves). The
unit is the **run**: one training run, logged publicly by its owner. A **question** is one
run at one query grid (Grid10 asks for `x = 0.3, ..., 1.0`, Grid20 for `0.4, 0.6, 0.8, 1.0`).
Checkpoints, history rows, and pairs of runs are never counted as runs.

## Test sets (682 runs)

| Folder | Stage | Source | Where it was read from | Runs | Questions |
|---|---|---|---|---:|---:|
| `pt-ft/` | Pretraining | Marin | experiment tracker | 100 | 200 |
| | Pretraining | DCLM | experiment tracker | 25 | 50 |
| | Pretraining | OLMoE | experiment tracker | 25 | 50 |
| | Fine-tuning | Marin (Tülu 3 recipe) | experiment tracker | 100 | 200 |
| | Fine-tuning | Open-R1 | experiment tracker | 50 | 100 |
| `rl/` | RL | 24 projects of labs and papers (below) | experiment tracker | 150 | 297 |
| `community/` | Fine-tuning, preference tuning, RL | 232 repositories of 184 owners | Hugging Face Hub (trainer state) | 232 | 464 |
| | | | **Total** | **682** | **1,361** |

The paper's headline counts 5,456 forecast targets: the 682 runs at the eight points of
Grid10.

**RL (`rl/`).** 126 runs log a reward (to be maximized) and 24 a loss. Runs per project:

| Project (tracker path) | Runs | | Project (tracker path) | Runs |
|---|---:|---|---|---:|
| `openrlbenchmark/lm_human_preference_details` | 30 | | `ai2-llm/Olmo-3-7B-RL-Zero` | 3 |
| `huggingface/open-r1` | 24 | | `cjreinforce/openrlhf_train_ppo` | 3 |
| `apiche/watermelon` | 13 | | `dzmitry-bahdanau/watermelon` | 3 |
| `lkevinzc/gem` | 13 | | `hashimoto-group/DeepScholarRLTTT-ablations` | 3 |
| `ai2-llm/open_instruct_public` | 11 | | `hashimoto-group/metagrad-rl` | 3 |
| `rafa/watermelon` | 7 | | `ehsk/watermelon` | 2 |
| `costa-huang/huggingface` | 6 | | `axolotl-ai/gsm8k-grpo-r1` | 1 |
| `huggingface/trl` | 6 | | `dongfu/acecoder` | 1 |
| `jiayipan/tinyrl` | 6 | | `dongfu/torl` | 1 |
| `lkevinzc/gem-tinker_train` | 5 | | `hamishivi/tmax-public` | 1 |
| `reallm-labs/slime-dev` | 5 | | `hashimoto-group/DeepScholarRL-ablations` | 1 |
| | | | `hashimoto-group/verl_examples` | 1 |
| | | | `upup-ashton-wang-usc/Tina` | 1 |

In 17 of the 150 runs the logged series does not move after the first 20%, so repeating the
last value has no error and the run has no error ratio; RL skill is computed on the other
133 runs. The prefix is the first 20% as logged (its density varies by run), not ten window
means.

**Community (`community/`).** Runs found on the Hugging Face Hub through their
`trainer_state.json`, selected by a rule fixed before any model was run: 12 strata of kind of
run × kind of owner × date, at most two runs per owner per stratum.

| Kind of run | Owner | Before Apr 2025 | Apr to Dec 2025 | From Jan 2026 | Runs |
|---|---|---:|---:|---:|---:|
| Fine-tuning and preference tuning | individual | 19 | 25 | 25 | 69 |
| Fine-tuning and preference tuning | institution | 25 | 25 | 21 | 71 |
| RL | individual | 25 | 25 | 25 | 75 |
| RL | institution | 3 | 6 | 8 | 17 |
| | | | | **Total** | **232** |

These logs are sparse (a median of 7 raw points per window, against 59 in the curated sets),
which is why every method loses most of its skill here. `labels/core-211.json` lists the 211
runs that Figure 2 of the paper scores (the runs every first-evaluated model answered
validly); `tea score FILE --set community --subset core-211` uses it.

## Matching pairs (`matching/`, 340 pairs)

A pair is two runs of one sweep whose settings differ. Each pair is shown in two orders
(on RL also at two curve grids) and two conditions: 920 items, 1,840 requests.

| Stage | Pairs | Projects |
|---|---:|---|
| Pretraining | 150 | `marin-community/marin_moe` 134, `samir/dcnlp` (DCLM) 11, `ai2-llm/olmoe` 5 |
| Fine-tuning | 70 | `marin-community/marin` 52, `huggingface/open-r1` 9, `hamishivi/open_instruct` 9 |
| RL | 120 | 18 projects, the largest `apiche/watermelon` (31), `rafa/watermelon` (15), `costa-huang/huggingface` (11) |

## Training data (`train/`)

`train/tea-reader-2b/train.jsonl` is the training file of
[TeaReader-2B](https://huggingface.co/xinranz3/TeaReader-2B): 7,724 examples from 918 runs.
Each line has `messages` (the request and the target answer, as a ratio to the last observed
value) and `meta` (run, stage, source, sweep family, the observed share `f`, and the recorded
values). Every run is shown at up to five observed shares (5%, 10%, 20%, 30%, 50%) and two
query grids. Every test sweep family was removed from this pool before training; the paper's
appendix reports the links to test runs that remain under a broader definition.

| Kind of run | Where it was read from | Runs | Examples |
|---|---|---:|---:|
| RL | Hugging Face Hub (trainer state) | 366 | 3,376 |
| Pretraining, Marin MoE | experiment tracker | 216 | 1,296 |
| Fine-tuning | Hugging Face Hub (trainer state) | 210 | 1,858 |
| Preference tuning (DPO 49, DPO-like 10, KTO 7, ORPO 4) | Hugging Face Hub (trainer state) | 70 | 642 |
| Synthetic-task study (`ai2-llm/Olmo-Hybrid-synth-eval`) | experiment tracker | 52 | 520 |
| Other (2 pretraining runs from a Hub dataset, 2 RL runs of `axolotl-ai`) | Hub and tracker | 4 | 32 |
| **Total** | | **918** | **7,724** |

The Hub runs of the training file come from 363 repository owners. The larger pool behind
[TeaReader-9B](https://huggingface.co/xinranz3/TeaReader-9B) (8,727 runs) is not in the
dataset yet.

## Files of a test set

```
<set>/inputs/prediction.json          questions: {case, conditions: {curve_only, enough_info_available}}
<set>/labels/prediction-truth.json    recorded values at the query points (null where the run logged none)
<set>/labels/allocation.json          per run: stage, source, lineage component; how the set is scored
<set>/requests/...                    request texts as sent, where they cannot be rebuilt from the inputs
```

`run_key` is the public path of a run: the tracker path, or the Hub repository. Configuration
fields are kept as logged, and missing information is marked as unknown. Some configurations
contain file paths of their owners' machines, as their owners published them.

## How each set is scored

| Set | Cells | Grids | Bootstrap clusters | Absurd-answer filter |
|---|---|---|---|---|
| `pt-ft` | five: stage × source | each grid separately (the paper reports Grid10) | lineage components | on |
| `rl` | one | both grids pooled | runs | off |
| `community` | one | each grid separately (the paper reports Grid10) | repository owners | off |

These are the conventions of the paper; each set's `allocation.json` carries them, so
`tea score` applies them without options.

[`data-card.md`](data-card.md) gives the counts of the collection funnel, and
[`third-party.md`](third-party.md) the terms of the third-party material.
