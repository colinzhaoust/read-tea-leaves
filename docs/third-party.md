# Third-party code, models, and data sources

Nothing third-party is vendored in this repository: no checkpoints, no model weights, no
wheels, and no upstream source trees. The table lists what the paper's experiments used and
the license evidence recorded for each. "Not recorded" means the project holds no license
record for the item; it is not a statement about the item's actual terms.

## Code and models

| Component | Use | Revision | License evidence |
|---|---|---|---|
| LC-PFN (Adriaensen, Rakotoarison, Müller, Hutter), code | curve-fitting baseline | `ba892f6f451027f69c50edf00c765ded98c75d30` | [MIT](https://github.com/automl/lcpfn/blob/ba892f6f451027f69c50edf00c765ded98c75d30/LICENSE) |
| LC-PFN, pretrained checkpoint | same | file `pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000` | no checkpoint-specific license recorded; obtain it from the authors' release, it is not redistributed here |
| Qwen3.5-2B | base model of TeaReader-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/LICENSE) |
| Qwen3.5-9B | base model of TeaReader-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Apache-2.0 (license tag of the [model repository](https://huggingface.co/Qwen/Qwen3.5-9B)) |
| Muse-Glimmer-30B | evaluated open-weight model | `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` | [Apache-2.0](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/LICENSE), with a separate [usage policy](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/USAGE_POLICY.md) |
| FT-PFN, Chronos-2, TimesFM-2.5, NCPL, NeuNeu | baselines | identifiers in the paper's appendix on baselines | not recorded |
| Other evaluated LLMs | forecasters | provider identifiers and pinned commits in the paper's appendix on models | not recorded |
| NumPy, SciPy, huggingface_hub (and, optionally, the Anthropic SDK) | dependencies, installed by the user | see `pyproject.toml` | each package's own license |

A public repository, an accessible model card, or a model license does not by itself
establish permission to redistribute a checkpoint, the logs of training runs, or the
datasets behind them.

## Data sources

| Source | What is used | License evidence |
|---|---|---|
| Public experiment-tracker projects (pretraining, fine-tuning, and RL runs) | logged metric histories and configurations | no license field on tracker runs; used under the terms on which their owners made them public; not audited run by run |
| Public Hugging Face repositories with a trainer-state file (fine-tuning runs, community set) | logged metric histories and configurations | no license field recorded per run; not audited run by run |
| Speedrun log archive (github.com/KellerJordan/modded-nanogpt) | training logs (the speedrun cell of the matching release) | MIT |
| Source repositories of the projects | short code excerpts shown in the Full condition, each with its source URL, line range, and file hash | upstream terms of each repository; not recorded by this project |

Consent was screened in the RL set only: runs from individual or unattributable accounts,
and runs of named researchers without an associated paper or official repository, were
excluded before any model was evaluated. The pretraining and fine-tuning release and the
community set were not screened for consent. Community runs are analyzed in aggregate and
are not attributed to individual authors.

In the reader's training export, 42 views from 6 run identities carry a redistribution hold
recorded from their sources (32 views: unknown repository license; 10 views: training
dataset under CC BY-NC 4.0 with model metadata under MIT). They are not part of this
repository.

## In this repository

| File | Third-party content |
|---|---|
| `examples/paper-example/input.json`, `request.txt` | prefix values and configuration of one public tracker run of the Marin project, and a 16-line excerpt of github.com/marin-community/marin with its file hash, as printed in the paper |
| `examples/paper-example/answer.json` | the quantile forecast returned by one evaluated model for that question, as printed in the paper |
