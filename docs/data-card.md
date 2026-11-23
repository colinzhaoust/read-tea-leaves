# Tea Leaves: data card

Tea Leaves is a benchmark of forecasting questions about real language-model training runs.
Each question shows the first 20% of a run's metric curve, with or without its configuration
and semantics, and asks for the rest of the curve. This card states where the runs come from,
what each count means, and what is and is not in this repository. Every number is taken from
the paper; the section or appendix is given with it.

## Where the data is

| Item | Location |
|---|---|
| One complete question with a model answer and its recorded values | this repository, `examples/paper-example/` |
| The four test sets (682 runs), the matching pairs (340), and the training file of TeaReader-2B (918 runs) | [Hugging Face dataset](https://huggingface.co/datasets/xinranz3/read-tea-leaves); tables in [data.md](data.md) |
| Trained readers | [TeaReader-2B](https://huggingface.co/xinranz3/TeaReader-2B), [TeaReader-9B](https://huggingface.co/xinranz3/TeaReader-9B) |
| Raw answers of the evaluated models, database of runs used for retrieval, training pool of TeaReader-9B | later releases |

## Test sets

The independent unit is the run. (Paper, Section 3 and the appendix on data collection.)

| Set | Metric (axis) | Sources | Runs |
|---|---|---|---:|
| Pretraining | loss (steps, tokens) | Marin, DCLM, OLMoE | 150 |
| Fine-tuning | loss (steps, epochs) | Marin (Tülu 3 recipe), Open-R1 | 150 |
| RL | reward, loss (steps, epochs) | more than twenty lab and paper repositories | 150 |
| Community ("in the wild") | mixed | Hugging Face Hub, 184 authors | 232 |
| Total | | | 682 |

- The pretraining and fine-tuning test set of the paper holds 300 forecasting runs and, in
  its own frozen files, 300 matching pairs (150 per stage; these are not the pairs of the
  paper's matching table, which are the 340 pairs of the released `matching/` folder, see
  [data.md](data.md)), drawn from 589 distinct run identities; lineage closure
  reserves 609 identities. The five cells are pretraining × {Marin, DCLM, OLMoE} with
  100, 25, and 25 runs and fine-tuning × {Marin, Open-R1} with 100 and 50 runs.
- The 150 RL runs come in two parts with no run in common (96 and 54 runs).
  In 17 of the 150 runs the logged series does not move after the prefix, so repeating the
  last value has zero error and no error ratio exists; RL skill is computed on the remaining
  133 runs for every method.
- The community set holds 232 runs by 184 authors, drawn from 1,513 Hub candidates after 36
  were removed for an author or token collision with the other sets.
- Matching results use 340 pairs: the 150 pretraining and 70 fine-tuning main pairs of a
  330-pair matching release (whose other 110 pairs form a speedrun cell and two extension
  sets), and 120 RL pairs.

The pretraining, fine-tuning, and RL test sets were frozen with content hashes before any
model was evaluated. For the community set, the selection rule and the requests were fixed
before any model was run on it.

## Views and requests

A view is one run at one query grid (Grid10: `x = 0.3, ..., 1.0`; Grid20: `x = 0.4, 0.6,
0.8, 1.0`). A request is one view in one input condition for one model. For each LLM the
pretraining and fine-tuning release gives 1,200 forecasting requests (300 runs, two grids,
the Curve and Full conditions) and 2,400 matching requests on its own 300 pairs (two orders,
two conditions, two grids); the released matching task has 1,840 requests on 340 pairs. These
are counts of requests, not of runs.

## Source inventories and the training pool

These inventories have different scopes and are not a deduplicated count of independent
experiments. (Paper, appendix on data collection.)

| Count | Unit | Value |
|---|---|---:|
| Tracker index | run identities recorded across 236 collection-log files, including metadata-only and failed lookups | 19,650 |
| Hugging Face Hub listing | distinct models returned by a capped listing | 47,929 |
| Hub repositories with a trainer state | ungated repositories after removing quantized, converted, and bulk-uploaded ones | 14,370 |
| Hub repositories fetched and parsed | repositories | 4,503 |
| Speedrun archive | parsed log records (1,476 distinct recorded run IDs) | 1,516 |
| Training-pool universe | collected records before filtering (16,650 distinct pool IDs) | 17,805 |
| Strict training pool | pool IDs after removing every test sweep family | 1,463 |

## What a question contains

`examples/paper-example/input.json` is one complete input object in the Full condition:
`case` (a 24-hex identifier), `grid`, `axis`, `metric`, `prefix` (ten `{x, value}` points),
`query_x`, `task_semantics`, and `context` with `reported_configuration`, `task_description`,
`source_facts`, `known_unknowns`, `interpretation_rule`, and `code_context` (excerpts with
source URL, line range, and file hash). The request text is the prompt, a line `INPUT:`, and
this object as one line of compact JSON with sorted keys (`tea_leaves/request.py`).

Labels are the recorded metric values at the query points. In the frozen releases they are
kept apart from the inputs, together with the run-to-cell and run-to-lineage maps that the
scorer needs (`labels/` in the dataset).

## Provenance, licensing, and consent

Every run in the evaluation sets is recorded with its source (tracker project, Hugging Face
repository, or log archive) and run identifier. A license is recorded per source only for the
speedrun log archive (MIT). Tracker and Hugging Face runs carry no license field; they are
used under the terms on which their owners made them public, and their licenses were not
audited run by run. Consent to redistribution was screened in the RL set only
([third-party.md](third-party.md)). Configuration fields are kept as logged; missing information is marked
as unknown and is never filled in.

## Known limits

- Splits isolate recorded lineage components; author identity alone does not form a
  component, so author-level isolation is not guaranteed.
- The exposure check covers the files declared as used in development and the recorded
  relations; it is weaker than a certificate that a run was never seen.
- The test sets are restricted to runs logged densely enough for a ten-point prefix, which
  favors projects that log often.
- Anonymizing a request does not show that a model has not seen the run during its own
  training.
