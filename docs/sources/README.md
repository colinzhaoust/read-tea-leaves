# Sources and acknowledgments

Tea Leaves exists because people train models in the open. Every curve in the benchmark and
in the training data was logged by someone who chose to make an experiment tracker project
or a model repository public, often with the configuration and the code next to it. We are
grateful to all of them. This folder lists every project, repository, and owner behind the
data, with links.

| Page | What it lists |
|---|---|
| [tracker-projects.md](tracker-projects.md) | the experiment-tracker projects: the five projects of the pretraining and fine-tuning test set, the 24 projects of the RL test set, the projects of the matching pairs, and the tracker projects of the training file |
| [community-runs.md](community-runs.md) | the 232 Hugging Face repositories of the community test set, by owner (184 owners) |
| [training-runs.md](training-runs.md) | the 646 Hugging Face repositories in the training file of TeaReader-2B, by owner (363 owners) |

The curated test sets draw on the projects of Marin, DCLM, OLMoE, and Open-R1, and on the RL
projects of the labs and authors named on the first page; the paper cites the ones that have
a publication.

Twenty of the Hugging Face repositories were public when their logs were read and are no
longer publicly visible; they are listed without a link. If you own a run and want it
removed from the dataset, or want a name or a link corrected, please open an issue.

The pages are generated from the dataset by [`build_sources.py`](build_sources.py).
