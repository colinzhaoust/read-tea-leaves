"""Rebuild the pages of this folder from the dataset:  python docs/sources/build_sources.py [--data data] [--check]

--check asks the Hugging Face Hub which repositories are public today (one request each) and
writes the answers to docs/sources/hub-status.json; without it, the saved answers are used.
"""
import argparse
import collections
import json
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WANDB, HUB = 'https://wandb.ai/', 'https://huggingface.co/'


def tracker(path):
    return f'[`{path}`]({WANDB}{path})'


def hub_status(repos, check):
    path = HERE / 'hub-status.json'
    status = json.loads(path.read_text()) if path.exists() else {}
    if check:
        for repo in sorted(repos):
            req = urllib.request.Request(HUB + 'api/models/' + urllib.request.quote(repo), method='HEAD',
                                         headers={'User-Agent': 'tea-leaves-sources'})
            try:
                status[repo] = urllib.request.urlopen(req, timeout=30).status
            except urllib.error.HTTPError as exc:
                status[repo] = exc.code
        path.write_text(json.dumps({k: status[k] for k in sorted(status)}, indent=0) + '\n')
    return status


def owner_table(repos_by_owner, status, extra=None):
    """One row per owner: owner link, number of runs, repositories."""
    lines = ['| Owner | Runs | Repositories |', '|---|---:|---|']
    for owner in sorted(repos_by_owner, key=str.lower):
        cells = []
        for repo in sorted(repos_by_owner[owner], key=str.lower):
            name = repo.split('/', 1)[1]
            tag = f' ({extra[repo]})' if extra and extra.get(repo) else ''
            public = status.get(repo, 200) == 200
            cells.append((f'[{name}]({HUB}{repo})' if public else f'{name} (no longer public)') + tag)
        lines.append(f'| [{owner}]({HUB}{owner}) | {len(repos_by_owner[owner])} | ' + ', '.join(cells) + ' |')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default='data')
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    data = Path(args.data)

    # ---- tracker projects of the test sets
    ptft = collections.Counter()
    for key in {r['run_key'] for r in json.loads((data / 'pt-ft/labels/prediction-truth.json').read_text())}:
        ptft['/'.join(key.split('/')[:2])] += 1
    names = {'marin-community/marin_moe': 'Marin, pretraining', 'marin-community/marin': 'Marin, fine-tuning (Tülu 3 recipe)',
             'samir/dcnlp': 'DCLM, pretraining', 'ai2-llm/olmoe': 'OLMoE, pretraining',
             'huggingface/open-r1': 'Open-R1, fine-tuning'}
    rl = collections.Counter(r['source'] for r in json.loads((data / 'rl/labels/allocation.json').read_text())['runs'])
    pairs = {}
    for row in json.loads((data / 'matching/labels/matching-answers.json').read_text()):
        pairs[row['pair_id']] = (row['stage'], row['project'])
    match = collections.Counter(pairs.values())

    # ---- training file
    train = {}
    for line in (data / 'train/tea-reader-2b/train.jsonl').read_text().splitlines():
        meta = json.loads(line)['meta']
        train[meta['pool_run_id']] = meta
    train_hub, train_kind, train_tracker = collections.defaultdict(set), {}, collections.Counter()
    for meta in train.values():
        key = meta['run_key']
        if key.startswith('hf/'):
            repo = key[3:]
            train_hub[repo.split('/')[0]].add(repo)
            train_kind[repo] = {'SFT': 'fine-tuning', 'RL': 'RL'}.get(meta['stage'], 'preference tuning')
        else:
            train_tracker['/'.join(key.split('/')[:2])] += 1

    # ---- community set
    community = collections.defaultdict(set)
    kind = {}
    for run in json.loads((data / 'community/labels/allocation.json').read_text())['runs']:
        community[run['run_key'].split('/')[0]].add(run['run_key'])
        kind[run['run_key']] = 'RL' if run['run_stage'] == 'RL' else 'fine-tuning'
    all_repos = {r for v in community.values() for r in v} | {r for v in train_hub.values() for r in v}
    status = hub_status(all_repos, args.check)
    gone = sum(1 for r in all_repos if status.get(r, 200) != 200)

    out = ['# Tracker projects', '',
           'Runs read from publicly visible experiment-tracker projects. Each path links to the project.', '',
           '## Pretraining and fine-tuning test set (300 runs)', '', '| Project | Source | Runs |', '|---|---|---:|']
    out += [f'| {tracker(p)} | {names.get(p, "")} | {n} |' for p, n in sorted(ptft.items(), key=lambda x: (-x[1], x[0]))]
    out += ['', f'## RL test set ({sum(rl.values())} runs, {len(rl)} projects)', '', '| Project | Runs |', '|---|---:|']
    out += [f'| {tracker(p)} | {n} |' for p, n in sorted(rl.items(), key=lambda x: (-x[1], x[0].lower()))]
    out += ['', f'## Matching pairs ({sum(match.values())} pairs)', '', '| Project | Stage | Pairs |', '|---|---|---:|']
    out += [f'| {tracker(p)} | {s} | {n} |' for (s, p), n in sorted(match.items(), key=lambda x: (x[0][0] != 'pretraining', x[0][0], -x[1]))]
    out += ['', '## Training file of TeaReader-2B', '', '| Project | Runs |', '|---|---:|']
    for p, n in sorted(train_tracker.items(), key=lambda x: -x[1]):
        link = f'[`{p[3:]}`]({HUB}datasets/{p[3:]}) (a Hugging Face dataset)' if p.startswith('hf:') else tracker(p)
        out.append(f'| {link} | {n} |')
    (HERE / 'tracker-projects.md').write_text('\n'.join(out) + '\n')

    n_c = sum(len(v) for v in community.values())
    out = ['# Community test set: repositories and owners', '',
           f'{n_c} runs from {n_c} public Hugging Face repositories of {len(community)} owners, read from each '
           "repository's `trainer_state.json`. The kind of run is given in parentheses.", '']
    out += owner_table(community, status, kind)
    (HERE / 'community-runs.md').write_text('\n'.join(out) + '\n')

    n_t = sum(len(v) for v in train_hub.values())
    out = ['# Training file of TeaReader-2B: repositories and owners', '',
           f'{n_t} of the {len(train)} training runs come from public Hugging Face repositories of {len(train_hub)} '
           "owners, read from each repository's `trainer_state.json`. The other "
           f'{len(train) - n_t} runs come from the projects listed in [tracker-projects.md](tracker-projects.md).', '']
    out += owner_table(train_hub, status, train_kind)
    (HERE / 'training-runs.md').write_text('\n'.join(out) + '\n')
    print(json.dumps({'tracker_projects_ptft': len(ptft), 'rl_projects': len(rl), 'matching_projects': len({p for _, p in match}),
                      'community_repos': n_c, 'community_owners': len(community), 'train_hub_runs': n_t,
                      'train_hub_owners': len(train_hub), 'train_other_runs': len(train) - n_t,
                      'no_longer_public': gone}))


if __name__ == '__main__':
    main()
