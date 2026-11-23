"""Test sets: download from the Hugging Face Hub and read.

A test set is a folder with

    inputs/prediction.json              questions, each as {case, conditions: {<condition>: input object}}
    requests/curve_plus_config.json     the requests of the Curve+Config condition, as sent
    labels/prediction-truth.json        recorded values at the query points
    labels/allocation.json              stage, source and lineage component of every run
"""
import json
from pathlib import Path

HF_DATASET = 'xinranz3/read-tea-leaves'
DEFAULT_ROOT = 'data'
DEFAULT_SET = 'pt-ft'

# Short names used on the command line -> keys used in the data files.
CONDITIONS = {'curve': 'curve_only', 'config': 'curve_plus_config', 'full': 'enough_info_available'}

# Keys of the stage-and-source cells. The sources are Marin, DCLM (mlfoundations), OLMoE (ai2) and
# Open-R1 (huggingface); the paper's tables abbreviate them as Marin, MLF, AI2 and HF.
CELLS = {('pretraining', 'marin-community'): 'PT/marin', ('pretraining', 'mlfoundations'): 'PT/mlfoundations',
         ('pretraining', 'ai2-llm'): 'PT/ai2', ('SFT', 'marin-community'): 'SFT/marin',
         ('SFT', 'huggingface'): 'SFT/huggingface'}


def download(root=DEFAULT_ROOT):
    """Download the dataset into `root` (a few megabytes) and return the folder."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('Downloading the data needs huggingface_hub:  pip install huggingface_hub') from exc
    snapshot_download(HF_DATASET, repo_type='dataset', local_dir=str(root))
    return Path(root)


def condition_key(name):
    """Accept the short name (curve, config, full) or the key used in the data files."""
    if name in CONDITIONS:
        return CONDITIONS[name]
    if name in CONDITIONS.values():
        return name
    raise ValueError(f'unknown condition {name!r}; use one of {sorted(CONDITIONS)}')


class TestSet:
    """Questions and labels of one test set."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = self.folder.name
        self._items = json.loads((self.folder / 'inputs/prediction.json').read_text(encoding='utf-8'))
        self._labels = None
        self.pool_grids = False
        self.absurd_filter = True
        self.maximize = set()

    def __repr__(self):
        return f'TestSet({self.name!r}, {len(self._items)} questions)'

    def questions(self, condition='full', grid=10):
        """Input objects of one condition. `grid` is 10, 20, or None for both query grids."""
        key = condition_key(condition)
        if key == 'curve_plus_config':
            from .request import split_request
            items = [split_request(r['text'])[1] for r in self._requests_file()]
        else:
            items = [item['conditions'][key] for item in self._items]
        return [q for q in items if grid is None or int(q['grid']) == int(grid)]

    def requests(self, condition='full', grid=10):
        """The exact text to send for each question: a list of {case, grid, condition, text}."""
        from .request import build_request
        key = condition_key(condition)
        if key == 'curve_plus_config':
            rows = [{'case': r['case'], 'grid': int(r['grid']), 'condition': key, 'text': r['text']}
                    for r in self._requests_file()]
        elif (self.folder / 'requests/prediction.json').exists():
            # a set that ships its requests as sent (the RL set: input keys in their original order)
            rows = [r for r in json.loads((self.folder / 'requests/prediction.json').read_text(encoding='utf-8'))
                    if r['condition'] == key]
        else:
            rows = [{'case': q['case'], 'grid': int(q['grid']), 'condition': key, 'text': build_request(q)}
                    for q in self.questions(key, None)]
        return [r for r in rows if grid is None or r['grid'] == int(grid)]

    def _requests_file(self):
        path = self.folder / 'requests/curve_plus_config.json'
        if not path.exists():
            raise FileNotFoundError(f'{self.name} has no Curve+Config requests ({path})')
        return json.loads(path.read_text(encoding='utf-8'))

    @property
    def inputs(self):
        """{case: curve-only input object}: prefix, grid and query points of every question."""
        return {item['case']: item['conditions']['curve_only'] for item in self._items}

    @property
    def labels(self):
        """{case: {targets, cell, cluster}}."""
        if self._labels is None:
            folder = self.folder / 'labels'
            alloc = json.loads((folder / 'allocation.json').read_text())
            runs = {r['run_key']: r for r in alloc['runs']}
            # A set may pool its two query grids into one score (the RL set does).
            self.pool_grids = bool(alloc.get('pool_grids', False))
            # The paper scores the RL set without the absurd-answer filter.
            self.absurd_filter = bool(alloc.get('absurd_filter', True))
            self._labels = {}
            for row in json.loads((folder / 'prediction-truth.json').read_text()):
                run = runs[row['run_key']]
                if run.get('direction') == 'maximize':
                    self.maximize.add(row['case'])
                # An explicit cell (the RL set: one cell 'RL') overrides the stage-and-source cell.
                cell = run.get('cell') or CELLS.get((run['stage'], run['source']), f"{run['stage']}/{run['source']}")
                self._labels[row['case']] = {'targets': row['targets'], 'cell': cell,
                                             'cluster': run['component_id']}
        return self._labels


def load(name=DEFAULT_SET, root=DEFAULT_ROOT):
    """Return a test set, downloading the data on first use."""
    folder = Path(root) / name
    if not (folder / 'inputs/prediction.json').exists():
        print(f'Downloading {HF_DATASET} into {root}/ ...')
        download(root)
    if not (folder / 'inputs/prediction.json').exists():
        raise FileNotFoundError(f'no test set named {name!r} under {root}/')
    return TestSet(folder)
