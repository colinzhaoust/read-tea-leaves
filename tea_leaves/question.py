"""Turn your own run into a Tea Leaves question.

The curve is put on the axis x = step / planned budget. The prefix is the benchmark's: ten
points at x = 0.02, 0.04, ..., 0.20, each the mean of the logged values in the trailing
window (x - 0.02, x]. If the run is already logged past a query point, the recorded value
there is computed the same way, so a finished run can be scored.
"""
import csv
import json
from pathlib import Path

import numpy as np

WINDOW = 0.02
PREFIX_X = [round(WINDOW * i, 2) for i in range(1, 11)]
GRIDS = {10: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 20: [0.4, 0.6, 0.8, 1.0]}
INTERPRETATION = ('Reported per-run configuration takes precedence over source example defaults. '
                  'Project-level code is explanatory context, not proof of execution. '
                  'Missing information is unknown, not zero.')


def window_mean(xs, ys, t):
    mask = (xs > t - WINDOW - 1e-12) & (xs <= t + 1e-12)
    return float(np.mean(ys[mask])) if mask.any() else None


def read_curve(path, step_col=None, value_col=None):
    """Read (steps, values) from a CSV file with a header row.

    The step column defaults to the first one named step, _step, global_step or iteration (or
    the first column), the value column to the first other column.
    """
    with open(path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f'{path} has no rows')
    cols = list(rows[0])
    step_col = step_col or next((c for c in cols if c.lower() in ('step', '_step', 'global_step', 'iteration')),
                                cols[0])
    value_col = value_col or next(c for c in cols if c != step_col)
    steps, values = [], []
    for r in rows:
        try:
            s, v = float(r[step_col]), float(r[value_col])
        except (TypeError, ValueError):
            continue  # rows where the metric was not logged
        if np.isfinite(s) and np.isfinite(v):
            steps.append(s)
            values.append(v)
    return steps, values, value_col


def make_question(steps, values, budget, metric='train/loss', config=None, description=None, code=None,
                  grid=10, case='my-run'):
    """Build a question from a logged curve.

    steps, values   the logged metric (at least the first 20% of the run)
    budget          the planned length of the run, in the unit of `steps`
    config          optional dict of settings; with it or `description`/`code` the question is
                    in the Full format, otherwise it is curve-only
    Returns (question, labels). `labels` is None unless the curve is logged at every query point.
    """
    xs = np.asarray(steps, float) / float(budget)
    ys = np.asarray(values, float)
    keep = (xs >= 0) & (xs <= 1 + 1e-9)
    order = np.argsort(xs[keep], kind='stable')
    xs, ys = xs[keep][order], ys[keep][order]
    prefix = [window_mean(xs, ys, t) for t in PREFIX_X]
    missing = [t for t, v in zip(PREFIX_X, prefix) if v is None]
    if missing:
        raise ValueError(f'no logged value in the window ending at x = {missing[0]} (step {missing[0] * budget:g}). '
                         'A question needs at least one point in every 2% of the planned budget up to 20%; '
                         'check that `budget` is the planned length of the run.')
    question = {'case': case, 'grid': grid,
                'prefix': [{'x': t, 'value': v} for t, v in zip(PREFIX_X, prefix)],
                'query_x': GRIDS[grid], 'axis': 'recorded step divided by the planned step budget',
                'metric': metric, 'task_semantics': 'Predict the recorded metric. Missing context is unknown.'}
    if config or description or code:
        context = {'task_description': description or 'Not provided.',
                   'reported_configuration': dict(config or {}, planned_step_budget=budget),
                   'source_facts': [], 'known_unknowns': [], 'interpretation_rule': INTERPRETATION}
        if code:
            context['code_context'] = [{'language': 'python', 'code': code}]
        question['context'] = context
    targets = [window_mean(xs, ys, t) for t in GRIDS[grid]]
    labels = None if any(v is None for v in targets) else {'case': case, 'targets': targets}
    return question, labels


def from_files(csv_path, budget, config_path=None, **kwargs):
    """`make_question` from a CSV file and an optional JSON file of settings."""
    steps, values, value_col = read_curve(csv_path, kwargs.pop('step_col', None), kwargs.pop('value_col', None))
    config = json.loads(Path(config_path).read_text()) if config_path else None
    kwargs.setdefault('metric', value_col)
    kwargs.setdefault('case', 'my-run')
    return make_question(steps, values, budget, config=config, **kwargs)
