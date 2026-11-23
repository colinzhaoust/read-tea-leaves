"""The skill score S.

For one question, the median forecast is scored by its mean absolute error over the query
points, divided by the error of repeating the last observed value:

    rho = MAE(forecast) / MAE(last value)

Ratios are averaged geometrically over the questions of a cell (a stage and a source) and
then over cells:

    R = exp( mean over cells of ( mean over questions of ln rho ) ),      S = 1 - R

S is 0 for repeating the last value, 1 for a perfect forecast, and negative for a forecast
worse than the last value. A question whose last-value error is zero has no ratio and does
not enter R. Answers that cannot be parsed are not scored, and neither are model answers whose
median leaves the range of the observed prefix widened by five ranges on each side (absurd
answers); both are counted and reported.
The 95% interval is a bootstrap over lineage components (2,000 resamples), so runs that
share part of a trajectory are resampled together. The bootstrap is seeded by the name of
the forecaster, the grid and the cell, so a rerun gives the same interval.

Parsing, the per-question metrics and the bootstrap are the functions of ``reference.py``.
"""
import json
import math
from pathlib import Path

import numpy as np

from . import reference as ref


# Display names of the stage-and-source cells (the keys are those of the data and of the bootstrap seed).
SOURCES = {'PT/marin': 'PT Marin', 'PT/mlfoundations': 'PT DCLM', 'PT/ai2': 'PT OLMoE',
           'SFT/marin': 'FT Marin', 'SFT/huggingface': 'FT Open-R1'}


def read_rows(path):
    """A JSON list, a single JSON object, or JSON lines."""
    text = Path(path).read_text(encoding='utf-8')
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    except ValueError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _inputs(questions):
    """{case: input object} from a path, a list of input objects, or a dict."""
    if isinstance(questions, dict) and 'case' not in questions:
        return questions
    rows = read_rows(questions) if isinstance(questions, (str, Path)) else questions
    rows = [rows] if isinstance(rows, dict) else rows
    return {q['case']: q.get('conditions', {}).get('curve_only', q) for q in rows}


def _labels(labels):
    """{case: {targets, cell, cluster}} from a path, a list of rows, or a dict."""
    if isinstance(labels, dict) and 'case' not in labels:
        return labels
    rows = read_rows(labels) if isinstance(labels, (str, Path)) else labels
    rows = [rows] if isinstance(rows, dict) else rows
    return {r['case']: {'targets': r['targets'], 'cell': r.get('cell'), 'cluster': r.get('cluster')} for r in rows}


def score_views(inputs, labels, forecasts, absurd_filter=True):
    """One record per answered question.

    With `absurd_filter`, a model answer (a row with `text`) that parses but whose median leaves
    the range of the observed prefix widened by five ranges on each side is not scored, as in the
    paper's tables. Numeric forecasts (rows with `predictions`) are never filtered."""
    by_case = {f['case']: f for f in forecasts if 'text' in f or 'predictions' in f}
    views = []
    for case, q in sorted(inputs.items()):
        f = by_case.get(case)
        if f is None or case not in labels:
            continue
        text = f['text'] if 'text' in f else json.dumps({'predictions': f['predictions']})
        try:
            payload, flags = ref.lenient_parse(text, 'prediction', q['query_x'])
        except ValueError as exc:
            views.append({'case': case, 'grid': int(q['grid']), 'status': 'invalid', 'reason': str(exc)})
            continue
        if absurd_filter and 'text' in f:
            band = ref.absurd_band([p['value'] for p in q['prefix']])
            if ref.is_absurd(band, payload['predictions'])[0]:
                views.append({'case': case, 'grid': int(q['grid']), 'status': 'absurd'})
                continue
        last = q['prefix'][-1]['value']
        lab = labels[case]
        m = ref.pred_metrics(payload['predictions'], lab['targets'], last)
        views.append({'case': case, 'grid': int(q['grid']), 'status': 'valid', 'flags': flags,
                      'cell': lab['cell'] or 'all', 'cluster': lab['cluster'], 'mae': m['mae'],
                      'mae_last_value': m['mae_last_value'], 'ratio': m['skill_ratio'],
                      'coverage80': m.get('coverage80')})
    return views


def aggregate(views, name, grid):
    """R, S and the 95% interval on S over the given views (all of one grid)."""
    views = [v for v in views if v['status'] == 'valid' and v['ratio'] is not None]
    if not views:
        return None
    cells = sorted({v['cell'] for v in views})
    clustered = all(v['cluster'] is not None for v in views)
    means, vecs, per_cell = [], [], {}
    for cell in cells:
        rs = [v for v in views if v['cell'] == cell]
        logs = [math.log(max(v['ratio'], 1e-12)) for v in rs]
        clusters = [v['cluster'] if clustered else v['case'] for v in rs]
        mean, vec, _ = ref.boot_vector(logs, clusters, ['skill', name, grid, cell])
        means.append(mean)
        vecs.append(vec if clustered else None)
        per_cell[cell] = {'S': 1 - math.exp(mean), 'with_ratio': len(rs)}
    r = math.exp(float(np.mean(means)))
    ci_r = ref.ci(np.mean(vecs, 0), math.exp) if all(v is not None for v in vecs) else None
    return {'S': 1 - r, 'S_ci95': [1 - ci_r[1], 1 - ci_r[0]] if ci_r else None, 'R': r,
            'with_ratio': len(views), 'cells': per_cell}


def evaluate(forecasts, data=None, name='forecast', questions=None, labels=None, absurd_filter=None,
             pool_grids=None, subset=None):
    """Score forecasts and return a summary dict.

    forecasts   a list of rows, or a path to a JSON or JSON-lines file. A row is
                {"case": ..., "text": "<raw model output>"} or
                {"case": ..., "predictions": [{"x": 0.3, "p50": ...}, ...]}.
    data        a TestSet or the folder of one (default: the pretraining and fine-tuning set,
                downloaded on first use). Alternatively pass `questions` and `labels` directly.
    name        the forecaster's name (it also seeds the bootstrap).
    absurd_filter   drop absurd model answers before scoring, as the paper does (default: the test
                set's own setting, True unless its allocation says otherwise; off for RL).
    pool_grids  score both query grids together under the key 'all' (default: the test set's
                own setting; the RL set pools them, as the paper does).
    subset      score only these cases (for example the community set's core-211 cohort).
    """
    forecasts = read_rows(forecasts) if isinstance(forecasts, (str, Path)) else list(forecasts)
    if questions is not None:
        inputs, labs, source = _inputs(questions), _labels(labels), 'custom questions'
    else:
        from .data import TestSet, load
        test = data if isinstance(data, TestSet) else (TestSet(data) if data else load())
        inputs, labs, source = test.inputs, test.labels, test.name
        pool_grids = test.pool_grids if pool_grids is None else pool_grids
        absurd_filter = test.absurd_filter if absurd_filter is None else absurd_filter
    absurd_filter = True if absurd_filter is None else absurd_filter
    if subset is not None:
        subset = set(subset)
        inputs = {c: q for c, q in inputs.items() if c in subset}
        source += f' ({len({c.rsplit("-g", 1)[0] for c in inputs})} runs of the subset)'
    views = score_views(inputs, labs, forecasts, absurd_filter)
    summary = {'name': name, 'data': source, 'grids': {}}
    groups = ['all'] if pool_grids and views else sorted({v['grid'] for v in views})
    for grid in groups:
        vs = [v for v in views if grid == 'all' or v['grid'] == grid]
        valid = [v for v in vs if v['status'] == 'valid']
        entry = {'questions': sum(1 for c, q in inputs.items() if (grid == 'all' or int(q['grid']) == grid) and c in labs),
                 'answered': len(vs), 'scored': len(valid),
                 'unparsed': sum(1 for v in vs if v['status'] == 'invalid'),
                 'absurd': sum(1 for v in vs if v['status'] == 'absurd'),
                 'no_ratio': sum(1 for v in valid if v['ratio'] is None)}
        entry.update(aggregate(vs, name, grid) or {'S': None, 'S_ci95': None, 'R': None, 'cells': {}})
        stages = sorted({v['cell'].split('/')[0] for v in valid if '/' in v['cell']})
        entry['stages'] = {s: aggregate([v for v in valid if v['cell'].split('/')[0] == s], name, grid)
                           for s in stages}
        summary['grids'][str(grid)] = entry
    summary['views'] = views
    return summary


def report(summary):
    """A short, readable report of an `evaluate` summary."""
    def fmt(entry):
        if not entry or entry.get('S') is None:
            return '   n/a'
        ci = entry.get('S_ci95')
        return f"{entry['S']:6.3f}" + (f'   95% interval [{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '')

    names = {'PT': 'pretraining', 'SFT': 'fine-tuning', 'RL': 'RL'}
    lines = []
    for grid, e in summary['grids'].items():
        lines.append(f"{summary['name']} on {summary['data']}, " + ('both grids' if grid == 'all' else f'Grid{grid}'))
        line = f"  scored {e['scored']} of {e['questions']} questions"
        if e['answered'] < e['questions']:
            line += f"; {e['questions'] - e['answered']} not answered"
        if e['unparsed']:
            line += f"; {e['unparsed']} answers could not be parsed"
        if e.get('absurd'):
            line += f"; {e['absurd']} removed as absurd"
        if e.get('no_ratio'):
            line += f"; {e['no_ratio']} without a ratio (the series does not move after the prefix)"
        lines.append(line)
        lines.append(f"  skill score S        {fmt(e)}")
        for stage, s in e.get('stages', {}).items():
            if len(e['stages']) > 1:
                lines.append(f"    {names.get(stage, stage):<19}{fmt(s)}")
        if len(e.get('cells', {})) > 1:
            lines.append('    by source: ' + ', '.join(f"{SOURCES.get(c, c)} {v['S']:.2f}" for c, v in e['cells'].items()))
    if not lines:
        lines.append(f"{summary['name']}: no forecast matched a question of {summary['data']}")
    lines.append('  (S = 0 is repeating the last observed value; S = 1 is a perfect forecast)')
    return '\n'.join(lines)
