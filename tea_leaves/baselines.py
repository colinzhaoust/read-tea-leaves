"""Baselines that see only the prefix: last value, power law, exponential, log-linear, and
the AIC-weighted ensemble of the three fits (AIC = n ln(SSE/n) + 2k; a failed fit falls back
to the last value)."""
import math

import numpy as np

from .reference import fit_family

METHODS = ['last_value', 'power_law', 'exponential', 'log_linear', 'aic_ensemble']


def windows(prefix):
    """Trailing 2% window means at x = 0.02, ..., 0.20 (empty windows omitted). On a
    pretraining or fine-tuning prefix, which is already these ten means, this returns it unchanged."""
    xs, ys = [], []
    for i in range(2, 21, 2):
        lo, hi = round(i / 100 - .02, 12), i / 100 + 1e-12
        win = [p['value'] for p in prefix if lo < p['x'] <= hi]
        if win:
            xs.append(i / 100)
            ys.append(sum(win) / len(win))
    return xs, ys


def forecast(question, maximize=False):
    """Return {method: [value per query_x]} for one input object.

    The fits see the ten 2% window means of the prefix; a metric to be maximized (an RL
    reward) is negated before fitting and the forecast negated back. The last value is the
    last point of the prefix as given."""
    sign = -1.0 if maximize else 1.0
    wx, wy = windows(question['prefix'])
    x = np.array(wx, float)
    y = sign * np.array(wy, float)
    qx = np.array(question['query_x'], float)
    last = float(question['prefix'][-1]['value'])
    preds = {'last_value': [last] * len(qx)}
    weights = {}
    for name, (pred, sse, k, status) in fit_family(x, y, qx).items():
        if status == 'ok':
            preds[name] = [float(sign * v) for v in pred]
            weights[name] = len(y) * math.log(max(sse, 1e-300) / len(y)) + 2 * k
        else:
            preds[name] = [last] * len(qx)
    if weights:
        best = min(weights.values())
        w = {n: math.exp(-.5 * (a - best)) for n, a in weights.items()}
        tot = sum(w.values())
        preds['aic_ensemble'] = [float(sum(w[n] / tot * preds[n][i] for n in w)) for i in range(len(qx))]
    else:
        preds['aic_ensemble'] = [last] * len(qx)
    return preds


def run(method, questions, maximize=()):
    """Forecast rows, ready for `evaluate`, for a list of input objects. `maximize`: the
    cases whose metric is maximized (TestSet.maximize)."""
    if method not in METHODS:
        raise ValueError(f'unknown baseline {method!r}; choose from {METHODS}')
    return [{'case': q['case'], 'predictions': [{'x': x, 'p50': v} for x, v in
                                                zip(q['query_x'], forecast(q, q['case'] in maximize)[method])]}
            for q in questions]
