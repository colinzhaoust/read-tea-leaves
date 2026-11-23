"""Reference functions of the paper's evaluation.

The functions below are copied without change from the scripts that produced the tables of
the paper: answer parsing (``lenient_parse``), the per-question metrics (``pred_metrics``),
the filter for absurd answers, the cluster bootstrap (``boot_vector``, ``ci``), and the parametric curve fits
(``fit_family``). The rest of the package is an interface on top of them. Change them only
if you intend to change the benchmark.

Lenient parsing, in order: parse the text as JSON (one outer code fence is tolerated); if
that fails, take the first balanced ``{...}`` object; if that fails, rewrite leading-decimal
numbers (``.5`` to ``0.5``). ``predictions`` must list exactly ``query_x`` in order with a
finite ``p50``; ``p10`` and ``p90`` are optional and are sorted per point if they cross. The
rationale is ignored.

A parsed answer of a model is then removed as absurd when any median lies outside the range
of the observed prefix widened by five ranges on each side (``absurd_band``, ``is_absurd``).

The bootstrap resamples lineage components within each cell (2,000 resamples, fixed seed).
"""
import json
import math
import re
import zlib

import numpy as np
from scipy.optimize import least_squares

SEED = 20260917
NBOOT = 2000
ABSURD_K = 5
STRING = re.compile(r'"(?:\\.|[^"\\])*"')
LEADING = re.compile(r'(?<![\w.])(-?)\.(\d)')


def first_balanced_object(text):
    start = text.find('{')
    if start < 0:
        raise ValueError('no object')
    depth, instr, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if instr:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    raise ValueError('unbalanced object')


def fix_leading_decimals(text):
    out, pos = [], 0
    for m in STRING.finditer(text):
        out.append(LEADING.sub(r'\g<1>0.\2', text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(LEADING.sub(r'\g<1>0.\2', text[pos:]))
    return ''.join(out)


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def lenient_parse(text, task, queries):
    """Return (payload, flags) or raise ValueError(reason)."""
    flags = []
    t = text.strip()
    if t.startswith('```') and t.endswith('```'):
        t = t.split('\n', 1)[1].rsplit('```', 1)[0].strip() if '\n' in t else t
    d = None
    try:
        d = json.loads(t)
    except ValueError:
        try:
            obj = first_balanced_object(t)
        except ValueError:
            raise ValueError('unparseable_json')
        flags.append('json_extracted')
        try:
            d = json.loads(obj)
        except ValueError:
            try:
                d = json.loads(fix_leading_decimals(obj))
                flags.append('leading_decimal_fixed')
            except ValueError:
                raise ValueError('unparseable_json')
    if not isinstance(d, dict):
        raise ValueError('not_an_object')
    if task == 'prediction':
        points = d.get('predictions')
        if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
            raise ValueError('predictions_missing_or_malformed')
        if len(points) != len(queries) or not all(number(p.get('x')) and abs(p['x'] - x) < 1e-9 for p, x in zip(points, queries)):
            raise ValueError('query_points_mismatch')
        if not all(number(p.get('p50')) for p in points):
            raise ValueError('nonfinite_or_missing_median')
        have = [('p10' in p and 'p90' in p) for p in points]
        out = []
        if all(have):
            if not all(number(p['p10']) and number(p['p90']) for p in points):
                raise ValueError('nonfinite_quantile')
            repaired = False
            for p in points:
                q = [p['p10'], p['p50'], p['p90']]
                if not q[0] <= q[1] <= q[2]:
                    q = sorted(q)
                    repaired = True
                out.append({'x': p['x'], 'p10': q[0], 'p50': q[1], 'p90': q[2]})
            if repaired:
                flags.append('quantile_repaired')
        else:
            flags.append('quantiles_missing')
            out = [{'x': p['x'], 'p50': p['p50']} for p in points]
        return {'predictions': out}, flags
    p = d.get('probability_A_curve1')
    if not (number(p) and 0 <= p <= 1):
        raise ValueError('probability_missing_or_out_of_range')
    return {'probability_A_curve1': p}, flags


def pred_metrics(preds, ys, last):
    """Additional per-request metrics.  preds: list of {p50[,p10,p90]}; last: prefix end value."""
    pairs = [(p, y) for p, y in zip(preds, ys) if y is not None]
    err = [p['p50'] - y for p, y in pairs]
    mae = sum(abs(e) for e in err) / len(err)
    mae_last = sum(abs(last - y) for _, y in pairs) / len(pairs)
    fy = pairs[-1][1]
    row = dict(mae=mae, rmse=math.sqrt(sum(e * e for e in err) / len(err)), final_abs_error=abs(err[-1]),
               final_rel_error=abs(err[-1]) / abs(fy) if fy else None, mae_last_value=mae_last,
               skill_ratio=mae / mae_last if mae_last > 1e-12 else None, pred_final=pairs[-1][0]['p50'], true_final=fy)
    if all('p10' in p for p, _ in pairs):
        pin = 0.
        for p, y in pairs:
            for q, k in ((.1, 'p10'), (.5, 'p50'), (.9, 'p90')):
                pin += max(q * (y - p[k]), (q - 1) * (y - p[k]))
        scale = sum(abs(y) for _, y in pairs)  # zero when an RL reward stays at exactly 0: WQL undefined
        row.update(mean_pinball=pin / (3 * len(pairs)), wql=2 * pin / (3 * scale) if scale else None,
                   coverage80=sum(p['p10'] <= y <= p['p90'] for p, y in pairs) / len(pairs),
                   width80=sum(p['p90'] - p['p10'] for p, _ in pairs) / len(pairs))
    return row


def rng_for(key):
    return np.random.default_rng([SEED, zlib.crc32(json.dumps(key, sort_keys=True).encode())])


def boot_vector(values, clusters, key):
    """Cluster bootstrap of the view-weighted mean.  Returns (mean, vector or None, n_clusters)."""
    values = np.asarray(values, float)
    ids = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(ids)}
    s = np.zeros(len(ids)); n = np.zeros(len(ids))
    for v, c in zip(values, clusters):
        s[pos[c]] += v; n[pos[c]] += 1
    mean = float(values.mean()) if len(values) else None
    if len(ids) < 2:
        return mean, None, len(ids)
    idx = rng_for(key).integers(len(ids), size=(NBOOT, len(ids)))
    return mean, s[idx].sum(1) / n[idx].sum(1), len(ids)


def ci(vec, transform=None):
    if vec is None:
        return None
    lo, hi = np.percentile(vec, [2.5, 97.5])
    if transform:
        lo, hi = transform(lo), transform(hi)
    return [float(lo), float(hi)]


def fit_family(x, y, qx):
    """Return {name: (pred, sse, k, status)} for the three parametric families."""
    res = {}
    ymin = float(np.min(y))
    hi = max(ymin, 1e-12)
    span = max(float(np.max(y) - np.min(y)), 1e-12)

    def run(name, fun, starts, lower, upper):
        best = None
        for p0 in starts:
            p0 = np.clip(np.array(p0, float), np.array(lower) + 1e-12, np.array(upper) - 1e-12)
            try:
                r = least_squares(lambda p: fun(p, x) - y, p0, bounds=(lower, upper), method='trf', max_nfev=2000)
            except Exception:
                continue
            sse = float(np.sum(r.fun ** 2))
            if np.isfinite(sse) and (best is None or sse < best[0]):
                best = (sse, r.x)
        if best is None:
            res[name] = (None, None, 3, 'fit_failed')
            return
        pred = fun(best[1], qx)
        if not np.all(np.isfinite(pred)):
            res[name] = (None, None, 3, 'nonfinite_prediction')
            return
        res[name] = (pred, best[0], 3, 'ok')

    lower_a = min(0.0, ymin - 1e-9)  # a in [0, min y] for positive losses; stays feasible if min y <= 0
    run('power_law', lambda p, t: p[0] + p[1] * np.power(t, -p[2]),
        [(0.8 * ymin, span * 0.2 ** c, c) for c in (0.1, 0.5, 1.0, 2.0)], [lower_a, 0.0, 0.01], [hi, np.inf, 5.0])
    run('exponential', lambda p, t: p[0] + p[1] * np.exp(-p[2] * t),
        [(0.8 * ymin, span, c) for c in (1.0, 5.0, 20.0, 60.0)], [lower_a, 0.0, 0.01], [hi, np.inf, 200.0])
    try:
        A = np.stack([np.ones_like(x), np.log(x)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = coef[0] + coef[1] * np.log(qx)
        sse = float(np.sum((A @ coef - y) ** 2))
        assert np.all(np.isfinite(pred)) and np.isfinite(sse)
        res['log_linear'] = (pred, sse, 2, 'ok')
    except Exception:
        res['log_linear'] = (None, None, 2, 'fit_failed')
    return res


def absurd_band(prefix_values):
    """Pre-declared plausibility band for a case: [min-5*range, max+5*range] over the shown prefix.

    A degenerate prefix (range 0) gets no band: the rule would then reject every answer that is
    not exactly the constant, which is not what it is for.  Such cases are reported as unguarded.
    """
    lo, hi = min(prefix_values), max(prefix_values)
    rng = hi - lo
    return None if rng <= 0 else (lo - ABSURD_K * rng, hi + ABSURD_K * rng)


def is_absurd(band, preds):
    """(flag, worst_excess_outside_the_band). Any p50 outside the band condemns the answer."""
    if band is None:
        return False, None
    lo, hi = band
    worst = 0.0
    for p in preds:
        worst = max(worst, lo - p['p50'], p['p50'] - hi)
    return worst > 0, worst
