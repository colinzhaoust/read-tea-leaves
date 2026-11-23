"""Answer validity.

``strict_validate`` is the validator used when the paper's models were run: it
raises on an invalid answer and returns the parsed object otherwise.

``lenient_parse`` (from ``reference.py``) is the rule that decides whether an answer is
scored: it ignores the rationale, accepts a JSON object embedded in text, repairs leading
decimals, and sorts crossing quantiles per point.
"""
import json
import math

from .reference import lenient_parse  # noqa: F401


def strict_validate(text, task, queries):
    # One outer code fence is tolerated uniformly, without repairing content.
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    d = json.loads(text)
    rationale = d['rationale']
    assert isinstance(rationale['uncertainty'], str) and rationale['uncertainty'].strip()
    for f in ['evidence', 'assumptions']:
        assert isinstance(rationale[f], list) and len(rationale[f]) <= 3
        assert all(isinstance(x, str) and x.strip() for x in rationale[f])
    def number(x):
        return type(x) in (int, float) and math.isfinite(x)
    if task == 'prediction':
        points = d['predictions']
        assert len(points) == len(queries)
        for point, x in zip(points, queries):
            assert number(point['x']) and abs(point['x'] - x) < 1e-9
            assert all(number(point[k]) for k in ['p10', 'p50', 'p90'])
            assert point['p10'] <= point['p50'] <= point['p90']
    else:
        assert number(d['probability_A_curve1']) and 0 <= d['probability_A_curve1'] <= 1
    return d
