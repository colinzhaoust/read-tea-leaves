"""Add your own forecaster in a few lines:  python examples/custom_forecaster.py

A forecaster maps a question (a dict with the observed `prefix`, the `query_x` to predict, and,
in the Full condition, the run's `context`) to one median per query point. This one continues
the straight line through the last two prefix points and never goes below zero.
"""
import tea_leaves as tea


def linear_extrapolation(question):
    (x0, y0), (x1, y1) = [(p['x'], p['value']) for p in question['prefix'][-2:]]
    slope = (y1 - y0) / (x1 - x0)
    return [{'x': x, 'p50': max(0.0, y1 + slope * (x - x1))} for x in question['query_x']]


test = tea.load()                                     # downloads the test set on first use
forecasts = [{'case': q['case'], 'predictions': linear_extrapolation(q)} for q in test.questions('curve')]
print(tea.report(tea.evaluate(forecasts, test, name='linear extrapolation')))
