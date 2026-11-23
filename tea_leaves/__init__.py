"""Tea Leaves: forecast a training run from its first 20%, and score the forecast.

    import tea_leaves as tea

    test = tea.load()                                  # pretraining and fine-tuning test set
    forecasts = [{"case": q["case"],
                  "predictions": [{"x": x, "p50": q["prefix"][-1]["value"]} for x in q["query_x"]]}
                 for q in test.questions("curve")]
    print(tea.report(tea.evaluate(forecasts, test, name="last value")))
"""
from .data import CONDITIONS, TestSet, download, load
from .question import make_question
from .request import build_request
from .score import evaluate, report

__all__ = ['CONDITIONS', 'TestSet', 'build_request', 'download', 'evaluate', 'load', 'make_question', 'report']
__version__ = '0.1.0'
