"""End-to-end check without network access:  python tests/smoke_test.py

1. the request text of the paper's worked example is rebuilt byte for byte;
2. the strict validator and the lenient parser behave as specified;
3. the scorer reproduces the numbers the paper prints for the example, and an absurd model
   answer is removed before scoring;
4. repeating the last value scores S = 0, and every baseline returns finite forecasts;
5. aggregation over cells and the cluster bootstrap behave as defined;
6. a test-set folder is read and scored through the command line;
7. a question is built from a logged curve with the benchmark's prefix statistic;
8. `tea run` and `tea read` talk to an OpenAI-compatible server; raw answers are kept, a
   restarted run sends nothing twice, and the answers are scored;
9. a matching pair is scored over its two presentation orders.
"""
import contextlib
import hashlib
import io
import json
import math
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import tea_leaves as tea  # noqa: E402
from tea_leaves import baselines, cli, question, request, score, validate  # noqa: E402

EX = REPO / 'examples/paper-example'
REQUEST_SHA256 = '1f5c187232f35379bb4508cf79f8395d54efd41bc8167bf3eb969538a0c82eaf'


def quiet(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(argv)
    return buf.getvalue()


def main():
    q = json.loads((EX / 'input.json').read_text(encoding='utf-8'))
    answer = json.loads((EX / 'answer.json').read_text())
    label = json.loads((EX / 'labels.json').read_text())

    # 1. request assembly
    text = request.build_request(q)
    assert text == (EX / 'request.txt').read_text(encoding='utf-8')
    assert hashlib.sha256(text.encode()).hexdigest() == REQUEST_SHA256
    assert request.split_request(text)[1] == q
    print(f'1. request rebuilt: {len(text.encode())} bytes, same SHA-256 as the request sent for the paper')

    # 2. validity
    full = dict(predictions=answer['predictions'],
                rationale=dict(evidence=['prefix falls'], assumptions=['no divergence'], uncertainty='moderate'))
    validate.strict_validate(json.dumps(full), 'prediction', q['query_x'])
    validate.strict_validate('```json\n' + json.dumps(full) + '\n```', 'prediction', q['query_x'])
    payload, flags = validate.lenient_parse('Here is my forecast: ' + json.dumps(full), 'prediction', q['query_x'])
    assert flags == ['json_extracted'] and len(payload['predictions']) == 8
    try:
        validate.lenient_parse(json.dumps({'predictions': answer['predictions'][:-1]}), 'prediction', q['query_x'])
        raise SystemExit('a truncated answer must be rejected')
    except ValueError:
        pass
    print('2. strict validator and lenient parser: ok')

    # 3. the paper's numbers for this example
    summary = tea.evaluate([answer], name='example', questions=[q], labels=[label])
    v, res = summary['views'][0], summary['grids']['10']
    assert res['scored'] == 1 and res['unparsed'] == 0
    assert abs(v['mae'] - 0.045) < 0.002 and abs(v['mae_last_value'] - 1.24) < 0.005
    assert abs(v['ratio'] - 0.036) < 0.002 and v['coverage80'] == 1.0
    assert abs(res['S'] - 0.96) < 0.005 and res['S_ci95'] is None
    print(f"3. example: MAE {v['mae']:.4f}, last-value MAE {v['mae_last_value']:.4f}, "
          f"R {res['R']:.4f}, S {res['S']:.4f}  (the paper prints 0.045, 1.24, 0.036, 0.96; "
          "the example's labels are the paper's two-decimal values)")

    # an absurd model answer (a median far outside the prefix range) is removed, not scored
    wild = {'predictions': [dict(p, p10=p['p50'] * 90, p50=p['p50'] * 100, p90=p['p50'] * 110)
                            for p in answer['predictions']]}
    as_text = tea.evaluate([{'case': q['case'], 'text': json.dumps(wild)}], name='x', questions=[q], labels=[label])
    as_numbers = tea.evaluate([dict(wild, case=q['case'])], name='x', questions=[q], labels=[label])
    assert as_text['grids']['10']['absurd'] == 1 and as_text['grids']['10']['scored'] == 0
    assert as_numbers['grids']['10']['absurd'] == 0 and as_numbers['grids']['10']['scored'] == 1

    # 4. baselines
    line = []
    for method in baselines.METHODS:
        rows = baselines.run(method, [q])
        assert all(math.isfinite(p['p50']) for p in rows[0]['predictions'])
        s = tea.evaluate(rows, name=method, questions=[q], labels=[label])['grids']['10']['S']
        assert method != 'last_value' or abs(s) < 1e-12
        line.append(f'{method} {s:.3f}')
    print('4. baselines on the example, S: ' + ', '.join(line))

    # 5. aggregation: geometric mean within a cell, then over cells; the bootstrap needs clusters
    def view(i, cell, ratio, cluster):
        return {'case': f'c{i}', 'status': 'valid', 'grid': 10, 'cell': cell, 'cluster': cluster, 'ratio': ratio}
    vs = [view(0, 'A', 0.5, 'k0'), view(1, 'A', 2.0, 'k1'), view(2, 'A', None, 'k1'),
          view(3, 'B', 0.25, 'k2'), view(4, 'B', 0.25, 'k3'), view(5, 'B', 0.25, 'k3')]
    out = score.aggregate(vs, 'toy', 10)
    assert abs(out['cells']['A']['S'] - 0.0) < 1e-12 and abs(out['cells']['B']['S'] - 0.75) < 1e-12
    assert abs(out['R'] - 0.5) < 1e-12 and abs(out['S'] - 0.5) < 1e-12 and out['with_ratio'] == 5
    lo, hi = out['S_ci95']
    assert lo <= out['S'] <= hi
    assert score.aggregate([dict(x, cluster=None) for x in vs], 'toy', 10)['S_ci95'] is None
    print(f'5. aggregation: R = {out["R"]:.3f} = sqrt(1.0 * 0.25), 95% interval on S [{lo:.3f}, {hi:.3f}]')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 6. a test-set folder (the example, laid out like the released data), scored through the CLI
        folder = tmp / 'data/toy'
        (folder / 'inputs').mkdir(parents=True)
        (folder / 'labels').mkdir()
        curve = {k: q[k] for k in ('case', 'grid', 'prefix', 'query_x')}
        (folder / 'inputs/prediction.json').write_text(json.dumps(
            [{'case': q['case'], 'conditions': {'curve_only': curve, 'enough_info_available': q}}]))
        (folder / 'labels/prediction-truth.json').write_text(json.dumps(
            [{'case': q['case'], 'run_key': 'run-0', 'targets': label['targets']}]))
        (folder / 'labels/allocation.json').write_text(json.dumps(
            {'runs': [{'run_key': 'run-0', 'stage': 'pretraining', 'source': 'marin-community',
                       'component_id': 'comp-0'}]}))
        raw = tmp / 'forecasts.jsonl'
        raw.write_text(json.dumps({'case': q['case'], 'text': json.dumps(full)}) + '\n')
        quiet(['score', str(raw), '--data', str(folder), '--json', str(tmp / 'summary.json')])
        got = json.loads((tmp / 'summary.json').read_text())['grids']['10']
        assert list(got['cells']) == ['PT/marin'] and abs(got['R'] - res['R']) < 1e-12
        test = tea.TestSet(folder)
        assert test.requests('full')[0]['text'] == text and len(test.questions('curve')) == 1
        print('6. test-set folder: read, requests rebuilt, scored through `tea score`')

        # 7. a question from a logged curve
        steps, values, name = question.read_curve(REPO / 'examples/my-run/curve.csv')
        mine, labels = tea.make_question(steps, values, budget=20000, metric=name, config={'optimizer': 'adamw'})
        assert [p['x'] for p in mine['prefix']] == question.PREFIX_X and len(labels['targets']) == 8
        first = [v for s, v in zip(steps, values) if 0 < s <= 400]
        assert abs(mine['prefix'][0]['value'] - sum(first) / len(first)) < 1e-12
        assert mine['context']['reported_configuration']['planned_step_budget'] == 20000
        try:
            tea.make_question(steps[::40], values[::40], budget=20000)
            raise SystemExit('a curve with empty prefix windows must be rejected')
        except ValueError:
            pass
        assert 'recorded' in quiet(['read', '--csv', str(REPO / 'examples/my-run/curve.csv'), '--budget', '20000'])
        print('7. own-run question: prefix windows, labels, and `tea read`')

        # 8. `tea run` against a local OpenAI-compatible server
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(body)
                assert self.path == '/v1/chat/completions' and len(body['messages']) == 1
                payload = {'choices': [{'message': {'content': 'Forecast: ' + json.dumps(full)},
                                        'finish_reason': 'stop'}], 'usage': {'total_tokens': 1}}
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        argv = ['run', '--model', 'mock-model', '--provider', 'openai',
                '--base-url', f'http://127.0.0.1:{server.server_port}/v1', '--data', str(folder),
                '--out', str(tmp / 'answers.jsonl'), '--json', str(tmp / 'run.json')]
        quiet(argv)
        quiet(argv)  # the second start finds the answer and sends nothing
        shown = quiet(['read', '--csv', str(REPO / 'examples/my-run/curve.csv'), '--budget', '20000',
                       '--model', 'mock-model', '--provider', 'openai',
                       '--base-url', f'http://127.0.0.1:{server.server_port}/v1'])
        server.shutdown()
        row = json.loads((tmp / 'answers.jsonl').read_text().splitlines()[0])
        assert len(calls) == 2 and calls[0]['messages'][0]['content'] == text
        assert row['model'] == 'mock-model' and row['text'].startswith('Forecast: ')
        assert 'mock-model' in shown and 'p90' in shown and 'skill of mock-model on this run' in shown
        assert abs(json.loads((tmp / 'run.json').read_text())['grids']['10']['S'] - res['S']) < 1e-12
        print('8. `tea run`: one request sent, raw answer kept, resumed without a second call, scored; '
              '`tea read` shows the model forecast')
    # 9. matching: one pair in two orders; the pair is right when the mean probability of the truth exceeds 0.5
    from tea_leaves import matching
    minputs = [{'case': 'm1', 'conditions': {}}, {'case': 'm2', 'conditions': {}}]
    mlabels = [{'case': 'm1', 'pair_id': 'p', 'stage': 'pretraining', 'grid': 10, 'A_is_curve1': True},
               {'case': 'm2', 'pair_id': 'p', 'stage': 'pretraining', 'grid': 10, 'A_is_curve1': False}]
    cond = matching.CONDITIONS[0]
    good = matching.score_matching(minputs, mlabels, [
        {'case': 'm1', 'condition': cond, 'text': '{"probability_A_curve1": 0.9}'},
        {'case': 'm2', 'condition': cond, 'probability_A_curve1': 0.3}])
    assert good['stages']['pretraining']['accuracy'] == 1.0 and good['stages']['pretraining']['both_orders'] == 1.0
    bad = matching.score_matching(minputs, mlabels, [
        {'case': 'm1', 'condition': cond, 'probability_A_curve1': 0.6},
        {'case': 'm2', 'condition': cond, 'probability_A_curve1': 0.9}])
    assert bad['stages']['pretraining']['accuracy'] == 0.0
    print('9. matching: a pair is scored over its two orders')
    print('smoke test passed')


if __name__ == '__main__':
    main()
