"""Command line: `tea <command>` (or `python -m tea_leaves <command>`).

    tea baseline power_law                  score a curve-fitting baseline on the test set
    tea run --model MODEL                   run an LLM on the test set and score it
    tea score outputs/answers.jsonl         score a file of forecasts
    tea read --csv run.csv --budget 20000   forecast your own run from its first 20%
    tea match --model MODEL                 the matching task: which setting drew which curve?
    tea request                             print the text a model receives for one question
    tea download                            fetch the data (done automatically on first use)
"""
import argparse
import json
import sys
from pathlib import Path

from . import backends, baselines, data, matching, question, request, run as runner, score


def _add_data(p):
    p.add_argument('--set', default=data.DEFAULT_SET, choices=['pt-ft', 'rl', 'community'],
                   help='test set: pt-ft (pretraining and fine-tuning, default), rl, or community')
    p.add_argument('--data', help='folder of a test set, instead of --set (default: data/<set>, downloaded on first use)')
    p.add_argument('--subset', help='score only a named subset of the set, e.g. core-211 on the community set')


def _add_model(p):
    p.add_argument('--model', help='model name, e.g. claude-opus-5 or the name served by your endpoint')
    p.add_argument('--provider', choices=['anthropic', 'openai'], default=None,
                   help="'anthropic' (official SDK) or 'openai' (any OpenAI-compatible endpoint, e.g. vLLM); "
                        "default: anthropic for claude-* models, openai otherwise")
    p.add_argument('--base-url', help='openai provider: endpoint, e.g. http://localhost:8000/v1')
    p.add_argument('--api-key-env', default='OPENAI_API_KEY', help='openai provider: variable holding the key')
    p.add_argument('--max-tokens', type=int, help='output token limit (default: 16000 for anthropic, unset for openai)')
    p.add_argument('--effort', help='anthropic provider: low, medium, high, xhigh or max')
    p.add_argument('--extra', type=json.loads, help='openai provider: JSON merged into the request body')


def _backend(args):
    provider = args.provider or ('anthropic' if args.model.startswith('claude') else 'openai')
    options = {k: v for k, v in {'base_url': args.base_url, 'api_key_env': args.api_key_env,
                                 'max_tokens': args.max_tokens, 'effort': args.effort,
                                 'extra': args.extra}.items() if v is not None}
    return backends.get(provider), options


def _load(args):
    folder = Path(args.data) if args.data else Path(data.DEFAULT_ROOT) / args.set
    test = data.load(folder.name, folder.parent)
    test.labels  # reads the allocation: how the set is scored, and which metrics are maximized
    return test


def _subset(args, test):
    if not getattr(args, 'subset', None):
        return None
    return json.loads((test.folder / 'labels' / f'{args.subset}.json').read_text())['cases']


def _grid(args, test):
    """The RL set is scored on both query grids together; the other sets on one grid."""
    return None if test.pool_grids else args.grid


def _finish(summary, json_out):
    print(score.report(summary))
    if json_out:
        Path(json_out).write_text(json.dumps(summary, indent=1) + '\n')
        print(f'full summary written to {json_out}')


def cmd_download(args):
    print(f'data in {data.download(args.root)}/')


def cmd_baseline(args):
    test = _load(args)
    grid = _grid(args, test)
    rows = baselines.run(args.method, test.questions('curve', grid), test.maximize)
    out = Path(args.out or f'outputs/{test.name}-{args.method}-grid{grid or "all"}.jsonl')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(f'{len(rows)} forecasts written to {out}')
    _finish(score.evaluate(rows, test, name=args.method, subset=_subset(args, test)), args.json)


def cmd_run(args):
    test = _load(args)
    grid = _grid(args, test)
    requests = test.requests(args.condition, grid)
    if args.limit:
        requests = requests[:args.limit]
    complete, options = _backend(args)
    out = Path(args.out) if args.out else runner.default_out(args.model, f'{test.name}-{args.condition}', grid or 'all')
    print(f'{args.model}: {len(requests)} requests ({test.name}, {args.condition} condition, '
          + ('both grids' if grid is None else f'Grid{grid}') + f') -> {out}')
    rows = runner.run(requests, complete, out, args.model, workers=args.workers, **options)
    _finish(score.evaluate(rows, test, name=args.name or args.model, subset=_subset(args, test)), args.json)


def cmd_score(args):
    if args.questions:
        summary = score.evaluate(args.forecasts, name=args.name or Path(args.forecasts).stem,
                                 questions=args.questions, labels=args.labels)
    else:
        test = _load(args)
        summary = score.evaluate(args.forecasts, test, name=args.name or Path(args.forecasts).stem,
                                 subset=_subset(args, test))
    _finish(summary, args.json)


def cmd_match(args):
    inputs, labels = matching.load(args.root)
    name = args.name or args.model or Path(args.answers).stem
    if args.answers:
        answers = score.read_rows(args.answers)
    else:
        conds = {'both': matching.CONDITIONS, 'minimal': matching.CONDITIONS[:1], 'full': matching.CONDITIONS[1:]}
        reqs = matching.requests(inputs, labels, conds[args.condition])
        if args.limit:
            reqs = reqs[:args.limit]
        complete, options = _backend(args)
        out = Path(args.out) if args.out else runner.default_out(args.model, 'matching-' + args.condition, 10)
        print(f'{args.model}: {len(reqs)} matching requests -> {out}')
        answers = runner.run(reqs, complete, out, args.model, workers=args.workers, **options)
    summary = matching.score_matching(inputs, labels, answers)
    print(matching.report(summary, name))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + '\n')


def cmd_request(args):
    if args.input:
        obj = json.loads(Path(args.input).read_text(encoding='utf-8'))
    else:
        test = _load(args)
        questions = test.questions(args.condition, _grid(args, test))
        obj = next((q for q in questions if q['case'] == args.case), None) if args.case else questions[0]
        if obj is None:
            sys.exit(f'no question with case {args.case}')
    print(request.build_request(obj))


def cmd_read(args):
    extra = {'metric': args.metric} if args.metric else {}
    q, labels = question.from_files(args.csv, args.budget, args.config, description=args.description,
                                    grid=args.grid, **extra)
    if args.save:
        Path(args.save).write_text(json.dumps(q, indent=1) + '\n')
        print(f'question written to {args.save}')
    print(f"{q['case']}: {q['metric']}, prefix from x = 0.02 to 0.20 of a budget of {args.budget:g}")
    print('  prefix   ' + '  '.join(f"{p['value']:.4g}" for p in q['prefix']))
    forecasts = {name: [p['p50'] for p in baselines.run(name, [q])[0]['predictions']]
                 for name in ('last_value', 'power_law')}
    quantiles = None
    if args.model:
        complete, options = _backend(args)
        answer = complete(request.build_request(q), model=args.model, **options)['text']
        try:
            from .reference import lenient_parse
            payload, _ = lenient_parse(answer, 'prediction', q['query_x'])
            quantiles = payload['predictions']
            forecasts[args.model] = [p['p50'] for p in quantiles]
        except ValueError as exc:
            print(f'  the answer of {args.model} could not be parsed ({exc}):\n{answer[:500]}')
    width = max(len(n) for n in forecasts) + 2
    print('  ' + 'x'.ljust(width) + ''.join(f'{x:>9.1f}' for x in q['query_x']))
    for name, values in forecasts.items():
        print('  ' + name.ljust(width) + ''.join(f'{v:>9.4g}' for v in values))
    if quantiles and all('p10' in p and 'p90' in p for p in quantiles):
        print('  ' + 'p10'.rjust(width - 2).ljust(width) + ''.join(f"{p['p10']:>9.4g}" for p in quantiles))
        print('  ' + 'p90'.rjust(width - 2).ljust(width) + ''.join(f"{p['p90']:>9.4g}" for p in quantiles))
    if labels:
        print('  ' + 'recorded'.ljust(width) + ''.join(f'{v:>9.4g}' for v in labels['targets']))
        for name, values in forecasts.items():
            if name == 'last_value':
                continue
            rows = [{'case': q['case'], 'predictions': [{'x': x, 'p50': v} for x, v in zip(q['query_x'], values)]}]
            s = score.evaluate(rows, name=name, questions=[q], labels=[labels])['grids'][str(args.grid)]['S']
            print(f'  skill of {name} on this run: ' + ('n/a' if s is None else f'{s:.3f}'))
    else:
        print('  (the run is not logged at every query point yet, so there is nothing to score)')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='tea', description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('download', help='fetch the data from the Hugging Face Hub')
    p.add_argument('--root', default=data.DEFAULT_ROOT)
    p.set_defaults(func=cmd_download)

    p = sub.add_parser('baseline', help='score a curve-fitting baseline')
    p.add_argument('method', choices=baselines.METHODS)
    p.add_argument('--grid', type=int, default=10, choices=[10, 20])
    p.add_argument('--out', help='forecast file (default: outputs/<set>-<method>-grid<grid>.jsonl)')
    p.add_argument('--json', help='also write the full summary here')
    _add_data(p)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser('run', help='run an LLM on the test set and score it')
    _add_model(p)
    p.add_argument('--condition', default='full', choices=sorted(data.CONDITIONS),
                   help='what the model sees: curve, config (curve and configuration), or full (default)')
    p.add_argument('--grid', type=int, default=10, choices=[10, 20])
    p.add_argument('--workers', type=int, default=8, help='parallel requests')
    p.add_argument('--limit', type=int, help='only the first N questions (for a trial run)')
    p.add_argument('--out', help='answer file (default: outputs/<model>-<set>-<condition>-grid<grid>.jsonl)')
    p.add_argument('--name', help='name shown in the report (default: the model)')
    p.add_argument('--json', help='also write the full summary here')
    _add_data(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('score', help='score a file of forecasts')
    p.add_argument('forecasts', help='JSON or JSON-lines file: {case, text} or {case, predictions}')
    p.add_argument('--name')
    p.add_argument('--questions', help='score against your own questions instead of the test set')
    p.add_argument('--labels', help='labels for --questions: rows {case, targets}')
    p.add_argument('--json', help='also write the full summary here')
    _add_data(p)
    p.set_defaults(func=cmd_score)

    p = sub.add_parser('match', help='the matching task: which setting drew which curve?')
    _add_model(p)
    p.add_argument('--answers', help='score this file instead of running a model: rows {case, condition, text}')
    p.add_argument('--condition', default='both', choices=['both', 'minimal', 'full'],
                   help='minimal: curves and settings; full: adds description and code; both (default)')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, help='only the first N requests (for a trial run)')
    p.add_argument('--out', help='answer file')
    p.add_argument('--name')
    p.add_argument('--json', help='also write the full summary here')
    p.add_argument('--root', default=data.DEFAULT_ROOT, help='data folder (default: data)')
    p.set_defaults(func=cmd_match)

    p = sub.add_parser('request', help='print the text a model receives for one question')
    p.add_argument('--input', help='a JSON file with one input object (default: a question of the test set)')
    p.add_argument('--case', help='case id of a test-set question (default: the first one)')
    p.add_argument('--condition', default='full', choices=sorted(data.CONDITIONS))
    p.add_argument('--grid', type=int, default=10, choices=[10, 20])
    _add_data(p)
    p.set_defaults(func=cmd_request)

    p = sub.add_parser('read', help='forecast your own run from its first 20%%')
    p.add_argument('--csv', required=True, help='logged curve: a step column and a metric column, with a header')
    p.add_argument('--budget', type=float, required=True, help='planned length of the run, in steps')
    p.add_argument('--config', help='JSON file of settings (makes the question a Full one)')
    p.add_argument('--description', help='one or two sentences about the run')
    p.add_argument('--metric', help='name of the metric (default: the CSV column name)')
    p.add_argument('--grid', type=int, default=10, choices=[10, 20])
    p.add_argument('--save', help='write the question here (JSON)')
    _add_model(p)
    p.set_defaults(func=cmd_read)

    args = ap.parse_args(argv)
    if args.command == 'run' and not args.model:
        ap.error('tea run needs --model')
    if args.command == 'match' and not (args.model or args.answers):
        ap.error('tea match needs --model or --answers')
    args.func(args)


if __name__ == '__main__':
    main()
