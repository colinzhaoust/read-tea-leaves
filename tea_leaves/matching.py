"""The matching task: which setting drew which curve?

A pair is two runs of one sweep whose settings differ. The model sees both curves and both
settings (anonymized as A and B) and returns `probability_A_curve1`. Each pair is shown in two
orders (the curves swapped; on RL also at two curve grids) and in two conditions:
`curves_plus_minimal_settings` (curves and settings) and `enough_info_available` (adds the
description, source facts and code).

Scoring, as in the paper's matching table. The unit is a pair in one condition: its valid answers
are averaged on the scale of the probability given to the true assignment, and the pair is correct
when the mean exceeds 0.5 (half credit at exactly 0.5). Answers that do not parse are dropped.
The accuracy of a stage pools both conditions; `avg` is the unweighted mean of the three stages;
chance is 0.5. `both_orders` is the share of pairs answered correctly in both orders (chance 0.25).
"""
import json
from pathlib import Path

import collections
import math

CONDITIONS = ('curves_plus_minimal_settings', 'enough_info_available')
STAGES = ('pretraining', 'fine-tuning', 'RL')


def _parser():
    from .reference import lenient_parse
    return lenient_parse


def probability(row, parse=None):
    """probability_A_curve1 of one answer row, or None when invalid."""
    if 'text' in row and row.get('text') is not None:
        parse = parse or _parser()
        try:
            payload, _ = parse(row['text'], 'matching', None)
        except ValueError:
            return None
        return float(payload['probability_A_curve1'])
    p = row.get('probability_A_curve1')
    if type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1:
        return float(p)
    return None


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def score_matching(inputs, labels, answers):
    cases = {x['case'] for x in inputs}
    lab = {l['case']: l for l in labels if l['case'] in cases}
    parse = None if not any('text' in a for a in answers) else _parser()

    views = collections.defaultdict(dict)        # (pair_id, condition) -> {case: p_true}
    counts = collections.Counter()
    seen = set()
    for a in answers:
        key = (a['case'], a['condition'])
        if a['case'] not in lab or a['condition'] not in CONDITIONS:
            counts['unknown_case_or_condition'] += 1
            continue
        if key in seen:
            counts['duplicate_ignored'] += 1
            continue
        seen.add(key)
        counts['answers'] += 1
        p = probability(a, parse)
        if p is None:
            counts['invalid'] += 1
            continue
        counts['valid'] += 1
        l = lab[a['case']]
        views[l['pair_id'], a['condition']][a['case']] = p if l['A_is_curve1'] else 1.0 - p

    pairs = {}
    for l in lab.values():
        pairs[l['pair_id']] = l['stage']
    rows = []
    missing = collections.Counter()
    for pid, stage in sorted(pairs.items()):
        for cond in CONDITIONS:
            v = views.get((pid, cond))
            if not v:
                missing[stage, cond] += 1
                continue
            m = _mean(list(v.values()))
            acc = 1.0 if m > 0.5 else (0.5 if m == 0.5 else 0.0)
            both = []
            for grid in sorted({lab[c]['grid'] for c in v}):
                g = [p for c, p in v.items() if lab[c]['grid'] == grid]
                if len(g) == 2:
                    both.append(1.0 if min(g) > 0.5 else 0.0)
            rows.append({'pair_id': pid, 'stage': stage, 'condition': cond, 'views': len(v),
                         'mean_p_true': m, 'accuracy': acc, 'both_orders': both})

    def summary(rs):
        b = [x for r in rs for x in r['both_orders']]
        return {'n': len(rs), 'accuracy': _mean([r['accuracy'] for r in rs]),
                'both_orders': _mean(b), 'both_orders_n': len(b),
                'one_view_rows': sum(r['views'] == 1 for r in rs)}

    out = {'counts': dict(counts), 'stages': {}}
    for stage in STAGES:
        rs = [r for r in rows if r['stage'] == stage]
        if not rs and not any(s == stage for s in pairs.values()):
            continue
        e = summary(rs)
        e['pairs'] = sum(1 for s in pairs.values() if s == stage)
        e['missing_rows'] = sum(n for (s, _), n in missing.items() if s == stage)
        e['by_condition'] = {c: summary([r for r in rs if r['condition'] == c]) for c in CONDITIONS}
        out['stages'][stage] = e
    accs = [out['stages'][s]['accuracy'] for s in STAGES if s in out['stages']]
    out['avg'] = _mean(accs) if len(accs) == len(STAGES) and None not in accs else None
    out['rows'] = rows
    return out


def load(root='data'):
    """(inputs, labels) of the matching set, downloading the data on first use."""
    from . import data
    folder = Path(root) / 'matching'
    if not (folder / 'inputs/matching.json').exists():
        print(f'Downloading {data.HF_DATASET} into {root}/ ...')
        data.download(root)
    return (json.loads((folder / 'inputs/matching.json').read_text(encoding='utf-8')),
            json.loads((folder / 'labels/matching-answers.json').read_text(encoding='utf-8')))


def requests(inputs, labels, conditions=CONDITIONS):
    """The exact text to send: rows {case, condition, grid, text}.

    The paper's pretraining and fine-tuning requests serialize the input object with sorted keys;
    its RL requests keep the stored key order. Both are reproduced here byte for byte."""
    from .request import prompt
    stage = {l['case']: l['stage'] for l in labels}
    rows = []
    for item in inputs:
        for cond in conditions:
            obj = item['conditions'][cond]
            payload = json.dumps(obj, sort_keys=stage[item['case']] != 'RL', separators=(',', ':'),
                                 ensure_ascii=False)
            rows.append({'case': item['case'], 'condition': cond, 'grid': obj.get('grid'),
                         'text': prompt('matching') + '\nINPUT:\n' + payload})
    return rows


def report(summary, name='answers'):
    lines = [f'{name} on the matching task (chance 0.5)']
    c = summary['counts']
    lines.append(f"  {c.get('valid', 0)} of {c.get('answers', 0)} answers parsed")
    for stage, e in summary['stages'].items():
        acc = 'n/a' if e['accuracy'] is None else f"{e['accuracy']:.3f}"
        lines.append(f"  {stage:<13}{acc}   ({e['n']} pair-condition rows of {2 * e['pairs']})")
    if summary.get('avg') is not None:
        lines.append(f"  {'average':<13}{summary['avg']:.3f}")
    return '\n'.join(lines)
