"""Run a model over a test set: one request per question, raw answers kept in a JSON-lines file.

The run can be interrupted and restarted: questions that already have an answer in the output
file are skipped, and failed calls are retried on the next start.
"""
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path



def default_out(model, condition, grid):
    slug = re.sub(r'[^A-Za-z0-9._-]+', '-', model).strip('-')
    return Path('outputs') / f'{slug}-{condition}-grid{grid}.jsonl'


def run(requests, complete, out, model, workers=8, **options):
    """Send each request ({case, grid, condition, text}) and append one row per answer to `out`.

    Returns every answered row of `out`, including those of earlier runs."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for line in out.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if 'text' in row:
                done[row['case'], row.get('condition')] = row
    todo = [r for r in requests if (r['case'], r['condition']) not in done]
    if done:
        print(f'{len(done)} answers already in {out}; {len(todo)} to go')
    lock = threading.Lock()
    failed = 0

    def one(req):
        row = {'case': req['case'], 'grid': req['grid'], 'condition': req['condition'], 'model': model}
        try:
            row.update(complete(req['text'], model=model, **options))
        except Exception as exc:  # recorded and retried on the next start; one bad call does not stop the run
            row['error'] = f'{type(exc).__name__}: {exc}'[:500]
        return row

    with out.open('a', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, r) for r in todo]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')
                fh.flush()
            if 'text' in row:
                done[row['case'], row.get('condition')] = row
            else:
                failed += 1
                if failed <= 3:
                    print(f"  call failed for {row['case']}: {row['error']}")
            if i % 25 == 0 or i == len(todo):
                print(f'  {i}/{len(todo)} requests done' + (f', {failed} failed' if failed else ''))
    if failed:
        print(f'{failed} calls failed; run the same command again to retry them')
    return list(done.values())
