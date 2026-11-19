"""Merge HLE runs over the same items into one result (several endpoints working one item set, or two subsets).

    python3 -m aa_evals.hle.merge --out qwen-diamond-medium-final --sources qwen-diamond-medium qwen-diamond-medium-any

Per (item, repeat), the first source in --sources order whose row has a finished generation wins (status ok or
model_error other than a cutoff). The rule never looks at the score, so nothing is cherry-picked: sources are in a
fixed priority, and a later source only fills items an earlier one did not finish. An item no source finished is a
cutoff (0) if any source cut it, else an infra error. Sources must share the generation config (model, sampling,
prompt, dataset) and the judge. The winning transcripts are hard-linked into the output's transcripts/, so the
merged run can be re-judged with run.py --generations-from.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

from . import scoring
from .run import GENERATION_KEYS, atomic_json, fingerprint, run_path
from aa_evals.common import jsonl


def finished(row):
    return row['status'] in ('ok', 'model_error') and row.get('finish_reason') != 'cutoff'


def src_dir(s):
    return Path(s) if (Path(s) / 'config.json').exists() else run_path(s)


def merge(sources, out_dir):
    cfgs = [json.loads((src_dir(s) / 'config.json').read_text()) for s in sources]
    for s, c in zip(sources[1:], cfgs[1:]):
        diff = [k for k in GENERATION_KEYS + ('judge_spec', 'judge_settings', 'n_repeats')
                if fingerprint(c)[k] != fingerprint(cfgs[0])[k]]
        if diff:
            raise SystemExit(f'{s} differs from {sources[0]} in {diff}')
        if c['dataset'].get('task_set', 'all') != cfgs[0]['dataset'].get('task_set', 'all'):
            raise SystemExit(f'{s} and {sources[0]} were drawn from different --task-set')
    rows_by_src = []
    for s in sources:
        path = src_dir(s) / 'items.jsonl'
        rows_by_src.append({(r['item_id'], r['repeat']): r
                            for r in jsonl.read(path, required=True)})
    keys = sorted(set().union(*rows_by_src))
    out_dir = Path(out_dir)
    (out_dir / 'transcripts').mkdir(parents=True, exist_ok=True)
    rows, origin = [], {}
    for key in keys:
        cands = [(s, rs[key]) for s, rs in zip(sources, rows_by_src) if key in rs]
        win = next(((s, r) for s, r in cands if finished(r)), None)
        if win is None:
            cut = next(((s, r) for s, r in cands if r.get('finish_reason') == 'cutoff'), None)
            win = cut or cands[0]
        s, r = win
        origin[s] = origin.get(s, 0) + 1
        r = {**r, 'source_run': s}
        src_t = Path(r['transcript'])
        if finished(r) and src_t.exists():
            dst_t = out_dir / 'transcripts' / src_t.name
            if dst_t.exists():
                dst_t.unlink()
            try:
                os.link(src_t, dst_t)
            except OSError:
                shutil.copy2(src_t, dst_t)
            r['transcript'] = str(dst_t)
        rows.append(r)
    base = cfgs[0]
    n_items = len({k[0] for k in keys})
    cfg = {**base, 'run_id': out_dir.name, 'merged_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
           'merged_from': [{'run_id': s, 'git': c.get('git'), 'model': c['model'].get('spec'),
                            'cutoff_min': c.get('cutoff_min'), 'note': c.get('note'), 'rows_used': origin.get(s, 0)}
                           for s, c in zip(sources, cfgs)],
           'selection': f'union of {", ".join(sources)} ({n_items} items)'}
    with open(out_dir / '.items.jsonl.tmp', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    (out_dir / '.items.jsonl.tmp').replace(out_dir / 'items.jsonl')
    atomic_json(out_dir / 'config.json', cfg)
    summary = scoring.summarize(
        rows, n_items=n_items, n_repeats=base['n_repeats'], dataset=base['dataset']['id'],
        aa_slug=base.get('aa_slug'),
        extra={'run_id': out_dir.name, 'model': base['model']['served_name'], 'judge': base['judge']['served_name'],
               'merged_from': sources, 'rows_from': origin, 'task_set': base['dataset'].get('task_set', 'all'),
               'n_cutoff': sum(r.get('finish_reason') == 'cutoff' for r in rows),
               'n_reasoning_unclosed': sum(bool(r.get('reasoning_unclosed')) for r in rows)})
    atomic_json(out_dir / 'summary.json', summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='run id of the merged result')
    ap.add_argument('--sources', nargs='+', required=True, help='run ids, highest priority first')
    a = ap.parse_args(argv)
    from .run import RUNS_ROOT
    s = merge(a.sources, RUNS_ROOT / a.out)
    print(json.dumps({k: s[k] for k in ('score', 'ci95', 'n_items', 'n_infra_errors', 'n_model_errors', 'n_cutoff',
                                        'valid', 'aa_reference', 'rows_from')}))


if __name__ == '__main__':
    main()
