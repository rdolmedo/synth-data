"""Re-score stored HLE runs on a question set, from their items.jsonl (no model or judge calls; #348).

    python3 -m aa_evals.hle.rescore --task-set text-only-v1 --out <dir> <run_id or run dir> ...

Each run's stored rows are restricted to the set's questions and summarised with scoring.summarize, exactly as a
run on that set would be. The run directory is never written: <out>/<run_id>.summary.<set>.json holds the new
summary (with the run's own score on its full selection beside it), and the command prints one line per run.
A run drawn from a subset (--limit / --ids) is re-scored on the intersection of its selection and the set.
"""
import argparse
import json
from pathlib import Path

from . import data, scoring
from .merge import src_dir
from .run import atomic_json
from aa_evals.common import jsonl


def rescore(run, task_set, items=None):
    """(summary on the set, the run's stored summary) for one run id or run directory."""
    d = src_dir(run)
    cfg = json.loads((d / 'config.json').read_text())
    stored = json.loads((d / 'summary.json').read_text())
    run_set = cfg['dataset'].get('task_set', 'all')
    if run_set not in ('all', task_set):
        raise SystemExit(f'{run}: drawn from --task-set {run_set}, which does not contain {task_set}')
    dataset = cfg['dataset']['id']
    items = items if items is not None else data.load(dataset, cfg['dataset']['revision'])
    keep = {it['id'] for it in data.task_set(items, task_set)}
    rows = jsonl.read(d / 'items.jsonl', required=True)
    for r in rows:   # the row's modality was recorded from the same dataset revision: they must agree
        if task_set == 'text-only-v1' and r.get('has_image') is not None and r['has_image'] == (r['item_id'] in keep):
            raise SystemExit(f'{run}: row {r["item_id"]} has_image={r["has_image"]} disagrees with the dataset')
    selected = cfg.get('item_ids') or sorted({r['item_id'] for r in rows})
    chosen = [i for i in selected if i in keep]
    rows = [r for r in rows if r['item_id'] in keep]
    out = scoring.summarize(
        rows, n_items=len(chosen), n_repeats=cfg['n_repeats'], dataset=dataset, aa_slug=cfg.get('aa_slug'),
        extra={'run_id': cfg['run_id'], 'model': cfg['model']['served_name'], 'judge': cfg['judge']['served_name'],
               'judge_arm': cfg['judge'].get('arm'), 'task_set': task_set, 'rescored_from': str(d),
               'source_task_set': run_set, 'source_n_items': len(selected),
               'source_score': stored.get('score'), 'source_ci95': stored.get('ci95'),
               'n_cutoff': sum(r.get('finish_reason') == 'cutoff' for r in rows),
               'n_reasoning_unclosed': sum(bool(r.get('reasoning_unclosed')) for r in rows)})
    out['delta_vs_source'] = (None if out['score'] is None or stored.get('score') is None
                              else round(out['score'] - stored['score'], 2))
    return out, stored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='run ids (looked up in every runs root) or run directories')
    ap.add_argument('--task-set', choices=data.TASK_SETS, default=data.DEFAULT_TASK_SET)
    ap.add_argument('--out', help='directory for <run_id>.summary.<set>.json (default: print only)')
    a = ap.parse_args(argv)
    cache = {}
    for run in a.runs:
        cfg = json.loads((src_dir(run) / 'config.json').read_text())
        key = (cfg['dataset']['id'], cfg['dataset']['revision'])
        if key not in cache:
            cache[key] = data.load(*key)
        s, stored = rescore(run, a.task_set, cache[key])
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            atomic_json(Path(a.out) / f'{s["run_id"]}.summary.{a.task_set}.json', s)
        print(json.dumps({'run_id': s['run_id'], 'task_set': a.task_set, 'n_items': s['n_items'],
                          'score': s['score'], 'ci95': s['ci95'], 'valid': s['valid'],
                          'source_n_items': s['source_n_items'], 'source_score': s['source_score'],
                          'delta': s['delta_vs_source']}), flush=True)


if __name__ == '__main__':
    main()
