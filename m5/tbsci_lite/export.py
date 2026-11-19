"""pi_eval attempts -> SFT export in the synth/ campaign layout that sft_qwen35/prepare.py reads (#353).

prepare.py --source pi_native --synth-roots <EXPORT> reads <EXPORT>/manifest.json (teacher, records with usable and
finalized) and <EXPORT>/attempts/index.json + shards (synth/archive.py's packed form: one JSON line per attempt with
its Pi sessions, first wire request and reward). This writes exactly that, so the M5 traces go through the same prep
as M1's production traces. Every finished (non-infrastructure) attempt is kept, flagged:
  meta.end: submitted (the model ended the episode), context_limit (ended at the 131k window), env_budget (2 h
  environment clock), stopped (tail cut / worker stopped); meta.max_context_tokens, model_requests, wall_sec.
`finalize` sets usable per the export policy (decontamination flags, ends) and writes manifest.json; until then
every record is usable, except one whose Pi finish notices are lost (meta.finish_notice_problem). Runs with the Pi
finish notice ship each trial's notice map as "finish_notices" (aa_evals/common/pi_eval/finish_map.py, #375 sftnotice).

  python -m m5.tbsci_lite.export collect --export DIR --runs RUN...
  python -m m5.tbsci_lite.export finalize --export DIR [--exclude-tasks FILE] [--drop-ends ...]
"""
import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

from .common import durable_write  # noqa: E402

TEACHER = 'nemotron35-super'
BATCH = 50


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def write_json(p, obj):
    durable_write(p, json.dumps(obj), check=json.loads)


def end_of(stats):
    s = stats or {}
    st = s.get('end_state')
    if st in ('finished', None) and s.get('context_limit'):
        return 'context_limit'
    if st == 'finished':
        return 'submitted'
    if st == 'env_budget_spent':
        return 'env_budget'
    return 'stopped' if st else 'unknown'


def attempt_line(adir):
    """One attempt dir (outcome.json + trial.tar.gz) -> archive line, or None (infra, or nothing to export)."""
    out = read_json(adir / 'outcome.json')
    att = read_json(adir / 'attempt.json', {}) or {}
    if not out or out.get('infrastructure_failure'):
        return None
    tar = adir / 'trial.tar.gz'
    if not tar.exists():
        return None
    from aa_evals.common.pi_eval import finish_map
    sessions, first, logs = [], None, {}
    with tarfile.open(tar) as tf:
        for m in tf:
            if not m.isfile():
                continue
            if finish_map.member_kind(m.name) and finish_map.member_kind(m.name) not in logs:   # #375 sftnotice
                logs[finish_map.member_kind(m.name)] = tf.extractfile(m).read().decode('utf-8', 'surrogateescape')
            elif '/agent/pi/sessions/' in m.name and m.name.endswith('.jsonl'):
                sessions.append({'path': m.name.split('/agent/', 1)[1],
                                 'text': tf.extractfile(m).read().decode('utf-8', 'surrogateescape')})
            elif '/pi-cont16k/first-request/' in m.name and m.name.endswith('.json') and first is None:
                try:
                    first = json.loads(tf.extractfile(m).read())
                except ValueError:
                    first = None
    if not sessions or not first:
        return None
    fn, window = finish_map.run_settings(str(adir.parent.parent))
    notices, problem = finish_map.attempt_notices(fn, window, logs.get('notices'),
                                                  (out.get('stats') or {}).get('max_prompt_tokens'))
    key = adir.name
    task = key.split('__a')[0]
    stats = out.get('stats') or {}
    meta = {'end': end_of(stats), 'end_state': stats.get('end_state'), 'context_limit': stats.get('context_limit'),
            'max_context_tokens': stats.get('max_context_tokens'), 'max_prompt_tokens': stats.get('max_prompt_tokens'),
            'model_requests': stats.get('model_requests'), 'wall_sec': stats.get('wall_sec'),
            'model_sec': stats.get('model_sec'), 'env_sec': stats.get('env_sec'),
            'output_cap_hits': stats.get('output_cap_hits'), 'run': adir.parent.parent.name,
            'engine': att.get('endpoint') or att.get('replica'), 'attempt_dir': str(adir)}
    meta['finish_notice_problem'] = problem   # the notices the model saw are lost: not usable for SFT
    rec = {'id': f'{adir.parent.parent.name}/{key}', 'task': task, 'usable': not problem, 'finalized': True,
           'teacher': TEACHER, 'raw': str(adir), 'native_sessions': [s['path'] for s in sessions], 'meta': meta}
    return {'id': rec['id'], 'task': task, 'record': rec, 'sessions': sessions, 'first_request': first,
            'reward': out.get('reward'), 'meta': meta, **({'finish_notices': notices} if notices is not None else {})}


class Exporter:
    """Incremental export: collect(run) picks up attempts not exported yet; lines are packed BATCH per shard."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / 'attempts').mkdir(parents=True, exist_ok=True)
        self.index = read_json(self.root / 'attempts' / 'index.json') or {'shards': {}, 'attempts': {}}
        self.meta = read_json(self.root / 'records.json') or {}     # id -> record (task, meta)
        self.buf = []
        self.skipped = set()   # finished attempts with nothing to export (infra, no session): not re-read
        self.last_flush = time.time()

    def collect(self, run):
        from aa_evals.common.pi_eval import finish_map
        finish_map.run_settings(str(run))   # a v3 notice run is refused before any of its attempts is buffered
        att = Path(run) / 'attempts'
        if not att.is_dir():
            return
        for e in os.scandir(att):
            rid = f'{Path(run).name}/{e.name}'
            if rid in self.meta or rid in self.skipped or rid in {x['id'] for x in self.buf}:
                continue
            if not os.path.exists(os.path.join(e.path, 'outcome.json')):
                continue
            line = attempt_line(Path(e.path))
            if line is None:
                o = read_json(os.path.join(e.path, 'outcome.json'), {}) or {}
                if o.get('infrastructure_failure') or os.path.exists(os.path.join(e.path, 'trial.tar.gz')):
                    self.skipped.add(rid)   # else the trial archive may still be on its way
                continue
            self.buf.append(line)
        if len(self.buf) >= BATCH or (self.buf and time.time() - self.last_flush > 1800):
            self.flush()

    def flush(self):
        if not self.buf:
            return
        n = len(self.index['shards'])
        name = f'shard-{n:05d}.jsonl.gz'
        data = gzip.compress(''.join(json.dumps(x) + '\n' for x in self.buf).encode(), mtime=0)
        want = len(self.buf)

        def check(b):   # the shard decompresses to exactly the buffered lines
            if gzip.decompress(b).count(b'\n') != want:
                raise ValueError('shard line count')
        durable_write(self.root / 'attempts' / name, data, check=check)
        self.index['shards'][str(n)] = {'lines': name, 'n': want, 'sha256': hashlib.sha256(data).hexdigest()}
        for x in self.buf:
            self.index['attempts'][x['id']] = [n]
            self.meta[x['id']] = {k: v for k, v in x['record'].items()}
        write_json(self.root / 'attempts' / 'index.json', self.index)
        write_json(self.root / 'records.json', self.meta)
        self.write_manifest()
        self.buf = []
        self.last_flush = time.time()

    def write_manifest(self, usable=None):
        recs = []
        for rid, r in self.meta.items():
            r = dict(r)
            if usable is not None:
                r['usable'] = rid in usable
            recs.append(r)
        write_json(self.root / 'manifest.json', {'teacher': TEACHER, 'records': recs, 'written': time.time(),
                                                 'finish_notice_map': 1})   # lines may carry finish_notices (#375)

    def all_meta(self):
        return list(self.meta.values()) + [x['record'] for x in self.buf]

    def count(self):
        return len(self.meta) + len(self.buf)

    def tokens(self):
        return sum((r['meta'].get('max_context_tokens') or 0) for r in self.all_meta())

    def mean_tokens(self):
        n = self.count()
        return self.tokens() / n if n else None


def finalize(a):
    ex = Exporter(a.export)
    drop_tasks = set(Path(a.exclude_tasks).read_text().split()) if a.exclude_tasks else set()
    drop_ends = set(a.drop_ends or [])
    usable = {rid for rid, r in ex.meta.items() if r['task'] not in drop_tasks and r['meta']['end'] not in drop_ends
              and not r['meta'].get('finish_notice_problem')}   # #375 sftnotice: the notices it saw are lost
    ex.write_manifest(usable)
    print(json.dumps({'records': len(ex.meta), 'usable': len(usable), 'dropped_tasks': len(drop_tasks),
                      'dropped_ends': sorted(drop_ends)}))


def collect(a):
    from aa_evals.common.pi_eval import finish_map
    for r in a.runs:   # #375 sftnotice: refuse a v3 (non-sticky) notice run before any shard is written
        finish_map.run_settings(str(r))
    ex = Exporter(a.export)
    for r in a.runs:
        ex.collect(r)
    ex.flush()
    print(json.dumps({'records': ex.count(), 'tokens': ex.tokens()}))


def combine(a):
    """One SFT root from several driver exports (prep: --synth-roots OUT): one trace per task (the first usable record of
    the task, sources in the order given), every record in the manifest, usable = kept. --budget N keeps whole traces
    of a seeded random order of tasks while the running total (meta.max_context_tokens, the drivers' count) stays
    <= N. Shards are rewritten with the kept lines only, written verified (temp, read back, line count) and checksummed
    (attempts/index.json, SHA256SUMS). Context-limit traces stay usable; meta.end flags them."""
    import random
    srcs = [Path(x) for x in a.sources]
    recs, order = {}, []
    for s in srcs:
        for r in json.loads((s / 'manifest.json').read_text())['records']:
            if r['id'] not in recs:
                recs[r['id']] = (s, r)
                order.append(r['id'])
    first = {}
    for rid in order:
        r = recs[rid][1]
        if r.get('usable', True) and r.get('finalized', True):
            first.setdefault(r['task'], rid)
    tok = lambda rid: int(recs[rid][1]['meta'].get('max_context_tokens') or 0)  # noqa: E731
    keep, total, over = set(), 0, 0
    tasks = sorted(first)
    if a.budget:
        random.Random(a.seed).shuffle(tasks)
    for t in tasks:
        rid = first[t]
        if a.budget and total + tok(rid) > a.budget:
            over += 1
            continue
        keep.add(rid)
        total += tok(rid)
    out = Path(a.out)
    (out / 'attempts').mkdir(parents=True, exist_ok=True)
    index, sums, buf, found = {'shards': {}, 'attempts': {}}, [], [], set()

    def flush():
        n = len(index['shards'])
        name = f'shard-{n:05d}.jsonl.gz'
        data = gzip.compress(''.join(json.dumps(x) + '\n' for x in buf).encode(), mtime=0)
        want = len(buf)

        def check(b):
            if gzip.decompress(b).count(b'\n') != want:
                raise ValueError('shard line count')
        durable_write(out / 'attempts' / name, data, check=check)
        h = hashlib.sha256(data).hexdigest()
        index['shards'][str(n)] = {'lines': name, 'n': want, 'sha256': h}
        sums.append(f'{h}  attempts/{name}')
        for x in buf:
            index['attempts'][x['id']] = [n]
        buf.clear()

    for s in srcs:
        sidx = json.loads((s / 'attempts' / 'index.json').read_text())
        for n in sorted(sidx['shards'], key=int):
            sh = sidx['shards'][n]
            raw = (s / 'attempts' / sh['lines']).read_bytes()
            if sh.get('sha256') and hashlib.sha256(raw).hexdigest() != sh['sha256']:   # shards before 78b5929: none
                raise SystemExit(f'{s}: shard {sh["lines"]} checksum mismatch')
            lines = gzip.decompress(raw).decode().splitlines()   # gzip's CRC checks every shard
            if len(lines) != sh['n']:
                raise SystemExit(f'{s}: shard {sh["lines"]} has {len(lines)} lines, index says {sh["n"]}')
            for line in lines:
                x = json.loads(line)
                if x['id'] in keep and x['id'] not in found and recs[x['id']][0] == s:
                    x['record'] = {**x['record'], 'usable': True, 'finalized': True}
                    buf.append(x)
                    found.add(x['id'])
                    if len(buf) >= 200:
                        flush()
    if buf:
        flush()
    missing = keep - found
    if missing:
        raise SystemExit(f'{len(missing)} kept records have no shard line (e.g. {sorted(missing)[:3]})')
    write_json(out / 'attempts' / 'index.json', index)
    records = [{**recs[rid][1], 'usable': rid in keep, 'finalized': True, 'source': str(recs[rid][0])} for rid in order]
    ends = {}
    for rid in keep:
        e = recs[rid][1]['meta'].get('end')
        ends[e] = ends.get(e, 0) + 1
    man = {'teacher': TEACHER, 'benchmark': 'tbsci_lite', 'finish_notice_map': 1, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'sources': [str(x) for x in srcs],
           'budget': ({'tokens': a.budget, 'seed': a.seed, 'order': 'seeded random order of tasks, whole traces'}
                      if a.budget else None),
           'counts': {'kept': len(keep), 'budget': over, 'duplicate_task': sum(1 for r in recs.values() if r[1].get('usable', True)) - len(first),
                      'not_usable': sum(1 for r in recs.values() if not r[1].get('usable', True))},
           'ends': ends, 'n_records': len(records), 'n_usable': len(keep), 'usable_tokens': total, 'records': records}
    write_json(out / 'manifest.json', man)
    durable_write(out / 'SHA256SUMS', '\n'.join(sums) + '\n', check=lambda b: None)
    print(json.dumps({k: v for k, v in man.items() if k != 'records'}))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    c = sp.add_parser('collect')
    c.add_argument('--export', required=True)
    c.add_argument('--runs', nargs='+', required=True)
    f = sp.add_parser('finalize')
    f.add_argument('--export', required=True)
    f.add_argument('--exclude-tasks')
    f.add_argument('--drop-ends', nargs='*')
    m = sp.add_parser('combine')
    m.add_argument('--out', required=True)
    m.add_argument('--sources', nargs='+', required=True)
    m.add_argument('--budget', type=float, default=0)
    m.add_argument('--seed', type=int, default=353)
    a = ap.parse_args(argv)
    {'collect': collect, 'finalize': finalize, 'combine': combine}[a.op](a)


if __name__ == '__main__':
    main()
