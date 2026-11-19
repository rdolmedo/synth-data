"""Issue #119: the thorough decontamination of issue #19 on a task_synthesis production pool, as tasks arrive.

Same criterion, rubrics, fingerprints and stopping rule as decontam/synth.py (student 1, 7,296 tasks): a candidate is
excluded if it is a variant of one specific TB2.1 or TB-Lite task (any REUSE among its pair's pooled samples, or
WEAK_REUSE in all of them) or carries a Terminal-Bench canary GUID. Pairs are never all-pairs: each test task's top-k
window per retrieval signal is judged, excluded candidates leave the window, and the next window is judged until a
window holds no unjudged pair. What changes is the scale (~150-300k tasks instead of 7k) and the input:

- Input: task streams (<stream>/chunks/NNNN.tar, the production layout) and bundle tars (tasks/<name>/...), read
  straight from the tars (no extraction). A real-asset task's environment/tasksynth/asset.tar.zst is opened: its
  files count as environment files (the setup script hands them to the agent), hashed for the identical-file signal,
  text heads shingled and shown to the judge, with the asset's provenance (title, source, URL).
- One pass near the end (user, 2026-11-03: only the final top-50 windows matter, so earlier passes waste judge
  time): while generation runs, each chunk is only read once into a per-unit cache (features, shingle hashes,
  embeddings, bundles). The main snapshot starts at ~0.9 of the planned pool (--checkpoints 0.9 --expected-tasks N),
  so it overlaps the last stretch of generation on the generator servers, or when the round writes STOP (token budget
  covered) if that comes first. The last snapshot, at the streams' END, recomputes
  the signals over the whole pool and judges only the pairs new in the windows, with the same stopping rule, so the
  list is as thorough as one pass at the end. (Earlier snapshots, --checkpoints, are possible but judge pairs that
  later fall out of the windows.)
- Forced pairs (a candidate whose spec carries a detail fitted on <= 2 TB tasks, or whose instruction names a test
  task) are a windowed signal, ordered by their best rank on the other six, instead of all being judged.
- Windowed signals (user, 2026-11-04; --signals): instr, instr_tests, embed, fp, forced. ids and overlap are still
  computed (the judge's prompt shows them) but get no windows: neither found a flagged task alone on student 1 or the
  #125 pilot, and they were ~26% of the pairs.
- Deep rule (#19's, --k-max 100): a flagged pair in the bottom half of a test task's window doubles that test task's
  window (50 -> 100; a larger --k-max keeps doubling), so a cluster of variants of one test task is followed past
  rank 50. Student 1 (fresh window loop, #19's verdicts): top-50 finds 34/36 of #19's list, the deep rule 36/36.
- Boilerplate thresholds (shingles or files shared by more than 8 / 3 tasks are ignored) scale with the pool size
  from student 1's 7,296 tasks, so they keep meaning "template text". Above --df-sample tasks, a candidate's own
  shingle count is taken against boilerplate estimated on a stable hash sample; everything touching a test task's
  shingles stays exact.

  python -m decontam.pool update   --work W --pool P [P ...]    read new chunks into W/units/ (CPU)
  python -m decontam.pool snapshot --work W [--serving S]       signals + judge windows until none unjudged + list
  python -m decontam.pool run      --work W --production <ROOT>/production/<name>/run.json --follow
        (= --pool <its stream> --expected-tasks <tasks_needed> --checkpoints 0.9 --snapshot-when <round>/STOP
           --serving <name>-ds1+... --hold <run dir>/decontam.HOLD --publish <ROOT>/decon/<name>
           --admission <ROOT>/rounds/<round>/admission.json: raised by the judge's load during the main pass)
  python -m decontam.pool final    --work W --out DIR            the list from the verdicts so far

P: a stream directory (chunks/), or a bundle tar. `run --follow` polls the streams until each has END.json and all
its chunks are read, takes the main snapshot when --snapshot-when appears (or at --checkpoints, or when W/CHECKPOINT
is touched), and a last one at the end, which it publishes (unless more than --hold-rate is excluded).
Lists: W/lists/<tasks>/decontamination-variants.json (SFT prep: --exclude-tasks), W/lists/LATEST.json.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time

import numpy as np

from . import render as RD
from . import retrieve as R
from . import synth as SY
from . import tasks as T
from . import variants as V

HERE = Path(__file__).resolve().parent
FIELDS = list(R.FIELDS)                       # instruction, tests, dockerfile, env, solution
SIX = ('instr', 'instr_tests', 'ids', 'overlap', 'embed', 'fp')
SIGNALS = SIX + ('forced',)
# signals whose windows are judged (user, 2026-11-04): ids and overlap never found a flagged task alone on student 1
# (36) or the #125 pilot (11) and were ~26% of the pairs. Both are still computed: the judge's prompt shows them and
# forced pairs are ordered by the best rank on all six.
WINDOWED = ('instr', 'instr_tests', 'embed', 'fp', 'forced')
NONE = 10 ** 9
ASSET = 'environment/tasksynth/asset.tar.zst'
ASSET_PREFIX = 'environment/tasksynth/asset/'
ASSET_FILES = 2000        # asset members read per task (the rest are counted)
ASSET_TEXT = 64 << 10     # text kept per asset file (the judge sees a ~16k-token environment budget in all)
TREE_CAP = 150
REF_TASKS = 7296          # student 1, the pool the boilerplate thresholds were set on
FINGERPRINTS = ('/data/aai/tbenchtrain/decontam_calibforge/variants/fp/fingerprints_v3.jsonl,'
                '/data/aai/tbenchtrain/decontam_calibforge/variants/fp/fingerprints_tblite_ds.jsonl')
RUBRIC_FP = HERE / 'rubric_reuse_synth.md'
RUBRIC_NOFP = HERE / 'rubric_reuse_nofp_synth.md'
LABELS = 'REUSE,WEAK_REUSE,NO_REUSE'
# judge arms: (verdict file, rubric, fingerprints?)  'fp' and 'nofp' split the test tasks by fingerprint; 'lite' is
# the second TB-Lite arm (no fingerprint, closest candidates only), since its fingerprints are DeepSeek's (#19)
ARMS = {'fp': ('verdicts-fp.jsonl', RUBRIC_FP, True), 'nofp': ('verdicts-nofp.jsonl', RUBRIC_NOFP, False),
        'lite': ('verdicts-nofp-lite.jsonl', RUBRIC_NOFP, False)}
FINAL_ARMS = (('fingerprint', ('fp', 'nofp')), ('nofp-tblite', ('lite',)))
TOKEN = re.compile(r'[a-z0-9_]+')


def log(work, msg):
    line = f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {msg}'
    print(line, flush=True)
    with open(Path(work) / 'events.log', 'a') as f:
        f.write(line + '\n')


def atomic_json(path, obj):
    tmp = Path(str(path) + '.partial')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little')


# ----------------------------------------------------------------------------------------------------- reading tars

def text_mode(data):
    """bytes -> str as Path.read_text(errors='replace') gives it (universal newlines)."""
    return data.decode(errors='replace').replace('\r\n', '\n').replace('\r', '\n')


def member_name(name):
    return (name[2:] if name.startswith('./') else name).lstrip('/')


def _asset_members(head, rest, info):
    """(path, size, first MAX_READ bytes) of the files in a zstd tar whose bytes are head + rest (a file object)."""
    try:
        p = subprocess.Popen(['zstd', '-dcq', '--long=31'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
    except OSError as e:
        info['asset_error'] = f'zstd: {e!r}'[:300]
        return []

    def feed():
        try:
            p.stdin.write(head)
            while block := rest.read(1 << 20):
                p.stdin.write(block)
        except (BrokenPipeError, OSError):
            pass
        finally:
            try:
                p.stdin.close()
            except OSError:
                pass
    th = threading.Thread(target=feed, daemon=True)
    th.start()
    out = []
    try:
        with tarfile.open(fileobj=p.stdout, mode='r|') as t:
            for m in t:
                if not m.isreg():
                    continue
                if len(out) >= ASSET_FILES:
                    info['asset_files_skipped'] = info.get('asset_files_skipped', 0) + 1
                    continue
                out.append((member_name(m.name), m.size, t.extractfile(m).read(T.MAX_READ)))
    except (tarfile.TarError, OSError, EOFError) as e:
        info['asset_error'] = repr(e)[:300]
    finally:
        p.stdout.close()
        th.join()
        p.wait()
    return out


def read_tasks(tar_path, assets=True):
    """Yield one task at a time from a bundle tar (tasks/<name>/...; each task's members contiguous):
    {'name', 'instruction', 'toml', 'members': [(rel, size, head)], 'tasksynth': dict|None, 'assets': [...],
    'info': {...}}."""
    cur = None

    def finish(c):
        c['members'].sort(key=lambda m: m[0].split('/'))
        return c
    seen = set()
    with tarfile.open(tar_path, 'r|') as t:
        for m in t:
            parts = member_name(m.name).split('/')
            if len(parts) < 3 or parts[0] != 'tasks' or not parts[2]:
                continue
            name, rel = parts[1], '/'.join(parts[2:])
            if cur is None or name != cur['name']:
                if cur is not None:
                    yield finish(cur)
                if name in seen:
                    raise ValueError(f'{tar_path}: members of task {name} are not contiguous')
                seen.add(name)
                cur = {'name': name, 'instruction': None, 'toml': '', 'members': [], 'tasksynth': None, 'assets': [],
                       'info': {}, 'data': {}}
            if m.issym() or not (m.isreg() or m.islnk()):
                continue
            if m.islnk():   # a hard link to an earlier member of the same task
                target = member_name(m.linkname).split('/', 2)
                data = cur['data'].get(target[2] if len(target) == 3 else '')
                if data is None:
                    continue
                cur['members'].append((rel, len(data), data))
                continue
            f = t.extractfile(m)
            if rel == 'instruction.md':
                cur['instruction'] = text_mode(f.read())
            elif rel == 'task.toml':
                cur['toml'] = text_mode(f.read())
            elif rel == 'tasksynth.json':
                raw = f.read()
                cur['members'].append((rel, len(raw), raw[:T.MAX_READ]))
                try:
                    cur['tasksynth'] = json.loads(raw)
                except ValueError:
                    cur['tasksynth'] = {}
            elif rel == ASSET and assets:
                head = f.read(T.MAX_READ)
                cur['members'].append((rel, m.size, head))
                cur['assets'] = _asset_members(head, f, cur['info'])
            else:
                data = f.read(T.MAX_READ)
                cur['members'].append((rel, m.size, data))
                if len(data) <= 1 << 16:
                    cur['data'][rel] = data
        if cur is not None:
            yield finish(cur)


def load_task(raw):
    """decontam.tasks' task dict (+ the asset's files as environment files) from read_tasks' record."""
    synth = raw['tasksynth'] is not None
    task = T.from_members(raw['name'], raw['instruction'] or '', raw['toml'],
                          [(r, s, d) for r, s, d in raw['members']], synth=synth)
    for path, size, data in raw['assets']:
        entry = T.file_entry(ASSET_PREFIX + path, size, data)
        if 'text' in entry:
            entry['text'] = data[:ASSET_TEXT].decode(errors='replace')
            entry['partial_read'] = size > ASSET_TEXT
        task['files'].append(entry)
    return task


def asset_section(meta, n_files, info):
    if not meta:
        return None
    desc = (meta.get('description') or '').strip().replace('\n', ' ')
    lines = ['### Real asset (tasksynth.json; the setup script unpacks its files for the agent)',
             f'- Title: {meta.get("title", "?")}',
             f'- Source: {meta.get("source", "?")} {meta.get("url") or meta.get("page_url") or ""} '
             f'(version {meta.get("version", "?")}, license {meta.get("license", "?")})',
             f'- Description: {desc[:600]}{"..." if len(desc) > 600 else ""}',
             f'- Files: {n_files} read from {ASSET} (listed below under {ASSET_PREFIX})'
             + (f', {info["asset_files_skipped"]} more not read' if info.get('asset_files_skipped') else '')
             + (f'; archive unreadable: {info["asset_error"]}' if info.get('asset_error') else '')]
    return '\n'.join(lines) + '\n'


# ------------------------------------------------------------------------------------------------------ unit caches

def doc_hashes(text):
    """retrieve.tfidf's unigram + bigram features of one document, hashed: (uint64 keys, counts)."""
    w = TOKEN.findall(text.lower())
    c = collections.Counter(w + [a + ' ' + b for a, b in zip(w, w[1:])])
    if not c:
        return np.zeros(0, np.uint64), np.zeros(0, np.uint32)
    keys = np.array([h64(k) for k in c], dtype=np.uint64)
    vals = np.array(list(c.values()), dtype=np.uint32)
    o = np.argsort(keys)
    return keys[o], vals[o]


def task_record(raw, side):
    """(features without shingles, {field: sorted uint64 shingles}, bundle row) of one task."""
    task = load_task(raw)
    f = R.task_features(task)
    spec, asset = {}, None
    if side == 'cf' and raw['tasksynth'] is not None:   # synth.candidate_features: the sketch stands in for tests
        sketch = T.field_text(task, 'solution')
        f['tests'] = sketch
        f['ids'] |= {x for x in T.IDENT.findall(sketch) if x != '/app'}
        spec = raw['tasksynth'].get('spec', {}) or {}
        asset = raw['tasksynth'].get('asset')
    sh = {k: np.array(sorted(v), dtype=np.uint64) for k, v in f.pop('shingles').items()}
    extra = asset_section(asset, len(raw['assets']), raw['info']) if asset else None
    row = RD.render_task(task, side, synth=side == 'cf' and raw['tasksynth'] is not None, extra=extra,
                         tree_cap=(ASSET_PREFIX, TREE_CAP))
    rec = {'name': f['name'], 'instruction': f['instruction'], 'tests': f['tests'], 'ids': sorted(f['ids']),
           'blobs': sorted(f['blobs']), 'canaries': f['canaries'], 'spec': spec,
           'asset': {k: asset.get(k) for k in ('asset_id', 'title', 'url', 'source')} if asset else None,
           'asset_files': len(raw['assets']), 'info': raw['info'],
           'bundle_ids': row['ids'], 'bundle_files': row['files'], 'tokens': row['tokens'], 'truncated': row['truncated'],
           'tf': {'instr': doc_hashes(f['instruction']), 'instr_tests': doc_hashes(f['instruction'] + '\n' + f['tests'])}}
    return rec, sh, row


_MODEL = None


def embed(texts):
    global _MODEL
    if _MODEL is None:
        import torch
        torch.set_num_threads(int(os.environ.get('DECONTAM_TORCH_THREADS', '2')))
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
    return _MODEL.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)


def save_shingles(path, shs):
    """{field: [sorted uint64 arrays per task]} -> one npz of CSR (offsets, values) per field."""
    arrs = {}
    for k, v in shs.items():
        arrs[f'{k}_off'] = np.cumsum([0] + [len(x) for x in v]).astype(np.int64)
        arrs[f'{k}_val'] = np.concatenate(v) if v else np.zeros(0, np.uint64)
    tmp = Path(str(path) + '.partial.npz')
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def load_shingles(path):
    with np.load(path) as z:
        return {k: (z[f'{k}_off'], z[f'{k}_val']) for k in FIELDS}


def _init_worker(bench, title, torch_threads=None):
    RD.REF_BENCH.update(bench)
    RD.CANDIDATE_TITLE = title
    if torch_threads:
        os.environ['DECONTAM_TORCH_THREADS'] = str(torch_threads)


def build_unit(args):
    """One tar -> W/units/<uid>.pkl (+ .bundles.jsonl). Returns (uid, names, seconds, info)."""
    work, uid, tar_path, side = args
    t0 = time.time()
    recs, shs, names = [], {k: [] for k in FIELDS}, []
    out = Path(work) / 'units'
    tmp_b = out / f'{uid}.bundles.jsonl.partial'
    with open(tmp_b, 'w') as fb:
        for raw in read_tasks(tar_path, assets=side == 'cf'):
            if raw['instruction'] is None:
                continue
            rec, sh, row = task_record(raw, side)
            fb.write(json.dumps(row) + '\n')
            recs.append(rec)
            names.append(rec['name'])
            for k in FIELDS:
                shs[k].append(sh[k])
    E = embed([r['instruction'] for r in recs]) if recs else np.zeros((0, 384), np.float32)
    blob = {'uid': uid, 'side': side, 'tar': str(tar_path), 'names': names, 'recs': recs, 'embed': E,
            'seconds': round(time.time() - t0, 1)}
    save_shingles(out / f'{uid}.sh.npz', shs)
    tmp = out / f'{uid}.pkl.partial'
    tmp.write_bytes(pickle.dumps(blob, protocol=5))
    os.replace(tmp_b, out / f'{uid}.bundles.jsonl')
    os.replace(tmp, out / f'{uid}.pkl')   # last: a unit counts as read once its .pkl exists
    info = collections.Counter()
    for r in recs:
        info['asset_tasks'] += bool(r['asset'])
        info['asset_files'] += r['asset_files']
        info['asset_errors'] += 'asset_error' in r['info']
    return uid, names, round(time.time() - t0, 1), dict(info)


# -------------------------------------------------------------------------------------------------------- the pool

def sources(spec):
    """[(uid, tar, identity)] of a pool entry: a stream's committed chunks, or one bundle tar."""
    p = Path(spec)
    if (p / 'chunks').is_dir():
        out, i = [], 0
        while (rec := p / 'chunks' / f'{i:04d}.json').exists():
            r = json.loads(rec.read_text())
            out.append((f'{p.name}.{i:04d}', str(p / r['tar']), r['sha256']))
            i += 1
        return out
    st = p.stat()
    name = p.parent.name if p.name == 'tasks.tar' else p.stem
    return [(name, str(p), f'{st.st_size}:{int(st.st_mtime)}')]


def stream_done(spec):
    """True for a bundle tar, or a stream with END.json whose chunks are all committed."""
    p = Path(spec)
    if not (p / 'chunks').is_dir():
        return True
    end = p / 'END.json'
    if not end.exists():
        return False
    n = json.loads(end.read_text()).get('chunks')
    return n is None or all((p / 'chunks' / f'{i:04d}.json').exists() for i in range(n))


def read_state(work):
    p = Path(work) / 'pool.json'
    return json.loads(p.read_text()) if p.exists() else {'refs': None, 'units': []}


def ensure_refs(work, workers):
    """TB2.1 + TB-Lite references (the tars of decontam.synth.REFS) as one unit, 'refs'."""
    st = read_state(work)
    if st['refs'] and (Path(work) / 'units' / 'refs.pkl').exists():
        return st
    raws = [(label, raw) for label, tar in SY.REFS.values() for raw in read_tasks(tar, assets=False)]
    bench = {raw['name']: label for label, raw in raws}
    _init_worker(bench, SY.CANDIDATE_TITLE)
    recs, shs, rows = [], {k: [] for k in FIELDS}, []
    for _, raw in raws:
        if raw['instruction'] is not None:
            rec, sh, row = task_record(raw, 'tb')
            recs.append(rec)
            rows.append(row)
            for k in FIELDS:
                shs[k].append(sh[k])
    order = sorted(range(len(recs)), key=lambda i: recs[i]['name'])
    recs, rows = [recs[i] for i in order], [rows[i] for i in order]
    shs = {k: [v[i] for i in order] for k, v in shs.items()}
    assert len({r['name'] for r in recs}) == len(recs), 'duplicate reference task'
    blob = {'uid': 'refs', 'side': 'tb', 'names': [r['name'] for r in recs], 'recs': recs,
            'embed': embed([r['instruction'] for r in recs]), 'bench': bench}
    (Path(work) / 'units').mkdir(parents=True, exist_ok=True)
    save_shingles(Path(work) / 'units' / 'refs.sh.npz', shs)
    (Path(work) / 'units' / 'refs.pkl').write_bytes(pickle.dumps(blob, protocol=5))
    with open(Path(work) / 'units' / 'refs.bundles.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    st['refs'] = {'tasks': len(recs), 'bench': dict(collections.Counter(bench.values()))}
    atomic_json(Path(work) / 'pool.json', st)
    log(work, f'references: {st["refs"]}')
    return st


def cmd_update(a):
    """Read every unit of the pool not cached yet (or whose chunk changed), in parallel."""
    from concurrent.futures import ProcessPoolExecutor
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    st = ensure_refs(work, a.workers)
    known = {u['uid']: u for u in st['units']}
    todo = []
    for spec in a.pool:
        for uid, tar, ident in sources(spec):
            u = known.get(uid)
            if u and u['identity'] != ident:
                raise SystemExit(f'{uid}: {tar} changed since it was read ({u["identity"]} -> {ident}); a published '
                                 f'chunk is immutable, so this is a different pool: use a new --work')
            if not u:
                todo.append((uid, tar, ident))
    if not todo:
        return st
    bench = pickle.loads((work / 'units' / 'refs.pkl').read_bytes())['bench']
    if not shutil.which('zstd'):
        log(work, 'WARNING: no zstd on PATH; real-asset archives will be listed but not opened')
    embed(['warm-up'])   # download the model once, before the workers load it
    t0, n = time.time(), 0
    import multiprocessing
    procs = min(a.workers, len(todo))   # few big units (bundle tars): fewer workers, more torch threads each
    with ProcessPoolExecutor(procs, mp_context=multiprocessing.get_context('spawn'), initializer=_init_worker,
                             initargs=(bench, SY.CANDIDATE_TITLE, max(1, a.workers // procs))) as pool:
        futs = {pool.submit(build_unit, (str(work), uid, tar, 'cf')): (uid, tar, ident) for uid, tar, ident in todo}
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            uid, tar, ident = futs[fut]
            _, names, secs, info = fut.result()
            st['units'].append({'uid': uid, 'tar': tar, 'identity': ident, 'tasks': len(names), 'seconds': secs,
                                'read_at': time.time(), **info})
            n += len(names)
            atomic_json(work / 'pool.json', st)
    log(work, f'update: {len(todo)} units, {n} tasks in {time.time() - t0:.0f}s; pool {sum(u["tasks"] for u in st["units"])} '
              f'tasks in {len(st["units"])} units')
    return st


class Pool:
    """The candidates of the units read so far (first unit wins a duplicate name), in a stable order: units by their
    read order in pool.json, tasks in tar order."""

    def __init__(self, work, units=None):
        self.work = Path(work)
        st = read_state(work)
        self.units = sorted(u['uid'] for u in st['units']) if units is None else units
        refs = pickle.loads((self.work / 'units' / 'refs.pkl').read_bytes())
        refs['shingles'] = load_shingles(self.work / 'units' / 'refs.sh.npz')
        self.refs, self.bench = refs, refs['bench']
        self.tbn = refs['names']
        self.cfn, self.recs, self.where, emb, self.dups, self._names = [], [], {}, [], 0, {}
        for uid in self.units:
            blob = self.unit(uid)
            self._names[uid] = blob['names']
            keep = []
            for i, n in enumerate(blob['names']):
                if n in self.where:
                    self.dups += 1
                    continue
                self.where[n] = (uid, i)
                keep.append(i)
                self.cfn.append(n)
                self.recs.append(blob['recs'][i])
            emb.append(blob['embed'][keep])
        self.E = np.concatenate(emb) if emb else np.zeros((0, 384), np.float32)
        self.ci = {n: j for j, n in enumerate(self.cfn)}
        self.ti = {n: i for i, n in enumerate(self.tbn)}

    def unit(self, uid):
        return pickle.loads((self.work / 'units' / f'{uid}.pkl').read_bytes())

    def shingle_units(self):
        """(global candidate index per unit task, -1 for a duplicate; {field: (offsets, values)}) per unit."""
        for uid in self.units:
            names = self.unit(uid)['names'] if uid not in self._names else self._names[uid]
            self._names[uid] = names
            g = np.array([self.ci[n] if self.where.get(n) == (uid, i) else -1 for i, n in enumerate(names)], np.int64)
            yield g, load_shingles(self.work / 'units' / f'{uid}.sh.npz')


# --------------------------------------------------------------------------------------------------------- signals

def tfidf_scores(tb_tf, cf_tf):
    """retrieve.cosine_tfidf on hashed features: identical up to float rounding and 64-bit hash collisions."""
    import scipy.sparse as sp
    docs = tb_tf + cf_tf
    keys = np.concatenate([k for k, _ in docs]) if docs else np.zeros(0, np.uint64)
    uniq, df = np.unique(keys, return_counts=True)
    vocab = uniq[df >= 2]
    vdf = df[df >= 2].astype(np.float64)
    idf = np.log(len(docs) / vdf)
    rows, cols, vals = [], [], []
    for i, (k, c) in enumerate(docs):
        pos = np.searchsorted(vocab, k)
        ok = (pos < len(vocab))
        ok[ok] = vocab[pos[ok]] == k[ok]
        rows.append(np.full(int(ok.sum()), i, np.int64))
        cols.append(pos[ok])
        vals.append((1 + np.log(c[ok].astype(np.float64))) * idf[pos[ok]])
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(docs), len(vocab)))
    X = sp.diags(1 / np.sqrt(X.multiply(X).sum(1).A1 + 1e-12)) @ X
    n = len(tb_tf)
    return (X[:n] @ X[n:].T).toarray()


def identifier_scores(tb_ids, cf_ids):
    """retrieve.identifier_scores through an inverted index over the test tasks' identifiers."""
    want = set().union(*tb_ids) if tb_ids else set()
    post = collections.defaultdict(list)
    df = collections.Counter()
    for j, ids in enumerate(cf_ids):
        for x in ids:
            df[x] += 1
            if x in want:
                post[x].append(j)
    S = np.zeros((len(tb_ids), len(cf_ids)), np.float64)
    n = len(cf_ids)
    for i, ids in enumerate(tb_ids):
        for x in sorted(ids):
            if post.get(x):
                S[i, post[x]] += math.log(n / df[x])
    return S


def stable_sample(names, size):
    """A hash sample of about `size` names whose membership only shrinks as the pool grows."""
    if len(names) <= size:
        return np.ones(len(names), bool)
    cut = size / len(names) * 2 ** 64
    return np.array([h64('df-sample:' + n) < cut for n in names])


def in_sorted(arr, vals):
    """(mask of vals found in the sorted unique arr, their positions there)."""
    pos = np.searchsorted(arr, vals)
    hit = pos < len(arr)
    hit[hit] = arr[pos[hit]] == vals[hit]
    return hit, pos


def overlap_signal(pool, thr, blob_thr, df_sample):
    """retrieve.overlap_scores (per-field shingle containment |A∩B|/min(|A|,|B|), both sides >= 25 shingles after
    boilerplate (df > thr over tb + cf) is dropped; plus identical rare files), streamed over the units with numpy.
    Exact while the pool has at most df_sample candidates; above, a candidate's own count of non-boilerplate
    shingles uses boilerplate estimated on a stable hash sample (the test tasks' shingles keep their exact df)."""
    n_tb, n_cf = len(pool.tbn), len(pool.cfn)
    tsh = pool.refs['shingles']
    tb_sets = {f: [tsh[f][1][tsh[f][0][i]:tsh[f][0][i + 1]] for i in range(n_tb)] for f in FIELDS}
    tb_union = {f: np.unique(np.concatenate(tb_sets[f])) for f in FIELDS}
    tb_df = {f: np.zeros(len(tb_union[f]), np.int64) for f in FIELDS}
    for f in FIELDS:
        for x in tb_sets[f]:
            tb_df[f][np.searchsorted(tb_union[f], x)] += 1
    sample = stable_sample(pool.cfn, df_sample)
    frac = float(sample.mean()) if n_cf else 1.0
    samp = {f: [] for f in FIELDS}
    for g, sh in pool.shingle_units():
        live = g >= 0
        for f in FIELDS:
            off, val = sh[f]
            owner = np.repeat(np.arange(len(g)), np.diff(off))
            use = live[owner]
            hit, pos = in_sorted(tb_union[f], val)
            tb_df[f] += np.bincount(pos[hit & use], minlength=len(tb_union[f]))
            keep = use & (sample[np.where(live, g, 0)][owner] if n_cf else use)
            if keep.any():
                samp[f].append(val[keep])
    boiler = {}
    for f in FIELDS:
        u, c = np.unique(np.concatenate(samp[f] + tb_sets[f]), return_counts=True)
        b = u[c > (thr if frac >= 1 else thr * frac)]
        boiler[f] = np.union1d(b[~in_sorted(tb_union[f], b)[0]], tb_union[f][tb_df[f] > thr])
    del samp
    inv, tb_len = {}, {}
    for f in FIELDS:
        kept = [x[~in_sorted(boiler[f], x)[0]] for x in tb_sets[f]]
        tb_len[f] = np.array([len(x) for x in kept])
        allk = np.concatenate(kept)
        owner = np.concatenate([np.full(len(x), i, np.int64) for i, x in enumerate(kept)])
        o = np.argsort(allk, kind='stable')
        allk, owner = allk[o], owner[o]
        u, start = np.unique(allk, return_index=True)
        inv[f] = (u, np.append(start, len(allk)), owner)
    contain = {f: np.zeros((n_tb, n_cf)) for f in FIELDS}
    for g, sh in pool.shingle_units():
        live = np.nonzero(g >= 0)[0]
        for f in FIELDS:
            off, val = sh[f]
            n_u = len(g)
            owner = np.repeat(np.arange(n_u), np.diff(off))
            isb = in_sorted(boiler[f], val)[0]
            nb = np.diff(off) - np.bincount(owner[isb], minlength=n_u)
            u, start, tbo = inv[f]
            hit, pos = in_sorted(u, val)
            hit &= (nb >= R.MIN_SHINGLES)[owner]
            p, cand = pos[hit], owner[hit]
            cnt = start[p + 1] - start[p]
            idx = np.repeat(start[p], cnt) + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
            M = np.bincount(tbo[idx] * n_u + np.repeat(cand, cnt), minlength=n_tb * n_u).reshape(n_tb, n_u)[:, live]
            ok = (M > 0) & (tb_len[f] >= R.MIN_SHINGLES)[:, None]
            ii, jj = np.nonzero(ok)
            contain[f][ii, g[live][jj]] = M[ii, jj] / np.minimum(tb_len[f][ii], nb[live][jj])
    blob_df = collections.Counter()
    for r in pool.refs['recs']:
        blob_df.update(r['blobs'])
    for r in pool.recs:
        blob_df.update(r['blobs'])
    tb_blobs = {}
    for i, r in enumerate(pool.refs['recs']):
        for b in r['blobs']:
            if blob_df[b] <= blob_thr:
                tb_blobs.setdefault(b, []).append(i)
    shared = np.zeros((n_tb, n_cf))
    for j, r in enumerate(pool.recs):
        for b in r['blobs']:
            for i in tb_blobs.get(b, ()):
                shared[i, j] += 1
    return contain, shared


def ranks(S):
    return R.ranks(S).astype(np.int32)


def forced_signal(pool, best):
    """Spec-detail and named-test-task pairs as a ranked signal: rank by best rank on the six others."""
    src = SY.spec_sources(set(pool.tbn))
    forced = collections.defaultdict(list)
    for j, r in enumerate(pool.recs):
        for d in SY.spec_details(r['spec']):
            for t in src.get(d, ()):
                forced[(pool.ti[t], j)].append('spec:' + ':'.join(d))
    pat = re.compile(r'(?<![\w/-])(' + '|'.join(map(re.escape, sorted(pool.tbn, key=len, reverse=True))) + r')(?![\w-])')
    named = {}
    for j, r in enumerate(pool.recs):
        hits = sorted(set(pat.findall(r['instruction'])))
        if hits:
            named[pool.cfn[j]] = hits
            for t in hits:
                forced[(pool.ti[t], j)].append('tb-named')
    Rf = np.full(best.shape, NONE, np.int32)
    by_tb = collections.defaultdict(list)
    for (i, j) in forced:
        by_tb[i].append(j)
    for i, js in by_tb.items():
        js = sorted(js, key=lambda j: (best[i, j], j))
        Rf[i, js] = np.arange(1, len(js) + 1)
    return Rf, forced, named, src


def compute_signals(work, fps, thr=None, blob_thr=None, df_sample=40000, save_scores=False):
    """All signal rank matrices over the pool so far -> (pool, {signal: ranks with NONE where it has no score},
    containment, shared files, report). Saves W/scores.npz (decontam.render.score_pairs layout)."""
    t0 = time.time()
    pool = Pool(work)
    n_cf = len(pool.cfn)
    thr = thr or max(R.BOILERPLATE_DF, int(R.BOILERPLATE_DF * n_cf / REF_TASKS))
    blob_thr = blob_thr or max(3, int(3 * n_cf / REF_TASKS))
    tb = pool.refs['recs']
    S = {'instr': tfidf_scores([r['tf']['instr'] for r in tb], [r['tf']['instr'] for r in pool.recs]),
         'instr_tests': tfidf_scores([r['tf']['instr_tests'] for r in tb], [r['tf']['instr_tests'] for r in pool.recs]),
         'ids': identifier_scores([set(r['ids']) for r in tb], [set(r['ids']) for r in pool.recs])}
    t1 = time.time()
    contain, shared = overlap_signal(pool, thr, blob_thr, df_sample)
    S['overlap'] = np.maximum.reduce([contain[f] for f in FIELDS]) + shared
    t2 = time.time()
    S['embed'] = pool.refs['embed'] @ pool.E.T
    have = [t for t in pool.tbn if t in fps]
    Q = embed([V.fp_query(fps[t]) if t in fps else '' for t in pool.tbn])
    S['fp'] = Q @ pool.E.T
    Rk = {s: ranks(S[s]) for s in SIX}
    Rm = {s: np.where(S[s] > 0, Rk[s], NONE).astype(np.int32) for s in SIX if s != 'fp'}
    Rm['fp'] = Rk['fp'].copy()
    Rm['fp'][[i for i, t in enumerate(pool.tbn) if t not in fps]] = NONE
    best = np.minimum.reduce([Rm[s] for s in SIX])
    Rm['forced'], forced, named, src = forced_signal(pool, best)
    t3 = time.time()
    if save_scores:   # for decontam.pool_check only (~5 GB at 300k tasks; nothing in the pipeline reads it)
        np.savez(Path(work) / 'scores.npz', tb=np.array(pool.tbn), cf=np.array(pool.cfn),
                 shared_files=shared.astype(np.float32), **{f'score_{s}': S[s].astype(np.float32) for s in SIX if s != 'fp'},
                 **{f'rank_{s}': Rk[s] for s in SIX if s != 'fp'}, **{f'contain_{f}': contain[f].astype(np.float32)
                                                                    for f in FIELDS})
    Rm['_raw'] = {s: Rk[s] for s in SIX if s != 'fp'}   # the five signals #19's judge prompt shows, unmasked
    report = {'refs': len(pool.tbn), 'cands': n_cf, 'duplicates_skipped': pool.dups, 'units': len(pool.units),
              'boilerplate_df': thr, 'blob_df': blob_thr, 'df_sample': min(df_sample, n_cf),
              'fingerprinted_refs': len(have), 'forced_pairs': len(forced), 'tb_named': named,
              'spec_sources': {':'.join(k): v for k, v in sorted(src.items())},
              'cf_canary_tasks': sorted(r['name'] for r in pool.recs if r['canaries']),
              'tb_with_canary': sum(bool(r['canaries']) for r in tb),
              'asset_tasks': sum(bool(r['asset']) for r in pool.recs),
              'seconds': {'tfidf_ids': round(t1 - t0), 'overlap': round(t2 - t1), 'embed_ranks_forced': round(t3 - t2),
                          'save': round(time.time() - t3), 'total': round(time.time() - t0)}}
    atomic_json(Path(work) / 'retrieval.json', report)
    return pool, Rm, contain, shared, forced, report


# --------------------------------------------------------------------------------------------------------- windows

def read_labels(path):
    """{(tb, cf): {sample: label}} of a verdict file (later lines win), without keeping the samples' text."""
    out = collections.defaultdict(dict)
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:   # a line cut short by a killed judge
                    continue
                if r.get('verdict'):
                    out[(r['tb'], r['cf'])][r['sample']] = r['verdict']['label']
    return out


def arm_labels(work):
    return {arm: {p: [v[k] for k in sorted(v)] for p, v in read_labels(Path(work) / f).items()}
            for arm, (f, _, _) in ARMS.items()}


def listed(labels, canaries):
    """Candidates the list excludes so far: any REUSE or all WEAK_REUSE in any arm, or a canary."""
    out = set(canaries)
    for ls in labels.values():
        out |= {c for _, c in V.excluded_pairs(ls, 'any', False) | V.all_weak(ls)}
    return out


def deep_k(pool, Rm, labels, signals, k, k_max):
    """{test-task index: window size} for test tasks whose windows go deeper than k: a flagged pair (any arm) in the
    bottom half of a test task's window doubles it, up to k_max (#19's deep rule, a flag at best rank 26-50 -> ranks
    51-100, applied until no flag sits in the bottom half). On student 1 the leak cluster (db-wal-recovery) had flags at
    ranks 55-93 that a fixed top-50 missed."""
    if k_max <= k:
        return {}
    best = np.minimum.reduce([Rm[s] for s in signals])
    ranks_of = collections.defaultdict(list)
    for ls in labels.values():
        for t, c in V.excluded_pairs(ls, 'any', False) | V.all_weak(ls):
            if t in pool.ti and c in pool.ci and best[pool.ti[t], pool.ci[c]] < NONE:
                ranks_of[pool.ti[t]].append(int(best[pool.ti[t], pool.ci[c]]))   # 0-based
    out = {}
    for i, rs in ranks_of.items():
        kk = k
        while kk < k_max and max(rs) >= kk // 2:
            kk = min(2 * kk, k_max)
        if kk > k:
            out[i] = kk
    return out


def windows(pool, Rm, fps, labels, gone, k, k_lite, signals=WINDOWED, k_max=None):
    """{arm: {(tb, cf): row}} of unjudged pairs in each test task's top-k per signal once listed candidates are out
    (top deep_k for test tasks with flags deep in their window); TB-Lite tasks also get the 'lite' arm's top-k_lite."""
    out = {arm: {} for arm in ARMS}
    deep = deep_k(pool, Rm, labels, signals, k, k_max or k)
    gone_mask = np.zeros(len(pool.cfn), bool)
    gone_mask[[pool.ci[c] for c in gone if c in pool.ci]] = True
    ng = int(gone_mask.sum())
    width = max([k, k_lite, *deep.values()]) + ng
    for s in signals:
        order = np.argsort(Rm[s], axis=1, kind='stable')[:, :width]   # rows: test tasks
        for i, t in enumerate(pool.tbn):
            arms = [('fp' if t in fps else 'nofp', deep.get(i, k))]
            if pool.bench[t] == 'Terminal-Bench Lite':
                arms.append(('lite', k_lite))
            cand = [j for j in order[i] if Rm[s][i, j] < NONE and not gone_mask[j]]
            for arm, kk in arms:
                for j in cand[:kk]:
                    c = pool.cfn[j]
                    if (t, c) not in labels[arm]:
                        out[arm].setdefault((t, c), {'tb': t, 'cf': c, 'via': []})['via'].append(s)
    return out, {pool.tbn[i]: kk for i, kk in sorted(deep.items())}


def write_signals(pool, Rm, contain, shared, pairs, path, blob_thr=3):
    """Each pair's overlap-signal block (decontam.render.signals_text), as in #19's prompts."""
    RD.CANDIDATE_POOL = f'{len(pool.cfn):,} synthetic training tasks'
    id_df = collections.Counter(x for r in pool.recs for x in r['bundle_ids'])
    blob_df = collections.Counter()
    for r in list(pool.refs['recs']) + pool.recs:
        blob_df.update({f[1] for f in r['bundle_files'] if f[2] >= 300})
    tbr = {r['name']: r for r in pool.refs['recs']}
    with open(path, 'w') as out:
        for p in pairs:
            i, j = pool.ti[p['tb']], pool.ci[p['cf']]
            a, b = tbr[p['tb']], pool.recs[j]
            row = {'rank': {s: int(r[i, j]) for s, r in Rm['_raw'].items()},
                   'containment': {f: round(float(contain[f][i, j]), 3) for f in FIELDS if contain[f][i, j]},
                   'shared_files': int(shared[i, j])}
            ids = sorted(set(a['bundle_ids']) & set(b['bundle_ids']), key=lambda x: (id_df[x], x))
            mine = {f[1]: f[0] for f in a['bundle_files'] if f[2] >= 300 and blob_df[f[1]] <= blob_thr}
            files = [(mine[f[1]], f[0]) for f in b['bundle_files'] if f[1] in mine]
            out.write(json.dumps({'tb': p['tb'], 'cf': p['cf'], 'text': RD.signals_text(row, ids, files)}) + '\n')


def write_bundles(pool, names, path):
    """The judge's bundle file for a pass: every reference plus the candidates in `names`."""
    by_unit = collections.defaultdict(set)
    for n in names:
        by_unit[pool.where[n][0]].add(n)
    with open(path, 'w') as out:
        with open(pool.work / 'units' / 'refs.bundles.jsonl') as f:
            for line in f:
                out.write(line)
        for uid, want in by_unit.items():
            with open(pool.work / 'units' / f'{uid}.bundles.jsonl') as f:
                for line in f:
                    row = json.loads(line)
                    if row['name'] in want and pool.where.get(row['name'], (None,))[0] == uid:
                        out.write(line)


def judge(work, arm, pairs_file, signals_file, bundles_file, samples, offset, a, tag):
    """One decontam.judge run (a subprocess) for an arm's pairs; returns its summary."""
    verdicts, rubric, with_fp = ARMS[arm]
    passdir = Path(pairs_file).parent
    cmd = [sys.executable, '-u', '-m', 'decontam.judge', '--work', str(work), '--out', str(Path(work) / verdicts),
           '--pairs', str(pairs_file), '--signals', os.path.relpath(signals_file, work),
           '--bundles', os.path.relpath(bundles_file, work), '--samples', str(samples), '--sample-offset', str(offset),
           '--effort', a.effort, '--concurrency', str(a.concurrency if arm == 'fp' else a.concurrency_small),
           '--ramp', '32', '--labels', LABELS, '--rubric', str(rubric), '--summary', str(passdir / f'{tag}.summary.json')]
    cmd += (['--serving', a.serving, '--load-file', str(Path(work) / 'judge-load' / f'{tag}.json')] if a.serving
            else ['--endpoint', a.endpoint])
    if with_fp:
        cmd += ['--fingerprints', str(Path(work) / 'fingerprints.jsonl')]
    with open(passdir / f'{tag}.log', 'w') as f:
        rc = subprocess.run(cmd, cwd=HERE.parent, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f'judge {tag} failed (exit {rc}); see {passdir / f"{tag}.log"}')
    return json.loads((passdir / f'{tag}.summary.json').read_text())


def resample_pairs(work, arm):
    """Pairs whose first two samples are not both NO_REUSE and that have fewer than 8 samples."""
    out = []
    for (t, c), v in sorted(read_labels(Path(work) / ARMS[arm][0]).items()):
        first = [v[k] for k in (0, 1) if k in v]
        if len(first) == 2 and first != ['NO_REUSE', 'NO_REUSE'] and len(v) < 8:
            out.append({'tb': t, 'cf': c})
    return out


def read_fps(paths, work):
    fps = SY.read_fps(paths)
    fp = Path(work) / 'fingerprints.jsonl'
    text = ''.join(open(p).read() for p in paths.split(','))
    if not fp.exists() or fp.read_text() != text:
        tmp = Path(str(fp) + '.partial')
        tmp.write_text(text)
        os.replace(tmp, fp)
    return fps


def snapshot(a):
    """Signals over the pool so far, then judge windows (2 samples; 6 more for any non-NO) until no window holds an
    unjudged pair; then the list for this pool. -> (list dir, counts)."""
    work = Path(a.work)
    fps = read_fps(a.fingerprints, work)
    pool, Rm, contain, shared, forced, rep = compute_signals(work, fps, df_sample=a.df_sample, save_scores=a.save_scores)
    n = len(pool.cfn)
    log(work, f'snapshot {n} tasks: signals in {rep["seconds"]["total"]}s (boilerplate df {rep["boilerplate_df"]}, '
              f'blob df {rep["blob_df"]}, forced pairs {rep["forced_pairs"]}, canary tasks {len(rep["cf_canary_tasks"])}, '
              f'asset tasks {rep["asset_tasks"]})')
    passes = work / 'passes'
    passes.mkdir(exist_ok=True)
    totals = collections.Counter()
    if a.dry_run:
        labels = arm_labels(work)
        gone = listed(labels, rep['cf_canary_tasks'])
        todo, deep = windows(pool, Rm, fps, labels, gone, a.k, a.k_lite, a.signals, a.k_max)
        d = passes / f'dry-{n:07d}'
        d.mkdir(exist_ok=True)
        for arm, rows in todo.items():
            V.write_pairs(sorted(rows.values(), key=lambda r: (r['tb'], r['cf'])), d / f'pairs-{arm}.jsonl')
        via = {arm: dict(collections.Counter(s for r in rows.values() for s in r['via'])) for arm, rows in todo.items()}
        log(work, f'dry run {n}: listed {len(gone)}; unjudged window pairs {({k: len(v) for k, v in todo.items()})}; '
                  f'by signal {via}; deeper windows {deep} -> {d}')
        return d, None
    for it in range(a.max_iterations):
        labels = arm_labels(work)
        gone = listed(labels, rep['cf_canary_tasks'])
        todo, deep = windows(pool, Rm, fps, labels, gone, a.k, a.k_lite, a.signals, a.k_max)
        sizes = {arm: len(v) for arm, v in todo.items()}
        if not any(sizes.values()):
            log(work, f'snapshot {n}: windows complete after {it} passes ({len(gone)} listed candidates)')
            break
        d = passes / f'{n:07d}-{it:02d}'
        d.mkdir(exist_ok=True)
        allp = [r for v in todo.values() for r in v.values()]
        write_signals(pool, Rm, contain, shared, allp, d / 'signals.jsonl', rep['blob_df'])
        write_bundles(pool, {r['cf'] for r in allp}, d / 'bundles.jsonl')
        log(work, f'snapshot {n} pass {it}: unjudged window pairs {sizes} (listed {len(gone)}; deeper windows '
                  f'{deep}) -> {d}')
        for arm, rows in todo.items():
            if rows:
                V.write_pairs(sorted(rows.values(), key=lambda r: (r['tb'], r['cf'])), d / f'pairs-{arm}.jsonl')
        runs = [(arm, d / f'pairs-{arm}.jsonl', 2, 0, f'{arm}-p') for arm in ARMS if todo[arm]]
        run_parallel(work, runs, d, a)
        # every pair whose first two samples are not both NO_REUSE gets six more (8 in all)
        runs = []
        for arm in ARMS:
            rs = resample_pairs(work, arm)
            if rs:
                V.write_pairs(rs, d / f'resample-{arm}.jsonl')
                runs.append((arm, d / f'resample-{arm}.jsonl', 6, 2, f'{arm}-rs'))
        if runs:
            log(work, f'snapshot {n} pass {it}: resampling {[(r[0], sum(1 for _ in open(r[1]))) for r in runs]}')
            # their candidates may come from earlier passes: rewrite this pass's bundles + signals to cover them
            extra = [json.loads(l) for _, p, *_ in runs for l in open(p)]
            write_signals(pool, Rm, contain, shared, allp + [e for e in extra if (e['tb'], e['cf']) not in
                                                             {(r['tb'], r['cf']) for r in allp}], d / 'signals.jsonl',
                          rep['blob_df'])
            write_bundles(pool, {r['cf'] for r in allp} | {e['cf'] for e in extra}, d / 'bundles.jsonl')
            run_parallel(work, runs, d, a)
        totals['passes'] += 1
        for arm, v in sizes.items():
            totals[f'pairs_{arm}'] += v
    else:
        log(work, f'snapshot {n}: WARNING stopped after {a.max_iterations} passes with unjudged window pairs left')
    out = work / 'lists' / f'{n:07d}'
    counts = final(work, out, pool, rep, fps, a.fingerprints)
    atomic_json(work / 'lists' / 'LATEST.json', {'tasks': n, 'units': pool.units, 'list': str(out / 'decontamination-variants.json'),
                                                'counts': counts, 'passes': dict(totals), 'time': time.time(),
                                                'windows': {'signals': list(a.signals), 'k': a.k, 'k_max': a.k_max,
                                                            'k_lite': a.k_lite}})
    log(work, f'snapshot {n}: list {out} excluded {counts["excluded_tasks"]} ({counts["exclusion_rate"]:.3%}), '
              f'pairs judged {counts["pairs_judged"]}')
    return out, counts


def missing_samples(work, arm, pairs_file, samples, offset):
    """Pairs of a judge run still missing one of their samples offset..offset+samples-1 (no verdict written: API
    errors or unparseable answers after the judge's retries)."""
    labels = read_labels(Path(work) / ARMS[arm][0])
    want = range(offset, offset + samples)
    return [r for r in map(json.loads, open(pairs_file))
            if any(s not in labels.get((r['tb'], r['cf']), {}) for s in want)]


def run_parallel(work, runs, d, a, fills=2):
    """The arms' judge runs side by side (each its own subprocess and concurrency). A pair left without one of its
    samples is judged again (up to `fills` more runs; the judge skips samples already there): a lost sample would
    otherwise count as judged with fewer samples (e.g. one WEAK_REUSE and no resampling, #234)."""
    errs = []

    def go(arm, pairs, samples, offset, tag):
        try:
            s = judge(work, arm, pairs, d / 'signals.jsonl', d / 'bundles.jsonl', samples, offset, a, tag)
            log(work, f'{d.name} {tag}: {json.dumps({k: v for k, v in s.items() if k not in ("args", "targets")})[:300]}')
            for i in range(fills):
                miss = missing_samples(work, arm, pairs, samples, offset)
                if not miss:
                    break
                fill = Path(pairs).with_name(f'{Path(pairs).stem}-fill{i}.jsonl')
                V.write_pairs(miss, fill)
                log(work, f'{d.name} {tag}: {len(miss)} pairs missing a sample; judging them again ({fill.name})')
                judge(work, arm, fill, d / 'signals.jsonl', d / 'bundles.jsonl', samples, offset, a, f'{tag}-fill{i}')
            else:
                if (miss := missing_samples(work, arm, pairs, samples, offset)):
                    log(work, f'{d.name} {tag}: WARNING {len(miss)} pairs still miss a sample after {fills} fills')
        except Exception as e:   # noqa: BLE001 -- reported after the others finish
            errs.append(e)
    ths = [threading.Thread(target=go, args=r) for r in runs]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise errs[0]


def final(work, out, pool, rep, fps, fingerprints):
    groups = [[str(Path(work) / ARMS[arm][0]) for arm in arms if (Path(work) / ARMS[arm][0]).exists()]
              for _, arms in FINAL_ARMS]
    report, labels, hit = SY.final_report(
        pool.bench, len(pool.cfn), rep['cf_canary_tasks'], fps, groups, [n for n, _ in FINAL_ARMS],
        f'task_synthesis pool: {len(pool.cfn)} tasks in {len(pool.units)} units', fingerprints,
        issue='internal issue 119')
    report['pool'] = {'units': pool.units, 'duplicates_skipped': pool.dups,
                      'boilerplate_df': rep['boilerplate_df'], 'blob_df': rep['blob_df']}
    SY.write_final(out, report, labels, hit)
    return report['counts']


def cmd_snapshot(a):
    snapshot(a)


def cmd_final(a):
    work = Path(a.work)
    fps = read_fps(a.fingerprints, work)
    pool = Pool(work)
    rep = json.loads((work / 'retrieval.json').read_text())
    print(json.dumps(final(work, Path(a.out), pool, rep, fps, a.fingerprints), indent=1))


def publish(work, src, dest, rate, hold_rate):
    if rate > hold_rate:
        log(work, f'HOLD: exclusion rate {rate:.3%} > {hold_rate:.0%}; list in {src}, not published')
        return False
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for f in ('decontamination-variants.pairs.jsonl', 'exclude-tasks.txt'):
        shutil.copy(src / f, dest / f)
    shutil.copy(src / 'decontamination-variants.json', dest / '.decontamination-variants.json.partial')
    os.replace(dest / '.decontamination-variants.json.partial', dest / 'decontamination-variants.json')
    log(work, f'PUBLISHED {dest} (exclusion rate {rate:.3%})')
    return True


def checkpoints(a):
    if not a.checkpoints:
        return []
    vals = [float(x) for x in a.checkpoints.split(',')]
    if all(v <= 1 for v in vals):
        if not a.expected_tasks:
            raise SystemExit('--checkpoints as fractions needs --expected-tasks')
        vals = [v * a.expected_tasks for v in vals]
    return sorted(int(v) for v in vals)


def admission_bump(work, path, base, judge_inflight, serving):
    """Raise the generation round's live admission target (<round>/admission.json "target_inflight", re-read by the
    round every 60 s) by the judge's in-flight requests per server, for a snapshot taken while generation still runs.
    The round counts every request on its servers as load, so without this it starts fewer tasks while the judge
    shares the servers (user, 2026-11-04). Other keys in the file are kept. The bump is recorded in
    W/admission-bump.json first, so admission_restore undoes it even after a crash. -> True if raised."""
    path = Path(path)
    if not path.parent.is_dir():
        log(work, f'admission: {path.parent} missing; target not raised')
        return False
    try:
        cur = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError) as e:
        log(work, f'admission: {path} unreadable ({e}); target not raised')
        return False
    if not isinstance(cur, dict):
        return False
    before = cur.get('target_inflight')
    now = before if before is not None else base
    if not now:   # the round is not load-gated (or its target is unknown): nothing to raise
        log(work, f'admission: target_inflight {now!r} ({path}, --admission-base {base!r}): not load-gated or '
                  f'unknown; target not raised')
        return False
    from . import judge as J
    servers = len({u for u, _ in J.serving_targets(serving)[1]}) if serving else 0
    per = math.ceil(judge_inflight / max(1, servers))
    target = int(now) + per
    atomic_json(Path(work) / 'admission-bump.json', {'path': str(path), 'had_key': 'target_inflight' in cur,
                                                     'before': before, 'target': target, 'time': time.time()})
    atomic_json(path, {**cur, 'target_inflight': target})
    log(work, f'admission: {path} target_inflight {now} -> {target} (judge ~{judge_inflight} in flight over '
              f'{servers} servers)')
    return True


def admission_restore(work):
    """Undo admission_bump (if W/admission-bump.json says one is in place). A target someone changed since is left."""
    rec = Path(work) / 'admission-bump.json'
    if not rec.exists():
        return
    st = json.loads(rec.read_text())
    path = Path(st['path'])
    try:
        cur = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        cur = None
    if not isinstance(cur, dict) or cur.get('target_inflight') != st['target']:
        log(work, f'admission: {path} changed since the bump (now {cur}); left as is')
    else:
        if st['had_key']:
            cur['target_inflight'] = st['before']
        else:
            cur.pop('target_inflight')
        if cur:
            atomic_json(path, cur)
        else:
            path.unlink(missing_ok=True)
        log(work, f'admission: {path} target_inflight restored to {st["before"] if st["had_key"] else "the round default"}')
    rec.unlink()


def production_defaults(a):
    """--production <ROOT>/production/<name>/run.json (task_synthesis.pipeline.production plan): the run's stream, its
    planned task count, its round's STOP, its generator servers, a hold file next to run.json and a publish dir."""
    run = json.loads(Path(a.production).read_text())
    from tbshared import serving   # the run's registry root (run.json serving_env), as its driver reads it
    serving.pin(run.get('serving_env'), f'run {run.get("name")}')
    from . import judge as J
    if J.serving_root() != serving.registry_root():   # the legacy DECONTAM_SERVING_ROOT may not move it elsewhere
        raise SystemExit(f'run {run.get("name")} uses registry {serving.registry_root()}, but '
                         f'{J.LEGACY_ROOT_ENV}={os.environ.get(J.LEGACY_ROOT_ENV)}: unset it')
    root = Path(os.path.abspath(a.production)).parents[2]
    st = Path(a.production).with_name('state.json')
    servers = sorted(json.loads(st.read_text()).get('server_jobs', {})) if st.exists() else []
    a.pool = a.pool or [run['stream']]
    a.expected_tasks = a.expected_tasks or run['tasks_needed']
    a.checkpoints = a.checkpoints or '0.9'
    a.snapshot_when = a.snapshot_when or str(root / 'rounds' / run['round'] / 'STOP')
    a.serving = a.serving or '+'.join(servers or [f'{run["name"]}-ds{i + 1}' for i in range(run.get('servers', 2))])
    a.hold = a.hold or str(Path(a.production).with_name('decontam.HOLD'))
    a.publish = a.publish or str(root / 'decon' / run['name'])
    rd = root / 'rounds' / run['round']
    a.admission = a.admission or str(rd / 'admission.json')
    if a.admission_base is None:
        try:
            a.admission_base = json.loads((rd / 'job.json').read_text())['args'].get('target_inflight')
        except (OSError, ValueError, KeyError):
            pass
    return a


class PassUpdater:
    """Reads new chunks into W/units every --interval while a judge pass runs (#234 R2), with --pass-update-workers
    processes, so the last snapshot at END only has the last chunks to read instead of everything streamed during the
    main pass (m1-hero: 9,874 tasks, 12 min of CPU while both DeepSeek servers waited). The running snapshot does not
    see the new units (it read pool.json when it started); stop() waits for an update in progress, so the run loop's
    own update never runs alongside it."""

    def __init__(self, a):
        self.a = argparse.Namespace(**vars(a))
        self.a.workers = a.pass_update_workers or max(1, a.workers // 4)
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        while not self.event.wait(self.a.interval):
            try:
                before = len(read_state(self.a.work)['units'])
                st = cmd_update(self.a)
                if len(st['units']) > before:
                    log(self.a.work, f'update during the pass: {len(st["units"]) - before} units read, pool '
                                     f'{sum(u["tasks"] for u in st["units"])} tasks')
            except Exception as e:   # noqa: BLE001 -- the run loop's own update retries after the pass
                log(self.a.work, f'update during the pass failed: {e!r}')

    def stop(self):
        self.event.set()
        self.thread.join()


def cmd_run(a):
    """update (every --interval) -> a snapshot when --snapshot-when appears (or at a checkpoint, or on W/CHECKPOINT)
    -> the last snapshot once every stream ended, which judges only the pairs new in the windows since."""
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    if a.production:
        production_defaults(a)
    cps = checkpoints(a)
    log(work, f'run: pool {a.pool}, serving {a.serving or a.endpoint}, snapshot when {a.snapshot_when}, '
              f'checkpoints {cps}, follow {a.follow}, admission {a.admission} (base {a.admission_base})')
    admission_restore(work)   # a bump left by a killed run
    if a.hold:   # the generation driver keeps its DeepSeek servers while this file exists (the last pass needs them)
        Path(a.hold).parent.mkdir(parents=True, exist_ok=True)
        Path(a.hold).write_text(json.dumps({'work': str(work), 'host': os.uname().nodename, 'pid': os.getpid(),
                                            'job': os.environ.get('CONDOR_JOB_ID'), 'time': time.time()}))
    while True:
        done = all(stream_done(p) for p in a.pool)   # read before update: chunks committed after END count too
        st = cmd_update(a)
        n = sum(u['tasks'] for u in st['units'])
        latest = json.loads((work / 'lists' / 'LATEST.json').read_text()) if (work / 'lists' / 'LATEST.json').exists() else {}
        last = latest.get('tasks', 0)
        trigger = (work / 'CHECKPOINT').exists()
        if trigger:
            (work / 'CHECKPOINT').unlink(missing_ok=True)
        # the main pass: at the first of --checkpoints (e.g. 0.9 of the planned pool) and the round's STOP (budget
        # covered); with --snapshot-when they are alternatives for one main pass, then the last one at END
        stop = bool(a.snapshot_when and not latest and Path(a.snapshot_when).exists())
        if stop:
            log(work, f'run: {a.snapshot_when} is there: main pass now')
        due = [c for c in cps if last < c <= n] if not (a.snapshot_when and latest) else []
        if (done or not a.follow or due or trigger or stop) and (n > last or not latest):
            # while generation runs, the judge's requests count against the round's admission target: raise it
            bumped = bool(a.admission and not done and admission_bump(
                work, a.admission, a.admission_base, a.concurrency + 2 * a.concurrency_small, a.serving))
            reader = None if done else PassUpdater(a)   # streams still open: keep reading chunks during the pass
            try:
                snapshot(a)
            finally:
                if reader:
                    reader.stop()
                if bumped:
                    admission_restore(work)
            latest = json.loads((work / 'lists' / 'LATEST.json').read_text())
        if done and a.publish:
            publish(work, Path(latest['list']).parent, a.publish, latest['counts']['exclusion_rate'], a.hold_rate)
        if done and a.hold and Path(a.hold).exists():
            Path(a.hold).unlink()
            log(work, f'released hold {a.hold}')
        if done or not a.follow:
            log(work, f'run: done ({n} tasks)' if done else f'run: stopped ({n} tasks; streams still open)')
            return
        time.sleep(a.interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['update', 'snapshot', 'run', 'final'])
    ap.add_argument('--work', required=True)
    ap.add_argument('--pool', nargs='*', default=[], help='stream directories and/or bundle tars')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('DECONTAM_WORKERS', '16')))
    ap.add_argument('--fingerprints', default=FINGERPRINTS)
    ap.add_argument('--serving', help="task_synthesis serving names joined by '+' (the judge's DeepSeek servers)")
    ap.add_argument('--endpoint', help='or one endpoint.json (decontam.serve_judge)')
    ap.add_argument('--effort', default='max')
    ap.add_argument('--concurrency', type=int, default=512, help='fingerprint arm, in flight over all servers')
    ap.add_argument('--concurrency-small', type=int, default=128, help='no-fingerprint and TB-Lite arms')
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--k-lite', type=int, default=10)
    ap.add_argument('--k-max', type=int, default=100, help='deep rule: a flag in the bottom half of a test task\'s '
                                                          'window doubles it, up to this (--k-max = --k: off)')
    ap.add_argument('--signals', default=','.join(WINDOWED), help=f'windowed signals, from {",".join(SIGNALS)}')
    ap.add_argument('--max-iterations', type=int, default=30)
    ap.add_argument('--df-sample', type=int, default=40000)
    ap.add_argument('--production', help='run: a task_synthesis production run.json; fills --pool, --expected-tasks, '
                                         '--checkpoints 0.9, --snapshot-when, --serving, --hold and --publish')
    ap.add_argument('--snapshot-when', help="run: take the main snapshot once this file exists (the production round's "
                                            "STOP, written when the token budget is covered), unless a checkpoint already "
                                            "took it; the last one follows at END")
    ap.add_argument('--checkpoints', help='pool sizes (or fractions of --expected-tasks) that trigger a snapshot')
    ap.add_argument('--expected-tasks', type=int)
    ap.add_argument('--follow', action='store_true', help='poll the streams until they end')
    ap.add_argument('--interval', type=int, default=600)
    ap.add_argument('--pass-update-workers', type=int, help='run: processes reading new chunks while a pass judges '
                                                            '(default: --workers / 4)')
    ap.add_argument('--publish', help='copy the last list here when every stream ended')
    ap.add_argument('--hold-rate', type=float, default=0.05, help='do not publish a list excluding more than this')
    ap.add_argument('--hold', help='run: a file that exists until the last list is written (the generation driver '
                                   'keeps its DeepSeek servers while it does)')
    ap.add_argument('--admission', help="run: the generation round's live admission.json; a snapshot taken while the "
                                        "streams are open raises its target_inflight by the judge's in-flight "
                                        "requests per server, and restores it after")
    ap.add_argument('--admission-base', type=int, help="run: the round's command-line --target-inflight (used when "
                                                       "admission.json has none; --production reads <round>/job.json)")
    ap.add_argument('--out', help='final: list directory')
    ap.add_argument('--dry-run', action='store_true', help='snapshot: signals and unjudged window pairs only')
    ap.add_argument('--save-scores', action='store_true', help='snapshot: write W/scores.npz (decontam.pool_check)')
    a = ap.parse_args()
    a.signals = tuple(x for x in a.signals.split(',') if x)
    if not a.signals or set(a.signals) - set(SIGNALS):
        ap.error(f'--signals: a comma list from {",".join(SIGNALS)}')
    if a.cmd in ('snapshot', 'run') and not (a.serving or a.endpoint or a.dry_run or a.production):
        ap.error('--serving or --endpoint')
    if a.cmd in ('snapshot', 'run') and (a.serving or a.production):
        from . import judge as J
        J.serving_root()   # a registry-root mismatch fails now, not hours later at the first judge pass
    {'update': cmd_update, 'snapshot': cmd_snapshot, 'run': cmd_run, 'final': cmd_final}[a.cmd](a)


if __name__ == '__main__':
    main()
