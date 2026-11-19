"""Export Super CritPt traces (pi_eval runs) as a packed synth/ campaign root that sft_qwen35/prepare.py reads as is.

  python -m m5.critpt.export --runs RUN_DIR [RUN_DIR ...] --out ROOT [--drop DECONTAM_DROP.txt] [--shard-size 2000]

  then: python sft_qwen35/prepare.py --source pi_native --arms nemotron35-super --synth-roots ROOT ...

Layout (synth/archive.py's, read by prepare.py `_synth_attempts`):
  ROOT/manifest.json         {"teacher": "nemotron35-super", "records": [{id, task, usable, finalized, ...}]}
  ROOT/attempts/index.json   {"shards": {"0": {"lines": "shard-00000.jsonl.gz", "n": ...}}, "attempts": {id: [0, i]}}
  ROOT/attempts/shard-*.jsonl.gz   one line per attempt: {id, task, record, sessions: [{path, text}],
                                   first_request (system prompt + tool schemas as sent), reward: null,
                                   finish_notices (runs with the Pi finish notice: the trial's notice map,
                                   aa_evals/common/pi_eval/finish_map.py; an attempt whose map is lost is dropped)}
Writes are verified (temp file, read-back, gzip/JSON parse, rename; shard sha256 in index.json). Kept: attempts whose Pi session exists and that wrote a parsable /work/answer.py (the trace submitted); the rest
(no submit, infrastructure failures) are counted in ROOT/export_stats.json and dropped. `reward` is null: generated
problems have no reference answers. --drop: problem ids removed by decontamination.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import io
import json
import tarfile
import zlib
from pathlib import Path

from aa_evals.common.pi_eval import finish_map

from . import safeio

TEACHER = 'nemotron35-super'


def read_attempt(adir: Path):
    """-> (record | None, reason). Reads only the files it needs from trial.tar.gz."""
    o = json.loads((adir / 'outcome.json').read_text())
    if o.get('infrastructure_failure'):
        return None, 'infrastructure_failure'
    arc = adir / 'trial.tar.gz'   # outcome.json's absolute trial_archive is stale once a run dir moves
    if not arc.exists():
        return None, 'no_archive'
    trial = o['trial_dir_in_archive']
    sessions, first, reward_txt, logs = [], None, None, {}
    with tarfile.open(arc, 'r:gz') as tf:
        for m in tf:
            if not m.isfile():
                continue
            n = m.name
            if finish_map.member_kind(n) and finish_map.member_kind(n) not in logs:   # #375 sftnotice
                logs[finish_map.member_kind(n)] = tf.extractfile(m).read().decode('utf-8', 'surrogateescape')
            elif n.startswith(trial + '/agent/pi/sessions/') and n.endswith('.jsonl'):
                sessions.append({'path': n[len(trial) + 1:],
                                 'text': tf.extractfile(m).read().decode('utf-8', 'surrogateescape')})
            elif '/pi-cont16k/first-request/' in n and n.endswith('.json') and first is None:
                first = json.loads(tf.extractfile(m).read())
            elif n == trial + '/verifier/reward.txt':
                reward_txt = tf.extractfile(m).read().decode().strip()
    if not sessions:
        return None, 'no_session'
    if not (first and first.get('messages') and first.get('tools')):
        return None, 'no_first_request'
    if reward_txt != '1':
        return None, 'no_answer_file'
    fn, window = finish_map.run_settings(str(adir.parent.parent))
    notices, problem = finish_map.attempt_notices(fn, window, logs.get('notices'),
                                                  (o.get('stats') or {}).get('max_prompt_tokens'))
    if problem:
        return None, problem.split(':')[0]
    st = o.get('stats') or {}
    task = adir.name.split('__')[0]
    return {'id': f'm5-critpt/{adir.parent.parent.name}/{adir.name}', 'task': task, 'sessions': sorted(
        sessions, key=lambda s: s['path']), 'first_request': first,
        **({'finish_notices': notices} if notices is not None else {}),
        'stats': {k: st.get(k) for k in ('end_state', 'model_requests', 'context_limit', 'output_cap_hits',
                                         'max_context_tokens', 'wall_sec')}}, 'ok'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--drop', default=None, help='file of problem ids to leave out (decontamination)')
    ap.add_argument('--shard-size', type=int, default=2000)
    ap.add_argument('--cut-from', default=None, help="a full export's manifest.json: export a seeded random subset of "
                    'its records (one per task) whose max_context_tokens sum to at most --budget-tokens')
    ap.add_argument('--budget-tokens', type=float, default=0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    drop = set(Path(a.drop).read_text().split()) if a.drop else set()
    keep = None
    if a.cut_from:   # seeded random by task: shuffle the full export's records, take them while they fit the budget
        import random
        recs = json.loads(Path(a.cut_from).read_text())['records']
        random.Random(a.seed).shuffle(recs)
        keep, tot = set(), 0
        for r in recs:
            t = r.get('max_context_tokens') or 0
            if tot + t > a.budget_tokens:
                continue
            keep.add(r['task'])
            tot += t
        print(f'cut: {len(keep)} of {len(recs)} records, {tot} tokens (budget {a.budget_tokens:.0f}, seed {a.seed})',
              flush=True)
    root = Path(a.out)
    (root / 'attempts').mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    records, index, seen = [], {'shards': {}, 'attempts': {}}, set()
    shard, buf = 0, []

    def flush():
        nonlocal shard, buf
        if not buf:
            return
        name = f'shard-{shard:05d}.jsonl.gz'
        raw = io.BytesIO()
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as g:
            for i, x in enumerate(buf):
                g.write((json.dumps(x, ensure_ascii=False) + '\n').encode('utf-8', 'surrogateescape'))
                index['attempts'][x['id']] = [shard, i]
        sha = safeio.write_verified(root / 'attempts' / name, raw.getvalue(), check=safeio.check_gzip_jsonl)
        index['shards'][str(shard)] = {'lines': name, 'n': len(buf), 'sha256': sha}
        shard, buf = shard + 1, []

    for run in a.runs:   # #375 sftnotice: refuse a v3 (non-sticky) notice run before any shard is written
        finish_map.run_settings(str(run))
    for run in a.runs:
        for adir in sorted((Path(run) / 'attempts').iterdir()):
            if not (adir / 'outcome.json').exists():
                counts['unfinished'] += 1
                continue
            if adir.name.split('__')[0] in drop:
                counts['decontam_dropped'] += 1
                continue
            if keep is not None and adir.name.split('__')[0] not in keep:
                counts['not_in_cut'] += 1
                continue
            try:
                r, why = read_attempt(adir)
            except (OSError, EOFError, tarfile.TarError, ValueError, zlib.error) as e:   # truncated/corrupt archive
                print(f'bad archive {adir}: {e!r}', flush=True)
                r, why = None, 'bad_archive'
            counts[why] += 1
            if r is None:
                continue
            if r['task'] in seen:   # one trace per problem
                counts['duplicate_task'] += 1
                continue
            seen.add(r['task'])
            rec = {'id': r['id'], 'task': r['task'], 'usable': True, 'finalized': True, 'teacher': TEACHER,
                   **r['stats']}
            records.append(rec)
            buf.append({'id': r['id'], 'task': r['task'], 'record': rec, 'sessions': r['sessions'],
                        'first_request': r['first_request'], 'reward': None,
                        **({'finish_notices': r['finish_notices']} if 'finish_notices' in r else {})})
            if len(buf) >= a.shard_size:
                flush()
    flush()
    safeio.write_json(root / 'attempts' / 'index.json', index)
    cut = {'cut_from': a.cut_from, 'budget_tokens': a.budget_tokens, 'seed': a.seed} if a.cut_from else None
    safeio.write_json(root / 'manifest.json', {'teacher': TEACHER, 'source': 'm5-critpt', 'runs': a.runs, 'cut': cut,
                                               'finish_notice_map': 1,   # lines may carry finish_notices (#375)
                                               'records': records}, indent=0)
    safeio.write_json(root / 'export_stats.json', {'kept': len(records), **counts}, indent=1)
    print(json.dumps({'kept': len(records), **counts}))


if __name__ == '__main__':
    main()
