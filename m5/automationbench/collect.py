"""Pack resolved Super attempts into M1's synth-archive layout and prune their trial archives (#359; /data budget).

  PYTHONPATH=. venvs/harbor/bin/python -m m5.automationbench.collect --runs 'GLOB' --out ROOT [--follow] [--prune]

ROOT/attempts/shard_NNNNN.jsonl.gz: one line per attempt, the line format sft_qwen35/prepare.py `_synth_attempts`
reads (synth/archive.py): {"id", "task", "record": {"usable", "finalized", "task"}, "sessions": [{"path", "text"}],
"first_request", "reward": null}, "finish_notices" (runs with the Pi finish notice: the trial's notice map,
aa_evals/common/pi_eval/finish_map.py), plus "m5": the run, the episode end, pi_eval's stats, the world-service call log
(heads), the final world, the closing message. ROOT/attempts/index.json {"shards": {n: {"lines"}}, "attempts":
{id: [n]}} is written atomically after each shard. ROOT/manifest.json (teacher, records) is written by export.py.
With --prune an attempt's trial.tar.gz is deleted once its shard and the index are on disk (outcome.json stays).
Run it as a CPU job: each archive is ~30 MB of gzip.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import tarfile
import time
from pathlib import Path

SHARD_SIZE = 200


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_attempt(pi_run: Path, row: dict) -> dict:
    """Session text, first wire request, world-service state and stats of one resolved attempt."""
    from aa_evals.common.pi_eval import port
    rec = port.attempt_record(pi_run, row)
    trial = rec.get('trial_dir_in_archive')
    archive = Path(rec.get('trial_archive') or row.get('transcript') or '/nonexistent')
    out = {'id': f'{pi_run.name}/{row["key"]}', 'task': row['item_id'], 'sessions': [], 'first_request': None,
           'reward': None, 'archive': str(archive), 'world': None, 'calls': [], 'why': None, 'logs': {}}
    if row['status'] != 'ok':
        out['why'] = f'infra: {row.get("infrastructure_reason")}'
        return out
    if not trial or not archive.is_file():
        out['why'] = 'no trial archive'
        return out
    from aa_evals.common.pi_eval.finish_map import member_kind
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            n = m.name
            if member_kind(n) and member_kind(n) not in out['logs']:   # #375 sftnotice: the notice map
                out['logs'][member_kind(n)] = tf.extractfile(m).read().decode('utf-8', 'surrogateescape')
            elif n.startswith(f'{trial}/agent/pi/sessions/') and n.endswith('.jsonl'):
                out['sessions'].append({'path': n[len(trial) + 1:],
                                        'text': tf.extractfile(m).read().decode('utf-8', 'surrogateescape')})
            elif '/first-request/' in n and n.endswith('.json') and out['first_request'] is None:
                try:
                    b = json.loads(tf.extractfile(m).read())
                except ValueError:
                    b = None
                if isinstance(b, dict) and b.get('messages') and b.get('tools'):
                    out['first_request'] = b
            elif n.endswith('/wire.jsonl') and out['first_request'] is None:
                for line in tf.extractfile(m):
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    b = e.get('body') if e.get('t') == 'req' else None
                    if isinstance(b, dict) and b.get('messages') and b.get('tools'):
                        out['first_request'] = b
                        break
            elif n == f'{trial}/artifacts/state/world.json':
                out['world'] = tf.extractfile(m).read().decode()
            elif n == f'{trial}/artifacts/state/calls.jsonl':
                out['calls'] = [json.loads(x) for x in tf.extractfile(m).read().decode().splitlines() if x.strip()]
    out['sessions'].sort(key=lambda s: s['path'])
    from aa_evals.common.pi_eval import finish_map
    fn, window = finish_map.run_settings(str(pi_run))
    served = row.get('max_prompt_tokens') or (rec.get('stats') or {}).get('max_prompt_tokens')
    out['finish_notices'], problem = finish_map.attempt_notices(fn, window, out['logs'].get('notices'), served)
    if problem:
        out['why'] = problem
    return out


def collect_once(runs: list[Path], root: Path, prune: bool) -> int:
    from aa_evals.common.pi_eval import port
    adir = root / 'attempts'
    adir.mkdir(parents=True, exist_ok=True)
    idx_p = adir / 'index.json'
    index = json.loads(idx_p.read_text()) if idx_p.exists() else {'shards': {}, 'attempts': {}}
    todo = []
    for run in runs:
        for row in port.resolved(run):
            aid = f'{run.name}/{row["key"]}'
            if aid not in index['attempts']:
                todo.append((run, row))
    n_new = 0
    for i in range(0, len(todo), SHARD_SIZE):
        chunk = todo[i:i + SHARD_SIZE]
        if len(chunk) < SHARD_SIZE and i + SHARD_SIZE > len(todo) and not getattr(collect_once, 'final', False):
            # a partial shard only when the runs are done (final pass) or the chunk is old enough
            if all(time.time() - os.path.getmtime(r / 'attempts' / w['key'] / 'outcome.json') < 1800
                   for r, w in chunk if (r / 'attempts' / w['key'] / 'outcome.json').exists()):
                break
        number = str(max(map(int, index['shards']), default=-1) + 1)
        name = f'shard_{int(number):05d}.jsonl.gz'
        lines, archives = [], []
        for run, row in chunk:
            a = read_attempt(run, row)
            usable = bool(a['sessions'] and a['first_request'] and row['status'] == 'ok'
                          and not str(a['why'] or '').startswith('finish_notice'))
            s = port.stats(row)
            lines.append({'id': a['id'], 'task': a['task'],
                          'record': {'usable': usable, 'finalized': True, 'task': a['task']},
                          'sessions': a['sessions'], 'first_request': a['first_request'], 'reward': None,
                          **({'finish_notices': a['finish_notices']} if a.get('finish_notices') is not None else {}),
                          'm5': {'run': run.name, 'key': row['key'], 'status': row['status'], 'end': port.end_of(row) if row['status'] == 'ok' else 'infra',
                                 'stats': s, 'why': a['why'], 'n_tool_calls': len(a['calls']),
                                 'n_tool_errors': sum(not c.get('ok', True) for c in a['calls']),
                                 'calls': [{k: c.get(k) for k in ('i', 'tool', 'args', 'ok', 'result_chars')} for c in a['calls']],
                                 'final_world': a['world']}})
            if usable:
                archives.append(a['archive'])
        tmp = adir / (name + '.partial')
        with gzip.open(tmp, 'wt') as f:
            for x in lines:
                f.write(json.dumps(x) + '\n')
        os.replace(tmp, adir / name)
        index['shards'][number] = {'lines': name, 'n': len(lines), 'at': time.time()}
        for x in lines:
            index['attempts'][x['id']] = [int(number)]
        atomic_json(idx_p, index)
        n_new += len(lines)
        print(time.strftime('%H:%M:%S'), f'{name}: {len(lines)} attempts ({sum(x["record"]["usable"] for x in lines)} usable)', flush=True)
        if prune:
            for p in archives:
                try:
                    os.remove(p)
                except OSError:
                    pass
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='glob of pi_eval run dirs')
    ap.add_argument('--out', required=True)
    ap.add_argument('--follow', action='store_true')
    ap.add_argument('--final', action='store_true', help='write a last partial shard')
    ap.add_argument('--prune', action='store_true')
    ap.add_argument('--interval', type=int, default=300)
    a = ap.parse_args()
    collect_once.final = a.final
    while True:
        runs = [Path(p) for p in sorted(glob.glob(a.runs)) if (Path(p) / 'config.json').exists()]
        from aa_evals.common.pi_eval import finish_map
        for r in runs:   # a run with the non-sticky (v3) Pi finish notice is refused (#375 sftnotice)
            finish_map.run_settings(str(r))
        n = collect_once(runs, Path(a.out), a.prune)
        print(time.strftime('%H:%M:%S'), f'collected {n} new attempts from {len(runs)} runs', flush=True)
        if not a.follow:
            return
        if Path(a.out, 'STOP').exists():
            collect_once.final = True
            collect_once(runs, Path(a.out), a.prune)
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
