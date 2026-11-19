"""Export (#359): the collected attempts as a synth/ campaign root that sft_qwen35/prepare.py reads
(`--source pi_native --synth-roots ROOT --arms nemotron35-super`), with exclusions applied in the manifest.

  PYTHONPATH=. venvs/harbor/bin/python -m m5.automationbench.export --traces ROOT [--exclude-tasks F,...] \
      [--drop-no-submit] [--token-target 330e6 --order FILE]

ROOT is collect.py's output (attempts/index.json + shards). This writes ROOT/manifest.json:
{"teacher": "nemotron35-super", "records": [{"id", "task", "usable", "finalized", "end", "flags", ...}]}. Every
attempt stays listed (no-submit and context-limit endings are flagged; infra failures are not usable); `usable` is
false for excluded ones. Excluded: decontamination (task names in --exclude-tasks files), infra failures, attempts
without a session or first request, and with --drop-no-submit the episodes that never finished by themselves
(`end` != finish, i.e. context or time limit). --token-target keeps attempts in --order (default: collection order)
until the target is covered and marks the rest `tail_cut` (not usable).
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path

TEACHER = 'nemotron35-super'


def lines(root: Path):
    idx = json.loads((root / 'attempts' / 'index.json').read_text())
    for n in sorted(idx['shards'], key=int):
        with gzip.open(root / 'attempts' / idx['shards'][n]['lines'], 'rt') as f:
            for line in f:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--exclude-tasks', default='')
    ap.add_argument('--drop-no-submit', action='store_true')
    ap.add_argument('--token-target', type=float, default=0)
    ap.add_argument('--tokens', default=None, help='JSON {attempt id: exported tokens} (tokenizer count)')
    a = ap.parse_args()
    root = Path(a.traces)
    excl = set()
    for f in filter(None, a.exclude_tasks.split(',')):
        excl |= set(Path(f).read_text().split())
    toks = json.loads(Path(a.tokens).read_text()) if a.tokens else {}
    recs, c = [], Counter()
    total = 0
    for x in lines(root):
        m = x['m5']
        flags = []
        if m['end'] != 'finish':
            flags.append(f'end:{m["end"]}')
        usable = bool(x['record']['usable'])
        why = None
        if not usable:
            why = 'no_session_or_infra'
        elif x['task'] in excl:
            usable, why = False, 'decontamination'
        elif a.drop_no_submit and m['end'] != 'finish':
            usable, why = False, 'no_submit'
        n_tok = int(toks.get(x['id']) or (m['stats'] or {}).get('max_context_tokens') or 0)
        if usable and a.token_target and total >= a.token_target:
            usable, why = False, 'tail_cut'
        if usable:
            total += n_tok
        c[why or 'kept'] += 1
        recs.append({'id': x['id'], 'task': x['task'], 'usable': usable, 'finalized': True, 'excluded': why,
                     'end': m['end'], 'flags': flags, 'tokens': n_tok, 'run': m['run']})
    man = {'teacher': TEACHER, 'source': 'm5 AutomationBench synthetic tasks (#359)', 'records': recs,
           'finish_notice_map': 1,   # shard lines may carry finish_notices (#375 sftnotice)
           'counts': dict(c), 'kept_tokens': total}
    tmp = root / f'manifest.json.tmp{os.getpid()}'
    tmp.write_text(json.dumps(man, indent=1))
    os.replace(tmp, root / 'manifest.json')
    print(json.dumps({'counts': dict(c), 'kept_tokens': total}))


if __name__ == '__main__':
    main()
