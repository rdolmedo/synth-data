"""Synthetic tasks as a Pi bundle, built by the eval's own code (aa_evals/automationbench/pi/tasks.py build_task):
the same instruction (domain system prompt with the turn-budget sentence replaced, the `ab` how-to and the upstream
tool schemas, the request), the same `ab` CLI, world service, images and compose file. Only the task data differ.

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.bundle ACCEPTED.jsonl OUT.tar VERSION [--ids FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import time
from pathlib import Path


def rows_of(accepted: list[dict]) -> list[dict]:
    """Upstream-shaped rows as aa_evals' builders expect them (prompt, info, example_id, _task_name)."""
    out = []
    for rec in accepted:
        t = rec['task']
        info = dict(t['info'])
        info.setdefault('assertions', [])
        out.append({'example_id': t['example_id'], 'prompt': t['prompt'], 'info': info,
                    '_task_name': info['task_name'], '_domain': info['task_name'].split('.')[0]})
    return out


def build(accepted: list[dict], out: Path, version: str) -> dict:
    from aa_evals.automationbench.pi import tasks as PT
    from aa_evals.common.pi_eval.task_inputs import sha256
    rows = rows_of(accepted)
    names = [r['_task_name'] for r in rows]
    assert len(set(names)) == len(names), 'duplicate task names'
    tools = PT.tool_schemas()
    PT.BUNDLE_VERSION = version   # task.toml metadata only
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') as tmp:
        root = Path(tmp) / 'tasks'
        root.mkdir()
        meta = {r['_task_name']: PT.build_task(r, tools, root) for r in rows}
        part = out.with_suffix('.tar.partial')
        with tarfile.open(part, 'w') as tf:
            for f in sorted(Path(tmp).rglob('*')):
                tf.add(f, arcname=f.relative_to(tmp).as_posix(), recursive=False, filter=PT._norm)
        part.replace(out)
    manifest = {'bundle_version': version, 'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'source': 'm5 synthetic (#359)', 'images': [PT.WORLD_REF, PT.AGENT_REF],
                'scripts': {n: sha256(PT.HERE / 'image' / n) for n in ('world_server.py', 'ab')},
                'tools_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest(),
                'tasks': meta, 'sha256': sha256(out)}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('accepted')
    ap.add_argument('out')
    ap.add_argument('version')
    ap.add_argument('--ids', default=None)
    a = ap.parse_args()
    recs = [json.loads(x) for x in Path(a.accepted).read_text().splitlines() if x.strip()]
    if a.ids:
        want = set(Path(a.ids).read_text().split())
        recs = [r for r in recs if r['task']['info']['task_name'] in want]
    m = build(recs, Path(a.out), a.version)
    print(f'{a.out}: {len(m["tasks"])} tasks, sha256 {m["sha256"]}')


if __name__ == '__main__':
    main()
