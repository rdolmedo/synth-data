"""Six-hour task bundle for the Pi protocol from any tasks tar or directory (tasks/<slug>/ or <slug>/).

gen/pi_cont16k.py runs only tasks whose [agent] timeout_sec is 21600 (the Pi campaigns' bundle family). This rewrites
that one value and nothing else (checked by re-parsing every task.toml), and records each task's original timeout in
the manifest. A shorter cap for a benchmark is applied by the environment clock instead (run.py --agent-cap-sec:
PI_ENVCLOCK_MULT = cap / 21600), so the Harbor files stay in the protocol's form.

  python -m aa_evals.common.pi_eval.bundle SRC OUT.tar
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
import tarfile
import tomllib
from pathlib import Path

SIX_HOURS = 21600
AGENT_TIMEOUT = re.compile(r'(?ms)^(\[agent\][^\[]*?^timeout_sec\s*=\s*)([0-9.eE+-]+)')


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _members(src: Path):
    """(relative path under tasks/, TarInfo-or-Path, bytes|None) for every entry of the source."""
    if src.is_dir():
        root = src / 'tasks' if (src / 'tasks').is_dir() else src
        for p in sorted(root.rglob('*')):
            yield p.relative_to(root).as_posix(), p, (p.read_bytes() if p.is_file() and not p.is_symlink() else None)
        return
    with tarfile.open(src) as tf:
        names = [m.name for m in tf.getmembers()]
        prefix = 'tasks/' if any(n.startswith('tasks/') for n in names) else ''
        for m in tf.getmembers():
            if not m.name.startswith(prefix) or m.name.rstrip('/') == prefix.rstrip('/'):
                continue
            rel = m.name[len(prefix):]
            yield rel, m, (tf.extractfile(m).read() if m.isfile() else None)


def six_hour_toml(text: str) -> tuple[str, float | None]:
    original = tomllib.loads(text).get('agent', {}).get('timeout_sec')
    if original == SIX_HOURS:
        return text, original
    if AGENT_TIMEOUT.search(text):
        new = AGENT_TIMEOUT.sub(lambda m: f'{m.group(1)}{float(SIX_HOURS)}', text, count=1)
    else:   # no [agent] timeout_sec: add it
        new = text + f'\n[agent]\ntimeout_sec = {float(SIX_HOURS)}\n' if '[agent]' not in text else \
            re.sub(r'(?m)^\[agent\]\s*$', f'[agent]\ntimeout_sec = {float(SIX_HOURS)}', text, count=1)
    a, b = tomllib.loads(new), tomllib.loads(text)
    assert a['agent']['timeout_sec'] == SIX_HOURS, 'rewrite failed'
    b.setdefault('agent', {})['timeout_sec'] = SIX_HOURS
    assert a == b, 'task.toml differs beyond [agent] timeout_sec'
    return new, original


def build(src, out) -> dict:
    src, out = Path(src), Path(out)
    tmp = out.with_suffix('.partial')
    tasks: dict[str, dict] = {}
    with tarfile.open(tmp, 'w') as dst:
        for rel, ref, data in _members(src):
            parts = rel.split('/')
            if data is not None and len(parts) == 2 and parts[1] == 'task.toml':
                text, original = six_hour_toml(data.decode())
                data = text.encode()
                cfg = tomllib.loads(text)
                tasks[parts[0]] = {'original_agent_timeout_sec': original,
                                   'verifier_timeout_sec': cfg.get('verifier', {}).get('timeout_sec'),
                                   'environment': cfg.get('environment', {})}
            if isinstance(ref, tarfile.TarInfo):
                info = ref
                info.name = 'tasks/' + rel
                if data is not None:
                    info.size = len(data)
                dst.addfile(info, io.BytesIO(data) if data is not None else None)
            else:
                info = dst.gettarinfo(str(ref), arcname='tasks/' + rel)
                if data is not None:
                    info.size = len(data)
                dst.addfile(info, io.BytesIO(data) if data is not None else None)
    tmp.replace(out)
    manifest = {'source': str(src), 'source_sha256': sha256(src) if src.is_file() else None, 'sha256': sha256(out),
                'task_count': len(tasks), 'policy': f'only [agent] timeout_sec rewritten to {SIX_HOURS}',
                'tasks': tasks}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


if __name__ == '__main__':
    m = build(sys.argv[1], sys.argv[2])
    print(sys.argv[2], m['sha256'], m['task_count'], 'tasks')
