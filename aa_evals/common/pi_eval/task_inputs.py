"""Per-task input files staged into the agent's sandbox before the agent starts (#348).

Benchmarks ported to the Pi harness (GDPval, AutomationBench) run every task on one shared image (`docker_image` in
task.toml), so harbor_ns builds and caches one rootfs instead of one multi-GB image per task. What differs between
tasks (GDPval's reference files, AutomationBench's initial app state and CLI) ships in the task directory instead:

  <task>/inputs/<absolute path in the sandbox>     files to place, e.g. inputs/home/user/report.xlsx
  <task>/inputs.json                               {"version": 1, "files": {rel: sha256}, "dirs": [rel, ...]}

`NsTaskInputsEnvironment` (`pi_eval.run --env-class aa_evals.common.pi_eval.task_inputs_env:NsTaskInputsEnvironment`)
is `NsSidecarEnvironment` plus this staging step after the sandbox starts and before Harbor installs the agent. It
checks every file against inputs.json (an inventory change or a link is an error, so an attempt never runs on
other inputs than the bundle recorded), creates the listed directories (e.g. an empty output directory), copies the
files in, and writes a receipt (`task_inputs.json`) into the trial directory. A task without inputs.json is staged
as before (nothing added), so the class is a no-op for TB-style tasks. A separate verifier environment gets
nothing. Staging failures raise, and pi_eval retries them as infrastructure (environment start).
"""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

VERSION = 1
INPUTS = 'inputs'
MANIFEST = 'inputs.json'
RECEIPT = 'task_inputs.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _rel(value: str) -> str:
    p = PurePosixPath(value)
    if (not isinstance(value, str) or not value or p.is_absolute() or '..' in p.parts or str(p) != value
            or '\\' in value):
        raise ValueError(f'unsafe task input path {value!r}')
    return value


def inventory(root: Path) -> tuple[dict[str, str], list[str]]:
    """({file: sha256}, [dirs]) of a task's inputs/ tree; links and special files are refused."""
    files, dirs = {}, []
    if not root.exists():
        return files, dirs
    for p in sorted(root.rglob('*')):
        mode = p.lstat().st_mode
        rel = p.relative_to(root).as_posix()
        if stat.S_ISDIR(mode):
            dirs.append(rel)
        elif stat.S_ISREG(mode):
            files[rel] = sha256(p)
        else:
            raise ValueError(f'task inputs contain a link or special file: {rel}')
    return files, dirs


def write_manifest(task_dir: Path, extra: dict | None = None) -> dict:
    """Record inputs/ of a task directory being built (bundle builders call this last)."""
    files, dirs = inventory(task_dir / INPUTS)
    m = {'version': VERSION, 'files': files, 'dirs': dirs, **(extra or {})}
    (task_dir / MANIFEST).write_text(json.dumps(m, indent=1, sort_keys=True) + '\n')
    return m


def load(task_dir: Path) -> dict | None:
    """The checked manifest of a task directory, or None when the task has no staged inputs."""
    mf = task_dir / MANIFEST
    if not mf.exists():
        return None
    m = json.loads(mf.read_text())
    if m.get('version') != VERSION or not isinstance(m.get('files'), dict) or not isinstance(m.get('dirs'), list):
        raise ValueError(f'{mf}: unknown task inputs manifest')
    for r in [*m['files'], *m['dirs']]:
        _rel(r)
    files, dirs = inventory(task_dir / INPUTS)
    if files != m['files'] or sorted(dirs) != sorted(m['dirs']):
        raise ValueError(f'{task_dir.name}: task inputs differ from {MANIFEST}')
    return m
