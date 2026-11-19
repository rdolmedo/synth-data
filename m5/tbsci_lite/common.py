"""Paths and helpers shared by the TB-Sci-lite synthetic-data pipeline (#353).

Code: this branch (m5/tbsci_lite, aa_evals) plus a frozen checkout of origin/task-synthesis (task_synthesis/, synth/,
its harbor_ns) for the M1 round machinery. Jobs run from frozen detached worktrees of both; PYTHONPATH puts the
task-synthesis checkout first for generation jobs (its harbor_ns), and this branch alone for trace jobs (pi_eval)."""
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKTREES = Path('/opt/aai/worktrees')
ROOT = Path('/data/aai/m5/tbsci_lite')
SHARED_SCIDATA = Path('/data/aai/m5/shared/scidata')
# Frozen checkouts live on /data (home is a small NFS quota; coordinator 2026-11-12). They are created by a short
# CPU job (ensure_worktree), since gpu123's own writes to /data stall.
CODE_ROOT = ROOT / 'code'
TS_SHA = 'e9ce566'                                    # origin/task-synthesis (M1 production code)
TS_CHECKOUT = CODE_ROOT / f'ts-{TS_SHA}'
DS_SHA, SFT_SHA = '60f67da', '0538ea6'                # origin/data-synthesis (decontam), origin/sft (prep)
DS_CHECKOUT = CODE_ROOT / f'ds-{DS_SHA}'
SFT_CHECKOUT = CODE_ROOT / f'sft-{SFT_SHA}'
PY = '/opt/aai/venvs/harbor/bin/python'
LITE_TASKS = Path('/data/aai/aa_evals/data/tbsci_lite/v2.1/tasks.tar')
LITE_MANIFEST = Path('/data/aai/aa_evals/data/tbsci_lite/v2.1/manifest.json')
PROXY_ENV = {'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
             'HTTP_PROXY': 'http://proxy.internal:8080', 'HTTPS_PROXY': 'http://proxy.internal:8080'}


EXCLUDED_NODES_FILE = Path('/opt/aai/m5-coord/excluded-nodes.txt')


def bad_nodes():
    """aa-serving's BAD_NODES plus the coordinator's m5-coord/excluded-nodes.txt (read at submit time)."""
    extra = set()
    try:
        for line in EXCLUDED_NODES_FILE.read_text().splitlines():
            tok = line.split('#', 1)[0].strip()
            extra |= set(tok.replace(',', ' ').split())
    except OSError:
        pass
    return sorted(set(_serving_bad_nodes()) | extra)


def _serving_bad_nodes():
    """aa-serving's BAD_NODES (serve_teacher.py), read from origin/aa-serving."""
    import ast
    src = subprocess.run(['git', '-C', str(REPO), 'show', 'origin/aa-serving:task_synthesis/serve/serve_teacher.py'],
                         capture_output=True, text=True, check=True).stdout
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'BAD_NODES' for t in node.targets):
            return sorted(ast.literal_eval(node.value))
    raise RuntimeError('BAD_NODES not found')


def requirements_excluding(nodes):
    return ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in nodes)


def ensure_worktree(path, sha, timeout=1800):
    """A detached worktree of `sha` at `path` (on /data), made by a CPU job; waits until it is complete."""
    import time
    path = Path(path)
    if (path / '.m5-complete').exists():
        return path
    jobs = ROOT / 'logs' / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    script = jobs / f'worktree-{path.name}.sh'
    script.write_text(f"""#!/bin/bash
set -e
if [ ! -e {path}/.git ]; then git -C {REPO} worktree add --detach {path} {sha}; fi
git -C {path} rev-parse HEAD > {path}/.m5-complete
""")
    script.chmod(0o755)
    sub = jobs / f'worktree-{path.name}.sub'
    sub.write_text(f"""universe = vanilla
executable = {script}
output = {jobs}/worktree-{path.name}.$(Cluster).out
error = {jobs}/worktree-{path.name}.$(Cluster).err
log = {jobs}/condor.log
request_cpus = 1
request_memory = 2048
request_disk = 2G
+JobBatchName = "m5-tbsci-worktree"
getenv = false
environment = "HOME=/home/user"
requirements = {requirements_excluding(bad_nodes())}
queue
""")
    subprocess.run(['condor_submit_bid', '36', str(sub)], capture_output=True, text=True, check=True)
    t0 = time.time()
    while not (path / '.m5-complete').exists():
        if time.time() - t0 > timeout:
            raise SystemExit(f'worktree {path} not ready after {timeout}s (see {jobs}/worktree-{path.name}.*.err)')
        time.sleep(10)
    return path


def frozen_checkout():
    """A detached worktree of this branch's HEAD for jobs (edits here never reach a running job), on /data."""
    sha = subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                         check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit('Commit first: jobs run from a frozen worktree of HEAD')
    return ensure_worktree(CODE_ROOT / f'run-{sha[:10]}', sha)


def job_pythonpath(repo):
    """Generation jobs: the task-synthesis checkout first (task_synthesis, synth, its harbor_ns), then this branch."""
    ensure_worktree(TS_CHECKOUT, TS_SHA)
    return f'{TS_CHECKOUT}:{repo}'


def durable_write(path, data, check=None, tries=10):
    """Write bytes so that a reader only ever sees a complete, verified file (2026-11-12 storage alert: the /data quota
    is enforced and writes fail with EDQUOT/EIO or leave 0-byte or truncated files). Temp file, fsync, read back and
    compare (and check(bytes) if given), then rename; retry with backoff for about 40 min before raising."""
    path = Path(path)
    if isinstance(data, str):
        data = data.encode()
    err = None
    for i in range(tries):
        tmp = path.with_name(f'{path.name}.tmp{os.getpid()}')
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            back = tmp.read_bytes()
            if back != data:
                raise OSError(f'read-back mismatch ({len(back)} of {len(data)} bytes)')
            if check is not None:
                check(back)
            os.replace(tmp, path)
            return
        except (OSError, ValueError) as e:
            err = e
            try:
                tmp.unlink()
            except OSError:
                pass
            time.sleep(min(600, 5 * 2 ** i))
    raise OSError(f'durable_write {path}: {err}')


def read_strict(path, tries=8):
    """Text of a file that must be read correctly (driver state): None if it does not exist; transient errors are
    retried and then raised, never mistaken for a missing file."""
    import json
    err = None
    for i in range(tries):
        try:
            return json.loads(Path(path).read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            err = e
            time.sleep(min(300, 5 * 2 ** i))
    raise OSError(f'read_strict {path}: {err}')
