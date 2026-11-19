"""Shared paths and helpers for the M5 SciCode pipeline."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

DATA = Path(os.environ.get('M5_SCICODE_DATA', '/data/aai/m5/scicode'))
ALLOC = Path('/opt/aai/m5-coord/alloc.json')
SERVING = Path('/data/aai/tbench/tasksynth/serving')
VENV_PY = os.environ.get('M5_SCICODE_PY', '/data/aai/venvs/m5-scicode/bin/python')
REPO = Path(__file__).resolve().parents[2]
BENCH = 'scicode'
CPU_BID = 36
BAD_NODES = ['cpu203', 'cpu205', 'cpu207', 'cpu208', 'gpu110', 'gpu132', 'gpu123', 'gpu143', 'gpu166', 'gpu172', 'gpu106', 'cpu305']
TEST_PROBLEMS = Path('/data/aai/aa_evals/data/scicode/verified_aa_v3/problems_test.jsonl')


EXCLUDED_FILE = Path('/opt/aai/m5-coord/excluded-nodes.txt')


# jobs import tb_agents lazily (aa_evals/{scicode,gdppdf}/run.py: the response budget, #446/#449; #451 test)
FROZEN_DIRS = ('m5', 'aa_evals', 'tb_agents')

def excluded_nodes() -> list[str]:
    """BAD_NODES plus the coordinator's shared exclusion list, read at submit time."""
    extra = []
    try:
        extra = [l.split('#')[0].strip() for l in EXCLUDED_FILE.read_text().splitlines()]
    except OSError:
        pass
    return sorted(set(BAD_NODES) | {n for n in extra if n})


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass   # torn last line from a killed job
    return out


RETRY_WAITS = (2, 5, 15, 30, 60, 120, 300, 600)   # /data quota enforcement: EDQUOT/EIO and short writes come and go


def _retry(fn, what):
    for k, w in enumerate(RETRY_WAITS + (None,)):
        try:
            return fn()
        except OSError as e:
            if w is None:
                raise
            print(f'write retry {k + 1} ({what}): {e!r}; sleeping {w}s', flush=True)
            time.sleep(w)


def append_line(path: Path, line: str):
    """Append one line and verify it landed (size grew by exactly its bytes, fsynced); on a failed or short write,
    truncate back to the old size and retry, so a torn line never swallows the next row."""
    path, data = Path(path), (line.rstrip('\n') + '\n').encode()

    def go():
        with open(path, 'ab') as f:
            n0 = f.tell()
            try:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                if os.fstat(f.fileno()).st_size != n0 + len(data):
                    raise OSError(f'short append to {path}')
            except OSError:
                try:
                    f.truncate(n0)
                except OSError:
                    pass
                raise
    _retry(go, path)


def safe_write(path: Path, data: str | bytes):
    """Write a whole file via temp file + fsync + size check + rename, with retries."""
    path = Path(path)
    b = data.encode() if isinstance(data, str) else data

    def go():
        tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
        with open(tmp, 'wb') as f:
            f.write(b)
            f.flush()
            os.fsync(f.fileno())
        if tmp.stat().st_size != len(b) or tmp.read_bytes() != b:
            tmp.unlink(missing_ok=True)
            raise OSError(f'verify failed for {tmp}')
        os.replace(tmp, path)
    _retry(go, path)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()

    async def write(self, row: dict):
        async with self.lock:
            line = json.dumps(row, ensure_ascii=False)
            await asyncio.to_thread(append_line, self.path, line)


class Alloc:
    """Concurrency caps from the coordinator's alloc.json, re-read at most every 60 s (last good value on errors)."""

    def __init__(self):
        self.t, self.d = 0.0, {}

    def get(self) -> dict:
        if time.time() - self.t > 60:
            try:
                self.d = json.loads(ALLOC.read_text())
            except (OSError, ValueError):
                pass
            self.t = time.time()
        return self.d

    def ds_cap(self) -> int:
        cap = int(((self.get().get('ds') or {}).get('max_inflight') or {}).get(BENCH, 0))
        return int(cap * float(os.environ.get('M5_DS_CAP_FRAC', '1')))   # jobs sharing the bench cap split it

    def super_engines(self) -> list[str]:
        return list((self.get().get('super') or {}).get('serving') or [])

    def super_cap(self) -> int:
        return int(((self.get().get('super') or {}).get('max_inflight_per_engine') or {}).get(BENCH, 0))


def git_sha() -> str:
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def frozen_checkout() -> Path:
    """A frozen copy of HEAD (git archive) under DATA/code/<sha>, for jobs (jobs on shared storage read scripts live;
    home holds only the worktree)."""
    dirty = subprocess.run(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit('commit first: jobs run from a frozen copy of HEAD')
    sha = git_sha()
    path = DATA / 'code' / sha[:12]
    if not path.exists():
        tmp = path.with_name(path.name + '.partial')
        tmp.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(['git', '-C', str(REPO), 'archive', sha, *FROZEN_DIRS], capture_output=True, check=True)
        subprocess.run(['tar', '-x', '-C', str(tmp)], input=arc.stdout, check=True)
        tmp.rename(path)
    return path


def submit(name: str, module_argv: list[str], *, cpus=2, mem_gb=8, disk_gb=10, log_dir: Path | None = None,
           env: dict | None = None) -> str:
    """Submit `python -m <module_argv>` as a CPU condor job from a frozen checkout of HEAD."""
    code = frozen_checkout()
    log_dir = Path(log_dir or DATA / 'logs' / 'jobs')
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%S')
    req = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in excluded_nodes())
    args = ' '.join(module_argv)
    extra = ''.join(f' {k}={v}' for k, v in (env or {}).items())
    sub = f'''universe = vanilla
executable = {VENV_PY}
arguments = "-u -m {args}"
initialdir = {code}
output = {log_dir}/{name}.{stamp}.$(Cluster).out
error = {log_dir}/{name}.{stamp}.$(Cluster).err
log = {log_dir}/condor.log
request_gpus = 0
request_cpus = {cpus}
request_memory = {mem_gb * 1024}
request_disk = {disk_gb}G
+BypassLXCfs = "true"
+JobBatchName = "m5-scicode-{name}"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONUNBUFFERED=1 PYTHONPATH={code}{extra}"
requirements = {req} && (TARGET.GPUs == 0 || TARGET.GPUs =?= undefined)
queue
'''
    path = log_dir / f'{name}.{stamp}.sub'
    path.write_text(sub)
    r = subprocess.run(['condor_submit_bid', str(CPU_BID), str(path)], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    return r.stdout
