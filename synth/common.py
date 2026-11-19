"""Durable control-plane I/O shared by every campaign process.

Lustre has no kernel file locks, can return transient EIO/ESTALE, and can run
out of inodes. Durable state uses tmp+fsync+rename with bounded retries;
telemetry uses best_effort() so a full or flaky filesystem never takes down a
serving engine, a controller, or a rollout.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/opt/aai/venvs/harbor/bin/python'
PROXY_ENV = {'http_proxy': 'http://proxy.internal:8080', 'https_proxy': 'http://proxy.internal:8080',
             'HTTP_PROXY': 'http://proxy.internal:8080', 'HTTPS_PROXY': 'http://proxy.internal:8080',
             'no_proxy': 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8',
             'NO_PROXY': 'localhost,127.0.0.1,::1,localnet,.internal.cluster.internal,10.0.0.0/8'}


def swap_path(text, old, new):
    """`text` with the path `old` replaced by `new` where it is a whole path or a prefix of one: starting the text or
    after : = or whitespace, and followed by / : or the end. Never inside a longer name (/mnt/data/..., vllm-x-other)."""
    return re.sub(r'(?<![^:=\s])' + re.escape(str(old)) + r'(?=[/:]|$)', lambda _: str(new), str(text))


def atomic(path, value, retries=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    for attempt in range(retries):
        # A fresh name per attempt is a fresh Lustre object, so a retry is not
        # pinned to the storage target that just failed.
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.{attempt}.tmp')
        try:
            with tmp.open('w') as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            return
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt == retries - 1:
                raise
            time.sleep(min(2 ** attempt, 10))


def best_effort(fn, *args, **kwargs):
    """Telemetry and heartbeats must never kill the process that writes them."""
    try:
        return fn(*args, **kwargs)
    except OSError as exc:
        print(f'best-effort write failed: {exc!r}', flush=True)
        return None


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(value, separators=(',', ':'), ensure_ascii=False) + '\n')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def harbor_result(folder):
    """An attempt's Harbor trial result.json, or None. A trial cut mid-write (TrainingClosed kills Harbor) leaves it
    truncated at a page boundary (seven 4096-byte files at the 2026-10-24 TMax hero close), which must not stop the
    controller's harvest or the exporter."""
    paths = sorted(Path(folder).glob('harbor/*__*/result.json'))
    if len(paths) != 1:
        return None
    try:
        return read(paths[0])
    except (OSError, ValueError):
        return None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed(campaign, kind, task, attempt):
    """Independent attempts; infrastructure retries reuse their attempt's seed."""
    digest = hashlib.sha256(f'{campaign}/{kind}/{task}/{attempt}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % (2 ** 31)


def read_task_list(path):
    lines = Path(path).read_text().split('\n')
    return [s.strip() for s in lines if s.strip() and not s.lstrip().startswith('#')]


def pi_log_paths(root, leaf, attempts=False):
    """Find Harbor's trial and milestone logs without walking task artifacts."""
    prefix = '*/' if attempts else ''
    return sorted({path for agent in ('agent', 'steps/*/agent')
        for path in Path(root).glob(f'{prefix}harbor/*__*/{agent}/pi/{leaf}')})


class Log:
    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, message):
        line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + message
        print(line, flush=True)
        best_effort(self._append, line)

    def many(self, messages):
        """Several lines in one append (each call is an open/append/close on Lustre)."""
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        lines = [f'{stamp} {m}' for m in messages]
        if lines:
            print('\n'.join(lines), flush=True)
            best_effort(self._append, '\n'.join(lines))

    def _append(self, line):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as f:
            f.write(line + '\n')


class MkdirLease:
    """Singleton ownership on Lustre (no flock): mkdir + owner record."""

    def __init__(self, path, owner, alive):
        self.path, self.owner, self.alive = Path(path), owner, alive

    def acquire(self, attempts=24, delay=5):
        for _ in range(attempts):
            try:
                self.path.mkdir(parents=True)
                atomic(self.path / 'owner.json', {'job_id': self.owner, 'created_at': time.time()})
                return self
            except FileExistsError:
                holder = read(self.path / 'owner.json', {}).get('job_id')
                if holder and not self.alive(holder):
                    (self.path / 'owner.json').unlink(missing_ok=True)
                    self.path.rmdir()
                else:
                    time.sleep(delay)
        raise RuntimeError(f'Another live owner holds {self.path}')

    def release(self):
        if read(self.path / 'owner.json', {}).get('job_id') == self.owner:
            (self.path / 'owner.json').unlink()
            self.path.rmdir()
