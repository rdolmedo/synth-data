"""/data writes that survive the enforced quota (EDQUOT / EIO / short writes, 11-12): verify and retry.

append_line: appends one line, fsyncs, checks the file grew by exactly that many bytes; on a short or failed write it
truncates back to the old size and retries with backoff, so a reader never sees a torn line followed by a good one.
write_atomic: temp file, verify size (and an optional check), then rename.
"""
import errno
import os
import random
import time
from pathlib import Path

RETRY_ERRNOS = {errno.EDQUOT, errno.EIO, errno.ENOSPC, errno.EAGAIN}


def _sleep(i):
    time.sleep(min(300, 2 ** min(i, 8)) * (0.5 + random.random()))


def append_line(path, line, attempts=200):
    data = (line if line.endswith('\n') else line + '\n').encode()
    for i in range(attempts):
        before = None
        try:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                before = os.fstat(fd).st_size
                n = os.write(fd, data)
                os.fsync(fd)
                after = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if n == len(data) and after == before + len(data):
                return
            err = f'short write {n}/{len(data)}, size {before}->{after}'
        except OSError as e:
            if e.errno not in RETRY_ERRNOS:
                raise
            err = str(e)
        if before is not None:   # undo a partial line (single writer per file)
            try:
                os.truncate(path, before)
            except OSError:
                pass
        print(f'append_line {path}: {err}; retry {i + 1}', flush=True)
        _sleep(i)
    raise OSError(errno.EIO, f'append_line failed after {attempts} attempts: {path}')


def write_atomic(path, text, check=None, attempts=100):
    path = Path(path)
    data = text.encode() if isinstance(text, str) else text
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    for i in range(attempts):
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if tmp.stat().st_size == len(data) and (check is None or check(tmp)):
                os.replace(tmp, path)
                return
            err = 'verify failed'
        except OSError as e:
            if e.errno not in RETRY_ERRNOS:
                raise
            err = str(e)
        print(f'write_atomic {path}: {err}; retry {i + 1}', flush=True)
        _sleep(i)
    raise OSError(errno.EIO, f'write_atomic failed after {attempts} attempts: {path}')


def verify_jsonl(path, n_expected=None):
    import json
    n = 0
    with open(path) as f:
        for line in f:
            if not line.endswith('\n'):
                return False
            json.loads(line)
            n += 1
    return n_expected is None or n == n_expected
