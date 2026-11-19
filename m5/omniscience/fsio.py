"""Verified writes for /data while the byte quota is enforced (EDQUOT/EIO, 0-byte or truncated files).

append_line: append one line, fsync, check the file grew by exactly that many bytes (truncate back and retry if not).
write_atomic: write a temp file, fsync, read back and compare, then rename over the target.
Both retry with backoff on OSError and never give up while `forever` is set (jobs keep running through quota trouble).
"""
import hashlib
import os
import random
import sys
import time
from pathlib import Path


def _sleep(attempt):
    time.sleep(min(300, 2 ** min(attempt, 8)) * (0.5 + random.random()))


def _warn(msg):
    print(time.strftime('%m-%d %H:%M:%S'), 'fsio:', msg, file=sys.stderr, flush=True)


def append_line(path, line, forever=True, tries=12):
    path = Path(path)
    data = (line if line.endswith('\n') else line + '\n').encode()
    attempt = 0
    while True:
        before = path.stat().st_size if path.exists() else 0
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                n = os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            after = path.stat().st_size
            if n == len(data) and after == before + len(data):
                return
            _warn(f'short append to {path}: wrote {n}/{len(data)}, size {before}->{after}; rolling back')
        except OSError as e:
            _warn(f'append to {path} failed: {e!r}')
        try:
            if path.exists() and path.stat().st_size > before:
                os.truncate(path, before)
        except OSError as e:
            _warn(f'rollback of {path} failed: {e!r}')
        attempt += 1
        if not forever and attempt >= tries:
            raise OSError(f'append to {path} failed {attempt} times')
        _sleep(attempt)


def write_atomic(path, data, forever=True, tries=12):
    path = Path(path)
    if isinstance(data, str):
        data = data.encode()
    want = hashlib.sha256(data).hexdigest()
    attempt = 0
    while True:
        tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            if tmp.stat().st_size == len(data) and hashlib.sha256(tmp.read_bytes()).hexdigest() == want:
                os.replace(tmp, path)
                return
            _warn(f'verify of {tmp} failed (size {tmp.stat().st_size} vs {len(data)})')
        except OSError as e:
            _warn(f'write of {path} failed: {e!r}')
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        attempt += 1
        if not forever and attempt >= tries:
            raise OSError(f'write of {path} failed {attempt} times')
        _sleep(attempt)


def verify_file(path, sha256=None):
    """-> (size, sha256) of a written file; raises if empty or the digest differs."""
    b = Path(path).read_bytes()
    if not b:
        raise OSError(f'{path} is empty')
    h = hashlib.sha256(b).hexdigest()
    if sha256 and h != sha256:
        raise OSError(f'{path} digest {h} != {sha256}')
    return len(b), h
