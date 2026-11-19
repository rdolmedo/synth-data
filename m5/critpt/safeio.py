"""Verified writes on /data while its byte quota is enforced (M5 storage alert, 2026-11-12): writes can fail with
EDQUOT/EIO or leave 0-byte or truncated files. Every write goes to a temp file, is fsynced, read back and checked
(size, sha256, optional parse), then renamed into place; appends are checked by the file's size and rolled back on a
short write. OSErrors are retried with backoff; the last one is raised.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

TRIES = 8


def _backoff(i: int):
    time.sleep(min(60, 2 ** i))


def write_verified(path, data: bytes, check=None, tries: int = TRIES) -> str:
    """Atomically write `data` to `path`; `check(path_of_temp)` may raise to reject the written copy. -> sha256."""
    path = Path(path)
    want = hashlib.sha256(data).hexdigest()
    tmp = path.with_name(f'.{path.name}.tmp{os.getpid()}')
    for i in range(tries):
        try:
            with open(tmp, 'wb') as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            got = tmp.read_bytes()
            if len(got) != len(data) or hashlib.sha256(got).hexdigest() != want:
                raise OSError(f'read-back mismatch on {tmp} ({len(got)} of {len(data)} bytes)')
            if check:
                check(tmp)
            os.replace(tmp, path)
            return want
        except (OSError, ValueError, EOFError) as e:
            if i == tries - 1:
                raise
            print(f'write_verified {path}: {e!r}; retry {i + 1}', flush=True)
            _backoff(i)
    raise AssertionError


def write_json(path, obj, indent=None) -> str:
    return write_verified(path, (json.dumps(obj, indent=indent, ensure_ascii=False) + '\n').encode(),
                          check=lambda p: json.loads(Path(p).read_text()))


def check_gzip_jsonl(p):
    with gzip.open(p, 'rt', encoding='utf-8', errors='surrogateescape') as g:
        for line in g:
            json.loads(line)


def append_line(path, line: str, tries: int = TRIES):
    """Append one line (newline added); verify the file grew by exactly its size, else truncate back and retry."""
    path = Path(path)
    data = (line.rstrip('\n') + '\n').encode()
    for i in range(tries):
        try:
            with open(path, 'ab') as f:
                start = f.tell()
                try:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                    if os.fstat(f.fileno()).st_size != start + len(data):
                        raise OSError(f'short append on {path}')
                except OSError:
                    f.truncate(start)
                    raise
            return
        except OSError as e:
            if i == tries - 1:
                raise
            print(f'append_line {path}: {e!r}; retry {i + 1}', flush=True)
            _backoff(i)
