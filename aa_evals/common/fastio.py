"""Write files to /data without filling the job's memory cgroup with dirty page cache (#378).

Buffered writes to Lustre leave dirty pages that count against the condor job's memory limit until written back.
/data's writeback is often slow (20 small buffered writes took 3.5 min on gpu103, 2026-11-13, vs ~10 ms each with
O_DIRECT; the cause is not established), dirty pages pile up, and the job is held "over memory limit" with a tiny RSS:
c30n's GDPval workers and collectors; c30's mass hold at 23:37 UTC on 11-11 came while /data refused writes (EDQUOT).
Raising RequestMemory only delays such holds. So bulk outputs are built on node-local disk and copied
with O_DIRECT (`dd oflag=direct`; GNU dd drops O_DIRECT for a short final block), which bypasses the page cache.

  copy_direct(src, dst)       # dst written with O_DIRECT; fallback: buffered copy + fsync + drop its cached pages
  copy_tree_direct(src, dst)  # a directory, file by file with copy_direct (dst/ must not exist; built as dst.partial)
  append_line(path, text)     # JSONL appends: append, fsync, drop the file's cached pages (bounded dirty pages)
  trim_torn_tail(path)        # a single writer restarting: cut a last line left without its newline by a kill
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def drop_cache(path) -> None:
    """fsync `path` and drop its pages from the page cache (best effort)."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except (OSError, AttributeError):
        pass


def copy_direct(src, dst, runner=subprocess.run) -> str:
    """Copy src to dst bypassing the page cache; returns the method used ('direct' or 'buffered'). The size of dst is
    checked against src; any failure raises OSError (dst may then be partial: write to a temporary name and rename)."""
    src, dst = Path(src), Path(dst)
    size = src.stat().st_size
    method = 'direct'
    try:
        r = runner(['dd', f'if={src}', f'of={dst}', 'bs=16M', 'oflag=direct', 'status=none'], capture_output=True,
                   text=True, timeout=3600)
        ok = r.returncode == 0 and dst.exists() and dst.stat().st_size == size
    except (OSError, subprocess.SubprocessError):
        ok = False
    if not ok:   # e.g. a filesystem without O_DIRECT (tmpfs): buffered, then flushed and dropped from the cache
        method = 'buffered'
        shutil.copyfile(src, dst)
        drop_cache(dst)
    if dst.stat().st_size != size:
        raise OSError(f'copy_direct: {dst} has {dst.stat().st_size} bytes, expected {size}')
    return method


def copy_tree_direct(src, dst, runner=subprocess.run) -> int:
    """Copy directory `src` to `dst` (which must not exist), every file with copy_direct. Built as `<dst>.partial`
    (removed first if a killed copy left one) and renamed, so `dst` appears complete or not at all. Returns the
    number of files."""
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    tmp = dst.with_name(dst.name + '.partial')
    shutil.rmtree(tmp, ignore_errors=True)
    n = 0
    for d, dirs, files in os.walk(src):
        rel = Path(d).relative_to(src)
        (tmp / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            p = Path(d) / f
            if p.is_symlink() or not p.is_file():
                continue
            copy_direct(p, tmp / rel / f, runner=runner)
            n += 1
    tmp.mkdir(parents=True, exist_ok=True)   # an empty src
    os.rename(tmp, dst)
    return n


def append_line(path, text: str) -> None:
    """Append one record (a newline is added if missing), then fsync and drop the file's cached pages: a collector
    appending large records (Pi sessions) to /data keeps at most one record's dirty pages in its memory cgroup.
    The fsync waits for /data's writeback; that slows the writer, it never grows its memory."""
    data = (text if text.endswith('\n') else text + '\n').encode('utf-8')
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (OSError, AttributeError):
            pass
    finally:
        os.close(fd)


def trim_torn_tail(path) -> int:
    """Cut a last line that has no newline (a record whose append was killed), so the next append does not glue onto
    it. Only for a file's single writer at its start (a concurrent writer's record in progress looks the same).
    Returns the number of bytes removed."""
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return 0
    if not size:
        return 0
    with open(p, 'rb+') as f:
        f.seek(size - 1)
        if f.read(1) == b'\n':
            return 0
        pos, chunk = size, 1 << 16
        while pos > 0:               # find the last newline, reading backwards
            start = max(0, pos - chunk)
            f.seek(start)
            i = f.read(pos - start).rfind(b'\n')
            if i >= 0:
                keep = start + i + 1
                break
            pos = start
        else:
            keep = 0
        f.truncate(keep)
    return size - keep
