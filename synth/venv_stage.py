"""Node-local copies of a vLLM venv (serve_teacher tunable `stage_venv`).

A vLLM main engine started from the /data venv spent ~2 min more before serving than the same engine from a
node-local copy: imports of torch / vLLM / FlashInfer from Lustre in every process (API servers, engine core,
workers). A/B on one B200 (2026-11-19, snowball-eval-main, fresh caches per start, job 17980343): healthy after 145 s
(local) / 288 s (/data) / 149 s (local again); launch -> engine init 27 vs 133 s; staging the 17.1 GB venv 21 s.

The copy comes from a tar of the venv next to it, `<venv>.tar`, read as one sequential O_DIRECT stream (the per-file
reads are what is slow on Lustre), with `<venv>.tar.json` recording what it holds. `build` writes both; run it in a
CPU job (it reads every file of the venv once, ~11 min for the vLLM main venv):
  python -m synth.venv_stage build /data/aai/venvs/<venv>
and again after any change to the venv. `stage` refuses a tar whose installed distributions differ from the venv's
(every *.dist-info name and its RECORD: an added, removed or reinstalled wheel), and the server then runs from /data
as before, saying why. A file edited in place inside an installed package is NOT detected (its RECORD stays):
rebuild the tar after such an edit. Same files, so the same numerics: serve_teacher records the engine's numerics
with the source venv's path.

The copy goes to <root>/venvs/<sha256(source path)[:12]>/<name>: a fixed path for a given venv in every job (the
engine-cache key and FlashInfer's JIT build lines embed it), and two venvs of one job never share a copy. FlashInfer's
ninja lines embed the venv path, so the first staged start of a recipe rebuilds its JIT modules once (seeded from the
/data recipe's snapshot, the rest is reused); warm one engine before a wave of staged starts.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


def tar_path(venv) -> Path:
    return Path(f'{venv}.tar')


def meta_path(venv) -> Path:
    return Path(f'{venv}.tar.json')


def dists(venv) -> list:
    """The venv's installed distributions (name-version), as engine_cache.config_key lists them."""
    return sorted(p.name for p in Path(venv).glob('lib/python3*/site-packages/*.dist-info'))


def fingerprint(venv) -> str:
    """sha256 over every *.dist-info name and its RECORD (the installed files' own hashes): a wheel added, removed
    or reinstalled with other files changes it; a file edited in place does not."""
    h = hashlib.sha256()
    for name in dists(venv):
        h.update(name.encode() + b'\0')
        for record in Path(venv).glob(f'lib/python3*/site-packages/{name}/RECORD'):
            h.update(record.read_bytes())
        h.update(b'\0')
    return h.hexdigest()


def relocatable(venv) -> bool:
    """A venv whose scripts find their interpreter relative to themselves (uv: `relocatable = true`)."""
    try:
        return any(line.replace(' ', '').lower() == 'relocatable=true'
                   for line in (Path(venv) / 'pyvenv.cfg').read_text().splitlines())
    except OSError:
        return False


def unusable(venv):
    """Why `venv` cannot be staged as such, or None: a symlink (its tar would hold the link back to the source) or
    not relocatable (absolute shebangs would run the source)."""
    venv = Path(venv)
    if venv.is_symlink():
        return f'{venv} is a symlink: build and stage its real path ({venv.resolve()})'
    if not (venv / 'pyvenv.cfg').is_file():
        return f'{venv}: not a venv (no pyvenv.cfg)'
    if not relocatable(venv):
        return f'{venv}: not relocatable (pyvenv.cfg has no relocatable = true)'
    return None


def copy_path(venv, root) -> Path:
    """Where `venv` is staged under root: one directory per source path, the venv's own name inside."""
    return Path(root) / 'venvs' / hashlib.sha256(str(venv).encode()).hexdigest()[:12] / Path(venv).name


def tar_files(tar) -> int:
    """Members of a tar that are not directories (files, symlinks, hard links): count_files of its extraction."""
    import tarfile
    with tarfile.open(tar) as t:
        return sum(1 for m in t if not m.isdir())


def count_files(root) -> int:
    """Files and symlinks under root (directories not counted), as `find -type f -o -type l`."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):   # symlinked dirs (lib64 -> lib) are listed, not followed
        n += len(filenames) + sum(1 for d in dirnames if os.path.islink(os.path.join(dirpath, d)))
    return n


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _bash(cmd, timeout):
    """Run a bash pipeline in its own process group; on timeout the whole group (dd, tar) is killed, then raises."""
    p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def _run_direct(pipe, flag, check=False, timeout=None):
    """Run `pipe` (bash) with dd's O_DIRECT `flag`; once more without it where the filesystem refuses O_DIRECT
    (tmpfs: EINVAL). O_DIRECT keeps a 17 GB stream out of the page cache (/data dirty pages, a job's memory limit)."""
    r = _bash(pipe.format(flag=flag), timeout)
    if r.returncode and 'Invalid argument' in r.stderr:
        r = _bash(pipe.format(flag=''), timeout)
    if check and r.returncode:
        raise SystemExit(f'failed (rc {r.returncode}): {r.stderr[-500:]}')
    return r


def build(venv) -> dict:
    """Write <venv>.tar (O_DIRECT: no dirty pages on /data) and <venv>.tar.json. Replaces both atomically."""
    venv = Path(venv)
    why = unusable(venv)
    if why:
        raise SystemExit(why)
    tar, part = tar_path(venv), tar_path(venv).with_name(tar_path(venv).name + '.partial')
    before = fingerprint(venv)
    t0 = time.time()
    pipe = (f'set -o pipefail; tar -C {shlex.quote(str(venv.parent))} -cf - {shlex.quote(venv.name)} | '
            f'dd of={shlex.quote(str(part))} {{flag}}iflag=fullblock bs=64M status=none')
    _run_direct(pipe, 'oflag=direct ', check=True)
    if fingerprint(venv) != before:
        part.unlink(missing_ok=True)
        raise SystemExit(f'{venv}: its distributions changed while it was being archived; run again')
    # the tar's own member count (not a walk of the venv afterwards: a .pyc an engine writes meanwhile is not in it)
    meta = {'venv': str(venv), 'top': venv.name, 'dists_sha256': before, 'dists': len(dists(venv)),
            'files': tar_files(part), 'bytes': part.stat().st_size, 'built': time.time(),
            'seconds': round(time.time() - t0, 1)}
    os.replace(part, tar)
    tmp = meta_path(venv).with_name(meta_path(venv).name + '.partial')
    tmp.write_text(json.dumps(meta, indent=1) + '\n')
    os.replace(tmp, meta_path(venv))
    return meta


def check(venv) -> tuple:
    """(meta, None) when <venv>.tar holds exactly this venv's distributions, else (None, reason)."""
    venv, tar = Path(venv), tar_path(venv)
    why = unusable(venv)
    if why:
        return None, why
    meta = _read(meta_path(venv))
    if not tar.is_file() or not meta:
        return None, f'no {tar.name} / {meta_path(venv).name} next to the venv (python -m synth.venv_stage build {venv})'
    if meta.get('top') != venv.name:
        return None, f'{tar.name} holds {meta.get("top")!r}, not {venv.name!r}'
    if meta.get('bytes') != tar.stat().st_size:
        return None, f'{tar.name} is {tar.stat().st_size} bytes, its record says {meta.get("bytes")}'
    if meta.get('dists_sha256') != fingerprint(venv):
        return None, f'the venv\'s distributions changed since {tar.name} was built: rebuild it'
    return meta, None


def stage(venv, root, timeout=1800) -> tuple:
    """(path, record): the venv extracted to copy_path(venv, root), or (None, record with the reason). Never raises:
    a server whose venv cannot be staged runs from the source venv (and a partial copy is removed)."""
    venv = Path(venv)
    rec = {'source': str(venv), 'tar': str(tar_path(venv))}
    dest = copy_path(venv, root)
    try:
        meta, reason = check(venv)
        if reason:
            return None, {**rec, 'staged': False, 'reason': reason}
        marker = dest.parent / f'.{venv.name}.staged.json'
        done = _read(marker)
        if done and done.get('dists_sha256') == meta['dists_sha256'] and dest.is_dir():
            return dest, {**rec, **done, 'staged': True, 'reused': True}
        shutil.rmtree(dest, ignore_errors=True)
        marker.unlink(missing_ok=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = _run_direct(f'set -o pipefail; dd if={shlex.quote(str(tar_path(venv)))} {{flag}}bs=64M status=none | '
                        f'tar -x -C {shlex.quote(str(dest.parent))}', 'iflag=direct ', timeout=timeout)
        seconds = round(time.time() - t0, 1)
        if r.returncode:
            shutil.rmtree(dest, ignore_errors=True)
            return None, {**rec, 'staged': False, 'reason': f'extract failed (rc {r.returncode}): {r.stderr[-300:]}'}
        files = count_files(dest)
        if fingerprint(dest) != meta['dists_sha256'] or files != meta.get('files'):
            shutil.rmtree(dest, ignore_errors=True)
            return None, {**rec, 'staged': False,
                          'reason': f'the copy differs from its record ({files} files vs {meta.get("files")})'}
        done = {'path': str(dest), 'dists_sha256': meta['dists_sha256'], 'files': files, 'bytes': meta['bytes'],
                'seconds': seconds, 'tar_built': meta.get('built')}
        marker.write_text(json.dumps(done))
        return dest, {**rec, **done, 'staged': True}
    except Exception as exc:  # noqa: BLE001  (a timeout, a full disk, ...)
        shutil.rmtree(dest, ignore_errors=True)
        return None, {**rec, 'staged': False, 'reason': f'{type(exc).__name__}: {exc}'[:500]}


def tar_gib(venv) -> float:
    """Disk a staged copy needs on the node (the tar's size), 0 without a tar."""
    tar = tar_path(venv)
    return tar.stat().st_size / 2 ** 30 if tar.is_file() else 0.0


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[1] not in ('build', 'check'):
        raise SystemExit('usage: python -m synth.venv_stage build|check VENV')
    if sys.argv[1] == 'build':
        print(json.dumps(build(sys.argv[2])))
    else:
        meta, reason = check(sys.argv[2])
        print(json.dumps(meta) if meta else reason)
        raise SystemExit(0 if meta else 1)
