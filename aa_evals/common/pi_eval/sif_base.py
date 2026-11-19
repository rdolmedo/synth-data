"""An Apptainer SIF as a harbor_ns base image, so a Harbor task can name it in `docker_image` (#348).

  python -m aa_evals.common.pi_eval.sif_base SIF REF [--workdir W] [--path-prefix DIR] [--env K=V ...] \
      [--shared /data/aai/tbench/harbor_ns]           # a CPU condor job: needs ~2x the SIF in node scratch

harbor_ns pulls a task's image from a registry, or unpacks it from its shared base cache
`<shared>/bases/<safe ref>.sqfs` (+ `.json` meta) when another node pulled it before (harbor_ns/builder.py
ensure_base). The AA sandboxes ported to the Pi harness (GDPval, CritPt) are SIFs, not registry images. This puts a
SIF's root filesystem (its squashfs partition: the files the Stirrup path's `apptainer exec` sees) into that cache
under REF, with the clean-up harbor_ns applies to a pulled base. The image environment is the SIF's: the docker
environment it was built from plus its `%environment` exports (what `apptainer exec` sets). REF should look like a
registry host (e.g. `aa-evals.local/name:tag`) that does not exist, so a node that misses the cache fails to pull
instead of fetching something else; that is an environment-start failure, retried as infrastructure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

SHARED = Path('/data/aai/tbench/harbor_ns')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def fs_partition(sif: Path) -> str:
    out = subprocess.run(['apptainer', 'sif', 'list', str(sif)], capture_output=True, text=True, check=True).stdout
    ids = [ln.split('|')[0].strip() for ln in out.splitlines() if 'FS (Squashfs' in ln]
    if len(ids) != 1:
        raise SystemExit(f'{sif}: expected one squashfs partition, found {ids}')
    return ids[0]


def sif_env(rootfs: Path) -> dict[str, str]:
    from harbor_ns.builder import _parse_apptainer_env
    env = _parse_apptainer_env(rootfs)
    f = rootfs / '.singularity.d/env/90-environment.sh'
    for line in f.read_text().splitlines() if f.exists() else []:
        w = shlex.split(line, comments=True)
        if w[:1] == ['export']:
            for kv in w[1:]:
                k, _, v = kv.partition('=')
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k) and '$' not in v:
                    env[k] = v
    return env


def build(sif: Path, ref: str, shared: Path, workdir: str | None, path_prefix: str | None, extra_env: dict) -> dict:
    from harbor_ns.builder import ImageMeta, _detect_libc, _safe_name
    want = sif.with_suffix('.sha256')
    digest = sha256(sif)
    if want.exists() and want.read_text().split()[0] != digest:
        raise SystemExit(f'{sif}: sha256 {digest} differs from {want}')
    name = _safe_name(ref)
    w = Path(tempfile.mkdtemp(prefix='sif_base.', dir=os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp'))
    try:
        part = w / 'sif.sqfs'
        with part.open('wb') as f:
            subprocess.run(['apptainer', 'sif', 'dump', fs_partition(sif), str(sif)], stdout=f, check=True)
        rootfs = w / 'rootfs'
        subprocess.run(['unsquashfs', '-q', '-n', '-no-xattrs', '-d', str(rootfs), str(part)], check=True)
        part.unlink()
        meta = ImageMeta(env=sif_env(rootfs), base=ref)
        meta.oci = True   # nothing to fetch from a registry
        if workdir:
            meta.workdir = workdir
        if path_prefix:
            meta.env['PATH'] = path_prefix + ':' + meta.env.get('PATH', '/usr/local/sbin:/usr/local/bin:/usr/sbin:'
                                                                      '/usr/bin:/sbin:/bin')
        meta.env.update(extra_env)
        # ImageStore._sanitize_rootfs: what a pulled base gets
        for p in ('environment', 'singularity', '.singularity.d', '.exec', '.run', '.shell', '.test'):
            q = rootfs / p
            if q.is_symlink() or q.is_file():
                q.unlink()
            elif q.is_dir():
                shutil.rmtree(q, ignore_errors=True)
        subprocess.run(['chmod', '-R', 'u+rwX', str(rootfs)], check=False)
        for d in ('proc', 'dev', 'sys', 'tmp', 'etc', 'root', 'run'):
            (rootfs / d).mkdir(exist_ok=True)
        os.chmod(rootfs / 'tmp', 0o1777)
        if workdir:
            (rootfs / workdir.lstrip('/')).mkdir(parents=True, exist_ok=True)
        meta.libc = _detect_libc(rootfs)
        d = {**meta.to_json(), 'source': {'sif': str(sif), 'sif_sha256': digest}}
        sq = w / 'base.sqfs'
        # the job's CPUs, not the node's: os.cpu_count() threads on a large node overran a 16 GB job's memory
        procs = min(len(os.sched_getaffinity(0)), 8)
        subprocess.run(['mksquashfs', str(rootfs), str(sq), '-comp', 'zstd', '-quiet', '-no-progress',
                        '-processors', str(procs), '-mem', '4G'], check=True)
        (shared / 'bases').mkdir(parents=True, exist_ok=True)
        dest = shared / 'bases' / f'{name}.sqfs'
        tmp = dest.with_name(dest.name + f'.partial{os.getpid()}')
        shutil.copyfile(sq, tmp)
        os.replace(tmp, dest)
        dest.with_suffix('.json').write_text(json.dumps(d, indent=1))
        print(json.dumps({'ref': ref, 'sqfs': str(dest), 'sqfs_sha256': sha256(dest), 'meta': d}, indent=1))
        return d
    finally:
        subprocess.run(['chmod', '-R', 'u+rwX', str(w)], check=False)
        shutil.rmtree(w, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sif')
    ap.add_argument('ref')
    ap.add_argument('--shared', default=str(SHARED))
    ap.add_argument('--workdir', default=None)
    ap.add_argument('--path-prefix', default=None)
    ap.add_argument('--env', action='append', default=[], metavar='K=V')
    a = ap.parse_args(argv)
    build(Path(a.sif), a.ref, Path(a.shared), a.workdir, a.path_prefix, dict(x.split('=', 1) for x in a.env))


if __name__ == '__main__':
    main()
