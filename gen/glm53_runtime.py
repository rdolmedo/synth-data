#!/usr/bin/env python3
"""Snapshot the pinned NFS runtime and relocate it without changing packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

OPS = Path('/data/aai/tbench/pi-cont16k-110m/20261015T215325Z/ops/glm53-flash')
SOURCE_VENV = Path('/opt/aai/venvs/vllm031')
SOURCE_PYTHON = Path('/home/user/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu').resolve()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as inp:
        for block in iter(lambda: inp.read(16 << 20), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def pack():
    root = OPS / 'runtime'
    root.mkdir(exist_ok=True)
    if (root / 'ready.json').exists():
        raise RuntimeError('An immutable runtime snapshot is already published')
    archive = root / f'vllm031-python312-{time.time_ns()}.tar'
    partial = archive.with_suffix('.partial')
    atomic(root / 'packing.json', {'started_at': time.time(), 'pid': os.getpid(),
                                  'venv': str(SOURCE_VENV), 'python': str(SOURCE_PYTHON)})
    tar = subprocess.Popen(['tar', '--sort=name', '-cf', '-', '-C', str(SOURCE_VENV.parent),
                            SOURCE_VENV.name, '-C', str(SOURCE_PYTHON.parent), SOURCE_PYTHON.name],
                           stdout=subprocess.PIPE)
    try:
        copied = subprocess.run(['dd', 'of=' + str(partial), 'bs=64M', 'iflag=fullblock',
                                 'oflag=direct', 'status=progress'], stdin=tar.stdout)
        tar.stdout.close()
        if tar.wait() or copied.returncode:
            raise RuntimeError('Runtime snapshot failed; no ready manifest published')
    finally:
        if tar.poll() is None:
            tar.terminate()
            tar.wait()
    partial.replace(archive)
    atomic(root / 'ready.json', {'archive': str(archive), 'archive_sha256': digest(archive),
                                'bytes': archive.stat().st_size, 'created_at': time.time(),
                                'source_venv': str(SOURCE_VENV), 'source_python': str(SOURCE_PYTHON),
                                'packages_changed': False,
                                'reason': 'vLLM API workers waiting on NFS filesystem RPC during imports'})


def stage():
    info = json.loads((OPS / 'runtime/ready.json').read_text())
    root = Path(os.environ['_CONDOR_SCRATCH_DIR']) / ('glm53-runtime-' + info['archive_sha256'][:12])
    venv, python = root / SOURCE_VENV.name, root / SOURCE_PYTHON.name
    if not (root / 'relocation.json').exists():
        root.mkdir(exist_ok=True)
        if digest(info['archive']) != info['archive_sha256']:
            raise RuntimeError('Pinned runtime archive checksum mismatch')
        subprocess.run(['tar', '--no-same-owner', '-xf', info['archive'], '-C', str(root)], check=True)
        # Only interpreter links/config and the CLI launcher need relocation;
        # package sources and binaries remain byte-for-byte the snapshot.
        for name in ('python', 'python3', 'python3.12'):
            link = venv / 'bin' / name
            link.unlink(missing_ok=True)
            link.symlink_to(python / 'bin/python3.12')
        cfg = venv / 'pyvenv.cfg'
        cfg.write_text('\n'.join('home = ' + str(python / 'bin') if line.startswith('home =') else line
                                 for line in cfg.read_text().splitlines()) + '\n')
        cli = venv / 'bin/vllm'
        cli.write_text('#!/bin/sh\nexec ' + shlex.quote(str(venv / 'bin/python'))
                       + ' -m vllm.entrypoints.cli.main "$@"\n')
        cli.chmod(0o755)
        result = subprocess.run([str(venv / 'bin/python'), '-c',
            'import json,sys; print(json.dumps({"prefix":sys.prefix,"base_prefix":sys.base_prefix}))'],
            capture_output=True, text=True, check=True)
        prefixes = json.loads(result.stdout)
        if Path(prefixes['prefix']) != venv or Path(prefixes['base_prefix']) != python:
            raise RuntimeError('Relocated Python still uses the shared runtime: ' + result.stdout)
        atomic(root / 'relocation.json', {**info, 'staged_at': time.time(), **prefixes})
    job = os.environ.get('CAMPAIGN_JOB_ID')
    if job:
        evidence = OPS / ('server-' + job)
        evidence.mkdir(exist_ok=True)
        atomic(evidence / 'runtime-relocation.json', json.loads((root / 'relocation.json').read_text()))
    print(venv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('pack', 'stage'))
    args = parser.parse_args()
    pack() if args.mode == 'pack' else stage()
