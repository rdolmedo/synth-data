"""Task streams: a running trace campaign takes new tasks while they are generated (`[train.stream]`).

Without a stream, `synth.prepare` freezes the task set: one bundle (ops/train_tasks.tar) whose list and sha256
ops/campaign.json records. The 330M task-synthesis proof of concept therefore traced in batches (s1, s2, s2b, ...):
Super idled while a batch finished, and each campaign had its own tail. A stream is an append-only directory of
immutable chunks that a producer publishes while the campaign runs:

  <stream>/chunks/NNNN.tar    tasks/<name>/... (the layout of synth.datasets.bundle)
  <stream>/chunks/NNNN.json   commit record, written last: the chunk's tasks in campaign order, the tar's sha256 and
                              size, and the producer's note on where the tasks came from
  <stream>/chunks/NNNN.meta.json  optional producer metadata per task (not read by campaigns)
  <stream>/END.json           end of stream, {"chunks": N, ...}: the campaign takes chunks 0..N-1 and no more

The controller admits committed chunks in index order (a missing index holds back the later ones). It checks each tar
against its record, rewrites agent limits as synth.prepare does for the frozen bundle, and keeps the result in
ops/stream/train-NNNN.tar with its audit ops/stream/train-NNNN.json; ops/stream/chunks.json lists every admitted
chunk with its source and sha256. A task whose task.toml cannot be rewritten is left out of its chunk (`invalid`); an
operator file ops/stream/skip.json ({"<index>": reason}) passes over a chunk that cannot be admitted at all. A worker
extracts a chunk the first time an attempt needs one of its tasks.

  python -m synth.stream publish <stream> <harbor task dir> [--order FILE] [--source JSON]
  python -m synth.stream end <stream> [--reason TEXT]
  python -m synth.stream status <stream>
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import time

from . import archive
from .common import atomic, read, read_task_list, sha

LOCK_STALE_SEC = 2 * 3600


def chunk_dir(stream):
    return Path(stream) / 'chunks'


def committed(stream):
    """Commit records of the published chunks, in index order, up to the first missing index."""
    records = []
    while (record := read(chunk_dir(stream) / f'{len(records):04d}.json')) is not None:
        records.append(record)
    return records


def ended(stream):
    """END.json of a closed stream, else None."""
    return read(Path(stream) / 'END.json')


class PublishLock:
    """One producer at a time (Lustre has no flock): an mkdir. A holder is gone when its process is dead on this host,
    or LOCK_STALE_SEC after it took the lock; `rm -r <stream>/publish.lock` frees it by hand."""

    def __init__(self, stream):
        self.path = Path(stream) / 'publish.lock'

    def held(self):
        owner = read(self.path / 'owner.json', {}) or {}
        if owner.get('host') == socket.gethostname() and owner.get('pid'):
            try:
                os.kill(owner['pid'], 0)
            except ProcessLookupError:
                return None
            except PermissionError:
                pass
        return owner if time.time() - self.path.stat().st_mtime < LOCK_STALE_SEC else None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            if (owner := self.held()) is not None:
                raise RuntimeError(f'{self.path} is held by {owner}; one producer per stream') from None
            shutil.rmtree(self.path, ignore_errors=True)
            self.path.mkdir()
        atomic(self.path / 'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time()})
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)


def invalid_task(toml_bytes):
    """Why a task.toml cannot be admitted (synth.prepare's agent-limit rewrite fails on it), or None."""
    from .prepare import rewrite_agent_timeouts
    try:
        rewrite_agent_timeouts(toml_bytes, 21600)
    except Exception as exc:  # noqa: BLE001  (TOML errors, no [agent].timeout_sec, an empty file)
        return repr(exc)[:300]
    return None


def publish(stream, tasks_dir, names=None, source=None, meta=None):
    """Publish the Harbor tasks `names` (campaign order; default: every task of tasks_dir, sorted) of tasks_dir as the
    next chunk. Task names must be new to the stream. The tar is built in node-local scratch and copied with dd (Lustre
    takes small writes at a few MB/s); `meta` ({task: ...}, the producer's) goes to NNNN.meta.json; the commit record
    is written last. -> the record."""
    import tarfile
    stream, tasks_dir = Path(stream), Path(tasks_dir)
    with PublishLock(stream):
        if ended(stream):
            raise ValueError(f'{stream} has ended (END.json); start a new stream')
        if names is None:
            names = sorted(p.name for p in tasks_dir.iterdir()
                           if (p / 'task.toml').is_file() and (p / 'instruction.md').is_file())
        if len(set(names)) != len(names):
            raise ValueError('duplicate task names in the chunk')
        records = committed(stream)
        seen = {t for r in records for t in r['tasks']}
        if seen & set(names):
            raise ValueError(f'tasks already in {stream}: {sorted(seen & set(names))[:10]}')
        bad = {n: 'no task.toml or instruction.md' for n in names
               if not ((tasks_dir / n / 'task.toml').is_file() and (tasks_dir / n / 'instruction.md').is_file())}
        bad.update({n: e for n in names if n not in bad and (e := invalid_task((tasks_dir / n / 'task.toml').read_bytes()))})
        if bad:  # a chunk is immutable: a task the campaign cannot admit must not be committed
            raise ValueError(f'tasks campaigns cannot admit in {tasks_dir}: {dict(list(bad.items())[:10])}')
        index = len(records)
        out = chunk_dir(stream)
        out.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synth-stream-', dir=archive.stage_dir()) as tmp:
            local = Path(tmp) / f'{index:04d}.tar'
            with tarfile.open(local, 'w') as tar:
                for name in names:
                    tar.add(tasks_dir / name, arcname=f'tasks/{name}')
            digest, size = sha(local), local.stat().st_size
            archive.publish(local, out / f'{index:04d}.tar')   # dd, size + sha256 check, rename
        record = {'index': index, 'tasks': list(names), 'count': len(names), 'tar': f'chunks/{index:04d}.tar',
                  'sha256': digest, 'bytes': size, 'source': source or {}, 'created_at': time.time(),
                  'host': socket.gethostname()}
        if meta is not None:
            atomic(out / f'{index:04d}.meta.json', meta)
            record['meta'] = f'chunks/{index:04d}.meta.json'
        atomic(out / f'{index:04d}.json', record)
        return record


def end(stream, reason=''):
    """Close the stream after its committed chunks."""
    stream = Path(stream)
    with PublishLock(stream):
        if ended(stream):
            raise ValueError(f'{stream} has already ended: {ended(stream)}')
        record = {'chunks': len(committed(stream)), 'reason': reason, 'time': time.time()}
        atomic(stream / 'END.json', record)
        return record


def admit(campaign, record, stream):
    """A committed chunk -> ops/stream/train-NNNN.tar (agent limits rewritten as in synth.prepare) and its audit
    ops/stream/train-NNNN.json, written last. A task whose task.toml cannot be rewritten (or that is missing from the
    tar) is left out and listed under `invalid`: one bad task must not hold back the stream. Idempotent: an admitted
    chunk returns its audit."""
    import tarfile
    from .prepare import rewrite_bundle
    folder = campaign.ops / 'stream'
    name = f'train-{record["index"]:04d}'
    target, audit_path = folder / f'{name}.tar', folder / f'{name}.json'
    if (audit := read(audit_path)) is not None:
        return audit
    source = Path(stream) / record['tar']
    if sha(source) != record['sha256']:
        raise ValueError(f'{source}: sha256 differs from its commit record')
    invalid = {t: 'not in the chunk tar' for t in record['tasks']}
    with tarfile.open(source) as tar:
        for member in tar:
            parts = member.name.split('/')
            if len(parts) == 3 and parts[0] == 'tasks' and parts[2] == 'task.toml' and parts[1] in invalid:
                data = tar.extractfile(member)
                invalid[parts[1]] = invalid_task(data.read()) if data else 'task.toml is not a file'
    invalid = {t: e for t, e in invalid.items() if e}
    tasks = [t for t in record['tasks'] if t not in invalid]
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synth-stream-', dir=archive.stage_dir()) as tmp:
        local = Path(tmp) / f'{name}.tar'
        audit = rewrite_bundle(source, set(tasks), local, campaign.protocol['agent_timeout_sec'])
        digest = sha(local)
        archive.publish(local, target)
    audit.update(path=str(target), sha256=digest, count=len(tasks), chunk=record['index'], invalid=invalid,
                 stream=str(stream), commit={k: v for k, v in record.items() if k != 'tasks'}, tasks_order=tasks,
                 admitted_at=time.time())
    atomic(audit_path, audit)
    return audit


def campaign_tars(root):
    """Task tars of a campaign: the frozen bundle (if any), then its admitted stream chunks in order."""
    ops = Path(root) / 'ops'
    frozen = [ops / 'train_tasks.tar'] if (ops / 'train_tasks.tar').exists() else []
    return frozen + sorted((ops / 'stream').glob('train-*.tar'))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('publish')
    p.add_argument('stream')
    p.add_argument('tasks_dir', help='directory of Harbor tasks (<name>/task.toml, instruction.md, ...)')
    p.add_argument('--order', help='task list (campaign order); default: every task of tasks_dir, sorted')
    p.add_argument('--source', default='{}', help='JSON note recorded with the chunk (where the tasks came from)')
    p = sub.add_parser('end')
    p.add_argument('stream')
    p.add_argument('--reason', default='')
    p = sub.add_parser('status')
    p.add_argument('stream')
    a = ap.parse_args()
    if a.op == 'publish':
        record = publish(a.stream, a.tasks_dir, read_task_list(a.order) if a.order else None, json.loads(a.source))
        print(json.dumps({k: v for k, v in record.items() if k != 'tasks'}))
    elif a.op == 'end':
        print(json.dumps(end(a.stream, a.reason)))
    else:
        records = committed(a.stream)
        print(json.dumps({'chunks': len(records), 'tasks': sum(r['count'] for r in records),
                          'last_chunk_at': records[-1]['created_at'] if records else None, 'end': ended(a.stream)}))


if __name__ == '__main__':
    main()
