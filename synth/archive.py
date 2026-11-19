"""Compact archive of a finished campaign's attempts (user, 2026-10-25: the /data quota of 41.9M files was full; an
attempt directory holds ~27 files). Each pack turns 1,000 attempt directories into two files, losslessly:

    <root>/<kind>/attempts/attempts-00000.jsonl.gz   one JSON line per attempt, what training and the bank read
    <root>/<kind>/attempts/raw-00000.zip             every file of those attempts under <uid>/ (deflate; read any
                                                     member in place with zipfile)
    <root>/<kind>/attempts/index.json                uid -> [shard, line, reward, verifier_infra_failure, truncated]

A line: id, campaign, teacher, kind, task, attempt, retry, record (the manifest record, if any), reward (max over
the trials' verifier/reward.txt), verifier_infra_failure, truncated (worker-result interrupted), worker_result,
first_request (the first wire request with system prompt and tools, as sft_qwen35/prepare.py finds it), sessions
([{path, text}], the native Pi sessions), export (the exported trajectory JSON text, if exported) and, for attempts run
with the Pi finish notice (#375, campaigns since 2026-11-14), finish_notices: the parsed lines of the guard's notice map
(pi-cont16k/finish-notices.jsonl; a v4 guard creates it empty at startup), which sft_qwen35/prepare.py joins to the
training messages. A line without the key had no notice (older campaigns' lines are unchanged).

Each shard is verified before anything is deleted: every member is in the zip with its size and CRC (testzip), and
the JSONL holds one line per attempt. With --delete the shard's attempt directories and export files are then
removed. Packing resumes from the index; readers (iter_attempts, read_raw) work on packed, partly packed and
unpacked campaigns alike.

    python -m synth.archive pack <campaign root> ... [--kind train] [--shard-size 1000] [--workers 16] [--delete]
    python -m synth.archive verify <campaign root> [--kind train]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import glob
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time
import zipfile

from .common import atomic, read

SKIP = {'agent', 'artifacts', 'verifier', 'sessions', 'setup'}


def folder(root, kind):
    return Path(root) / kind / 'attempts'


def load_index(root, kind='train'):
    return read(folder(root, kind) / 'index.json', None) or {'version': 1, 'shards': {}, 'attempts': {}}


def first_request(raw):
    """The first chat request with a system prompt and tool schemas (the rule of sft_qwen35/prepare.py)."""
    for d, dirs, files in os.walk(raw):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if 'wire.jsonl' in files:
            with open(os.path.join(d, 'wire.jsonl'), errors='replace') as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    x = e.get('body') if e.get('t') == 'req' else None
                    if isinstance(x, dict) and x.get('messages') and x.get('tools') \
                            and x['messages'][0].get('role') in ('system', 'developer'):
                        return x
    return None


MISSING_NOTICE_MAP = {'missing': 'finish-notices.jsonl'}


def finish_notices(raw, errors=None):
    """An attempt's Pi finish notice map (pi-cont16k/finish-notices.jsonl, or the not yet moved log directory
    .pi-cont16k/<name>-*/ of an attempt whose wrapper was killed) as parsed lines, or None if it has none.
    Split on '\\n' only (JSON strings may hold U+2028); a line that is not JSON becomes {"malformed": <its start>}, so
    the SFT splice (sft_qwen35/prepare.py _splice_finish_notices) drops the attempt instead of using a partial map."""
    for d, dirs, files in os.walk(raw):
        dirs[:] = sorted(x for x in dirs if x not in SKIP)
        if (os.path.basename(d) == 'pi-cont16k' or os.path.basename(os.path.dirname(d)) == '.pi-cont16k') \
                and 'finish-notices.jsonl' in files:
            try:
                with open(os.path.join(d, 'finish-notices.jsonl'), encoding='utf-8', errors='surrogateescape') as f:
                    text = f.read()
            except OSError as e:
                if errors is not None:
                    errors.append(f'finish-notices.jsonl: {type(e).__name__}: {e}')
                return None
            out = []
            for line in text.split('\n'):
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        out.append({'malformed': line[:200]})
            return out
    return None


def verdict(raw):
    """(reward, verifier_infra_failure, truncated) of an attempt directory: the max reward over its Harbor trials
    (harbor/<trial>/ and harbor/trial/<trial>/), whether a verifier failed for infrastructure reasons, whether the
    worker was interrupted. The archive lines, their index entries and the bank all use this rule."""
    from .bank import INFRA
    raw = Path(raw)
    reward, infra = None, False
    for trial in sorted(glob.glob(f'{raw}/harbor/*/')) + sorted(glob.glob(f'{raw}/harbor/trial/*/')):
        try:
            value = float((Path(trial) / 'verifier' / 'reward.txt').read_text().strip())
            reward = value if reward is None else max(reward, value)
        except (OSError, ValueError):
            pass
        try:
            infra = infra or bool(INFRA.search((Path(trial) / 'verifier' / 'test-stdout.txt').read_text(errors='replace')))
        except OSError:
            pass
    return reward, infra, bool((read(raw / 'worker-result.json', None) or {}).get('interrupted'))


def summarize(raw, record, campaign, kind, errors=None):
    """One attempt directory as an archive line (everything training and the bank read from it). `errors`: a list
    that gets the text of every session or export that could not be read (the line then lacks it); a reader uses it
    to tell a damaged read from a complete one (read_attempts)."""
    raw = Path(raw)
    reward, infra, _ = verdict(raw)
    worker = read(raw / 'worker-result.json', None)
    item = read(raw / 'attempt.json', {}) or {}
    names = (record or {}).get('native_sessions') or sorted(
        str(Path(p).relative_to(raw)) for p in glob.glob(f'{raw}/harbor/*/agent/pi/sessions/*.jsonl'))
    sessions = []
    for name in names:
        try:
            sessions.append({'path': name, 'text': (raw / name).read_text(errors='surrogateescape')})
        except OSError as e:
            if errors is not None:
                errors.append(f'{name}: {type(e).__name__}: {e}')
    export = read_export(record, errors)
    notices = finish_notices(raw, errors)
    if notices is None and sessions and ((item.get('protocol') or {}).get('finish_notice')):
        # A v4 guard creates the map at startup, so an attempt run with the notice whose map is gone has lost it
        # (aa_evals finish_map: finish_notices_missing). Shipped as an entry without the map keys, the SFT splice
        # drops the attempt ("unmapped") instead of training on a trace without the notices the model saw.
        notices = [dict(MISSING_NOTICE_MAP)]
    return {'id': raw.name, 'campaign': campaign['id'], 'teacher': campaign['teacher']['slug'], 'kind': kind,
            'task': item.get('task') or (record or {}).get('task'), 'attempt': item.get('attempt'),
            'retry': item.get('retry'), 'record': record, 'reward': reward, 'verifier_infra_failure': infra,
            'truncated': bool((worker or {}).get('interrupted')), 'worker_result': worker,
            'first_request': first_request(raw), 'sessions': sessions, 'export': export,
            **({'finish_notices': notices} if notices is not None else {})}


def read_export(record, errors=None):
    """The exported trajectory JSON text of a manifest record, or None if it has no export file."""
    if not (record and record.get('path') and os.path.exists(record['path'])):
        return None
    try:
        with gzip.open(record['path'], 'rt') as f:
            return f.read()
    except (OSError, EOFError) as e:
        if errors is None:
            raise
        errors.append(f'export {record["path"]}: {type(e).__name__}: {e}')
        return None


def stage_dir():
    """Node-local scratch for building shards: zip and gzip write in small chunks, which Lustre takes at a few MB/s
    (2026-10-25: ~6.5 MB/s per packer, 100 min per shard). Condor scratch first, then TMPDIR, then /tmp."""
    return Path(os.environ.get('SYNTH_ARCHIVE_STAGE') or os.environ.get('_CONDOR_SCRATCH_DIR')
                or os.environ.get('TMPDIR') or '/tmp')


def publish(local, dest):
    """Copy a staged file to /data in 64 MiB blocks (dd), check size and SHA-256 of the copy, rename into place."""
    import hashlib
    import subprocess
    partial = dest.with_name(dest.name + '.partial')
    # oflag=direct: through the page cache Lustre took ~34 MB/s (a 10 GB stream chunk, 2026-10-31); direct I/O ~1 GB/s
    subprocess.run(['dd', f'if={local}', f'of={partial}', 'bs=64M', 'oflag=direct', 'status=none'], check=True)
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).stat().st_size < 2**31 else \
        subprocess.run(['sha256sum', str(p)], capture_output=True, text=True, check=True).stdout.split()[0]
    if partial.stat().st_size != local.stat().st_size or digest(partial) != digest(local):
        raise ValueError(f'{partial}: copy differs from {local}')
    partial.replace(dest)
    local.unlink()


def restore_access(raw):
    """Give the owner back read access to an attempt directory: an agent's sandbox can leave a directory without
    execute permission (or a file without read), which made the whole pack fail. Only paths we own are changed;
    the zip records the restored modes."""
    uid = os.getuid()
    def fix(p, need):
        st = os.lstat(p)
        if st.st_uid == uid and not stat.S_ISLNK(st.st_mode) and st.st_mode & need != need:
            os.chmod(p, stat.S_IMODE(st.st_mode) | need)
    fix(raw, stat.S_IRWXU)
    for d, dirs, files in os.walk(raw):
        for x in dirs:
            fix(os.path.join(d, x), stat.S_IRWXU)
        for x in files:
            fix(os.path.join(d, x), stat.S_IRUSR)


def add_tree(z, src, prefix, expected):
    """Add directory `src` to zip `z` under `prefix` (ends in '/'), losslessly: empty directories as entries, symlinks
    as link entries (content = target; directories they point to are not descended), FIFOs and other special files
    refused. `expected[name] = size` for every entry, for check_zip."""
    src = Path(src)
    for d, dirs, files in os.walk(src):
        dirs.sort()
        rel = Path(d).relative_to(src)
        base = prefix if str(rel) == '.' else f'{prefix}{rel.as_posix()}/'
        if not files and not dirs:   # keep empty directories
            z.writestr(base, b'')
            expected[base] = 0
        links = [x for x in dirs if (Path(d) / x).is_symlink()]   # symlinks to directories: not descended
        dirs[:] = [x for x in dirs if x not in links]
        for f in sorted(files + links):
            p = Path(d) / f
            name = base + f
            if p.is_symlink():   # stored as a symlink entry whose content is the link target
                target = os.readlink(p)
                info = zipfile.ZipInfo(name)
                info.create_system, info.external_attr = 3, (0o120777 << 16)
                z.writestr(info, target)
                expected[name] = len(target.encode())
            elif not p.is_file():   # a FIFO would block the read forever: fail this shard instead
                raise ValueError(f'{p}: not a regular file')
            else:
                z.write(p, name)
                expected[name] = p.stat().st_size


def check_zip(path, expected):
    """The zip holds exactly the `expected` members with their sizes, and every member passes its CRC check."""
    with zipfile.ZipFile(path) as z:
        sizes = {i.filename: i.file_size for i in z.infolist()}
        if sizes != expected:
            raise ValueError(f'{path}: members differ from the directories packed')
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f'{path}: CRC check failed at {bad}')


def pack_shard(job):
    """Write and verify one shard (zip + JSONL) from attempt directories; returns its index entry. Both files are
    built and verified in node-local scratch, then copied to the campaign (see stage_dir, publish)."""
    out, number, raws, records, campaign, kind = job
    zip_path, lines_path = out / f'raw-{number:05d}.zip', out / f'attempts-{number:05d}.jsonl.gz'
    stage = Path(tempfile.mkdtemp(prefix=f'synth-archive-{number:05d}-', dir=stage_dir()))
    zip_tmp, lines_tmp = stage / zip_path.name, stage / lines_path.name
    expected, lines = {}, []
    with zipfile.ZipFile(zip_tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for raw in raws:
            raw = Path(raw)
            restore_access(raw)
            add_tree(z, raw, f'{raw.name}/', expected)
            lines.append(summarize(raw, records.get(raw.name), campaign, kind))
    with gzip.open(lines_tmp, 'wt', compresslevel=6) as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    check_zip(zip_tmp, expected)
    with gzip.open(lines_tmp, 'rt') as f:
        ids = [json.loads(line)['id'] for line in f]
    if ids != [Path(r).name for r in raws]:
        raise ValueError(f'{lines_tmp}: lines differ from the attempts')
    publish(zip_tmp, zip_path)
    publish(lines_tmp, lines_path)
    shutil.rmtree(stage, ignore_errors=True)
    return {'out': str(out), 'shard': number, 'zip': zip_path.name, 'lines': lines_path.name, 'files': len(expected),
            'bytes': sum(expected.values()), 'zip_bytes': zip_path.stat().st_size,
            'lines_bytes': lines_path.stat().st_size, 'packed_at': time.time(),
            'attempts': [[x['id'], x['reward'], x['verifier_infra_failure'], x['truncated']] for x in lines]}


def remove_packed(root, kind, shard, uids, records, check):
    """Delete the attempt directories (and export files) of a packed shard. With check, first confirm that every
    file still on disk is in the shard's zip with its size (a shard packed by an earlier run without --delete)."""
    root, raw_root = Path(root), Path(root) / kind / 'raw'
    if check:
        with zipfile.ZipFile(folder(root, kind) / shard['zip']) as z:
            sizes = {i.filename: i.file_size for i in z.infolist()}
        for uid in uids:
            for d, dirs, files in os.walk(raw_root / uid):
                for f in files + [x for x in dirs if (Path(d) / x).is_symlink()]:
                    p = Path(d) / f
                    size = len(os.readlink(p).encode()) if p.is_symlink() else p.stat().st_size
                    if sizes.get(f'{uid}/{p.relative_to(raw_root / uid).as_posix()}') != size:
                        raise ValueError(f'{p}: not in {shard["zip"]} as on disk; not deleting')
    freed = 0
    for uid in uids:
        export = (records.get(uid) or {}).get('path')
        if export and Path(export).resolve().is_relative_to(root.resolve()) and os.path.exists(export):
            os.unlink(export)
            freed += 1
        if (raw_root / uid).exists():
            freed += sum(len(files) for _, _, files in os.walk(raw_root / uid))
            shutil.rmtree(raw_root / uid)
    return freed


def adopt_published(out, index):
    """Shards an interrupted run published (final zip and JSONL) but never indexed: index them instead of packing
    their attempts again. Their directories are deleted later only after the size check against the zip."""
    adopted = 0
    for lines_path in sorted(out.glob('attempts-*.jsonl.gz')):
        number = int(lines_path.name[len('attempts-'):-len('.jsonl.gz')])
        zip_path = out / f'raw-{number:05d}.zip'
        if str(number) in index['shards'] or not zip_path.exists():
            continue
        try:
            with gzip.open(lines_path, 'rt') as f:
                lines = [json.loads(line) for line in f]
            with zipfile.ZipFile(zip_path) as z:
                infos = z.infolist()
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            continue   # unreadable: left in place, its attempts are packed again
        index['shards'][str(number)] = {'shard': number, 'zip': zip_path.name, 'lines': lines_path.name,
                                        'files': len(infos), 'bytes': sum(i.file_size for i in infos),
                                        'zip_bytes': zip_path.stat().st_size, 'lines_bytes': lines_path.stat().st_size,
                                        'packed_at': lines_path.stat().st_mtime, 'adopted': True}
        for line, x in enumerate(lines):
            index['attempts'][x['id']] = [number, line, x['reward'], x['verifier_infra_failure'], x['truncated']]
        adopted += 1
    return adopted


def plan(root, kind, shard_size, only=None, campaign=None):
    """(campaign context, shard jobs) of one campaign: its attempt directories not yet in the index. A running
    campaign is packed only through `only` (the controller's list of finished attempts with final exports)."""
    root = Path(root)
    campaign = campaign or read(root / 'ops' / 'campaign.json', None)
    if not campaign:
        raise SystemExit(f'{root}: no ops/campaign.json')
    if kind == 'train' and only is None and not (root / 'ops' / 'completion.json').exists():
        raise SystemExit(f'{root}: campaign not complete; pack finished campaigns only (or pass the attempts to pack)')
    out = folder(root, kind)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('*.partial'):   # an interrupted earlier pack; its attempts are packed again below
        stale.unlink()
    index = load_index(root, kind)
    index.update(campaign=campaign['id'], kind=kind)
    adopted = adopt_published(out, index)
    if adopted:
        atomic(out / 'index.json', index)
    raw_root = root / kind / 'raw'
    raws = sorted(str(p) for p in raw_root.iterdir() if p.is_dir()) if raw_root.exists() else []
    todo = [r for r in raws if Path(r).name not in index['attempts'] and (only is None or Path(r).name in only)]
    records = {r['id']: r for r in (read(root / kind / 'manifest.json', {}) or {}).get('records', [])}
    first = max((int(k) for k in index['shards']), default=-1) + 1
    jobs = [(out, first + i, todo[o:o + shard_size], {Path(r).name: records.get(Path(r).name) for r in todo[o:o + shard_size]},
             campaign, kind) for i, o in enumerate(range(0, len(todo), shard_size))]
    return {'root': root, 'out': out, 'index': index, 'records': records, 'raws': len(raws), 'adopted': adopted}, jobs


def pack_many(roots, kind='train', shard_size=1000, workers=16, delete=False, log=print, only=None, campaign=None):
    """Pack several campaigns with one process pool, so small campaigns do not idle the workers; each shard's
    index entry is written (and with delete its directories removed) as soon as that shard is verified."""
    contexts, jobs = {}, []
    for root in roots:
        context, campaign_jobs = plan(root, kind, shard_size, only, campaign)
        contexts[str(context['out'])] = context
        jobs += campaign_jobs
        log(f'{context["root"].name}/{kind}: {context["raws"]} attempt dirs, '
            f'{sum(len(j[2]) for j in campaign_jobs)} to pack in {len(campaign_jobs)} shards'
            + (f', {context["adopted"]} published shards adopted' if context['adopted'] else ''))
    # Deletions run in threads (one shard's ~27k unlinks took ~1-2 min and serialized the whole job); a shard's
    # directories are only deleted after its index entry is written, and deleted_at is recorded here, not in threads.
    deleter, pending, failures = ThreadPoolExecutor(16) if delete else None, {}, []

    def reap(wait=False):
        for future in [f for f in pending if wait or f.done()]:
            context, number, how = pending.pop(future)
            freed = future.result()
            context['index']['shards'][number]['deleted_at'] = time.time()
            atomic(context['out'] / 'index.json', context['index'])
            log(f'  {context["root"].name} deleted shard {int(number):05d}: {freed} files{how}')

    def schedule(context, number, check):
        uids = [u for u, v in context['index']['attempts'].items() if v[0] == int(number)]
        future = deleter.submit(remove_packed, context['root'], kind, context['index']['shards'][number], uids,
                                context['records'], check)
        pending[future] = (context, number, ' (packed earlier)' if check else '')

    if delete and only is None:   # shards packed (or adopted) earlier and not deleted yet
        for context in contexts.values():
            for number, shard in sorted(context['index']['shards'].items(), key=lambda kv: int(kv[0])):
                if not shard.get('deleted_at'):
                    schedule(context, number, check=True)
    # spawn: the controller calls this with live threads, and fork after threads can deadlock
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(pack_shard, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                entry = future.result()
            except Exception as exc:   # this shard's attempts stay unpacked; the others go on
                failures.append(exc)
                job = futures[future]
                log(f'  FAILED shard {job[1]:05d} of {Path(job[0]).parents[1].name}: {exc!r}')
                continue
            context = contexts[str(Path(entry['out']))]
            index, out = context['index'], context['out']
            number = str(entry['shard'])
            index['shards'][number] = {k: v for k, v in entry.items() if k not in ('attempts', 'out')}
            for line, (uid, reward, infra, truncated) in enumerate(entry['attempts']):
                index['attempts'][uid] = [entry['shard'], line, reward, infra, truncated]
            atomic(out / 'index.json', index)
            log(f'  {context["root"].name} shard {entry["shard"]:05d}: {len(entry["attempts"])} attempts, '
                f'{entry["files"]} files, {entry["bytes"] / 2**30:.1f} GiB -> '
                f'{(entry["zip_bytes"] + entry["lines_bytes"]) / 2**30:.2f} GiB')
            if delete:   # this shard was verified (testzip, line ids, copy checksum) just now
                schedule(context, number, check=False)
            reap()
    if deleter is not None:
        reap(wait=True)
        deleter.shutdown()
    if failures:
        raise RuntimeError(f'{len(failures)} shards failed (their attempts stay unpacked): {failures[0]!r}')
    return {str(c['root']): c['index'] for c in contexts.values()}


def pack(root, kind='train', shard_size=1000, workers=16, delete=False, log=print, only=None):
    return pack_many([root], kind, shard_size, workers, delete, log, only)[str(Path(root))]


def verify(root, kind='train'):
    """Re-check every shard of an archive: zip CRCs and one JSONL line per indexed attempt."""
    out, index = folder(root, kind), load_index(root, kind)
    for shard in index['shards'].values():
        with zipfile.ZipFile(out / shard['zip']) as z:
            if z.testzip() is not None or len(z.infolist()) != shard['files']:
                raise ValueError(f'{shard["zip"]}: verification failed')
        with gzip.open(out / shard['lines'], 'rt') as f:
            n = sum(1 for _ in f)
        if n != sum(1 for v in index['attempts'].values() if v[0] == shard['shard']):
            raise ValueError(f'{shard["lines"]}: line count differs from the index')
    return len(index['shards']), len(index['attempts'])


class Incomplete(RuntimeError):
    """Usable, finalized manifest records that came back from neither a shard nor their directory."""
    def __init__(self, root, missing, wanted):
        self.missing = missing
        shown = '; '.join(f'{k} ({v})' for k, v in sorted(missing.items())[:10])
        super().__init__(f'{root}: {len(missing)} of {wanted} usable, finalized attempts could not be read from a '
                         f'shard or their directory (issues #166, #236), e.g. {shown}')


def usable(record):
    return bool(record and record.get('usable') and record.get('finalized'))


def consistent_reads(paths, read_one, packed_now, workers=8, chunk=256):
    """Read loose directories consistently while a packer may pack and delete them (issues #166, #236, #245).

    `paths`: {name: directory}. `read_one(directory)` runs on every directory of a chunk (`workers` threads; an
    OSError becomes `error`), then `packed_now()` returns the packer's index read again; `name in` it means packed.
    Yields (name, value, error, index) chunk by chunk. A name the re-read index lists may have lost files during its
    read, so the caller takes it from its shard (`index` is the index that lists it; `value` and `error` are None).
    A name it does not list was not being deleted during its read: a packer writes the shard, then the index
    (atomically), and only then deletes the directories, so a deletion that overlapped the read would have come
    after an index write that lists it. Its read stands (`index` None)."""
    names = list(paths)

    def one(name):
        try:
            return name, read_one(paths[name]), None
        except OSError as e:
            return name, None, f'{type(e).__name__}: {e}'
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for o in range(0, len(names), chunk):
            done = list(pool.map(one, names[o:o + chunk]))
            index = packed_now()
            for name, value, error in done:
                if name in index:
                    yield name, None, None, index
                else:
                    yield name, value, error, None


def read_dirs(root, kind, uids, read_one, workers=8, chunk=256):
    """consistent_reads of a campaign's attempt directories; `index` is the archive index (load_index)."""
    raw_root = Path(root) / kind / 'raw'

    class Packed(dict):   # `in` asks the attempts; the caller gets the whole index
        def __contains__(self, uid):
            return uid in self['attempts']
    for uid, value, error, index in consistent_reads({u: raw_root / u for u in uids}, read_one,
                                                     lambda: Packed(load_index(root, kind)), workers, chunk):
        yield uid, value, error, (dict(index) if index is not None else None)


def shard_lines(root, kind, index, numbers, keep=None):
    """Archive lines of the given shards (a shard never changes once it is in the index); `keep(line text)`: a cheap
    filter before parsing."""
    out = folder(root, kind)
    for number in sorted({str(n) for n in numbers}, key=int):
        with gzip.open(out / index['shards'][number]['lines'], 'rt') as f:
            for text in f:
                if keep is None or keep(text):
                    yield json.loads(text)


def complete(line, record):
    """Why an archive line does not hold a usable, finalized record's training data, or None if it does: the
    record, the first wire request and every native session."""
    if not usable(line.get('record')):
        return 'not usable and finalized in its archive line'
    if not line.get('first_request'):
        return 'no first request'
    names = record.get('native_sessions') or []
    if not line.get('sessions') or len(line['sessions']) < len(names):
        return f'{len(line.get("sessions") or [])} of {len(names) or "≥1"} sessions'
    return None


def read_attempts(root, kind='train', known=(), tasks=None, workers=8, chunk=256, check='raise', log=print):
    """Every attempt of a campaign as an archive line, read consistently while synth/archive.py packs it (#236):
      1. the index (I1) and its shards: the packed attempts, in shard order;
      2. the attempt directories not in I1, `chunk` at a time (read_dirs): one packed during its read is taken from
         its shard instead;
      3. the index again: attempts packed between I1 and the directory listing come from their shards (last).
    `known`: attempt ids to leave out (a --follow rescan); a shard whose attempts are all known is not opened.
    `tasks`: only those tasks.

    A shard line whose attempt was packed after it finished but before its export was finalized has "record": null
    (1,526 of 102,099 on m1-hero, #166/#167) or a record not yet finalized. When the manifest marks the attempt
    usable and finalized and the line holds its sessions and first request, the line gets the manifest record (and the export, if its file is still
    there) and `record_from_manifest: true`, as sft_qwen35/prepare.py takes it.

    Completeness: every usable, finalized record of manifest.json (as read at the start; minus `known` and other
    tasks) must come back with its record, first request and sessions. check='raise' raises Incomplete at the end,
    naming the missing attempts; 'warn' logs them (a live rescan reads them again next time); None skips the check.
    Attempts that are neither in the manifest nor in the index (running ones) are yielded as they read."""
    root, known = Path(root), set(known)
    campaign = read(root / 'ops' / 'campaign.json', {}) or {}
    records = {r['id']: r for r in (read(root / kind / 'manifest.json', {}) or {}).get('records', [])}
    want = {i for i, r in records.items() if usable(r) and i not in known and (tasks is None or r.get('task') in tasks)}
    missing, done, recovered = {}, set(), 0

    def take(line):
        nonlocal recovered
        uid = line['id']
        task = line.get('task') or (line.get('record') or records.get(uid) or {}).get('task')
        if uid in known or uid in done or (tasks is not None and task not in tasks):
            return None
        done.add(uid)
        if not usable(line.get('record')) and usable(records.get(uid)) and line.get('sessions') \
                and line.get('first_request'):
            line = {**line, 'record': records[uid], 'record_from_manifest': True,
                    'export': line.get('export') or read_export(records[uid], [])}
            recovered += 1
        if uid in want:
            why = complete(line, records[uid])
            if why:
                missing[uid] = why
                return None
        return line

    keep = None if tasks is None else (lambda text: any(t in text for t in tasks))
    first = load_index(root, kind)
    by_shard = {}
    for uid, loc in first['attempts'].items():
        by_shard.setdefault(str(loc[0]), []).append(uid)
    numbers = [n for n in first['shards'] if not (by_shard.get(n) and all(u in known for u in by_shard[n]))]
    for line in shard_lines(root, kind, first, numbers, keep):
        line = take(line)
        if line is not None:
            yield line
    raw_root = root / kind / 'raw'
    uids = sorted(p.name for p in raw_root.iterdir() if p.is_dir()) if raw_root.exists() else []
    uids = [u for u in uids if u not in first['attempts'] and u not in known
            and (tasks is None or u in want or u not in records or records[u].get('task') in tasks)]

    def summary(raw):
        errors = []
        line = summarize(raw, records.get(raw.name), campaign, kind, errors)
        return line, errors
    late = set()
    for uid, value, error, index in read_dirs(root, kind, uids, summary, workers, chunk):
        if index is not None:   # packed while it was read: from its shard, below
            late.add(uid)
            continue
        line, errors = value or (None, [error])
        if errors and uid in want:
            missing[uid] = '; '.join(errors)
            continue
        if line is None:
            log(f'{uid}: unreadable this read ({error})')
            continue
        line = take(line)
        if line is not None:
            yield line
    last = load_index(root, kind)
    late |= {u for u in last['attempts'] if u not in first['attempts'] and u not in done and u not in known}
    for uid in late:
        missing.pop(uid, None)
    if late:
        for line in shard_lines(root, kind, last, {last['attempts'][u][0] for u in late}, keep):
            if line['id'] in late:
                line = take(line)
                if line is not None:
                    yield line
    for uid in want - done - set(missing):
        missing[uid] = 'in neither a shard nor its directory'
    if recovered:
        log(f'{root.name}/{kind}: {recovered} packed attempts without a record in their archive line took the '
            f'manifest record (usable, finalized)')
    if missing and check == 'raise':
        raise Incomplete(root, missing, len(want))
    if missing and check == 'warn':
        log(str(Incomplete(root, missing, len(want))))


def iter_attempts(root, kind='train', **kw):
    """Every attempt of a campaign as an archive line (read_attempts: consistent while it is packed, and it raises
    if a usable, finalized manifest record does not come back)."""
    return read_attempts(root, kind, **kw)


def read_raw(root, uid, relpath, kind='train'):
    """Bytes of one file of an attempt, from its directory or its shard's zip. The directory is tried first; if the
    file is not there the index is read after that try, so an attempt packed and deleted in between is found in
    its shard (KeyError: the attempt is in neither)."""
    path = Path(root) / kind / 'raw' / uid / relpath
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    index = load_index(root, kind)
    if uid not in index['attempts']:
        raise FileNotFoundError(path)
    shard = index['shards'][str(index['attempts'][uid][0])]
    with zipfile.ZipFile(folder(root, kind) / shard['zip']) as z:
        return z.read(f'{uid}/{relpath}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pack')
    p.add_argument('roots', nargs='+')
    p.add_argument('--kind', default='train')
    p.add_argument('--shard-size', type=int, default=1000)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--delete', action='store_true', help='remove attempt dirs and export files of verified shards')
    p.add_argument('--only', help='JSON list of attempt ids to pack (a running campaign; written by the controller)')
    v = sub.add_parser('verify')
    v.add_argument('roots', nargs='+')
    v.add_argument('--kind', default='train')
    a = ap.parse_args()
    if a.cmd == 'pack':
        os.nice(15)   # behind the controller and the exporter on a controller job's CPUs
        only = set(json.loads(Path(a.only).read_text())) if a.only else None
        pack_many(a.roots, a.kind, a.shard_size, a.workers, a.delete, log=lambda m: print(m, flush=True), only=only)
    else:
        for r in a.roots:
            print(r, 'shards, attempts:', verify(r, a.kind), flush=True)
