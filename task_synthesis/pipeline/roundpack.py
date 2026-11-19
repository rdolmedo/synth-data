"""Pack a finished round's loose files into a few archives (#245), and read packed and unpacked rounds alike.

A round writes ~38 loose files per accepted task (tasks/ ~27, work/ ~9, assets/ ~2): m1-hero's round added ~2.45M
files against /data's 41.9M-file quota. Packing turns work/<spec>/ and tasks/<spec>/ into, per shard of specs:

    <round>/packed/round-00000.zip         every file under work/<spec>/... and tasks/<spec>/... (deflate; symlinks
                                            as links, empty directories kept; read any member in place)
    <round>/packed/records-00000.jsonl.gz  one line per spec: {"spec_id", "record", "error"} (work/<spec>/record.json
                                            and error.json parsed, null when absent), what the round's readers read
    <round>/packed/index.json              spec -> [shard, line, status, judge pass]; shards with their sizes

Each shard is built in node-local scratch and verified (every member with its size, CRC check, one line per spec)
before it is copied to the round (dd + SHA-256), and the index is written before anything is deleted. With --delete
a spec's loose files are then removed one by one, each only after its size and CRC32 on disk match its zip member;
a file that differs or was added since stays, and readers take a loose file first. assets/, bind/, specs.jsonl and
the round's logs stay loose (a few files per round, or large tarballs).

Readers go through RoundFiles, which reads packed, partly packed and unpacked rounds and is consistent while a pack
runs (the #166/#236 rule: the index, then the loose files, then the index again).

When: a round is packed once it is finished (STOP or every spec has its record or error), no round job of it is in
the queue, and, for a production run's round, its stream has ENDed and its decontamination list is published; or on
demand with --force (never while a round job is queued or running).

    python -m task_synthesis.pipeline.roundpack plan   --round r1             # dry run: what would be packed
    python -m task_synthesis.pipeline.roundpack submit --round r1 --delete    # CPU job from a frozen checkout
    python -m task_synthesis.pipeline.roundpack pack   --round r1 [--delete] [--force] [--shard-size 5000]
    python -m task_synthesis.pipeline.roundpack verify --round r1
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
import zlib

from synth import archive
from synth.common import atomic, read

from ..common import ROOT

ROUNDS = ROOT / 'rounds'
AREAS = ('work', 'tasks')


def folder(rd):
    return Path(rd) / 'packed'


def load_index(rd):
    return read(folder(rd) / 'index.json', None) or {'version': 1, 'shards': {}, 'specs': {}}


def round_dir(name_or_dir):
    p = Path(name_or_dir)
    return p if p.is_absolute() or p.exists() else ROUNDS / str(name_or_dir)


def outcome(record, error):
    """(status, judge pass) kept in the index, so `finished` and `accepted` need no shard read."""
    if record is not None:
        return record.get('status'), bool((record.get('judge') or {}).get('pass'))
    return ('error' if error is not None else None), False


class Packed(dict):
    """A round's pack index; `spec in` it asks whether the spec is packed."""
    def __contains__(self, sid):
        return sid in self['specs']


# ------------------------------------------------------------------------------------------------------------- read
class RoundFiles:
    """The files of one round, packed or not. A loose file is read first; if it is not there, the index is read
    (again) and the file comes from its shard's zip. Shards never change once indexed."""

    def __init__(self, rd):
        self.rd = round_dir(rd)
        self._index = None
        self._zips, self._names = {}, {}

    # ---- index
    def index(self, reload=False):
        if self._index is None or reload:
            self._index = Packed(load_index(self.rd))
        return self._index

    def is_packed(self):
        return bool(self.index()['specs'])

    def _locate(self, sid):
        index = self.index()
        if sid not in index:
            index = self.index(reload=True)   # packed since the last read
        return index['specs'].get(sid)

    def _zip(self, number):
        path = folder(self.rd) / self.index()['shards'][str(number)]['zip']
        z = self._zips.get(path)
        if z is None:
            z = self._zips[path] = zipfile.ZipFile(path)   # reads are thread-safe (zipfile locks the shared file)
            self._names[path] = set(z.namelist())
        return z, self._names[path]

    # ---- one spec
    def read(self, sid, rel, area='work'):
        """Bytes of <area>/<sid>/<rel>, loose or packed (FileNotFoundError: in neither)."""
        path = self.rd / area / sid / rel
        try:
            return path.read_bytes()
        except FileNotFoundError:
            pass
        loc = self._locate(sid)
        if loc is None:
            raise FileNotFoundError(path)
        z, names = self._zip(loc[0])
        name = f'{area}/{sid}/{rel}'
        if name not in names:
            raise FileNotFoundError(path)
        return z.read(name)

    def text(self, sid, rel, area='work'):
        return self.read(sid, rel, area).decode()

    def json(self, sid, rel, area='work', default=None):
        try:
            return json.loads(self.read(sid, rel, area))
        except FileNotFoundError:
            return default

    def exists(self, sid, rel, area='work'):
        if (self.rd / area / sid / rel).exists():
            return True
        loc = self._locate(sid)
        if loc is None:
            return False
        _, names = self._zip(loc[0])
        name = f'{area}/{sid}/{rel}'
        return name in names or f'{name}/' in names or any(n.startswith(f'{name}/') for n in names)

    def record(self, sid):
        """work/<sid>/record.json, or None."""
        return self.json(sid, 'record.json')

    def task_dir(self, sid):
        """tasks/<sid> as a loose directory, or None when it is packed (use copy_task)."""
        d = self.rd / 'tasks' / sid
        return d if d.is_dir() and (d / 'task.toml').exists() else None

    def copy_task(self, sid, dest):
        """Copy tasks/<sid> to `dest` (see copy_area)."""
        return self.copy_area(sid, 'tasks', dest)

    def copy_area(self, sid, area, dest):
        """Copy <area>/<sid> to `dest`: shutil.copytree(..., symlinks=True) of a loose directory (tasks/: one with its
        task.toml), else extracted from the zip with its symlinks, empty directories and file modes."""
        d = self.rd / area / sid
        if d.is_dir() and (sid not in self.index() or area != 'tasks' or (d / 'task.toml').exists()):
            try:
                shutil.copytree(d, dest, symlinks=True)
            except (shutil.Error, OSError):
                if sid not in self.index(reload=True):
                    raise
            if sid not in self.index(reload=True):   # not packed (so not deleted) while it was copied
                return Path(dest)
            shutil.rmtree(dest, ignore_errors=True)   # packed meanwhile: its copy may be partial; from the zip
        loc = self._locate(sid)
        if loc is None:
            raise FileNotFoundError(d)
        z, _ = self._zip(loc[0])
        prefix = f'{area}/{sid}/'
        members = [i for i in z.infolist() if i.filename.startswith(prefix)]
        if not members:
            raise FileNotFoundError(d)
        dest = Path(dest)
        dest.mkdir(parents=True)
        for info in members:
            target = dest / info.filename[len(prefix):]
            mode = info.external_attr >> 16
            if info.filename.endswith('/'):
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISLNK(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(z.read(info).decode(), target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
                if stat.S_IMODE(mode):
                    os.chmod(target, stat.S_IMODE(mode))
        return dest

    # ---- the whole round
    def specs(self):
        """Every spec id with a work/ or tasks/ directory, loose or packed (sorted)."""
        loose = set()
        for area in AREAS:
            try:
                loose |= set(os.listdir(self.rd / area))
            except FileNotFoundError:
                pass
        return sorted(loose | set(self.index(reload=True)['specs']))

    def has_outcome(self, sids):
        """Every one of `sids` has its record or error (production/bind/campaign `round finished` checks)."""
        index = self.index(reload=True)
        work = self.rd / 'work'
        for s in sids:
            loc = index['specs'].get(s)
            if loc is not None and loc[2] is not None:
                continue
            if (work / s / 'record.json').exists() or (work / s / 'error.json').exists():
                continue
            loc = self._locate(s)   # packed after the index read above, its directory deleted since
            if loc is None or loc[2] is None:
                return False
        return True

    def outcomes(self, workers=16, chunk=2048):
        """{spec: (record, error)} of every spec, read consistently: packed specs from their records lines, loose
        ones from work/<spec>/{record,error}.json in chunks with the index re-read after each (a spec packed during
        its read comes from its shard), then specs packed between the first index read and the listing. Sorted.
        A record.json or error.json that cannot be parsed raises (it used to be skipped or crash later)."""
        first = self.index(reload=True)
        got = {}
        for line in self._lines(first, first['shards']):
            got[line['spec_id']] = (line['record'], line['error'])
        try:
            loose = sorted(s for s in os.listdir(self.rd / 'work') if s not in first)
        except FileNotFoundError:
            loose = []

        def one(wd):
            out = []
            for name in ('record.json', 'error.json'):
                try:
                    out.append(json.loads((wd / name).read_text()))
                except FileNotFoundError:
                    out.append(None)
                except ValueError as e:
                    raise ValueError(f'{wd / name}: {e}') from e
            return tuple(out)
        late = set()
        for sid, value, error, index in archive.consistent_reads({s: self.rd / 'work' / s for s in loose}, one,
                                                                 lambda: self.index(reload=True), workers, chunk):
            if index is not None:
                late.add(sid)
            elif error is not None:
                raise OSError(f'{self.rd}/work/{sid}: {error}')
            else:
                got[sid] = value
        last = self.index(reload=True)
        late |= {s for s in last['specs'] if s not in first and s not in got}
        if late:
            for line in self._lines(last, {str(last['specs'][s][0]) for s in late}):
                if line['spec_id'] in late:
                    got[line['spec_id']] = (line['record'], line['error'])
        return dict(sorted(got.items()))

    def records(self, errors=False):
        """record.json of every spec in spec order (round.collect); with errors, error.json of specs without one."""
        out = []
        for record, error in self.outcomes().values():
            if record is not None:
                out.append(record)
            elif errors and error is not None:
                out.append(error)
        return out

    def _lines(self, index, numbers):
        for number in sorted({str(n) for n in numbers}, key=int):
            with gzip.open(folder(self.rd) / index['shards'][number]['records'], 'rt') as f:
                for text in f:
                    yield json.loads(text)


# ------------------------------------------------------------------------------------------------------------- pack
def spec_files(rd, sid):
    """{zip member name: path} of a spec's loose files (work/ and tasks/)."""
    out = {}
    for area in AREAS:
        base = Path(rd) / area / sid
        if not base.exists():
            continue
        for d, dirs, files in os.walk(base):
            links = [x for x in dirs if (Path(d) / x).is_symlink()]
            for f in files + links:
                p = Path(d) / f
                out[f'{area}/{sid}/{p.relative_to(base).as_posix()}'] = p
    return out


def pack_shard(job):
    """Write and verify one shard (zip + records lines) of specs; returns its index entry (built in node-local
    scratch, then copied to <round>/packed)."""
    rd, number, sids = job
    rd, out = Path(rd), folder(rd)
    zip_path, lines_path = out / f'round-{number:05d}.zip', out / f'records-{number:05d}.jsonl.gz'
    stage = Path(tempfile.mkdtemp(prefix=f'roundpack-{number:05d}-', dir=archive.stage_dir()))
    zip_tmp, lines_tmp = stage / zip_path.name, stage / lines_path.name
    expected, lines = {}, []
    with zipfile.ZipFile(zip_tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for sid in sids:
            parsed = {}
            for area in AREAS:
                base = rd / area / sid
                if base.is_dir() and not base.is_symlink():
                    archive.restore_access(base)
                    archive.add_tree(z, base, f'{area}/{sid}/', expected)
            for name in ('record', 'error'):
                p = rd / 'work' / sid / f'{name}.json'
                try:
                    parsed[name] = json.loads(p.read_text())
                except FileNotFoundError:
                    parsed[name] = None
                except ValueError:   # kept byte for byte in the zip; the line says it did not parse
                    parsed[name], parsed[f'{name}_unparsable'] = None, True
            lines.append({'spec_id': sid, **parsed})
    with gzip.open(lines_tmp, 'wt', compresslevel=6) as f:
        for line in lines:
            f.write(json.dumps(line) + '\n')
    archive.check_zip(zip_tmp, expected)
    with gzip.open(lines_tmp, 'rt') as f:
        if [json.loads(x)['spec_id'] for x in f] != list(sids):
            raise ValueError(f'{lines_tmp}: lines differ from the specs')
    archive.publish(zip_tmp, zip_path)
    archive.publish(lines_tmp, lines_path)
    shutil.rmtree(stage, ignore_errors=True)
    return {'shard': number, 'zip': zip_path.name, 'records': lines_path.name, 'files': len(expected),
            'bytes': sum(expected.values()), 'zip_bytes': zip_path.stat().st_size,
            'records_bytes': lines_path.stat().st_size, 'packed_at': time.time(),
            'specs': [[x['spec_id'], i, *outcome(x['record'], x['error'])] for i, x in enumerate(lines)]}


def remove_packed(rd, shard, sids):
    """Delete the loose files of a packed shard's specs. A file is deleted only when its zip member has the same
    size and CRC32 (read back from disk now); one that differs or is not in the zip stays (it is reported), and so
    do the directories that hold it. Returns (files deleted, files kept)."""
    rd = Path(rd)
    with zipfile.ZipFile(folder(rd) / shard['zip']) as z:
        members = {i.filename: i for i in z.infolist()}
    parents = set()   # every directory prefix the zip holds
    for name in members:
        parts = name.rstrip('/').split('/')
        parents.update('/'.join(parts[:k]) + '/' for k in range(2, len(parts) + (name.endswith('/'))))
    deleted, kept = 0, []
    for sid in sids:
        for name, p in sorted(spec_files(rd, sid).items()):
            info = members.get(name)
            if p.is_symlink():
                data = os.readlink(p).encode()
                same = info is not None and info.file_size == len(data) and info.CRC == zlib.crc32(data)
            else:
                try:
                    same = info is not None and info.file_size == p.stat().st_size and info.CRC == crc32(p)
                except OSError:
                    same = False
            if same:
                p.unlink()
                deleted += 1
            else:
                kept.append(name)
        for area in AREAS:   # directories the zip holds (as entries or by their files), deepest first; one that
            base = rd / area / sid   # still holds a kept file stays
            if not base.is_dir() or base.is_symlink():
                continue
            for d, _, _ in sorted(os.walk(base), key=lambda x: -len(Path(x[0]).parts)):
                rel = Path(d).relative_to(base).as_posix()
                if (f'{area}/{sid}/' if rel == '.' else f'{area}/{sid}/{rel}/') in parents:
                    try:
                        os.rmdir(d)
                    except OSError:
                        pass
    return deleted, kept


def crc32(path, block=1 << 20):
    value = 0
    with open(path, 'rb') as f:
        while chunk := f.read(block):
            value = zlib.crc32(chunk, value)
    return value


def live_round_job(rd):
    """The round's condor job id if it is still queued or running (round.submit writes <round>/job.json), else None."""
    job = read(Path(rd) / 'job.json', None) or {}
    if not job.get('job_id'):
        return None
    r = subprocess.run(['condor_q', str(job['job_id']).split('.')[0], '-af', 'JobStatus'], capture_output=True,
                       text=True)
    if r.returncode != 0:
        return f'{job["job_id"]} (condor_q failed: cannot tell)'
    return job['job_id'] if r.stdout.split() else None


def production_run_of(rd):
    """The production run (run.json) whose round this is, or None."""
    for p in sorted((ROOT / 'production').glob('*/run.json')):
        info = read(p, {}) or {}
        if info.get('round') == Path(rd).name:
            return p.parent, info
    return None


def ready(rd):
    """Why the round may not be packed yet ([] = ready): a live round job; not finished; for a production run's
    round, a stream without END.json or no published decontamination list (or a decontam hold still on)."""
    rd, why = Path(rd), []
    if (job := live_round_job(rd)):
        why.append(f'round job {job} is queued or running')
    files = RoundFiles(rd)
    if not (rd / 'STOP').exists():
        try:
            sids = [json.loads(l)['spec_id'] for l in (rd / 'specs.jsonl').read_text().splitlines() if l.strip()]
        except OSError:
            sids = None
        if sids is None or not files.has_outcome(sids):
            why.append('not finished (no STOP, and not every spec has its record or error)')
    run = production_run_of(rd)
    if run:
        path, info = run
        if not (Path(info['stream']) / 'END.json').exists():
            why.append(f'stream {info["stream"]} has not ENDed')
        if not (ROOT / 'decon' / path.name / 'decontamination-variants.json').exists():
            why.append(f'decontamination list {ROOT / "decon" / path.name}/decontamination-variants.json not published')
        if (path / 'decontam.HOLD').exists():
            why.append(f'{path}/decontam.HOLD is on')
    return why


def plan(rd, shard_size=5000):
    """(index, shard jobs, files to pack): the specs with loose work/ or tasks/ directories not in the index."""
    rd = Path(rd)
    out = folder(rd)
    index = load_index(rd)
    loose = set()
    for area in AREAS:
        try:
            loose |= {s for s in os.listdir(rd / area) if (rd / area / s).is_dir()}
        except FileNotFoundError:
            pass
    todo = sorted(s for s in loose if s not in index['specs'])
    first = max((int(k) for k in index['shards']), default=-1) + 1
    jobs = [(str(rd), first + i, todo[o:o + shard_size]) for i, o in enumerate(range(0, len(todo), shard_size))]
    return index, jobs, out


def pack(rd, shard_size=5000, workers=8, delete=False, force=False, log=print):
    """Pack a round (see the module doc). Resumable: specs already in the index are skipped; with delete, shards
    packed earlier and not yet deleted are deleted (after the per-file check)."""
    rd = round_dir(rd)
    why = ready(rd)
    blocking = [w for w in why if w.startswith('round job')]
    if blocking or (why and not force):
        raise SystemExit(f'{rd}: not packing: {"; ".join(why)}' + ('' if blocking else ' (--force packs on demand)'))
    if why:
        log(f'{rd.name}: packing on demand although: {"; ".join(why)}')
    index, jobs, out = plan(rd, shard_size)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob('*.partial'):
        stale.unlink()
    log(f'{rd.name}: {sum(len(j[2]) for j in jobs)} specs to pack in {len(jobs)} shards'
        + (f'; {len(index["specs"])} packed earlier' if index['specs'] else ''))
    deleter, pending, failures, totals = ThreadPoolExecutor(8) if delete else None, {}, [], {'deleted': 0, 'kept': []}

    def reap(wait=False):
        for future in [f for f in pending if wait or f.done()]:
            number = pending.pop(future)
            deleted, kept = future.result()
            index['shards'][number]['deleted_at'] = time.time()
            index['shards'][number]['kept_files'] = kept[:1000]
            atomic(out / 'index.json', index)
            totals['deleted'] += deleted
            totals['kept'] += kept
            log(f'  deleted shard {int(number):05d}: {deleted} files' + (f', {len(kept)} kept (differ from the zip '
                                                                         f'or not in it), e.g. {kept[:3]}' if kept else ''))

    def schedule(number):
        sids = [s for s, v in index['specs'].items() if v[0] == int(number)]
        pending[deleter.submit(remove_packed, rd, index['shards'][number], sids)] = number

    if delete:
        for number, shard in sorted(index['shards'].items(), key=lambda kv: int(kv[0])):
            if not shard.get('deleted_at'):
                schedule(number)
    with ProcessPoolExecutor(max(1, workers), mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(pack_shard, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                entry = future.result()
            except Exception as exc:   # this shard's specs stay loose; the others go on
                failures.append(exc)
                log(f'  FAILED shard {futures[future][1]:05d}: {exc!r}')
                continue
            number = str(entry['shard'])
            index['shards'][number] = {k: v for k, v in entry.items() if k != 'specs'}
            for sid, line, status, passed in entry['specs']:
                index['specs'][sid] = [entry['shard'], line, status, passed]
            index['round'] = rd.name
            atomic(out / 'index.json', index)
            log(f'  shard {entry["shard"]:05d}: {len(entry["specs"])} specs, {entry["files"]} files, '
                f'{entry["bytes"] / 2**20:.0f} MiB -> {(entry["zip_bytes"] + entry["records_bytes"]) / 2**20:.0f} MiB')
            if delete:
                schedule(number)
            reap()
    if deleter is not None:
        reap(wait=True)
        deleter.shutdown()
    if failures:
        raise RuntimeError(f'{len(failures)} shards failed (their specs stay loose): {failures[0]!r}')
    return {'shards': len(index['shards']), 'specs': len(index['specs']), 'deleted': totals['deleted'],
            'kept': len(totals['kept'])}


def verify(rd):
    """Re-check every shard: zip CRCs and member count, and one records line per indexed spec."""
    rd = round_dir(rd)
    out, index = folder(rd), load_index(rd)
    for shard in index['shards'].values():
        with zipfile.ZipFile(out / shard['zip']) as z:
            if z.testzip() is not None or len(z.infolist()) != shard['files']:
                raise ValueError(f'{shard["zip"]}: verification failed')
        with gzip.open(out / shard['records'], 'rt') as f:
            n = sum(1 for _ in f)
        if n != sum(1 for v in index['specs'].values() if v[0] == shard['shard']):
            raise ValueError(f'{shard["records"]}: line count differs from the index')
    return len(index['shards']), len(index['specs'])


def survey(rd, shard_size=5000, workers=16):
    """Dry run: what a pack would do (specs, shards, loose files and bytes it would pack; nothing is written)."""
    rd = round_dir(rd)
    index, jobs, _ = plan(rd, shard_size)
    sids = [s for j in jobs for s in j[2]]

    def count(sid):
        files = spec_files(rd, sid)
        size = 0
        for p in files.values():
            try:
                size += os.lstat(p).st_size
            except OSError:
                pass
        return len(files), size
    with ThreadPoolExecutor(workers) as pool:
        counted = list(pool.map(count, sids))
    files, size = sum(c[0] for c in counted), sum(c[1] for c in counted)
    return {'round': rd.name, 'ready': not ready(rd), 'not_ready': ready(rd), 'specs_packed': len(index['specs']),
            'specs_to_pack': len(sids), 'shards': len(jobs), 'loose_files_to_pack': files,
            'loose_bytes_to_pack': size, 'files_after': 2 * len(jobs) + (0 if index['shards'] else 1)}


def submit(a):
    """A CPU job running `roundpack pack` from a frozen checkout (node-local scratch stages the shards)."""
    from ..common import frozen_checkout
    from .production import BAD_CPU_NODES
    rd = round_dir(a.round)
    code = frozen_checkout()
    args = f'pack --round {rd} --shard-size {a.shard_size} --workers {a.workers}' + \
        (' --delete' if a.delete else '') + (' --force' if a.force else '')
    sub = rd / 'roundpack.sub'
    sub.write_text(f'''universe = vanilla
executable = /opt/aai/venvs/harbor/bin/python
arguments = -u -m task_synthesis.pipeline.roundpack {args}
initialdir = {code}
output = {rd}/roundpack.$(Cluster).out
error = {rd}/roundpack.$(Cluster).err
log = {rd}/roundpack.condor.log
request_cpus = {a.workers}
request_memory = {max(8, 2 * a.workers) * 1024}
request_disk = {a.disk_gib}G
+JobBatchName = "roundpack-{rd.name}"
+BypassLXCfs = "true"
getenv = false
environment = "HOME=/home/user USER=user PATH=/usr/local/bin:/usr/bin:/bin PYTHONPATH={code} PYTHONNOUSERSITE=1"
requirements = {' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in BAD_CPU_NODES)}
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    job = re.search(r'cluster (\d+)', r.stdout).group(1) + '.0'
    atomic(rd / 'roundpack.json', {'job_id': job, 'code': str(code), 'time': time.time(), 'args': vars(a)})
    print(job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='op', required=True)
    for op in ('plan', 'pack', 'submit', 'verify'):
        p = sub.add_parser(op)
        p.add_argument('--round', required=True, help='round name or directory')
        if op != 'verify':
            p.add_argument('--shard-size', type=int, default=5000)
            p.add_argument('--workers', type=int, default=8)
        if op in ('pack', 'submit'):
            p.add_argument('--delete', action='store_true', help='remove loose files of verified shards')
            p.add_argument('--force', action='store_true', help='pack on demand before the round is ready '
                                                                '(never while a round job is queued or running)')
        if op == 'submit':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--disk-gib', type=int, default=50)
    a = ap.parse_args()
    if a.op == 'plan':
        print(json.dumps(survey(a.round, a.shard_size, a.workers), indent=1))
    elif a.op == 'pack':
        os.nice(10)
        print(json.dumps(pack(a.round, a.shard_size, a.workers, a.delete, a.force,
                              log=lambda m: print(m, flush=True))))
    elif a.op == 'submit':
        submit(a)
    else:
        print('shards, specs:', verify(a.round))


if __name__ == '__main__':
    main()
