"""Judge-passed tasks of audit rounds -> a synth campaign task bundle (synth.prepare's train.bundle_source), or chunks
of a task stream that a running campaign takes as they come ([train.stream], synth/stream.py).

  python -m task_synthesis.pipeline.campaign bundle --rounds r4 --name r4
  python -m task_synthesis.pipeline.campaign stream --rounds r7a r7b --stream <dir> --exclude s1 s2 --follow --end

Takes judge-passed tasks without decon flags or near-duplicates (the round's summary.json). Writes
<ROOT>/campaigns/<name>/: tasks.tar (tasks/tasksynth-<spec id>/...), tasks.tar.tasks.txt (sorted),
order.txt (seeded shuffle, the campaign's train.tasks), smoke.json (two short tasks without services for [smoke])
and manifest.json (task -> round, spec, slug, minutes). The tar is built in node-local scratch and copied to /data.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from synth.common import atomic
from synth.datasets import bundle

from ..common import ROOT
from .round import ROUNDS
from .roundpack import RoundFiles
from .textsim import near_duplicates_fast
from tbshared.fastio import copy_direct  # noqa: E402  (the shared layer, tbshared/README.md)

CAMPAIGNS = ROOT / 'campaigns'


def passed(round_name):
    """Judge-passed tasks of a round, minus decon-flagged ones and the second task of each near-duplicate pair
    (from the round's summary.json, so run `round report` first)."""
    summary = json.loads((ROUNDS / round_name / 'summary.json').read_text())
    if 'decon_flagged_ids' not in summary:
        raise ValueError(f'{round_name}: summary.json predates decon ids; rerun `round report --round {round_name}`')
    if summary.get('near_duplicates_skipped'):
        raise ValueError(f'{round_name}: the round job skipped near-duplicates; run `round report --round {round_name}`')
    drop = set(summary['decon_flagged_ids']) | {b for _, b, _ in summary['near_duplicates']}
    for rec in RoundFiles(ROUNDS / round_name).records():
        if rec['status'] == 'valid' and rec.get('judge', {}).get('pass') and rec['spec_id'] not in drop:
            yield rec


def build(rounds, name, seed=0, exclude=()):
    """exclude: earlier bundles; their tasks are left out, and new tasks near-duplicating them are dropped."""
    out = CAMPAIGNS / name
    out.mkdir(parents=True, exist_ok=True)
    taken = {}
    for b in exclude:
        for m in json.loads((CAMPAIGNS / b / 'manifest.json').read_text()).values():
            taken[m['spec_id']] = m['round']
    manifest, recs = {}, [(r, rec) for r in rounds for rec in passed(r) if rec['spec_id'] not in taken]
    # near-duplicates across rounds (each round's report only checks within the round) and against the excluded
    # bundles: keep the earlier bundle's task, else the first of a pair
    texts = {rec['spec_id']: rec['instruction'] for _, rec in recs}
    for sid, r in taken.items():
        texts[sid] = RoundFiles(ROUNDS / r).record(sid)['instruction']
    dups = near_duplicates_fast(texts) if len(rounds) > 1 or taken else []
    drop = {b if b not in taken else a for a, b, _ in dups} - set(taken)
    if dups:
        print(f'{len(dups)} near-duplicate pairs; dropping {len(drop)} tasks')
    with tempfile.TemporaryDirectory(prefix='tasksynth-bundle-') as tmp:
        stage = Path(tmp) / 'tasks'
        stage.mkdir()
        for r, rec in recs:
            if rec['spec_id'] in drop:
                continue
            task = f'tasksynth-{rec["spec_id"]}'
            RoundFiles(ROUNDS / r).copy_task(rec['spec_id'], stage / task)
            manifest[task] = {'round': r, 'spec_id': rec['spec_id'], 'slug': rec['meta'].get('slug'),
                              'minutes': rec['meta'].get('expert_minutes') or rec['spec'].get('minutes'),
                              'services': bool(rec['meta'].get('services'))}
        tar = Path(tmp) / 'tasks.tar'
        bundle(stage, tar)
        for f in (tar, tar.with_name('tasks.tar.tasks.txt')):   # O_DIRECT: no dirty page cache left behind (#389)
            copy_direct(f, out / f'{f.name}.partial')
            (out / f'{f.name}.partial').replace(out / f.name)
    names = sorted(manifest)
    random.Random(seed).shuffle(names)
    (out / 'order.txt').write_text('\n'.join(names) + '\n')
    short = sorted((m['minutes'] or 999, t) for t, m in manifest.items() if not m['services'])
    atomic(out / 'smoke.json', [t for _, t in short[:2]])
    atomic(out / 'manifest.json', manifest)
    print(f'{out}: {len(manifest)} tasks from {", ".join(rounds)}; smoke {[t for _, t in short[:2]]}')


def build_cohorts(cohorts, name, seed=0, exclude=()):
    """Valid, judge-passed tasks of generator cohorts (task_synthesis.pipeline.cohort) -> a campaign bundle. Tasks are
    named tasksynth-<cohort>-<spec id> (the round's own task for the same spec is tasksynth-<spec id>). Eval-set
    decontamination and near-duplicates within the cohorts are checked here (cohorts have no round report)."""
    from .cohort import COHORTS
    from .textsim import decon
    # tasks of earlier bundles stay out, except ones blocked there (`exclude` entries name:blocked_tasks.json)
    taken = set()
    for b in exclude:
        b, _, blocked = b.partition(':')
        skip = set(json.loads(Path(blocked).read_text())) if blocked else set()
        taken |= set(json.loads((CAMPAIGNS / b / 'manifest.json').read_text())) - skip
    recs = []
    for c in cohorts:
        for p in sorted((COHORTS / c / 'records').glob('s*.json')):
            if p.name.endswith('.check.json'):
                continue
            r = json.loads(p.read_text())
            if r['status'] == 'valid' and (r.get('judge') or {}).get('pass') and \
                    f'tasksynth-{c}-{r["spec_id"]}' not in taken:
                recs.append((c, r))
    texts = {f'{c}/{r["spec_id"]}': r['instruction'] for c, r in recs}
    flagged = {k for k, v in decon(texts).items() if v['flagged']}
    dups = near_duplicates_fast(texts)
    drop = flagged | {b for _, b, _ in dups}
    print(f'{len(recs)} judge-passed cohort tasks; decon-flagged {len(flagged)}, near-duplicate pairs {len(dups)}')
    out = CAMPAIGNS / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with tempfile.TemporaryDirectory(prefix='tasksynth-bundle-') as tmp:
        stage = Path(tmp) / 'tasks'
        stage.mkdir()
        for c, r in recs:
            if f'{c}/{r["spec_id"]}' in drop:
                continue
            task = f'tasksynth-{c}-{r["spec_id"]}'
            src = COHORTS / c / 'tasks' / r['spec_id']
            shutil.copytree(src, stage / task, symlinks=True)
            manifest[task] = {'cohort': c, 'spec_id': r['spec_id'], 'dir': str(src), 'slug': r['meta'].get('slug'),
                              'minutes': r['meta'].get('expert_minutes'), 'services': bool(r['meta'].get('services'))}
        tar = Path(tmp) / 'tasks.tar'
        bundle(stage, tar)
        for f in (tar, tar.with_name('tasks.tar.tasks.txt')):   # O_DIRECT: no dirty page cache left behind (#389)
            copy_direct(f, out / f'{f.name}.partial')
            (out / f'{f.name}.partial').replace(out / f.name)
    names = sorted(manifest)
    random.Random(seed).shuffle(names)
    (out / 'order.txt').write_text('\n'.join(names) + '\n')
    short = sorted((m['minutes'] or 999, t) for t, m in manifest.items() if not m['services'])
    atomic(out / 'smoke.json', [t for _, t in short[:2]])
    atomic(out / 'manifest.json', manifest)
    print(f'{out}: {len(manifest)} tasks from {", ".join(cohorts)}; smoke {[t for _, t in short[:2]]}')


def judge_passed(round_name):
    """Valid, judge-passed tasks of a round as their records appear (a running round has no summary.json yet)."""
    for rec in RoundFiles(ROUNDS / round_name).records():
        if rec['status'] == 'valid' and (rec.get('judge') or {}).get('pass'):
            yield rec


def round_done(round_name):
    """Every spec of the round has its record (or error). Tasks that pass only later (`round rejudge` of judge errors,
    `round run --resume` of infrastructure failures) need another `campaign stream` run before the stream ends."""
    d = ROUNDS / round_name
    if not (d / 'specs.jsonl').exists():
        return False
    sids = [json.loads(line)['spec_id'] for line in (d / 'specs.jsonl').read_text().splitlines() if line.strip()]
    return RoundFiles(d).has_outcome(sids)


class StreamProducer:
    """Judge-passed tasks of audit rounds -> chunks of a synth task stream (synth/stream.py), while the rounds run.

    Each publish takes the rounds' judge-passed tasks that are not in the stream yet. A new task is dropped when the
    decontamination check against TB2.1 + TB-Lite flags it, or when it near-duplicates a task published earlier, a
    task of an `exclude` bundle, or an earlier task of the same chunk (a round's report checks only within the round,
    and only at its end). Dropped tasks are listed in <stream>/producer/rejected.json and not reconsidered. The
    published tasks' metadata and instructions are kept with each chunk (chunks/NNNN.meta.json), so the producer's
    view is rebuilt from the stream itself; <stream>/manifest.json (task -> round, spec, ...) and smoke.json mirror
    a campaign bundle's."""

    def __init__(self, rounds, stream_dir, seed=0, exclude=()):
        from synth import stream as chunks
        self.chunks, self.rounds, self.dir, self.seed = chunks, list(rounds), Path(stream_dir), seed
        self.metas = {}      # chunk index -> its meta.json
        self.excluded = {}   # spec id (cohort tasks: cohort/spec id) -> instruction of the excluded bundles' tasks
        for b in exclude:
            for m in json.loads((CAMPAIGNS / b / 'manifest.json').read_text()).values():
                key = f'{m["cohort"]}/{m["spec_id"]}' if m.get('cohort') else m['spec_id']
                self.excluded[key] = (Path(m['dir']) / 'instruction.md').read_text() if m.get('dir') else \
                    RoundFiles(ROUNDS / m['round']).text(m['spec_id'], 'instruction.md', 'tasks')

    def published(self):
        """task -> meta (round, spec_id, slug, minutes, services, instruction) of every committed chunk."""
        out = {}
        for record in self.chunks.committed(self.dir):
            if record['index'] not in self.metas:
                if not record.get('meta'):
                    raise ValueError(f'{self.dir} chunk {record["index"]} has no task metadata (another producer?)')
                self.metas[record['index']] = json.loads((self.dir / record['meta']).read_text())
            out.update(self.metas[record['index']])
        return out

    def publish(self, min_tasks=1, final=False):
        """The next chunk, or None (no new valid task, or fewer than min_tasks of them and not final)."""
        from .textsim import decon, near_duplicates_new
        published = self.published()
        rejected_path = self.dir / 'producer' / 'rejected.json'
        rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else {}
        taken = set(self.excluded) | {m['spec_id'] for m in published.values()}
        # decon and near-duplicate verdicts stand; an invalid task is checked again (its directory may be repaired)
        standing = {t for t, v in rejected.items() if v['reason'] != 'invalid_task'}
        new = [(r, rec) for r in self.rounds for rec in judge_passed(r)
               if rec['spec_id'] not in taken and f'tasksynth-{rec["spec_id"]}' not in standing]
        invalid = {}   # a task campaigns could not admit (e.g. a task.toml emptied by overlapping jobs, 2026-10-31)
        for r, rec in new:
            files = RoundFiles(ROUNDS / r)
            if not (files.exists(rec['spec_id'], 'task.toml', 'tasks')
                    and files.exists(rec['spec_id'], 'instruction.md', 'tasks')):
                invalid[rec['spec_id']] = 'no task.toml or instruction.md'
            elif error := self.chunks.invalid_task(files.read(rec['spec_id'], 'task.toml', 'tasks')):
                invalid[rec['spec_id']] = error
        invalid_entries = {f'tasksynth-{s}': {'reason': 'invalid_task', 'error': e, 'time': time.time()}
                           for s, e in invalid.items()}
        new_texts = {rec['spec_id']: rec['instruction'] for _, rec in new if rec['spec_id'] not in invalid}
        if not new_texts or (len(new_texts) < min_tasks and not final):
            # wait; record newly invalid tasks only (no decon or near-duplicate pass, which costs minutes at scale)
            if set(invalid_entries) - set(rejected):
                atomic(rejected_path, {**invalid_entries, **rejected})
            return None
        for sid in new_texts:   # repaired since an earlier publish
            rejected.pop(f'tasksynth-{sid}', None)
        flagged = {k for k, v in decon(new_texts).items() if v['flagged']}
        texts = {**self.excluded, **{m['spec_id']: m['instruction'] for m in published.values()},
                 **{k: v for k, v in new_texts.items() if k not in flagged}}
        pos = {k: i for i, k in enumerate(texts)}
        drop = {}   # a new task near-duplicating one that is kept (published, excluded, or earlier and kept here)
        for a, b, c in sorted(near_duplicates_new(texts, [k for k in new_texts if k not in flagged]),
                              key=lambda p: (pos[p[1]], pos[p[0]])):
            if a not in drop and b not in drop:
                drop[b] = (a, c)
        rejected.update(invalid_entries)
        rejected.update({f'tasksynth-{s}': {'reason': 'decon', 'time': time.time()} for s in flagged})
        rejected.update({f'tasksynth-{b}': {'reason': 'near_duplicate', 'of': a, 'cosine': c, 'time': time.time()}
                         for b, (a, c) in drop.items()})
        atomic(rejected_path, rejected)
        keep = [(r, rec) for r, rec in new if rec['spec_id'] in new_texts and rec['spec_id'] not in flagged
                and rec['spec_id'] not in drop]
        print(f'{len(new)} new judge-passed tasks; invalid {len(invalid)}, decon-flagged {len(flagged)}, '
              f'near-duplicates {len(drop)}, publishing {len(keep)}', flush=True)
        if not keep:
            return None
        meta = {}
        with tempfile.TemporaryDirectory(prefix='tasksynth-stream-') as tmp:
            stage = Path(tmp) / 'tasks'
            stage.mkdir()
            for r, rec in keep:
                task = f'tasksynth-{rec["spec_id"]}'
                RoundFiles(ROUNDS / r).copy_task(rec['spec_id'], stage / task)
                meta[task] = {'round': r, 'spec_id': rec['spec_id'], 'slug': rec['meta'].get('slug'),
                              'minutes': rec['meta'].get('expert_minutes') or rec['spec'].get('minutes'),
                              'services': bool(rec['meta'].get('services')), 'instruction': rec['instruction']}
            names = sorted(meta)
            index = len(self.chunks.committed(self.dir))
            random.Random(self.seed * 100003 + index).shuffle(names)
            record = self.chunks.publish(self.dir, stage, names, meta=meta, source={
                'producer': 'task_synthesis.pipeline.campaign stream', 'rounds': sorted({r for r, _ in keep}),
                'invalid': len(invalid), 'decon_flagged': len(flagged), 'near_duplicates_dropped': len(drop)})
        manifest = {t: {k: v for k, v in m.items() if k != 'instruction'} for t, m in {**published, **meta}.items()}
        atomic(self.dir / 'manifest.json', manifest)
        if not (self.dir / 'smoke.json').exists():
            short = sorted((m['minutes'] or 999, t) for t, m in meta.items() if not m['services'])
            atomic(self.dir / 'smoke.json', [t for _, t in short[:2]])
        print(f'{self.dir}: chunk {record["index"]}, {record["count"]} tasks from {", ".join(record["source"]["rounds"])}',
              flush=True)
        return record

    def follow(self, interval=600, min_tasks=50, end=False):
        """Publish every `interval` seconds while the rounds run (chunks of at least min_tasks), then what is left once
        every round is done, then END.json if `end`."""
        while True:
            try:   # a heartbeat per pass (production status, #237): a chunk is published only when tasks are ready
                (self.dir / 'producer').mkdir(parents=True, exist_ok=True)
                atomic(self.dir / 'producer' / 'HEARTBEAT', {'time': time.time(), 'pid': os.getpid(),
                                                             'interval': interval})
            except OSError:
                pass
            try:
                done = all(round_done(r) for r in self.rounds)
                self.publish(min_tasks, final=done)
            except OSError as e:   # a transient Lustre EIO: retry the whole publish (skipping a record would drop its
                print(f'{self.dir}: publish failed ({e}); retrying in {interval}s', flush=True)   # task for good at END)
                time.sleep(interval)
                continue
            if done:
                if end:
                    print(json.dumps(self.chunks.end(self.dir, f'rounds {" ".join(self.rounds)} done')), flush=True)
                return
            time.sleep(interval)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('bundle')
    p.add_argument('--rounds', nargs='+', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude', nargs='*', default=[], help='earlier bundle names (their tasks are left out)')
    p = sub.add_parser('stream', help='publish judge-passed round tasks to a synth task stream ([train.stream])')
    p.add_argument('--rounds', nargs='+', required=True)
    p.add_argument('--stream', required=True, help='stream directory (synth/stream.py)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude', nargs='*', default=[], help='earlier bundle names (their tasks are left out)')
    p.add_argument('--min-tasks', type=int, default=1, help='publish only once this many new tasks are ready')
    p.add_argument('--follow', action='store_true', help='keep publishing while the rounds run')
    p.add_argument('--interval', type=int, default=600, help='--follow: seconds between publishes')
    p.add_argument('--end', action='store_true', help='close the stream (END.json) after the last publish '
                   '(--follow: once every spec has a record; tasks that pass later, after `round rejudge` or '
                   '`round run --resume`, then need a new stream, so rejudge first or end by hand)')
    p = sub.add_parser('bundle-cohorts')
    p.add_argument('--cohorts', nargs='+', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude', nargs='*', default=[],
                   help='earlier bundles (name, or name:blocked-tasks.json to let that campaign\'s blocked tasks back in)')
    a = ap.parse_args()
    if a.op == 'bundle-cohorts':
        return build_cohorts(a.cohorts, a.name, a.seed, a.exclude)
    if a.op == 'stream':
        producer = StreamProducer(a.rounds, a.stream, a.seed, a.exclude)
        if a.follow:
            return producer.follow(a.interval, a.min_tasks, a.end)
        producer.publish(a.min_tasks, final=a.end)
        if a.end:
            print(json.dumps(producer.chunks.end(producer.dir, 'closed by hand')))
        return
    build(a.rounds, a.name, a.seed, a.exclude)


if __name__ == '__main__':
    main()
