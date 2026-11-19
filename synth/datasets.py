"""External task datasets -> Harbor task bundles for synth campaigns.

  python -m synth.datasets tmax-base <base_intricate.def> <context_dir>
      build context of TMax's shared "intricate" base (its %post as one layer)
  python -m synth.datasets register-base <context_dir> <ref> [--submit]
      build that context with harbor_ns (in a CPU job with --submit) and publish the image as the
      shared harbor_ns base <ref>, so task Dockerfiles `FROM <ref>` reuse it instead of rebuilding
  python -m synth.datasets bundle <harbor_dir> <out.tar>
      tasks/<name>/... tarball (synth.prepare's bundle_source) + <out>.tasks.txt

TMax tasks come from TMax's own rl_data/scripts/analyze/convert_to_harbor.py
(--intricate-base-image tmax/base-intricate:1.0); CalibForge tasks are already Harbor tasks.
"""
import argparse
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

from . import condor
from .common import PROXY_ENV, PYTHON, REPO, atomic

HARBOR_NS_SHARED = Path('/data/aai/tbench/harbor_ns')


def section(text, name):
    m = re.search(rf'^%{name}\b[^\n]*\n(.*?)(?=^%\w+|\Z)', text, re.S | re.M)
    return m.group(1).rstrip() if m else ''


def tmax_base(definition, context):
    """Dockerfile + base_install.sh for base_intricate.def; its %environment goes to base.env.json."""
    text = Path(definition).read_text()
    post = re.sub(r'^\s*export\s+DEBIAN_FRONTEND=noninteractive\s*\n?', '', section(text, 'post'), count=1)
    env = {}
    path = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'  # ubuntu:22.04
    for line in section(text, 'environment').splitlines():
        m = re.match(r'\s*export\s+([A-Za-z_]\w*)=(.*)$', line)
        if m:
            value = m.group(2).strip().replace('$PATH', path)
            env[m.group(1)] = value
            if m.group(1) == 'PATH':
                path = value
    context = Path(context)
    context.mkdir(parents=True, exist_ok=True)
    (context / 'base_install.sh').write_text(post + '\n')
    lines = ['FROM ubuntu:22.04', 'ENV DEBIAN_FRONTEND=noninteractive']
    lines += [f'ENV {k}={v}' for k, v in env.items()]
    lines += ['COPY base_install.sh /tmp/base_install.sh', 'RUN bash /tmp/base_install.sh && rm /tmp/base_install.sh', '']
    (context / 'Dockerfile').write_text('\n'.join(lines))
    atomic(context / 'base.env.json', env)
    print(f'{context}: {len(post.splitlines())} %post lines, env {sorted(env)}')


def safe_name(ref):
    from harbor_ns.builder import _safe_name
    return _safe_name(ref)


def register_base(context, ref):
    """Build `context` with harbor_ns, then publish its squashfs + metadata as shared base `ref`."""
    import asyncio
    from harbor_ns.cli import default_store
    store = default_store()
    key, rootfs, meta = asyncio.run(store.build(Path(context), label=f'base:{ref}'))
    image = HARBOR_NS_SHARED / 'images' / f'{key}.sqfs'
    for _ in range(120):  # the build persists its squashfs to the shared store
        if image.exists() and image.with_suffix('.json').exists():
            break
        time.sleep(5)
    target = HARBOR_NS_SHARED / 'bases' / f'{safe_name(ref)}.sqfs'
    record = json.loads(image.with_suffix('.json').read_text())
    record.update(base=ref, oci=True)  # oci: no registry lookup for this local-only ref
    record['env'] = {**record.get('env', {}), **json.loads((Path(context) / 'base.env.json').read_text())}
    # A built image keeps nsfake's user.harborns.owner xattrs, and unsquashing those onto read-only
    # files (postfix's postdrop) fails without root; bases pulled from a registry have neither.
    work = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / 'synth-base'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    subprocess.run(['unsquashfs', '-q', '-no-xattrs', '-d', str(work / 'rootfs'), str(image)], check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run(['chmod', '-R', 'u+rwX', str(work / 'rootfs')], check=True)
    portable = work / 'base.sqfs'
    subprocess.run(['mksquashfs', str(work / 'rootfs'), str(portable), '-comp', 'zstd', '-no-xattrs', '-quiet',
                    '-no-progress', '-processors', '8'], check=True)
    tmp = target.with_name(f'.{target.name}.{os.getpid()}')
    subprocess.run(['dd', f'if={portable}', f'of={tmp}', 'bs=64M', 'oflag=direct', 'conv=fsync', 'status=none'], check=True)
    tmp.replace(target)
    atomic(target.with_suffix('.json'), record)
    print(json.dumps({'ref': ref, 'image_key': key, 'base': str(target), 'env': record['env']}))


def submit_register(context, ref, bid=56):
    context = Path(context).resolve()
    env = {'USER': 'user', 'HOME': os.environ.get('HOME', '/home/user'), 'PYTHONNOUSERSITE': '1',
           'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONPATH': str(REPO), **PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    sub = context / 'register.sub'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m synth.datasets register-base {context} {ref}
initialdir = {REPO}
output = {context}/register.$(Cluster).out
error = {context}/register.$(Cluster).err
log = {context}/register.condor.log
request_cpus = 8
request_memory = 32768
request_disk = 120G
+JobBatchName = "synth-base-{safe_name(ref)}"
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {condor.requirements({"gpu132"} | condor.bad_nodes(), False)}
queue
''')
    print(subprocess.run(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, check=True).stdout.strip())


WORDS = re.compile(r'\w+')
CODE = re.compile(r'\w+|[^\w\s]')
FIELDS = {'instruction': (WORDS, 5), 'tests': (CODE, 10), 'dockerfile': (CODE, 8), 'env': (CODE, 10)}
# seeds/METHODOLOGY.md section 6.3 lexical rules (the embedding rules need a model and are not used).
FLAGS = {'tests': 0.15, 'env': 0.30, 'dockerfile': 0.50, 'instruction': 0.10}
MAX_FILE_BYTES = 1 << 20


def task_fields(task_dir):
    """Text per decontamination field and the hashes of files >= 300 bytes."""
    task_dir = Path(task_dir)
    text = {k: [] for k in FIELDS}
    blobs = set()
    text['instruction'].append((task_dir / 'instruction.md').read_text(errors='replace'))
    for path in sorted(task_dir.rglob('*')):
        if not path.is_file() or path.name in ('instruction.md', 'task.toml'):
            continue
        rel = path.relative_to(task_dir).parts
        size = path.stat().st_size
        data = path.read_bytes()[:MAX_FILE_BYTES]
        if size >= 300:
            blobs.add(hashlib.sha1(data).hexdigest())
        if b'\0' in data[:4096]:
            continue
        field = ('tests' if rel[0] == 'tests' else
                 'dockerfile' if rel[-1] in ('Dockerfile', 'docker-compose.yaml', 'docker-compose.yml') else
                 'env' if rel[0] == 'environment' else None)
        if field:
            text[field].append(data.decode(errors='replace'))
    return text, blobs


def shingles(text, field):
    pattern, n = FIELDS[field]
    tokens = pattern.findall(text.lower() if field == 'instruction' else text)
    # Deterministic across worker processes (str hash() is salted per process).
    return {int.from_bytes(hashlib.blake2b(' '.join(tokens[i:i + n]).encode(), digest_size=8).digest(), 'little')
            for i in range(len(tokens) - n + 1)}


def decontaminate(candidates, references, out):
    """Flag candidate tasks that share content with any reference (held-out) task."""
    import collections
    from concurrent.futures import ProcessPoolExecutor
    def load(dirs):
        with ProcessPoolExecutor(32) as pool:
            return dict(zip([Path(d).name for d in dirs], pool.map(_features, dirs, chunksize=64)))
    cand, refs = load(candidates), load(references)
    df = {f: collections.Counter() for f in FIELDS}
    blob_df = collections.Counter()
    for feats, blobs in list(cand.values()) + list(refs.values()):
        for f in FIELDS:
            df[f].update(feats[f])
        blob_df.update(blobs)
    keep = lambda f, s: {x for x in s if df[f][x] <= 8}  # template boilerplate dropped
    index = {f: collections.defaultdict(list) for f in FIELDS}
    ref_sets = {}
    for name, (feats, blobs) in refs.items():
        ref_sets[name] = {f: keep(f, feats[f]) for f in FIELDS}
        ref_sets[name]['blobs'] = {b for b in blobs if blob_df[b] <= 3}
        for f in FIELDS:
            for x in ref_sets[name][f]:
                index[f][x].append(name)
    ref_blobs = collections.defaultdict(list)
    for name, sets in ref_sets.items():
        for b in sets['blobs']:
            ref_blobs[b].append(name)
    flagged = {}
    for name, (feats, blobs) in cand.items():
        reasons = []
        for f in FIELDS:
            mine = keep(f, feats[f])
            if len(mine) < 25:
                continue
            hits = collections.Counter(r for x in mine for r in index[f].get(x, ()))
            for ref, shared in hits.items():
                other = len(ref_sets[ref][f])
                if other >= 25 and shared / min(len(mine), other) >= FLAGS[f]:
                    reasons.append({'reference': ref, 'field': f, 'containment': round(shared / min(len(mine), other), 3)})
        for b in blobs:
            if blob_df[b] <= 3:
                reasons += [{'reference': ref, 'field': 'shared_file', 'sha1': b} for ref in ref_blobs.get(b, ())]
        if reasons:
            flagged[name] = reasons
    atomic(out, {'candidates': len(cand), 'references': len(refs), 'flagged': flagged,
                 'rules': {**FLAGS, 'shared_file': '>= 300 bytes, in <= 3 tasks'},
                 'method': 'seeds/METHODOLOGY.md 6.1/6.3 lexical features; flagged tasks are excluded without review'})
    print(f'{len(flagged)} of {len(cand)} candidates flagged against {len(refs)} references -> {out}')


def _features(task_dir):
    text, blobs = task_fields(task_dir)
    return {f: set().union(*(shingles(t, f) for t in text[f])) if text[f] else set() for f in FIELDS}, blobs


def order(names, seed, exclude=()):
    """Deterministic random order of the tasks not excluded (decontamination flags)."""
    return sorted((n for n in names if n not in set(exclude)),
                  key=lambda n: hashlib.sha256(f'{seed}/{n}'.encode()).hexdigest())


TMAX_VALIDATION_SH = """#!/bin/bash
# Environment validation (no agent): the initial-state tests pass and the final-state tests fail.
mkdir -p /logs/verifier
cd /tests
python3 -m pytest test_initial_state.py -q > /logs/verifier/initial-state.txt 2>&1; I=$?
python3 -m pytest test_final_state.py -q > /logs/verifier/final-state.txt 2>&1; F=$?
echo "{\\"initial_exit\\": $I, \\"final_exit\\": $F}" > /logs/verifier/validation.json
if [ $I -eq 0 ] && [ $F -ne 0 ]; then echo 1 > /logs/verifier/reward.txt; else echo 0 > /logs/verifier/reward.txt; fi
"""


def tmax_validation(harbor_dir, source_dir, out_dir):
    """Validation copies of TMax Harbor tasks: + test_initial_state.py, validation test.sh."""
    harbor_dir, source_dir, out_dir = Path(harbor_dir), Path(source_dir), Path(out_dir)
    count = 0
    for task in sorted(p for p in harbor_dir.iterdir() if (p / 'task.toml').is_file()):
        target = out_dir / task.name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(task, target, symlinks=True)
        shutil.copy2(source_dir / task.name / 'test_initial_state.py', target / 'tests' / 'test_initial_state.py')
        (target / 'tests' / 'test.sh').write_text(TMAX_VALIDATION_SH)
        count += 1
    print(f'{out_dir}: {count} validation tasks')


def tests_ran(verifier):
    """The task's tests executed (a 0 from a verifier that broke before testing does not count)."""
    try:
        if json.loads((verifier / 'ctrf.json').read_text())['results']['summary'].get('tests', 0) > 0:
            return True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        return bool(re.search(r'=+ .*\d+ (failed|error)', (verifier / 'test-stdout.txt').read_text(errors='replace')))
    except OSError:
        return False


_RESULTS = {}  # result.json path -> ((mtime_ns, size), the fields validation_status reads)


def validation_status(runs, kind, out, min_failures=1, assume_valid=None):
    """{task: valid} from no-agent Harbor runs. kind tmax: the validation verifier's reward is 1;
    kind nop: the task's own verifier ran and scored 0 on the untouched environment.

    A pass from any run stands. min_failures > 1: a task is invalid only after that many failed runs (or a nop reward
    of 1, a task solved by doing nothing); with fewer it gets no verdict yet, for a later sweep. RTS 2026-10-26: an
    8-minute proxy outage turned ~750 of 3,441 verdicts into verifier timeouts, tests that never ran and failed
    builds, and a controller never reopens a task once its verdict was false."""
    import glob
    passed, failures, reasons, solved = set(), collections.Counter(), {}, set()
    # Trial results sit at <run>/<trial>/result.json: a recursive glob walked every trial's logs and artifacts, and
    # re-reading ~20k results per pass took 10+ min on /data (RTS 2026-10-26); --loop keeps parsed results by mtime.
    for path in sorted(p for pattern in runs for p in glob.glob(f'{pattern.rstrip("/")}/*/result.json')):
        try:
            st = os.stat(path)
            hit = _RESULTS.get(path)
            if hit and hit[0] == (st.st_mtime_ns, st.st_size):
                r = hit[1]
            else:
                r = json.loads(Path(path).read_text())
                if r.get('finished_at') is not None:
                    _RESULTS[path] = ((st.st_mtime_ns, st.st_size), {k: r.get(k) for k in (
                        'trial_name', 'finished_at', 'task_name', 'verifier_result', 'exception_info')})
        except (OSError, ValueError):
            continue
        if not r.get('trial_name') or r.get('finished_at') is None:
            continue
        task = r['task_name'].split('/')[-1]
        rewards = (r.get('verifier_result') or {}).get('rewards') or {}
        reward = rewards.get('reward')
        error = (r.get('exception_info') or {}).get('exception_type')
        if 'is not published' in str((r.get('exception_info') or {}).get('exception_message')):
            # A shared prefix base not built yet (synth.prefix_bases): no verdict, a later sweep decides. A task with
            # no verdict holds the controller's opening cursor, so datasets with such bases set a validation_window
            # covering the whole list (RTS: 40000). A verifier that wrote no reward counts as a failed run.
            continue
        valid = (error is None and reward == 1) if kind == 'tmax' else (error is None and reward == 0
                                                                        and tests_ran(Path(path).parent / 'verifier'))
        if valid:
            passed.add(task)
            reasons[task] = error or f'reward={reward}'
            continue
        failures[task] += 1
        if kind == 'nop' and error is None and reward == 1:
            solved.add(task)
        if task not in passed:
            reasons[task] = error or (f'reward={reward}' if kind == 'tmax' or reward != 0 else 'tests did not run')
    verdicts = {t: True for t in passed}
    for t, n in failures.items():
        if t not in passed and (n >= min_failures or t in solved):
            verdicts[t] = False
    pending = sum(1 for t in failures if t not in verdicts)
    # assume_valid: tasks a campaign may open before their sweep decides (RTS 2026-10-27: network-heavy tasks, whose
    # sweep crawls behind the proxy, would otherwise be skipped and the corpus would lose them); a sweep verdict wins.
    for t in (assume_valid or ()):
        if t not in verdicts:
            verdicts[t] = True
            reasons[t] = 'assumed valid (not validated yet)'
    atomic(out, verdicts)
    atomic(Path(out).with_name(Path(out).stem + '-reasons.json'), reasons)
    print(f'{out}: {sum(verdicts.values())} valid of {len(verdicts)}; {pending} failed fewer than {min_failures} times '
          f'(no verdict yet)')


def calibforge(checkout, out_dir):
    """CalibForge (Harbor tasks) -> one directory; environments build from their Dockerfiles.

    Each task.toml names a prebuilt Docker Hub image; 5,431 distinct images would be throttled
    by Docker Hub's pull limit (200/h for our account), while the Dockerfiles use 13 bases. The
    `docker_image` line is removed and recorded in <out_dir>.images.json."""
    checkout, out_dir = Path(checkout), Path(out_dir)
    images = {}
    for subset in ('multi_solver', 'contrastive_solver'):
        for task in sorted(p for p in (checkout / subset).iterdir() if (p / 'task.toml').is_file()):
            if task.name in images:
                raise ValueError(f'duplicate CalibForge task id {task.name}')
            target = out_dir / task.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(task, target)
            text = (target / 'task.toml').read_text()
            m = re.search(r'^docker_image\s*=\s*"([^"]*)"\s*\n?', text, re.M)
            images[task.name] = {'subset': subset, 'docker_image': m and m.group(1)}
            if m:
                (target / 'task.toml').write_text(text[:m.start()] + text[m.end():])
    atomic(out_dir.with_name(out_dir.name + '.images.json'), images)
    print(f'{out_dir}: {len(images)} tasks')


def bundle(harbor_dir, out):
    """tasks/<name>/... tarball of every task directory (instruction.md + task.toml) in harbor_dir."""
    harbor_dir, out = Path(harbor_dir), Path(out)
    names = sorted(p.name for p in harbor_dir.iterdir()
                   if (p / 'task.toml').is_file() and (p / 'instruction.md').is_file())
    tmp = out.with_name(f'.{out.name}.partial')
    with tarfile.open(tmp, 'w') as tar:
        for name in names:
            tar.add(harbor_dir / name, arcname=f'tasks/{name}')
    tmp.replace(out)
    out.with_name(out.name + '.tasks.txt').write_text('\n'.join(names) + '\n')
    print(f'{out}: {len(names)} tasks')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('tmax-base')
    p.add_argument('definition')
    p.add_argument('context')
    p = sub.add_parser('register-base')
    p.add_argument('context')
    p.add_argument('ref')
    p.add_argument('--submit', action='store_true')
    p = sub.add_parser('bundle')
    p.add_argument('harbor_dir')
    p.add_argument('out')
    p = sub.add_parser('decontaminate')
    p.add_argument('out')
    p.add_argument('--candidates', nargs='+', required=True, help='task dirs or globs')
    p.add_argument('--references', nargs='+', required=True, help='task dirs or globs (held-out sets)')
    p = sub.add_parser('order')
    p.add_argument('tasks_txt')
    p.add_argument('seed')
    p.add_argument('out')
    p.add_argument('--exclude-json', help='decontamination output (its flagged tasks are excluded)')
    p = sub.add_parser('tmax-validation')
    p.add_argument('harbor_dir')
    p.add_argument('source_dir')
    p.add_argument('out_dir')
    p = sub.add_parser('validation-status')
    p.add_argument('kind', choices=['tmax', 'nop'])
    p.add_argument('out')
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--loop', type=int, default=0, help='rescan every N seconds')
    p.add_argument('--min-failures', type=int, default=1, help='failed runs before a task is invalid')
    p.add_argument('--assume-valid', help='file of tasks treated as valid until a sweep decides')
    p = sub.add_parser('calibforge')
    p.add_argument('checkout')
    p.add_argument('out_dir')
    a = ap.parse_args()
    if a.op == 'calibforge':
        calibforge(a.checkout, a.out_dir)
    elif a.op == 'order':
        names = [x.strip() for x in Path(a.tasks_txt).read_text().splitlines() if x.strip()]
        flagged = json.loads(Path(a.exclude_json).read_text())['flagged'] if a.exclude_json else {}
        ordered = order(names, a.seed, flagged)
        Path(a.out).write_text('\n'.join(ordered) + '\n')
        print(f'{a.out}: {len(ordered)} tasks ({len(names) - len(ordered)} excluded), seed {a.seed!r}')
    elif a.op == 'tmax-validation':
        tmax_validation(a.harbor_dir, a.source_dir, a.out_dir)
    elif a.op == 'validation-status':
        while True:
            assume = Path(a.assume_valid).read_text().split() if a.assume_valid else None
            validation_status(a.runs, a.kind, a.out, a.min_failures, assume)
            if not a.loop:
                break
            time.sleep(a.loop)
    elif a.op == 'decontaminate':
        import glob
        expand = lambda xs: sorted(d for x in xs for d in glob.glob(x) if (Path(d) / 'instruction.md').is_file())
        decontaminate(expand(a.candidates), expand(a.references), a.out)
    elif a.op == 'tmax-base':
        tmax_base(a.definition, a.context)
    elif a.op == 'register-base':
        submit_register(a.context, a.ref) if a.submit else register_base(a.context, a.ref)
    elif a.op == 'bundle':
        bundle(a.harbor_dir, a.out)
