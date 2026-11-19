"""Shared harbor_ns bases from Dockerfile install prefixes (user, 2026-10-26: one shared base per package set, so
per-task images do not blow up /data storage).

A task Dockerfile's install prefix is its FROM line plus the leading ENV / WORKDIR / SHELL / LABEL instructions and
RUN instructions that only install software (apt, yum, apk, pip, npm, gem, repository keys), up to the first COPY,
ADD, ARG, USER, heredoc RUN or RUN that also writes task files. Tasks whose prefixes are the same shell words share
one base: that prefix plus tmux (RTS Dockerfiles end with a harbor tmux block, which then exits at
`command -v tmux`). The task Dockerfile becomes `FROM local/<dataset>-prefix-<id>:1` followed by its own remaining
lines verbatim, as Docker's layer cache would build it, so each task still gets exactly its own filesystem. The
installs exist once per group on /data (harbor_ns/bases, never pulled from a registry: LOCAL_REF_PREFIX) and once
per worker in scratch; the task layer is rebuilt locally (HARBOR_NS_PERSIST_IMAGES=0).

  python -m synth.prefix_bases plan TASKS_DIR ORDER.txt OUT_DIR --dataset rts [--min-group 2]
  python -m synth.prefix_bases bundle TASKS_DIR OUT_DIR/manifest.json BUNDLE.tar
  python -m synth.prefix_bases build OUT_DIR/manifest.json --shard I --of N [--parallel 4] [--limit P]   (CPU job)
  python -m synth.prefix_bases submit OUT_DIR/manifest.json --jobs N [--limit P] [--bid 40]
  python -m synth.prefix_bases wait OUT_DIR/manifest.json SHARD.txt      (blocks until the shard's bases settled)
  python -m synth.prefix_bases status OUT_DIR/manifest.json

--limit P builds only bases used by one of the first P tasks of ORDER.txt (bases are built in that order).
Results: OUT_DIR/results/<id>.json (published | failed, with reasons); failed builds keep their log in OUT_DIR/logs.
Tasks of a failed base fail to build, so the nop validation sweep marks them invalid and campaigns skip them.
"""
import argparse
import asyncio
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import time

from .common import PROXY_ENV, PYTHON, REPO, atomic, read

SHARED = Path(os.environ.get('HARBOR_NS_SHARED', '/data/aai/tbench/harbor_ns'))
VERSION = 1
# Commands an install-only RUN may consist of (anything else ends the prefix: the task's own part starts there).
INSTALLERS = {'apt-get', 'apt', 'aptitude', 'yum', 'dnf', 'microdnf', 'apk', 'npm', 'gem', 'conda', 'mamba',
              'update-ca-certificates', 'locale-gen', 'add-apt-repository', 'apt-key', 'gpg', 'dpkg-reconfigure'}
NEUTRAL = {'true', ':', 'set'}
# paths `rm` may delete in an install-only RUN (package-manager caches)
CLEAN_PATHS = {'/var/lib/apt/lists/*', '/var/lib/apt/lists', '/var/cache/apt/*', '/var/cache/apt/archives/*',
               '/var/cache/apt', '/root/.cache', '/root/.cache/*', '/root/.cache/pip', '/var/cache/yum',
               '/var/cache/dnf', '/var/cache/apk/*', '/tmp/*'}
SEPARATORS = {'&&', '||', ';', '|', '&'}
TMUX_COMMENT = '# harbor-agent-runtime: tmux is required by the terminal agent'
TMUX_BLOCK = re.compile(r'^RUN if command -v tmux >/dev/null 2>&1; then \\\n(?:.*\\\n)*?\s*fi[ \t]*$', re.M)


def shell_tokens(raw):
    """Words and operators of a shell command line (None: not tokenizable)."""
    lex = shlex.shlex(raw, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    lex.commenters = ''
    try:
        return tuple(lex)
    except ValueError:
        return None


def install_only(tokens):
    """The RUN only installs software: every command is a package manager (apt/yum/apk/pip/npm/gem/cargo/go
    install), a cache cleanup, or curl/wget piped into gpg/apt-key/sh; no redirection except to /dev/null and no
    subshells, conditionals or variables set for later commands."""
    if not tokens:
        return False
    commands, seps, i = [[]], [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in SEPARATORS:
            commands.append([])
            seps.append(t)
        elif t in ('>', '>>', '<', '>&', '<&', '&>', '>|'):
            target = tokens[i + 1] if i + 1 < len(tokens) else ''
            if target not in ('/dev/null', '1', '2'):
                return False
            if commands[-1] and commands[-1][-1].isdigit():
                commands[-1].pop()  # the fd of `2>/dev/null`
            i += 1
        elif t in ('(', ')', '{', '}', ';;') or '$(' in t or '`' in t:
            return False
        else:
            commands[-1].append(t)
        i += 1
    installs, consumed = 0, set()
    for k, words in enumerate(commands):
        if k in consumed:
            continue
        while words and (re.fullmatch(r'[A-Za-z_]\w*=\S*', words[0]) or words[0] in ('sudo', 'env', 'command')):
            words = words[1:]
        if not words:
            if k == 0 or k == len(commands) - 1:
                return False  # an assignment alone (VAR=x && ...) sets state for the task part
            continue
        name = os.path.basename(words[0])
        piped_into = commands[k + 1][:1] if k < len(seps) and seps[k] == '|' else []
        if name in INSTALLERS or re.fullmatch(r'pip[0-9.]*', name):
            installs += name not in ('gpg', 'apt-key', 'dpkg-reconfigure')
        elif re.fullmatch(r'python[0-9.]*', name) and words[1:3] == ['-m', 'pip']:
            installs += 1
        elif name in ('cargo', 'go') and words[1:2] == ['install']:
            installs += 1
        elif name in ('curl', 'wget') and piped_into and os.path.basename(piped_into[0]) in ('gpg', 'apt-key', 'sh', 'bash'):
            installs += 1
            consumed.add(k + 1)  # the installer script / key import it feeds
        elif name == 'rm' and all(w.startswith('-') or w in CLEAN_PATHS for w in words[1:]):
            pass
        elif name in NEUTRAL:
            pass
        else:
            return False
    return installs > 0


def split_prefix(text):
    """(key, prefix_end_line, stock_base, reason) of a Dockerfile. key: hashable install prefix (None: no prefix);
    prefix_end_line: 1-based line where the task's own part starts (None: the whole file is prefix)."""
    from harbor_ns import dockerfile as dfp
    if re.match(r'\s*#\s*(escape|syntax)\s*=', text):
        return None, None, None, 'parser directive'
    try:
        stages, global_args = dfp.parse(text)
    except Exception as e:  # noqa: BLE001
        return None, None, None, f'parse error: {e}'
    if len(stages) != 1 or global_args:
        return None, None, None, 'multi-stage or global ARG'
    stage = stages[0]
    if stage.platform:
        return None, None, None, 'FROM --platform'
    key = [('FROM', stage.base)]
    for ins in stage.instructions:
        if ins.cmd in ('ENV', 'WORKDIR', 'SHELL', 'LABEL'):
            key.append((ins.cmd, ' '.join(ins.raw.split())))
            continue
        if ins.cmd == 'RUN' and not ins.json_form and not ins.heredocs and not ins.flags:
            tokens = shell_tokens(ins.raw)
            if install_only(tokens):
                key.append(('RUN', tokens))
                continue
        break
    while key[-1][0] not in ('RUN', 'FROM'):  # trailing ENV/WORKDIR belong to the task part
        key.pop()
    kept = len(key) - 1
    if not kept:
        return None, None, stage.base, 'no install prefix'
    end = stage.instructions[kept].lineno if kept < len(stage.instructions) else None
    return tuple(key), end, stage.base, 'prefix'


def base_id(prefix_text, tmux):
    return hashlib.sha256(f'prefix-bases v{VERSION}\n{prefix_text}\n{tmux}\n'.encode()).hexdigest()[:16]


def ref_of(dataset, bid):
    return f'local/{dataset}-prefix-{bid}:1'


def safe_name(ref):
    from harbor_ns.builder import _safe_name
    return _safe_name(ref)


def shared_paths(ref):
    name = safe_name(ref)
    return SHARED / 'bases' / f'{name}.sqfs', SHARED / 'bases' / f'{name}.json'


def plan(tasks_dir, order_path, out, dataset, min_group=2):
    """Group every task of tasks_dir by install prefix; write OUT/manifest.json and OUT/contexts/<id>/Dockerfile."""
    tasks_dir, out = Path(tasks_dir), Path(out)
    order = [x.strip() for x in Path(order_path).read_text().splitlines() if x.strip()]
    position = {t: i for i, t in enumerate(order)}
    names = sorted(p.name for p in tasks_dir.iterdir() if (p / 'environment' / 'Dockerfile').is_file())
    info, groups, blocks = {}, collections.defaultdict(list), collections.Counter()
    for name in names:
        text = (tasks_dir / name / 'environment' / 'Dockerfile').read_text()
        key, end, stock, reason = split_prefix(text)
        m = TMUX_BLOCK.search(text)
        if m:
            blocks[m.group(0)] += 1
        info[name] = {'reason': reason, 'stock': stock, 'end': end, 'tmux_block': bool(m),
                      'upstream_sha256': hashlib.sha256(text.encode()).hexdigest()}
        if key:
            groups[key].append(name)
    tmux = blocks.most_common(1)[0][0] if blocks else ''
    bases = {}
    contexts = out / 'contexts'
    for key, members in groups.items():
        if len(members) < min_group:
            for t in members:
                info[t]['reason'] = f'prefix shared by fewer than {min_group} tasks'
            continue
        first = min(members, key=lambda t: position.get(t, len(order)))
        text = (tasks_dir / first / 'environment' / 'Dockerfile').read_text()
        end = info[first]['end']
        lines = text.split('\n')
        prefix_text = '\n'.join(lines[:end - 1] if end else lines).rstrip() + '\n'
        bid = base_id(prefix_text, tmux)
        dockerfile = (f'# synth.prefix_bases v{VERSION}: install prefix shared by {len(members)} {dataset} tasks '
                      f'(as written in {first}), plus tmux\n{prefix_text}\n{TMUX_COMMENT}\n{tmux}\n')
        (contexts / bid).mkdir(parents=True, exist_ok=True)
        (contexts / bid / 'Dockerfile').write_text(dockerfile)
        priority = min(position.get(t, len(order)) for t in members)
        bases[bid] = {'ref': ref_of(dataset, bid), 'stock': key[0][1], 'tasks': sorted(members), 'priority': priority,
                      'apt_packages': apt_packages(key), 'context': str(contexts / bid)}
        for t in members:
            info[t].update(base=bid, reason='shared')
    manifest = {'version': VERSION, 'dataset': dataset, 'tasks_dir': str(tasks_dir), 'order': str(order_path),
                'min_group': min_group, 'tmux_block': tmux, 'created_at': time.time(),
                'bases': dict(sorted(bases.items(), key=lambda kv: kv[1]['priority'])), 'tasks': info}
    atomic(out / 'manifest.json', manifest)
    shared = sum(1 for v in info.values() if v.get('base'))
    reasons = collections.Counter(v['reason'] for v in info.values())
    print(f'{out}/manifest.json: {len(bases)} bases for {shared} of {len(info)} tasks; {dict(reasons)}')
    for limit in (2000, 5000, 10000, 16000, len(order)):
        need = {b for b, v in bases.items() if v['priority'] < limit}
        print(f'  first {limit} tasks of the order need {len(need)} bases')


def apt_packages(key):
    """Package names the prefix's apt-get/apt install commands name (for the post-build check)."""
    names = []
    for kind, words in key:
        if kind != 'RUN':
            continue
        words = list(words)
        i = 0
        while i < len(words):
            if words[i] in ('apt-get', 'apt') and 'install' in words[i + 1:i + 6]:
                j = words.index('install', i + 1) + 1
                while j < len(words) and words[j] not in ('&&', ';', '||', '|', ')'):
                    w = words[j]
                    if w in ('-o',):
                        j += 2
                        continue
                    if not w.startswith('-') and re.fullmatch(r'[a-z0-9][a-z0-9.+_-]*(:[a-z0-9]+)?(=\S+)?', w):
                        names.append(re.split(r'[:=]', w)[0])
                    j += 1
                i = j
            else:
                i += 1
    return sorted(set(names))


def rewrite(text, ref, end, bid):
    lines = text.split('\n')
    rest = '\n'.join(lines[end - 1:]) if end else ''
    return (f'# synth.prefix_bases v{VERSION}: lines 1-{(end or len(lines) + 1) - 1} of the upstream Dockerfile are '
            f'the shared base {ref} (+ tmux); the rest is verbatim\nFROM {ref}\n{rest}')


# RTS's own verifier patch (upstream "harbor-reward-fallback"), present in 35,043 of 37,484 test.sh files. Of the
# rest, 2,082 run `set -e` and write the reward only after pytest passes, so a failing run leaves no reward file
# (Harbor: RewardFileNotFoundError); `bundle --reward-fallback` inserts the same block after their shebang.
REWARD_FALLBACK = """
# harbor-reward-fallback: ensure failed verifiers still emit reward
mkdir -p /logs/verifier
_harbor_write_missing_reward() {
  if [ ! -f /logs/verifier/reward.txt ] && [ ! -f /logs/verifier/reward.json ]; then
    echo 0 > /logs/verifier/reward.txt
  fi
}
trap _harbor_write_missing_reward EXIT
"""


def add_reward_fallback(text):
    """test.sh with REWARD_FALLBACK after its shebang (None: it already has one, or another EXIT trap)."""
    if 'harbor-reward-fallback' in text or re.search(r'^\s*trap\b.*\bEXIT\b', text, re.M):
        return None
    head, sep, rest = text.partition('\n') if text.startswith('#!') else ('', '', text)
    return f'{head}{sep}{REWARD_FALLBACK}{rest}' if head else f'#!/bin/bash{REWARD_FALLBACK}{text}'


def bundle(tasks_dir, manifest_path, out, reward_fallback=False):
    """tasks/<name>/... tarball of every task, Dockerfiles of grouped tasks rewritten onto their shared base (and, with
    reward_fallback, the fallback block in test.sh files without one; listed in <out>.reward-fallback.json)."""
    manifest = json.loads(Path(manifest_path).read_text())
    tasks_dir, out = Path(tasks_dir), Path(out)
    tmp = out.with_name(f'.{out.name}.partial')
    names = sorted(manifest['tasks'])
    patched = []

    def add_bytes(path, arc, data):
        ti = tar.gettarinfo(str(path), arc)
        ti.size = len(data)
        tar.addfile(ti, io.BytesIO(data))

    with tarfile.open(tmp, 'w') as tar:
        for name in names:
            meta = manifest['tasks'][name]
            for path in sorted((tasks_dir / name).rglob('*')):
                rel = path.relative_to(tasks_dir / name).as_posix()
                arc = f'tasks/{name}/{rel}'
                regular = path.is_file() and not path.is_symlink()
                if regular and rel == 'environment/Dockerfile' and meta.get('base'):
                    add_bytes(path, arc, rewrite(path.read_text(), manifest['bases'][meta['base']]['ref'], meta['end'],
                                                 meta['base']).encode())
                elif regular and rel == 'tests/test.sh' and reward_fallback and \
                        (new := add_reward_fallback(path.read_text(errors='replace'))) is not None:
                    add_bytes(path, arc, new.encode())
                    patched.append(name)
                else:
                    tar.add(str(path), arc, recursive=False)
    tmp.replace(out)
    out.with_name(out.name + '.tasks.txt').write_text('\n'.join(names) + '\n')
    if reward_fallback:
        atomic(out.with_name(out.name + '.reward-fallback.json'), {'block': REWARD_FALLBACK, 'tasks': patched})
    print(f'{out}: {len(names)} tasks, {sum(1 for v in manifest["tasks"].values() if v.get("base"))} on shared bases, '
          f'{len(patched)} test.sh given the reward fallback')


# ---------------------------------------------------------------------------------------------------- building

def dpkg_installed(rootfs):
    """Installed package names and the virtual names they provide, from the rootfs's dpkg status."""
    status = Path(rootfs) / 'var/lib/dpkg/status'
    if not status.is_file():
        return None
    have = set()
    for stanza in status.read_text(errors='replace').split('\n\n'):
        fields = dict(re.findall(r'^([A-Za-z-]+): ?(.*)$', stanza, re.M))
        if 'install ok installed' not in fields.get('Status', ''):
            continue
        have.add(fields.get('Package', ''))
        for p in fields.get('Provides', '').split(','):
            if p.strip():
                have.add(p.strip().split()[0].split(':')[0])
    return have


class Mutex:
    """Cross-node build mutex on Lustre (no kernel locks there): an mkdir, stale after `stale` seconds."""

    def __init__(self, path, stale=3 * 3600):
        self.path, self.stale = Path(path), stale

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            try:
                age = time.time() - self.path.stat().st_mtime
            except OSError:
                return False
            if age < self.stale:
                return False
            shutil.rmtree(self.path, ignore_errors=True)
            try:
                self.path.mkdir()
            except FileExistsError:
                return False
        atomic(self.path / 'owner.json', {'host': socket.gethostname(), 'pid': os.getpid(), 'time': time.time(),
                                          'job': os.environ.get('SYNTH_JOB_ID')})
        return True

    def release(self):
        shutil.rmtree(self.path, ignore_errors=True)


def squash(rootfs, dest, xattrs):
    argv = ['mksquashfs', str(rootfs), str(dest), '-comp', 'zstd', '-quiet', '-no-progress', '-processors', '4']
    if not xattrs:
        argv.append('-no-xattrs')
    subprocess.run(argv, check=True, capture_output=True)


def unsquash_ok(image, work):
    """The published image unpacks the way ImageStore.ensure_base unpacks it (nsfake owner xattrs included)."""
    target = Path(work) / 'unsquash-check'
    shutil.rmtree(target, ignore_errors=True)
    r = subprocess.run(['unsquashfs', '-q', '-n', '-d', str(target), str(image)], capture_output=True, text=True)
    subprocess.run(['chmod', '-R', 'u+rwX', str(target)], capture_output=True)
    shutil.rmtree(target, ignore_errors=True)
    return r.returncode == 0, (r.stderr or r.stdout)[-800:]


async def build_one(store, manifest, bid, out, work, rounds=3):
    base = manifest['bases'][bid]
    ref = base['ref']
    sqfs, meta_path = shared_paths(ref)
    result_path = out / 'results' / f'{bid}.json'
    if sqfs.exists() and meta_path.exists():
        if not result_path.exists():
            atomic(result_path, {'status': 'published', 'ref': ref, 'adopted': True, 'time': time.time()})
        return 'published'
    mutex = Mutex(SHARED / 'bases' / '.building' / safe_name(ref))
    if not await asyncio.to_thread(mutex.acquire):
        return 'busy'
    started, reasons = time.time(), []
    pins = []   # (image dir, token): the built image is pinned against harbor_ns eviction while it is checked and squashed
    try:
        for attempt in range(rounds):
            if attempt:
                await asyncio.sleep(120 * attempt)
            pin = f'prefix-base-{bid}-{os.getpid()}-{attempt}'
            try:  # a failed build's log lands in <shared>/logs/prefix-base-<id>.<key>.failed.log
                key, rootfs, meta = await store.build_pinned(Path(base['context']), label=f'prefix-base-{bid}', token=pin)
            except Exception as e:  # noqa: BLE001  (BuildError, timeouts)
                reasons.append(f'round {attempt}: build failed: {str(e)[-600:]}')
                continue
            pins.append((store.image_dir(key), pin))
            image_dir = store.image_dir(key)
            built = read(image_dir / 'meta.json', {})
            logs = out / 'logs'
            problems = []
            if built.get('proxy_failures_last_attempt'):
                problems.append(f'{built["proxy_failures_last_attempt"]} downloads failed at the proxy in the last build')
            if not any((Path(rootfs) / p).exists() for p in ('usr/bin/tmux', 'bin/tmux', 'usr/local/bin/tmux')):
                problems.append('tmux missing')
            have = await asyncio.to_thread(dpkg_installed, rootfs)
            if have is not None:
                missing = [p for p in base['apt_packages'] if p not in have]
                if missing:
                    problems.append(f'apt packages not installed: {missing[:20]}')
            if problems:
                reasons.append(f'round {attempt}: ' + '; '.join(problems))
                logs.mkdir(parents=True, exist_ok=True)
                shutil.copy(image_dir / 'build.log', logs / f'{bid}.round{attempt}.log')
                subprocess.run(['chmod', '-R', 'u+rwX', str(image_dir)], capture_output=True)
                shutil.rmtree(image_dir, ignore_errors=True)  # rebuild from scratch next round
                continue
            record = {**meta.to_json(), 'base': ref, 'oci': True}
            local = Path(work) / f'{bid}.sqfs'
            xattrs = True
            await asyncio.to_thread(squash, rootfs, local, True)
            ok, err = await asyncio.to_thread(unsquash_ok, local, work)
            if not ok:  # nsfake owner xattrs on read-only files cannot be restored without root
                xattrs = False
                local.unlink(missing_ok=True)
                await asyncio.to_thread(squash, rootfs, local, False)
                ok, err = await asyncio.to_thread(unsquash_ok, local, work)
            if not ok:
                reasons.append(f'round {attempt}: squashfs does not unpack: {err}')
                local.unlink(missing_ok=True)
                continue
            tmp = sqfs.with_name(f'.{sqfs.name}.{os.getpid()}')
            await asyncio.to_thread(subprocess.run, ['dd', f'if={local}', f'of={tmp}', 'bs=64M', 'oflag=direct',
                                                     'conv=fsync', 'status=none'], check=True)
            tmp.replace(sqfs)
            atomic(meta_path, record)
            size = local.stat().st_size
            local.unlink(missing_ok=True)
            atomic(result_path, {'status': 'published', 'ref': ref, 'bytes': size, 'xattrs': xattrs,
                                 'seconds': round(time.time() - started), 'rounds': attempt + 1,
                                 'build_attempts': built.get('build_attempts', 1), 'reasons': reasons,
                                 'host': socket.gethostname(), 'time': time.time()})
            subprocess.run(['chmod', '-R', 'u+rwX', str(image_dir)], capture_output=True)
            shutil.rmtree(image_dir, ignore_errors=True)
            return 'published'
        atomic(result_path, {'status': 'failed', 'ref': ref, 'reasons': reasons, 'seconds': round(time.time() - started),
                             'host': socket.gethostname(), 'time': time.time()})
        return 'failed'
    finally:
        for image_dir, pin in pins:
            store.unmark_in_use(image_dir, pin)
        await asyncio.to_thread(mutex.release)


def selected(manifest, limit=None):
    return [b for b, v in manifest['bases'].items() if limit is None or v['priority'] < limit]


async def build_shard(manifest_path, shard, of, parallel, limit, rounds, only=None):
    manifest_path = Path(manifest_path)
    out = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    from harbor_ns.cli import default_store
    os.environ.setdefault('HARBOR_NS_NETWORK_REBUILDS', '3')
    store = default_store()
    store.persist_images = False
    store.network_rebuilds = int(os.environ['HARBOR_NS_NETWORK_REBUILDS'])
    store.network_backoff = (60.0, 180.0, 300.0)
    work = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp') / 'prefix-bases'
    work.mkdir(parents=True, exist_ok=True)
    mine = [b for i, b in enumerate(selected(manifest, limit)) if i % of == shard and (not only or b in only)]
    pending = [b for b in mine if (read(out / 'results' / f'{b}.json', {}) or {}).get('status') != 'failed']
    sem = asyncio.Semaphore(parallel)
    counts = collections.Counter()

    async def one(bid):
        async with sem:
            t0 = time.time()
            status = await build_one(store, manifest, bid, out, work, rounds)
            counts[status] += 1
            print(f'{time.strftime("%H:%M:%S")} {bid} {status} {time.time() - t0:.0f}s ({dict(counts)})', flush=True)
            return bid, status

    results = dict(await asyncio.gather(*(one(b) for b in pending)))
    # Bases another job was building: wait for them to settle so this job's exit means "shard done".
    busy = [b for b, s in results.items() if s == 'busy']
    deadline = time.time() + 4 * 3600
    while busy and time.time() < deadline:
        await asyncio.sleep(60)
        retry = []
        for b in busy:
            status = await one(b)
            if status[1] == 'busy':
                retry.append(b)
        busy = retry
    print(json.dumps({'shard': shard, 'of': of, 'bases': len(mine), **counts}), flush=True)


def submit(manifest_path, jobs, limit, bid, parallel, exclude):
    manifest_path = Path(manifest_path).resolve()
    out = manifest_path.parent
    logs = out / 'jobs'
    logs.mkdir(parents=True, exist_ok=True)
    env = {'USER': 'user', 'HOME': os.environ.get('HOME', '/home/user'), 'PYTHONNOUSERSITE': '1',
           'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin', 'PYTHONPATH': str(REPO),
           'SYNTH_JOB_ID': '$(Cluster).$(Process)', **PROXY_ENV}
    environment = ' '.join(f"{k}='{v}'" for k, v in env.items())
    from . import condor   # tbshared's BAD_NODES on top of --exclude (#451)
    reqs = ' && '.join(f'(TARGET.UtsnameNodename != "{n}")' for n in sorted(set(exclude) | condor.bad_nodes())) or 'True'
    limit_arg = f'--limit {limit}' if limit is not None else ''
    sub = logs / f'build-{int(time.time())}.sub'
    sub.write_text(f'''universe = vanilla
executable = {PYTHON}
arguments = -u -m synth.prefix_bases build {manifest_path} --shard $(Process) --of {jobs} --parallel {parallel} {limit_arg}
initialdir = {REPO}
output = {logs}/build-$(Cluster).$(Process).out
error = {logs}/build-$(Cluster).$(Process).err
log = {logs}/build.condor.log
request_cpus = 8
request_memory = 32768
request_disk = 150G
+JobBatchName = "prefix-bases-{manifest_path.parent.name}"
+MaxRuntime = 172800
+BypassLXCfs = "true"
getenv = false
environment = "{environment}"
requirements = {reqs}
queue {jobs}
''')
    r = subprocess.run(['condor_submit_bid', str(bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip())


def status(manifest_path, limit=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    counts, size, failed = collections.Counter(), 0, []
    for b in selected(manifest, limit):
        r = read(manifest_path.parent / 'results' / f'{b}.json', {}) or {}
        sqfs, _ = shared_paths(manifest['bases'][b]['ref'])
        s = r.get('status') or ('published' if sqfs.exists() else 'pending')
        counts[s] += 1
        size += r.get('bytes') or 0
        if s == 'failed':
            failed.append((b, len(manifest['bases'][b]['tasks']), (r.get('reasons') or ['?'])[-1][:160]))
    print(json.dumps({**counts, 'published_bytes': size}))
    for f in failed[:20]:
        print('  failed', *f)
    return counts


def wait(manifest_path, shard_file, timeout):
    """Block until every base the shard's tasks use is published or failed (nop validation runs after it)."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    tasks = [x.strip() for x in Path(shard_file).read_text().splitlines() if x.strip()]
    need = {manifest['tasks'][t]['base'] for t in tasks if manifest['tasks'].get(t, {}).get('base')}
    deadline = time.time() + timeout
    while True:
        left = [b for b in need if not shared_paths(manifest['bases'][b]['ref'])[0].exists()
                and (read(manifest_path.parent / 'results' / f'{b}.json', {}) or {}).get('status') != 'failed']
        if not left or time.time() > deadline:
            print(f'{len(need) - len(left)} of {len(need)} bases settled', flush=True)
            return
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('plan')
    p.add_argument('tasks_dir')
    p.add_argument('order')
    p.add_argument('out')
    p.add_argument('--dataset', required=True)
    p.add_argument('--min-group', type=int, default=2)
    p = sub.add_parser('bundle')
    p.add_argument('tasks_dir')
    p.add_argument('manifest')
    p.add_argument('out')
    p.add_argument('--reward-fallback', action='store_true')
    p = sub.add_parser('build')
    p.add_argument('manifest')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--of', type=int, default=1)
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--only', nargs='*', help='base ids (tests)')
    p = sub.add_parser('submit')
    p.add_argument('manifest')
    p.add_argument('--jobs', type=int, required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--bid', type=int, default=40)
    p.add_argument('--parallel', type=int, default=4)
    p.add_argument('--exclude', nargs='*', default=['gpu110', 'gpu132', 'gpu158', 'gpu180', 'cpu105', 'cpu301', 'cpu207', 'cpu208'])
    p = sub.add_parser('status')
    p.add_argument('manifest')
    p.add_argument('--limit', type=int)
    p = sub.add_parser('wait')
    p.add_argument('manifest')
    p.add_argument('shard')
    p.add_argument('--timeout', type=float, default=8 * 3600)
    a = ap.parse_args()
    if a.op == 'plan':
        plan(a.tasks_dir, a.order, a.out, a.dataset, a.min_group)
    elif a.op == 'bundle':
        bundle(a.tasks_dir, a.manifest, a.out, a.reward_fallback)
    elif a.op == 'build':
        asyncio.run(build_shard(a.manifest, a.shard, a.of, a.parallel, a.limit, a.rounds, a.only))
    elif a.op == 'submit':
        submit(a.manifest, a.jobs, a.limit, a.bid, a.parallel, a.exclude)
    elif a.op == 'status':
        status(a.manifest, a.limit)
    elif a.op == 'wait':
        wait(a.manifest, a.shard, a.timeout)


if __name__ == '__main__':
    main()
