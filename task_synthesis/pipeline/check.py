"""Build a materialized task with harbor_ns and check its environment. None of these checks tests correctness
(BLUEPRINT.md, cheap checks): the image builds, services start offline, every input exists, no output pre-exists,
the planted defect is present, construction files are gone, and every path the instruction names makes sense.
The check also snapshots the built environment (tree, heads of the inputs) for the judge."""
import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import time
import tomllib
from pathlib import Path

INSIDE = r'''
import json, os, subprocess, sys
cfg = json.loads(sys.stdin.readline())
out = {}
out['inputs_missing'] = [p for p in cfg['inputs'] if not os.path.lexists(p)]
out['outputs_present'] = [p for p in cfg['outputs'] if os.path.lexists(p)]
out['leftover'] = os.path.exists('/tasksynth')
covered = set(cfg['inputs']) | set(cfg['outputs'])
def under(p, roots):
    return any(p == r or p.startswith(r.rstrip('/') + '/') for r in roots)
bad, undeclared, not_fs = [], [], []
for p in cfg['mentioned']:
    if os.path.lexists(p) or under(p, covered):
        continue
    top = '/' + p.strip('/').split('/')[0]
    if not os.path.isdir(top):  # /healthz, /api/v1/..., /path/to/...: an HTTP route or a placeholder, not a file
        not_fs.append(p)
        continue
    parent = os.path.dirname(p.rstrip('/')) or '/'
    (undeclared if os.path.isdir(parent) else bad).append(p)
out['mentioned_missing_parent'] = bad
out['mentioned_undeclared'] = undeclared
out['mentioned_not_fs'] = not_fs
if cfg['defect_check']:
    try:
        r = subprocess.run(['bash', '-c', cfg['defect_check']], cwd='/app', capture_output=True, text=True, timeout=120)
        out['defect_rc'], out['defect_out'] = r.returncode, (r.stdout + r.stderr)[-600:]
    except subprocess.TimeoutExpired:
        out['defect_rc'], out['defect_out'] = 'timeout', ''
roots = sorted({'/app'} | {os.path.dirname(p.rstrip('/')) or '/' for p in cfg['inputs'] + cfg['outputs']})
SYSTEM = ('/', '/etc', '/usr', '/var', '/home', '/tmp', '/opt', '/srv', '/root', '/run', '/var/run', '/var/lib',
          '/var/log', '/usr/bin', '/usr/sbin', '/usr/local', '/usr/local/bin', '/usr/local/lib', '/usr/lib', '/bin')
roots = [r for r in roots if r not in SYSTEM] + \
        [p for p in cfg['inputs'] if os.path.isdir(p) and p.count('/') <= 2]
tree = []
for root in roots[:12]:
    if not os.path.isdir(root):
        continue
    base = root.rstrip('/').count('/')
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x not in ('.git', '__pycache__', 'node_modules', '.venv'))
        if d.count('/') - base > 3:
            dirs[:] = []
            continue
        for f in sorted(files)[:60]:
            p = os.path.join(d, f)
            try:
                tree.append(f'{os.path.getsize(p):>10}  {p}')
            except OSError:
                tree.append(f'         ?  {p}')
        if len(tree) > 400:
            break
out['tree'] = sorted(set(tree))[:400]
heads, budget = {}, 24000
for p in cfg['inputs']:
    if budget <= 0 or not os.path.isfile(p):
        continue
    try:
        with open(p, 'rb') as f:
            raw = f.read(4000)
        text = raw.decode('utf-8')
    except (OSError, UnicodeDecodeError):
        r = subprocess.run(['file', '-b', p], capture_output=True, text=True)
        heads[p] = '[binary] ' + r.stdout.strip()
        continue
    lines = text.splitlines()[:40]
    heads[p] = '\n'.join(l[:300] for l in lines)
    budget -= len(heads[p])
out['heads'] = heads
if os.path.exists('/var/log/tasksynth-services.log'):
    out['services_log'] = open('/var/log/tasksynth-services.log', errors='replace').read()[-1500:]
print('TASKSYNTH_CHECK ' + json.dumps(out))
'''

PATH_RE = re.compile(r'(?<![\w.~:/\-])(/(?:[A-Za-z0-9_.@+\-]+/?)+)')
NOT_PATHS = {'/', '/s', '/h', '/min', '/sec', '/day', '/month', '/year', '/kg', '/km', '/n'}


def mentioned_paths(instruction):
    found = []
    for m in PATH_RE.finditer(instruction):
        p = m.group(1).rstrip('.,;:)')
        if p in NOT_PATHS or len(p) < 3 or p.count('/') < 2 and len(p) < 5:
            continue
        found.append(p.rstrip('/') if len(p) > 1 else p)
    return sorted(set(found))


BUILD_NOISE = re.compile(
    r'^\s*(\(Reading database|Get:\d|Hit:\d|Ign:\d|Fetched |Selecting previously|Preparing to unpack|Unpacking |'
    r'Setting up |Processing triggers|Reading package lists|Building dependency tree|Reading state information|'
    r'Collecting |Downloading |Requirement already satisfied|Using cached|Stored in directory|Created wheel|'
    r'Building wheels? for|Installing collected|Successfully installed|update-alternatives|Compiling |Checking |'
    r'Downloaded |Updating crates|Locking \d+ packages|Adding .* \(latest|Need to get|After this operation|'
    r'The following (NEW|additional)|Suggested packages|Recommended packages|Progress:|\d+% |#+\s*\d+)')
APT_LISTING = re.compile(r'^\s{2,}[a-z0-9][\w.+\-:]*(\s+[a-z0-9][\w.+\-:]*)*\s*$')   # apt's indented package lists
ERROR_LINE = re.compile(r'error|Error|ERROR|failed|FAILED|Traceback|No such file|not found|cannot|Cannot|denied|'
                        r'exit code|Exception|undefined|panicked|fatal')


def failed_build_log(store, label, limit=2800):
    """What the generator sees of a failed build: the FIRST failure (harbor_ns retries a failed RUN once and the
    retry repeats or masks it), without package-manager progress lines, error lines first."""
    logs = sorted((store.shared_root / 'logs').glob(f'{label}.*.failed.log'), key=lambda p: p.stat().st_mtime) \
        if store.shared_root else []
    if not logs:
        return ''
    text = logs[-1].read_text(errors='replace')
    text = text.split('[harbor_ns] RUN failed', 1)[0]
    lines = [l for l in text.splitlines() if l.strip() and not BUILD_NOISE.match(l) and not APT_LISTING.match(l)]
    tail = '\n'.join(lines)[-limit:]
    errors = [l[:300] for l in lines if ERROR_LINE.search(l)][-12:]
    head = 'Error lines:\n' + '\n'.join(errors) + '\n\nLog tail (progress lines removed):\n' if errors else ''
    return head + tail


def fetch_record(image_dir):
    """What the build fetched through the job's package cache (synth.package_cache): 'fetch' = counts (served from the
    store vs fetched upstream), 'fetches' = the packages (URL, version, SHA-256; pinning, licensing)."""
    out = {}
    try:
        out['fetch'] = json.loads((image_dir / 'fetch.json').read_text())
        lines = (image_dir / 'fetch.jsonl').read_text().splitlines() if (image_dir / 'fetch.jsonl').exists() else []
    except (OSError, ValueError):
        return out
    fetches = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('kind') in ('apt', 'pypi-file') and e.get('package') and e.get('status') in (200, 206, 304):
            fetches.append({k: e.get(k) for k in ('url', 'package', 'version', 'sha256', 'cache')})
    if fetches:
        out['fetches'] = fetches
    return out


def network_failures(store, label, timed_out=False):
    """Downloads the last failed build of `label` lost at the proxy or package cache (harbor_ns.builder.proxy_failures),
    counted in its final rebuild only. `timed_out`: the check's timeout stopped the build; if it stopped a network
    rebuild (harbor_ns rebuilds after proxy failures, with a backoff), the earlier attempts' failures count: the time
    went to the outage, not to the task's setup."""
    from harbor_ns.builder import proxy_failures
    logs = sorted((store.shared_root / 'logs').glob(f'{label}.*.failed.log'), key=lambda p: p.stat().st_mtime) \
        if store.shared_root else []
    if not logs:
        return 0
    text = logs[-1].read_text(errors='replace')
    head, rebuilt, last = text.rpartition('rebuilding from scratch')
    n = proxy_failures(last if rebuilt else text, 8080)
    if not n and timed_out and rebuilt:
        n = proxy_failures(head, 8080)
    return n


def drop_dir(d):
    subprocess.run(['chmod', '-R', 'u+rwX', str(d)], capture_output=True)
    shutil.rmtree(d, ignore_errors=True)


# Checks using each image in this process. Tasks without materials can have byte-identical environments, hence the
# same image key: deleting the image when one check ended pulled the rootfs from under another ("nsenter: failed to
# execute /bin/sh", ~1% of checks in r4-r6, shown to the generator as its error).
_IMAGE_USERS = {}


def release_image(store, key):
    _IMAGE_USERS[key] -= 1
    if not _IMAGE_USERS[key]:
        del _IMAGE_USERS[key]
        drop_dir(store.image_dir(key))


# #171: memory caps, so a task that allocates without bound fails its own check instead of taking the round job (and
# every other spec in it) down with it. No cgroup is delegated to HTCondor jobs here (cgroup v1, the job's memory
# cgroup is root's), and ulimit -v breaks runtimes that reserve address space (V8's cages and wasm, ASan's shadow), so:
# the check's sandbox runs under harbor_ns's PSS guard at the task's memory_mb (task.toml [environment]; the largest
# process is killed), build steps under the builder's resident-memory guard (a build is not the task's runtime, so its
# own cap), and the job's stack limit is Docker's 8 MiB (round run --sandbox-stack-kb).
DEFAULT_MEMORY_MB = 8192
BUILD_MEMORY_MB = 24 * 1024


def task_memory_mb(task_dir):
    """task.toml [environment] memory_mb, or DEFAULT_MEMORY_MB if the field is missing or unreadable."""
    try:
        env = tomllib.loads((Path(task_dir) / 'task.toml').read_text()).get('environment') or {}
        mb = int(env.get('memory_mb') or 0)
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError):
        mb = 0
    return mb if mb > 0 else DEFAULT_MEMORY_MB


async def check(store, task_dir, timeout=600, memory_mb=None, build_memory_mb=BUILD_MEMORY_MB):
    # r2-r4: successful builds took at most 222 s (p99 118 s); the 25-minute builds were all hung setups.
    """-> {'ok', 'errors': [...for the generator...], 'warnings', 'build_seconds', 'snapshot': {...}}
    memory_mb: the check sandbox's memory cap (default: the task's memory_mb); build_memory_mb: each build step's
    (0: none)."""
    from harbor_ns.builder import _sets_jvm_opts
    from harbor_ns.sandbox import Sandbox
    task_dir = Path(task_dir)
    meta = json.loads((task_dir / 'tasksynth.json').read_text())['meta']
    label = f'tasksynth-{task_dir.name}'
    t0 = time.time()
    res = {'ok': False, 'errors': [], 'warnings': [], 'snapshot': {}}
    env_dir = task_dir / 'environment'
    # the key harbor_ns gives this build (ImageStore.build with the default Dockerfile, no tmux), to delete the image
    # once the check is done (images are not reused; harbor_ns itself deletes a failed build's directory, which r6,
    # 1000 specs per job, once filled 200 GB of scratch with)
    key = store.context_key(env_dir, 'Dockerfile' + ('+jvmproxy' if _sets_jvm_opts(env_dir / 'Dockerfile') else ''))
    _IMAGE_USERS[key] = _IMAGE_USERS.get(key, 0) + 1
    memory_mb = memory_mb or task_memory_mb(task_dir)
    res['memory_mb'] = memory_mb
    # pinned against harbor_ns eviction while the check sandbox runs on it (it is no harbor trial: #451 lesson 7)
    pin = f'check-{os.getpid()}-{secrets.token_hex(4)}'
    try:
        key, rootfs, imeta = await asyncio.wait_for(
            store.build_pinned(env_dir, label=label, token=pin, memory_mb=build_memory_mb or None), timeout)
    except Exception as e:  # noqa: BLE001  (BuildError, timeout)
        res['stage'] = 'build'
        log = failed_build_log(store, label)
        release_image(store, key)
        if 'No space left on device' in f'{e}\n{log}' or 'could not be pinned' in str(e):
            # a pin that failed (harbor_ns build_pinned: the scratch is full, or evicted by other trials) is never the
            # generator's build
            res['infra'] = f"{'scratch full' if 'No space left' in f'{e}{log}' else 'image pin failed'}: {str(e)[-300:]}"
            res['build_seconds'] = round(time.time() - t0)
            return res
        failures = network_failures(store, label, timed_out=isinstance(e, asyncio.TimeoutError))
        if failures:
            # downloads failed at the cluster proxy or the package cache after harbor_ns's network rebuilds: an outage,
            # not the generator's package (round.checked retries it later)
            res['infra'] = f'network: {failures} downloads failed at the proxy or package cache: {str(e)[-200:]}'
            res['build_seconds'] = round(time.time() - t0)
            return res
        if isinstance(e, asyncio.TimeoutError):
            head = (f'The image build did not finish within {timeout // 60} minutes. Setup must finish in a few '
                    'minutes: generate data with fast code, start no long-running or interactive process, and '
                    'prefer packages with prebuilt wheels.')
        else:
            head = f'The image build failed: {str(e)[-500:]}'
        res['errors'].append(f'{head}\nBuild log tail:\n{log}')
        res['build_seconds'] = round(time.time() - t0)
        return res
    res['build_seconds'] = round(time.time() - t0)
    res.update(fetch_record(store.image_dir(key)))
    box = Sandbox(cfg=store.rt, lower=[rootfs], hostname='task', base_env={**imeta.env, 'HOME': '/root'},
                  internet=False, memory_limit_mb=memory_mb, memory_guard_required=False, memory_poll_s=0.5)
    try:
        await box.start()
        # The entrypoint starts the services and puts back setup's /etc/hosts.
        if meta.get('services') or '/usr/local/bin/tasksynth-entrypoint' in (imeta.entrypoint or []):
            await box.exec('setsid nohup /usr/local/bin/tasksynth-entrypoint sleep 7200 >/dev/null 2>&1 &', timeout=30)
            for _ in range(36):
                r = await box.exec("pgrep -xf 'sleep 7200' >/dev/null && echo up", timeout=20)
                if 'up' in r.stdout:
                    break
                await asyncio.sleep(5)
            else:
                res['errors'].append('The task entrypoint (services script) did not finish within 180 s (the services '
                                     'script must start daemons in the background and return once they accept '
                                     'connections).')
        cfg = {'inputs': list(meta.get('inputs') or []), 'outputs': list(meta.get('outputs') or []),
               'defect_check': meta.get('defect_check') or '',
               'mentioned': mentioned_paths((task_dir / 'instruction.md').read_text())}
        r = await box.exec('python3 -c "$TASKSYNTH_INSIDE"', env={'TASKSYNTH_INSIDE': INSIDE}, timeout=300,
                           stdin=(json.dumps(cfg) + '\n').encode())
        line = next((l for l in r.stdout.splitlines() if l.startswith('TASKSYNTH_CHECK ')), None)
        if line is None:
            out = (r.stdout + r.stderr)[-800:]
            if 'failed to execute /bin/sh' in out:
                res['infra'] = f'sandbox has no /bin/sh: {out[-300:]}'
            else:
                res['errors'].append(f'The environment check could not run: {out}')
            return res
        snap = json.loads(line[len('TASKSYNTH_CHECK '):])
        res['snapshot'] = snap
    finally:
        try:
            await box.stop()
        finally:
            store.unmark_in_use(store.image_dir(key), pin)
            release_image(store, key)
            if box.memory_kills:   # also when the inside check itself was the process killed
                k = box.memory_kills[0]
                res['errors'].insert(0, f'The task environment ran out of memory: its processes used '
                                        f'{k["total_pss_kib"] / 2**20:.1f} GB, over the task\'s memory_mb ({memory_mb} MB), '
                                        f'and `{k["comm"]}` was killed. Setup, services and checks must stay well within '
                                        f'memory_mb.')
                res['memory_kills'] = box.memory_kills[:5]
    e = res['errors']
    if snap['inputs_missing']:
        e.append(f'Declared inputs do not exist in the built image: {snap["inputs_missing"][:10]}')
    if snap['outputs_present']:
        e.append(f'Declared outputs already exist at the start (the agent must create them): {snap["outputs_present"][:10]}')
    if snap['leftover']:
        e.append('/tasksynth still exists after setup.')
    if meta.get('defect_check') and snap.get('defect_rc') != 0:
        e.append(f'defect_check did not exit 0 in the built image (rc={snap.get("defect_rc")}): {snap.get("defect_out", "")}')
    if snap['mentioned_missing_parent']:
        e.append('The instruction names paths that do not exist and whose parent directory does not exist either: '
                 f'{snap["mentioned_missing_parent"][:10]}')
    if 'services failed' in snap.get('services_log', ''):
        e.append(f'The services script failed: {snap["services_log"][-800:]}')
    if snap['mentioned_undeclared']:
        res['warnings'].append(f'instruction paths neither present nor declared outputs: {snap["mentioned_undeclared"][:10]}')
    if snap.get('mentioned_not_fs'):
        res['warnings'].append(f'instruction paths whose top-level directory does not exist (routes or placeholders?): '
                               f'{snap["mentioned_not_fs"][:10]}')
    res['ok'] = not e
    return res


def make_store():
    """harbor_ns image store on node scratch; task images are never persisted to /data."""
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    from harbor_ns.cli import default_store
    return default_store()
