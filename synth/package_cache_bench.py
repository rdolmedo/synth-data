"""Validation of the package cache (issue #60): replayed builds with the cache on vs off, and load on the cluster proxy.

Every request a build makes to the cluster proxy goes through a counting relay (CountingRelay: an HTTP/1.1 forwarder,
keep-alive and CONNECT, that labels requests by the proxy user name), in both arms, so "requests the cluster proxy
sees" is measured the same way: plain-HTTP requests plus CONNECT tunnels (requests inside a tunnel are invisible to
the proxy too). The cache's own upstream traffic goes through the relay as well (label "cache").

  python -m synth.package_cache_bench sample --out DIR [--n 600]          # stratified r6a-h tasks
  python -m synth.package_cache_bench replay --out DIR                     # inside a CPU job (submit-replay)
  python -m synth.package_cache_bench submit-replay --out DIR
  python -m synth.package_cache_bench report --out DIR
"""
import argparse
import asyncio
import base64
import contextlib
import contextvars
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import time

ROUNDS = Path('/data/aai/tbench/tasksynth/rounds')
ARM = contextvars.ContextVar('pkgcache_bench_arm', default=None)


# ---------------------------------------------------------------------- counting relay to the cluster proxy
class CountingRelay:
    def __init__(self, upstream='http://proxy.internal:8080'):
        from urllib.parse import urlsplit
        u = urlsplit(upstream)
        self.up_host, self.up_port = u.hostname, u.port or 8080
        self.tags = {}
        self.hosts = {}     # origin host (CONNECT target or URL host) -> requests, bytes_down
        self.port = None
        self.errors = 0
        self.down = False   # failure injection: refuse every request
        self.delays = []    # failure injection: [(url substring, seconds)] held before forwarding

    def count(self, tag, key, n=1, host=None):
        t = self.tags.setdefault(tag or '-', {'http': 0, 'connect': 0, 'bytes_down': 0, 'bytes_up': 0, 'errors': 0,
                                               'connections': 0})
        t[key] += n
        if host and key in ('http', 'connect', 'bytes_down'):
            h = self.hosts.setdefault(host, {'requests': 0, 'bytes_down': 0})
            h['requests' if key != 'bytes_down' else 'bytes_down'] += n

    async def start(self, host='127.0.0.1'):
        self.server = await asyncio.start_server(self.handle, host, 0, backlog=2048, limit=1 << 17)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def close(self):
        self.server.close()

    @staticmethod
    def tag_of(headers):
        for k, v in headers:
            if k.lower() == 'proxy-authorization' and v.lower().startswith('basic '):
                with contextlib.suppress(ValueError, UnicodeDecodeError):
                    return base64.b64decode(v[6:].strip()).decode().split(':', 1)[0]
        return None

    async def handle(self, reader, writer):
        up_r = up_w = None
        tag = None
        try:
            while True:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 300)
                except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                    return
                line, *rest = head.decode('latin-1').split('\r\n')
                method, target, version = line.split(' ', 2)
                headers = [tuple(x.strip() for x in h.split(':', 1)) for h in rest if ':' in h]
                tag = self.tag_of(headers) or tag
                fwd = [(k, v) for k, v in headers if k.lower() != 'proxy-authorization']
                out = '\r\n'.join([line, *(f'{k}: {v}' for k, v in fwd), '', '']).encode('latin-1')
                if self.down:
                    self.count(tag, 'errors')
                    writer.close()
                    return
                if up_w is None:
                    try:
                        up_r, up_w = await asyncio.wait_for(asyncio.open_connection(self.up_host, self.up_port), 30)
                    except (OSError, asyncio.TimeoutError):
                        self.count(tag, 'errors')
                        self.errors += 1
                        writer.write(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
                        await writer.drain()
                        return
                    self.count(tag, 'connections')
                if method == 'CONNECT':
                    host = target.rsplit(':', 1)[0]
                    self.count(tag, 'connect', host=host)
                    up_w.write(out)
                    await up_w.drain()
                    await asyncio.gather(self.pipe(reader, up_w, tag, 'bytes_up'),
                                         self.pipe(up_r, writer, tag, 'bytes_down', host))
                    return
                from urllib.parse import urlsplit
                host = urlsplit(target).hostname
                self.count(tag, 'http', host=host)
                h = {k.lower(): v for k, v in headers}
                body = await reader.readexactly(int(h['content-length'])) if h.get('content-length') else b''
                for pat, secs in self.delays:
                    if pat in target:
                        await asyncio.sleep(secs)
                up_w.write(out + body)
                await up_w.drain()
                self.count(tag, 'bytes_up', len(out) + len(body))
                resp = await asyncio.wait_for(up_r.readuntil(b'\r\n\r\n'), 300)
                writer.write(resp)
                self.count(tag, 'bytes_down', len(resp), host)
                rl, *rh = resp.decode('latin-1').split('\r\n')
                status = int(rl.split(' ', 2)[1])
                rhd = {k.strip().lower(): v.strip() for k, _, v in (x.partition(':') for x in rh if x)}
                close = 'close' in rhd.get('connection', '').lower() or 'close' in h.get('connection', '').lower() \
                    or 'close' in h.get('proxy-connection', '').lower()
                if method == 'HEAD' or status in (204, 304) or 100 <= status < 200:
                    pass
                elif 'content-length' in rhd:
                    await self.copy(up_r, writer, int(rhd['content-length']), tag, host)
                elif 'chunked' in rhd.get('transfer-encoding', '').lower():
                    while True:
                        size_line = await up_r.readuntil(b'\r\n')
                        writer.write(size_line)
                        size = int(size_line.split(b';')[0].strip(), 16)
                        if size == 0:
                            while (t := await up_r.readuntil(b'\r\n')) != b'\r\n':
                                writer.write(t)
                            writer.write(b'\r\n')
                            break
                        await self.copy(up_r, writer, size + 2, tag, host)
                else:
                    await self.pipe(up_r, writer, tag, 'bytes_down', host)
                    return
                await writer.drain()
                if close:
                    return
        except (ConnectionError, OSError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError,
                IndexError, asyncio.TimeoutError):
            self.count(tag, 'errors')
        finally:
            for w in (writer, up_w):
                if w is not None:
                    with contextlib.suppress(Exception):
                        w.close()

    async def copy(self, r, w, n, tag, host=None):
        while n > 0:
            data = await r.read(min(n, 1 << 18))
            if not data:
                raise ConnectionError('upstream closed mid-body')
            w.write(data)
            n -= len(data)
            self.count(tag, 'bytes_down', len(data), host)
            await w.drain()

    async def pipe(self, r, w, tag, key, host=None):
        try:
            while data := await r.read(1 << 18):
                w.write(data)
                self.count(tag, key, len(data), host)
                await w.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                w.close()


# ---------------------------------------------------------------------- installed versions of a built rootfs
def installed(rootfs):
    """{'dpkg:<name>': version, 'py:<name>': version} of a rootfs (dpkg status; *.dist-info / *.egg-info dirs)."""
    rootfs = Path(rootfs)
    out = {}
    status = rootfs / 'var/lib/dpkg/status'
    if status.exists():
        for block in status.read_text(errors='replace').split('\n\n'):
            f = dict(re.findall(r'^(Package|Status|Version): (.*)$', block, re.M))
            if f.get('Package') and 'installed' in f.get('Status', '') and 'not-installed' not in f.get('Status', ''):
                out[f'dpkg:{f["Package"]}'] = f.get('Version')
    for base in ('usr/lib/python3', 'usr/lib/python3.12', 'usr/local/lib', 'opt', 'app', 'root', 'srv', 'home'):
        b = rootfs / base
        if not b.is_dir():
            continue
        for dirpath, dirs, _ in os.walk(b):
            depth = dirpath[len(str(b)):].count('/')
            if depth > 6:
                dirs[:] = []
                continue
            for d in list(dirs):
                m = re.match(r'^(.+?)-([0-9][^-]*?)\.(dist-info|egg-info)$', d)
                if m:
                    where = 'sys' if dirpath.startswith(str(rootfs / 'usr')) else 'venv'
                    out[f'py-{where}:{m.group(1).lower().replace("_", "-")}'] = m.group(2)
                    dirs.remove(d)
            dirs[:] = [d for d in dirs if d not in ('node_modules', '.git', '__pycache__', 'proc', 'sys', 'dev')]
    return out


# ---------------------------------------------------------------------- sample
def sample(out, n=600, seed=60):
    rows = [json.loads(line) for line in (Path(out).parent / 'analysis' / 'scan.jsonl').read_text().splitlines()] \
        if (Path(out).parent / 'analysis' / 'scan.jsonl').exists() else \
        [json.loads(line) for line in Path('/data/aai/tbench/tasksynth/pkgcache/analysis/scan.jsonl').read_text().splitlines()]
    rng = random.Random(seed)
    strata = {'apt': [], 'pip': [], 'apt+pip': [], 'none': []}
    for r in rows:
        apt, pip = r['apt_install'], r['pip_install']
        strata['apt+pip' if apt and pip else 'apt' if apt else 'pip' if pip else 'none'].append(r)
    quota = {'apt': int(n * 0.3), 'pip': int(n * 0.25), 'apt+pip': int(n * 0.25), 'none': int(n * 0.2)}
    chosen = {}
    for k, q in quota.items():
        for r in rng.sample(strata[k], min(q, len(strata[k]))):
            chosen[r['task']] = {**r, 'stratum': k}
    for r in rows:   # every task that configures pip, and some npm / curl / git / failing builds
        if r['index_url'] or (r['npm'] or r['curl_wget'] or r['git_clone']) and rng.random() < 0.3 \
                or r['status'] == 'env_fail' and rng.random() < 0.08:
            s = 'apt+pip' if r['apt_install'] and r['pip_install'] else 'apt' if r['apt_install'] else \
                'pip' if r['pip_install'] else 'none'
            chosen.setdefault(r['task'], {**r, 'stratum': s})
    items = sorted(chosen.values(), key=lambda r: r['task'])
    for r in items:
        r['dir'] = str(ROUNDS / r['round'] / 'tasks' / r['task'])
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / 'sample.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in items))
    from collections import Counter
    print(len(items), Counter(r['stratum'] for r in items), 'pip-config', sum(r['index_url'] for r in items),
          'env_fail', sum(r['status'] == 'env_fail' for r in items))


# ---------------------------------------------------------------------- replay
class ProxyWatch:
    """Cluster proxy health: a small direct request every 30 s; dispatch pauses after 2 failures in a row and
    resumes after 2 successes (the bench must never add load to a struggling proxy)."""

    def __init__(self, log):
        self.ok = True
        self.fail = self.good = 0
        self.log = log
        self.history = []

    async def probe(self):
        t0 = time.time()
        p = await asyncio.create_subprocess_exec(
            'curl', '-s', '-o', '/dev/null', '-m', '20', '-x', 'http://proxy.internal:8080', '-w', '%{http_code}',
            'http://archive.ubuntu.com/ubuntu/dists/noble/Release.gpg', stdout=asyncio.subprocess.PIPE)
        out = (await p.communicate())[0].decode().strip()
        return out == '200', round(time.time() - t0, 2)

    async def run(self):
        while True:
            ok, secs = await self.probe()
            self.history.append((time.time(), ok, secs))
            if ok:
                self.fail, self.good = 0, self.good + 1
                if not self.ok and self.good >= 2:
                    self.ok = True
                    self.log(f'proxy probe recovered ({secs}s): dispatch resumes')
            else:
                self.good, self.fail = 0, self.fail + 1
                if self.ok and self.fail >= 2:
                    self.ok = False
                    self.log('proxy probe failed twice: dispatch paused')
            await asyncio.sleep(30)


async def replay(a):
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    import harbor_ns.builder as hb
    from harbor_ns.cli import default_store
    from harbor_ns.sandbox import package_cache_env as real_env
    from .package_cache import CacheProcess, publish
    out = Path(a.out)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkgbench')
    logf = open(out / 'replay.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()

    # one relay per arm and one for the cache's own upstream traffic: pip sends CONNECT without the proxy URL's
    # credentials, so its tunnels arrive untagged; separate relays keep them on the right side of the comparison
    relays = {k: await CountingRelay().start() for k in ('off', 'on', 'cache')}
    relay_url = {k: f'127.0.0.1:{r.port}' for k, r in relays.items()}
    os.environ['PKGCACHE_UPSTREAM'] = f'http://{relay_url["cache"]}'

    def relay_tags():
        return {'relays': {k: r.tags for k, r in relays.items()}, 'hosts': {k: r.hosts for k, r in relays.items()}}

    def bench_env(tag='', pip=True):
        arm = ARM.get() or 'on'
        via_relay = {k: f'http://{tag}@{relay_url[arm]}' for k in ('http_proxy', 'HTTP_PROXY', 'https_proxy',
                                                                     'HTTPS_PROXY')}
        if arm == 'off':
            return via_relay
        env = real_env(tag, pip)
        return {**via_relay, **env}   # https stays on the relay (cluster proxy), http and pip on the cache
    hb.package_cache_env = bench_env

    store = default_store()
    tasks = [json.loads(line) for line in (out / 'sample.jsonl').read_text().splitlines()]
    if a.limit:
        tasks = tasks[:a.limit]
    results = out / 'results.jsonl'
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            r = json.loads(line)
            done.add((r['phase'], r['arm'], r['task']))
    watch = ProxyWatch(log)
    watch_task = asyncio.ensure_future(watch.run())
    slots = asyncio.Semaphore(a.parallel)

    async def build(phase, arm, t):
        if (phase, arm, t['task']) in done:
            return
        while not watch.ok:
            await asyncio.sleep(10)
        async with slots:
            ARM.set(arm)
            label = f'{phase}-{arm}-{t["task"]}'
            t0 = time.time()
            rec = {'phase': phase, 'arm': arm, 'task': t['task'], 'stratum': t['stratum'], 'start': t0}
            key = None
            # each (phase, arm) builds its own copy of the context: a marker file makes the image keys differ, so
            # harbor_ns never hands one arm the other's image (identical contexts share a key and a build lock)
            ctx = scratch / 'ctx' / label
            shutil.rmtree(ctx, ignore_errors=True)
            await asyncio.to_thread(shutil.copytree, Path(t['dir']) / 'environment', ctx, symlinks=True)
            (ctx / f'.pkgcache-bench-{phase}-{arm}').write_text(label)
            try:
                key, rootfs, meta = await asyncio.wait_for(store.build(ctx, label=label), 1800)
                rec['ok'] = True
                rec['versions'] = await asyncio.to_thread(installed, rootfs)
                with contextlib.suppress(OSError, ValueError):
                    rec['fetch'] = json.loads((store.image_dir(key) / 'fetch.json').read_text())
                rec['tag'] = (rec.get('fetch') or {}).get('tag')
            except Exception as e:  # noqa: BLE001
                rec['ok'] = False
                rec['error'] = f'{type(e).__name__}: {str(e)[-300:]}'
                logs = sorted((store.shared_root / 'logs').glob(f'{label}.*.failed.log'))
                if logs:
                    text = logs[-1].read_text(errors='replace')
                    rec['proxy_failures'] = hb.proxy_failures(text.rsplit('rebuilding from scratch', 1)[-1], 8080)
                    rec['log_tail'] = text[-1500:]
            finally:
                shutil.rmtree(ctx, ignore_errors=True)
                if key is not None:
                    d = store.image_dir(key)
                    subprocess.run(['chmod', '-R', 'u+rwX', str(d)], capture_output=True)
                    shutil.rmtree(d, ignore_errors=True)
            rec['seconds'] = round(time.time() - t0, 1)
            with open(results, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            log(f'{phase} {arm} {t["task"]} {"ok" if rec["ok"] else "FAIL"} {rec["seconds"]}s '
                f'{(rec.get("fetch") or {}).get("upstream", "")}')

    async def resources(stop):
        while not stop.is_set():
            du = shutil.disk_usage(scratch)
            entry = {'t': time.time(), 'scratch_used_gib': round(du.used / 2**30, 1),
                     'cache_gib': round(sum(f.stat().st_size for f in (scratch / 'pkgcache').rglob('*')
                                            if f.is_file()) / 2**30, 2) if (scratch / 'pkgcache').exists() else 0,
                     'relay': relay_sum(relays['cache'].tags)}
            with open(out / 'resources.jsonl', 'a') as f:
                f.write(json.dumps(entry) + '\n')
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(stop.wait(), 120)

    stop = asyncio.Event()
    res_task = asyncio.ensure_future(resources(stop))
    seed_dir = out / 'seed'
    try:
        # phase cold: cache-off and cache-on (empty cache) interleaved in time, same build slots
        cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', seed_dir=None, log=log,
                                   args=['--seed-dir', 'none', '--max-gib', '30']).start()
        jobs = [('cold', arm, t) for t in tasks for arm in ('off', 'on')]
        random.Random(1).shuffle(jobs)
        log(f'cold phase: {len(jobs)} builds, {a.parallel} at once')
        await asyncio.gather(*(build(*j) for j in jobs))
        stats = cache.stats()
        (out / 'cache-stats-cold.json').write_text(json.dumps(stats, indent=1))
        (out / 'relay-cold.json').write_text(json.dumps(relay_tags(), indent=1))
        await cache.close()
        keep_log(scratch / 'pkgcache', out / 'cache-requests-cold.jsonl')
        summary = await asyncio.to_thread(publish, scratch / 'pkgcache', seed_dir, 12.0, scratch)
        log(f'seed published: {summary}')
        shutil.rmtree(scratch / 'pkgcache', ignore_errors=True)
        # phase warm: a new cache restored from the seed (a fresh round job), cache-on only, plus cache-off again
        # interleaved for version parity at the same time
        cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', seed_dir=seed_dir, log=log,
                                   args=['--max-gib', '30']).start()
        jobs = [('warm', arm, t) for t in tasks for arm in ('on', 'off')]
        random.Random(2).shuffle(jobs)
        log(f'warm phase: {len(jobs)} builds')
        await asyncio.gather(*(build(*j) for j in jobs))
        (out / 'cache-stats-warm.json').write_text(json.dumps(cache.stats(), indent=1))
        await cache.close()
        keep_log(scratch / 'pkgcache', out / 'cache-requests-warm.jsonl')
        if a.publish_production:
            from .package_cache import SEED_DIR
            summary = await asyncio.to_thread(publish, scratch / 'pkgcache', SEED_DIR, 12.0, scratch)
            log(f'production seed published: {summary}')
    finally:
        stop.set()
        await res_task
        watch_task.cancel()
        (out / 'relay.json').write_text(json.dumps(relay_tags(), indent=1))
        (out / 'proxy-probes.json').write_text(json.dumps(watch.history))
        for r in relays.values():
            await r.close()
        log('replay done')


# ---------------------------------------------------------------------- report
def pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def relay_sum(tags, keys=('http', 'connect', 'bytes_down', 'bytes_up', 'errors')):
    return {k: sum(t.get(k, 0) for t in tags.values()) for k in keys}


def relay_diff(final, cold):
    return {tag: {k: v - cold.get(tag, {}).get(k, 0) for k, v in t.items()} for tag, t in final.items()}


def report(out):
    """Markdown tables for the replay: parity, versions, proxy requests and bytes, hit rates, build times, scratch."""
    from collections import Counter, defaultdict
    out = Path(out)
    rows = [json.loads(line) for line in (out / 'results.jsonl').read_text().splitlines()]
    sample = {json.loads(line)['task']: json.loads(line) for line in (out / 'sample.jsonl').read_text().splitlines()}
    by = defaultdict(dict)
    for r in rows:
        by[(r['phase'], r['task'])][r['arm']] = r
    relay_cold = json.loads((out / 'relay-cold.json').read_text()) if (out / 'relay-cold.json').exists() else {}
    relay_final = json.loads((out / 'relay.json').read_text()) if (out / 'relay.json').exists() else relay_cold
    lines = [f'# Package cache replay ({len(sample)} tasks)', '']

    split = 'relays' in relay_final or 'relays' in relay_cold

    def phase_relay(phase):
        """-> {'off': tags, 'on': tags, 'cache': tags, 'untagged': tags} for the phase. Single-relay runs (replay1)
        put the cache's traffic and untagged build traffic (pip tunnels) together under 'untagged'."""
        if split:
            cold = relay_cold.get('relays', {})
            fin = relay_final.get('relays', cold)
            rel = cold if phase == 'cold' else {k: relay_diff(fin[k], cold.get(k, {})) for k in fin}
            return {**rel, 'untagged': {}}
        rel = relay_cold if phase == 'cold' else relay_diff(relay_final, relay_cold)
        by = {'off': {}, 'on': {}, 'cache': {}, 'untagged': {}}
        for tag, t in rel.items():
            arm = tag.split('-')[1] if tag.startswith(phase + '-') and tag.count('-') >= 2 else None
            by[arm if arm in ('off', 'on') else 'untagged'][tag] = t
        return by

    for phase in ('cold', 'warm'):
        pairs = [v for (ph, _), v in by.items() if ph == phase and 'on' in v and 'off' in v]
        if not pairs:
            continue
        c = Counter((v['off']['ok'], v['on']['ok']) for v in pairs)
        lines += [f'## {phase}: {len(pairs)} tasks built with the cache off and on', '',
                  '| off \\ on | ok | failed |', '| --- | --- | --- |',
                  f'| ok | {c[True, True]} | {c[True, False]} |', f'| failed | {c[False, True]} | {c[False, False]} |', '']
        disc = [v for v in pairs if v['off']['ok'] != v['on']['ok']]
        for v in disc:
            lines.append(f'- {v["on"]["task"]}: off {"ok" if v["off"]["ok"] else v["off"].get("error", "")[:120]} / '
                         f'on {"ok" if v["on"]["ok"] else v["on"].get("error", "")[:120]}')
        same = diff = 0
        diffs = Counter()
        for v in pairs:
            if v['off']['ok'] and v['on']['ok']:
                a, b = v['off'].get('versions') or {}, v['on'].get('versions') or {}
                if a == b:
                    same += 1
                else:
                    diff += 1
                    for k in set(a) | set(b):
                        if a.get(k) != b.get(k):
                            diffs[f'{k} {a.get(k)} -> {b.get(k)}'] += 1
        lines += ['', f'Installed versions (dpkg + Python dists), both arms ok: {same} identical, {diff} different.', '']
        lines += [f'- {k} ({n} tasks)' for k, n in diffs.most_common(20)]
        rel = phase_relay(phase)
        stats = json.loads((out / f'cache-stats-{phase}.json').read_text()) \
            if (out / f'cache-stats-{phase}.json').exists() else {}
        n_req = lambda t: t.get('http', 0) + t.get('connect', 0)   # noqa: E731  (what the proxy sees)
        off, on, cache, untagged = (relay_sum(rel[k]) for k in ('off', 'on', 'cache', 'untagged'))
        per_off = [n_req(rel['off'].get(v['off'].get('tag') or '', {})) for v in pairs if v['off'].get('tag')]
        per_on = [v['on']['fetch'].get('upstream', 0) + n_req(rel['on'].get(v['on'].get('tag') or '', {}))
                  for v in pairs if v['on'].get('fetch')]
        lines += ['', f'### {phase}: load on the cluster proxy (requests it sees: plain HTTP + CONNECT tunnels)', '']
        if split:
            on_total = n_req(on) + n_req(cache)
            on_bytes = on['bytes_down'] + cache['bytes_down']
            lines += ['| | cache off | cache on |', '| --- | --- | --- |',
                      f'| proxy requests, all builds | {n_req(off)} | {on_total} (builds {n_req(on)} + cache '
                      f'{n_req(cache)}) |',
                      f'| reduction | | {100 * (1 - on_total / max(1, n_req(off))):.1f}% |',
                      f'| bytes from the proxy | {off["bytes_down"] / 2**30:.2f} GiB | {on_bytes / 2**30:.2f} GiB |',
                      f'| bytes reduction | | {100 * (1 - on_bytes / max(1, off["bytes_down"])):.1f}% |']
        else:
            # replay1: pip's tunnels (no credentials on CONNECT) and the cache's own traffic share one bucket. Bound:
            # all of it charged to the cache arm. Bytes: the cache's upstream bytes are exact (its own counter).
            up = stats.get('upstream_bytes', 0)
            on_hi = n_req(on) + n_req(untagged)
            off_bytes = off['bytes_down'] + max(0, untagged['bytes_down'] - up)
            on_bytes = on['bytes_down'] + up
            lines += ['| | cache off | cache on |', '| --- | --- | --- |',
                      f'| proxy requests, all builds | >= {n_req(off)} (tagged; pip tunnels arrive untagged) | '
                      f'<= {on_hi} (builds {n_req(on)} + cache and all untagged traffic {n_req(untagged)}) |',
                      f'| reduction (lower bound) | | {100 * (1 - on_hi / max(1, n_req(off))):.1f}% |',
                      f'| bytes from the proxy | {off_bytes / 2**30:.2f} GiB (tagged {off["bytes_down"] / 2**30:.2f} + '
                      f'untagged minus the cache\'s) | {on_bytes / 2**30:.2f} GiB (builds {on["bytes_down"] / 2**30:.2f}'
                      f' + cache {up / 2**30:.2f}) |',
                      f'| bytes reduction | | {100 * (1 - on_bytes / max(1, off_bytes)):.1f}% |']
        lines += [f'| per build p50 / p90 / mean | {pctl(per_off, .5)} / {pctl(per_off, .9)} / '
                  f'{sum(per_off) / max(1, len(per_off)):.1f}{"" if split else " (apt GETs; tagged only)"} | '
                  f'{pctl(per_on, .5)} / {pctl(per_on, .9)} / {sum(per_on) / max(1, len(per_on)):.1f} '
                  f'(cache log: misses + revalidations + own tunnels) |', '']
        if stats:
            req = max(1, stats['requests'])
            lines += [f'Cache ({phase}): {stats["requests"]} requests: hit {stats["hit"]}, coalesced {stats["coalesced"]}, '
                      f'revalidated {stats["revalidated"]}, miss {stats["miss"]}, stale {stats["stale"]}, '
                      f'bypass {stats["bypass"]}, error {stats["error"]}; hit rate {100 * stats["hit"] / req:.1f}% '
                      f'(no download: {100 * (stats["hit"] + stats["coalesced"] + stats["revalidated"]) / req:.1f}%); '
                      f'bytes served {stats["bytes_out"] / 2**30:.2f} GiB, fetched upstream '
                      f'{stats["upstream_bytes"] / 2**30:.2f} GiB; store {stats["store_bytes"] / 2**30:.2f} GiB, '
                      f'{stats["entries"]} files; upstream errors {stats["upstream_errors"]}.', '']
        lines += ['### Build seconds (p50 / p90), by stratum', '', '| stratum | n | cache off | cache on |',
                  '| --- | --- | --- | --- |']
        for st in ('apt', 'pip', 'apt+pip', 'none', 'all'):
            sel = [v for v in pairs if st == 'all' or sample[v['on']['task']]['stratum'] == st]
            off = [v['off']['seconds'] for v in sel if v['off']['ok']]
            on = [v['on']['seconds'] for v in sel if v['on']['ok']]
            lines.append(f'| {st} | {len(sel)} | {pctl(off, .5)} / {pctl(off, .9)} | {pctl(on, .5)} / {pctl(on, .9)} |')
        lines.append('')
    if (out / 'resources.jsonl').exists():
        res = [json.loads(line) for line in (out / 'resources.jsonl').read_text().splitlines()]
        lines += [f'Scratch: max {max(r["scratch_used_gib"] for r in res)} GiB used (builds + cache); cache store max '
                  f'{max(r["cache_gib"] for r in res)} GiB.', '']
    if (out / 'proxy-probes.json').exists():
        probes = json.loads((out / 'proxy-probes.json').read_text())
        lines += [f'Cluster proxy probe (every 30 s, direct): {len(probes)} probes, {sum(not p[1] for p in probes)} failed, '
                  f'p50 {pctl([p[2] for p in probes], .5)} s, max {max((p[2] for p in probes), default=0)} s.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(out / 'REPORT.md')


def stress_report(out):
    """Per minute and per load phase: requests/s, GB/s, latency, errors, integrity, cache CPU/RSS, upstream."""
    out = Path(out)
    plan = None
    tl = [json.loads(x) for x in (out / 'timeline.jsonl').read_text().splitlines()]
    t0 = tl[0]['t'] - tl[0]['elapsed']
    loads = {}
    for f in sorted(out.glob('load-*.jsonl')):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            loads.setdefault((f.stem[5:], r['t']), []).append(r)
    windows = {}
    for (name, t), rs in loads.items():
        w = windows.setdefault(name, {'n': 0, 'err': 0, 'bytes': 0, 'bad': 0, 'p50': [], 'p99': [], 't': [],
                                      'why': {}})
        for r in rs:
            w['n'] += r['n']
            w['err'] += r['err']
            w['bytes'] += r['bytes']
            w['bad'] += r['bad']
            for k, v in (r.get('why') or {}).items():
                w['why'][k] = w['why'].get(k, 0) + v
            if r['p50'] is not None:
                w['p50'].append(r['p50'])
                w['p99'].append(r['p99'])
        w['t'].append(t)
    lines = ['| load | duration | requests | req/s | GB/s | p50 latency (s) | p99 latency (s, worst 10 s) | errors | '
             'integrity mismatches |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, w in windows.items():
        dur = max(w['t']) - min(w['t']) + 10
        lines.append(f'| {name} | {dur / 60:.0f} min | {w["n"]} | {w["n"] / dur:.0f} | {w["bytes"] / dur / 1e9:.2f} | '
                     f'{pctl(w["p50"], .5):.3f} | {max(w["p99"]):.2f} | {w["err"]} | {w["bad"]} |')
    cpu = [r['cache_cpu'] for r in tl if r.get('cache_cpu') is not None]
    lines += ['', f'Cache process: CPU p50 {pctl(cpu, .5)} / max {max(cpu)} cores; RSS max '
              f'{max(r["cache_rss_mib"] for r in tl)} MiB; clients peak {max(r.get("clients_peak") or 0 for r in tl)}; '
              f'upstream requests during the run {tl[-1]["upstream_requests"] - tl[0]["upstream_requests"]}; '
              f'upstream errors {tl[-1]["upstream_errors"] - tl[0]["upstream_errors"]}; proxy probe failures '
              f'{sum(not r["proxy_ok"] for r in tl)} of {len(tl)} samples; stats probes unanswered '
              f'{sum(r.get("requests") is None for r in tl)}; event-loop stalls > 3 s '
              f'{max((r.get("loop_stalls") or 0) for r in tl)}, worst loop lag '
              f'{max((r.get("loop_lag_max_s") or 0) for r in tl)} s.']
    for name, w in windows.items():
        if w['why']:
            lines.append(f'- {name} errors by cause: ' + ', '.join(f'{k} {v}' for k, v in
                                                                 sorted(w['why'].items(), key=lambda x: -x[1])))
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


# ---------------------------------------------------------------------- stress (load on one cache, upstream ~ 0)
def build_sequences(logs, min_requests=1):
    """Per build (tag), its requests in order with the idle gap before each: [(gap_s, target, accept, sha256, size)].
    apt targets stay absolute (sent to the cache as to a proxy); PyPI ones become /pypi/... paths."""
    by = {}
    for path in logs:
        if not Path(path).exists():
            continue
        for line in Path(path).read_text().splitlines():
            e = json.loads(line)
            if not e.get('tag') or e.get('status') not in (200,) or e.get('kind') not in ('apt', 'pypi-page', 'pypi-file'):
                continue
            by.setdefault(e['tag'], []).append(e)
    seqs = []
    for tag, es in by.items():
        es.sort(key=lambda e: e['t'] - e['ms'] / 1000)
        seq, last_end = [], None
        for e in es:
            start = e['t'] - e['ms'] / 1000
            gap = 0.0 if last_end is None else max(0.0, start - last_end)
            last_end = e['t']
            if e['kind'] == 'apt':
                target, accept = e['url'], None
            elif e['kind'] == 'pypi-page':
                target, accept = '/pypi/simple/' + e['url'].rstrip('/').rsplit('/', 1)[-1] + '/', \
                    'application/vnd.pypi.simple.v1+json'
            else:
                target, accept = '/pypi/packages/' + e['url'].split('/packages/', 1)[1], None
            seq.append((min(gap, 120.0), target, accept, e.get('sha256'), e.get('bytes')))
        if len(seq) >= min_requests:
            seqs.append(seq)
    return seqs


def sandbox_sequences(path):
    """S1 attempts' allowed plain-HTTP (apt) requests through the package proxy, as targets."""
    seqs = []
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        t = [q['target'] for q in r['requests'] if q.get('allowed') and q.get('method') == 'GET'
             and str(q.get('target', '')).startswith('http://')]
        if t:
            seqs.append(t)
    return seqs


class Http1:
    """A minimal keep-alive HTTP/1.1 client (apt-like) for the load generator."""

    def __init__(self, port=None, unix=None):
        self.port, self.unix = port, unix
        self.r = self.w = None

    async def request(self, target, accept=None, keep=True):
        import hashlib
        if self.w is None:
            if self.unix:
                self.r, self.w = await asyncio.open_unix_connection(self.unix, limit=1 << 20)
            else:
                self.r, self.w = await asyncio.open_connection('127.0.0.1', self.port, limit=1 << 20)
        hs = f'Accept: {accept}\r\n' if accept else ''
        conn = 'keep-alive' if keep else 'close'
        self.w.write(f'GET {target} HTTP/1.1\r\nHost: x\r\nUser-Agent: pkgcache-bench\r\n{hs}Connection: {conn}\r\n\r\n'
                     .encode())
        await self.w.drain()
        head = await self.r.readuntil(b'\r\n\r\n')
        line, *rest = head.decode('latin-1').split('\r\n')
        h = {k.strip().lower(): v.strip() for k, _, v in (x.partition(':') for x in rest if x)}
        n = int(h.get('content-length', -1))
        sha = hashlib.sha256()
        got = 0
        if n >= 0:
            while got < n:
                data = await self.r.read(min(1 << 20, n - got))
                if not data:
                    raise ConnectionError('short body')
                sha.update(data)
                got += len(data)
        else:
            while data := await self.r.read(1 << 20):
                sha.update(data)
                got += len(data)
        if not keep or 'close' in h.get('connection', '').lower() or n < 0:
            self.close()
        return int(line.split()[1]), got, sha.hexdigest(), h.get('x-pkgcache'), h.get('x-pkgcache-sha256')

    def close(self):
        if self.w is not None:
            with contextlib.suppress(Exception):
                self.w.close()
        self.r = self.w = None


async def load(a):
    """One load-generator process: a.clients clients started over a.ramp seconds, each looping over sequences until
    a.duration. role build: keep-alive to the cache, recorded gaps (x a.gap_scale); role sandbox: one PackageProxy
    per client (unix socket, chained to the cache), one request per connection. Writes per-10 s aggregates."""
    import statistics
    from .package_proxy import PackageProxy
    rng = random.Random(a.seed)
    seqs = json.loads(Path(a.seqs).read_text())
    reg = json.loads(Path(a.registry).read_text())
    t_end = time.time() + a.duration
    buckets = {}
    proxies = []
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    def record(ok, secs, nbytes, bad, why=None):
        b = buckets.setdefault(int(time.time() // 10) * 10, {'n': 0, 'err': 0, 'bytes': 0, 'lat': [], 'bad': 0,
                                                             'why': {}})
        b['n'] += 1
        b['err'] += 0 if ok else 1
        b['bytes'] += nbytes
        b['bad'] += bad
        if why:
            b['why'][why] = b['why'].get(why, 0) + 1
        if len(b['lat']) < 5000:
            b['lat'].append(secs)

    from harbor_ns.sandbox import package_cache
    slow_log = Path(str(a.out).replace('load-', 'slow-'))
    slow_n = [0]

    def slow(target, secs, what, how):
        """Requests that failed or took over 30 s, one line each (first 2,000 per process)."""
        if slow_n[0] < 2000:
            slow_n[0] += 1
            with open(slow_log, 'a') as f:
                f.write(json.dumps({'t': round(time.time(), 1), 'pid': os.getpid(), 'role': a.role, 'target': target,
                                    'secs': round(secs, 1), 'what': what, 'cache': how}) + '\n')

    def cache_reg():
        os.environ['HARBOR_NS_PACKAGE_CACHE'] = a.registry
        return package_cache()

    async def client(i):
        await asyncio.sleep(a.ramp * i / max(1, a.clients))
        if a.role == 'sandbox':
            sock = tmp / f'p{os.getpid()}-{i}.sock'
            px = await PackageProxy(sock, tmp / f'p{os.getpid()}-{i}.jsonl', 'http://127.0.0.1:9', cache=cache_reg,
                                    tag=f'sbx{i}').start()
            proxies.append(px)
            conn = Http1(unix=str(sock))
        else:
            conn = Http1(port=reg['port'])
        while time.time() < t_end:
            seq = rng.choice(seqs)
            for item in seq:
                if time.time() >= t_end:
                    break
                if a.role == 'build':
                    gap, target, accept, sha, size = item
                    if gap * a.gap_scale > 0:
                        await asyncio.sleep(gap * a.gap_scale)
                else:
                    target, accept, sha, size = item, None, None, None
                    await asyncio.sleep(rng.uniform(0, 0.5) * a.gap_scale)
                t0 = time.time()
                try:
                    code, n, got_sha, how, stored_sha = await asyncio.wait_for(
                        conn.request(target, accept, keep=a.role == 'build'), 600)
                    # integrity: the body against the stored file's SHA-256 the cache names, and immutable files
                    # (pool, by-hash, PyPI files) against the hash recorded in the replay. Indexes (InRelease, PyPI
                    # pages) legitimately change upstream between the replay and this run.
                    immutable = '/pool/' in target or '/by-hash/' in target or '/pypi/packages/' in target
                    bad = int(code == 200 and (bool(stored_sha) and got_sha != stored_sha
                                               or immutable and sha is not None and got_sha != sha))
                    record(code == 200, time.time() - t0, n, bad, None if code == 200 else f'http {code}')
                    if code != 200 or time.time() - t0 > 30:
                        slow(target, time.time() - t0, f'http {code}', how)
                except Exception as e:  # noqa: BLE001
                    conn.close()
                    record(False, time.time() - t0, 0, 0, type(e).__name__)
                    slow(target, time.time() - t0, type(e).__name__, None)
            if a.role == 'sandbox':
                await asyncio.sleep(rng.uniform(1, 5) * a.gap_scale)

    async def flush():
        while True:
            await asyncio.sleep(10)
            now_b = int(time.time() // 10) * 10
            done = {k: v for k, v in buckets.items() if k < now_b}
            with open(a.out, 'a') as f:
                for k in sorted(done):
                    v = buckets.pop(k)
                    lat = sorted(v['lat'])
                    f.write(json.dumps({'t': k, 'pid': os.getpid(), 'role': a.role, 'n': v['n'], 'err': v['err'],
                                        'bytes': v['bytes'], 'bad': v['bad'], 'why': v['why'],
                                        'p50': lat[len(lat) // 2] if lat else None,
                                        'p99': lat[int(len(lat) * 0.99)] if lat else None,
                                        'mean': statistics.fmean(lat) if lat else None}) + '\n')
    fl = asyncio.ensure_future(flush())
    await asyncio.gather(*(client(i) for i in range(a.clients)), return_exceptions=True)
    await asyncio.sleep(11)
    fl.cancel()
    for px in proxies:
        await px.close()


def proc_usage(pid):
    """(cpu seconds, rss bytes) of a process."""
    try:
        f = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        rss = int(Path(f'/proc/{pid}/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        return (int(f[11]) + int(f[12])) / os.sysconf('SC_CLK_TCK'), rss
    except (OSError, IndexError, ValueError):
        return None, None


async def stress(a):
    """Load on one package cache: build clients (recorded request sequences and gaps, keep-alive) and sandbox clients
    (one synth.package_proxy each, S1 apt sequences), on a cache restored from the replay seed (warm) or empty (cold,
    upstream through the counting relay). Dispatch pauses (SIGSTOP) while the cluster proxy probe fails."""
    import signal as sig
    from .package_cache import CacheProcess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkgstress')
    scratch.mkdir(parents=True, exist_ok=True)
    logf = open(out / 'stress.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    relay = await CountingRelay().start()
    os.environ['PKGCACHE_UPSTREAM'] = f'http://127.0.0.1:{relay.port}'
    replay_dir = Path(a.replay)
    seqs_b = build_sequences([replay_dir / 'cache-requests-cold.jsonl', replay_dir / 'cache-requests-warm.jsonl'])
    seqs_s = sandbox_sequences(a.sandbox_seqs)
    (scratch / 'seqs-build.json').write_text(json.dumps(seqs_b))
    (scratch / 'seqs-sandbox.json').write_text(json.dumps(seqs_s))
    log(f'{len(seqs_b)} build sequences ({sum(map(len, seqs_b))} requests), {len(seqs_s)} sandbox sequences')
    cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json',
                               seed_dir=None if a.cold else replay_dir / 'seed', log=log,
                               args=(['--seed-dir', 'none'] if a.cold else []) + ['--max-gib', '40']).start()
    # pre-warm: every target once through the cache (8 at a time: the cold part, at or below a round job's rate);
    # targets that do not answer 200 now (S1's by-hash indexes of 10-30 may be gone) are dropped from the sequences,
    # so the load never sends repeated uncacheable requests upstream
    reg = json.loads((scratch / 'pkgcache.json').read_text())
    targets = {(i[1], i[2]) for seq in seqs_b for i in seq} | {(t, None) for seq in seqs_s for t in seq}
    good, sem = set(), asyncio.Semaphore(8)

    async def warm(t):
        async with sem:
            c = Http1(port=reg['port'])
            try:
                code, *_ = await asyncio.wait_for(c.request(t[0], t[1], keep=False), 900)
                if code == 200:
                    good.add(t)
            except Exception:  # noqa: BLE001
                pass
            finally:
                c.close()
    t_warm = time.time()
    await asyncio.gather(*(warm(t) for t in targets))
    seqs_b = [[i for i in seq if (i[1], i[2]) in good] for seq in seqs_b]
    seqs_s = [[t for t in seq if (t, None) in good] for seq in seqs_s]
    seqs_b, seqs_s = [q for q in seqs_b if q], [q for q in seqs_s if q]
    (scratch / 'seqs-build.json').write_text(json.dumps(seqs_b))
    (scratch / 'seqs-sandbox.json').write_text(json.dumps(seqs_s))
    log(f'pre-warm: {len(good)}/{len(targets)} targets ok in {time.time() - t_warm:.0f}s; cache {cache.stats()}')
    (out / 'relay-prewarm.json').write_text(json.dumps(relay.tags, indent=1))
    plan = json.loads(a.plan if a.plan.lstrip().startswith('[') else Path(a.plan).read_text())
    procs = []
    watch = ProxyWatch(log)
    wt = asyncio.ensure_future(watch.run())
    t0 = time.time()
    paused = False

    async def launch(step):
        await asyncio.sleep(max(0, step['at'] - (time.time() - t0)))
        n = step['procs']
        log(f'start {step}')
        for k in range(n):
            cmd = [os.sys.executable, '-m', 'synth.package_cache_bench', 'load', '--role', step['role'],
                   '--seqs', str(scratch / f'seqs-{step["role"]}.json'), '--registry', str(scratch / 'pkgcache.json'),
                   '--clients', str(step['clients'] // n), '--duration', str(step['duration']), '--ramp',
                   str(step.get('ramp', 600)), '--gap-scale', str(step.get('gap_scale', 1.0)),
                   '--seed', str(hash((step['name'], k)) % 10**6), '--tmp', str(scratch / 'socks'),
                   '--out', str(out / f'load-{step["name"]}.jsonl')]
            procs.append(await asyncio.create_subprocess_exec(*cmd))
    launches = [asyncio.ensure_future(launch(st)) for st in plan]
    pid = json.loads((scratch / 'pkgcache.json').read_text())['pid']
    last_cpu, last_t = proc_usage(pid)[0], time.time()
    end = max(st['at'] + st['duration'] + 30 for st in plan)
    while time.time() - t0 < end:
        await asyncio.sleep(10)
        try:
            pid = json.loads((scratch / 'pkgcache.json').read_text())['pid']
        except (OSError, ValueError):
            pass
        cpu, rss = proc_usage(pid)
        now_t = time.time()
        stats = cache.stats() or {}
        entry = {'t': round(now_t), 'elapsed': round(now_t - t0), 'cache_cpu': round((cpu - last_cpu) / (now_t - last_t), 2)
                 if cpu is not None and last_cpu is not None else None, 'cache_rss_mib': round((rss or 0) / 2**20),
                 'relay': relay.tags.get('-', {}), 'proxy_ok': watch.ok, **{k: stats.get(k) for k in (
                     'requests', 'hit', 'miss', 'coalesced', 'revalidated', 'stale', 'bypass', 'error', 'bytes_out',
                     'upstream_requests', 'upstream_bytes', 'upstream_errors', 'clients', 'clients_peak',
                     'store_bytes', 'breaker_open', 'loop_stalls', 'loop_lag_max_s')}}
        last_cpu, last_t = cpu, now_t
        with open(out / 'timeline.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
        if not watch.ok and not paused:
            paused = True
            for p in procs:
                with contextlib.suppress(ProcessLookupError):
                    p.send_signal(sig.SIGSTOP)
            log('cluster proxy probe failing: load generators paused')
        elif watch.ok and paused:
            paused = False
            for p in procs:
                with contextlib.suppress(ProcessLookupError):
                    p.send_signal(sig.SIGCONT)
            log('cluster proxy probe ok: load generators resumed')
    for p in procs:
        with contextlib.suppress(ProcessLookupError):
            p.send_signal(sig.SIGCONT)
        await p.wait()
    for t in launches:
        t.cancel()
    wt.cancel()
    (out / 'cache-stats.json').write_text(json.dumps(cache.stats(), indent=1))
    (out / 'relay.json').write_text(json.dumps(relay.tags, indent=1))
    (out / 'proxy-probes.json').write_text(json.dumps(watch.history))
    keep_log_tail(scratch / 'pkgcache', out / 'cache-errors.jsonl')
    with contextlib.suppress(OSError):   # the cache's stdout: counters and any event-loop stall stacks
        err = (scratch / 'pkgcache.err').read_bytes()
        (out / 'pkgcache.err').write_bytes(err[-(4 << 20):])
    await cache.close()
    log('stress done')


# ---------------------------------------------------------------------- agent installs (offline sandboxes)
VERSIONS_CMD = ("dpkg-query -W -f='dpkg:${Package} ${Version}\\n' 2>/dev/null; "
                "python3 -m pip list --format=freeze 2>/dev/null | sed 's/^/py:/'")


async def agent_run(store, rootfs, imeta, commands, arm, cache_reg, log_dir, pip):
    """The attempt's install commands in an offline sandbox whose package proxy has (or lacks) the cache."""
    import tempfile
    from harbor_ns.environment import offline_bypass_names
    from harbor_ns.sandbox import Sandbox, proxy_env
    from .endpoint_relay import EndpointRelay
    from .package_proxy import DEFAULT_ALLOW, PIP_INDEX, SANDBOX_PORT, SANDBOX_URL, PackageProxy
    os.environ['HARBOR_NS_OFFLINE_PROXY_HOSTS'] = ','.join(DEFAULT_ALLOW)
    env = {**imeta.env, 'HOME': '/root', **proxy_env(offline_bypass_names(rootfs) or None, url=SANDBOX_URL, offline=True)}
    if pip:
        env['PIP_INDEX_URL'] = PIP_INDEX
    box = Sandbox(cfg=store.rt, lower=[rootfs], hostname='task', internet=False, base_env=env)
    await box.start()
    log_dir.mkdir(parents=True, exist_ok=True)
    relay = EndpointRelay(log_dir)
    relay.scratch = tempfile.TemporaryDirectory(prefix='pkgb-', dir='/tmp')
    sock = Path(relay.scratch.name) / 'p.sock'
    proxy = await PackageProxy(sock, log_dir / 'requests.jsonl', os.environ['PKGCACHE_UPSTREAM'],
                               cache=cache_reg if arm != 'off' else None, pip=pip, tag=log_dir.name).start()
    out = {'rcs': [], 'tails': []}
    try:
        ns = await relay.spawn(['nsenter', '-t', str(box.init_pid), '-U', '-n', '--preserve-credentials', '--',
                                '/usr/bin/socat', '-d', '-d', f'TCP4-LISTEN:{SANDBOX_PORT},bind=127.0.0.1,reuseaddr,fork',
                                f'UNIX-CONNECT:{sock}'], 'namespace-relay.log')
        try:
            await relay.wait_for('listening on', 'namespace-relay.log', (ns,), 'Package proxy relay', timeout=10)
        except TimeoutError:
            pass  # as before: the commands run and show the failure
        t0 = time.time()
        for cmd in commands[:8]:
            r = await box.exec(f'cd /app 2>/dev/null; {cmd}', timeout=600)
            out['rcs'].append(r.return_code)
            out['tails'].append(((r.stdout or '') + (r.stderr or ''))[-400:])
        out['seconds'] = round(time.time() - t0, 1)
        v = await box.exec(VERSIONS_CMD, timeout=120)
        out['versions'] = dict(line.split(' ', 1) if ' ' in line else line.split('==', 1)
                               for line in (v.stdout or '').splitlines() if line.strip() and (' ' in line or '==' in line))
    finally:
        await relay.close()
        await proxy.close()
        await box.stop()
    reqs = [json.loads(x) for x in (log_dir / 'requests.jsonl').read_text().splitlines()] \
        if (log_dir / 'requests.jsonl').exists() else []
    from collections import Counter
    out['proxy_log'] = dict(Counter(f'{q.get("method")} {q.get("cache") or "-"}' for q in reqs if q.get('allowed')))
    out['corrupt_marks'] = sum(len(CORRUPT.findall(t)) for t in out['tails'])
    return out


async def agent_replay(a):
    """S1 attempts' install commands, cache off then on (apt via the cache; pip via the cache when --pip), then on
    again (warm). Phases run one after the other; the relay is snapshot between them."""
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    from harbor_ns.cli import default_store
    from harbor_ns.sandbox import package_cache
    from .package_cache import CacheProcess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkgagent')
    logf = open(out / 'agent.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    relay = await CountingRelay().start()
    os.environ['PKGCACHE_UPSTREAM'] = f'http://127.0.0.1:{relay.port}'
    attempts = [json.loads(x) for x in Path(a.attempts).read_text().splitlines()]
    attempts = [r for r in attempts if r['commands']]
    rng = random.Random(11)
    rng.shuffle(attempts)
    attempts = attempts[:a.n]
    cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', seed_dir=a.seed, log=log).start()
    store = default_store()
    images = {}
    if (packed := [d.name for d in sorted(ROUNDS.glob('r6[a-h]')) if (d / 'packed' / 'index.json').exists()]):
        raise SystemExit(f'rounds {packed} are packed (task_synthesis roundpack, #245): this bench reads loose tasks')
    for r in attempts:
        rd = next((d for d in sorted(ROUNDS.glob('r6[a-h]')) if (d / 'tasks' / r['task']).is_dir()), None)
        r['dir'] = str(rd / 'tasks' / r['task']) if rd else None
    attempts = [r for r in attempts if r['dir']]
    slots = asyncio.Semaphore(a.parallel)

    async def image(r):
        if r['task'] not in images:
            images[r['task']] = await store.build(Path(r['dir']) / 'environment', label=f'agent-{r["task"]}')
        return images[r['task']]
    results = out / 'results.jsonl'
    log(f'{len(attempts)} attempts; images first')
    await asyncio.gather(*(image(r) for r in attempts), return_exceptions=True)
    snaps = {}
    for phase, arm in (('off', 'off'), ('on', 'on'), ('on-warm', 'on')):
        async def one(r):
            async with slots:
                try:
                    key, rootfs, imeta = await image(r)
                    res = await agent_run(store, rootfs, imeta, r['commands'], arm, package_cache,
                                          scratch / 'logs' / f'{phase}-{r["uid"][-60:]}', a.pip and arm != 'off')
                except Exception as e:  # noqa: BLE001
                    res = {'error': f'{type(e).__name__}: {e}'[:300]}
                with open(results, 'a') as f:
                    f.write(json.dumps({'phase': phase, 'uid': r['uid'], 'task': r['task'], **res}) + '\n')
                log(f'{phase} {r["task"]} rcs={res.get("rcs")} {res.get("proxy_log")}')
        await asyncio.gather(*(one(r) for r in attempts))
        snaps[phase] = json.loads(json.dumps(relay.tags))
        (out / f'relay-{phase}.json').write_text(json.dumps(snaps[phase], indent=1))
        (out / f'cache-stats-{phase}.json').write_text(json.dumps(cache.stats(), indent=1))
    await cache.close()
    log('agent replay done')


# ---------------------------------------------------------------------- author installs (internet sandboxes)
INSTALL_CMD = re.compile(r'\bapt(-get)?\s+(-\S+\s+)*(install|update)\b|\b(pip3?|python3? -m pip)\s+(-\S+\s+)*install\b')
AUTHOR_ARMS = ('off', 'apt', 'apt+pip')


def author_sample(a):
    """Workspace authors' recorded install commands (apt/pip), in order, with their asset -> <out>/authors.jsonl."""
    rows = []
    for rnd in a.rounds.split(','):
        if (ROUNDS / rnd / 'packed' / 'index.json').exists():   # task_synthesis roundpack (#245): loose files only
            raise SystemExit(f'{ROUNDS / rnd} is packed: this bench reads loose work/ directories only')
        for wd in sorted((ROUNDS / rnd / 'work').iterdir()):
            try:
                log = json.loads((wd / 'explore.json').read_text())['log']
                asset = Path(json.loads((wd / 'record.json').read_text())['spec']['asset']['dir']) / 'asset.tar.zst'
            except (OSError, ValueError, KeyError, TypeError):
                continue
            cmds = [c for c in log if INSTALL_CMD.search(c.get('cmd') or '')][:a.max_commands]
            if cmds and asset.exists():
                rows.append({'uid': f'{rnd}/{wd.name}', 'asset_tar': str(asset), 'commands': [c['cmd'] for c in cmds],
                             'recorded': [[c.get('rc'), c.get('secs')] for c in cmds]})
    random.Random(a.seed).shuffle(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'authors.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:a.n]))
    print(f'{min(len(rows), a.n)} of {len(rows)} authors with install commands -> {out / "authors.jsonl"}')


def registry_of(path):
    """() -> the cache's registry while its process lives, else None (harbor_ns.sandbox.package_cache for one path)."""
    def registry():
        try:
            reg = json.loads(Path(path).read_text())
            os.kill(int(reg['pid']), 0)
            return reg
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return registry


def version_map(text):
    return dict(line.split(' ', 1) if ' ' in line else line.split('==', 1)
                for line in (text or '').splitlines() if line.strip() and (' ' in line or '==' in line))


async def author_run(store, r, arm, registry, proxy_env, log_dir):
    """One author's install commands, in order, in an author sandbox built by the round's own code
    (explore.Explorer: internet through the cluster proxy); arms apt and apt+pip use the arm's package cache."""
    from collections import Counter
    from task_synthesis.pipeline import explore
    log_dir.mkdir(parents=True, exist_ok=True)
    ex = explore.Explorer(store, r['asset_tar'], log_dir.name, package_cache=None if arm == 'off' else arm,
                          fetch_log=log_dir / 'fetch.jsonl')
    ex.proxy_env, ex.cache_registry = proxy_env, registry
    out = {'rcs': [], 'secs': [], 'tails': []}
    async with ex:
        out['via_cache'] = ex.package_proxy is not None
        t0 = time.time()
        for cmd in r['commands']:
            t = time.time()
            res = await ex.box.exec(cmd, cwd='/tasksynth/asset', timeout=explore.CMD_TIMEOUT)
            out['rcs'].append(res.return_code)
            out['secs'].append(round(time.time() - t, 1))
            out['tails'].append(((res.stdout or '') + (res.stderr or ''))[-400:])
        out['seconds'] = round(time.time() - t0, 1)
        v = await ex.box.exec(VERSIONS_CMD, timeout=120)
        out['versions'] = version_map(v.stdout)
    fetch = log_dir / 'fetch.jsonl'
    reqs = [json.loads(x) for x in fetch.read_text().splitlines()] if fetch.exists() else []
    out['proxy_log'] = dict(Counter(f'{q.get("method")} {q.get("cache") or "-"}' for q in reqs if q.get('allowed')))
    out['refused'] = sum(1 for q in reqs if not q.get('allowed'))
    out['corrupt_marks'] = sum(len(CORRUPT.findall(t)) for t in out['tails'])
    return out


async def author_replay(a):
    """Recorded author install commands in author sandboxes, the arms side by side for each author (same proxy
    conditions): off (cluster proxy), apt (apt through the cache: the round default), apt+pip. Each cache arm has
    its own cache restored from the seed and its own counting relays, so every arm's cluster-proxy load is exact."""
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    import socket
    from task_synthesis.pipeline.check import make_store
    from .common import PROXY_ENV
    from .package_cache import CacheProcess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkgauthor')
    logf = open(out / 'author.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    # author sandboxes reach the network through slirp4netns without host loopback: their relays listen on the
    # address this node uses to reach the proxy
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('proxy', 8080))
    ip = s.getsockname()[0]
    s.close()
    arms = [x for x in AUTHOR_ARMS if x in a.arms.split(',')]
    relays = {arm: await CountingRelay().start(ip) for arm in arms}
    cache_relays = {arm: await CountingRelay().start() for arm in arms if arm != 'off'}
    caches, registries = {}, {}
    for arm in cache_relays:
        root = scratch / f'pkgcache-{arm.replace("+", "-")}'
        caches[arm] = await CacheProcess(root, root.parent / f'{root.name}.json', seed_dir=a.seed or None, log=log,
                                         args=['--upstream', f'http://127.0.0.1:{cache_relays[arm].port}',
                                               '--max-gib', '20']).start()
        registries[arm] = registry_of(caches[arm].registry)
    # warm from the seed first, as a round job is by the time its authors install (the registry file is written once:
    # ask the caches themselves)
    for _ in range(1800):
        stats = {k: await asyncio.to_thread(c.stats) for k, c in caches.items()}
        if not any((st or {}).get('restoring') for st in stats.values()):
            break
        await asyncio.sleep(1)
    log('caches ready: ' + ', '.join(f'{k} {(st or {}).get("restored")} restored' for k, st in stats.items()))

    def proxy_env(arm):
        u = f'http://{ip}:{relays[arm].port}'
        return {**PROXY_ENV, 'http_proxy': u, 'HTTP_PROXY': u, 'https_proxy': u, 'HTTPS_PROXY': u}

    def snapshot():
        (out / 'relays.json').write_text(json.dumps({'arms': {k: r.tags for k, r in relays.items()},
                                                     'caches': {k: r.tags for k, r in cache_relays.items()},
                                                     'hosts': {k: r.hosts for k, r in relays.items()},
                                                     'cache_hosts': {k: r.hosts for k, r in cache_relays.items()}},
                                                    indent=1))
        (out / 'cache-stats.json').write_text(json.dumps({k: c.stats() for k, c in caches.items()}, indent=1))
    authors = [json.loads(x) for x in Path(a.authors).read_text().splitlines()][:a.n]
    store = make_store()
    slots = asyncio.Semaphore(a.parallel)
    results = out / 'results.jsonl'
    done = 0

    async def one(i, r):
        nonlocal done
        async with slots:
            async def run(arm):
                try:
                    return await author_run(store, r, arm, registries.get(arm), proxy_env(arm),
                                            scratch / 'logs' / f'{arm.replace("+", "-")}-{i:04d}')
                except Exception as e:  # noqa: BLE001
                    return {'error': f'{type(e).__name__}: {e}'[:300]}
            res = await asyncio.gather(*(run(arm) for arm in arms))
            with open(results, 'a') as f:
                for arm, x in zip(arms, res):
                    f.write(json.dumps({'arm': arm, 'i': i, 'uid': r['uid'], 'commands': r['commands'],
                                        'recorded': r['recorded'], **x}) + '\n')
            done += 1
            log(f'{done}/{len(authors)} {r["uid"]} ' + ' | '.join(f'{arm} rcs={x.get("rcs")} {x.get("seconds")}s'
                                                                  + (f' {x["error"]}' if 'error' in x else '')
                                                                  for arm, x in zip(arms, res)))
            if done % 10 == 0:
                snapshot()
    log(f'{len(authors)} authors x {arms}, {a.parallel} at once')
    await asyncio.gather(*(one(i, r) for i, r in enumerate(authors)))
    snapshot()
    for c in caches.values():
        await c.close()
    log('author replay done')


def author_report(out):
    out = Path(out)
    rows = [json.loads(x) for x in (out / 'results.jsonl').read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r['uid'], {})[r['arm']] = r
    arms = [a for a in AUTHOR_ARMS if any(a in v for v in by.values())]
    full = {u: v for u, v in by.items() if all(a in v and 'error' not in v[a] for a in arms)}
    rel = json.loads((out / 'relays.json').read_text()) if (out / 'relays.json').exists() else {'arms': {}, 'caches': {}}
    lines = [f'# Author installs: {len(full)} authors with every arm complete ({len(by)} started)', '']
    lines += ['| arm | commands | same exit code as off | timeouts (600 s) | install seconds (sum) | per command p50 / p90 / max |'
              ' versions differing from off | via cache | proxy requests | proxy GiB | corrupt |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm in arms:
        rs = [v[arm] for v in full.values()]
        secs = [x for r in rs for x in r['secs']]
        same = sum(x == y for u, v in full.items() for x, y in zip(v[arm]['rcs'], v['off']['rcs'])) if 'off' in arms else 0
        diff = both = 0
        if 'off' in arms:
            for v in full.values():
                a_, o = v[arm]['versions'], v['off']['versions']
                common = set(a_) & set(o)
                both += len(common)
                diff += sum(a_[k] != o[k] for k in common)
        t = list(rel['arms'].get(arm, {}).values()) + list(rel['caches'].get(arm, {}).values())
        reqs = sum(x['http'] + x['connect'] for x in t)
        gib = sum(x['bytes_down'] for x in t) / 2**30
        lines.append(f'| {arm} | {len(secs)} | {same}/{len(secs)} | {sum(x == 124 for r in rs for x in r["rcs"])} | '
                     f'{sum(secs):.0f} | {pctl(secs, .5):.0f} / {pctl(secs, .9):.0f} / {max(secs or [0]):.0f} | '
                     f'{diff}/{both} | {sum(bool(r.get("via_cache")) for r in rs)}/{len(rs)} | {reqs} | {gib:.2f} | '
                     f'{sum(r.get("corrupt_marks", 0) for r in rs)} |')
    lines += ['', 'Errors: ' + json.dumps({a: sum('error' in v.get(a, {}) for v in by.values()) for a in arms})]
    if 'off' in arms:
        lines += ['', 'Commands whose exit code differs from off (arm, uid, rcs off -> arm, command):']
        for u, v in full.items():
            for arm in arms[1:]:
                for k, (x, y) in enumerate(zip(v['off']['rcs'], v[arm]['rcs'])):
                    if x != y:
                        lines.append(f'- {arm} {u}: {x} -> {y} ({v["off"]["secs"][k]:.0f}s -> {v[arm]["secs"][k]:.0f}s) '
                                     f'`{v[arm]["commands"][k][:120]}`')
    if (out / 'cache-stats.json').exists():
        for k, st in json.loads((out / 'cache-stats.json').read_text()).items():
            lines.append(f'\nCache {k}: ' + ', '.join(f'{x} {st.get(x)}' for x in ('requests', 'hit', 'miss', 'revalidated',
                                                                                 'stale', 'passthrough', 'error', 'restored')))
    text = '\n'.join(lines) + '\n'
    (out / 'REPORT.md').write_text(text)
    print(text)


# ---------------------------------------------------------------------- failure injection (production check() path)
CORRUPT = re.compile(r'Hash Sum mismatch|Hashes do not match|BADSIG|invalid signature|THESE PACKAGES DO NOT MATCH THE '
                     r'HASHES|does not match the expected hash|Bad header data|Truncated file|unexpected end of file')


def verify_store(root):
    """Every stored file against the SHA-256 in its metadata -> (files, mismatches)."""
    import hashlib
    n = bad = 0
    for meta in Path(root, 'objects').glob('*/*.json'):
        with contextlib.suppress(OSError, ValueError):
            m = json.loads(meta.read_text())
            h = hashlib.sha256()
            with open(meta.with_suffix(''), 'rb') as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            n += 1
            bad += h.hexdigest() != m['sha256']
    return n, bad


async def run_checks(tasks, parallel, scenario, results, log, reference, during=None):
    """task_synthesis check() for each task (the round's build+check path) -> rows appended to results."""
    from task_synthesis.pipeline import check as check_mod
    from harbor_ns.builder import proxy_failures
    store = check_mod.make_store()
    versions = {}
    real_release = check_mod.release_image

    def release(st, key):
        d = st.image_dir(key)
        if (d / 'rootfs').exists() and key not in versions:
            versions[key] = installed(d / 'rootfs')
            with contextlib.suppress(OSError):
                versions[key, 'log'] = (d / 'build.log').read_text(errors='replace')
        return real_release(st, key)
    check_mod.release_image = release
    slots = asyncio.Semaphore(parallel)

    async def one(t):
        async with slots:
            t0 = time.time()
            res = await check_mod.check(store, t['dir'])
            label = f'tasksynth-{Path(t["dir"]).name}'
            row = {'scenario': scenario, 'task': t['task'], 'ok': res['ok'], 'infra': res.get('infra'),
                   'task_errors': [e[:200] for e in res['errors']], 'seconds': round(time.time() - t0),
                   'fetch': res.get('fetch')}
            logs = sorted((store.shared_root / 'logs').glob(f'{label}.*.failed.log'), key=lambda p: p.stat().st_mtime)
            text = ''
            if not res['ok'] and logs and logs[-1].stat().st_mtime > t0:
                text = logs[-1].read_text(errors='replace')
                row['proxy_failures'] = proxy_failures(text.rsplit('rebuilding from scratch', 1)[-1], 8080)
            row['corrupt_marks'] = len(CORRUPT.findall(text))
            ref = reference.get(t['task'])
            row['ref_ok'] = ref is not None
            row['scored_as_task_defect'] = bool(res['errors']) and not res.get('infra') and \
                bool(ref and ref['ok'])
            with open(results, 'a') as f:
                f.write(json.dumps(row) + '\n')
            log(f'{scenario} {t["task"]} ok={res["ok"]} infra={bool(res.get("infra"))} {row["seconds"]}s '
                f'{(res.get("fetch") or {}).get("upstream", "")}')
    tasks_ = [asyncio.ensure_future(one(t)) for t in tasks]
    if during:
        await during()
    await asyncio.gather(*tasks_)
    check_mod.release_image = real_release
    for k, v in versions.items():
        if isinstance(k, tuple):
            n = len(CORRUPT.findall(v))
            if n:
                with open(results, 'a') as f:
                    f.write(json.dumps({'scenario': scenario, 'image': k[0], 'corrupt_marks_in_ok_build': n}) + '\n')


async def mini(a):
    """A stand-in round driver for the preemption scenario: its own package cache, then checks until killed."""
    from .package_cache import CacheProcess
    scratch = Path(a.scratch)
    os.environ['HARBOR_NS_LOCAL'] = str(scratch / 'harbor_ns')
    cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', seed_dir=a.seed,
                               log=lambda m: print(m, flush=True)).start()
    tasks = [json.loads(line) for line in Path(a.tasks).read_text().splitlines()]
    await run_checks(tasks, 8, a.scenario, Path(a.results), lambda m: print(m, flush=True), {})
    stats = cache.stats()
    Path(a.results).with_suffix('.stats.json').write_text(json.dumps(stats))
    await cache.close()


async def inject(a):
    from .package_cache import CacheProcess
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkginject')
    logf = open(out / 'inject.log', 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        logf.write(line + '\n')
        logf.flush()
    replay_dir = Path(a.replay)
    reference = {}
    for line in (replay_dir / 'results.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['phase'] == 'cold' and r['arm'] == 'off':
            reference[r['task']] = r
    sample = {json.loads(line)['task']: json.loads(line) for line in (replay_dir / 'sample.jsonl').read_text().splitlines()}
    rng = random.Random(7)
    pool = [sample[t] for t, r in sorted(reference.items()) if r['ok'] and sample[t]['stratum'] != 'none']
    rng.shuffle(pool)
    batches = [pool[i * a.batch:(i + 1) * a.batch] for i in range(6)]
    results = out / 'results.jsonl'
    relay = await CountingRelay().start()
    os.environ['PKGCACHE_UPSTREAM'] = f'http://127.0.0.1:{relay.port}'
    import harbor_ns.sandbox as hs
    # the builds' own proxy traffic (https) through the relay too, so "upstream unreachable" covers everything
    hs.DEFAULT_PROXY_HOST, hs.DEFAULT_PROXY_PORT, hs._PROXY_URL = '127.0.0.1', relay.port, False
    seed = replay_dir / 'seed'
    root = scratch / 'pkgcache'

    def summary(name, extra=None):
        rows = [json.loads(x) for x in results.read_text().splitlines()] if results.exists() else []
        rows = [r for r in rows if r.get('scenario') == name and 'task' in r]
        s = {'scenario': name, 'checks': len(rows), 'ok': sum(r['ok'] for r in rows),
             'infra': sum(bool(r['infra']) for r in rows), 'task_defect': sum(r['scored_as_task_defect'] for r in rows),
             'corrupt_marks': sum(r.get('corrupt_marks', 0) for r in rows), **(extra or {})}
        with open(out / 'summary.jsonl', 'a') as f:
            f.write(json.dumps(s) + '\n')
        log(f'SUMMARY {s}')

    # 1. kill -9 the cache twice while 16 builds run; the supervisor restarts it on the same port
    cache = await CacheProcess(root, scratch / 'pkgcache.json', seed_dir=seed, log=log).start()

    async def killer():
        for at in (45, 150):
            await asyncio.sleep(at if at == 45 else 105)
            pid = json.loads((scratch / 'pkgcache.json').read_text())['pid']
            os.kill(pid, 9)
            log(f'killed cache pid {pid}')
    await run_checks(batches[0], 16, 'kill-restart', results, log, reference, during=killer)
    summary('kill-restart', {'restarts': cache.restarts})

    # 2. upstream unreachable with a warm store: cached builds keep working, the rest fail as infrastructure
    relay.down = True
    await run_checks(batches[1], 16, 'upstream-down', results, log, reference)
    relay.down = False
    summary('upstream-down', {'cache': cache.stats()})

    # 3. stale index: InRelease validated 3 h ago (within stale-max) and 7 h ago (beyond), upstream down
    await cache.close()
    for hours, name in ((3, 'stale-3h-upstream-down'), (7, 'stale-7h-upstream-down')):
        for meta in root.glob('objects/*/*.json'):
            m = json.loads(meta.read_text())
            if m.get('mutable'):
                m['validated'] = time.time() - hours * 3600
                meta.write_text(json.dumps(m))
        cache = await CacheProcess(root, scratch / 'pkgcache.json', seed_dir=None, log=log,
                                   args=['--seed-dir', 'none']).start()
        relay.down = True
        await run_checks(batches[2][:a.batch // 2] if hours == 3 else batches[2][a.batch // 2:], 16, name, results,
                         log, reference)
        relay.down = False
        summary(name, {'cache': cache.stats()})
        await cache.close()
    # stale index with upstream up: revalidated, never served stale
    for meta in root.glob('objects/*/*.json'):
        m = json.loads(meta.read_text())
        if m.get('mutable'):
            m['validated'] = time.time() - 30 * 86400
            meta.write_text(json.dumps(m))
    cache = await CacheProcess(root, scratch / 'pkgcache.json', seed_dir=None, log=log,
                               args=['--seed-dir', 'none']).start()
    await run_checks(batches[3], 16, 'stale-30d-upstream-up', results, log, reference)
    summary('stale-30d-upstream-up', {'cache': cache.stats()})

    # 4. disk: scratch filled to below the cache's floor (10 GiB free): the cache stops storing, builds go on
    fill = scratch / 'fill.bin'
    free = shutil.disk_usage(scratch).free
    target_free = 6 * 2**30
    if free > target_free:
        subprocess.run(['fallocate', '-l', str(free - target_free), str(fill)], check=False)
    log(f'disk filled: {shutil.disk_usage(scratch).free / 2**30:.1f} GiB free')
    await run_checks(batches[4][:a.batch // 2], 8, 'disk-below-floor', results, log, reference)
    summary('disk-below-floor', {'cache': cache.stats(), 'free_gib': round(shutil.disk_usage(scratch).free / 2**30, 1)})
    fill.unlink(missing_ok=True)
    n, bad = verify_store(root)
    log(f'store verified: {n} files, {bad} mismatches')
    with open(out / 'summary.jsonl', 'a') as f:
        f.write(json.dumps({'scenario': 'store-verify', 'files': n, 'mismatches': bad}) + '\n')
    await cache.close()

    # 5. preemption: a driver killed with SIGKILL mid-run; its cache must exit, the restarted job restores the seed
    tasks_file = scratch / 'preempt-tasks.jsonl'
    tasks_file.write_text(''.join(json.dumps(t) + '\n' for t in batches[5]))
    env = {**os.environ}
    for attempt in ('preempt-1', 'preempt-2'):
        sc = scratch / attempt
        sc.mkdir(exist_ok=True)
        p = await asyncio.create_subprocess_exec(
            os.sys.executable, '-m', 'synth.package_cache_bench', 'mini', '--scratch', str(sc), '--seed', str(seed),
            '--tasks', str(tasks_file), '--results', str(results), '--scenario', attempt, env=env)
        if attempt == 'preempt-1':
            reg = None
            for _ in range(1200):   # the driver's cache is up, then the job runs 30 s (builds and the seed
                with contextlib.suppress(OSError, ValueError):   # restore in flight) before it is killed
                    reg = json.loads((sc / 'pkgcache.json').read_text())
                if reg or p.returncode is not None:
                    break
                await asyncio.sleep(0.5)
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(p.wait(), 30)
            if reg is None or p.returncode is not None:
                log(f'preempt-1: the driver ended before it could be killed (registry {reg})')
                continue
            p.kill()
            await p.wait()
            alive = True
            for _ in range(30):
                await asyncio.sleep(1)
                try:
                    os.kill(reg['pid'], 0)
                except ProcessLookupError:
                    alive = False
                    break
            log(f'driver killed; its cache (pid {reg["pid"]}) {"STILL ALIVE" if alive else "exited"}')
            with open(out / 'summary.jsonl', 'a') as f:
                f.write(json.dumps({'scenario': 'preempt-orphan', 'cache_alive_after_30s': alive}) + '\n')
            shutil.rmtree(sc, ignore_errors=True)   # the node's scratch is gone with the job
        else:
            await p.wait()
            stats = json.loads(results.with_suffix('.stats.json').read_text())
            summary(attempt, {'cache': stats})
    (out / 'relay.json').write_text(json.dumps(relay.tags, indent=1))
    log('inject done')


def keep_log_tail(root, dst):
    """The cache's non-hit log lines (misses, errors), not the millions of hits."""
    src = Path(root) / 'logs' / 'requests.jsonl'
    if src.exists():
        with open(src) as f, open(dst, 'w') as g:
            for line in f:
                if '"cache": "hit"' not in line:
                    g.write(line)


def submit_job(a, op, args, name, cpus, memory_gib, disk_gib=300):
    from task_synthesis.common import frozen_checkout
    from .common import PYTHON
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = frozen_checkout()
    sub = out / f'{op}.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export HARBOR_NS_LOCAL=$_CONDOR_SCRATCH_DIR/harbor_ns TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $_CONDOR_SCRATCH_DIR/tmp; exec {PYTHON} -u -m synth.package_cache_bench {op} --out {out} {args}'"
initialdir = {code}
output = {out}/job.$(Cluster).out
error = {out}/job.$(Cluster).err
log = {out}/job.condor.log
request_cpus = {cpus}
request_memory = {memory_gib * 1024}
request_disk = {disk_gib}G
+JobBatchName = "{name}"
+MaxRuntime = 43200
+BypassLXCfs = "true"
getenv = false
environment = "USER=user HOME=/home/user PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH={code} HARBOR_NS_PERSIST_IMAGES=0 HARBOR_NS_BUILD_CPUS=2 http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy=localhost,127.0.0.1,::1,localnet"
requirements = (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "gpu110") && (TARGET.UtsnameNodename != "cpu105") && (TARGET.UtsnameNodename != "cpu301") && (TARGET.UtsnameNodename != "gpu180") && (TARGET.UtsnameNodename != "gpu158") && (TARGET.UtsnameNodename != "cpu208") && (TARGET.UtsnameNodename != "gpu109") && (TARGET.UtsnameNodename != "gpu123") && (TARGET.UtsnameNodename != "gpu143")
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip(), 'code', code)


async def flaky(a):
    """Tasks whose cache-on and cache-off builds disagreed in the replay, rebuilt --repeats times per arm (cache warm
    from the replay's seed): a disagreement that does not follow the arm is the task's own flakiness."""
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    import harbor_ns.builder as hb
    from harbor_ns.cli import default_store
    from harbor_ns.sandbox import package_cache_env as real_env
    from .package_cache import CacheProcess
    out, replay_dir = Path(a.out), Path(a.replay)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or '/tmp/pkgflaky')
    rows = [json.loads(x) for x in (replay_dir / 'results.jsonl').read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault((r['phase'], r['task']), {})[r['arm']] = r['ok']
    tasks = sorted({t for (ph, t), v in by.items() if len(v) == 2 and v['on'] != v['off']
                    and not (a.on_failures_only and v['on'])})
    sample = {json.loads(x)['task']: json.loads(x) for x in (replay_dir / 'sample.jsonl').read_text().splitlines()}
    print(f'{len(tasks)} discordant tasks: {tasks}', flush=True)

    def bench_env(tag='', pip=True):
        return {} if ARM.get() == 'off' else real_env(tag, pip)
    hb.package_cache_env = bench_env
    cache = await CacheProcess(scratch / 'pkgcache', scratch / 'pkgcache.json', seed_dir=replay_dir / 'seed',
                               log=print).start()
    store = default_store()
    slots = asyncio.Semaphore(a.parallel)
    # builds of one task never overlap: builds share the host network, and tasks' setup scripts start services on
    # fixed ports (8080, 5432), so two copies of one task collide whatever the arm
    task_locks = {t: asyncio.Lock() for t in tasks}

    async def one(task, arm, i):
        async with task_locks[task], slots:
            ARM.set(arm)
            label = f'flaky-{arm}-{i}-{task}'
            ctx = scratch / 'ctx' / label
            await asyncio.to_thread(shutil.copytree, Path(sample[task]['dir']) / 'environment', ctx, symlinks=True)
            (ctx / f'.pkgcache-bench-{arm}-{i}').write_text(label)
            ok, err = True, None
            try:
                key, rootfs, meta = await asyncio.wait_for(store.build(ctx, label=label), 1800)
                d = store.image_dir(key)
                subprocess.run(['chmod', '-R', 'u+rwX', str(d)], capture_output=True)
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:  # noqa: BLE001
                ok, err = False, f'{type(e).__name__}: {str(e)[-200:]}'
            shutil.rmtree(ctx, ignore_errors=True)
            with open(out / 'flaky.jsonl', 'a') as f:
                f.write(json.dumps({'task': task, 'arm': arm, 'i': i, 'ok': ok, 'error': err}) + '\n')
            print(task, arm, i, ok, err or '', flush=True)
    jobs = [(t, arm, i) for t in tasks for arm in ('off', 'on') for i in range(a.repeats)]
    random.Random(3).shuffle(jobs)
    await asyncio.gather(*(one(*j) for j in jobs))
    await cache.close()


def keep_log(root, dst):
    src = Path(root) / 'logs' / 'requests.jsonl'
    if src.exists():
        shutil.copyfile(src, dst)


def submit_replay(a):
    from task_synthesis.common import frozen_checkout
    from .common import PYTHON
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    code = frozen_checkout()
    extra = ' --publish-production' if a.publish_production else ''
    extra += f' --limit {a.limit}' if a.limit else ''
    sub = out / 'replay.sub'
    sub.write_text(f'''universe = vanilla
executable = /bin/bash
arguments = "-c 'export HARBOR_NS_LOCAL=$_CONDOR_SCRATCH_DIR/harbor_ns TMPDIR=$_CONDOR_SCRATCH_DIR/tmp; mkdir -p $_CONDOR_SCRATCH_DIR/tmp; exec {PYTHON} -u -m synth.package_cache_bench replay --out {out} --parallel {a.parallel}{extra}'"
initialdir = {code}
output = {out}/job.$(Cluster).out
error = {out}/job.$(Cluster).err
log = {out}/job.condor.log
request_cpus = {a.cpus}
request_memory = {a.memory_gib * 1024}
request_disk = 300G
+JobBatchName = "tasksynth-pkgcache-replay"
+MaxRuntime = 43200
+BypassLXCfs = "true"
getenv = false
environment = "USER=user HOME=/home/user PYTHONNOUSERSITE=1 PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin PYTHONPATH={code} HARBOR_NS_PERSIST_IMAGES=0 HARBOR_NS_BUILD_CPUS=2 http_proxy=http://proxy.internal:8080 https_proxy=http://proxy.internal:8080 no_proxy=localhost,127.0.0.1,::1,localnet"
requirements = (TARGET.UtsnameNodename != "gpu132") && (TARGET.UtsnameNodename != "gpu110") && (TARGET.UtsnameNodename != "cpu105") && (TARGET.UtsnameNodename != "cpu301") && (TARGET.UtsnameNodename != "gpu180") && (TARGET.UtsnameNodename != "gpu158") && (TARGET.UtsnameNodename != "cpu208") && (TARGET.UtsnameNodename != "gpu109") && (TARGET.UtsnameNodename != "gpu123") && (TARGET.UtsnameNodename != "gpu143")
queue
''')
    r = subprocess.run(['condor_submit_bid', str(a.bid), str(sub)], capture_output=True, text=True, check=True)
    print(r.stdout.strip(), 'code', code)


# warm stress (seconds from start): 600 build clients with recorded gaps for 70 min; 2,400 sandbox clients for 40 min
# from minute 15; then 600 build clients back to back (no gaps) for 10 min
DEFAULT_PLAN = [{'name': 'build', 'role': 'build', 'at': 0, 'clients': 600, 'procs': 6, 'duration': 4200, 'ramp': 600},
                {'name': 'sandbox', 'role': 'sandbox', 'at': 900, 'clients': 2400, 'procs': 8, 'duration': 2400,
                 'ramp': 600},
                {'name': 'burst', 'role': 'build', 'at': 4260, 'clients': 600, 'procs': 6, 'duration': 600, 'ramp': 60,
                 'gap_scale': 0.0}]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='op', required=True)
    p = sub.add_parser('sample')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=600)
    for name in ('replay', 'submit-replay'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--parallel', type=int, default=48)
        p.add_argument('--limit', type=int, default=0)
        p.add_argument('--publish-production', action='store_true')
        if name == 'submit-replay':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=96)
    for name in ('stress', 'submit-stress'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--replay', required=True, help='replay output dir: request logs and seed')
        p.add_argument('--sandbox-seqs', default='/data/aai/tbench/tasksynth/pkgcache/agent/s1-installs.jsonl')
        p.add_argument('--cold', action='store_true')
        p.add_argument('--plan', default=json.dumps(DEFAULT_PLAN))
        if name == 'submit-stress':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=64)
    for name in ('inject', 'submit-inject'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--replay', required=True)
        p.add_argument('--batch', type=int, default=24)
        if name == 'submit-inject':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=96)
    for name in ('agent', 'submit-agent'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--attempts', default='/data/aai/tbench/tasksynth/pkgcache/agent/s1-installs.jsonl')
        p.add_argument('--seed', required=True, help='seed dir to warm the cache from (the replay\'s)')
        p.add_argument('--n', type=int, default=120)
        p.add_argument('--parallel', type=int, default=16)
        p.add_argument('--pip', action='store_true', help='also serve pip through the cache (harbor.package_cache_pip)')
        if name == 'submit-agent':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=96)
    p = sub.add_parser('author-sample')
    p.add_argument('--out', required=True)
    p.add_argument('--rounds', default='wy0')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--max-commands', type=int, default=6)
    p.add_argument('--seed', type=int, default=60)
    for name in ('author', 'submit-author'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--authors', required=True)
        p.add_argument('--seed', default='', help="seed dir to warm the caches from (default: the production seed)")
        p.add_argument('--arms', default=','.join(AUTHOR_ARMS))
        p.add_argument('--n', type=int, default=300)
        p.add_argument('--parallel', type=int, default=16)
        if name == 'submit-author':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=128)
    p = sub.add_parser('author-report')
    p.add_argument('--out', required=True)
    for name in ('flaky', 'submit-flaky'):
        p = sub.add_parser(name)
        p.add_argument('--out', required=True)
        p.add_argument('--replay', required=True)
        p.add_argument('--repeats', type=int, default=3)
        p.add_argument('--parallel', type=int, default=8)
        p.add_argument('--on-failures-only', action='store_true', help='only tasks that failed with the cache on')
        if name == 'submit-flaky':
            p.add_argument('--bid', type=int, default=35)
            p.add_argument('--cpus', type=int, default=32)
            p.add_argument('--memory-gib', type=int, default=96)
    p = sub.add_parser('mini')
    for k in ('--scratch', '--seed', '--tasks', '--results', '--scenario'):
        p.add_argument(k, required=True)
    p = sub.add_parser('report')
    p.add_argument('--out', required=True)
    p = sub.add_parser('stress-report')
    p.add_argument('--out', required=True)
    p = sub.add_parser('load')
    p.add_argument('--role', choices=['build', 'sandbox'], required=True)
    p.add_argument('--seqs', required=True)
    p.add_argument('--registry', required=True)
    p.add_argument('--clients', type=int, required=True)
    p.add_argument('--duration', type=float, required=True)
    p.add_argument('--ramp', type=float, default=600)
    p.add_argument('--gap-scale', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tmp', required=True)
    p.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.op == 'load':
        import resource
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        asyncio.run(load(a))
        return
    if a.op == 'agent':
        asyncio.run(agent_replay(a))
        return
    if a.op == 'submit-agent':
        submit_job(a, 'agent', f'--attempts {a.attempts} --seed {a.seed} --n {a.n} --parallel {a.parallel}'
                   + (' --pip' if a.pip else ''), 'tasksynth-pkgcache-agent', a.cpus, a.memory_gib, disk_gib=200)
        return
    if a.op == 'author-sample':
        author_sample(a)
        return
    if a.op == 'author':
        if not a.seed:
            from .package_cache import SEED_DIR
            a.seed = str(SEED_DIR)
        asyncio.run(author_replay(a))
        return
    if a.op == 'submit-author':
        submit_job(a, 'author', f'--authors {a.authors} --arms {a.arms} --n {a.n} --parallel {a.parallel}'
                   + (f' --seed {a.seed}' if a.seed else ''), 'tasksynth-pkgcache-author', a.cpus, a.memory_gib,
                   disk_gib=400)
        return
    if a.op == 'author-report':
        author_report(a.out)
        return
    if a.op == 'report':
        report(a.out)
        return
    if a.op == 'stress-report':
        stress_report(a.out)
        return
    if a.op == 'inject':
        asyncio.run(inject(a))
        return
    if a.op == 'submit-inject':
        submit_job(a, 'inject', f'--replay {a.replay} --batch {a.batch}', 'tasksynth-pkgcache-inject', a.cpus,
                   a.memory_gib, disk_gib=120)
        return
    if a.op == 'flaky':
        asyncio.run(flaky(a))
        return
    if a.op == 'submit-flaky':
        submit_job(a, 'flaky', f'--replay {a.replay} --repeats {a.repeats} --parallel {a.parallel}'
                   + (' --on-failures-only' if a.on_failures_only else ''),
                   'tasksynth-pkgcache-flaky', a.cpus, a.memory_gib, disk_gib=200)
        return
    if a.op == 'mini':
        asyncio.run(mini(a))
        return
    if a.op == 'stress':
        asyncio.run(stress(a))
        return
    if a.op == 'submit-stress':
        Path(a.out).mkdir(parents=True, exist_ok=True)
        plan = Path(a.out) / 'plan.json'
        plan.write_text(a.plan if a.plan.lstrip().startswith('[') else Path(a.plan).read_text())
        args = f"--replay {a.replay} --sandbox-seqs {a.sandbox_seqs} --plan {plan}" + (' --cold' if a.cold else '')
        submit_job(a, 'stress', args, 'tasksynth-pkgcache-stress', a.cpus, a.memory_gib)
        return
    if a.op == 'sample':
        sample(a.out, a.n)
    elif a.op == 'replay':
        asyncio.run(replay(a))
    elif a.op == 'submit-replay':
        submit_replay(a)


if __name__ == '__main__':
    main()
