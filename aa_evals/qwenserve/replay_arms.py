"""Chained replay arms on one engine start (#110): gates are separate processes, so gate variants, pooling and client
splits are compared on the same engines without restarting vLLM.

  python -m aa_evals.qwenserve.replay_arms --endpoints serving/<name>/endpoints.json --launch .../launch.json \
      --arms arms.json --runs DIR --gpus 2

arms.json: [{"arm": "pool", "warmup": 900, "measure": 1200,
             "gates": {"pool": {"engines": [0, 1], "policy": {"client_share": true}}},
             "clients": [{"name": "agents", "base": "@pool", "corpus": ..., "count": 96, ...}, ...]}, ...]
Per arm: wait until every engine is idle (nothing running or waiting), reset vLLM's prefix cache including the CPU
tier (POST /reset_prefix_cache?reset_external=true; the engines need VLLM_SERVER_DEV_MODE=1), so no arm inherits
the previous arm's cached prefixes; start the arm's gates fresh (no learned state carries over); run mixed_replay
with every client's "@<gate>" (or "@engine<i>") base resolved, server totals summed over all engines; stop the gates.
Appends one row per arm to <runs>/arms.jsonl (wall time, B200-hours = gpus x wall, cache reset result).
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'inference'))
from replay_bench import scrape  # noqa: E402

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def post(url, timeout=120):
    req = urllib.request.Request(url, data=b'', method='POST')
    return json.loads(OPENER.open(req, timeout=timeout).read() or b'{}')


def get(url, timeout=10):
    return json.loads(OPENER.open(url, timeout=timeout).read())


def wait_idle(engines, timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snaps = [scrape(e) for e in engines]
        if all(s and not s.get('run') and not s.get('wait') for s in snaps):
            return round(time.time() - t0)
        time.sleep(5)
    return None


def reset_caches(engines, tries=30):
    out = []
    for e in engines:
        ok = False
        for _ in range(tries):
            try:
                ok = bool(post(e + '/reset_prefix_cache?reset_external=true').get('success'))
            except OSError as exc:
                ok = f'error: {exc!r}'[:200]
                break
            if ok:
                break
            time.sleep(5)
        out.append(ok)
    return out


def start_gates(spec, engines, logs, d, max_inflight):
    procs, urls = {}, {}
    for name, g in spec.items():
        port = free_port()
        ups = [engines[i] for i in g['engines']]
        argv = [sys.executable, '-u', '-m', 'aa_evals.qwenserve.gate', '--upstream', ','.join(ups), '--port', str(port),
                '--max-inflight', str(g.get('max_inflight', max_inflight)), '--policy', json.dumps(g.get('policy', {})),
                '--log', str(d / f'gate-{name}.jsonl'), '--status', str(d / f'gate-{name}.status.json'),
                '--vllm-log', ','.join(logs[i] for i in g['engines'])]
        procs[name] = subprocess.Popen(argv, cwd=str(ROOT), stdout=(d / f'gate-{name}.out').open('ab'),
                                       stderr=subprocess.STDOUT, start_new_session=True)
        urls[name] = f'http://127.0.0.1:{port}'
    for name, url in urls.items():
        for _ in range(60):
            try:
                st = get(url + '/gate/status')
                if all(e.get('capacity') for e in st.get('engines', [])):
                    break
            except OSError:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f'gate {name} not ready')
    return procs, urls


def stop(procs):
    for p in procs.values():
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
    for p in procs.values():
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)


def window_stats(engines, start, seconds, out):
    """Each engine's counters over the measured window (scraped directly, not through a gate): gen tok/s, prompt
    tok/s, prefix hit (GPU + CPU tier), preemptions; for twin arms, where server totals sum all engines."""
    time.sleep(max(0.0, start - time.time()))
    s0 = [scrape(e) for e in engines]
    time.sleep(seconds)
    s1 = [scrape(e) for e in engines]
    for a, b in zip(s0, s1):
        if not (a and b):
            out.append(None)
            continue
        d = {k: b[k] - a[k] for k in ('gen', 'prompt', 'src_compute', 'src_gpu', 'src_cpu', 'preempt')}
        seen = d['src_compute'] + d['src_gpu'] + d['src_cpu']
        out.append({'gen_tok_s': round(d['gen'] / seconds), 'prompt_tok_s': round(d['prompt'] / seconds),
                    'hit': round((d['src_gpu'] + d['src_cpu']) / seen, 3) if seen else None,
                    'preemptions': d['preempt']})


def resolve(clients, urls, engines):
    out = []
    for c in clients:
        c = dict(c)
        base = c.get('base', '')
        if base.startswith('@engine'):
            c['base'] = engines[int(base[len('@engine'):])] + '/v1'
        elif base.startswith('@'):
            c['base'] = urls[base[1:]] + '/v1'
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoints', required=True)
    ap.add_argument('--launch', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--gpus', type=int, required=True)
    ap.add_argument('--only', default='', help='comma-separated arm names')
    a = ap.parse_args()
    ep = json.loads(Path(a.endpoints).read_text())
    launch = json.loads(Path(a.launch).read_text())
    engines = [u.removesuffix('/v1').rstrip('/') for u in ep['engine_urls']]
    logs = [r['log'] for r in launch['replicas']]
    runs = Path(a.runs)
    only = set(filter(None, a.only.split(',')))
    for arm in json.loads(Path(a.arms).read_text()):
        if only and arm['arm'] not in only:
            continue
        d = runs / 'arms' / arm['arm']
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        idle = wait_idle(engines)
        reset = reset_caches(engines)
        procs, urls = start_gates(arm.get('gates', {}), engines, logs, d, arm.get('max_inflight', 128))
        clients = d / 'clients.json'
        clients.write_text(json.dumps(resolve(arm['clients'], urls, engines), indent=1))
        print(f'{time.strftime("%FT%TZ", time.gmtime())} arm {arm["arm"]}: idle wait {idle}s, reset {reset}, '
              f'gates {urls}', flush=True)
        t1 = time.time()
        per_engine = []
        watcher = threading.Thread(target=window_stats, args=(engines, t1 + arm['warmup'], arm['measure'], per_engine),
                                   daemon=True)
        watcher.start()
        rc = subprocess.call([sys.executable, '-u', '-m', 'aa_evals.qwenserve.mixed_replay', '--base',
                              engines[0] + '/v1', '--served-name', ep['served_name'], '--clients', str(clients),
                              '--metrics', ','.join(engines), '--warmup', str(arm['warmup']),
                              '--measure', str(arm['measure']), '--arm', arm['arm'], '--out',
                              str(runs / 'mixed.jsonl')], cwd=str(ROOT))
        for name, url in urls.items():
            try:
                (d / f'gate-{name}.final.json').write_text(json.dumps(get(url + '/gate/status')))
            except OSError:
                pass
        stop(procs)
        watcher.join(timeout=5)
        t2 = time.time()
        row = {'arm': arm['arm'], 'rc': rc, 'idle_wait_s': idle, 'cache_reset': reset, 'replay_s': round(t2 - t1),
               'wall_s': round(t2 - t0), 'b200_h': round(a.gpus * (t2 - t0) / 3600, 3), 't0': t0, 't2': t2,
               'engines': per_engine, 'gates': {n: g['engines'] for n, g in arm.get('gates', {}).items()}}
        with open(runs / 'arms.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
