"""#171 end to end, inside one CPU condor job: a few-spec round on a fake model endpoint with 3 shards (real `round run`
processes, real builds and checks). Mid-flight, shard 1 is killed (SIGKILL: a job held at its memory limit) and shard 2
is stopped (SIGSTOP: a hung event loop) and resumed later. Shard 0 must take over both jobs' open claims and finish
them, shard 2 must drop the claims it lost without writing their records, and every job must exit by itself.

  python -m task_synthesis.tests.shards_e2e_job <out_dir> [--specs 12]

No model server: the endpoint answers generator requests with a small valid task (after a delay, so specs are in
flight when the shards are hit) and judge requests with a passing verdict. Writes <out_dir>/e2e.json.
"""
import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from task_synthesis.pipeline import specs as specs_mod
from task_synthesis.pipeline.selftest import LEDGER

PY = sys.executable
JUDGE = json.dumps({'scores': {'clarity': 5, 'consistency': 5, 'solvability': 5, 'difficulty': 4, 'realism': 4,
                               'originality': 4}, 'flags': {}, 'estimated_minutes': 20, 'issues': [], 'verdict': 'pass'})


class Fake(BaseHTTPRequestHandler):
    delay = (20, 40)
    calls = {'gen': 0, 'judge': 0}

    def log_message(self, *a):
        pass

    def send(self, obj, ctype='application/json'):
        body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith('/models'):
            return self.send({'data': [{'id': 'fake-gen', 'object': 'model'}]})
        if self.path.endswith('/metrics'):
            return self.send(b'vllm:num_requests_running{m="x"} 0\nvllm:num_requests_waiting{m="x"} 0\n', 'text/plain')
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        system = req['messages'][0]['content']
        if '<task_json>' in system:
            Fake.calls['gen'] += 1
            time.sleep(random.uniform(*self.delay))
            n = Fake.calls['gen']
            text = LEDGER.replace('"ledger-check"', f'"ledger-{n}"').replace('random.seed(0)', f'random.seed({n})')
        else:
            Fake.calls['judge'] += 1
            text = JUDGE
        self.send({'id': 'x', 'object': 'chat.completion', 'created': int(time.time()), 'model': 'fake-gen',
                   'choices': [{'index': 0, 'finish_reason': 'stop',
                                'message': {'role': 'assistant', 'content': text}}],
                   'usage': {'prompt_tokens': 100, 'completion_tokens': 200, 'total_tokens': 300}})


def owner_of(claim):
    """Current owner id of a .claim directory (highest take-<n>)."""
    gens = [int(n[5:]) for n in os.listdir(claim) if n.startswith('take-')]
    d = claim / f'take-{max(gens)}' if gens else claim
    try:
        return json.loads((d / 'owner').read_text())['owner']
    except (OSError, ValueError, KeyError):
        return None


def main(a):
    out = Path(a.out)
    rounds = out / 'rounds'
    rd = rounds / 'e2e'
    rd.mkdir(parents=True, exist_ok=True)
    specs = specs_mod.sample(a.specs, 171)
    (rd / 'specs.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in specs))
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    srv = ThreadingHTTPServer(('127.0.0.1', port), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{port}/v1'
    boot = (f'import sys; from pathlib import Path; import task_synthesis.pipeline.round as R; '
            f'R.ROUNDS = Path({str(rounds)!r}); sys.argv = ["round"] + sys.argv[1:]; R.main()')
    common = ['run', '--round', 'e2e', '--n', str(a.specs), '--gen', url, '--judge', url, '--shards', '3',
              '--package-cache', 'off', '--build-parallel', '4', '--repairs', '0', '--claim-stale-minutes', '1',
              '--authors', '2',   # at most 2 open specs per job, so every job holds claims
              '--hang-exit-minutes', '10']
    procs = {}
    for k in range(3):
        env = {**os.environ, 'HARBOR_NS_LOCAL': f'{os.environ.get("HARBOR_NS_LOCAL", "/tmp/hns")}-{k}'}
        procs[k] = subprocess.Popen([PY, '-u', '-c', boot, *common, '--shard', str(k)], env=env,
                                    stdout=open(out / f'shard{k}.out', 'w'), stderr=subprocess.STDOUT,
                                    start_new_session=True)
    t0 = time.time()
    events = []

    def log(msg):
        line = f'{time.time() - t0:6.0f}s {msg}'
        events.append(line)
        print(line, flush=True)

    def owners():
        beats = {}
        for p in (rd / 'shards').glob('*.json'):
            try:
                b = json.loads(p.read_text())
                beats[b['shard']] = b['owner']
            except (OSError, ValueError, KeyError):
                pass
        return beats

    def open_claims(owner):
        res = []
        for sp in specs:
            wd = rd / 'work' / sp['spec_id']
            if (wd / '.claim').is_dir() and not (wd / 'record.json').exists() and not (wd / 'error.json').exists():
                if owner_of(wd / '.claim') == owner:
                    res.append(sp['spec_id'])
        return res

    # wait until shards 1 and 2 each have in-flight specs
    while time.time() - t0 < 900:
        o = owners()
        if 1 in o and 2 in o and len(open_claims(o[1])) >= 2 and len(open_claims(o[2])) >= 2:
            break
        time.sleep(2)
    o = owners()
    killed, stopped = open_claims(o[1]), open_claims(o[2])
    os.killpg(procs[1].pid, signal.SIGKILL)
    log(f'SIGKILL shard 1 ({o[1]}) with open claims {killed}')
    os.killpg(procs[2].pid, signal.SIGSTOP)
    log(f'SIGSTOP shard 2 ({o[2]}) with open claims {stopped}')
    time.sleep(150)                       # > 1 min stale limit + a sweep
    os.killpg(procs[2].pid, signal.SIGCONT)
    log('SIGCONT shard 2')
    rc = {}
    while time.time() - t0 < 3600 and len(rc) < 2:
        for k in (0, 2):
            if k not in rc and procs[k].poll() is not None:
                rc[k] = procs[k].returncode
                log(f'shard {k} exited rc={rc[k]}')
        time.sleep(2)
    procs[1].wait()
    recs = {}
    for sp in specs:
        wd = rd / 'work' / sp['spec_id']
        for name in ('record.json', 'error.json'):
            if (wd / name).exists():
                recs[sp['spec_id']] = json.loads((wd / name).read_text())
    lost_logs = (rd / 'round.s2.log').read_text() if (rd / 'round.s2.log').exists() else ''
    res = {
        'specs': len(specs), 'finished': len(recs), 'status': {s: r['status'] for s, r in recs.items()},
        'killed_open': killed, 'stopped_open': stopped, 'owners': o,
        'taken_over': {s: r.get('taken_over_from') for s, r in recs.items() if r.get('taken_over_from')},
        'rc': rc, 'events': events, 'calls': Fake.calls,
        'shard2_dropped': [l for l in lost_logs.splitlines() if 'taken over by other jobs' in l or 'claim lost' in l],
    }
    checks = {
        'every spec finished': len(recs) == len(specs),
        'shard 1 open specs re-run, by shard 0': all(res['taken_over'].get(s) == o[1] for s in killed),
        'shard 2 open specs taken over': all(res['taken_over'].get(s) == o[2] for s in stopped),
        'shard 0 and 2 exited by themselves (rc 0)': rc == {0: 0, 2: 0},
        'shard 2 wrote no record for the specs it lost': not any(
            f' {sid} ' in l for sid in stopped for l in lost_logs.splitlines() if l.split(' ', 1)[-1].startswith('[')),
    }
    res['checks'] = checks
    (out / 'e2e.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print('E2E', 'PASS' if all(checks.values()) else 'FAIL', flush=True)
    srv.shutdown()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--specs', type=int, default=12)
    main(ap.parse_args())
    os._exit(0)
