"""#171 memory caps, inside a CPU condor job (never on a login node or gpu106): synthetic memory bombs fail their own check
and the job survives; ordinary tasks and JVM / Go / node runtimes still pass; ulimit -v (the rejected mechanism) is shown
to break them; m1-hero tasks re-checked under the caps.

  python -m task_synthesis.tests.memcap_job <out_dir> [--tasks DIR ...] [--sample N --round m1-hero]

Writes <out_dir>/memcap.json and prints one line per case. The job's own peak memory (VmHWM of this process plus the
children it waited for) is part of the result: it must stay far below the bombs' targets.
"""
import argparse
import asyncio
import json
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')

from harbor_ns.sandbox import Sandbox, apply_stack_limit  # noqa: E402
from task_synthesis.pipeline.check import check, make_store  # noqa: E402
from task_synthesis.pipeline.package import materialize, parse  # noqa: E402
from task_synthesis.pipeline.selftest import LEDGER, SERVICE, WORDS  # noqa: E402

BOMB_PY = 'import time\nb = []\nwhile len(b) < 60:\n    b.append(bytearray(1 << 30))\n    time.sleep(0.2)\nprint("survived")\n'
RECURSE_C = r'''#include <stdio.h>
#include <string.h>
static int depth(int n) { char buf[4096]; memset(buf, n, sizeof buf); return depth(n + 1) + buf[n % 4096]; }
int main(void) { printf("%d\n", depth(0)); return 0; }
'''


def package(slug, setup, defect='', services=None, files=None, apt=()):
    files = {'/app/README.md': f'# {slug}\n', **(files or {})}
    body = ''.join(f'<file path="{p}">\n{t}\n</file>\n' for p, t in files.items())
    meta = {'slug': slug, 'title': slug, 'kind': 'substrate', 'expert_minutes': 15, 'apt': list(apt), 'pip': [],
            'inputs': ['/app/README.md'], 'outputs': [], 'defect_check': defect, 'services': bool(services),
            'summary': 's'}
    text = (f'<task_json>\n{json.dumps(meta)}\n</task_json>\n<instruction>\n{WORDS} Read /app/README.md.\n</instruction>\n'
            f'{body}<setup>\n#!/bin/bash\nset -euo pipefail\n{setup}\n</setup>\n'
            + (f'<services>\n#!/bin/bash\n{services}\n</services>\n' if services else '')
            + '<solution_sketch>\nx\n</solution_sketch>')
    return text


CASES = {
    # (package text, expected ok, what must be in the errors)
    'ledger': (LEDGER, True, None),
    'service': (SERVICE, True, None),
    'bomb-in-setup': (package('bomb-setup', 'python3 /tasksynth/bomb.py', files={'/tasksynth/bomb.py': BOMB_PY}),
                      False, 'out of memory'),
    'bomb-in-defect-check': (package('bomb-defect', 'mkdir -p /opt/b && cp /tasksynth/bomb.py /opt/b/',
                                     defect='python3 /opt/b/bomb.py', files={'/tasksynth/bomb.py': BOMB_PY}),
                             False, 'ran out of memory'),
    'bomb-in-service': (package('bomb-service', 'mkdir -p /opt/b && cp /tasksynth/bomb.py /opt/b/',
                                services='nohup python3 /opt/b/bomb.py >/dev/null 2>&1 &\nsleep 8\nexit 0',   # grows while services start
                                files={'/tasksynth/bomb.py': BOMB_PY}),
                        False, 'ran out of memory'),
    # the m1-hero culprit's pattern: a crash reproducer that recurses without end and expects SIGSEGV (rc 139)
    'recursion-expects-segv': (package('recurse', 'cd /tasksynth && gcc -O0 -o /usr/local/bin/recurse recurse.c\n'
                                       'set +e; /usr/local/bin/recurse >/dev/null 2>&1; rc=$?; set -e\n'
                                       'echo "rc=$rc"; [ "$rc" -eq 139 ]',
                                       defect='/usr/local/bin/recurse >/dev/null 2>&1; [ $? -eq 139 ]',
                                       files={'/tasksynth/recurse.c': RECURSE_C}),
                               True, None),
}

RUNTIME_APT = ['nodejs', 'default-jdk-headless', 'golang-go']
RUNTIME_CMDS = {
    'node': "node -e 'const m = new WebAssembly.Memory({initial: 256}); const a = new Array(1e6).fill(1); "
            "console.log(\"node ok\", m.buffer.byteLength, a.length)'",
    'java': 'cd /tmp && printf "public class H { public static void main(String[] a) { long[] x = new long[50000000]; '
            'System.out.println(\\"java ok \\" + x.length + \\" max heap \\" + Runtime.getRuntime().maxMemory()); } }" '
            '> H.java && javac H.java && java -cp /tmp H',
    'go': 'mkdir -p /tmp/g && cd /tmp/g && printf "package main\\nimport \\"fmt\\"\\nfunc main() { s := make([]byte, 1<<28); '
          's[0] = 1; fmt.Println(\\"go ok\\", len(s)) }\\n" > m.go && GOCACHE=/tmp/gocache go run m.go',
    'python-numpy': 'python3 -c "import numpy; a = numpy.ones((4000, 4000)); print(\'numpy ok\', a.sum())"',
    'gcc-asan': "cd /tmp && printf 'int main(){char*p=__builtin_malloc(10);p[0]=1;return p[0]-1;}' > a.c && "
                'gcc -fsanitize=address -o a a.c && ./a && echo asan ok',
}
LIMITS = {'none': '', 'ulimit -v 8 GiB': 'ulimit -v 8388608; ', 'ulimit -v 16 GiB': 'ulimit -v 16777216; '}


def peak_gb():
    self_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids_kib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return round(self_kib / 2**20, 2), round(kids_kib / 2**20, 2)


async def run_case(store, out, name, text, mem=None):
    d = materialize(parse(text), {'spec_id': name, 'minutes': 15}, {'model': 'memcap'}, out / 'tasks' / name)
    t0 = time.time()
    try:
        chk = await check(store, d, memory_mb=mem)
    except Exception as e:  # noqa: BLE001  (sandbox infrastructure)
        chk = {'ok': False, 'errors': [], 'infra': f'{type(e).__name__}: {e}'}
    return {'ok': chk['ok'], 'errors': [e[:600] for e in chk['errors']], 'infra': chk.get('infra'),
            'memory_kills': chk.get('memory_kills'), 'seconds': round(time.time() - t0), 'peak_gb': peak_gb()}


async def runtimes(store, out):
    """Build one image with node, a JDK, Go, numpy and gcc; run each runtime under the PSS guard (8 GiB) and under
    ulimit -v (the mechanism #171 suggested as a fallback)."""
    text = package('runtimes', 'true', apt=RUNTIME_APT)
    pkg = parse(text)
    d = materialize(pkg, {'spec_id': 'runtimes', 'minutes': 15}, {'model': 'memcap'}, out / 'tasks' / 'runtimes')
    key, rootfs, imeta = await store.build(d / 'environment', label='memcap-runtimes', memory_mb=24 * 1024)
    res = {}
    box = Sandbox(cfg=store.rt, lower=[rootfs], hostname='rt', base_env={**imeta.env, 'HOME': '/root'},
                  internet=False, memory_limit_mb=8192, memory_guard_required=False)
    await box.start()
    try:
        for lname, prefix in LIMITS.items():
            for cname, cmd in RUNTIME_CMDS.items():
                r = await box.exec(prefix + cmd, timeout=300)
                res[f'{cname} | {lname}'] = {'rc': r.return_code, 'out': (r.stdout + r.stderr)[-300:].strip()}
                print('runtime', cname, '|', lname, r.return_code, (r.stdout + r.stderr)[-160:].strip().replace('\n', ' '),
                      flush=True)
        res['guard'] = {'proc_mode': box.proc_mode, 'kills': box.memory_kills}
    finally:
        await box.stop()
    return res


async def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {'host': os.uname().nodename, 'stack_kib': None, 'cases': {}, 'tasks': {}}
    store = make_store()
    if a.stack_kb:
        results['stack_kib'] = apply_stack_limit(a.stack_kb)
    print('stack limit KiB', results['stack_kib'], 'rlimit', resource.getrlimit(resource.RLIMIT_STACK), flush=True)
    if not a.no_cases:
        for name, (text, want_ok, want_err) in CASES.items():
            r = await run_case(store, out, name, text)
            good = r['ok'] == want_ok and (want_err is None or any(want_err in e for e in r['errors']))
            results['cases'][name] = {**r, 'expected_ok': want_ok, 'as_expected': good}
            print('case', name, 'PASS' if good else 'FAIL', json.dumps(r)[:900], flush=True)
        results['runtimes'] = await runtimes(store, out)
    tasks = [Path(t) for t in a.tasks]
    if a.sample:
        rd = Path('/data/aai/tbench/tasksynth/rounds') / a.round
        valid = []
        for line in (rd / 'specs.jsonl').read_text().splitlines():
            sid = json.loads(line)['spec_id']
            rec = rd / 'work' / sid / 'record.json'
            if rec.exists() and (rd / 'tasks' / sid / 'task.toml').exists():
                valid.append(sid)
        rng = random.Random(a.seed)
        rng.shuffle(valid)
        picked, runtime_picked = [], []
        for sid in valid:
            if len(picked) >= a.sample and len(runtime_picked) >= a.runtime_sample:
                break
            try:
                r = json.loads((rd / 'work' / sid / 'record.json').read_text())
            except (OSError, ValueError):
                continue
            if r.get('status') != 'valid':
                continue
            df = (rd / 'tasks' / sid / 'environment' / 'Dockerfile').read_text(errors='replace')
            heavy = any(w in df for w in ('openjdk', 'default-jdk', 'default-jre', 'nodejs', 'npm', 'golang', 'maven',
                                         'gradle'))
            if heavy and len(runtime_picked) < a.runtime_sample:
                runtime_picked.append(sid)
            elif not heavy and len(picked) < a.sample:
                picked.append(sid)
        tasks += [rd / 'tasks' / sid for sid in picked + runtime_picked]
        results['sample'] = {'plain': picked, 'runtimes': runtime_picked}
    sem = asyncio.Semaphore(a.parallel)

    async def one(src):
        dst = out / 'copies' / src.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)
        async with sem:
            t0 = time.time()
            try:
                chk = await check(store, dst)
            except Exception as e:  # noqa: BLE001
                chk = {'ok': False, 'errors': [], 'infra': f'{type(e).__name__}: {e}'}
        r = {'ok': chk['ok'], 'errors': [e[:400] for e in chk['errors']], 'infra': chk.get('infra'),
             'memory_kills': chk.get('memory_kills'), 'seconds': round(time.time() - t0), 'peak_gb': peak_gb()}
        results['tasks'][src.name] = r
        print('task', src.name, 'ok' if r['ok'] else 'FAIL', json.dumps(r)[:700], flush=True)
    await asyncio.gather(*(one(t) for t in tasks))
    results['peak_gb'] = peak_gb()
    (out / 'memcap.json').write_text(json.dumps(results, indent=1))
    n_ok = sum(r['ok'] for r in results['tasks'].values())
    print('summary: cases', sum(c['as_expected'] for c in results['cases'].values()), '/', len(results['cases']),
          'as expected; tasks ok', n_ok, '/', len(results['tasks']), '; peak GB (self, children)', results['peak_gb'],
          flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--tasks', nargs='*', default=[])
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--runtime-sample', type=int, default=0)
    ap.add_argument('--round', default='m1-hero')
    ap.add_argument('--seed', type=int, default=171)
    ap.add_argument('--parallel', type=int, default=8)
    ap.add_argument('--stack-kb', type=int, default=8192, help='0: leave the job\'s (unlimited) stack')
    ap.add_argument('--no-cases', action='store_true')
    asyncio.run(main(ap.parse_args()))
    sys.stdout.flush()
    os._exit(0)
