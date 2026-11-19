#!/usr/bin/env python3
"""Restart dead serving jobs of the IRT-subset k=9 evals (launch.py), as orchestrate.refresh does for requests.json
runs: a server that is gone without serve/COMPLETE is resubmitted (avoiding the nodes it died on), up to 3 times; a
held one is removed first. Scores a completed run if the ops loop has not within 30 min. Exits when all are scored,
or when STOP exists. Events: events.log."""
import subprocess
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from launch import EVAL, names  # noqa: E402

sys.path.insert(0, EVAL)
from common import ROOT, read  # noqa: E402
from orchestrate import GPU_BID, PY, condor_jobs, sh  # noqa: E402

GRACE_S, SCORE_AFTER_S, MAX_RESTARTS = 600, 1800, 3


def ev(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(D / 'events.log', 'a') as f:
        f.write(line + '\n')


def try_sh(*args):
    try:
        sh(*args)
        return True
    except Exception as e:   # a failed condor_rm/submit must not end the babysitter
        ev(f'TROUBLE {args[:4]}: {e}')
        return False


def main():
    last_submit = {n: time.time() for n in names()}
    restarts = {n: 0 for n in names()}
    scored = set()
    ev(f'BABYSIT_START {len(last_submit)} runs')
    while not (D / 'STOP').exists():
        jobs = condor_jobs()
        for n in names():
            run = ROOT / n
            done, scores = run / 'serve' / 'COMPLETE', run / 'scores.json'
            if done.exists():
                if scores.exists() and scores.stat().st_mtime >= done.stat().st_mtime:
                    if n not in scored:
                        s = read(scores, {})
                        scored.add(n)
                        ev(f"SCORED {n} pass@1 {s.get('pass_at_1')} coverage {s.get('coverage')} "
                           f"per_attempt {[r.get('passed') for r in s.get('per_attempt', [])]}")
                elif time.time() - done.stat().st_mtime > SCORE_AFTER_S:
                    subprocess.run([PY, 'score.py', str(run)], capture_output=True, text=True, cwd=EVAL)
                    ev(f'SCORE_RUN {n} (ops loop had not scored it)')
                continue
            serve = [j for j in jobs if j['batch'] == f'q35-tblite-serve-{n}']
            for j in serve:
                if j['status'] == 5:
                    try_sh('condor_rm', j['id']) and ev(f"HELD_SERVER_REMOVED {n} {j['id']}")
            if any(j['status'] in (1, 2) for j in serve) or time.time() - last_submit[n] < GRACE_S:
                continue
            if restarts[n] >= MAX_RESTARTS:
                if restarts[n] == MAX_RESTARTS:
                    ev(f'TROUBLE {n}: serving job died {MAX_RESTARTS + 1} times without COMPLETE')
                    restarts[n] += 1
                continue
            bad = sorted({l.split()[2] for f in (run / 'serve').glob('serve.*.out') for l in [f.read_text().split('\n')[0]]
                          if l.startswith('[serve]') and len(l.split()) > 2})
            if not try_sh(PY, 'submit.py', 'serve', '--name', n, '--exclude-nodes', ','.join(bad), '--gpu-bid', GPU_BID):
                continue
            restarts[n] += 1
            last_submit[n] = time.time()
            ev(f'SERVER_RESTARTED {n} restart {restarts[n]} excluding {bad}')
        if len(scored) == len(last_submit):
            ev('ALL_SCORED')
            return
        time.sleep(600)
    ev('STOPPED')


if __name__ == '__main__':
    main()
