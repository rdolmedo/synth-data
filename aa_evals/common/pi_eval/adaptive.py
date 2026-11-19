#!/usr/bin/env python3
"""Adaptive in-flight limiter for pi_eval runs sharing one vLLM engine (the r4 TB runs' limiter, #66, made generic).

  python -m aa_evals.common.pi_eval.adaptive --metrics http://host:port RUN [RUN ...] [--start 24] [--ceil 64]

Every 60 s it reads the engine's /metrics (KV usage, waiting requests, preemptions) and the runs' queues, then
writes each run's control.json (`max_in_flight`, read live by pi_eval's workers):
  - target +2 per calm check (KV < 0.85, nothing waiting, no new preemption, in flight >= target - 1), up to --ceil;
  - back off to (in flight - 1), not below --floor, on a new preemption or on waiting requests at KV >= 0.95;
  - free slots (target - in flight) are admitted only while KV < 0.95, nothing waits and nothing was preempted; they go
    to the runs in the order given (a run with nothing left to claim passes its share on). Admission stops at 0 for a
    run while the engine is under pressure; running attempts are never cut.
--metrics: the raw engine base URL (the registry's urls point at the admission gate when the spec has one; the gate
also decides per request, so this limiter only bounds how many attempts start). Prints on change only.
"""
from __future__ import annotations

import argparse
import os
import time
import urllib.request

OPEN = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def state(run: str) -> tuple[int, int]:
    """(attempts in flight, attempts left to claim) of one pi_eval run, from the one pending-work helper
    (tbsched.work.pending, #451): in flight = live claims (fresh heartbeat); left = unclaimed items + unclaimed
    same-seed retries + dead workers' claims (their reaper queues the retry), 0 once STOP_WORKERS is set. A dead
    worker's claim no longer holds engine share as "in flight" or keeps a stopped run from draining."""
    from pathlib import Path
    from tbsched.work import pending
    p = pending(Path(run))
    if not p.known:   # items.txt unreadable (a /data EIO): never "drained", never a limit of 0 -- the loop retries
        raise OSError(f'{run}: queue/items.txt unreadable')
    stopped = os.path.exists(os.path.join(run, 'STOP_WORKERS'))
    return p.live, 0 if stopped else p.claimable


def write(run: str, cap: int, why: str) -> None:
    """Read-merge-write (control.update, #451): a dispatcher's max_starts, a fence or a hand edit stays."""
    from aa_evals.common.pi_eval import control
    control.update(run, {'max_in_flight': int(cap), 'burst': 4, 'why': why})


def metric(text: str, name: str) -> float:
    return sum(float(ln.split()[-1]) for ln in text.splitlines() if ln.startswith(name + '{'))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--metrics', required=True, help='raw vLLM base URL (http://host:port)')
    ap.add_argument('--start', type=int, default=24)
    ap.add_argument('--ceil', type=int, default=64)
    ap.add_argument('--floor', type=int, default=8)
    ap.add_argument('--every', type=float, default=60)
    a = ap.parse_args()
    target, pre, last = a.start, None, None
    while True:
        try:
            text = OPEN.open(a.metrics.rstrip('/').removesuffix('/v1') + '/metrics', timeout=10).read().decode()
            kv = metric(text, 'vllm:kv_cache_usage_perc')
            w = metric(text, 'vllm:num_requests_waiting')
            p = metric(text, 'vllm:num_preemptions_total')
            dp = 0 if pre is None else int(p - pre)
            pre = p
            st = [state(r) for r in a.runs]
            act = sum(x for x, _ in st)
            old = target
            if dp > 0 or (w > 0 and kv >= 0.95):
                target = max(a.floor, act - 1)
            elif kv < 0.85 and w == 0 and dp == 0 and act >= target - 1:
                target = min(a.ceil, target + 2)
            if target != old:
                print(f'{time.strftime("%H:%M:%S", time.gmtime())} target {old} -> {target} (kv {kv:.2f}, '
                      f'waiting {w:g}, preempt +{dp}, in flight {act})', flush=True)
            free = max(0, target - act) if (kv < 0.95 and w == 0 and dp == 0) else 0
            why = f'adaptive T={target} kv={kv:.2f} waiting={w:g} preempt+={dp} in-flight {act}'
            admits = []
            for run, (inflight, left) in zip(a.runs, st):
                give = min(free, left)
                free -= give
                admits.append(give)
                write(run, inflight + give if give else 0, why)
            if admits != last:
                print(f'{time.strftime("%H:%M:%S", time.gmtime())} admit {admits} ({why}; left '
                      f'{[x for _, x in st]})', flush=True)
            last = admits
            if all(left == 0 and inflight == 0 for inflight, left in st):
                print('all runs drained', flush=True)
                return
        except Exception as e:   # noqa: BLE001 -- keep limiting through transient metric/FS errors
            print(f'{time.strftime("%H:%M:%S", time.gmtime())} error: {e!r}', flush=True)
        time.sleep(a.every)


if __name__ == '__main__':
    main()
