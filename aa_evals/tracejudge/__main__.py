"""python -m aa_evals.tracejudge {judge,watch,fold,digest,eval} ... (see README.md)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _judge(a):
    if a.judge in (None, 'none') or a.llm == 'none':
        return None
    from aa_evals.tracejudge.judge import LLMJudge
    return LLMJudge(a.judge, effort=a.effort, thinking=a.thinking, max_tokens=a.max_tokens, save_raw=a.save_raw)


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m aa_evals.tracejudge')
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp):
        sp.add_argument('--run-dir', required=True)
        sp.add_argument('--judge', default='serving:aa-judge-gemma4-31b',
                        help='model spec of the LLM judge (serving:<name> or url:<base>@<served>); none = rules only')
        sp.add_argument('--effort', default=None, help='reasoning_effort override (e.g. low, medium)')
        sp.add_argument('--thinking', choices=['on', 'off'], default=None)
        sp.add_argument('--max-tokens', type=int, default=8192)
        sp.add_argument('--save-raw', action='store_true',
                        help='retain each returned judge JSON body, reasoning and content in the local flags row')
        sp.add_argument('--llm', choices=['fails', 'suspicious', 'all', 'every', 'none'], default='fails',
                        help='which attempts the LLM reads; all skips decisive model outcomes, every forces '
                             'an LLM opinion on each attempt (default: failed or suspicious ones)')
        sp.add_argument('--out', default=None, help='flags file (default <run-dir>/flags.jsonl)')
        sp.add_argument('--concurrency', type=int, default=16)
        sp.add_argument('--layout', default=None, help='force a layout (pi_eval, hle, omniscience, gdppdf, critpt, '
                                                       'automationbench, gdpval, generic)')

    j = sub.add_parser('judge', help='judge a finished run (only attempts not in flags.jsonl yet)')
    common(j)
    j.add_argument('--rejudge', action='store_true')
    j.add_argument('--keys', default=None, help='comma-separated attempt keys')
    j.add_argument('--fold', action='store_true', help='fold n_flagged into summary.json afterwards')

    w = sub.add_parser('watch', help='follow a running run and judge each attempt as it finishes')
    common(w)
    w.add_argument('--interval', type=float, default=60)
    w.add_argument('--no-fold', action='store_true')
    w.add_argument('--idle-exit-h', type=float, default=0)

    f = sub.add_parser('fold', help='fold flags.jsonl into summary.json (or print with --dry-run)')
    f.add_argument('--run-dir', required=True)
    f.add_argument('--flags', default=None)
    f.add_argument('--dry-run', action='store_true')

    d = sub.add_parser('digest', help='print the digest and rule hits of one attempt (debugging)')
    d.add_argument('--run-dir', required=True)
    d.add_argument('--key', required=True)
    d.add_argument('--layout', default=None)

    e = sub.add_parser('eval', help='score judges on the labeled set')
    e.add_argument('--labels', required=True)
    e.add_argument('--flags', nargs='+', required=True, help='name=flags.jsonl pairs (one per judge arm)')
    e.add_argument('--out', default=None)

    a = p.parse_args(argv)
    if a.cmd in ('judge', 'watch'):
        from aa_evals.tracejudge import core
        judge = _judge(a)

        async def go():
            try:
                if a.cmd == 'judge':
                    keys = a.keys.split(',') if a.keys else None
                    rows = await core.judge_run(a.run_dir, judge, a.llm, a.out, a.concurrency, a.rejudge, keys, a.layout)
                    n = sum(1 for r in rows if r.get('flagged') and not r.get('known_infra') and r.get('counted', True))
                    print(f'judged {len(rows)}; new flags on counted attempts: {n}')
                    if a.fold:
                        from aa_evals.tracejudge.hook import fold_file
                        s = fold_file(a.run_dir, path=a.out)
                        print('summary.json:', None if s is None else {k: s.get(k) for k in ('n_flagged', 'n_flagged_by_category')})
                else:
                    await core.watch_run(a.run_dir, judge, a.llm, a.out, a.concurrency, a.interval, not a.no_fold,
                                         a.idle_exit_h, a.layout)
            finally:
                if judge:
                    await judge.close()
        asyncio.run(go())
    elif a.cmd == 'fold':
        from aa_evals.tracejudge.hook import fold_file, fold_summary
        if a.dry_run:
            s = fold_summary(a.run_dir, {}, path=a.flags)
            print(json.dumps(s, indent=1))
        else:
            s = fold_file(a.run_dir, path=a.flags)
            print(json.dumps(None if s is None else {k: s.get(k) for k in ('n_flagged', 'n_flagged_by_category', 'tracejudge')}, indent=1))
    elif a.cmd == 'digest':
        from aa_evals.tracejudge.digest import render
        from aa_evals.tracejudge.layouts import open_run
        from aa_evals.tracejudge.rules import decide, run_rules
        t = open_run(a.run_dir, a.layout).load(a.key)
        h = run_rules(t)
        print(render(t, h))
        print(f'\n== prefilter: {decide(h)}')
    elif a.cmd == 'eval':
        from aa_evals.tracejudge.labels import evaluate
        res = evaluate(a.labels, dict(x.split('=', 1) for x in a.flags))
        txt = json.dumps(res, indent=1)
        if a.out:
            Path(a.out).write_text(txt)
        print(txt)


if __name__ == '__main__':
    sys.exit(main())
