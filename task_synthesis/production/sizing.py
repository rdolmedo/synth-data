"""Generation sizing for M1 production from measured rates (#124 -> #133): task count, B200-h and wall time for a
GPU layout, and the Super engine count that matches the DeepSeek servers' task supply.

Every number is an input with a default measured on the m1-pilot (2026-11-03/04, #124); override any of them.
The outputs follow from the assumptions below; none of them is a threshold.

  python -m task_synthesis.production.sizing                       # M1 defaults: 12B SFT tokens, E=2, M=1
  python -m task_synthesis.production.sizing --epochs 1 --ds-servers 4 --engines 8
  python -m task_synthesis.production.sizing --ds-tok-s 60000      # what a higher session cap would buy
  python -m task_synthesis.production.sizing --assumptions         # print the assumptions table only

Model (steady state, then fixed overheads):
  unique tokens = SFT tokens / E;  traces = unique / saved tokens per trace;  tasks = traces / M
  DeepSeek server-h = tasks x DeepSeek tokens per accepted task (real-asset/generated mix + binding, x (1 + waste)) / (gen tok/s per server x 3600)
  Super engine-h    = unique tokens / (saved tok/s per engine at capacity x 3600)
  supply rate (tasks/h) = servers x gen tok/s x 3600 / DeepSeek tokens per task; demand per engine (tasks/h) =
  saved tok/s per engine x 3600 / (M x saved tokens per trace). Balanced engines = supply / demand per engine.
  Wall = the slower side (task supply vs Super), plus startup; the faster side idles for the difference unless it is
  resized. Overheads: server starts (one per 24 h of wall per server), the round's ramp, the stop rule's
  over-generation (supply_margin), smoke and campaign startup.
"""
import argparse
import math

# name: (default, unit, source)
ASSUMPTIONS = {
    'sft_tokens': (12e9, 'tokens', '#49 target'),
    'epochs': (2, '', '#49 (E=2 accepted; #132 decides)'),
    'traces_per_task': (1, '', '#116 (M=1; the pilot runs M=2)'),
    'saved_per_trace': (46100, 'saved tokens', 'm1-pilot: 1.109B saved / 24,033 trajectories (32% real-asset tasks)'),
    'ds_tokens_bound': (199800, 'DeepSeek completion tokens per accepted real-asset task (author + judge)',
                        'm1-pilot records (3,634 accepted)'),
    'ds_tokens_generated': (93700, 'DeepSeek completion tokens per accepted generated task (author + judge)',
                            'm1-pilot records (7,802 accepted)'),
    'asset_task_share': (0.32, 'of accepted tasks', 'm1-pilot: 3,634 / 11,436 (plan: 0.36)'),
    'ds_tokens_bind': (6600, 'DeepSeek completion tokens per accepted task for binding',
                       'm1-pilot bind log: ~12.4k per bind-pending spec, 6,131 specs / 11,436 accepted'),
    'ds_waste': (0.10, 'extra DeepSeek tokens in no record (tail-cut sessions, retries, restarts)',
                 'm1-pilot: 1.700B generated vs 1.457B in records + 0.076B binding (9.8%)'),
    'ds_tok_s': (45800, 'gen tok/s per 4xB200 server',
                 'm1-pilot 02:50-04:20: 91.6k for 2 servers at ~1,850-2,060 running (target_sessions 1000), KV 0.16, hit 0.90'),
    'ds_gpus': (4, 'B200 per server', 'm1.toml'),
    'super_saved_tok_s': (12000, 'saved tok/s per 1xB200 engine at capacity',
                          'm1-pilot: 4 KV-full engines 48-50k saved/s; 24.6 busy engine-h for 1.109B'),
    'ds_startup_h': (0.27, 'h per server start', 'pilot: submit -> healthy 16 min (staging 3 + vLLM 13)'),
    'server_hours': (24, 'h', 'm1.toml: no server longer than ~24 h -> one restart per 24 h'),
    'ramp_h': (2.0, 'h', 'pilot: the round was bind-window-limited for ~2 h after start'),
    'ramp_efficiency': (0.76, 'of steady DeepSeek output during the ramp', 'm1-pilot: 606M in 2.4 h vs 91.6k tok/s steady'),
    'supply_margin': (0.03, 'extra tasks generated past the budget', 'm1.toml [stop] supply_margin'),
    'super_startup_h': (0.5, 'h of every engine', 'pilot: engine ready ~5 min + smoke 26.5 min'),
}


def ds_tokens_per_task(p):
    return ((p['asset_task_share'] * p['ds_tokens_bound'] + (1 - p['asset_task_share']) * p['ds_tokens_generated']
             + p['ds_tokens_bind']) * (1 + p['ds_waste']))


def size(p, ds_servers, engines):
    tok = ds_tokens_per_task(p)
    unique = p['sft_tokens'] / p['epochs']
    traces = unique / p['saved_per_trace']
    tasks = traces / p['traces_per_task']
    tasks_gen = tasks * (1 + p['supply_margin'])
    ds_tok = tasks_gen * tok
    supply_tph = ds_servers * p['ds_tok_s'] * 3600 / tok           # tasks per hour
    demand_tph = p['super_saved_tok_s'] * 3600 / (p['traces_per_task'] * p['saved_per_trace'])   # per engine
    balanced = supply_tph / demand_tph
    ramp_loss_h = p['ramp_h'] * (1 - p['ramp_efficiency'])
    ds_wall = ds_tok / (ds_servers * p['ds_tok_s'] * 3600) + ramp_loss_h + p['ds_startup_h']
    super_wall = unique / (engines * p['super_saved_tok_s'] * 3600) + p['super_startup_h']
    wall = max(ds_wall, super_wall)
    restarts = ds_servers * max(0, math.ceil(ds_wall / p['server_hours']) - 1)
    ds_busy = ds_tok / (p['ds_tok_s'] * 3600) * p['ds_gpus']
    ds_h = ds_servers * p['ds_gpus'] * ds_wall + restarts * p['ds_gpus'] * p['ds_startup_h']
    super_busy = unique / (p['super_saved_tok_s'] * 3600)
    super_h = engines * wall          # engines stay up until the campaign ends
    return {
        'unique_tokens': unique, 'tasks': tasks, 'tasks_generated': tasks_gen,
        'supply_tasks_per_h': supply_tph, 'engine_tasks_per_h': demand_tph, 'balanced_engines': balanced,
        'ds_wall_h': ds_wall, 'super_wall_h': super_wall, 'wall_h': wall, 'bound': 'DeepSeek' if ds_wall >= super_wall else 'Super',
        'ds_restarts': restarts, 'ds_b200h': ds_h, 'ds_busy_b200h': ds_busy, 'super_b200h': super_h,
        'super_busy_b200h': super_busy, 'total_b200h': ds_h + super_h,
        'idle_b200h': ds_h - ds_busy + super_h - super_busy,
        'ds_per_1k_tasks': 1000 * ds_h / tasks, 'super_per_1B': 1e9 * super_h / unique,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, (v, unit, src) in ASSUMPTIONS.items():
        ap.add_argument('--' + k.replace('_', '-'), type=float, default=v, help=f'{unit} ({src})')
    ap.add_argument('--ds-servers', type=int, nargs='+', default=[2, 3, 4])
    ap.add_argument('--engines', type=int, nargs='+', default=None,
                    help='Super engines per layout (default: 8, and the balanced count for each server count)')
    ap.add_argument('--assumptions', action='store_true')
    a = ap.parse_args()
    p = {k: getattr(a, k) for k in ASSUMPTIONS}
    print('| assumption | value | unit | source |\n|---|---|---|---|')
    for k, (v, unit, src) in ASSUMPTIONS.items():
        mark = '' if p[k] == v else ' (overridden)'
        print(f'| {k} | {p[k]:,.6g}{mark} | {unit} | {src} |')
    if a.assumptions:
        return
    rows = []
    for n in a.ds_servers:
        bal = size(p, n, 1)['balanced_engines']
        for e in (a.engines or sorted({8, max(1, math.ceil(bal))})):
            rows.append((n, e, size(p, n, e)))
    r0 = rows[0][2]
    print(f"\nDeepSeek tokens per accepted task: {ds_tokens_per_task(p) / 1e3:.0f}k")
    print(f"\n{r0['unique_tokens'] / 1e9:.2f}B unique tokens, {r0['tasks']:,.0f} tasks "
          f"({r0['tasks_generated']:,.0f} generated with the stop margin); one DeepSeek server supplies "
          f"{r0['supply_tasks_per_h'] / a.ds_servers[0]:,.0f} tasks/h, one engine traces {r0['engine_tasks_per_h']:,.0f} tasks/h")
    print('\n| DeepSeek servers | Super engines | balanced engines | bound | wall h | DeepSeek B200-h (restarts) '
          '| Super B200-h | total | idle B200-h | DeepSeek per 1k tasks | Super per 1B |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for n, e, r in rows:
        print(f"| {n} | {e} | {r['balanced_engines']:.1f} | {r['bound']} | {r['wall_h']:.0f} | "
              f"{r['ds_b200h']:,.0f} ({r['ds_restarts']}) | {r['super_b200h']:,.0f} | {r['total_b200h']:,.0f} | "
              f"{r['idle_b200h']:,.0f} | {r['ds_per_1k_tasks']:.2f} | {r['super_per_1B']:.0f} |")


if __name__ == '__main__':
    main()
