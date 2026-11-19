"""Auto budget (#378), offline: the controller (autobudget.AutoBudget, pure) against simulated servers built from c30's
knee curves (audits/c30_efficiency_20261113/serving: vLLM 10-s stats binned by running, per model).

Each simulated server: generated tok/s = curve(running) x a slow workload drift (AR(1), its 5-min window-to-window
spread measured in c30 at steady running) x per-tick noise, with a prefill dip when many attempts start at once.
running = (running / in-flight ratio, measured in c30's loaded phases) x in flight, capped by max-num-seqs (the rest
waits). KV (p50 and p90) and the waiting share follow the c30 bins too. Beyond the highest running c30 reached, the curve continues
with its last measured elasticity, decaying as (r_last / r)^2, and KV linearly; KV p90 >= 92% then thrashes
(tok/s falls, requests wait, preemptions). In flight follows the budget: up within a minute (burst pacing), down only
as attempts finish (mean 30 min). Super's curve falls above ~110 running (c30: 5.9k -> 4.4k tok/s at 129-192).

The claim tested: started from its c30 budget, the controller settles near each model's knee: Super well below 182
(about 128), Gemma, Qwen3.6 and Qwen3.5-4B above their c30 budgets (121, 130, 132; Qwen3.6 then stops at its
max-num-seqs of 128 running), Lightning (at its knee in c30) roughly where it was.
"""
import bisect
import math
import random

import pytest

from aa_evals.common.dispatch import autobudget as ab

TICK = 15.0
MEAN_SEC = 1800.0   # mean attempt duration: how fast in flight drains after a cut

# c30 per-model curves: running -> generated tok/s (medians of 16-wide running bins with >= 20 samples, smoothed
# where the bins are thin), KV p50, waiting share; running / in-flight ratio in loaded phases; max-num-seqs;
# window-to-window log tok/s spread at steady running (5-min windows).
MODELS = {
    'super-vl': dict(budget=182, mns=512, ratio=0.90, sigma=0.12,
                     r=[0, 16, 56, 80, 104, 120, 136, 160, 192, 256],
                     tps=[0, 1500, 4800, 5400, 6100, 5800, 4800, 4400, 3500, 2500],
                     kv=[0, .07, .21, .25, .33, .44, .64, .76, .86, .95],
                     kv90=[0, .095, .235, .25, .337, .573, .90, .955, .97, .99],
                     wait=[0, 0, 0, 0, .02, .10, .20, .30, .35, .50]),
    'gemma26': dict(budget=121, mns=256, ratio=0.88, sigma=0.045,
                    r=[0, 16, 40, 56, 72, 88, 104, 120], tps=[0, 4000, 7733, 9170, 9994, 10604, 12402, 13158],
                    kv=[0, .048, .089, .114, .154, .19, .234, .233],
                    kv90=[0, .072, .107, .148, .188, .289, .277, .259], wait=[0] * 8),
    'qwen36': dict(budget=130, mns=128, ratio=0.70, sigma=0.07,
                   r=[0, 16, 40, 56, 72, 88, 104], tps=[0, 2898, 3155, 4337, 5457, 6397, 7274],
                   kv=[0, .08, .162, .252, .367, .391, .434], kv90=[0, .119, .328, .495, .585, .573, .605],
                   wait=[.01, .01, .01, .02, .04, .04, .03]),
    'qwen35-4b': dict(budget=132, mns=256, ratio=0.82, sigma=0.08,
                      r=[0, 16, 40, 56, 72, 88, 104, 120], tps=[0, 1800, 2543, 3036, 3176, 5303, 5315, 5913],
                      kv=[0, .138, .211, .26, .287, .302, .345, .376],
                      kv90=[0, .175, .252, .304, .335, .334, .404, .461], wait=[0] * 8),
    'lightning': dict(budget=139, mns=512, ratio=0.80, sigma=0.06,
                      r=[0, 16, 40, 56, 72, 88, 104, 120, 136], tps=[0, 2455, 3194, 4611, 5290, 5482, 5501, 6170, 5935],
                      kv=[0, .099, .215, .304, .412, .573, .618, .719, .783],
                      kv90=[0, .141, .256, .438, .485, .788, .757, .787, .825],
                      wait=[0, .01, .04, 0, .01, .03, .17, .18, .19]),
}


def interp(xs, ys, x):
    i = bisect.bisect_right(xs, x)
    if i <= 0:
        return ys[0]
    if i >= len(xs):
        return ys[-1]
    f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return ys[i - 1] + f * (ys[i] - ys[i - 1])


class Server:
    def __init__(self, m: dict):
        self.m = m
        r, t = m['r'], m['tps']
        # last measured elasticity (d ln tps / d ln running), continued beyond the data with decay (r_last / r)^2
        self.e0 = max(0.0, math.log(t[-1] / t[-3]) / math.log(r[-1] / r[-3]))
        self.kv_slope = m['kv'][-1] / r[-1]

    def at(self, running: float) -> tuple[float, float, float, float]:
        """(tok/s, KV p50, KV p90, waiting share) at this many running sequences."""
        m, r = self.m, self.m['r']
        if running <= r[-1]:
            tps, kv, wait = interp(r, m['tps'], running), interp(r, m['kv'], running), interp(r, m['wait'], running)
            kv90 = interp(r, m['kv90'], running)
        else:
            tps = m['tps'][-1] * math.exp(self.e0 / 2 * (1 - (r[-1] / running) ** 2))
            kv, wait = max(m['kv'][-1], self.kv_slope * running), m['wait'][-1]
            kv90 = kv * m['kv90'][-1] / m['kv'][-1]
            if kv90 >= 0.92:   # KV-bound beyond the data: thrash
                tps *= max(0.4, 1 - 3 * (kv90 - 0.92))
                wait = max(wait, 0.3)
        return tps, kv, max(kv90, kv), wait


def simulate(model: str, hours: float = 8.0, seed: int = 0, start: int | None = None, auto: bool = True) -> dict:
    m = MODELS[model]
    srv, rnd = Server(m), random.Random(seed)
    b0 = start or m['budget']
    ctl = ab.AutoBudget(b0, max(16, b0 // 2), 2 * b0, max_num_seqs=m['mns'])
    in_flight, drift, ratio_dev = 0.0, 0.0, 0.0
    gen, pre, starts = 0.0, 0.0, []
    phi = math.exp(-TICK / 600.0)   # drift correlation time 10 min
    trace = []
    t = 0.0
    while t < hours * 3600:
        b = ctl.step(t, TICK)[0] if auto else b0
        # in flight: fills toward the budget at 16 starts/min per lane x 5 lanes; drains only as attempts finish
        fin = in_flight * TICK / MEAN_SEC
        room = b - (in_flight - fin)
        new = max(0.0, min(room, 20.0))
        in_flight = in_flight - fin + new
        starts.append(new - fin)          # excess starts over replacement: prefill bursts
        burst = sum(max(0.0, x) for x in starts[-8:])
        drift = phi * drift + math.sqrt(1 - phi * phi) * m['sigma'] * 1.4 * rnd.gauss(0, 1)
        ratio_dev = 0.98 * ratio_dev + 0.2 * 0.03 * rnd.gauss(0, 1)
        want = max(0.0, (m['ratio'] + ratio_dev) * in_flight)
        running = min(want, m['mns'])
        tps, kv50, kv90, wshare = srv.at(running)
        tps *= math.exp(drift + 0.08 * rnd.gauss(0, 1)) * max(0.7, 1 - 0.01 * burst)
        spread = math.log(max(kv90, 1e-6) / max(kv50, 1e-6)) / 1.2816 if kv50 > 0 else 0.0   # p90 as in c30
        kv = min(1.0, kv50 * math.exp(spread * rnd.gauss(0, 1)))
        waiting = (want - running) + (1 if rnd.random() < wshare else 0)
        if kv >= 0.97 and rnd.random() < 0.5:
            pre += 1
        gen += tps * TICK
        filled = in_flight >= b - max(2, 0.05 * b)
        ctl.sample(t, {'gen_total': gen, 'preempt_total': pre, 'running': running, 'waiting': waiting, 'kv': kv,
                       'engines': 1}, filled, False)
        trace.append((t, b, running, tps))
        t += TICK
    half = [x for x in trace if x[0] >= hours * 3600 / 2]
    bs = sorted(x[1] for x in half)
    return {'median_budget': bs[len(bs) // 2], 'p10': bs[len(bs) // 10], 'p90': bs[9 * len(bs) // 10],
            'mean_tps': sum(x[3] for x in half) / len(half), 'mean_running': sum(x[2] for x in half) / len(half),
            'changes': sum(1 for a, c in zip(trace, trace[1:]) if a[1] != c[1])}


def summary(model, seeds=range(6)):
    runs = [simulate(model, seed=s) for s in seeds]
    med = sorted(r['median_budget'] for r in runs)[len(runs) // 2]
    fixed = sum(simulate(model, seed=s, auto=False)['mean_tps'] for s in seeds) / len(runs)
    return med, runs, sum(r['mean_tps'] for r in runs) / len(runs) / fixed


def test_super_settles_near_its_knee_from_182():
    med, runs, gain = summary('super-vl')
    print('super', med, gain, runs)
    assert 105 <= med <= 150                       # about 128, not 182
    assert all(r['median_budget'] < 165 for r in runs)
    assert gain > 1.10                             # +10% or more generated tok/s over the fixed 182


@pytest.mark.parametrize('model', ['gemma26', 'qwen36', 'qwen35-4b'])
def test_underfilled_models_go_above_their_c30_budget(model):
    med, runs, gain = summary(model)
    print(model, med, gain, runs)
    assert med > MODELS[model]['budget'] * 1.08
    assert gain > 1.0
    if model == 'qwen36':   # stops at max-num-seqs 128 running: the rest would only wait in the engine
        assert all(r['mean_running'] <= 128 for r in runs) and med < 2 * MODELS[model]['budget']


def test_lightning_stays_near_its_knee():
    med, runs, gain = summary('lightning')
    print('lightning', med, gain, runs)
    assert 100 <= med <= 175 and gain > 0.95


# ------------------------------------------------------------------------------------------- unit rules

def feed(ctl, t0, n, tps, kv=0.3, waiting=0, preempt=0, filled=True, gate=False, running=50, engines=1, gen0=0.0):
    gen, pre = gen0, 0.0
    t = t0
    for _ in range(n):
        gen += tps * TICK
        pre += preempt
        ctl.sample(t, {'gen_total': gen, 'preempt_total': pre, 'running': running, 'waiting': waiting, 'kv': kv,
                       'engines': engines}, filled, gate)
        t += TICK
    return t, gen


def window(ctl, t, **kw):
    t, _ = feed(ctl, t, int(ab.DWELL / TICK) + 1, **kw)
    return t, ctl.step(t, TICK)


def test_raise_keep_revert_and_block():
    c = ab.AutoBudget(100, 50, 200, now=0.0)
    t, (b, why) = window(c, 0.0, tps=5000)
    assert b == 110 and 'raise' in why
    t, (b, why) = window(c, t, tps=5500)            # +10%: kept, raised again
    assert b == 121
    t, (b, why) = window(c, t, tps=5500)            # flat: reverted to 110, 121 blocked
    assert b == 110 and 'reverted' in why
    t, (b, why) = window(c, t, tps=5500)
    assert b == 110 and 'blocked' in why
    t, (b, _) = window(c, t + ab.BLOCK, tps=5500)    # block expired: probed again
    assert b == 121


def test_overload_cuts_and_unfilled_holds():
    c = ab.AutoBudget(180, 50, 300, now=0.0)
    t, (b, why) = window(c, 0.0, tps=4000, kv=0.95, waiting=3)
    assert b == 162 and 'overload' in why
    t, (b, why) = window(c, t, tps=4000, kv=0.5, preempt=1)
    assert b == round(162 * ab.CUT) and 'overload' in why
    t, (b2, why) = window(c, t, tps=4000, filled=False)
    assert b2 == b and 'not filled' in why


def test_engine_limit_reverts_a_raise_and_hi_holds():
    c = ab.AutoBudget(100, 50, 200, max_num_seqs=128, now=0.0)
    t, (b, _) = window(c, 0.0, tps=5000, running=100)
    assert b == 110
    t, (b, why) = window(c, t, tps=6000, running=125)   # 98% of max-num-seqs: the raise only queued
    assert b == 100 and 'limit' in why
    c = ab.AutoBudget(100, 50, 105, now=0.0)
    t, (b, why) = window(c, 0.0, tps=5000)
    assert b == 105                                     # a raise stops at --budget-max
    t, (b, why) = window(c, t, tps=6000)
    assert b == 105 and 'budget-max' in why
    c = ab.AutoBudget(100, 50, 200, now=0.0)
    t, (b, why) = window(c, 0.0, tps=5000, gate=True)    # the gate holds: more budget waits at the gate
    assert b == 100 and 'gate' in why


def test_a_raise_is_judged_after_an_unfilled_window():
    c = ab.AutoBudget(100, 50, 200, now=0.0)
    t, (b, _) = window(c, 0.0, tps=1000)
    assert b == 110
    t, (b, why) = window(c, t, tps=1000, filled=False)   # grown workers still starting
    assert b == 110 and 'judged in the next window' in why
    t, (b, why) = window(c, t, tps=1000)                  # flat: reverted, not raised again
    assert b == 100 and 'reverted' in why


def test_counter_reset_and_missing_scrapes():
    c = ab.AutoBudget(100, 50, 200, now=0.0)
    t, gen = feed(c, 0.0, 10, tps=5000)
    feed(c, t, 5, tps=5000, gen0=0.0)                     # engine restarted: counter went back, sample dropped
    assert all(s['tps'] > 0 for s in c.samples)
    c = ab.AutoBudget(100, 50, 200, now=0.0)
    b, why = c.step(ab.DWELL + 1, TICK)
    assert b == 100 and 'too few' in why


def test_parse_metrics_sums_engines_and_takes_the_fullest_kv():
    text = '\n'.join([
        '# HELP vllm:generation_tokens_total x',
        'vllm:generation_tokens_total{engine="0",model_name="m"} 100.0',
        'vllm:generation_tokens_total{engine="1",model_name="m"} 50.0',
        'vllm:num_requests_running{engine="0",model_name="m"} 30.0',
        'vllm:num_requests_running{engine="1",model_name="m"} 20.0',
        'vllm:num_requests_waiting{engine="0",model_name="m"} 1.0',
        'vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.4',
        'vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.7',
        'vllm:num_preemptions_total{engine="0",model_name="m"} 2.0'])
    m = ab.parse_metrics(text)
    assert m == {'gen_total': 150.0, 'preempt_total': 2.0, 'running': 50.0, 'waiting': 1.0, 'kv': 0.7, 'engines': 2}
    with pytest.raises(ValueError):
        ab.parse_metrics('nothing')


# ------------------------------------------------------------------------------------------- in the dispatcher

def test_dispatcher_auto_budget_raises_on_a_filled_improving_server(tmp_path, monkeypatch):
    import json
    import time as _time
    from aa_evals.common.dispatch import dispatcher as dsp
    from aa_evals.common.dispatch.lane import Lane
    from aa_evals.common.dispatch.test_dispatch import table
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '20', '--durations', str(table(tmp_path)), '--expect-lanes', '1',
              '--metrics', 'http://engine:1', '--max-num-seqs', '512'])
    cfg = json.loads((d / 'dispatch.json').read_text())
    assert cfg['auto_budget'] is True and cfg['pi_idle_exit_sec'] == 600.0
    lane = Lane.open(d, 'hle', run=tmp_path / 'hlerun', start_poller=False)
    lane.expect([(f'q{i}.r0', f'q{i}', 100) for i in range(500)])
    clock = [_time.time()]
    monkeypatch.setattr(_time, 'time', lambda: clock[0])
    D = dsp.Dispatcher(d)
    gen = [0.0]

    def scrape():   # an engine whose tok/s grows with the budget (no knee in reach)
        gen[0] += 100.0 * D.budget() * TICK
        return (f'vllm:generation_tokens_total{{engine="0"}} {gen[0]}\nvllm:num_requests_running{{engine="0"}} 10\n'
                'vllm:kv_cache_usage_perc{engine="0"} 0.2\nvllm:num_requests_waiting{engine="0"} 0\n')
    monkeypatch.setattr(D, 'scrape', scrape)
    budgets = []
    for _ in range(int(2 * ab.DWELL / TICK) + 3):
        lane.flush()
        st = D.tick()
        budgets.append(st['budget'])
        clock[0] += TICK
    assert budgets[0] == 20 and budgets[-1] == 24 and st['auto_budget']['budget'] == 24   # 20 -> 22 -> 24
    assert 'auto budget 20 -> 22' in (d / 'log.txt').read_text()
    assert json.loads((d / 'auto_budget.json').read_text())['budget'] == 24
    # a manual budget turns it off; `budget DIR auto` back on, from where the controller was
    dsp.main(['budget', str(d), '15'])
    lane.flush()
    assert D.tick()['budget'] == 15 and D.tick()['auto_budget'] is None
    dsp.main(['budget', str(d), 'auto'])
    lane.flush()
    assert D.tick()['budget'] == 24
    # a restarted dispatcher keeps the budget it had found
    assert dsp.Dispatcher(d).budget() == 24


def test_old_and_no_auto_dirs_keep_a_fixed_budget(tmp_path):
    import json
    from aa_evals.common.dispatch import dispatcher as dsp
    from aa_evals.common.dispatch.test_dispatch import table
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '20', '--durations', str(table(tmp_path)), '--metrics', 'http://e:1'])
    cfg = json.loads((d / 'dispatch.json').read_text())
    for k in ('auto_budget', 'budget_min', 'budget_max', 'max_num_seqs', 'pi_idle_exit_sec'):
        del cfg[k]                                    # a dir made before #378's auto budget (c30n)
    (d / 'dispatch.json').write_text(json.dumps(cfg))
    D = dsp.Dispatcher(d)
    assert not D.auto_on() and D.budget() == 20
    d2 = tmp_path / 'disp2'
    dsp.main(['init', str(d2), '--budget', '20', '--durations', str(table(tmp_path)), '--metrics', 'http://e:1',
              '--no-auto-budget'])
    assert not dsp.Dispatcher(d2).auto_on()
    d3 = tmp_path / 'disp3'                           # no --metrics: nothing to climb on
    dsp.main(['init', str(d3), '--budget', '20', '--durations', str(table(tmp_path))])
    assert not dsp.Dispatcher(d3).auto_on()


def test_max_num_seqs_from_the_registry_launch(tmp_path, monkeypatch):
    import json
    import aa_evals.common.harbor_mswea.endpoint as ep
    from aa_evals.common.dispatch import dispatcher as dsp
    from aa_evals.common.dispatch.test_dispatch import table
    monkeypatch.setattr(ep, 'SERVING_ROOT', tmp_path / 'serving')
    (tmp_path / 'serving' / 'reg').mkdir(parents=True)
    (tmp_path / 'serving' / 'reg' / 'launch.json').write_text(json.dumps(
        {'replicas': [{'argv': ['vllm', 'serve', 'm', '--max-num-seqs', '128']}, {'argv': ['--max-num-seqs=256']}]}))
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '20', '--durations', str(table(tmp_path)), '--metrics', 'http://e:1',
              '--registry', 'reg'])
    D = dsp.Dispatcher(d)
    assert D.max_num_seqs() == 256 and D.auto_budget().max_num_seqs == 256
