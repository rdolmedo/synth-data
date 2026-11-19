"""c30 live-run loop (#344, round 2) reproduced in a simulation of one gated server, then the fix.

The serving gate admits about G requests' worth of expected KV. A SciCode chain in flight always has a request at the
gate (its steps run back to back); a Pi attempt's live turns are admitted without waiting and occupy the gate only
part of the time. pi_eval workers start no attempt while the gate holds ANY new conversation, rechecking every 30 s.
A short gate hold at the start (here: 60 s of another client's burst) pauses the Pi lane. The old, work-conserving
policy then hands the whole budget to SciCode, whose requests keep the gate holding, so Pi stays paused and its long
attempts, the head of the queue, start only after SciCode drains. The fix (decide: gate back-off + reservation for
gate-paused lanes) keeps the head at the head.
"""
from aa_evals.common.dispatch.dispatcher import decide

DT, G, BUDGET = 15.0, 70.0, 131
PI_N, PI_SEC, PI_PRED, PI_CAP, PI_GATE_SHARE = 200, 3000.0, 3000.0, 120, 0.3
SC_N, SC_STEPS, SC_STEP_SEC, SC_PRED = 400, 4, 450.0, 1800.0


def simulate(policy: str, horizon_h: float = 12.0, transient_sec: float = 60.0, pi_sec: float = PI_SEC) -> dict:
    t = 0.0
    pi_pending, pi_running, pi_starts, pi_credit, pi_start_t = PI_N, [], 0, 0, []
    sc_pending, sc_running, sc_starts, sc_credit = SC_N, [], 0, 0   # running: [steps left, sec left in this step]
    pi_paused, unspent_since, last_starts, gate_log = True, None, 0, []
    while t < horizon_h * 3600 and (pi_pending or pi_running or sc_pending or sc_running):
        # the gate: Pi live turns first, then SciCode requests up to what is left; the rest is held
        room_sc = max(0, int(G - PI_GATE_SHARE * len(pi_running)))
        queued = max(0, len(sc_running) - room_sc) + (10 if t < transient_sec else 0)
        gate_log.append(queued)
        # progress
        pi_running = [x - DT for x in pi_running]
        pi_running = [x for x in pi_running if x > 0]
        for c in sc_running[:room_sc]:
            c[1] -= DT
            if c[1] <= 0:
                c[0] -= 1
                c[1] = SC_STEP_SEC
        sc_running = [c for c in sc_running if c[0] > 0]
        # pi_eval workers recheck the gate every 30 s
        if t % 30 == 0:
            pi_paused = queued > 0
        # dispatcher tick
        views = {
            'pi': {'in_flight': len(pi_running), 'starts': pi_starts, 'credit': pi_credit, 'dead': False,
                   'capacity': 0 if pi_paused else PI_CAP, 'withdrawable': True,
                   'gate_paused': pi_paused and policy == 'new', 'worker_capacity': PI_CAP,
                   'unspent_since': unspent_since, 'pending': [(pi_sec, 1)] * min(pi_pending, BUDGET)},
            'sc': {'in_flight': len(sc_running), 'starts': sc_starts, 'credit': sc_credit, 'dead': False,
                   'capacity': 10 ** 6, 'withdrawable': False, 'gate_paused': False, 'worker_capacity': 10 ** 6,
                   'unspent_since': None, 'pending': [(SC_PRED, 1)] * min(sc_pending, BUDGET)}}
        grants, withdraw, _ = decide(views, BUDGET, queued if policy == 'new' else None, t)
        pi_credit = withdraw.get('pi', max(pi_credit, pi_starts)) + grants['pi']
        sc_credit = max(sc_credit, sc_starts) + grants['sc']
        # lanes start what their credit allows (Pi only while its workers see the gate clear)
        n = min(sc_credit - sc_starts, sc_pending)
        sc_running += [[SC_STEPS, SC_STEP_SEC] for _ in range(n)]
        sc_pending -= n
        sc_starts += n
        if not pi_paused:
            n = max(0, min(pi_credit - pi_starts, pi_pending, PI_CAP - len(pi_running)))
            pi_running += [pi_sec] * n
            pi_pending -= n
            pi_starts += n
            pi_start_t += [t] * n
        if pi_credit > pi_starts:
            unspent_since = t if (unspent_since is None or pi_starts > last_starts) else unspent_since
        else:
            unspent_since = None
        last_starts = pi_starts
        t += DT
    pi_start_t.sort()
    return {'makespan_h': t / 3600, 'pi_first_start_h': pi_start_t[0] / 3600,
            'pi_median_start_h': pi_start_t[len(pi_start_t) // 2] / 3600,
            'pi_120th_start_h': pi_start_t[119] / 3600, 'pi_started': len(pi_start_t),
            'gate_held_share': sum(1 for q in gate_log if q) / len(gate_log)}


def test_old_policy_loops_new_policy_keeps_the_head():
    old, new = simulate('old', pi_sec=7200.0), simulate('new', pi_sec=7200.0)
    # the loop: the Pi lane (queue head: attempts at the 2 h cap) waits until SciCode has drained
    assert old['pi_first_start_h'] > 2.0 and old['gate_held_share'] > 0.4
    # fixed: the first 120 Pi attempts start within 2 min of the transient, and the tail is gone
    assert new['pi_120th_start_h'] < 3 / 60 and new['pi_started'] == 200
    assert new['makespan_h'] < 0.9 * old['makespan_h']   # 6.5 h -> 5.5 h
    print('old', old, '\nnew', new)


def test_non_gated_server_unchanged():
    """No gate (gate_queued None) and no gate pause: decide() is the round-1 allocation."""
    views = {'a': {'in_flight': 2, 'starts': 5, 'credit': 6, 'capacity': 10, 'gate_paused': False,
                   'worker_capacity': 10, 'dead': False, 'withdrawable': True, 'unspent_since': 0.0,
                   'pending': [(900, 1), (800, 1), (700, 1)]},
             'b': {'in_flight': 0, 'starts': 0, 'credit': 0, 'capacity': 10 ** 6, 'gate_paused': False,
                   'worker_capacity': 10 ** 6, 'dead': False, 'withdrawable': False, 'unspent_since': None,
                   'pending': [(1000, 1), (50, 1)]}}
    grants, withdraw, info = decide(views, 6, None, 10 ** 9)
    # used: a = 2 + 1 unspent; free 3 -> b 1000, a 800 (900 is covered by a's unspent credit), a 700
    assert grants == {'a': 2, 'b': 1} and not withdraw and not info['gate_hold']


def test_withdrawn_lane_not_regranted_and_bounded_without_gate():
    v = {'p': {'in_flight': 0, 'starts': 3, 'credit': 8, 'capacity': 0, 'gate_paused': True, 'worker_capacity': 8,
               'dead': False, 'withdrawable': True, 'unspent_since': 0.0, 'pending': [(100, 1)] * 5},
         's': {'in_flight': 0, 'starts': 0, 'credit': 0, 'capacity': 10 ** 6, 'gate_paused': False,
               'worker_capacity': 10 ** 6, 'dead': False, 'withdrawable': False, 'unspent_since': None,
               'pending': [(10, 1)] * 5}}
    for gq in (0, None):   # gate clear, or no gate signal (unreachable / --gate-backoff off): bounded either way
        grants, withdraw, _ = decide(v, 10, gq, 10 ** 4)
        assert withdraw == {'p': 3} and grants == {'p': 0, 's': 5}   # not handed straight back (8 -> 3 -> 8)
    v['p'].update(credit=3, withdrawn=True, unspent_since=None)
    assert decide(v, 10, 0, 10 ** 4)[0]['p'] == 0                     # still gate-paused: no new reservation
    v['p'].update(withdrawn=False, gate_paused=False, capacity=8)
    assert decide(v, 10, 0, 10 ** 4)[0]['p'] == 5                     # its workers see the gate clear again


def test_withdrawal_only_with_gate_clear():
    v = {'p': {'in_flight': 0, 'starts': 3, 'credit': 8, 'capacity': 0, 'gate_paused': True, 'worker_capacity': 8,
               'dead': False, 'withdrawable': True, 'unspent_since': 0.0, 'pending': [(100, 1)] * 5}}
    assert decide(v, 10, 4, 10 ** 4)[1] == {}                          # gate still holding: the reservation stays
    assert decide(v, 10, 0, 10 ** 4)[1] == {'p': 3}                    # clear for > reserve_sec and unused: back
    assert decide(v, 10, 0, 100.0)[1] == {}                            # not yet


# ---------------------------------------------------------------------------------------------------------------------
# #378 round 3: per-replica Pi pause (pi_eval worker) and the single-turn hold (decide).
#
# c30 Qwen3.8 Xhigh, 2 replicas: SciCode chains (round-robin over replicas) filled the gate of the replica with less
# room; the other had room. Pi workers paused while ANY replica's gate held, and SciCode's client, at its own
# concurrency cap, had ~46 credits unspent: during the back-off each finished chain started a new one from them, so
# the gate queue never drained and every Pi lane paused ~50 min. Round 3: a Pi worker pauses only while EVERY replica's
# gate holds (a new attempt whose hashed replica holds runs on a clear one), and while a Pi lane is gate-paused the
# single-turn lanes get no credit and their unspent credit is withdrawn.

def simulate_replicas(policy: str, rooms=(50.0, 30.0), horizon_h: float = 14.0, pi_n: int = 200,
                      pi_sec: float = 3000.0, sc_conc: int = 84, sc_unspent: int = 46, transient_sec: float = 0.0) -> dict:
    """Starts where c30 Xhigh was when the loop began: SciCode at its client's concurrency (sc_conc chains in
    flight, spread over the replicas, `sc_unspent` credits granted and not yet spent), the Pi lane paused."""
    t, reps = 0.0, len(rooms)
    pi_pending, pi_running, pi_starts, pi_credit, pi_start_t = pi_n, [], 0, 0, []   # running: [sec left, replica]
    # running chains: [steps left, sec left in this step, replica of its current request]
    sc_running = [[1 + i % SC_STEPS, SC_STEP_SEC * (1 + i % 3) / 3, i % reps] for i in range(sc_conc)]
    sc_pending, sc_starts, sc_credit = SC_N - sc_conc, sc_conc, sc_conc + sc_unspent
    pi_paused, held, unspent_since, last_starts, n = True, set(range(reps)), None, 0, sc_conc
    paused_ticks = ticks = 0
    while t < horizon_h * 3600 and (pi_pending or pi_running or sc_pending or sc_running):
        queued, admitted = [], []
        for r in range(reps):   # each replica's gate: Pi live turns first, then SciCode requests up to what is left
            room = max(0, int(rooms[r] - PI_GATE_SHARE * sum(1 for x in pi_running if x[1] == r)))
            mine = [c for c in sc_running if c[2] == r]
            admitted += mine[:room]
            queued.append(max(0, len(mine) - room) + (10 if t < transient_sec else 0))
        for x in pi_running:
            x[0] -= DT
        pi_running = [x for x in pi_running if x[0] > 0]
        for c in admitted:
            c[1] -= DT
            if c[1] <= 0:
                c[0] -= 1
                c[1] = SC_STEP_SEC
                c[2] = (c[2] + 1) % reps          # the chain's next request goes to the next replica (round-robin)
        sc_running = [c for c in sc_running if c[0] > 0]
        if t % 30 == 0:                           # pi_eval workers re-read every replica's gate every 30 s
            held = {r for r in range(reps) if queued[r] > 0}
            pi_paused = bool(held) if policy == 'r2' else len(held) == reps
        ticks += pi_pending > 0
        paused_ticks += pi_pending > 0 and pi_paused
        new = policy == 'r3'   # 'repin': the per-replica pause alone (no single-turn hold)
        views = {
            'pi': {'in_flight': len(pi_running), 'starts': pi_starts, 'credit': pi_credit, 'dead': False, 'kind': 'pi',
                   'capacity': 0 if pi_paused else PI_CAP, 'withdrawable': True, 'gate_paused': pi_paused,
                   'worker_capacity': PI_CAP, 'unspent_since': unspent_since,
                   'pending': [(pi_sec, 1)] * min(pi_pending, BUDGET)},
            'sc': {'in_flight': len(sc_running), 'starts': sc_starts, 'credit': sc_credit, 'dead': False,
                   'kind': 'lane', 'capacity': 10 ** 6, 'withdrawable': False, 'gate_paused': False,
                   'worker_capacity': 10 ** 6, 'unspent_since': None, 'shrinkable': new,
                   'pending': [(SC_PRED, 1)] * min(sc_pending, BUDGET)}}
        grants, withdraw, _ = decide(views, BUDGET, min(queued), t, st_hold_on=new)
        pi_credit = withdraw.get('pi', max(pi_credit, pi_starts)) + grants['pi']
        sc_credit = withdraw.get('sc', max(sc_credit, sc_starts)) + grants['sc']
        # SciCode starts chains while its credit allows, up to its client's own concurrency
        k = max(0, min(sc_credit - sc_starts, sc_pending, sc_conc - len(sc_running)))
        for _ in range(k):
            sc_running.append([SC_STEPS, SC_STEP_SEC, n % reps])
            n += 1
        sc_pending -= k
        sc_starts += k
        if not pi_paused:
            clear = [r for r in range(reps) if r not in held] or list(range(reps))
            k = max(0, min(pi_credit - pi_starts, pi_pending, PI_CAP - len(pi_running)))
            for i in range(k):
                hashed = (pi_starts + i) % reps
                rep = hashed if (policy == 'r2' or hashed not in held) else clear[i % len(clear)]
                pi_running.append([pi_sec, rep])
            pi_pending -= k
            pi_starts += k
            pi_start_t += [t] * k
        if pi_credit > pi_starts:
            unspent_since = t if (unspent_since is None or pi_starts > last_starts) else unspent_since
        else:
            unspent_since = None
        last_starts = pi_starts
        t += DT
    pi_start_t.sort()
    return {'makespan_h': round(t / 3600, 2), 'pi_first_start_h': round(pi_start_t[0] / 3600, 2),
            'pi_median_start_h': round(pi_start_t[len(pi_start_t) // 2] / 3600, 2),
            'pi_120th_start_h': round(pi_start_t[119] / 3600, 2), 'pi_started': len(pi_start_t),
            'pi_paused_share': round(paused_ticks / max(1, ticks), 3)}


def test_round3_two_replicas_xhigh():
    """One replica's gate full of SciCode, the other with room: round 2 pauses Pi until SciCode drains; round 3 keeps
    Pi claiming on the clear replica and the single-turn hold stops SciCode refilling the gate."""
    r2, r3 = simulate_replicas('r2', pi_sec=7200.0), simulate_replicas('r3', pi_sec=7200.0)
    print('\nround 2', r2, '\nround 3', r3)
    assert r2['pi_paused_share'] > 0.3 and r2['pi_120th_start_h'] > 1.0
    assert r3['pi_paused_share'] < 0.1 and r3['pi_120th_start_h'] < 1.0 and r3['pi_started'] == 200
    assert r3['makespan_h'] < 0.9 * r2['makespan_h']             # 6.8 h -> 5.6 h
    repin = simulate_replicas('repin', pi_sec=7200.0)             # per-replica pause alone: helps, not enough
    print('repin only', repin)
    assert r3['pi_120th_start_h'] < repin['pi_120th_start_h'] < r2['pi_120th_start_h']


def test_round3_one_replica_single_turn_hold():
    """Fix 2 alone (one replica: per-replica pausing changes nothing). SciCode at its client's concurrency cap with
    unspent credit: round 2's back-off still lets it refill the gate from that credit; round 3 withdraws it."""
    r2, r3 = simulate_replicas('r2', rooms=(65.0,), pi_sec=7200.0), simulate_replicas('r3', rooms=(65.0,), pi_sec=7200.0)
    print('\nround 2', r2, '\nround 3', r3)
    assert r3['pi_120th_start_h'] < r2['pi_120th_start_h'] and r3['pi_paused_share'] < r2['pi_paused_share']
    assert r3['makespan_h'] <= r2['makespan_h']
