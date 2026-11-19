"""#477's collapse gate, its HTTP front and the client's fail-open; the verbatim copies stay verbatim."""
import inspect
from pathlib import Path
import unittest

from tbsched import gate, kv_fit, turngate
from tbsched.gate import CollapseGate, GateServer, _post, report_done, report_end, wait_turn

ROOT = Path(__file__).resolve().parents[2]


def turn(g, k, prompt, cached, now, completion=500):
    """One request of session k: ask, then report it finished with `cached` prefix tokens."""
    go = g.ask(k, prompt, now)
    if go:
        g.finished(k, prompt, completion, cached, now + 1)
    return go


class Collapse(unittest.TestCase):
    def test_open_until_the_cache_collapses_then_hold_the_youngest(self):
        g = CollapseGate(reopen_sec=300)
        for i in range(10):
            g.g.register(f's{i}', order=i, now=0)
        t, prompt = 0, 50_000
        for rnd in range(12):    # a round every 20 s; rounds 6+ find nothing cached (thrash)
            for i in range(10):
                turn(g, f's{i}', prompt + 1000, 0 if rnd >= 6 else prompt if rnd else 0, t)
                t += 2
            prompt += 1000
            g.tick(t)
            if rnd == 5:
                self.assertEqual(g.mode, 'open')
        self.assertEqual(g.mode, 'hold')
        self.assertEqual(g.triggers[-1][1], 'gate_efficiency')
        allowed = [g.ask(f's{i}', prompt + 1000, t + 1) for i in range(10)]
        self.assertTrue(allowed[0])            # the oldest claims keep going
        self.assertFalse(allowed[-1])          # the youngest wait at their turn boundary
        self.assertLess(sum(allowed), 10)

    def test_reopens_after_clean_minutes_with_everyone_allowed(self):
        g = CollapseGate(reopen_sec=300)
        for i in range(4):
            g.g.register(f's{i}', order=i, now=0)
            g.finished(f's{i}', 40_000, 100, 0, 0)
        g.trigger(0, 'test', capacity=1_000_000)
        self.assertEqual(g.mode, 'hold')
        g.g.fill = 10.0                         # room for everyone again
        t, prev = 1, 40_000
        while t < 2000 and g.mode == 'hold':
            for i in range(4):
                turn(g, f's{i}', prev + 1000, prev, t)
            prev += 1000
            t += 30
            g.tick(t)
        self.assertEqual(g.mode, 'open')

    def test_a_trigger_from_the_server_uses_the_engine_capacity_as_the_unit(self):
        g = CollapseGate()
        for i in range(10):
            g.g.register(f's{i}', order=i, now=0)
            g.finished(f's{i}', 100_000, 100, 0, 0)
        g.trigger(5, 'hit_collapse', capacity=20_000_000)
        self.assertAlmostEqual(g.g.fill * g.g.capacity, 0.8 * 10 * 100_100)
        self.assertEqual(g.g.capacity, 20_000_000)


    def test_reopens_when_the_parked_sessions_ended_and_no_traffic_is_left(self):
        g = CollapseGate(reopen_sec=300)
        for i in range(10):
            g.g.register(f's{i}', order=i, now=0)
            g.finished(f's{i}', 100_000, 100, 0, 0)
        g.trigger(0, 'test')
        held = [k for k in g.g.r if k not in g.g.allowed]
        self.assertTrue(held)
        for k in list(g.g.allowed):
            g.end(k)                            # the allowed sessions finished
        g.sync({k: i for i, k in enumerate(held)}, 10)
        for t in range(20, 2000, 30):
            g.tick(t)
        self.assertEqual(g.mode, 'open')
        self.assertTrue(all(g.ask(k, 1, 2001) for k in held))


class Http(unittest.TestCase):
    def test_round_trip_and_fail_open(self):
        s = GateServer(host='127.0.0.1', tick=3600)
        try:
            s.add('e0')
            self.assertLess(wait_turn(s.url, 'k1', 'e0', need=1000), 1.0)
            report_done(s.url, 'k1', 'e0', prompt=1000, completion=10, cached=0)
            self.assertIn('k1', s.gates['e0'].g.r)
            self.assertLess(wait_turn(s.url, 'k2', 'unknown-engine'), 1.0)   # no gate for it: go
            report_end(s.url, 'k1', 'e0')
            self.assertNotIn('k1', s.gates['e0'].g.r)
            _post(s.url, '/sync', {'engine': 'e0', 'keys': {'a': 1, 'b': 2}}, 5)
            self.assertEqual(set(s.gates['e0'].g.r), {'a', 'b'})
            g = s.gates['e0']
            g.g.capacity, g.g.fill, g.mode = 1.0, 1e-9, 'hold'     # holds everyone but the first
            g.g.r['a']['ctx'] = g.g.r['b']['ctx'] = 10
            g.g.allowed = {'a'}
            waited = wait_turn(s.url, 'b', 'e0', poll=0.05, max_wait=0.3)
            self.assertTrue(0.25 <= waited < 2.0)                  # held, then failed open at max_wait
        finally:
            s.close()
        self.assertLess(wait_turn('http://127.0.0.1:9', 'k', timeout=0.5), 2.0)   # nothing listening: go


class Copies(unittest.TestCase):
    """Moved verbatim; until the pipelines import them from tbsched, both copies must stay identical."""

    def test_kv_fit_is_synth_admission(self):
        self.assertEqual(Path(kv_fit.__file__).read_text(), (ROOT / 'synth' / 'admission.py').read_text())

    def test_turngate_is_eval_q35s(self):
        src = (ROOT / 'eval_q35' / 'turn_gate.py').read_text()
        self.assertEqual(inspect.getsource(turngate.TurnGate).strip(), src[src.index('class TurnGate:'):].strip())


if __name__ == '__main__':
    unittest.main()
