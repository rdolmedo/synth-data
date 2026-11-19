"""Offline tests: python -m aa_evals.omniscience.test_omniscience  (pytest also collects them).

The end-to-end test runs run.main against a fake OpenAI-compatible server that answers, truncates, fails with
HTTP 500, and returns unparseable grades. It checks that infra failures are excluded and model failures scored."""
from __future__ import annotations

import json
import re
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import client, data, judge, metrics, run
from .prompts import OMNISCIENCE_ANSWER_PROMPT, OMNISCIENCE_GRADER_TEMPLATE


def test_metrics_reproduce_aa_super():
    # AA leaderboard 2026-11-02, Nemotron 3.5 Super VL on 3,000 questions: accuracy 0.211667, index -2.8667,
    # non-hallucination 0.695137. c = 635 and i = 721 give all three.
    m = metrics.from_counts(c=635, i=721, p=0, a=3000 - 635 - 721)
    assert abs(m['accuracy'] - 21.1667) < 1e-3
    assert abs(m['omniscience_index'] - -2.8667) < 1e-3
    assert abs(m['non_hallucination_rate'] - 69.5137) < 1e-3
    # Partial answers count like abstentions in both rates.
    m = metrics.from_counts(1, 1, 1, 1)
    assert m['accuracy'] == 25.0 and m['non_hallucination_rate'] == 100 * 2 / 3 and m['omniscience_index'] == 0.0
    assert m['non_hallucination_rate_cutoff_incorrect'] == m['non_hallucination_rate']
    # cutoffs: in N for accuracy, out of the non-hallucination denominator; the conservative variant counts them wrong
    m = metrics.from_counts(1, 1, 1, 1, x=2)
    assert m['accuracy'] == 100 / 6 and m['non_hallucination_rate'] == 100 * 2 / 3
    assert m['non_hallucination_rate_cutoff_incorrect'] == 100 * (1 - 3 / 5)
    assert m['omniscience_index_cutoff_incorrect'] == 100 * -2 / 6


def test_parse_letter():
    ok = {'A': 'A', ' B\n': 'B', '"C"': 'C', '**D**': 'D', 'A.': 'A', 'B: INCORRECT': 'B', 'C) PARTIAL_ANSWER': 'C',
          'NOT_ATTEMPTED': 'D', 'correct': 'A', 'Incorrect.': 'B'}
    for text, want in ok.items():
        assert judge.parse_letter(text) == want, text
    for text in ['', 'The answer is A', 'A or B', 'A: INCORRECT', 'E', 'AB']:
        assert judge.parse_letter(text) is None, text


def test_prompts_match_card():
    spec = data.DATASETS['v1.1']
    card = (spec.dir / 'README.md')
    if not card.exists():
        return
    text = card.read_text()
    ans = re.search(r'OMNISCIENCE_ANSWER_PROMPT = """(.*?)"""', text, re.S).group(1)
    grader = re.search(r'OMNISCIENCE_GRADER_TEMPLATE = """(.*?)"""\.strip\(\)', text, re.S).group(1).strip()
    assert ans == OMNISCIENCE_ANSWER_PROMPT
    assert grader == OMNISCIENCE_GRADER_TEMPLATE


def test_external_api_needs_flag():
    # User decision 2026-11-02: no closed API judge. openai: specs are refused without --allow-external-api.
    with tempfile.TemporaryDirectory() as tmp:
        for argv in (['--model', 'url:http://127.0.0.1:9/v1@x', '--judge', 'openai:gpt-5.6-luna'],
                     ['--model', 'openai:gpt-5.6-luna', '--judge', 'url:http://127.0.0.1:9/v1@x']):
            try:
                run.main(argv + ['--run-id', 'x', '--runs-root', tmp])
            except SystemExit as e:
                assert e.code == 2
            else:
                raise AssertionError('openai: accepted without --allow-external-api')
            assert not (Path(tmp) / 'x').exists()
    assert run.DEFAULT_JUDGE.startswith('serving:')


def test_subset_balanced_and_stable():
    _, items = data.load('v1.1')
    a = data.subset(items, 60)
    assert [x['item_id'] for x in a] == [x['item_id'] for x in data.subset(items, 60)]
    counts = {}
    for x in a:
        counts[x['domain']] = counts.get(x['domain'], 0) + 1
    assert sorted(counts.values()) == [10] * 6
    assert len(data.subset(items, None)) == 300


class _Fake(BaseHTTPRequestHandler):
    # Behaviour by question_id: 1-2 always HTTP 500 (infra), 3 truncated with empty content (model failure),
    # 4 judge always unparseable (infra), 5 abstains, 6 wrong; others answer correctly. 7 fails once then works.
    calls: dict = {}
    gold: dict = {}
    slow: set = set()  # question ids whose answer takes 2 s (deadline test)
    slow_judge: set = set()  # question ids whose grading takes 2 s

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {'data': [{'id': 'fake', 'max_model_len': 32768}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        msgs = req['messages']
        if msgs[0]['role'] == 'system':  # answering
            q = msgs[1]['content']
            qid = self.gold[q][0]
            n = self.calls[qid] = self.calls.get(qid, 0) + 1
            if qid in self.slow:
                time.sleep(2)
            if qid in ('1', '2') or (qid == '7' and n == 1):
                return self._send(500, {'error': 'boom'})
            if qid == '3':
                return self._choice('', 'length', reasoning='thinking...' * 10)
            if qid == '5':
                return self._choice("I don't know.", 'stop')
            if qid == '6':
                return self._choice('definitely-wrong', 'stop')
            return self._choice(self.gold[q][1], 'stop', reasoning='hmm')
        tail = msgs[0]['content'].split('Here is a new example.')[1]
        pred = re.search(r'Predicted answer: (.*)\n\n\nGrade the', tail, re.S).group(1)
        target = re.search(r'Gold target: (.*)\nPredicted answer:', tail, re.S).group(1)
        q = re.search(r'Question: (.*)\nGold target:', tail, re.S).group(1)
        if self.gold[q][0] in self.slow_judge:
            time.sleep(2)
        if self.gold[q][0] == '4':
            return self._choice('I think it is probably A', 'stop')
        if pred == target:
            return self._choice('A', 'stop')
        if "don't know" in pred:
            return self._choice('D', 'stop')
        return self._choice('B', 'stop')

    def _choice(self, content, finish, reasoning=None):
        self._send(200, {'choices': [{'message': {'role': 'assistant', 'content': content,
                                                  'reasoning_content': reasoning},
                                      'finish_reason': finish}],
                         'usage': {'completion_tokens': 5}})


def test_end_to_end_fake_server():
    _, items = data.load('v1.1')
    first = [it for it in items if int(it['item_id']) <= 12]
    _Fake.gold = {it['question']: (it['item_id'], it['answer']) for it in items}
    _Fake.calls = {}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'url:http://127.0.0.1:{srv.server_port}/v1@fake'
    # Run on exactly question ids 1..12 by monkeypatching the subset.
    orig_subset = data.subset
    data.subset = lambda its, limit, seed=0: [it for it in its if int(it['item_id']) <= 12]
    orig_sleep = run.client.time.sleep
    run.client.time.sleep = lambda s: None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['--model', base, '--judge', base, '--run-id', 't', '--limit', '12', '--repeats', '1', '--runs-root', tmp,
                    '--concurrency', '4']
            rc = run.main(argv)
            s = json.loads((Path(tmp) / 't' / 'summary.json').read_text())
            rows = run.load_rows(Path(tmp) / 't' / 'items.jsonl')
            st = {k[0]: r['status'] for k, r in rows.items()}
            assert st['1'] == st['2'] == st['4'] == 'infra_error', st
            assert st['3'] == 'model_error' and rows[('3', 0)]['grade'] == 'NOT_ATTEMPTED'
            assert rows[('3', 0)]['grade_source'] == 'rule_empty'
            assert rows[('5', 0)]['grade'] == 'NOT_ATTEMPTED' and rows[('6', 0)]['grade'] == 'INCORRECT'
            assert rows[('7', 0)]['grade'] == 'CORRECT'
            assert s['n_infra_errors'] == 3 and s['n_scored'] == 9 and s['n_truncated'] == 1
            assert s['grades'] == {'CORRECT': 6, 'INCORRECT': 1, 'PARTIAL_ANSWER': 0, 'NOT_ATTEMPTED': 2, 'CUTOFF': 0}
            assert s['valid'] is False and rc == 2  # 3/12 infra errors > 2%
            assert abs(s['score'] - 100 * 6 / 9) < 0.01
            assert len(first) == 12
            # Resume: item 4 is regraded without a new model call; 1 and 2 are retried.
            n4 = _Fake.calls['4']
            run.main(argv)
            assert _Fake.calls['4'] == n4
            assert _Fake.calls['1'] > 8
            # Regrade from the answers of run t with a fresh run id: no model calls at all.
            before = dict(_Fake.calls)
            run.main(['--answers-from', 't', '--judge', base, '--run-id', 't2', '--limit', '12', '--repeats', '1',
                      '--runs-root', tmp])
            assert _Fake.calls == before
            s2 = json.loads((Path(tmp) / 't2' / 'summary.json').read_text())
            assert s2['grades']['CORRECT'] == 6 and s2['n_infra_errors'] == 3
    finally:
        data.subset = orig_subset
        run.client.time.sleep = orig_sleep
        srv.shutdown()


def test_deadline_cutoff():
    """No tails: an item unanswered at --deadline-s counts as 0 (cutoff); an answer in hand at the deadline is still
    graded; regrades carry a cutoff over and a resume retries it."""
    _, items = data.load('v1.1')
    _Fake.gold = {it['question']: (it['item_id'], it['answer']) for it in items}
    _Fake.calls, _Fake.slow, _Fake.slow_judge = {}, {'8'}, {'9'}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'url:http://127.0.0.1:{srv.server_port}/v1@fake'
    orig_subset = data.subset
    data.subset = lambda its, limit, seed=0: [it for it in its if int(it['item_id']) in (5, 6, 8, 9)]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['--model', base, '--judge', base, '--run-id', 'd', '--limit', '4', '--repeats', '1', '--runs-root', tmp,
                    '--concurrency', '4', '--deadline-s', '1']
            rc = run.main(argv)
            s = json.loads((Path(tmp) / 'd' / 'summary.json').read_text())
            rows = run.load_rows(Path(tmp) / 'd' / 'items.jsonl')
            assert rows[('8', 0)]['status'] == 'cutoff' and rows[('8', 0)]['grade'] == 'CUTOFF'
            assert rows[('9', 0)]['status'] == 'ok' and rows[('9', 0)]['grade'] == 'CORRECT'  # judged past the deadline
            assert s['n_cutoff'] == 1 and s['n_scored'] == 4 and s['valid'] is True and rc == 0
            assert s['grades'] == {'CORRECT': 1, 'INCORRECT': 1, 'PARTIAL_ANSWER': 0, 'NOT_ATTEMPTED': 1, 'CUTOFF': 1}
            # 5 abstains, 6 is wrong: non-hallucination 1 - 1/2 without the cutoff, 1 - 2/3 counting it INCORRECT
            assert s['metrics']['non_hallucination_rate']['score'] == 50.0 and s['metrics']['accuracy']['score'] == 25.0
            assert s['metrics']['non_hallucination_rate_cutoff_incorrect']['score'] == 33.33
            time.sleep(2.5)  # let the abandoned request finish before the next run
            before = dict(_Fake.calls)
            run.main(['--answers-from', 'd', '--judge', base, '--run-id', 'd2', '--limit', '4', '--repeats', '1', '--runs-root', tmp])
            assert _Fake.calls == before
            s2 = json.loads((Path(tmp) / 'd2' / 'summary.json').read_text())
            assert s2['n_cutoff'] == 1 and s2['grades']['CUTOFF'] == 1
            _Fake.slow, _Fake.slow_judge = set(), set()
            run.main(argv)  # resume: only item 8 is asked again
            rows = run.load_rows(Path(tmp) / 'd' / 'items.jsonl')
            assert rows[('8', 0)]['grade'] == 'CORRECT' and _Fake.calls['9'] == before['9']
    finally:
        data.subset = orig_subset
        _Fake.slow, _Fake.slow_judge = set(), set()
        srv.shutdown()


def test_timeout_on_healthy_server_is_model_error():
    """A request with no response within --timeout while the server answers /models is a runaway generation:
    model_error with no answer (NOT_ATTEMPTED), not retried, not infra."""
    _, items = data.load('v1.1')
    _Fake.gold = {it['question']: (it['item_id'], it['answer']) for it in items}
    _Fake.calls, _Fake.slow = {}, {'8'}
    srv = ThreadingHTTPServer(('127.0.0.1', 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'url:http://127.0.0.1:{srv.server_port}/v1@fake'
    orig_subset = data.subset
    data.subset = lambda its, limit, seed=0: [it for it in its if int(it['item_id']) in (8, 9)]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run.main(['--model', base, '--judge', base, '--run-id', 'to', '--limit', '2', '--repeats', '1', '--runs-root', tmp,
                      '--timeout', '1'])
            s = json.loads((Path(tmp) / 'to' / 'summary.json').read_text())
            r = run.load_rows(Path(tmp) / 'to' / 'items.jsonl')[('8', 0)]
            assert r['status'] == 'model_error' and r['finish_reason'] == 'timeout' and r['grade'] == 'NOT_ATTEMPTED'
            assert _Fake.calls['8'] == 1 and s['n_timeouts'] == 1 and s['n_infra_errors'] == 0
    finally:
        data.subset = orig_subset
        _Fake.slow = set()
        srv.shutdown()



def test_gate_backpressure_uses_no_attempt():
    """A gate's 503 server_overloaded (Retry-After) is waited out, not counted against `attempts`."""
    n = {'calls': 0}

    class Gate(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            n['calls'] += 1
            if n['calls'] <= 5:
                body, code, hdr = json.dumps({'error': {'message': 'gate: server at capacity', 'type': 'server_overloaded',
                                                        'code': 503}}).encode(), 503, {'Retry-After': '0', 'X-Gate': '1'}
            else:
                body, code, hdr = json.dumps({'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}]}).encode(), 200, {}
            self.send_response(code)
            for k, v in hdr.items():
                self.send_header(k, v)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(('127.0.0.1', 0), Gate)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    orig = client.time.sleep
    client.time.sleep = lambda s: None
    try:
        ep = client.resolve(f'url:http://127.0.0.1:{srv.server_port}/v1@fake')
        resp = client.chat(ep, {'messages': []}, timeout=10, attempts=2)
        assert n['calls'] == 6 and resp['_gate_rejections'] == 5 and resp['_attempts'] == 1, (n, resp)
    finally:
        client.time.sleep = orig
        srv.shutdown()

if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('ok', name)


def test_stages_do_not_contact_inactive_server(tmp_path, monkeypatch):
    """Generate without resolving judge, then grade with the original model endpoint forbidden."""
    from types import SimpleNamespace
    items = [{'item_id': str(i), 'domain': 'Math', 'subtopic': 'arithmetic',
              'question': f'What is {i}+{i}?', 'answer': str(2*i)} for i in (3, 8, 9)]
    spec = SimpleNamespace(name='v1.1', repo='synthetic', revision='synthetic', sha256='synthetic', n=3, aa_version='1.1')
    monkeypatch.setattr(data, 'load', lambda dataset: (spec, items))
    monkeypatch.setattr(data, 'subset', lambda items, limit, seed=0: items)
    monkeypatch.setattr(run, 'git_info', lambda: {'sha': 'generation-sha', 'dirty': False})
    _Fake.gold = {it['question']: (it['item_id'], it['answer']) for it in items}
    _Fake.calls, _Fake.slow, _Fake.slow_judge = {}, set(), set()
    srv = ThreadingHTTPServer(('127.0.0.1', 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{srv.server_port}/v1'
    model, grader = f'url:{base}@fake', f'url:{base}@judge'
    original = client.resolve
    active = [model]
    resolved = []

    def resolve(value):
        assert value == active[0], f'inactive endpoint resolved: {value}'
        resolved.append(value)
        return original(value)

    monkeypatch.setattr(client, 'resolve', resolve)
    args = ['--run-id', 'stages', '--runs-root', str(tmp_path), '--repeats', '1', '--limit', '3', '--judge', grader]
    try:
        assert run.main(args + ['--stage', 'generate', '--model', model]) == 0
        root = tmp_path / 'stages'
        summary = json.loads((root / 'summary.json').read_text())
        assert summary['score'] is None and not summary['valid'] and not summary['complete']
        assert summary['n_pending'] == 3 and summary['generation_complete']
        assert all(r['score'] is None for r in run.load_rows(root / 'items.jsonl').values())
        before = {p.name: json.loads(p.read_text())['model'] for p in (root / 'transcripts').glob('*.json')}
        calls = dict(_Fake.calls)
        assert run.main(args + ['--stage', 'generate', '--model', model]) == 0
        assert _Fake.calls == calls
        active[0] = grader
        monkeypatch.setattr(run, 'git_info', lambda: {'sha': 'grading-sha', 'dirty': False})
        assert run.main(args + ['--stage', 'grade']) == 0
        assert _Fake.calls == calls
        after = {p.name: json.loads(p.read_text())['model'] for p in (root / 'transcripts').glob('*.json')}
        assert before == after
        summary = json.loads((root / 'summary.json').read_text())
        assert summary['score'] == 66.67 and summary['valid'] and summary['complete'] and summary['n_pending'] == 0
        cfg = json.loads((root / 'config.json').read_text())
        assert cfg['git']['sha'] == 'generation-sha' and cfg['stage_git']['sha'] == 'grading-sha'
        assert resolved == [model, model, grader]
    finally:
        srv.shutdown()


def test_grade_missing_answer_stays_infra_without_generation(tmp_path, monkeypatch):
    it = {'item_id': 'missing', 'domain': 'Math', 'subtopic': 'synthetic', 'answer': 'x'}
    monkeypatch.setattr(run, 'answer_item', lambda *a: (_ for _ in ()).throw(AssertionError('model contacted')))
    row = run.process((it, 0, None), {'stage': 'grade', 'tdir': tmp_path, 'run_dir': tmp_path, 'answers_from': None})
    assert row['status'] == 'infra_error' and row['stage'] == 'model' and row['score'] is None
