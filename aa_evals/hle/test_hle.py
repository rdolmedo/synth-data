"""Unit tests on synthetic items (no HLE data): python3 -m unittest aa_evals.hle.test_hle"""
import asyncio
import json
import math
import random
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aa_evals.hle import prompts, scoring
from aa_evals.hle.client import ChatClient, InfraError
from aa_evals.hle.run import Runner, parse_args, select

IMG = 'data:image/png;base64,iVBORw0KGgo='


def item(i, answer_type='exactMatch', image='', partition='reasoning'):
    return {'id': f'q{i}', 'question': f'What is {i}+{i}?', 'image': image, 'answer': str(2 * i),
            'answer_type': answer_type, 'category': 'Math', 'raw_subject': 'x', 'partition': partition}


def completion(content, finish='stop', reasoning='thinking...'):
    return {'id': 'cmpl', 'choices': [{'index': 0, 'finish_reason': finish,
                                       'message': {'role': 'assistant', 'content': content,
                                                   'reasoning_content': reasoning}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}}


class Prompts(unittest.TestCase):
    def test_system_prompt_by_type(self):
        self.assertIn('Exact Answer:', prompts.system_prompt(item(1, 'exactMatch')))
        self.assertIn('\nAnswer:', prompts.system_prompt(item(1, 'multipleChoice')))
        # the official two-prompt code compared against 'exact_match', so it sent the MC prompt to everything
        self.assertEqual(prompts.system_prompt(item(1, 'exactMatch'), 'code'), prompts.SYSTEM_MC)
        self.assertEqual(prompts.system_prompt(item(1, 'exactMatch'), 'unified'), prompts.SYSTEM_MC)

    def test_messages_image_after_text(self):
        m = prompts.build_messages(item(1, image=IMG))
        self.assertEqual(m[0]['role'], 'system')
        self.assertEqual([p['type'] for p in m[1]['content']], ['text', 'image_url'])
        self.assertEqual(m[1]['content'][1]['image_url']['url'], IMG)
        t = prompts.build_messages(item(1, image=IMG), text_only=True)
        self.assertEqual([p['type'] for p in t[1]['content']], ['text'])
        u = prompts.build_messages(item(1), system_role='user')
        self.assertEqual(len(u), 1)
        self.assertTrue(u[0]['content'][0]['text'].startswith('Your response should be'))

    def test_judge_prompt_matches_official(self):
        # official string literal is non-raw with "\%", which Python keeps as backslash-percent
        self.assertIn('between 0|\\%| and 100|\\%|', prompts.JUDGE_PROMPT)
        msg = prompts.judge_messages(item(3), 'Answer: 6')[0]['content']
        self.assertIn('[correct_answer]: 6', msg)
        self.assertIn('[response]: Answer: 6', msg)

    def test_parse_judgment(self):
        j = {'extracted_final_answer': '6', 'reasoning': 'r', 'correct': 'yes', 'confidence': 80, 'strict': True}
        self.assertEqual(prompts.parse_judgment(json.dumps(j))['correct'], 'yes')
        self.assertEqual(prompts.parse_judgment('```json\n' + json.dumps(j) + '\n```')['confidence'], 80)
        with self.assertRaises(ValueError):
            prompts.parse_judgment(json.dumps({**j, 'correct': 'maybe'}))
        with self.assertRaises(ValueError):
            prompts.parse_judgment('no json here')
        with self.assertRaises(ValueError):
            prompts.parse_judgment(None)

    def test_confidence_contract_rejects_runaway_integers(self):
        j = {'extracted_final_answer': '6', 'reasoning': 'r', 'correct': 'no', 'confidence': 0}
        for confidence in (0, 60, 95, 100):
            self.assertEqual(prompts.parse_judgment(json.dumps({**j, 'confidence': confidence}))['confidence'], confidence)
        for confidence in (-1, 101, 880, 5000, 9500, 60 * 10 ** 100,
                           True, False, 95.9, 100.99, '95', None, float('inf')):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(ValueError, 'integer in 0..100'):
                prompts.parse_judgment(json.dumps({**j, 'confidence': confidence}))


class Scoring(unittest.TestCase):
    def test_calib_err_matches_official(self):
        import numpy as np
        def official(confidence, correct, beta=100):
            idxs = np.argsort(confidence, kind='stable')
            confidence, correct = confidence[idxs], correct[idxs]
            bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
            bins[-1] = [bins[-1][0], len(confidence)]
            cerr = 0
            for i in range(len(bins) - 1):
                bc, bk = confidence[bins[i][0]:bins[i][1]], correct[bins[i][0]:bins[i][1]]
                cerr += len(bc) / len(confidence) * np.square(np.abs(np.nanmean(bc) - np.nanmean(bk)))
            return np.sqrt(cerr)
        rng = random.Random(0)
        conf = [rng.random() for _ in range(357)]
        corr = [int(rng.random() < c * 0.5) for c in conf]
        self.assertAlmostEqual(scoring.calib_err(conf, corr), float(official(np.array(conf), np.array(corr))), 9)
        self.assertIsNone(scoring.calib_err(conf[:150], corr[:150]))

    def test_pass_at_1_wald(self):
        rows = [{'item_id': f'q{i}', 'score': int(i < 30), 'status': 'ok'} for i in range(100)]
        p = scoring.pass_at_1(rows)
        self.assertEqual(p['score'], 30.0)
        self.assertEqual(p['ci95'], round(1.96 * math.sqrt(30 * 70 / 100), 2))   # HLE's Wald interval

    def test_summary_infra_excluded(self):
        rows = [{'item_id': 'a', 'score': 1, 'status': 'ok', 'partition': 'reasoning', 'has_image': True},
                {'item_id': 'b', 'score': 0, 'status': 'model_error', 'partition': 'knowledge'},
                {'item_id': 'c', 'score': None, 'status': 'infra_error', 'partition': 'knowledge'}]
        s = scoring.summarize(rows, n_items=3, n_repeats=1, dataset='cais/hle-diamond')
        self.assertEqual(s['score'], 50.0)
        self.assertEqual(s['n_infra_errors'], 1)
        self.assertFalse(s['valid'])
        self.assertEqual(s['by_partition']['reasoning']['score'], 100.0)
        self.assertEqual(s['by_modality']['text']['n'], 1)

    def test_model_error_kind(self):
        me = lambda **kw: {'status': 'model_error', **kw}
        self.assertEqual(scoring.model_error_kind(me(finish_reason='cutoff')), 'cutoff')
        self.assertEqual(scoring.model_error_kind(me(finish_reason='length')), 'length')
        self.assertEqual(scoring.model_error_kind(me(finish_reason='stop', reasoning_unclosed=True)),
                         'reasoning_unclosed')
        self.assertEqual(scoring.model_error_kind(me(finish_reason='stop')), 'empty_stop')
        self.assertEqual(scoring.model_error_kind(me(finish_reason='abort')), 'empty_abort')
        self.assertIsNone(scoring.model_error_kind({'status': 'ok', 'finish_reason': 'stop'}))
        s = scoring.summarize([me(item_id='a', score=0, finish_reason='length'), me(item_id='b', score=0,
                               finish_reason='cutoff')], n_items=2, n_repeats=1, dataset='cais/hle-diamond')
        self.assertEqual(s['model_errors_by_kind'], {'cutoff': 1, 'length': 1})

    def test_aa_reference(self):
        if scoring.AA_LEADERBOARD.exists():
            self.assertEqual(scoring.aa_reference('nemotron-3-5-super-vl', 'cais/hle-diamond'), 8.8)

    def test_select_stratified(self):
        from aa_evals.hle.run import stratum
        items = [item(i, answer_type='multipleChoice' if i % 5 == 0 else 'exactMatch',
                      image=IMG if i % 4 == 0 else '', partition='reasoning' if i % 2 else 'knowledge')
                 for i in range(1000)]
        sub = select(items, 100, stratify=True)
        self.assertEqual(len(sub), 100)
        full, got = {}, {}
        for it in items:
            full[stratum(it)] = full.get(stratum(it), 0) + 1
        for it in sub:
            got[stratum(it)] = got.get(stratum(it), 0) + 1
        self.assertEqual(set(got), set(full))   # every stratum covered
        for k in full:
            self.assertLessEqual(abs(got[k] - full[k] / 10), 1)   # proportional
        self.assertEqual([x['id'] for x in sub], [x['id'] for x in select(list(reversed(items)), 100, stratify=True)])

    def test_judge_effort_reaches_served_template(self):
        from aa_evals.hle.client import Endpoint
        from aa_evals.hle.run import judge_settings
        qwen = Endpoint('serving:j', 'serving', 'j', ['http://x'], request={'overrides': {
            'temperature': 1.0, 'top_k': 20, 'reasoning_effort': 'xhigh',
            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'}}})
        s = judge_settings(args(judge_reasoning_effort='medium'), qwen)
        self.assertEqual((s['reasoning_effort'], s['chat_template_kwargs']['reasoning_effort']), ('medium', 'medium'))
        self.assertEqual((s['top_k'], s['max_tokens']), (20, 32768))
        self.assertEqual(judge_settings(args(), qwen)['chat_template_kwargs']['reasoning_effort'], 'xhigh')
        super_ = Endpoint('serving:s', 'serving', 's', ['http://x'], request={'overrides': {
            'temperature': 1.0, 'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}}})
        s = judge_settings(args(), super_)   # registry defaults untouched
        self.assertNotIn('reasoning_effort', s)
        self.assertEqual(s['chat_template_kwargs'], {'enable_thinking': True, 'medium_effort': False})
        oa = Endpoint('openai:g', 'openai', 'g', ['http://x'])
        self.assertEqual(judge_settings(args(), oa)['reasoning_effort'], 'medium')

    def test_adaptive_limiter(self):
        from aa_evals.hle.run import AdaptiveLimiter
        lim = AdaptiveLimiter(80, 'http://x/metrics', start=48)
        self.assertEqual(lim.limit, 48)
        lim.inflight = 48
        lim.adjust(0, 0.5, 0)              # room and demand: up by 4
        self.assertEqual(lim.limit, 52)
        lim.adjust(0, 0.9, 0)              # KV above target: hold
        self.assertEqual(lim.limit, 52)
        lim.adjust(10, 0.99, 0)            # vLLM queues: down to what it runs
        self.assertEqual(lim.limit, 38)
        lim.adjust(10, 0.99, 0)            # still draining (48 in flight > 38): hold, don't cut twice
        self.assertEqual(lim.limit, 38)
        lim.inflight = 38
        lim.adjust(0, 0.95, 3)             # preempted, no queue: down by 2
        self.assertEqual(lim.limit, 36)
        lim.inflight = 36
        lim.adjust(500, 1.0, 3)
        self.assertEqual(lim.limit, 8)     # floor
        self.assertEqual(AdaptiveLimiter(16).limit, 16)   # no metrics: a plain cap
        m = AdaptiveLimiter.parse('# HELP x\nvllm:num_requests_waiting{engine="0"} 3.0\n'
                                  'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 3.0\n'
                                  'vllm:kv_cache_usage_perc{engine="0"} 0.5\nvllm:num_preemptions_total{e="0"} 7.0\n')
        self.assertEqual(m, {'num_requests_waiting': 3.0, 'kv_cache_usage_perc': 0.5, 'num_preemptions_total': 7.0})
        async def gate():
            lim = AdaptiveLimiter(2)
            order = []
            async def job(i):
                async with lim:
                    order.append(('in', i, lim.inflight))
                    await asyncio.sleep(0.01)
            await asyncio.gather(*(job(i) for i in range(5)))
            return max(n for _, _, n in order)
        self.assertEqual(asyncio.run(gate()), 2)

    def test_select_deterministic(self):
        items = [item(i) for i in range(50)]
        a = select(items, 10)
        b = select(list(reversed(items)), 10)
        self.assertEqual([x['id'] for x in a], [x['id'] for x in b])


class FakeServer:
    """A MockTransport handler: the model answers by item, the judge says yes iff the answer contains '6'."""

    def __init__(self, model_plan):
        self.model_plan = model_plan      # question text -> list of responses (dict body or int status)
        self.calls = {}

    def __call__(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith('/tokenize'):
            return httpx.Response(200, json={'count': len(body['prompt'].split())})
        self.bodies = getattr(self, 'bodies', []) + [body]
        if body['model'] == 'judge':
            text = body['messages'][0]['content']
            resp = text.split('[response]: ', 1)[1].split('\n\nYour judgement', 1)[0]
            ok = 'yes' if '6' in resp else 'no'
            j = {'extracted_final_answer': resp, 'reasoning': 'r', 'correct': ok, 'confidence': 90, 'strict': True}
            return httpx.Response(200, json=completion(json.dumps(j)))
        q = body['messages'][-1]['content'][0]['text']
        n = self.calls[q] = self.calls.get(q, 0) + 1
        plan = self.model_plan[q]
        r = plan[min(n, len(plan)) - 1]
        return httpx.Response(r, text='boom') if isinstance(r, int) else httpx.Response(200, json=r)


def args(**kw):
    a = parse_args(['--model', 'url:http://fake/v1@m', '--run-id', 't', '--judge', 'url:http://fake/v1@judge'])
    for k, v in {'concurrency': 4, 'judge_concurrency': 4, 'progress_every': 3600, **kw}.items():
        setattr(a, k, v)
    return a


def run_once(server, items, run_dir, a, jset=None):
    t = httpx.MockTransport(server)
    model = ChatClient('url:http://fake/v1@m', transport=t, max_attempts=3, backoff_cap=0)
    judge = ChatClient('url:http://fake/v1@judge', transport=t, max_attempts=3, backoff_cap=0)
    (run_dir / 'transcripts').mkdir(parents=True, exist_ok=True)
    runner = Runner(a, items, model, judge, {'temperature': 1.0}, jset or {'max_tokens': 100}, run_dir)
    return asyncio.run(runner.run([(it['id'], 0) for it in items]))


class EndToEnd(unittest.TestCase):
    def test_resume_regrades_only_out_of_range_confidence_without_generation(self):
        items = [item(3), item(4)]
        server = FakeServer({it['question']: [completion('Answer: 6')] for it in items})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            run_once(server, items, path, args())
            old = {p.name: p.read_bytes() for p in (path / 'transcripts').glob('*.json')}
            affected = path / 'transcripts/q3.r0.json'
            record = json.loads(affected.read_text())
            generation = record['generation']
            record['judgment']['confidence'] = 9500
            affected.write_text(json.dumps(record))
            server.bodies = []
            model_calls = dict(server.calls)
            rows = run_once(server, items, path, args(stage='grade'))
            self.assertEqual(server.calls, model_calls)
            self.assertEqual(len(server.bodies), 1)
            self.assertEqual(server.bodies[0]['model'], 'judge')
            self.assertEqual(json.loads(affected.read_text())['generation'], generation)
            self.assertEqual(json.loads(affected.read_text())['judgment']['confidence'], 90)
            self.assertEqual((path / 'transcripts/q4.r0.json').read_bytes(), old['q4.r0.json'])
            self.assertEqual(len(rows), 2)

    def test_statuses_and_resume(self):
        items = [item(3), item(4, image=IMG, partition='knowledge'), item(5), item(7)]
        q = {it['id']: it['question'] for it in items}
        server = FakeServer({
            q['q3']: [503, completion('Explanation: x\nExact Answer: 6\nConfidence: 90%')],   # retried, correct
            q['q4']: [completion('Answer: 9')],                                            # wrong
            q['q5']: [completion(None, finish='length')],                                  # no final answer
            q['q7']: [500, 500, 500, completion('Answer: 6')],                             # infra, then fixed
        })
        with tempfile.TemporaryDirectory() as d:
            rows = {r['item_id']: r for r in run_once(server, items, Path(d), args())}
            self.assertEqual((rows['q3']['status'], rows['q3']['score'], rows['q3']['model_attempts']), ('ok', 1, 2))
            self.assertEqual((rows['q4']['status'], rows['q4']['score']), ('ok', 0))
            self.assertEqual((rows['q5']['status'], rows['q5']['score']), ('model_error', 0))
            self.assertEqual((rows['q7']['status'], rows['q7']['score'], rows['q7']['error_stage']),
                             ('infra_error', None, 'generation'))
            # resume: finished items are not re-asked; the infra item is retried and now succeeds
            before = dict(server.calls)
            rows = {r['item_id']: r for r in run_once(server, items, Path(d), args())}
            self.assertEqual(server.calls[q['q3']], before[q['q3']])
            self.assertEqual((rows['q7']['status'], rows['q7']['score']), ('ok', 1))

    def test_context_window_caps_max_tokens_per_item(self):
        from aa_evals.hle import run as R
        items = [item(3), item(4, image=IMG)]
        q = {it['id']: it['question'] for it in items}
        server = FakeServer({q['q3']: [completion('Answer: 6')], q['q4']: [completion('Answer: 6')]})
        with tempfile.TemporaryDirectory() as d:
            run_once(server, items, Path(d), args(window=131072))
            mt = {b['messages'][-1]['content'][0]['text']: b['max_tokens'] for b in server.bodies if b['model'] == 'm'}
            words = lambda it: len((prompts.system_prompt(it, 'by_type') + '\n' + it['question']).split())
            self.assertEqual(mt[q['q3']], 131072 - words(items[0]) - R.TEMPLATE_SLACK)
            self.assertEqual(mt[q['q4']], 131072 - words(items[1]) - R.TEMPLATE_SLACK - R.IMAGE_ALLOWANCE)
        server = FakeServer({q['q3']: [completion('Answer: 6')], q['q4']: [completion('Answer: 6')]})
        with tempfile.TemporaryDirectory() as d:   # no window (endpoint context <= cap, or calibration): unset
            run_once(server, items, Path(d), args(window=None))
            self.assertTrue(all('max_tokens' not in b for b in server.bodies if b['model'] == 'm'))

    def test_effort_sets_template_kwarg_and_refuses_models_without_levels(self):
        from aa_evals.hle.client import Endpoint
        from aa_evals.hle.run import generation_settings
        qwen = Endpoint('serving:q', 'serving', 'qwen38-27b', ['http://x'], request={'overrides': {
            'temperature': 1.0, 'reasoning_effort': 'xhigh',
            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'}}})
        g = generation_settings(args(reasoning_effort='medium'), qwen)
        self.assertEqual((g['reasoning_effort'], g['chat_template_kwargs']['reasoning_effort']), ('medium', 'medium'))
        self.assertEqual(qwen.request['overrides']['reasoning_effort'], 'xhigh')   # registry left untouched
        gemma = Endpoint('serving:g', 'serving', 'gemma', ['http://x'], request={'overrides': {
            'temperature': 1.0, 'chat_template_kwargs': {'enable_thinking': True}}})
        with self.assertRaises(SystemExit):
            generation_settings(args(reasoning_effort='medium'), gemma)

    def test_rejudge_from_other_run(self):
        from aa_evals.hle.run import seed_generations
        items = [item(3), item(4)]
        q = {it['id']: it['question'] for it in items}
        server = FakeServer({q['q3']: [completion('Answer: 6')], q['q4']: [500]})
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            run_once(server, items, Path(d1), args())
            (Path(d2) / 'transcripts').mkdir()
            self.assertEqual(seed_generations(d1, d2, [('q3', 0), ('q4', 0)]), 1)   # infra-failed q4 not copied
            calls = dict(server.calls)
            rows = {r['item_id']: r for r in run_once(server, items, Path(d2), args(generations_from='src'))}
            self.assertEqual(server.calls, calls)   # a re-judge never calls the model
            self.assertEqual((rows['q3']['status'], rows['q3']['score']), ('ok', 1))
            self.assertEqual(rows['q4']['status'], 'infra_error')   # resume the source run to fix it
            self.assertFalse((Path(d2) / 'transcripts' / 'q4.r0.json').exists())

    def test_rejudge_on_judge_change(self):
        items = [item(3)]
        server = FakeServer({items[0]['question']: [completion('Answer: 6')]})
        judge_calls = []
        def handler(request):
            body = json.loads(request.content)
            if body['model'] == 'judge':
                judge_calls.append(body)
            return server(request)
        with tempfile.TemporaryDirectory() as d:
            run_once(handler, items, Path(d), args())
            run_once(handler, items, Path(d), args(rejudge=True))   # same judge: kept
            self.assertEqual(len(judge_calls), 1)
            rows = run_once(handler, items, Path(d), args(rejudge=True), jset={'max_tokens': 200})
            self.assertEqual(len(judge_calls), 2)   # re-graded with the new settings
            self.assertEqual(judge_calls[-1]['max_tokens'], 200)
            self.assertEqual(server.calls[items[0]['question']], 1)   # the model is never re-asked
            self.assertEqual(rows[0]['score'], 1)

    def test_merge_priority_never_looks_at_score(self):
        from aa_evals.hle import merge
        cfg = {'model': {'served_name': 'm', 'spec': 'url:a'}, 'generation': {'temperature': 1.0}, 'prompt': {},
               'dataset': {'id': 'cais/hle-diamond', 'revision': 'r'}, 'judge': {'spec': 'j', 'settings': {},
               'served_name': 'j'}, 'n_repeats': 1, 'aa_slug': None}
        def row(i, status, score, finish='stop', d=None):
            t = Path(d) / 'transcripts' / f'q{i}.r0.json'
            t.write_text('{}')
            return {'item_id': f'q{i}', 'repeat': 0, 'status': status, 'score': score, 'finish_reason': finish,
                    'transcript': str(t), 'partition': 'reasoning', 'has_image': False}
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2, \
                tempfile.TemporaryDirectory() as out:
            for d, rows in ((d1, lambda: [row(1, 'ok', 0, d=d1), row(2, 'model_error', 0, 'cutoff', d1),
                                          row(3, 'infra_error', None, d=d1), row(4, 'model_error', 0, 'cutoff', d1)]),
                            (d2, lambda: [row(1, 'ok', 1, d=d2), row(2, 'ok', 1, d=d2), row(3, 'model_error', 0, d=d2),
                                          row(4, 'model_error', 0, 'cutoff', d2)])):
                (Path(d) / 'transcripts').mkdir()
                (Path(d) / 'config.json').write_text(json.dumps(cfg))
                (Path(d) / 'items.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows()))
            s = merge.merge([d1, d2], Path(out) / 'm')
            got = {r['item_id']: (r['source_run'] == d1, r['score'], r['finish_reason'])
                   for r in map(json.loads, (Path(out) / 'm' / 'items.jsonl').read_text().splitlines())}
            self.assertEqual(got['q1'], (True, 0, 'stop'))     # the first source's finished row wins, even if wrong
            self.assertEqual(got['q2'], (False, 1, 'stop'))    # cut off in the first, finished in the second
            self.assertEqual(got['q3'], (False, 0, 'stop'))    # infra in the first
            self.assertEqual(got['q4'][2], 'cutoff')           # nobody finished: a cutoff 0
            self.assertEqual((s['n_cutoff'], s['n_items']), (1, 4))
            self.assertTrue((Path(out) / 'm' / 'transcripts' / 'q2.r0.json').exists())

    def test_cutoff_scores_unfinished_zero(self):
        items = [item(3), item(4)]
        q = {it['id']: it['question'] for it in items}
        async def handler(request):
            body = json.loads(request.content)
            if body['model'] != 'judge' and body['messages'][-1]['content'][0]['text'] == q['q4']:
                await asyncio.sleep(30)   # the tail
            return FakeServer({q['q3']: [completion('Answer: 6')], q['q4']: [completion('Answer: 8')]})(request)
        with tempfile.TemporaryDirectory() as d:
            t0 = time.time()
            rows = {r['item_id']: r for r in run_once(handler, items, Path(d), args(cutoff_min=0.02))}
            self.assertLess(time.time() - t0, 10)
            self.assertEqual((rows['q3']['status'], rows['q3']['score']), ('ok', 1))
            self.assertEqual((rows['q4']['status'], rows['q4']['score'], rows['q4']['finish_reason']),
                             ('model_error', 0, 'cutoff'))
            self.assertFalse((Path(d) / 'transcripts' / 'q4.r0.json').exists())   # regenerated on resume

    def test_bad_request_is_infra_not_model(self):
        items = [item(3)]
        server = FakeServer({items[0]['question']: [400]})
        with tempfile.TemporaryDirectory() as d:
            rows = run_once(server, items, Path(d), args())
        self.assertEqual(rows[0]['status'], 'infra_error')
        self.assertEqual(server.calls[items[0]['question']], 3)   # max_4xx

    def test_judge_off_schema_is_infra(self):
        items = [item(3)]
        def handler(request):
            body = json.loads(request.content)
            if body['model'] == 'judge':
                return httpx.Response(200, json=completion('not json'))
            return httpx.Response(200, json=completion('Answer: 6'))
        with tempfile.TemporaryDirectory() as d:
            rows = run_once(handler, items, Path(d), args(judge_format_attempts=2))
        self.assertEqual((rows[0]['status'], rows[0]['error_stage']), ('infra_error', 'judge'))


class Client(unittest.TestCase):
    def test_gives_up(self):
        c = ChatClient('url:http://fake/v1@m', transport=httpx.MockTransport(lambda r: httpx.Response(502)),
                       max_attempts=4, backoff_cap=0)
        with self.assertRaises(InfraError) as e:
            asyncio.run(c.chat({'messages': []}))
        self.assertEqual(len(e.exception.attempts), 4)

    def test_gate_backpressure_uses_no_attempt(self):
        n = {'calls': 0}
        def handler(request):
            n['calls'] += 1
            if n['calls'] <= 6:   # more gate rejections than max_attempts
                return httpx.Response(503, headers={'Retry-After': '0', 'X-Gate': '1'},
                                      json={'error': {'message': 'gate: server at capacity for new conversations',
                                                      'type': 'server_overloaded', 'code': 503}})
            return httpx.Response(200, json=completion('Answer: 6'))
        c = ChatClient('url:http://fake/v1@m', transport=httpx.MockTransport(handler), max_attempts=3, backoff_cap=0)
        c_sleep = asyncio.sleep
        async def fast(_):
            await c_sleep(0)
        import aa_evals.hle.client as cl
        old, cl.asyncio.sleep = cl.asyncio.sleep, fast
        try:
            data, attempts = asyncio.run(c.chat({'messages': []}))
        finally:
            cl.asyncio.sleep = old
        self.assertEqual(n['calls'], 7)
        self.assertEqual(attempts[-1]['gate_rejections'], 6)
        # a plain 503 (no gate) still counts as an attempt
        c2 = ChatClient('url:http://fake/v1@m', transport=httpx.MockTransport(lambda r: httpx.Response(503)),
                        max_attempts=3, backoff_cap=0)
        with self.assertRaises(InfraError) as e:
            asyncio.run(c2.chat({'messages': []}))
        self.assertEqual(len(e.exception.attempts), 3)


if __name__ == '__main__':
    unittest.main()


def test_main_generate_then_grade_without_live_other_server(tmp_path, monkeypatch):
    """Separate stages keep provenance and never create/resolve the inactive server's client."""
    from aa_evals.hle import run as R
    items = [item(3), item(4), item(5)]
    (tmp_path / 'manifest.json').write_text(json.dumps({'revision': 'synthetic', 'sha256': 'synthetic'}))
    monkeypatch.setattr(R, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(R.data, 'load', lambda dataset: items)
    monkeypatch.setattr(R.data, 'local_dir', lambda dataset: tmp_path)
    monkeypatch.setattr(R, 'endpoint_context', lambda ep: 131072)
    monkeypatch.setattr(R.protocol, 'image_inputs', lambda spec: (True, 'synthetic'))
    monkeypatch.setattr(R, 'git_state', lambda: {'sha': 'generation-sha', 'dirty': False})
    server = FakeServer({it['question']: [completion(None, finish='length') if it['id'] == 'q5' else completion('Answer: 6')] for it in items})
    real_resolve = R.resolve_now
    calls = []
    active = ['m']

    def resolve(spec, wait):
        assert spec.endswith('@' + active[0]), f'inactive endpoint resolved: {spec}'
        calls.append(('resolve', active[0]))
        return real_resolve(spec, wait)

    def client(spec, **kw):
        assert spec.endswith('@' + active[0]), f'inactive client constructed: {spec}'
        calls.append(('client', active[0]))
        return ChatClient(spec, transport=httpx.MockTransport(server), max_attempts=1, backoff_cap=0)

    monkeypatch.setattr(R, 'resolve_now', resolve)
    monkeypatch.setattr(R, 'ChatClient', client)
    argv = ['--model', 'url:http://offline-model/v1@m', '--judge', 'url:http://offline-judge/v1@judge',
            '--run-id', 'stages', '--limit', '3', '--repeats', '1']
    assert R.main(argv + ['--stage', 'generate']) == 0
    root = tmp_path / 'runs' / 'stages'
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['score'] is None and not summary['valid'] and not summary['complete']
    assert summary['n_pending'] == 3 and summary['generation_complete']
    assert all(json.loads(l)['score'] is None for l in (root / 'items.jsonl').read_text().splitlines())
    before = {p.name: json.loads(p.read_text())['generation'] for p in (root / 'transcripts').glob('*.json')}
    assert all(b['model'] == 'm' for b in server.bodies)
    model_calls = dict(server.calls)
    assert R.main(argv + ['--stage', 'generate']) == 0
    assert server.calls == model_calls
    active[0] = 'judge'
    monkeypatch.setattr(R, 'git_state', lambda: {'sha': 'grading-sha', 'dirty': False})
    assert R.main(['--run-id', 'stages', '--stage', 'grade', '--judge', 'url:http://offline-judge/v1@judge']) == 0
    assert server.calls == model_calls
    after = {p.name: json.loads(p.read_text())['generation'] for p in (root / 'transcripts').glob('*.json')}
    assert after == before
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['score'] == 66.67 and summary['valid'] and summary['complete'] and not summary['n_pending']
    cfg = json.loads((root / 'config.json').read_text())
    assert cfg['git']['sha'] == 'generation-sha' and cfg['stage_git']['sha'] == 'grading-sha'
    assert calls.count(('client', 'm')) == 2 and calls.count(('client', 'judge')) == 1


def test_grade_with_missing_generation_does_not_regenerate(tmp_path):
    rows = run_once(FakeServer({}), [item(3)], tmp_path, args(stage='grade'))
    assert rows[0]['status'] == 'infra_error' and rows[0]['score'] is None


def _offline_main(tmp_path, monkeypatch, items):
    """R.main against a MockTransport model 'm' and judge 'judge' on synthetic items (no data, no servers)."""
    from aa_evals.hle import run as R
    (tmp_path / 'manifest.json').write_text(json.dumps({'revision': 'synthetic', 'sha256': 'synthetic'}))
    monkeypatch.setattr(R, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(R, 'run_path', lambda run_id: tmp_path / 'runs' / run_id)
    monkeypatch.setattr(R.data, 'load', lambda dataset: items)
    monkeypatch.setattr(R.data, 'local_dir', lambda dataset: tmp_path)
    monkeypatch.setattr(R, 'endpoint_context', lambda ep: 131072)
    monkeypatch.setattr(R.protocol, 'image_inputs', lambda spec: (False, 'synthetic text-only'))
    monkeypatch.setattr(R, 'git_state', lambda: {'sha': 'sha', 'dirty': False})
    server = FakeServer({it['question']: [completion('Answer: 6' if it['id'] != 'q4' else 'Answer: 9')]
                         for it in items})
    monkeypatch.setattr(R, 'ChatClient', lambda spec, **kw: ChatClient(
        spec, transport=httpx.MockTransport(server), max_attempts=1, backoff_cap=0))
    base = ['--model', 'url:http://offline/v1@m', '--judge', 'url:http://offline/v1@judge', '--repeats', '1']
    return R, server, base


def test_task_set_default_drops_image_questions(tmp_path, monkeypatch):
    from aa_evals.hle import data
    items = [item(3), item(4, image=IMG), item(5), item(6, image=IMG)]
    assert [r['id'] for r in data.task_set(items, 'text-only-v1')] == ['q3', 'q5']
    assert data.task_set(items, 'all') == items
    R, server, base = _offline_main(tmp_path, monkeypatch, items)
    assert R.main(base + ['--run-id', 'new']) == 0
    cfg = json.loads((tmp_path / 'runs/new/config.json').read_text())
    summary = json.loads((tmp_path / 'runs/new/summary.json').read_text())
    assert sorted(cfg['item_ids']) == ['q3', 'q5'] and cfg['dataset']['task_set'] == 'text-only-v1'
    assert cfg['dataset']['n_task_set'] == 2 and summary['task_set'] == 'text-only-v1' and summary['n_items'] == 2
    assert R.main(base + ['--run-id', 'full', '--task-set', 'all']) == 0
    cfg = json.loads((tmp_path / 'runs/full/config.json').read_text())
    assert len(cfg['item_ids']) == 4 and cfg['dataset']['task_set'] == 'all'
    # an image id is refused under the default set, with the hint
    (tmp_path / 'ids').write_text('q3\nq4\n')
    with pytest.raises(SystemExit, match='--task-set all'):
        R.main(base + ['--run-id', 'ids', '--ids', str(tmp_path / 'ids')])
    assert R.main(base + ['--run-id', 'ids', '--ids', str(tmp_path / 'ids'), '--task-set', 'all']) == 0


def test_pre_348_run_resumes_and_grades_on_all(tmp_path, monkeypatch):
    items = [item(3), item(4, image=IMG), item(5)]
    R, server, base = _offline_main(tmp_path, monkeypatch, items)
    assert R.main(base + ['--run-id', 'old', '--task-set', 'all', '--stage', 'generate']) == 0
    cfg_path = tmp_path / 'runs/old/config.json'
    cfg = json.loads(cfg_path.read_text())
    del cfg['dataset']['task_set']   # as written before #348
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match='pass --task-set all'):
        R.main(base + ['--run-id', 'old', '--stage', 'generate'])
    # grading a pre-#348 run keeps all its items, with no --task-set
    assert R.main(['--run-id', 'old', '--stage', 'grade', '--judge', 'url:http://offline/v1@judge']) == 0
    summary = json.loads((tmp_path / 'runs/old/summary.json').read_text())
    assert summary['n_items'] == 3 and summary['task_set'] == 'all' and summary['score'] == 66.67


def test_rescore_restricts_stored_rows(tmp_path, monkeypatch):
    from aa_evals.hle import rescore
    items = [item(3), item(4, image=IMG), item(5), item(6, image=IMG)]
    R, server, base = _offline_main(tmp_path, monkeypatch, items)
    assert R.main(base + ['--run-id', 'full', '--task-set', 'all']) == 0
    run_dir = tmp_path / 'runs/full'
    before = {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()}
    s, stored = rescore.rescore(run_dir, 'text-only-v1', items)
    assert stored['score'] == 75.0 and stored['n_items'] == 4   # q4 wrong
    assert s['score'] == 100.0 and s['n_items'] == 2 and s['valid'] and s['task_set'] == 'text-only-v1'
    assert s['source_score'] == 75.0 and s['delta_vs_source'] == 25.0
    same, _ = rescore.rescore(run_dir, 'all', items)
    assert same['score'] == stored['score'] and same['ci95'] == stored['ci95']
    assert {p.name: p.read_bytes() for p in run_dir.iterdir() if p.is_file()} == before   # run dir untouched


def test_per_item_window_when_capped():
    """m5l (2026-11-15): Snowball SFT servers have max_model_len == the 131,072 window, so no per-item max_tokens was
    set and the capped budget's 130,850 cap made vLLM reject every prompt over 222 tokens. Under a capped budget the
    item's room is always fitted; uncapped runs keep the endpoint's own bound."""
    from aa_evals.hle.run import per_item_window
    same = {'window': 131072, 'endpoint_max_model_len': 131072}
    big = {'window': 131072, 'endpoint_max_model_len': 1048576}
    assert per_item_window('generate', None, same, capped=False) is None
    assert per_item_window('generate', None, same, capped=True) == 131072
    assert per_item_window('generate', None, big, capped=False) == 131072
    assert per_item_window('all', None, big, capped=True) == 131072
    assert per_item_window('grade', None, same, capped=True) is None
    assert per_item_window('generate', 'src-run', big, capped=True) is None
    assert per_item_window('generate', None, {'window': None}, capped=True) is None


def test_response_budget_on_reads_env(monkeypatch):
    from aa_evals.hle.run import response_budget_on
    monkeypatch.delenv('PI_RESPONSE_BUDGET', raising=False)
    assert not response_budget_on()
    monkeypatch.setenv('PI_RESPONSE_BUDGET', '{"warn_after_response_tokens":130850,"finish_response_tokens":32000,'
                       '"total_response_tokens":162850,"token_counting":"super","near_window":true}')
    assert response_budget_on()['near_window']
