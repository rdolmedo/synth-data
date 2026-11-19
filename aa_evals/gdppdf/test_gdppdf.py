"""Offline tests (no endpoint, no judge, no PDFs): python -m aa_evals.gdppdf.test_gdppdf"""
import io
import json
from decimal import Decimal
from pathlib import Path

from aa_evals.gdppdf import inputs, metrics
from aa_evals.gdppdf.data import Criterion, Task
from aa_evals.gdppdf.judge import ParseError, judge_prompt, parse_verdict, passes


def test_parse_verdict():
    assert parse_verdict('{"score": "1", "rationale": "ok"}') == (Decimal('1'), 'ok')
    assert parse_verdict('```json\n{"rationale": "x", "score": 0}\n```')[0] == Decimal('0')
    for bad in ['', 'nope', '[1]', '{"score": "1"}', '{"score": true, "rationale": ""}', '{"score": "a", "rationale": ""}',
                '{"score": "1", "rationale": 3}']:
        try:
            parse_verdict(bad)
        except ParseError:
            continue
        raise AssertionError(f'parsed {bad!r}')
    assert passes(Decimal('1')) and passes(Decimal('0.99995')) and not passes(Decimal('0.9999')) and not passes(Decimal('0'))


def test_judge_prompt():
    p = judge_prompt('TASK?', 'ANSWER', 'CRIT')
    assert 'prompt: TASK?,' in p and 'response: ANSWER,' in p and 'criteria: CRIT,' in p
    assert p.index('TASK?') < p.index('ANSWER') < p.index('CRIT') and '{{' not in p


def test_nemotron_rule():
    # values checked against vLLM 0.32's DynamicResolutionImageTiler (+2 image delimiters; /tokenize on aa-super
    # counts 2,082 for one 1275x1650 image)
    for (w, h), v in {(1275, 1650): 2080, (612, 792): 500, (2562, 1714): 3290, (300, 200): 280}.items():
        assert inputs.nemotron_vl_tokens(w, h) == v + 2, (w, h)


def test_composite_factor():
    assert inputs.composite_factor(46, 256) == (1, 46)
    assert inputs.composite_factor(100, 64) == (2, 100)
    assert inputs.composite_factor(200, 64) == (4, 200)
    assert inputs.composite_factor(300, 64) == (4, 256)


class FakeRenderer:
    """Blank letter pages at the requested DPI."""

    def render(self, pdf, dpi, n):
        from PIL import Image
        out = []
        for p in range(1, n + 1):
            buf = io.BytesIO()
            Image.new('RGB', (round(8.5 * dpi), round(11 * dpi)), (255, 255, 255)).save(buf, format='PNG')
            out.append(inputs.Page(num=p, opaque=True, png=buf.getvalue()))
        return out


def _task(n_pages):
    return Task(task_id='t', domain='D', prompt='What is X?', pdf_path=Path('/nonexistent/doc.pdf'),
                criteria=[Criterion('r1', 'mentions X')]), [f'text of page {i}' for i in range(1, n_pages + 1)]


def test_ladder():
    r = FakeRenderer()
    task, pages = _task(10)
    # roomy: 150 DPI, one image per page
    plan, imgs, text = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 256), 1000, r)
    assert (plan.mode, plan.dpi, plan.k, plan.n_images, plan.covered) == ('images', 150, 1, 10, 10)
    assert plan.reason == 'fits' and 'image i is page i' in text and text.index('What is X?') < text.index('--- Page 1 ---')
    # 10 pages x 2082 tokens = 20.8k at 150 DPI; a 25k budget after output must step down
    plan, _, _ = inputs.plan_and_build(task, pages, inputs.Limits(1000 + 512 + 65536 + 15000, 65536, 256), 1000, r)
    assert plan.mode == 'images' and plan.dpi < 150, plan.describe()
    # image cap 4: 4-up composites (10 pages -> 3 images), labelled
    plan, imgs, text = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 4), 1000, r)
    assert (plan.k, plan.n_images, plan.covered) == (4, 3, 10) and 'combined 4 consecutive pages' in text
    # image cap 2: covers only 8 leading pages
    plan, imgs, text = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 2), 1000, r)
    assert (plan.k, plan.n_images, plan.covered) == (4, 2, 8) and 'pages 9-10 are available only as extracted text' in text
    # nothing fits but text: text only
    plan, imgs, text = inputs.plan_and_build(task, pages, inputs.Limits(1000 + 512 + 65536 + 100, 65536, 256), 1000, r)
    assert plan.mode == 'text' and not imgs and plan.reason == 'images do not fit'
    # text-only model
    plan, imgs, _ = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 0), 1000, r)
    assert plan.mode == 'text' and plan.reason == 'text-only model'
    # even text too long
    plan, _, _ = inputs.plan_and_build(task, pages, inputs.Limits(60000, 65536, 256), 1000, r)
    assert plan.reason == 'input_too_long'
    # payload cap forces JPEG
    # vLLM's decoded-media cap (W*H*4 per image): 10 pages = 84 MB at 150 DPI, 54 MB at 120, 34 MB at 96
    plan, _, _ = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 256, max_decoded_bytes=50_000_000), 1000, r)
    assert plan.mode == 'images' and plan.dpi == 96, plan.describe()
    # per-image template overhead counts once there is more than one image
    plan, _, _ = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 256), 1000, r)
    note_tokens = len(inputs.coverage_note(plan, 10, 'doc.pdf')) // 3 + 2
    assert plan.est_prompt_tokens == 1000 + note_tokens + 10 * (inputs.nemotron_vl_tokens(1275, 1650) + 8) + 512
    # payload cap: JPEG is tried at each DPI before stepping down (blank 72-DPI PNGs then fit)
    plan, imgs, _ = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 256, max_payload_mb=0.05), 1000, r)
    assert plan.tried[0]['jpeg'] and plan.dpi < 150 and plan.payload_mb <= 0.05, plan.describe()


def test_message_content():
    r = FakeRenderer()
    task, pages = _task(2)
    plan, imgs, text = inputs.plan_and_build(task, pages, inputs.Limits(1_000_000, 65536, 256), 100, r)
    parts = inputs.message_content(imgs, text)
    assert [p['type'] for p in parts] == ['image_url', 'image_url', 'text']
    assert parts[0]['image_url']['url'].startswith('data:image/png;base64,')


def test_metrics():
    rows = []
    for t, dom, scores in [('a', 'X', [1, 1]), ('b', 'X', [0, 1]), ('c', 'Y', [0, 0])]:
        for r, s in enumerate(scores):
            rows.append({'item_id': t, 'domain': dom, 'status': 'ok', 'score': float(s), 'mean_pass': 0.5 + s / 2})
    rows.append({'item_id': 'd', 'domain': 'Y', 'status': 'infra_error', 'score': None, 'mean_pass': None})
    rows.append({'item_id': 'd', 'domain': 'Y', 'status': 'model_error', 'score': 0.0, 'mean_pass': 0.0})
    s = metrics.summarize(rows, 4, 2)
    assert s['n_infra_errors'] == 1 and s['n_scored'] == 7 and s['n_model_errors'] == 1
    assert abs(s['score'] - 100 * 3 / 7) < 0.01
    assert s['mean_pass_by_domain']['X']['n_tasks'] == 2 and s['valid'] is False
    assert json.dumps(s)
    rows.append({'item_id': 'e', 'domain': 'Y', 'status': 'cutoff', 'score': 0.0, 'mean_pass': 0.0})
    s = metrics.summarize(rows, 5, 2)
    assert s['n_cutoff'] == 1 and s['n_scored'] == 8 and abs(s['score'] - 100 * 3 / 8) < 0.01
    rows.append({'item_id': 'e', 'domain': 'Y', 'status': 'ungraded', 'score': 0.0, 'mean_pass': 0.0})
    s = metrics.summarize(rows, 5, 2)
    assert s['n_ungraded'] == 1 and s['n_scored'] == 9 and s['n_missing'] == 0


def test_pending_judgments_are_not_infrastructure():
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    from aa_evals.gdppdf import run as R

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task, _ = _task(1)
        args = SimpleNamespace(repeats=1, cutoff=False, model='synthetic', judge='synthetic', run_id='test')
        cfg = {'generation': {'body': {}}, 'selection': {'task_ids': [task.task_id]},
               'context': {'window': 131072}, 'judge': {'arm': 'synthetic'}, 'git_sha': 'frozen'}
        generation = {'task_id': task.task_id, 'repeat': 0, 'status': 'ok', 'at': 'answer-time',
                      'answer': 'answer', 'finish_reason': 'stop', 'usage': {}, 'plan': {}}
        (root / 'generations.jsonl').write_text(json.dumps(generation) + '\n')
        grade = {'task_id': task.task_id, 'repeat': 0, 'judge_key': 'current', 'answer_at': 'answer-time',
                 'n_unscored': 0, 'n_passed': 1, 'n_criteria': 1, 'all_pass': 1., 'mean_pass': 1., 'grades': []}
        with patch.object(R, 'current_judge_key', return_value='current'), patch.object(R, 'aa_reference', return_value=None):
            pending = R.stage_summarize(args, root, [task], cfg)
            assert pending['n_infra'] == 0 and pending['n_pending_judgments'] == 1
            assert pending['generation_complete'] and not pending['complete'] and not pending['valid']
            assert pending['score'] is None
            assert json.loads((root / 'items.jsonl').read_text())['status'] == 'pending'
            for mismatch in ({'judge_key': 'old'}, {'answer_at': 'old-answer'}):
                (root / 'grades.jsonl').write_text(json.dumps({**grade, **mismatch}) + '\n')
                result = R.stage_summarize(args, root, [task], cfg)
                assert result['n_pending_judgments'] == 1 and result['n_infra'] == 0
            (root / 'grades.jsonl').write_text(json.dumps(grade) + '\n')
            complete = R.stage_summarize(args, root, [task], cfg)
            assert complete['score'] == 100 and complete['complete'] and complete['valid']
            assert complete['n_pending_judgments'] == complete['n_infra'] == 0
            (root / 'grades.jsonl').write_text(json.dumps({**grade, 'n_unscored': 1}) + '\n')
            failed = R.stage_summarize(args, root, [task], cfg)
            assert failed['n_infra'] == 1 and failed['n_pending_judgments'] == 0 and not failed['valid']
            args.cutoff = True
            (root / 'grades.jsonl').unlink()
            deadline = R.stage_summarize(args, root, [task], cfg)
            assert deadline['n_ungraded'] == 1 and deadline['score'] == 0 and deadline['n_infra'] == 0


def test_partial_judging_never_publishes_a_final_score():
    rows = [{'item_id': 'a', 'domain': 'X', 'status': 'ok', 'score': 1., 'mean_pass': 1.},
            {'item_id': 'b', 'domain': 'X', 'status': 'pending', 'score': None, 'mean_pass': None}]
    result = metrics.summarize(rows, 2, 1)
    assert result['n_scored'] == 1 and result['n_pending_judgments'] == 1 and result['n_infra'] == 0
    assert result['score'] is None and result['ci95'] is None and not result['valid']


def test_judge_settings():
    from aa_evals.gdppdf.judge import judge_settings
    qwen_reg = {'overrides': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'reasoning_effort': 'xhigh',
                              'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True,
                                                       'reasoning_effort': 'xhigh'}}}
    q = judge_settings('serving:aa-judge-qwen38-27b', 'qwen38-27b', qwen_reg)
    assert q['reasoning_effort'] == 'medium' and q['chat_template_kwargs']['reasoning_effort'] == 'medium'
    assert (q['temperature'], q['top_p'], q['top_k'], q['max_tokens']) == (1.0, 0.95, 20, 32768)
    assert qwen_reg['overrides']['chat_template_kwargs']['reasoning_effort'] == 'xhigh'   # registry not mutated
    sup_reg = {'overrides': {'temperature': 1.0, 'top_p': 0.95,
                             'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}}}
    n = judge_settings('serving:aa-super', 'nemotron-3.5-super-vl', sup_reg)
    assert n == {'temperature': 1.0, 'top_p': 0.95, 'max_tokens': 32768,
                 'chat_template_kwargs': {'enable_thinking': True, 'medium_effort': False}}
    from aa_evals.gdppdf.run import judge_key
    assert judge_key('serving:x', q) != judge_key('serving:x', dict(q, reasoning_effort='xhigh'))


def test_read_jsonl_partial(tmp=None):
    import tempfile
    from aa_evals.gdppdf.run import read_jsonl
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'g.jsonl'
        f.write_text('{"task_id": "a", "repeat": 0, "x": 1}\n{"task_id": "a", "repeat": 1, "x"')
        assert list(read_jsonl(f)) == [('a', 0)]



def test_protocol_config():
    """M3.2: the registry's lab sampling as the base, the effort in both places, the slug by effort, the 128k cap's
    output reserve, the pinned judge arm, and Super's deliberate preset."""
    import tempfile
    from unittest import mock
    from aa_evals.common import protocol
    qwen = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'reasoning_effort': 'xhigh',
            'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'}}
    with tempfile.TemporaryDirectory() as tmp:
        for name, served, ov in [('q', 'qwen38-27b', qwen), ('s', 'nemotron-3.5-super-vl', {'temperature': 1.0}),
                                 ('g', 'gemma-4-26b-a4b-it', {'temperature': 1.0, 'top_k': 64,
                                                              'chat_template_kwargs': {'enable_thinking': True}})]:
            (Path(tmp) / name).mkdir()
            (Path(tmp) / name / 'endpoints.json').write_text(json.dumps(
                {'served_name': served, 'request': {'overrides': ov}}))
        # build_config straight from parsed args (main() needs the dataset)
        import aa_evals.gdppdf.run as R

        def parse(*extra):
            ap_args = ['--run-id', 'x', *extra]
            seen = {}
            orig = R.argparse.ArgumentParser.parse_args

            def grab(self, v=None, ns=None):
                a = orig(self, v, ns)
                seen['a'] = a
                raise SystemExit(0)
            with mock.patch.object(R.argparse.ArgumentParser, 'parse_args', grab):
                try:
                    R.main(ap_args)
                except SystemExit:
                    pass
            return seen['a']
        with mock.patch.object(protocol, 'SERVING_ROOT', Path(tmp)):
            a = parse('--model', 'serving:q', '--reasoning-effort', 'medium')
            c = R.build_config(a, Path(tmp) / 'run', [])
            b = c['generation']['body']
            assert (b['reasoning_effort'], b['chat_template_kwargs']['reasoning_effort'], b['top_k']) == \
                ('medium', 'medium', 20) and 'max_tokens' not in b
            assert c['aa_slug'] == 'qwen3-8-27b-medium' and c['effort'] == 'medium'
            assert c['generation']['output_reserve'] == R.CAPPED_OUTPUT_RESERVE
            a = parse('--model', 'serving:q', '--context-window', '0')
            c = R.build_config(a, Path(tmp) / 'run', [])
            assert c['aa_slug'] == 'qwen3-8-27b' and c['generation']['output_reserve'] == 65536
            a = parse('--model', 'serving:s')
            c = R.build_config(a, Path(tmp) / 'run', [])
            assert c['preset'] == 'super-vl' and c['generation']['body']['temperature'] == 0.6
            a = parse('--model', 'serving:g', '--reasoning-effort', 'high')
            try:
                R.build_config(a, Path(tmp) / 'run', [])
            except SystemExit:
                pass
            else:
                raise AssertionError('an effort on a model without effort levels must be refused')
    from aa_evals.judges import arms, pins
    assert pins.default_arm('gdppdf') in arms.ARMS



def test_gate_backpressure_uses_no_attempt():
    """A gate's 503 server_overloaded is waited out, not counted against max_attempts."""
    import asyncio
    import httpx
    from aa_evals.gdppdf import client as cl
    n = {'calls': 0}

    def handler(request):
        n['calls'] += 1
        if n['calls'] <= 5:
            return httpx.Response(503, headers={'Retry-After': '0', 'X-Gate': '1'},
                                  json={'error': {'message': 'gate: server at capacity', 'type': 'server_overloaded'}})
        return httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}],
                                         'usage': {}})

    async def go():
        c = cl.Client('url:http://fake/v1@m', max_attempts=2, backoff_cap=0)
        await c.http.aclose()
        c.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        orig = cl.asyncio.sleep
        async def fast(_):
            await orig(0)
        cl.asyncio.sleep = fast
        try:
            return await c.chat({'messages': []}, stream=False)
        finally:
            cl.asyncio.sleep = orig
    out = asyncio.run(go())
    assert n['calls'] == 6 and out['content'] == 'ok' and out['attempts'][-1]['gate_rejections'] == 5, (n, out)

def test_regrade_copy_preserves_generation_protocol():
    """A new judge may run with the original generation server offline."""
    import tempfile
    from unittest.mock import patch
    from aa_evals.gdppdf import run as R

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / 'source'
        src.mkdir()
        original = {
            'model': 'serving:offline-original', 'repeats': 1, 'git_sha': 'original-generation-sha',
            'selection': {'task_ids': ['t']},
            'generation': {'body': {'temperature': 1, 'reasoning_effort': 'medium'},
                           'output_reserve': 32768},
            'endpoint': {'served_name': 'original'}, 'limits': {'max_model_len': 262144},
            'context': {'window': 131072, 'requested': 131072, 'endpoint_max_model_len': 262144},
            'image_inputs': {'endpoint': False, 'why': 'text-only model'},
        }
        (src / 'config.json').write_text(json.dumps(original))
        generations = b'{"task_id":"t","repeat":0,"answer":"frozen answer"}\n'
        (src / 'generations.jsonl').write_bytes(generations)
        task, _ = _task(1)
        captured = []

        async def grade(a, run_dir, tasks, cfg):
            captured.append(cfg)

        async def forbidden(*args, **kwargs):
            raise AssertionError('a regrade must not generate')

        with patch.object(R, 'RUNS', root), patch.object(R, 'registry_request', return_value=(None, {})), \
                patch.object(R.data, 'load_tasks', return_value=[task]), \
                patch.object(R, 'stage_generate', forbidden), patch.object(R, 'stage_grade', grade), \
                patch.object(R, 'stage_summarize'):
            R.main(['--run-id', 'copy', '--generations-from', str(src), '--stage', 'grade',
                    '--judge', 'url:http://fake-judge/v1@judge'])
        cfg = captured[0]
        assert (root / 'copy' / 'generations.jsonl').read_bytes() == generations
        for key in ('generation', 'endpoint', 'limits', 'context', 'image_inputs'):
            assert cfg[key] == original[key], key
        assert cfg['generation_git_sha'] == original['git_sha']
        assert cfg['model'] == original['model']
        assert cfg['judge']['spec'] == 'url:http://fake-judge/v1@judge'


def test_stacked_regrade_copy_preserves_original_generation_identity():
    """Changing judges twice never replaces the generator SHA with a grader SHA."""
    import tempfile
    from unittest.mock import patch
    from aa_evals.gdppdf import run as R

    for prior_generation_sha in ('original-generation-code', None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source'
            src.mkdir()
            original = {
                'model': 'serving:offline-generator', 'repeats': 1,
                'git_sha': 'prior-grading-code' if prior_generation_sha else 'original-generation-code',
                'generation_git_sha': prior_generation_sha, 'selection': {'task_ids': ['t']},
                'generation': {'body': {'temperature': 1, 'reasoning_effort': 'max'}, 'output_reserve': 32768},
                'endpoint': {'served_name': 'original'}, 'limits': {'max_model_len': 262144},
                'context': {'window': 131072}, 'image_inputs': {'endpoint': False},
            }
            (src / 'config.json').write_text(json.dumps(original))
            generations = b'{"task_id":"t","repeat":0,"answer":"immutable stored answer"}\n'
            (src / 'generations.jsonl').write_bytes(generations)
            task, _ = _task(1)

            async def grade(a, run_dir, tasks, cfg):
                assert cfg['generation_git_sha'] == 'original-generation-code'

            async def forbidden(*args, **kwargs):
                raise AssertionError('stacked grading must not generate')

            for number in (1, 2):
                before = {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()}
                target = root / f'copy{number}'
                with patch.object(R, 'RUNS', root), patch.object(R, 'git_sha', return_value=f'grading-{number}'), \
                        patch.object(R, 'registry_request', return_value=(None, {})), \
                        patch.object(R.data, 'load_tasks', return_value=[task]), \
                        patch.object(R, 'stage_generate', forbidden), patch.object(R, 'stage_grade', grade), \
                        patch.object(R, 'stage_summarize'):
                    R.main(['--run-id', target.name, '--generations-from', str(src), '--stage', 'grade',
                            '--judge', 'url:http://fake-judge/v1@judge'])
                cfg = json.loads((target / 'config.json').read_text())
                assert cfg['generation_git_sha'] == 'original-generation-code'
                assert cfg['git_sha'] == f'grading-{number}'
                for key in ('generation', 'endpoint', 'limits', 'context', 'image_inputs', 'model', 'repeats'):
                    assert cfg[key] == original[key], key
                assert (target / 'generations.jsonl').read_bytes() == generations
                assert {p.name: p.read_bytes() for p in src.iterdir() if p.is_file()} == before
                src = target


def test_existing_grade_retains_protocol_with_generation_server_offline():
    import tempfile
    from unittest.mock import patch
    from aa_evals.gdppdf import run as R

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = root / 'same'
        run.mkdir()
        task, _ = _task(1)
        original = {'model': 'serving:gone', 'repeats': 1, 'git_sha': 'generation-code',
                    'selection': {'limit': 1, 'task_ids': [task.task_id]},
                    'generation': {'body': {'temperature': 1., 'reasoning_effort': 'max'}, 'output_reserve': 32768},
                    'context': {'window': 131072}, 'image_inputs': {'endpoint': False},
                    'judge': {'spec': 'old'}}
        (run / 'config.json').write_text(json.dumps(original))
        generations = b'preserved-generation-bytes\n'
        (run / 'generations.jsonl').write_bytes(generations)
        captured = []

        async def grade(a, run_dir, tasks, cfg):
            assert a.model == original['model'] and a.repeats == 1 and len(tasks) == 1
            captured.append(cfg)

        async def forbidden(*args, **kwargs):
            raise AssertionError('grading must not generate')

        with patch.object(R, 'RUNS', root), patch.object(R, 'git_sha', return_value='grading-code'), \
                patch.object(R, 'registry_request', side_effect=AssertionError('generation registry unavailable')), \
                patch.object(R.data, 'load_tasks', return_value=[task]), patch.object(R, 'stage_generate', forbidden), \
                patch.object(R, 'stage_grade', grade), patch.object(R, 'stage_summarize'):
            R.main(['--run-id', 'same', '--stage', 'grade', '--judge', 'url:http://fake/v1@judge'])
        cfg = captured[0]
        for key in ('generation', 'context', 'image_inputs', 'model', 'repeats'):
            assert cfg[key] == original[key], key
        assert cfg['generation_git_sha'] == 'generation-code'
        assert cfg['stage_git']['grade'] == cfg['git_sha'] == 'grading-code'
        assert (run / 'generations.jsonl').read_bytes() == generations




def test_text_only_task_set():
    """#348: the text-only set drops the 25 figure-bound tasks; 'all' keeps every task."""
    from aa_evals.gdppdf import data
    assert len(data.TEXT_ONLY_EXCLUDED) == 25 and data.DEFAULT_TASK_SET == 'text-only-v1' and 'all' in data.TASK_SETS
    assert {v[0] for v in data.TEXT_ONLY_EXCLUDED.values()} == {'clear', 'borderline'}
    out = next(iter(data.TEXT_ONLY_EXCLUDED))
    tasks = [Task(task_id=i, domain='D', prompt='p', pdf_path=Path('/x.pdf'), criteria=[]) for i in (out, 'keep')]
    assert [t.task_id for t in data.task_set(tasks, 'text-only-v1')] == ['keep']
    assert data.task_set(tasks, 'all') == tasks
    try:
        data.task_set(tasks, 'nope')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown set accepted')


def test_summary_headline_follows_the_runs_task_set():
    """A run's headline is its own set; an 'all' run (before #348) also reports text-only-v1 from the same rows."""
    import tempfile
    from types import SimpleNamespace
    from unittest.mock import patch
    from aa_evals.gdppdf import data, run as R

    out = next(iter(data.TEXT_ONLY_EXCLUDED))
    tasks = [Task(task_id=i, domain='D', prompt='p', pdf_path=Path('/x.pdf'), criteria=[]) for i in (out, 'keep')]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        args = SimpleNamespace(repeats=1, cutoff=False, model='synthetic', judge='synthetic', run_id='test')
        cfg = {'generation': {'body': {}}, 'selection': {'task_ids': [t.task_id for t in tasks]},
               'context': {'window': 131072}, 'judge': {'arm': 'synthetic'}, 'git_sha': 'frozen',
               'image_inputs': {'endpoint': True, 'vision': 'off', 'sent': False}}
        gens, grades = [], []
        for t, ok in ((out, 0.), ('keep', 1.)):
            gens.append({'task_id': t, 'repeat': 0, 'status': 'ok', 'at': 'a', 'answer': 'x', 'finish_reason': 'stop',
                         'usage': {}, 'plan': {'mode': 'text'}})
            grades.append({'task_id': t, 'repeat': 0, 'judge_key': 'k', 'answer_at': 'a', 'n_unscored': 0,
                           'n_passed': int(ok), 'n_criteria': 1, 'all_pass': ok, 'mean_pass': ok, 'grades': []})
        (root / 'generations.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in gens))
        (root / 'grades.jsonl').write_text(''.join(json.dumps(g) + '\n' for g in grades))
        with patch.object(R, 'current_judge_key', return_value='k'), patch.object(R, 'aa_reference', return_value=None):
            s = R.stage_summarize(args, root, tasks, cfg)   # a config without selection.task_set: before #348
            assert s['task_set'] == 'all' and s['score'] == 50 and s['n_items'] == 2
            assert s['task_sets']['all']['score'] == 50 and s['task_sets']['text-only-v1']['score'] == 100
            assert s['task_sets']['text-only-v1']['n_items'] == 1 and s['task_sets']['text-only-v1']['valid']
            assert s['protocol']['image_inputs'] is False   # what was sent, not what the endpoint could take
            assert len((root / 'items.jsonl').read_text().splitlines()) == 2
            keep = [t for t in tasks if t.task_id == 'keep']
            cfg['selection'] = {'task_ids': ['keep'], 'task_set': 'text-only-v1'}
            s = R.stage_summarize(args, root, keep, cfg)
            assert s['task_set'] == 'text-only-v1' and s['score'] == 100 and list(s['task_sets']) == ['text-only-v1']


def test_selection_defaults_to_the_text_only_set():
    import tempfile
    from unittest.mock import patch
    from aa_evals.gdppdf import data, run as R

    out = next(iter(data.TEXT_ONLY_EXCLUDED))
    tasks = [Task(task_id=i, domain='D', prompt='p', pdf_path=Path('/x.pdf'), criteria=[]) for i in (out, 'keep')]
    seen = []

    async def generate(a, run_dir, tasks, cfg):
        seen.append(([t.task_id for t in tasks], cfg['selection']))

    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(R, 'RUNS', Path(tmp)), patch.object(R, 'registry_request', return_value=(None, {})), \
                patch.object(R.data, 'load_tasks', return_value=tasks), patch.object(R, 'stage_generate', generate):
            base = ['--model', 'url:http://m/v1@m', '--stage', 'generate']
            R.main(['--run-id', 'new', *base])
            assert seen[-1] == (['keep'], {'limit': None, 'task_ids': ['keep'], 'task_set': 'text-only-v1'})
            R.main(['--run-id', 'full', '--task-set', 'all', *base])
            assert seen[-1] == ([out, 'keep'], {'limit': None, 'task_ids': [out, 'keep']})   # as before #348
            for argv, hint in ((['--run-id', 'ids', '--task-ids', out], '--task-set all'),
                               (['--run-id', 'full'], 'pass --task-set all')):
                try:
                    R.main(argv + base)
                except SystemExit as e:
                    assert hint in str(e), e
                else:
                    raise AssertionError(argv)


def test_vision_off_by_default_and_old_runs_resume_with_images():
    import tempfile
    from unittest import mock
    from aa_evals.gdppdf import run as R

    def parse(*extra):
        seen = {}
        orig = R.argparse.ArgumentParser.parse_args

        def grab(self, v=None, ns=None):
            seen['a'] = orig(self, v, ns)
            raise SystemExit(0)
        with mock.patch.object(R.argparse.ArgumentParser, 'parse_args', grab):
            try:
                R.main(['--run-id', 'x', '--model', 'url:http://m/v1@m', *extra])
            except SystemExit:
                pass
        return seen['a']
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / 'run'
        run.mkdir()
        a = parse()
        c = R.build_config(a, run, [])
        assert c['inputs']['vision'] == 'off' and 'alone' in c['harness']
        old = {**c, 'inputs': {k: v for k, v in c['inputs'].items() if k != 'vision'}}   # as written before #348
        (run / 'config.json').write_text(json.dumps(old))
        try:
            R.build_config(a, run, [])
        except SystemExit as e:
            assert '--vision auto' in str(e)
        else:
            raise AssertionError('a pre-#348 run resumed without images')
        assert R.build_config(parse('--vision', 'auto'), run, [])['inputs']['vision'] == 'auto'
    try:
        R.main(['--run-id', 'x', '--model', 'url:http://m/v1@m', '--max-images', '8'])
    except SystemExit as e:
        assert '--vision auto' in str(e)
    else:
        raise AssertionError('--max-images with --vision off accepted')


if __name__ == '__main__':
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            n += 1
            print('ok', name)
    print(f'{n} tests passed')


def test_aai_open_version_of_a_regrade_is_the_generation_runs(monkeypatch):
    """#481: a grade of stored generations keeps the generation run's AAI-open label (an old run stays 'unrecorded',
    also through a re-grade of a re-grade); a judge that is not the pinned arm makes it '+custom'; a stray launcher
    value in this process never relabels it."""
    import tempfile
    from unittest.mock import patch
    from aa_evals.gdppdf import run as R
    monkeypatch.setenv('AAI_OPEN_VERSION', 'v9')
    for recorded, judge_args, want in ((None, [], 'unrecorded'),
                                       ({'version': 'v0.1', 'deviations': [], 'source': 'launcher'}, [], 'v0.1'),
                                       ({'version': 'v0.1', 'deviations': [], 'source': 'launcher'},
                                        ['--judge', 'url:http://fake-judge/v1@judge'], 'v0.1+custom')):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'source'
            src.mkdir()
            original = {'model': 'serving:offline-generator', 'repeats': 1, 'git_sha': 'gen',
                        'selection': {'task_ids': ['t']},
                        'generation': {'body': {'temperature': 1}, 'output_reserve': 32768},
                        'context': {'window': 131072}, **({'aai_open': recorded} if recorded else {})}
            (src / 'config.json').write_text(json.dumps(original))
            (src / 'generations.jsonl').write_bytes(b'{"task_id":"t","repeat":0,"answer":"a"}\n')
            task, _ = _task(1)

            async def noop(*args, **kwargs):
                return None
            prev = src
            for number in (1, 2):
                target = root / f'copy{number}'
                with patch.object(R, 'RUNS', root), patch.object(R, 'git_sha', return_value='g'), \
                        patch.object(R, 'registry_request', return_value=(None, {})), \
                        patch.object(R.data, 'load_tasks', return_value=[task]), \
                        patch.object(R, 'stage_generate', noop), patch.object(R, 'stage_grade', noop), \
                        patch.object(R, 'stage_summarize'):
                    R.main(['--run-id', target.name, '--generations-from', str(prev), '--stage', 'grade', *judge_args])
                assert json.loads((target / 'config.json').read_text())['aai_open']['version'] == want, (recorded, judge_args)
                prev = target
