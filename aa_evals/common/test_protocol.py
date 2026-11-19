"""Offline tests of the M3.0 protocol helpers (python -B -m aa_evals.common.test_protocol)."""
import json
import tempfile
from pathlib import Path

from aa_evals.common import protocol as P

QWEN = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'reasoning_effort': 'xhigh',
        'chat_template_kwargs': {'enable_thinking': True, 'preserve_thinking': True, 'reasoning_effort': 'xhigh'}}
GEMMA = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64, 'chat_template_kwargs': {'enable_thinking': True}}


def test_effort_sets_both_places_and_refuses_models_without_levels():
    b = P.apply_effort(QWEN, 'medium')
    assert b['reasoning_effort'] == 'medium' and b['chat_template_kwargs']['reasoning_effort'] == 'medium'
    assert b['chat_template_kwargs']['preserve_thinking'] is True and QWEN['reasoning_effort'] == 'xhigh'
    only_kw = {'chat_template_kwargs': {'reasoning_effort': 'max', 'thinking': True}}
    assert P.apply_effort(only_kw, 'high')['reasoning_effort'] == 'high'
    try:
        P.apply_effort(GEMMA, 'medium')
    except ValueError:
        pass
    else:
        raise AssertionError('an effort on a model without effort levels must be refused')


def test_aa_variant_follows_the_effort():
    assert P.aa_model('qwen38-27b', P.apply_effort(QWEN, 'medium'))[1] == 'qwen3-8-27b-medium'
    assert P.aa_model('qwen38-27b', QWEN)[1] == 'qwen3-8-27b'
    assert P.aa_model('gemma-4-26b-a4b-it', GEMMA)[1] == 'gemma-4-26b-a4b'
    assert P.aa_model('qwen38-27b', P.apply_effort(QWEN, 'high')) == (None, None)   # AA has no High variant
    assert P.thinking_of(GEMMA) is True and P.effort_of(GEMMA) is None


def test_window():
    assert P.window(262144, 131072)[0] == 131072
    assert P.window(65536, 131072)[0] == 65536
    assert P.window(262144, 0)[0] == 262144
    assert P.window(None, 131072)[0] == 131072


def test_image_inputs_from_launch_argv():
    with tempfile.TemporaryDirectory() as tmp:
        old = P.SERVING_ROOT
        P.SERVING_ROOT = Path(tmp)
        try:
            for name, argv, want in [
                    ('vl', ['vllm', 'serve', '--limit-mm-per-prompt', '{"image":256,"video":0}'], True),
                    ('lmo', ['vllm', 'serve', '--language-model-only'], False),
                    ('none', ['vllm', 'serve', '--max-model-len', '1000'], False),
                    ('zero', ['vllm', 'serve', '--limit-mm-per-prompt', '{"image":0}'], False)]:
                (Path(tmp) / name).mkdir()
                (Path(tmp) / name / 'launch.json').write_text(json.dumps({'argv': argv}))
                assert P.image_inputs(f'serving:{name}')[0] is want, name
            (Path(tmp) / 'rep').mkdir()
            (Path(tmp) / 'rep' / 'launch.json').write_text(json.dumps({'replicas': [{'argv': ['--language-model-only']}]}))
            assert P.image_inputs('serving:rep')[0] is False
            assert P.image_inputs('url:http://x/v1@m')[0] is None
        finally:
            P.SERVING_ROOT = old


def test_n_flagged_from_trace_judge():
    from aa_evals.tracejudge import fold_summary
    with tempfile.TemporaryDirectory() as tmp:
        base = P.summary_fields(model='serving:x', served_name='qwen38-27b', body={'reasoning_effort': 'medium'},
                                context=None, judge_arm=None, judge_settings_key=None, code_sha=None, repeats=5,
                                image_inputs=True)
        assert 'n_flagged' not in base and base['protocol']['aa_variant'] == 'Qwen3.8 27B (Medium)'
        assert fold_summary(Path(tmp), base)['n_flagged'] is None
        (Path(tmp) / 'flags.jsonl').write_text('{"key": "a.r0", "flagged": true, "flags": []}\n'
                                               '{"key": "b.r0", "flagged": false}\n')
        assert fold_summary(Path(tmp), base)['n_flagged'] == 1


if __name__ == '__main__':
    for k, fn in list(globals().items()):
        if k.startswith('test_'):
            fn()
            print('ok', k)
