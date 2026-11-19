"""Neutral fixtures only: emitted PNG integrity, legacy budget boundary and unscored grading."""
import asyncio
import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from aa_evals.gdpval import judge as J
from aa_evals.gdpval.render_cache import RenderIntegrityError, publish_png


def test_atomic_publish_preserves_valid_encoder_bytes(tmp_path):
    image = Image.new('RGB', (3, 5), '#123456')
    old, new = tmp_path / 'old.png', tmp_path / 'new.png'
    image.save(old, 'PNG', optimize=True)
    publish_png(new, lambda path: image.save(path, 'PNG', optimize=True))
    assert new.read_bytes() == old.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['new.png', 'old.png']


@pytest.mark.parametrize('data', [b'', b'nonempty but not PNG'])
def test_failed_png_never_replaces_valid_previous_file(tmp_path, data):
    path = tmp_path / 'page.png'
    Image.new('RGB', (1, 1)).save(path)
    before = path.read_bytes()
    with pytest.raises(RenderIntegrityError):
        publish_png(path, lambda target: target.write_bytes(data))
    assert path.read_bytes() == before and len(list(tmp_path.iterdir())) == 1


def test_temporary_file_creation_failure_is_grader_owned(tmp_path, monkeypatch):
    from aa_evals.gdpval import render_cache as C
    def denied(*args, **kwargs):
        raise OSError(28, 'neutral disk full')
    monkeypatch.setattr(C.tempfile, 'mkstemp', denied)
    with pytest.raises(RenderIntegrityError, match='disk full'):
        publish_png(tmp_path / 'page.png', lambda target: pytest.fail('must not encode'))


def load_standalone_renderer(monkeypatch):
    directory = Path(__file__).parent
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('neutral_standalone_renderer', directory / 'render.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_script_finds_sibling_module_without_install(tmp_path):
    source, output = tmp_path / 'in', tmp_path / 'out'
    source.mkdir(); output.mkdir()
    result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('render.py')), str(source), str(output)],
                            capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((output / 'manifest.json').read_text()) == {'files': [], 'n_files': 0}


def test_generated_failure_is_explicit_not_unreadable_model_file(tmp_path, monkeypatch):
    renderer = load_standalone_renderer(monkeypatch)
    source, output = tmp_path / 'in', tmp_path / 'out'
    source.mkdir(); output.mkdir(); (source / 'neutral.txt').write_text('neutral')
    def broken(*args):
        raise renderer.RenderIntegrityError('img/page.png', 'generated PNG is empty')
    monkeypatch.setattr(renderer, 'render_file', broken)
    assert renderer.main(source, output) is False
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['render_failed'] is True
    assert manifest['render_integrity_error']['kind'] == 'gdpval_render_png_cache_integrity'
    with pytest.raises(J.RenderIntegrityError):
        J.side_content('SUBMISSION A', output, 100, 1, {}, 'empty')


def test_malformed_original_stays_an_unreadable_deliverable(tmp_path, monkeypatch):
    renderer = load_standalone_renderer(monkeypatch)
    source = tmp_path / 'invalid.png'; source.write_bytes(b'not an image')
    images, scratch = tmp_path / 'images', tmp_path / 'scratch'
    images.mkdir(); scratch.mkdir()
    monkeypatch.setattr(renderer, 'run', lambda *args, **kw: SimpleNamespace(returncode=1, stderr='invalid source'))
    record = renderer.render_file(source, 'invalid.png', images, scratch, 1)[0]
    assert record['images'] == [] and record['notes']
    assert 'render_integrity_error' not in record


def manifest(directory, *, first_empty):
    (directory / 'img').mkdir(parents=True)
    (directory / 'img/good.png').write_bytes(b'legacy nonempty image fixture')
    (directory / 'img/empty.png').write_bytes(b'')
    images = ['empty.png', 'good.png'] if first_empty else ['good.png', 'empty.png']
    (directory / 'manifest.json').write_text(json.dumps({'files': [
        {'name': 'neutral.pdf', 'kind': 'pdf', 'size': 100, 'text': 'neutral', 'images': images, 'notes': []}]}))


def test_legacy_empty_is_rejected_only_when_selected(tmp_path):
    manifest(tmp_path, first_empty=False)
    assert any(p['type'] == 'image_url' for p in J.side_content('SUBMISSION A', tmp_path, 100, 1, {}, 'empty'))
    with pytest.raises(RenderIntegrityError, match='selected legacy PNG is empty'):
        J.side_content('SUBMISSION A', tmp_path, 100, 2, {}, 'empty')


@pytest.mark.parametrize('retry_errors', [False, True])
def test_pending_grading_no_model_call_no_follow_retry_and_no_numeric_score(tmp_path, monkeypatch, retry_errors):
    from aa_evals.gdpval import run as R
    run = tmp_path / 'runs/neutral'; attempt = run / 'attempts/task__r0'
    manifest(attempt / 'render', first_empty=True)
    (attempt / 'outcome.json').write_text(json.dumps({'status': 'ok', 'end': 'finish', 'turns': 1}))
    cfg = {'selection': {'task_ids': ['task']}, 'repeats': 1, 'model': {'spec': 'neutral', 'served_name': 'neutral',
           'body': {}, 'vision': False}, 'context': {}, 'git_sha': 'a' * 40}
    (run / 'config.json').write_text(json.dumps(cfg)); (run / 'matches').mkdir()
    expert = tmp_path / 'prep/task/expert'; expert.mkdir(parents=True)
    (expert / 'manifest.json').write_text('{"files": []}')
    task = SimpleNamespace(task_id='task', occupation='neutral', sector='neutral', prompt='neutral',
                           reference_files=[], deliverable_files=['expert'])
    monkeypatch.setattr(R, 'RUNS', tmp_path / 'runs'); monkeypatch.setattr(R, 'PREP', tmp_path / 'prep')
    monkeypatch.setattr(R.T, 'load_tasks', lambda: [task])
    class NoModel:
        def __init__(self, *args, **kwargs):
            pass
        async def ask(self, *args, **kwargs):
            pytest.fail('broken evaluator cache must not call model')
        async def aclose(self):
            pass
    monkeypatch.setattr(J, 'Judge', NoModel)
    original_sleep = asyncio.sleep; sleeps = 0
    async def yield_only(*args):
        nonlocal sleeps
        sleeps += 1
        assert sleeps < 10, 'follow repeated a terminal grading-cache error'
        await original_sleep(0)
    monkeypatch.setattr(R.asyncio, 'sleep', yield_only)
    args = SimpleNamespace(run_id='neutral', judge_arm='qwen38-27b-medium-pin', judge_endpoint=None,
                           judge_images=1, concurrency=2, follow=True, retry_judge_errors=retry_errors, bootstrap=10)
    asyncio.run(R.stage_judge(args))
    records = R.read_jsonl(run / 'matches/qwen38-27b-medium-pin.jsonl')
    assert len(records) == 2 and all(r['winner'] is None and r['tries'] == 0 and r['grading_pending'] for r in records)
    assert all('model_score' not in r for r in records)
    R.stage_score(args)
    item = R.read_jsonl(run / 'items.jsonl')[0]
    summary = json.loads((run / 'summary.json').read_text())
    assert item['score'] is None and item['judge_status'] == 'pending'
    assert summary['score'] is None and summary['ci95'] is None and summary['valid'] is False
    assert summary['complete'] is False and summary['n_pending_judgments'] == 2

@pytest.mark.parametrize('converted', [b'', b'nonempty corrupt converted PNG'])
def test_successful_converter_bad_output_stays_typed_through_parent(tmp_path, monkeypatch, converted):
    from aa_evals.gdpval import run as R
    renderer = load_standalone_renderer(monkeypatch)
    source, output = tmp_path / 'in', tmp_path / 'out'
    source.mkdir(); (source / 'neutral.psd').write_bytes(b'neutral unsupported-by-Pillow source')
    def convert(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(converted)
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(renderer, 'run', convert)
    def child(cmd, **kwargs):
        # Exercise the actual child main and parent result handling without a container/model.
        ok = renderer.main(source, output)
        return SimpleNamespace(returncode=0 if ok else 2, stdout='', stderr='typed child PNG failure')
    monkeypatch.setattr(R.subprocess, 'run', child)
    assert R.render(source, output, 'neutral-unused.sif', Path(__file__).parent) is True
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['render_integrity_error']['kind'] == 'gdpval_render_png_cache_integrity'
    assert 'convert succeeded' in manifest['render_integrity_error']['detail']
    assert manifest['render_failed'] is True and manifest['files'] == []
    assert not (output / 'img/f001.png').exists()
    assert (output / 'render_error.txt').read_text() == 'typed child PNG failure'
    with pytest.raises(J.RenderIntegrityError):
        J.side_content('SUBMISSION A', output, 100, 1, {}, 'empty')


@pytest.mark.parametrize('returncode', [0, 1])
def test_usable_converter_output_preserves_legacy_png_bytes(tmp_path, monkeypatch, returncode):
    renderer = load_standalone_renderer(monkeypatch)
    source = tmp_path / 'neutral.psd'; source.write_bytes(b'neutral unsupported-by-Pillow source')
    images, scratch = tmp_path / 'images', tmp_path / 'scratch'
    images.mkdir(); scratch.mkdir()
    image = Image.new('RGB', (3, 5), '#345678')
    expected = tmp_path / 'old-normalized.png'; image.save(expected, 'PNG', optimize=True)
    def convert(cmd, **kwargs):
        image.save(cmd[-1], 'PNG')
        assert cmd[:2] == ['convert', str(source) + '[0]'] and kwargs['timeout'] == 120
        return SimpleNamespace(returncode=returncode, stderr='neutral decoder warning' if returncode else '')
    monkeypatch.setattr(renderer, 'run', convert)
    record = renderer.render_file(source, source.name, images, scratch, 1)[0]
    assert record['images'] == ['f001.png']
    assert (images / 'f001.png').read_bytes() == expected.read_bytes()
    assert sorted(p.name for p in images.iterdir()) == ['f001.png']
