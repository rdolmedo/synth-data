import hashlib
import json
from pathlib import Path
import pytest

from tb_agents.harness_agents import cached_binary
from tb_agents.pi_continuation import CONTINUE, annotate_sessions, resampling_policy


def test_same_version_same_size_bundles_cannot_alias(tmp_path, monkeypatch):
    monkeypatch.setenv('HARNESS_AGENT_CACHE_LOCAL', str(tmp_path / 'local'))
    outputs = []
    for root, content in [('stock', b'stock'), ('patch', b'patch')]:
        cache = tmp_path / root
        src = cache / 'pi/1.2.0/linux-x64/harbor-pi.tar'
        src.parent.mkdir(parents=True)
        src.write_bytes(content)
        src.with_suffix('.tar.sha256').write_text(hashlib.sha256(content).hexdigest())
        outputs.append(cached_binary('pi', '1.2.0', 'linux-x64', 'harbor-pi.tar', shared_root=cache))
    assert outputs[0] != outputs[1]
    assert [p.read_bytes() for p in outputs] == [b'stock', b'patch']


def test_truncated_and_failed_turns_are_context_only(tmp_path):
    entries = [
        {'id': 'a', 'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'length',
         'content': [{'type': 'thinking', 'thinking': 'unfinished'}]}},
        {'id': 'b', 'parentId': 'a', 'type': 'message', 'message': {'role': 'user',
         'content': [{'type': 'text', 'text': CONTINUE}]}},
        {'id': 'c', 'parentId': 'b', 'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'toolUse',
         'content': [{'type': 'toolCall', 'name': 'bash', 'arguments': {'command': 'true'}}]}},
        {'id': 'd', 'parentId': 'c', 'type': 'message', 'message': {'role': 'assistant', 'stopReason': 'aborted'}},
        {'id': 'e', 'parentId': 'd', 'type': 'compaction', 'summary': 'history'},
    ]
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    original = '\n'.join(json.dumps(e) for e in entries) + '\n'
    session.write_text(original)
    result = annotate_sessions(tmp_path)
    annotated = result['sessions'][0]['entries']
    assert [e['sft_loss_weight'] for e in annotated[:4]] == [0, 0, 1, 0]
    assert annotated[4] == entries[4]
    assert [{k: v for k, v in e.items() if k != 'sft_loss_weight'} for e in annotated] == entries
    assert session.read_text() == original


def test_truncated_session_file_is_not_exportable(tmp_path):
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    session.write_text('{"type":"message","message":{"role":"assistant","stopReason":"stop"}}\n{"type":')
    result = annotate_sessions(tmp_path)
    assert not result['valid_for_export']
    assert result['parse_errors'] == [{'source': 'pi/sessions/test.jsonl', 'line': 2}]


def test_annotation_publish_retries_io_error_without_truncating_previous_masks(tmp_path, monkeypatch):
    import errno
    from tb_agents import pi_continuation as continuation
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    session.write_text(json.dumps({'id': 'a', 'type': 'message', 'message': {
        'role': 'assistant', 'stopReason': 'length', 'content': 'reasoning'}}) + '\n')
    target = session.parent.parent / 'sft-events.json'
    original = '{"previous_mask": 0}\n'
    target.write_text(original)
    replace = continuation.os.replace
    attempts = []
    def transient_replace(source, destination):
        attempts.append(source)
        assert target.read_text() == original
        if len(attempts) < 3:
            raise OSError(errno.EIO, 'Input/output error')
        replace(source, destination)
    monkeypatch.setattr(continuation.os, 'replace', transient_replace)
    monkeypatch.setattr(continuation.time, 'sleep', lambda _: None)
    result = annotate_sessions(tmp_path)
    assert len(attempts) == 3
    assert json.loads(target.read_text()) == result
    assert result['sessions'][0]['entries'][0]['sft_loss_weight'] == 0
    assert not list(target.parent.glob('.sft-events.*'))


def test_annotation_publish_persistent_io_error_preserves_previous_sidecar(tmp_path, monkeypatch):
    import errno
    from tb_agents import pi_continuation as continuation
    target = tmp_path / 'pi/sft-events.json'
    target.parent.mkdir()
    target.write_text('{"previous_mask": 0}\n')
    def broken_fsync(_):
        raise OSError(errno.EIO, 'Input/output error')
    monkeypatch.setattr(continuation.os, 'fsync', broken_fsync)
    monkeypatch.setattr(continuation.time, 'sleep', lambda _: None)
    with pytest.raises(OSError):
        annotate_sessions(tmp_path)
    assert target.read_text() == '{"previous_mask": 0}\n'
    assert not list(target.parent.glob('.sft-events.*'))


@pytest.mark.parametrize('separator', ['\u0085', '\u2028', '\u2029'])
def test_unicode_separators_preserve_native_events_and_loss_weights(tmp_path, separator):
    entries = [
        {'id': 'tool', 'type': 'message', 'message': {'role': 'toolResult',
         'content': [{'type': 'text', 'text': 'observation' + separator + 'continued'}]}},
        {'id': 'thought', 'parentId': 'tool', 'type': 'message',
         'message': {'role': 'assistant', 'stopReason': 'length',
         'content': [{'type': 'thinking', 'thinking': 'reasoning' + separator + 'continued'}]}},
        {'id': 'answer', 'parentId': 'thought', 'type': 'message',
         'message': {'role': 'assistant', 'stopReason': 'stop',
         'content': [{'type': 'text', 'text': 'final' + separator + 'answer'}]}},
    ]
    session = tmp_path / 'pi/sessions/unicode.jsonl'
    session.parent.mkdir(parents=True)
    original = ('\n'.join(json.dumps(e, ensure_ascii=False) for e in entries) + '\n').encode()
    session.write_bytes(original)
    result = annotate_sessions(tmp_path)
    assert result['valid_for_export']
    assert result['parse_errors'] == []
    annotated = result['sessions'][0]['entries']
    assert [e['sft_loss_weight'] for e in annotated] == [0, 0, 1]
    assert [{k: v for k, v in e.items() if k != 'sft_loss_weight'} for e in annotated] == entries
    assert session.read_bytes() == original


@pytest.mark.parametrize('previous_sidecar', [False, True])
def test_failed_annotation_write_keeps_published_sidecar_and_native_bytes(tmp_path, monkeypatch, previous_sidecar):
    from tb_agents import pi_continuation
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    native = b'{"type":"session","id":"s"}\n'
    session.write_bytes(native)
    target = tmp_path / 'pi/sft-events.json'
    previous = b'{"sessions":[],"preserved":true}\n'
    if previous_sidecar:
        target.write_bytes(previous)

    def interrupted_write(_):
        raise OSError('Interrupted annotation write')

    monkeypatch.setattr(pi_continuation.os, 'fsync', interrupted_write)
    with pytest.raises(OSError, match='Interrupted annotation write'):
        annotate_sessions(tmp_path)
    if previous_sidecar:
        assert target.read_bytes() == previous
    else:
        assert not target.exists()
    assert session.read_bytes() == native
    assert not list(target.parent.glob('sft-events.json.*.tmp'))


def test_unicode_line_separators_in_native_messages_are_not_record_boundaries(tmp_path):
    message = {'type': 'message', 'id': 'a', 'message': {'role': 'assistant',
        'stopReason': 'stop', 'content': [{'type': 'text', 'text': 'NEL\x85LS\u2028PS\u2029end'}]}}
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    original = json.dumps(message, ensure_ascii=False) + '\n'
    session.write_text(original)
    result = annotate_sessions(tmp_path)
    assert result['valid_for_export']
    assert result['parse_errors'] == []
    assert result['sessions'][0]['entries'][0]['message'] == message['message']
    assert session.read_text() == original


def test_text_only_export_cannot_drop_continuation_weights(tmp_path):
    from gen.export_sft import load_trial
    annotations = tmp_path / 'agent/pi/sft-events.json'
    annotations.parent.mkdir(parents=True)
    annotations.write_text('{}')
    with pytest.raises(ValueError, match='Pi-aware training loader'):
        load_trial(tmp_path, {})


def test_retry_usage_is_added_without_double_counting_selected_sample(tmp_path):
    from gen.pi_cont16k import usage_summary
    logs = tmp_path / 'proxy'
    logs.mkdir()
    (logs / 'turns.jsonl').write_text(json.dumps({'sample_usage': [
        {'completion_tokens': 512}, {'completion_tokens': 512}, {'completion_tokens': 128}]}) + '\n')
    trial = tmp_path / 'jobs/task__trial'
    trial.mkdir(parents=True)
    (trial / 'result.json').write_text(json.dumps({'agent_result': {'n_output_tokens': 1000}}))
    result = usage_summary(logs, trial.parent)
    assert result['discarded_output_tokens'] == 1024
    assert result['backend_output_tokens'] == 2024


def test_new_default_and_explicit_resampling():
    assert resampling_policy({}) == (0, 'default')
    assert resampling_policy({'PI_CONT16K_RESAMPLES': '2'}) == (2, 'environment')
    with pytest.raises(ValueError, match='nonnegative'):
        resampling_policy({'PI_CONT16K_RESAMPLES': '-1'})


def test_running_and_queued_workers_keep_the_manifest_policy(tmp_path):
    run = tmp_path / 'campaign/pi-cont16k'
    (run / 'results').mkdir(parents=True)
    (run / 'manifest.json').write_text(json.dumps({'pi_continuation': {'proxy_resamples': 2}}))
    replica = tmp_path / 'campaign/replicas/a1/pi-cont16k'
    replica.mkdir(parents=True)
    (replica / 'results').symlink_to(run / 'results', target_is_directory=True)
    for queue in [run, replica]:
        ad = tmp_path / 'job.ad'
        ad.write_text('Args = ' + json.dumps(f'{queue} /server/endpoint.url pi-cont16k 8') + '\n')
        assert resampling_policy({'_CONDOR_JOB_AD': str(ad)})[0] == 2
        assert resampling_policy({'PI_CONT16K_RUN': str(queue)})[0] == 2
        with pytest.raises(ValueError, match='conflicts'):
            resampling_policy({'PI_CONT16K_RUN': str(queue), 'PI_CONT16K_RESAMPLES': '0'})
