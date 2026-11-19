"""CPU-only native GDPval trace adapter; synthetic outcomes, no model or grader."""
import gzip
import json
from dataclasses import asdict

from aa_evals.tracejudge.core import build_row
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.rules import decide, run_rules


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))


def make_run(tmp_path):
    run = tmp_path / 'gdpval' / 'r'
    write(run / 'config.json', {'benchmark': 'GDPval-AA', 'selection': {'task_ids': ['task']}, 'repeats': 1,
                               'agent_network': {'mode': 'offline'}, 'limits': {'max_infra_tries': 2},
                               'harness': {'max_turns': 250}, 'context': {'window': 131072}})
    (run / 'attempts').mkdir()
    return run


def outcome(run, status='ok', **extra):
    path = run / 'attempts/task__r0'
    write(path / 'outcome.json', {'task_id': 'task', 'repeat': 0, 'status': status,
                                 'end': 'finish' if status == 'ok' else 'infra', 'n_submitted': 1, **extra})
    history = [[{'role': 'user', 'content': 'original task'},
                {'role': 'assistant', 'content': 'before summary'}],
               [{'role': 'user', 'content': 'retained task'},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'c', 'function': {
                    'name': 'code_exec', 'arguments': '{"cmd":"ls"}'}}]},
                {'role': 'tool', 'content': 'file.pdf', 'tool_call_id': 'c'},
                {'role': 'assistant', 'content': 'finished'}]]
    with gzip.open(path / 'transcript.json.gz', 'wt') as f:
        json.dump({'history': history, 'events': [{'source': 'sandbox', 'text': 'local execution completed'}]}, f)
    return path


def test_native_completed_before_items_preserves_histories_and_paths(tmp_path):
    run = make_run(tmp_path)
    path = outcome(run)
    layout = open_run(run)
    assert layout.name == 'gdpval' and layout.finished() == ['task/r0'] and layout.complete()
    t = layout.load('task/r0')
    assert t.recorded['status'] == 'ok' and t.recorded['score'] is None
    assert t.recorded['agent_network'] == {'mode': 'offline'}
    assert t.recorded['end_state'] == 'finish' and t.recorded['context_window'] == 131072
    assert {x.ref.split(':')[0] for x in t.turns} == {'history[0]', 'history[1]'}
    assert any(x.kind == 'tool_result' and x.text == 'file.pdf' for x in t.turns)
    assert t.source == str(path / 'transcript.json.gz')
    assert t.events[-1].ref == str(path / 'outcome.json')


def test_incomplete_or_unreadable_outcomes_are_not_finished(tmp_path):
    run = make_run(tmp_path)
    path = run / 'attempts/task__r0/outcome.json'
    for content in ('{"status":', {'status': 'running'}, []):
        write(path, content)
        layout = open_run(run)
        assert layout.finished() == [] and not layout.complete()


def test_retryable_infra_is_not_frozen_under_the_final_logical_key(tmp_path):
    run = make_run(tmp_path)
    path = outcome(run, 'infra_error', error='endpoint: unavailable')
    assert open_run(run).finished() == []
    write(path / 'outcome.infra1.json', {'status': 'infra_error'})
    layout = open_run(run)
    assert layout.finished() == ['task/r0'] and layout.complete()
    t = layout.load('task/r0')
    hits = run_rules(t)
    row = build_row(t, hits, decide(hits), None, None, None)
    assert row['known_infra'] and row['flagged']
    # A subsequent successful outcome keeps the logical key and replaces its source status.
    outcome(run, 'ok')
    assert open_run(run).load('task/r0').recorded['status'] == 'ok'


def test_postscore_parity_and_explicit_empty_policy_override(tmp_path):
    run = make_run(tmp_path)
    path = outcome(run)
    item = {'key': 'task/r0', 'item_id': 'task', 'repeat': 0, 'status': 'ok', 'score': 0.0,
            'agent_network': {}, 'transcript': str(path / 'transcript.json.gz')}
    write(run / 'items.jsonl', json.dumps(item) + '\n')
    native, generic = open_run(run).load('task/r0'), open_run(run, 'generic').load('task/r0')
    assert native.key == generic.key and native.source == generic.source
    assert native.recorded['score'] == generic.recorded['score'] == 0.0
    assert native.recorded['agent_network'] == generic.recorded['agent_network'] == {}
    assert [asdict(x) for x in native.turns] == [asdict(x) for x in generic.turns]
    assert [asdict(x) for x in native.events[:-1]] == [asdict(x) for x in generic.events]


def test_stale_scored_item_does_not_expose_an_in_progress_retry(tmp_path):
    run = make_run(tmp_path)
    path = run / 'attempts/task__r0'
    path.mkdir()
    write(run / 'items.jsonl', json.dumps({'key': 'task/r0', 'item_id': 'task', 'repeat': 0,
                                         'status': 'infra_error', 'score': None}) + '\n')
    assert open_run(run).finished() == []
    assert not open_run(run).complete()


def test_items_only_export_retains_generic_contract(tmp_path):
    run = make_run(tmp_path)
    write(run / 'items.jsonl', json.dumps({'key': 'task/r0', 'item_id': 'task', 'repeat': 0,
                                         'status': 'ok', 'score': 0.75}) + '\n')
    assert open_run(run).load('task/r0').recorded['score'] == 0.75


def test_relative_run_directory_resolves_native_transcript(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    path = outcome(run)
    monkeypatch.chdir(tmp_path)
    t = open_run('gdpval/r').load('task/r0')
    assert t.source == str(path / 'transcript.json.gz')
    assert t.turns[-1].text == 'finished'


def test_native_index_read_once_per_batch_and_score_write_refreshes_it(tmp_path, monkeypatch):
    from aa_evals.tracejudge import layouts
    run = make_run(tmp_path)
    outcome(run)
    original_read = layouts._read_json
    reads = []

    def read(path, default=None):
        if path.name == 'outcome.json':
            reads.append(path)
        return original_read(path, default)

    monkeypatch.setattr(layouts, '_read_json', read)
    layout = open_run(run)
    layout.finished()
    assert layout.load('task/r0').recorded['score'] is None
    assert layout.load('task/r0').recorded['score'] is None
    assert layout.complete() and len(reads) == 1
    write(run / 'items.jsonl', json.dumps({'key': 'task/r0', 'item_id': 'task', 'repeat': 0,
                                         'status': 'ok', 'score': 0.5}) + '\n')
    assert layout.load('task/r0').recorded['score'] == 0.5
    assert len(reads) == 2
