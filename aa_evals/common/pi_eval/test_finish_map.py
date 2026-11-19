"""#375 sftnotice: the finish notice map SFT exporters ship (finish_map.py), and a synthetic trial archive helper
the M5 exporter tests share."""
import io
import json
import tarfile

import pytest

from aa_evals.common.pi_eval import finish_map as F
from tb_agents import pi_context_limit as G

V4 = {'warn_at': 40000, 'reserve': 16384, 'policy': G.STICKY_POLICY, 'sticky': True}
V3 = {'warn_at': 40000, 'reserve': 16384, 'policy': G.FINISH_POLICY}
CONT = [{'type': 'text', 'text': 'Thinking exceeded the maximum thinking budget, please continue.'}]


def notice_line(index=3, role='user', content=CONT, text='[Harness notice] About 30,000 tokens', **kw):
    return {'policy': G.STICKY_POLICY, 'notice': 'warn', 'message_count': index + 1, 'index': index, 'role': role,
            'base_sha256': G.notice_base_sha256(content), 'text': text, 'appended': False, **kw}


def capped_turns(*prompts):
    """sticky_proxy's --turn-log as it really is: one line per CAPPED turn only (a trial with 41 requests had 4 lines),
    so it says nothing about the requests near the window that got notices."""
    return ''.join(json.dumps({'t': 1.0, 'path': '/v1/chat/completions', 'n_messages': 2 * k + 2, 'capped_attempts': 1,
                               'attempts': 1, 'continued': 0, 'outcome': 'gave_up',
                               'sample_usage': [{'prompt_tokens': p, 'completion_tokens': 16384}]}) + '\n'
                   for k, p in enumerate(prompts))


def trial_tar(path, trial, session_text, first_request, notices=None, turns_text=None, extra=None):
    """A trial.tar.gz laid out as pi_eval's: <job>/<trial>/agent/pi/sessions/*.jsonl, <job>/pi-cont16k/{first-request,
    finish-notices.jsonl, turns.jsonl}. `trial` is the trial dir in the archive (<job>/<trial name>)."""
    job = trial.rsplit('/', 1)[0]
    files = {f'{trial}/agent/pi/sessions/2026-11-13T00-00-00-000Z_s.jsonl': session_text,
             f'{job}/pi-cont16k/first-request/0001_x_v1_chat_completions.json': json.dumps(first_request),
             **(extra or {})}
    if notices is not None:
        files[f'{job}/pi-cont16k/finish-notices.jsonl'] = ''.join(json.dumps(x) + '\n' for x in notices)
    if turns_text is not None:
        files[f'{job}/pi-cont16k/turns.jsonl'] = turns_text
    with tarfile.open(path, 'w:gz') as tf:
        for name, text in files.items():
            data = text if isinstance(text, bytes) else text.encode('utf-8', 'surrogateescape')
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return path


def test_member_kind():
    assert F.member_kind('job/job/pi-cont16k/finish-notices.jsonl') == 'notices'
    assert F.member_kind('job/job/pi-cont16k/turns.jsonl') is None
    assert F.member_kind('job/.pi-cont16k/finish-notices.jsonl') is None
    assert F.member_kind('job/job/pi-cont16k/wire-requests.jsonl') is None


def test_run_policy():
    assert F.run_policy({'protocol': {}}, 'r') is None
    assert F.run_policy({'protocol': {'finish_notice': None}}, 'r') is None
    assert F.run_policy({'protocol': {'finish_notice': V4}}, 'r') == V4
    with pytest.raises(SystemExit, match='non-sticky'):
        F.run_policy({'protocol': {'finish_notice': V3}}, 'r')


def test_parse():
    line = notice_line(text='a b')               # U+2028 inside a JSON string must not split the line
    assert F.parse(json.dumps(line, ensure_ascii=False) + '\n\n') == [line]
    with pytest.raises(ValueError):
        F.parse(json.dumps({**line, 'policy': G.FINISH_POLICY}))   # a v3 line has no map
    with pytest.raises(ValueError):
        F.parse(json.dumps({k: v for k, v in line.items() if k != 'base_sha256'}))
    with pytest.raises(ValueError):
        F.parse('{"policy": ')


def test_attempt_notices():
    text = json.dumps(notice_line()) + '\n'
    assert F.attempt_notices(None, 131072, text) == (None, None)                  # run without a notice
    assert F.attempt_notices(None, 131072, None) == (None, None)
    assert F.attempt_notices(V4, 131072, text) == ([notice_line()], None)
    assert F.attempt_notices(V4, 131072, '') == ([], None)                        # the guard's empty file: no notice
    # SHOULD-1: a v4 guard creates the file at startup, so no file = lost map, whatever else the trial logged
    assert F.attempt_notices(V4, 131072, None) == (None, 'finish_notices_missing')
    assert F.attempt_notices(V4, 131072, None, max_prompt_tokens=3000) == (None, 'finish_notices_missing')
    # secondary check: an empty map although the largest served prompt was inside the threshold
    near = 131072 - 40000 + G.NOTICE_MARGIN
    assert F.attempt_notices(V4, 131072, '', max_prompt_tokens=near) == (None, 'finish_notices_empty_near_window')
    assert F.attempt_notices(V4, 131072, '', max_prompt_tokens=near - 1) == ([], None)
    assert F.attempt_notices(V4, 131072, text, max_prompt_tokens=near + 9000) == ([notice_line()], None)
    assert F.attempt_notices(V4, 131072, '{"x": 1}\n')[1].startswith('finish_notices_malformed')
    appended = json.dumps(notice_line(appended=True)) + '\n'
    assert F.attempt_notices(V4, 131072, appended) == (None, 'finish_notice_appended')


def test_threshold_covers_the_reserve_clamp():
    assert F.threshold({'warn_at': 0, 'reserve': 16384}) == 16384 + G.TURN - 1
    assert F.threshold(V4) == 40000
    for remaining in (40000, 32767, 20000):
        assert G.finish_plan(remaining, 16384, 40000, 16384)[1] is not None
    assert G.finish_plan(40001, 16384, 40000, 16384)[1] is None
