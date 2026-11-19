"""Datakit-format text (Marin's post-trained Snowball releases: <|start_think|>/<|end_think|> reasoning, JSON calls in
<tool_call> blocks) through the capped protocol's raw-text paths, which before only knew the Snowball/Super format: a
cut turn keeps its reasoning apart from its content, and an open call at the warning point is completed and executed
exactly as a Snowball call is. Snowball text must parse as before."""
import json

from tb_agents import response_budget as B
from tb_agents.response_budget_output import is_datakit, parse_tool_text, process, text_message
from tb_agents.test_response_budget_pi import guard, post
from tb_agents.test_response_budget_warning_tool import HEAD, TAIL, TOOLS, backend, health

DK_HEAD = ('<|start_think|>I will write the file now.<|end_think|><tool_call>\n{"name": "write", "arguments": '
           '{"path": "/home/user/a.txt", "content": "hello wor')
DK_TAIL = 'ld\\nsecond line"}}\n</tool_call>'


def test_datakit_calls_parse():
    m = parse_tool_text(DK_HEAD + DK_TAIL, TOOLS)
    assert m['reasoning_content'] == 'I will write the file now.' and m['content'] == ''
    (call,) = m['tool_calls']
    assert call['function'] == {'name': 'write', 'arguments': {'path': '/home/user/a.txt',
                                                               'content': 'hello world\nsecond line'}}
    two = parse_tool_text('<|start_think|>r<|end_think|>Running both.<tool_call>\n{"name": "bash", "arguments": '
                          '{"command": "ls"}}\n</tool_call><tool_call>\n{"name": "bash", "arguments": "{\\"command\\": '
                          '\\"pwd\\", \\"timeout\\": 30}"}\n</tool_call>' + B.EOT, TOOLS)
    assert two['content'] == 'Running both.'
    assert [c['function']['arguments'] for c in two['tool_calls']] == [{'command': 'ls'},
                                                                       {'command': 'pwd', 'timeout': 30}]
    assert parse_tool_text(DK_HEAD, TOOLS) is None                           # still open
    assert parse_tool_text(DK_HEAD + DK_TAIL + ' trailing', TOOLS) is None   # unparsed text after the call
    assert parse_tool_text('<|start_think|>r<|end_think|><tool_call>\n{"name": "write", "arguments": {"path": "/a"}}'
                           '\n</tool_call>', TOOLS) is None                  # required parameter missing
    assert parse_tool_text('<|start_think|>r<|end_think|><tool_call>\n{not json}\n</tool_call>', TOOLS) is None


def test_snowball_text_is_untouched():
    assert not is_datakit(HEAD + TAIL)
    m = parse_tool_text(HEAD + TAIL, TOOLS)
    assert m['tool_calls'][0]['function']['arguments']['content'] == 'hello world\nsecond line'
    assert text_message('abc</think>\n\ndone', {}) == {'role': 'assistant', 'reasoning_content': 'abc', 'content': 'done'}
    assert text_message('plain', {}) == {'role': 'assistant', 'reasoning_content': '', 'content': 'plain'}


def test_datakit_cut_turns_keep_reasoning_apart():
    cut_in_reasoning = text_message('<|start_think|>Let me think about th', {'reasoning_content': 'x'})
    assert cut_in_reasoning == {'role': 'assistant', 'reasoning_content': 'Let me think about th', 'content': ''}
    cut_in_content = text_message('<|start_think|>plan<|end_think|>The answer is 4' + B.EOT, {})
    assert cut_in_content == {'role': 'assistant', 'reasoning_content': 'plan', 'content': 'The answer is 4'}
    # a cut call stays plain, non-executable content, as a cut XML call does
    cut_call = text_message(DK_HEAD, {})
    assert cut_call['reasoning_content'] == 'I will write the file now.' and cut_call['content'].startswith('<tool_call>')


def test_datakit_cut_turn_through_process():
    raw = '<|start_think|>' + 'thinking hard. ' * 300
    ids = list(range(1000))
    result = {'choices': [{'message': {'role': 'assistant', 'content': None, 'reasoning_content': raw[15:]},
                           'token_ids': ids, 'finish_reason': 'length'}], 'usage': {'completion_tokens': len(ids)}}
    out = process(result, raw, 50, 1000, {'token_counting': 'backend-native'})
    msg = result['choices'][0]['message']
    assert out['clipped'] and msg['content'] == '' and DK_START_NOT_IN(msg['reasoning_content'])


def DK_START_NOT_IN(text):
    return '<|start_think|>' not in text and text.startswith('thinking hard.')


def test_datakit_open_call_at_warning_is_completed_executed_and_free(tmp_path, monkeypatch):
    with backend(DK_HEAD, DK_TAIL) as (url, chats, completions, warn):
        budget = {'warn_after_response_tokens': warn, 'finish_response_tokens': 200,
                  'total_response_tokens': warn + 200, 'token_counting': 'backend-native'}
        with guard(url, tmp_path, budget, monkeypatch) as endpoint:
            messages = [{'role': 'user', 'content': 'Write the file.'}]
            r = post(endpoint, {'model': 'm', 'messages': messages, 'tools': TOOLS, 'max_tokens': 16384,
                                'temperature': 1.0, 'top_p': 0.95})
            c = r['choices'][0]
            assert c['finish_reason'] == 'tool_calls'
            (call,) = c['message']['tool_calls']
            assert json.loads(call['function']['arguments'])['content'] == 'hello world\nsecond line'
            (comp,) = completions
            assert comp['stop'] == ['</tool_call>'] and comp['skip_special_tokens'] is False
            assert health(endpoint)['used'] == warn                       # the completion is free


def test_snowball_completion_request_is_unchanged(tmp_path, monkeypatch):
    with backend(HEAD, TAIL) as (url, chats, completions, warn):
        budget = {'warn_after_response_tokens': warn, 'finish_response_tokens': 200,
                  'total_response_tokens': warn + 200, 'token_counting': 'backend-native'}
        with guard(url, tmp_path, budget, monkeypatch) as endpoint:
            post(endpoint, {'model': 'm', 'messages': [{'role': 'user', 'content': 'W'}], 'tools': TOOLS,
                            'max_tokens': 16384})
            (comp,) = completions
            assert 'skip_special_tokens' not in comp


def test_omniscience_strips_leaked_datakit_reasoning():
    from aa_evals.omniscience.client import split_response
    resp = {'choices': [{'message': {'content': 'draft<|end_think|>Paris', 'reasoning_content': 'first'},
                         'finish_reason': 'stop'}]}
    out = split_response(resp)
    assert out['answer'] == 'Paris' and out['reasoning'] == 'first\ndraft'
    leak = {'choices': [{'message': {'content': 'Paris<|start_think|>hmm<|end_think|>', 'reasoning_content': 'r'},
                         'finish_reason': 'stop'}]}
    assert split_response(leak)['answer'] == ''
    unclosed = {'choices': [{'message': {'content': 'Paris<|start_think|>hmm, maybe Lyon', 'reasoning_content': 'r'},
                             'finish_reason': 'length'}]}
    assert split_response(unclosed)['answer'] == 'Paris' and split_response(unclosed)['reasoning'] == 'r\nhmm, maybe Lyon'
    plain = {'choices': [{'message': {'content': 'Paris', 'reasoning_content': 'r'}, 'finish_reason': 'stop'}]}
    assert split_response(plain) == {'answer': 'Paris', 'reasoning': 'r', 'finish_reason': 'stop', 'usage': {}}
