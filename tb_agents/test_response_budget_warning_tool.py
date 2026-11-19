"""A response cut by the warning threshold inside an open tool call: the call is completed, executed, not charged,
and the notice follows its tool result. A cut in reasoning keeps CUT + notice."""
import contextlib
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import urllib.request

from tb_agents import response_budget as B
from tb_agents.response_budget_output import parse_tool_text
from tb_agents.response_budget_proxy import PER_RESPONSE_CAP
from tb_agents.test_response_budget_pi import guard, post

TOOLS = [{'type': 'function', 'function': {'name': 'write', 'description': 'Write a file',
          'parameters': {'type': 'object', 'required': ['path', 'content'],
                         'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}}}}},
         {'type': 'function', 'function': {'name': 'bash', 'description': 'Run a command',
          'parameters': {'type': 'object', 'required': ['command'],
                         'properties': {'command': {'type': 'string'}, 'timeout': {'type': 'number'}}}}}]
HEAD = 'I will write the file now.</think>\n\n<tool_call>\n<function=write>\n<parameter=path>\n/home/user/a.txt\n</parameter>\n<parameter=content>\nhello wor'
TAIL = 'ld\nsecond line\n</parameter>\n</function>\n</tool_call>'


def test_parse_tool_text():
    m = parse_tool_text(HEAD + TAIL, TOOLS)
    assert m['reasoning_content'] == 'I will write the file now.' and m['content'] == ''
    (call,) = m['tool_calls']
    assert call['function'] == {'name': 'write', 'arguments': {'path': '/home/user/a.txt',
                                                               'content': 'hello world\nsecond line'}}
    typed = parse_tool_text('x</think>\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n'
                            '<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>', TOOLS)
    assert typed['tool_calls'][0]['function']['arguments'] == {'command': 'ls', 'timeout': 30}
    assert parse_tool_text(HEAD, TOOLS) is None                          # still open
    assert parse_tool_text(HEAD + TAIL + ' trailing', TOOLS) is None     # unparsed text after the call
    missing = 'x</think>\n<tool_call>\n<function=write>\n<parameter=path>\n/a\n</parameter>\n</function>\n</tool_call>'
    assert parse_tool_text(missing, TOOLS) is None                       # required parameter missing


@contextlib.contextmanager
def backend(head, tail):
    chats, completions = [], []
    head_ids = B.encode(head)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/detokenize'):
                result = {'prompt': B.tokenizer().decode(payload['tokens'], skip_special_tokens=False)}
            elif self.path.endswith('/tokenize'):
                result = {'tokens': [101, 102, 103]}
            elif self.path.endswith('/v1/completions'):
                completions.append(copy.deepcopy(payload))
                ids = B.encode(tail)
                result = {'choices': [{'index': 0, 'text': tail, 'token_ids': ids, 'finish_reason': 'stop',
                                       'stop_reason': '</tool_call>'}], 'usage': {'completion_tokens': len(ids)}}
            else:
                chats.append(copy.deepcopy(payload))
                if len(chats) == 1:
                    ids = head_ids[:payload['max_tokens']]
                    result = {'id': 'c1', 'object': 'chat.completion', 'created': 1, 'model': 'm',
                              'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None,
                                           'reasoning_content': 'I will write the file now.'},
                                           'finish_reason': 'length', 'token_ids': ids}],
                              'usage': {'completion_tokens': len(ids)}}
                else:
                    ids = B.encode('Done.' + B.EOT)
                    result = {'id': 'c2', 'object': 'chat.completion', 'created': 2, 'model': 'm',
                              'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'Done.'},
                                           'finish_reason': 'stop', 'token_ids': ids}],
                              'usage': {'completion_tokens': len(ids)}}
            data = json.dumps(result).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', chats, completions, len(head_ids)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def health(endpoint):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(endpoint + '/__budget_health', timeout=10) as r:
        return json.loads(r.read())


def test_open_call_at_warning_is_completed_executed_and_free(tmp_path, monkeypatch):
    with backend(HEAD, TAIL) as (url, chats, completions, warn):
        budget = {'warn_after_response_tokens': warn, 'finish_response_tokens': 200,
                  'total_response_tokens': warn + 200, 'token_counting': 'backend-native'}
        with guard(url, tmp_path, budget, monkeypatch) as endpoint:
            messages = [{'role': 'user', 'content': 'Write the file.'}]
            r = post(endpoint, {'model': 'm', 'messages': messages, 'tools': TOOLS, 'max_tokens': 16384,
                                'temperature': 0.7, 'top_p': 0.9})
            c = r['choices'][0]
            assert c['finish_reason'] == 'tool_calls'
            (call,) = c['message']['tool_calls']
            assert call['function']['name'] == 'write'
            assert json.loads(call['function']['arguments'])['content'] == 'hello world\nsecond line'
            (comp,) = completions
            assert comp['prompt'][:3] == [101, 102, 103] and comp['stop'] == ['</tool_call>']
            assert comp['max_tokens'] == PER_RESPONSE_CAP - warn          # bounded by the per-response cap
            assert comp['temperature'] == 0.7 and comp['top_p'] == 0.9
            assert health(endpoint)['used'] == warn                       # the completion is free
            # Pi executes the call and sends the tool result: the notice follows it, with the full allowance.
            nxt = messages + [c['message'], {'role': 'tool', 'tool_call_id': call['id'], 'content': 'written'}]
            post(endpoint, {'model': 'm', 'messages': nxt, 'tools': TOOLS, 'max_tokens': 16384})
            last = chats[1]['messages'][-1]
            assert last['role'] == 'tool' and B.notice(200) in json.dumps(last['content'])
            assert B.CUT not in json.dumps(chats[1]['messages']) and B.CONTINUE not in json.dumps(chats[1]['messages'])
        wire = [json.loads(l) for l in (tmp_path / 'response-budget-wire.jsonl').read_text().splitlines()]
        acc = wire[0]['generation_accounting']['warning_tool_completion']
        assert acc['executed'] and not acc['charged'] and acc['extra_native_tokens'] == len(B.encode(TAIL))


def test_cut_in_reasoning_is_not_completed(tmp_path, monkeypatch):
    head = 'Thinking about the task. ' * 40
    with backend(head, TAIL) as (url, chats, completions, n):
        warn = n // 2
        budget = {'warn_after_response_tokens': warn, 'finish_response_tokens': 200,
                  'total_response_tokens': warn + 200, 'token_counting': 'backend-native'}
        with guard(url, tmp_path, budget, monkeypatch) as endpoint:
            r = post(endpoint, {'model': 'm', 'messages': [{'role': 'user', 'content': 'Q'}], 'tools': TOOLS,
                                'max_tokens': 16384})
            assert r['choices'][0]['finish_reason'] == 'length' and not completions
            assert health(endpoint)['used'] == warn


def test_open_call_cut_by_the_context_clamp_is_completed_and_charged(tmp_path, monkeypatch):
    with backend(HEAD, TAIL) as (url, chats, completions, n):
        # a large response budget (no warning in reach); Pi's request carries the guard's clamp as max_tokens
        budget = {'warn_after_response_tokens': 10 ** 6, 'finish_response_tokens': 200,
                  'total_response_tokens': 10 ** 6 + 200, 'token_counting': 'backend-native'}
        with guard(url, tmp_path, budget, monkeypatch) as endpoint:
            r = post(endpoint, {'model': 'm', 'messages': [{'role': 'user', 'content': 'Write the file.'}],
                                'tools': TOOLS, 'max_tokens': n})
            assert r['choices'][0]['finish_reason'] == 'tool_calls' and len(completions) == 1
            assert completions[0]['max_tokens'] <= 12288
            assert health(endpoint)['used'] == n + len(B.encode(TAIL))     # charged in full
        wire = [json.loads(l) for l in (tmp_path / 'response-budget-wire.jsonl').read_text().splitlines()]
        acc = wire[0]['generation_accounting']['clamp_tool_completion']
        assert acc['executed'] and acc['charged']
