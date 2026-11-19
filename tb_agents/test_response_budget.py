import json
from tb_agents import response_budget as B
from tb_agents.response_budget_proxy import sse


def mock_generation(message, finish, requested):
    """A vLLM-like response with raw generation, before parsed-message loss."""
    ids = B.encode(B.body(message))
    ids = ids[:requested]
    msg, _, clipped = B.clip(message, requested)
    return {'id': 'mock', 'created': 1, 'model': 'test', 'object': 'chat.completion',
            'choices': [{'index': 0, 'message': msg,
                         'finish_reason': 'length' if clipped else finish, 'token_ids': ids}],
            'usage': {'completion_tokens': len(ids)}}


def test_partial_reasoning_keeps_exact_tokens():
    m = {'role': 'assistant', 'reasoning_content': 'Consider the possible choices. ' * 200, 'content': '42'}
    cut, ids, clipped = B.clip(m, 101)
    assert clipped and len(ids) == 101
    assert ids == B.encode(B.body(m))[:101]
    assert not cut['content'] and not cut.get('tool_calls')


def test_partial_call_is_not_executed():
    m = {'role': 'assistant', 'reasoning_content': 'Act.', 'content': '',
         'tool_calls': [{'id': 'call_x', 'type': 'function', 'function': {'name': 'bash',
                         'arguments': {'command': 'echo ' + 'hello ' * 100}}}]}
    cap = B.length(m) - 10
    cut, ids, clipped = B.clip(m, cap)
    assert clipped and not cut.get('tool_calls')
    assert '<tool_call>' in cut['content']
    assert ids == B.encode(B.body(m))[:cap]


def test_natural_stop_and_eot_kept():
    m = {'role': 'assistant', 'reasoning_content': '', 'content': '42'}
    cut, ids, clipped = B.clip(m, 1000)
    assert not clipped and cut == m
    assert ids[-1] == 128009


def test_sse_tool_index():
    m = {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'x', 'type': 'function',
         'function': {'name': 'bash', 'arguments': '{}'}}]}
    data = sse({'id': 'x', 'created': 1, 'model': 'x', 'usage': {},
                'choices': [{'message': m, 'finish_reason': 'tool_calls'}]})
    d = json.loads(data.splitlines()[0].removeprefix(b'data: '))
    assert d['choices'][0]['delta']['tool_calls'][0]['index'] == 0


def test_single_task_interrupt_and_finish():
    from tb_agents.response_budget_single import run
    budget={'warn_after_response_tokens':100,'finish_response_tokens':50,'total_response_tokens':150}
    requests=[]
    def call(payload):
        requests.append(json.loads(json.dumps(payload)))
        m=({'role':'assistant','reasoning_content':'Thinking carefully. '*200,'content':''} if len(requests)==1
           else {'role':'assistant','reasoning_content':'','content':'42'})
        return {'choices':[{'message':m,'finish_reason':'length' if len(requests)==1 else 'stop'}], 'usage':{}}
    r=run(call,{'messages':[{'role':'user','content':'What is the answer?'}],'max_tokens':1000},budget)
    assert len(requests)==2
    assert requests[0]['max_tokens']==100 and requests[1]['max_tokens']==45
    assert B.notice(50) in requests[1]['messages'][-1]['content']
    assert r['choices'][0]['message']['content'].strip()=='42'
    assert r['choices'][0]['finish_reason']=='stop'


def test_notice_extra_only_for_its_benchmark():
    base = {'warn_after_response_tokens': 100, 'finish_response_tokens': 50, 'total_response_tokens': 150}
    omni = B.with_benchmark(base, 'omniscience')
    assert omni['notice_extra'] == B.NOTICE_EXTRA['omniscience'] and 'notice_extra' not in base
    assert 'notice_extra' not in B.with_benchmark(base, 'gdppdf')
    assert B.with_benchmark(None, 'omniscience') is None
    assert B.with_benchmark(dict(base, notice_extra='X'), 'omniscience')['notice_extra'] == 'X'
    text = B.notice(50, extra=omni['notice_extra'])
    assert text.startswith(B.notice(50)) and text.endswith(B.NOTICE_EXTRA['omniscience'])
    assert 'aim for at most 12 words' in text           # the word target stays
    fin = B.notice(4, final=True, extra=omni['notice_extra'])
    assert B.NOTICE_EXTRA['omniscience'] in fin and fin.endswith('Do not start another repair or expand the work.')
    assert B.notice(50, extra=None) == B.notice(50)      # other benchmarks: unchanged text


def test_single_task_carries_omniscience_sentence():
    from tb_agents.response_budget_single import run
    budget = B.with_benchmark({'warn_after_response_tokens': 100, 'finish_response_tokens': 50,
                               'total_response_tokens': 150}, 'omniscience')
    requests = []
    def call(payload):
        requests.append(json.loads(json.dumps(payload)))
        m = ({'role': 'assistant', 'reasoning_content': 'Thinking carefully. ' * 200, 'content': ''} if len(requests) == 1
             else {'role': 'assistant', 'reasoning_content': '', 'content': 'I do not know.'})
        return {'choices': [{'message': m, 'finish_reason': 'length' if len(requests) == 1 else 'stop'}], 'usage': {}}
    r = run(call, {'messages': [{'role': 'user', 'content': 'Q?'}], 'max_tokens': 1000}, budget)
    assert B.notice(50, extra=B.NOTICE_EXTRA['omniscience']) in requests[1]['messages'][-1]['content']
    assert r['_response_budget']['budget']['notice_extra'] == B.NOTICE_EXTRA['omniscience']   # recorded per attempt
    assert r['_response_budget']['used']<=150


def test_proxy_counts_tool_turns_refreshes_sticky_notice_and_deduplicates_retry(tmp_path, monkeypatch):
    import copy
    import http.server
    import threading
    import urllib.request
    from tb_agents.response_budget_proxy import start
    seen=[]
    replies=[{'role':'assistant','reasoning_content':'Analyze this carefully. '*300,'content':''},
             {'role':'assistant','content':'','tool_calls':[{'id':'t1','type':'function','function':{
                 'name':'bash','arguments':'{"command":"true"}'}}]},
             {'role':'assistant','content':'Done.'}]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/detokenize'):
                result={'prompt':B.tokenizer().decode(payload['tokens'],skip_special_tokens=False)}
            elif self.path.endswith('/tokenize'):
                seen.append((self.path,copy.deepcopy(payload)))
                result={'count':1}
            else:
                seen.append((self.path,copy.deepcopy(payload)))
                n=sum(p.endswith('/chat/completions')for p,_ in seen)-1
                result=mock_generation(replies[n],['length','tool_calls','stop'][n],payload['max_tokens'])
            data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('PI_RESPONSE_BUDGET',json.dumps({'warn_after_response_tokens':100,'finish_response_tokens':100,'total_response_tokens':200}))
    child,url=start(f'http://127.0.0.1:{server.server_port}',tmp_path)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def post(path,payload):
        req=urllib.request.Request(url+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=20) as r:return json.loads(r.read())
    try:
        msgs=[{'role':'user','content':'Complete the task.'}]
        r1=post('/v1/chat/completions',{'model':'test','messages':msgs,'max_tokens':16384})
        msgs += [r1['choices'][0]['message'],{'role':'user','content':B.CONTINUE}]
        post('/tokenize',{'model':'test','messages':msgs})
        assert B.notice(100) in seen[-1][1]['messages'][-1]['content']
        p2={'model':'test','messages':copy.deepcopy(msgs),'max_tokens':16384}
        r2=post('/v1/chat/completions',p2)
        assert post('/v1/chat/completions',p2)==r2
        msgs += [r2['choices'][0]['message'],{'role':'tool','tool_call_id':'t1','content':'ok'}]
        post('/v1/chat/completions',{'model':'test','messages':msgs,'max_tokens':16384})
        chats=[p for path,p in seen if path.endswith('/chat/completions')]
        assert len(chats)==3
        left=100-B.length(r2['choices'][0]['message'])
        assert B.notice(left) in chats[-1]['messages'][-1]['content']
        assert B.notice(100) in chats[-1]['messages'][2]['content']
        logs=[json.loads(l)for l in (tmp_path/'response-budget-wire.jsonl').read_text().splitlines()]
        assert logs[0]['used_after']==100 and logs[-1]['used_after']<=200
    finally:
        child.terminate();child.wait(timeout=10);server.shutdown();thread.join()


def test_reserved_finalization_can_finish_after_another_length_cut():
    from tb_agents.response_budget_single import run
    requests=[]
    def call(p):
        requests.append(json.loads(json.dumps(p)))
        n=len(requests)
        m=({'role':'assistant','reasoning_content':'Think carefully. '*1000,'content':''}if n==1 else
           {'role':'assistant','content':'Long answer. '*1000}if n==2 else {'role':'assistant','content':'42'})
        return {'choices':[{'message':m,'finish_reason':'length'if n<3 else'stop'}],'usage':{}}
    budget={'warn_after_response_tokens':1000,'finish_response_tokens':100,'total_response_tokens':1100}
    r=run(call,{'messages':[{'role':'user','content':'Answer.'}],'max_tokens':10000},budget)
    assert [p['max_tokens']for p in requests]==[1000,90,10]
    assert 'reserved finalization allowance' in requests[-1]['messages'][-1]['content']
    assert r['choices'][0]['finish_reason']=='stop'
    assert r['_response_budget']['used']<=1100


def test_full_finishing_allowance_restored_after_one_token_boundary_request():
    from tb_agents.response_budget_single import run
    first={'role':'assistant','reasoning_content':'An unfinished analysis. '*30,'content':''}
    warn=B.length(first)+1
    budget={'warn_after_response_tokens':warn,'finish_response_tokens':100,'total_response_tokens':warn+100}
    requests=[]
    def call(p):
        requests.append(json.loads(json.dumps(p)))
        n=len(requests)
        m=first if n==1 else {'role':'assistant','reasoning_content':'More analysis. '*100,'content':''}if n==2 else {'role':'assistant','content':'42'}
        return {'choices':[{'message':m,'finish_reason':'length'if n<3 else'stop'}],'usage':{}}
    r=run(call,{'messages':[{'role':'user','content':'Answer.'}],'max_tokens':10000},budget)
    assert [p['max_tokens']for p in requests]==[warn,1,90]
    assert r['choices'][0]['finish_reason']=='stop'
    assert r['choices'][0]['message']['content'].strip()=='42'


def test_proxy_preserves_native_pi_max_completion_tokens_clamp(tmp_path,monkeypatch):
    import http.server
    import threading
    import urllib.request
    from tb_agents.response_budget_proxy import start
    seen=[]
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path.endswith('/detokenize'):
                result={'prompt':B.tokenizer().decode(payload['tokens'],skip_special_tokens=False)}
            else:
                seen.append(payload)
                result=mock_generation({'role':'assistant','content':'42'},'stop',payload['max_tokens'])
            data=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('PI_RESPONSE_BUDGET',json.dumps({'warn_after_response_tokens':100,'finish_response_tokens':100,'total_response_tokens':200}))
    child,url=start(f'http://127.0.0.1:{server.server_port}',tmp_path)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        payload={'model':'test','messages':[{'role':'user','content':'Answer.'}],'max_completion_tokens':25}
        req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=20)as r:assert json.loads(r.read())['choices'][0]['message']['content']=='42'
        assert seen[0]['max_tokens']==25
        assert 'max_completion_tokens'not in seen[0]
    finally:
        child.terminate();child.wait(timeout=10);server.shutdown();thread.join()


def test_single_request_history_preserves_reasoning_for_native_vllm_parser():
    from tb_agents.response_budget_single import request_payload
    payload={'messages':[{'role':'user','content':'Answer.'},
                         {'role':'assistant','reasoning_content':'Important unfinished reasoning.','content':''}]}
    sent=request_payload(payload)
    assert sent['messages'][1]['reasoning']=='Important unfinished reasoning.'
    # Native vLLM _parse_chat_message_content maps this alias to both template fields.
    normalized={'role':'assistant','reasoning_content':sent['messages'][1]['reasoning'],'content':''}
    assert B.body(normalized)==B.body(payload['messages'][1])
    assert 'reasoning'not in payload['messages'][1]


def test_warned_cut_is_not_followed_by_please_continue():
    from tb_agents.response_budget_single import run
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 50, 'total_response_tokens': 150}
    requests = []
    def call(payload):
        requests.append(json.loads(json.dumps(payload)))
        m = ({'role': 'assistant', 'reasoning_content': 'Thinking carefully. ' * 200, 'content': ''} if len(requests) == 1
             else {'role': 'assistant', 'reasoning_content': '', 'content': '42'})
        return {'choices': [{'message': m, 'finish_reason': 'length' if len(requests) == 1 else 'stop'}], 'usage': {}}
    run(call, {'messages': [{'role': 'user', 'content': 'Q?'}], 'max_tokens': 1000}, budget)
    last = requests[1]['messages'][-1]['content']
    assert last.startswith(B.CUT) and B.notice(50) in last and B.CONTINUE not in last


def test_replace_continue_only_touches_budget_notices():
    note = B.notice(500)
    pi_list = {'role': 'user', 'content': [{'type': 'text', 'text': B.CONTINUE}, {'type': 'text', 'text': note}]}
    as_str = {'role': 'user', 'content': B.CONTINUE + '\n\n' + note}
    plain = {'role': 'user', 'content': [{'type': 'text', 'text': B.CONTINUE}]}          # 16k cap, no notice
    tool = {'role': 'tool', 'content': 'ok\n\n' + note}
    other = {'role': 'user', 'content': B.CONTINUE + '\n\n[Harness notice] context is almost full'}
    msgs = [plain, pi_list, as_str, tool, other]
    out = B.replace_continue(msgs)
    assert out[0] == plain and out[3] == tool and out[4] == other
    assert out[1]['content'][0]['text'] == B.CUT and out[1]['content'][1]['text'] == note
    assert out[2]['content'] == B.CUT + '\n\n' + note
    assert msgs[1]['content'][0]['text'] == B.CONTINUE                                      # input not modified


def test_measurement_backstop_does_not_cut_at_the_stated_allowance():
    b = {'warn_after_response_tokens': 100, 'finish_response_tokens': 50, 'total_response_tokens': 150}
    assert B.hard_limit(b) == 150 and B.output_cap(120, b) == 150 - 120 - B.reserve(b)
    m = dict(b, measure_backstop_tokens=1000)
    assert B.hard_limit(m) == 1100
    assert B.output_cap(40, m) == 60                      # before the warning: unchanged
    assert B.output_cap(120, m) == 980 and B.output_cap(400, m) == 700   # past the stated total: still generating
    assert B.output_cap(1100, m) == 1


def test_near_window_warning_binds_first_on_long_items():
    b = {'warn_after_response_tokens': 130850, 'finish_response_tokens': 32000, 'total_response_tokens': 162850,
         'near_window': True}
    it = B.item_budget(b, 130000)                       # HLE: Super's p95 sits past the window
    assert it['near_window_bound'] and it['warn_after_response_tokens'] == 130000 - 512 - 32000
    assert it['total_response_tokens'] == 130000 - 512
    assert it['total_response_tokens'] - it['warn_after_response_tokens'] == it['finish_response_tokens']
    short = B.item_budget({'warn_after_response_tokens': 20192, 'finish_response_tokens': 10000,
                           'total_response_tokens': 30192, 'near_window': True}, 129000)
    assert not short['near_window_bound'] and short['warn_after_response_tokens'] == 20192   # the suite cap binds
    assert B.item_budget(dict(b, near_window=False), 130000)['warn_after_response_tokens'] == 130850   # flag off


def test_async_chat_adapter_near_window_and_cut_marker():
    import asyncio
    from tb_agents.response_budget_single import run_async_chat
    budget = {'warn_after_response_tokens': 10**6, 'finish_response_tokens': 50, 'total_response_tokens': 10**6 + 50,
              'near_window': True}
    seen = []

    async def call(payload):
        seen.append(json.loads(json.dumps(payload)))
        m = ({'role': 'assistant', 'reasoning_content': 'Thinking carefully. ' * 400, 'content': ''} if len(seen) == 1
             else {'role': 'assistant', 'reasoning_content': '', 'content': '42'})
        return {'choices': [{'message': m, 'finish_reason': 'length' if len(seen) == 1 else 'stop'}], 'usage': {}}
    r = asyncio.run(run_async_chat(call, {'messages': [{'role': 'user', 'content': 'Q?'}], 'max_tokens': 900}, budget))
    warn = 900 - B.NEAR_WINDOW_MARGIN - 50
    assert seen[0]['max_tokens'] == warn                       # the near-window point binds, not the 1M suite cap
    last = seen[1]['messages'][-1]['content']
    assert last.startswith(B.CUT) and '[Harness notice]' in last and B.CONTINUE not in last
    assert r['choices'][0]['message']['content'].strip() == '42' and r['_response_budget']['budget']['near_window_bound']


def test_super_counting_measures_text_in_super_tokens():
    m = {'role': 'assistant', 'reasoning_content': 'Consider the possible choices carefully. ' * 120, 'content': '42'}
    full = len(B.super_ids(B.body(m)))
    same, _, clipped, n = B.super_clip(m, full + 5)
    assert not clipped and n == full and same['content'] == '42'
    cut, ids, clipped, n = B.super_clip(m, 100)
    assert clipped and n == 100 and not cut['content'] and len(B.super_ids(cut['reasoning_content'])) <= 101
    from tb_agents.response_budget_single import run
    budget = {'warn_after_response_tokens': 100, 'finish_response_tokens': 50, 'total_response_tokens': 150,
              'token_counting': 'super'}
    calls = []
    def call(payload):
        calls.append(json.loads(json.dumps(payload)))
        msg = m if len(calls) == 1 else {'role': 'assistant', 'reasoning_content': '', 'content': '42'}
        return {'choices': [{'message': msg, 'finish_reason': 'length' if len(calls) == 1 else 'stop'}], 'usage': {}}
    r = run(call, {'messages': [{'role': 'user', 'content': 'Q?'}], 'max_tokens': 1000}, budget)
    assert r['_response_budget']['used'] == 100 + len(B.super_ids(B.body({'role': 'assistant', 'reasoning_content': '', 'content': '42'})))


def _windowed_backend(window, p0, ratio, answer_after_notice=True):
    """A vLLM-like async backend whose own tokens are `ratio` x Super tokens of every text: it rejects a request whose
    prompt + max_tokens exceeds `window` (HTTP 400), fills max_tokens with reasoning (finish length), and answers once
    the final notice is in the prompt. Reports prompt/completion usage in its own tokens."""
    def toks(text):
        return int(round(ratio * len(B.super_ids(text))))
    seen = []

    async def call(payload):
        prompt = p0 + sum(toks(B.body(m) if m['role'] == 'assistant' else m['content']) + 8
                         for m in payload['messages'][1:])
        seen.append({'prompt': prompt, 'max_tokens': payload['max_tokens'], 'last': payload['messages'][-1]})
        if prompt + payload['max_tokens'] > window:
            raise RuntimeError(f'HTTP 400: {prompt} + {payload["max_tokens"]} > {window}')
        last = payload['messages'][-1]['content']
        if answer_after_notice and payload['messages'][-1]['role'] == 'user' and '[Harness notice]' in last:
            m, finish = {'role': 'assistant', 'reasoning_content': '', 'content': '42'}, 'stop'
        else:
            n = max(1, int(payload['max_tokens'] / ratio) - 8)   # ' think' is one Super token: fills max_tokens
            m, finish = {'role': 'assistant', 'reasoning_content': ' think' * n, 'content': ''}, 'length'
        return {'choices': [{'message': m, 'finish_reason': finish}],
                'usage': {'prompt_tokens': prompt, 'completion_tokens': toks(B.body(m))}}
    return call, seen


HLE_BUDGET = {'warn_after_response_tokens': 130850, 'finish_response_tokens': 32000, 'total_response_tokens': 162850,
              'token_counting': 'super', 'near_window': True}


def test_near_window_holds_in_model_tokens_when_they_outgrow_super():
    """m5l-18k-1p0b HLE (11-15): Snowball-tokenizer output ran ahead of its Super count, so the continuation prompt
    outgrew the window and every such item ended in an HTTP 400 (an infra error) instead of the final notice."""
    import asyncio
    from tb_agents.response_budget_single import run_async_chat
    window, p0 = 131072, 300
    call, seen = _windowed_backend(window, p0, ratio=1.5)
    r = asyncio.run(run_async_chat(call, {'messages': [{'role': 'user', 'content': 'Q?'}],
                                          'max_tokens': window - p0 - 64}, HLE_BUDGET))
    assert all(s['prompt'] + s['max_tokens'] <= window for s in seen)
    assert r['choices'][0]['message']['content'].strip() == '42'
    w = r['_response_budget']['window']
    assert w['forced_warning'] and w['grown'] <= w['room']
    notice = next(s for s in seen if '[Harness notice]' in (s['last'].get('content') or ''))
    assert notice['last']['content'].startswith(B.CUT)
    # the warning came with about the allowance of the model's room left (not past the window)
    assert window - notice['prompt'] >= HLE_BUDGET['finish_response_tokens'] - 2 * B.NEAR_WINDOW_MARGIN


def test_near_window_room_spent_ends_as_length_stop():
    import asyncio
    from tb_agents.response_budget_single import run_async_chat
    window, p0 = 131072, 300
    call, seen = _windowed_backend(window, p0, ratio=1.5, answer_after_notice=False)
    r = asyncio.run(run_async_chat(call, {'messages': [{'role': 'user', 'content': 'Q?'}],
                                          'max_tokens': window - p0 - 64}, HLE_BUDGET))
    assert all(s['prompt'] + s['max_tokens'] <= window for s in seen)
    assert r['choices'][0]['finish_reason'] == 'length' and not r['choices'][0]['message']['content'].strip()


def test_near_window_guard_barely_moves_matched_tokens():
    """When the model's tokens are Super's (ratio 1), the guard changes the call sequence by at most the template
    tokens of the re-sent turns: same calls, same warning call, max_tokens within NEAR_WINDOW_MARGIN."""
    import asyncio
    from tb_agents.response_budget_single import run_async_chat
    window, p0 = 131072, 300
    call, seen = _windowed_backend(window, p0, ratio=1.0)
    asyncio.run(run_async_chat(call, {'messages': [{'role': 'user', 'content': 'Q?'}],
                                      'max_tokens': window - p0 - 64}, HLE_BUDGET))

    async def no_usage(payload):   # the same backend with usage hidden: the guard is off (the previous behaviour)
        d = await call2(payload)
        return dict(d, usage={})
    call2, ref = _windowed_backend(window, p0, ratio=1.0)
    asyncio.run(run_async_chat(no_usage, {'messages': [{'role': 'user', 'content': 'Q?'}],
                                          'max_tokens': window - p0 - 64}, HLE_BUDGET))
    assert len(seen) == len(ref)
    assert [('[Harness notice]' in (s['last'].get('content') or '')) for s in seen] == \
        [('[Harness notice]' in (s['last'].get('content') or '')) for s in ref]
    assert all(abs(a['max_tokens'] - b['max_tokens']) <= B.NEAR_WINDOW_MARGIN for a, b in zip(seen, ref))
