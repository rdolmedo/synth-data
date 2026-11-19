"""sticky_proxy / pi_cont16k options as synth campaigns combine them (task-synthesis), and the eval profile's
defaults (#480: one launcher, PI_CONT16K_PROFILE)."""
import asyncio
import json
from pathlib import Path
import socket
import subprocess
import sys

import aiohttp
from aiohttp import web
import pytest

from sticky_proxy import sse
from synth.wire import load as load_wire


async def run_proxy(tmp_path, extra, payloads, stream=True):
    requests = []

    async def generate(req):
        requests.append(await req.json())
        message = {'role': 'assistant', 'content': f'sample-{len(requests)}', 'reasoning_content': 'thought'}
        if stream:
            body = sse({'choices': [{'index': 0, 'delta': message, 'finish_reason': None}]})
            body += sse({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
            body += sse({'choices': [], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}) + b'data: [DONE]\n\n'
            return web.Response(body=body, content_type='text/event-stream')
        return web.json_response({'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]})

    async def tokenize(req):
        return web.json_response({'count': 1000})

    async def models(req):
        return web.json_response({'data': []})

    app = web.Application()
    app.router.add_post('/v1/chat/completions', generate)
    app.router.add_post('/tokenize', tokenize)
    app.router.add_get('/v1/models', models)
    runner = web.AppRunner(app)
    await runner.setup()
    backend = socket.socket()
    backend.bind(('127.0.0.1', 0))
    backend.listen(128)
    await web.SockSite(runner, backend).start()
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(Path(__file__).with_name('sticky_proxy.py')),
        '--port', str(port), '--listen-fd', str(listener.fileno()),
        '--backends', f'http://127.0.0.1:{backend.getsockname()[1]}',
        '--turn-max-tokens', '512', '--turn-retries', '0', '--turn-continue', '0',
        '--hold', '0', '--shutdown-timeout', '0', '--no-load-poll', *extra,
        pass_fds=(listener.fileno(),), stdout=asyncio.subprocess.DEVNULL)
    listener.close()
    try:
        async with aiohttp.ClientSession() as client:
            for _ in range(100):
                try:
                    async with client.get(f'http://127.0.0.1:{port}/v1/models') as r:
                        await r.read()
                    break
                except aiohttp.ClientConnectionError:
                    await asyncio.sleep(.05)
            for payload in payloads:
                async with client.post(f'http://127.0.0.1:{port}/v1/chat/completions',
                                       json={**payload, 'stream': stream}) as r:
                    assert r.status == 200
                    await r.read()
        return requests
    finally:
        process.terminate()
        await asyncio.wait_for(process.wait(), 5)
        await runner.cleanup()


HISTORY = [{'role': 'user', 'content': 'task'},
           {'role': 'assistant', 'content': '', 'reasoning_content': 'unfinished'},
           {'role': 'user', 'content': 'Thinking exceeded the maximum thinking budget, please continue.'}]


def test_request_overrides_template_kwargs_mirror_and_wire_log(tmp_path):
    wire = tmp_path / 'wire.jsonl'
    extra = ['--request-overrides', json.dumps({'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'max', 'seed': 100}),
             '--request-overrides-mode', 'update', '--chat-template-kwargs', json.dumps({'clear_thinking': False}), '--mirror-reasoning-content',
             '--wire-log', str(wire)]
    payloads = [{'model': 'm', 'messages': HISTORY[:1], 'max_tokens': 16384, 'seed': 7,
                 'chat_template_kwargs': {'keep': 1}},
                {'model': 'm', 'messages': HISTORY, 'max_tokens': 16384}]
    requests = asyncio.run(run_proxy(tmp_path, extra, payloads))
    assert [r['seed'] for r in requests] == [100, 100], 'one per-attempt seed on every request'
    assert all(r['temperature'] == 1.0 and r['reasoning_effort'] == 'max' for r in requests)
    assert requests[0]['chat_template_kwargs'] == {'keep': 1, 'clear_thinking': False}
    assert requests[1]['messages'][1]['reasoning'] == 'unfinished'
    assert all(r['max_tokens'] == 512 for r in requests)
    calls = load_wire(wire)
    assert [c['body'] for c in calls] == requests
    assert 'sample-2' in bytes(calls[1]['response']).decode()


def test_no_optional_pinning_by_default(tmp_path):
    payload = {'model': 'm', 'messages': HISTORY, 'max_tokens': 16384, 'seed': 5}
    requests = asyncio.run(run_proxy(tmp_path, [], [payload]))
    assert requests[0]['seed'] == 5 and 'temperature' not in requests[0]
    assert 'reasoning' not in requests[0]['messages'][1]
    assert requests[0]['max_tokens'] == 512


@pytest.mark.parametrize('flag', ['--no-load-poll', '--disable-load-polling',
                                  '--skip-single-backend-load-poll', '--single-backend-no-poll'])
def test_every_teacher_poll_flag_requires_one_backend(flag):
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('sticky_proxy.py')), '--port', '0',
                             '--backends', 'http://127.0.0.1:1,http://127.0.0.1:2', flag],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and 'requires exactly one backend' in result.stderr


def test_request_overrides_merge_mode_keeps_client_template_kwargs(tmp_path):
    """eval profile (aa_evals M3.2): overrides on the per-turn path, chat_template_kwargs merged key by key."""
    extra = ['--request-overrides', json.dumps({'temperature': 0.6, 'chat_template_kwargs': {'enable_thinking': True}})]
    payloads = [{'model': 'm', 'messages': HISTORY[:1], 'max_tokens': 16384, 'chat_template_kwargs': {'keep': 1}}]
    requests = asyncio.run(run_proxy(tmp_path, extra, payloads))
    assert requests[0]['chat_template_kwargs'] == {'keep': 1, 'enable_thinking': True}
    assert requests[0]['temperature'] == 0.6
    requests = asyncio.run(run_proxy(tmp_path, extra + ['--request-overrides-mode', 'update'], payloads))
    assert requests[0]['chat_template_kwargs'] == {'enable_thinking': True}   # synth: shallow update


def test_pi_cont16k_eval_profile_defaults(tmp_path):
    """Unset PI_CONT16K_PROFILE = eval: aa-evals' capture files, gate identity headers, routing polls."""
    from gen.pi_cont16k import proxy_options
    args, record = proxy_options(tmp_path, {'PI_GATE_CLIENT': 'tb21'}, conversation_id='c1')
    assert args == ['--dump-dir', str(tmp_path / 'first-request'), '--dump-max', '1',
                    '--response-dir', str(tmp_path / 'wire-responses'),
                    '--request-events', str(tmp_path / 'wire-requests.jsonl'),
                    '--conversation-id', 'c1', '--gate-client', 'tb21']
    assert record['gate_conversation_id'] == 'c1' and record['routing_load_polling'] is True
    args, record = proxy_options(tmp_path, {'PI_REQUEST_OVERRIDES': '{"reasoning_effort": "high"}',
                                            'PI_SAMPLING_SEED': '7', 'PI_KV_GATE': '0.8,0.7', 'PI_DP_RANK': '1',
                                            'PI_PROXY_CAPTURE_FILES': '0', 'PI_GATE_HEADERS': '0'})
    assert args == ['--sampling-seed', '7', '--request-overrides', '{"reasoning_effort": "high"}',
                    '--dp-rank', '1', '--kv-gate', '0.8,0.7']
    assert record['request_overrides_mode'] == 'merge' and record['reasoning_effort'] == 'high'


def test_pi_cont16k_eval_q35_and_pi_eval_flags_match_aa_evals_launcher(tmp_path):
    """The complete proxy flags for an eval_q35-style and a pi_eval-style attempt environment are the ones aa-evals'
    gen/pi_cont16k.py passed (its campaign_args + capture flags; argument order aside)."""
    from gen.pi_cont16k import proxy_options
    eval_q35 = {'PI_SAMPLING_SEED': '123', 'PI_KV_GATE': '0.80,0.72', 'PI_KV_GATE_FILE': '/r/serve/kv.json',
                'PI_GATE_KEY': 'task__a0__r0', 'PI_HTTP_FORCE_CLOSE': '1', 'PI_DP_RANK': '1',
                'PI_REASONING_EFFORT': 'high', 'PI_GUARD_BOUND_SHORTCUT': '1'}
    pi_eval = {**eval_q35, 'PI_GATE_CLIENT': 'tb21', 'PI_INLINE_REASONING_HISTORY': '1',
               'PI_REQUEST_OVERRIDES': '{"temperature": 1.0, "chat_template_kwargs": {"enable_thinking": true}}'}

    def aa_evals_args(env):   # aa-evals' gen/pi_cont16k.py (f69fd04a), lines 112-131, and its capture flags
        args = ['--dump-dir', str(tmp_path / 'first-request'), '--dump-max', '1',
                '--response-dir', str(tmp_path / 'wire-responses'),
                '--request-events', str(tmp_path / 'wire-requests.jsonl'),
                '--conversation-id', 'c', '--gate-client', env.get('PI_GATE_CLIENT', 'pi')]
        if env.get('PI_SAMPLING_SEED'):
            args += ['--sampling-seed', env['PI_SAMPLING_SEED']]
        if env.get('PI_REASONING_EFFORT'):
            args += ['--reasoning-effort', env['PI_REASONING_EFFORT']]
        if env.get('PI_REQUEST_OVERRIDES'):
            args += ['--request-overrides', env['PI_REQUEST_OVERRIDES']]
        if env.get('PI_INLINE_REASONING_HISTORY') == '1':
            args += ['--inline-reasoning-history']
        if env.get('PI_DP_RANK'):
            args += ['--dp-rank', env['PI_DP_RANK']]
        if env.get('PI_HTTP_FORCE_CLOSE') == '1':
            args += ['--backend-force-close']
        if env.get('PI_KV_GATE'):
            args += ['--kv-gate', env['PI_KV_GATE']]
            if env.get('PI_KV_GATE_FILE'):
                args += ['--kv-gate-file', env['PI_KV_GATE_FILE']]
            if env.get('PI_GATE_KEY'):
                args += ['--gate-key', env['PI_GATE_KEY']]
        return args

    def pairs(args):   # flag -> value (or True), order-free
        out, i = {}, 0
        while i < len(args):
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                out[args[i]] = args[i + 1]
                i += 2
            else:
                out[args[i]] = True
                i += 1
        return out

    for env in (eval_q35, pi_eval):
        args, _ = proxy_options(tmp_path, env, conversation_id='c')
        assert pairs(args) == pairs(aa_evals_args(env)) and len(args) == len(aa_evals_args(env))
    with pytest.raises(SystemExit, match='does not combine'):
        proxy_options(tmp_path, {**pi_eval, 'PI_REQUEST_OVERRIDES_MODE': 'update'})


def test_pi_cont16k_proxy_option_mapping(tmp_path):
    from gen.pi_cont16k import proxy_options, stop_settings
    import signal
    synth = {'PI_CONT16K_PROFILE': 'synth'}
    args, record = proxy_options(tmp_path, synth)
    assert args == ['--no-load-poll'] and record['routing_load_polling'] is False
    args, _ = proxy_options(tmp_path, {**synth, 'PI_PROXY_LOAD_POLLING': '1'})
    assert args == []
    args, record = proxy_options(tmp_path, {**synth, 'PI_REQUEST_OVERRIDES': '{"seed": 9}', 'PI_MIRROR_REASONING': '1',
                                            'PI_CAMPAIGN_DP_RANK': '2', 'PI_HTTP_FORCE_CLOSE': '1', 'PI_WIRE_LOG': '1'})
    assert json.loads(args[args.index('--request-overrides') + 1]) == {'seed': 9}
    assert args[args.index('--request-overrides-mode') + 1] == 'update'
    assert json.loads(args[args.index('--backend-headers') + 1]) == {'X-data-parallel-rank': '2'}
    assert '--mirror-reasoning-content' in args and '--backend-force-close' in args
    assert args[args.index('--wire-log') + 1] == str(tmp_path / 'wire.jsonl')
    assert record['history_reasoning_adapter'] == 'mirror' and record['wire_log'] and record['data_parallel_rank'] == 2
    args, record = proxy_options(tmp_path, {'PI_INLINE_REASONING_HISTORY': '1'})
    assert '--inline-reasoning-history' in args and record['history_reasoning_adapter'] == 'inline'
    assert stop_settings({}) == (signal.SIGTERM, 10.0)
    assert stop_settings({'PI_CONT16K_STOP_SIGNAL': 'INT', 'PI_CONT16K_STOP_TIMEOUT': '30'}) == (signal.SIGINT, 30.0)


def test_eval_q35_sampling_temperature_reaches_the_backend(tmp_path):
    """#167: --temperature/--top-p (from an eval_q35 attempt's PI_TEMPERATURE/PI_TOP_P) replace the seeded path's
    1.0 / 0.95; without them every request keeps 1.0 / 0.95, as before."""
    seeded = ['--sampling-seed', '5']
    req = asyncio.run(run_proxy(tmp_path, seeded + ['--temperature', '0.6', '--top-p', '0.9'], [{'messages': HISTORY}]))
    assert (req[0]['temperature'], req[0]['top_p'], req[0]['seed']) == (0.6, 0.9, 5)
    req = asyncio.run(run_proxy(tmp_path, seeded, [{'messages': HISTORY}]))
    assert (req[0]['temperature'], req[0]['top_p']) == (1.0, 0.95)


@pytest.mark.parametrize('extra', [['--temperature', '0.6'],
                                   ['--sampling-seed', '5', '--top-p', '0.9', '--request-overrides', '{"top_p": 1}']])
def test_sampling_temperature_needs_the_seeded_path(extra):
    r = subprocess.run([sys.executable, str(Path(__file__).with_name('sticky_proxy.py')), '--port', '1',
                        '--backends', 'http://127.0.0.1:9', *extra], capture_output=True, text=True)
    assert r.returncode == 2 and 'apply only with --sampling-seed' in r.stderr


def test_pi_cont16k_temperature_flags_and_record(tmp_path):
    from gen.pi_cont16k import proxy_options
    eval_q35 = {'PI_SAMPLING_SEED': '123', 'PI_PROXY_CAPTURE_FILES': '0', 'PI_GATE_HEADERS': '0'}
    args, record = proxy_options(tmp_path, {**eval_q35, 'PI_TEMPERATURE': '0.6', 'PI_TOP_P': '0.95'})
    assert args == ['--sampling-seed', '123', '--temperature', '0.6', '--top-p', '0.95']
    assert (record['temperature'], record['top_p']) == (0.6, 0.95)
    args, record = proxy_options(tmp_path, eval_q35)   # a run without temperatures: flags as before
    assert args == ['--sampling-seed', '123'] and (record['temperature'], record['top_p']) == (1.0, 0.95)
    _, record = proxy_options(tmp_path, {'PI_CONT16K_PROFILE': 'synth'})   # synth: no seeded path, no record
    assert 'temperature' not in record and 'top_p' not in record
    with pytest.raises(SystemExit, match='needs PI_SAMPLING_SEED'):
        proxy_options(tmp_path, {'PI_TEMPERATURE': '0.6'})
    with pytest.raises(SystemExit, match='no PI_REQUEST_OVERRIDES'):
        proxy_options(tmp_path, {**eval_q35, 'PI_TEMPERATURE': '0.6', 'PI_REQUEST_OVERRIDES': '{"temperature": 1}'})
