"""tb_agents/terminus_eval.py without a GPU: seeds, vLLM's context-length 400, the episode end at the limit.

  /opt/aai/venvs/harbor/bin/python tb_agents/test_terminus_eval.py   (or pytest)
"""
import asyncio
import contextlib
import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harbor.agents.terminus_2.terminus_2 import Terminus2  # noqa: E402
from harbor.llms.base import ContextLengthExceededError  # noqa: E402
from harbor.models.agent.context import AgentContext  # noqa: E402
from harbor.trial.errors import AgentTimeoutError  # noqa: E402
from tb_agents.terminus_eval import Terminus2Eval  # noqa: E402

# vLLM 0.27.1 renderers/params.py:_token_len_check (prompt + max_tokens > max_model_len)
VLLM_400 = ("This model's maximum context length is 132000 tokens. However, you requested 16384 output tokens and "
            "your prompt contains 120000 input tokens, for a total of 136384 tokens. Please reduce the length of "
            "the input prompt or the number of requested output tokens.")


@contextlib.contextmanager
def backend(replies):
    """Fake OpenAI chat endpoint: pops one reply per request ('400' = vLLM's context-length error)."""
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            reply = replies.pop(0)
            if reply == '400':
                code, body = 400, {'error': {'message': VLLM_400, 'type': 'BadRequestError', 'param': 'input_tokens',
                                             'code': 400}, 'object': 'error'}
            else:
                code, body = 200, {'id': 'x', 'object': 'chat.completion', 'model': 'test', 'choices': [
                    {'index': 0, 'finish_reason': 'stop',
                     'message': {'role': 'assistant', 'content': reply, 'reasoning_content': 'thinking'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', bodies
    finally:
        server.shutdown()
        server.server_close()


def agent(logs, api_base, **kw):
    return Terminus2Eval(logs_dir=logs, model_name='hosted_vllm/test', api_base=api_base, temperature=1.0,
                         model_info={'max_input_tokens': 132000, 'max_output_tokens': 16384,
                                     'input_cost_per_token': 0, 'output_cost_per_token': 0},
                         llm_kwargs={'max_tokens': 16384, 'top_p': 0.95, 'timeout': 60},
                         interleaved_thinking=True, enable_summarize=False, proactive_summarization_threshold=0,
                         record_terminal_session=False, **kw)


def test_harbor_accepts_the_worker_kwargs():
    """The --ak list of gen/run_harbor.sh (terminus-2 branch) plus eval_q35/worker.py's, through Harbor's parser."""
    from harbor.cli.utils import parse_kwargs
    kw = parse_kwargs(['api_base=http://h:1/v1', 'temperature=1.0',
                       'model_info={"max_input_tokens":132000,"max_output_tokens":16384,"input_cost_per_token":0,"output_cost_per_token":0}',
                       'llm_kwargs={"max_tokens":16384,"timeout":3600,"top_p":0.95}', 'interleaved_thinking=true',
                       'record_terminal_session=false', 'store_all_messages=true',
                       'trajectory_config={"raw_content":true,"linear_history":true}',
                       'enable_summarize=false', 'proactive_summarization_threshold=0', 'sampling_seed=284929153',
                       'kv_gate=0.80,0.72', 'kv_gate_file=/data/x/serve/kv.json', 'env_budget_multiplier=1',
                       'tail_cut_file=/data/x/queue/cut/k.json'])
    Terminus2Eval.preflight(kw)
    with tempfile.TemporaryDirectory() as d:
        a = Terminus2Eval(logs_dir=Path(d), model_name='hosted_vllm/test', **kw)
        assert a._seed == 284929153 and a._gate == (0.8, 0.72) and a._env_mult == 1.0
        assert a.options.enable_summarize is False and a.options.proactive_summarization_threshold == 0
        assert a.options.interleaved_thinking is True


def test_seeds_sampling_and_context_400():
    with tempfile.TemporaryDirectory() as d, backend(['{"a": 1}', '{"a": 2}', '400']) as (base, bodies):
        a = agent(Path(d), base, sampling_seed=2147483646)
        r = asyncio.run(a._llm.call(prompt='p0', message_history=[]))
        assert r.reasoning_content == 'thinking'
        asyncio.run(a._llm.call(prompt='p1', message_history=[{'role': 'user', 'content': 'p0'},
                                                             {'role': 'assistant', 'content': '{"a": 1}',
                                                              'reasoning_content': 'thinking'}]))
        try:
            asyncio.run(a._llm.call(prompt='p2', message_history=[]))
            raise AssertionError('no ContextLengthExceededError')
        except ContextLengthExceededError:
            pass
        assert [b['seed'] for b in bodies] == [2147483646, 0, 1], [b.get('seed') for b in bodies]
        assert all(b['max_tokens'] == 16384 and b['top_p'] == 0.95 and b['temperature'] == 1.0 for b in bodies)
        # interleaved thinking: the history's reasoning goes out on the wire
        assert bodies[1]['messages'][1].get('reasoning_content') == 'thinking', bodies[1]['messages'][1]
        recs = [json.loads(x) for x in (Path(d) / 'requests.jsonl').read_text().splitlines()]
        assert [r['history'] for r in recs] == [0, 2, 0] and recs[2]['error'] == 'ContextLengthExceededError'
        assert recs[0]['prompt_tokens'] == 10


def task_env(d, timeout):
    (d / 'environment').mkdir()
    (d / 'task.toml').write_text(f'version = "1.0"\n[agent]\ntimeout_sec = {timeout}\n')
    return SimpleNamespace(environment_dir=d / 'environment')


def test_context_limit_ends_episode_normally():
    async def parent_run(self, instruction, environment, context):
        await self._llm.call(prompt='p', message_history=[])   # model time: not charged
        raise ContextLengthExceededError()

    with tempfile.TemporaryDirectory() as d, backend(['{}']) as (base, _):
        d = Path(d)
        a = agent(d, base)
        with patch.object(Terminus2, 'run', parent_run):
            assert asyncio.run(a.run('x', task_env(d, 600), AgentContext())) is None
        clock = json.loads((d / 'env_clock.json').read_text())
        assert clock['state'] == 'context_limit' and clock['requests'] == 1 and clock['budget_sec'] == 600


def test_env_budget_counts_environment_time_only():
    async def parent_run(self, instruction, environment, context):
        while True:
            await asyncio.sleep(0.5)   # "environment" time

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a = agent(d, 'http://127.0.0.1:9/v1', env_budget_multiplier=0.002)   # 1000 s x 0.002 = 2 s
        with patch.object(Terminus2, 'run', parent_run):
            try:
                asyncio.run(a.run('x', task_env(d, 1000), AgentContext()))
                raise AssertionError('no AgentTimeoutError')
            except AgentTimeoutError:
                pass
        clock = json.loads((d / 'env_clock.json').read_text())
        assert clock['state'] == 'env_budget_spent' and 2 <= clock['env_sec'] < 6, clock


def test_tail_cut_file_stops_like_a_timeout():
    async def parent_run(self, instruction, environment, context):
        while True:
            await asyncio.sleep(0.5)

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cut = d / 'cut.json'
        threading.Timer(1.0, lambda: cut.write_text('{}')).start()
        a = agent(d, 'http://127.0.0.1:9/v1', tail_cut_file=str(cut))
        with patch.object(Terminus2, 'run', parent_run):
            try:
                asyncio.run(a.run('x', task_env(d, 1000), AgentContext()))
                raise AssertionError('no AgentTimeoutError')
            except AgentTimeoutError:
                pass
        assert json.loads((d / 'env_clock.json').read_text())['state'] == 'tail_cut'


def test_kv_gate_holds_until_low():
    import time
    with tempfile.TemporaryDirectory() as d, backend(['{}']) as (base, _):
        d = Path(d)
        f = d / 'kv.json'
        f.write_text(json.dumps({'t': time.time(), 'kv': 0.9}))
        a = agent(d, base, kv_gate='0.80,0.72', kv_gate_file=str(f))
        threading.Timer(1.0, lambda: f.write_text(json.dumps({'t': time.time(), 'kv': 0.5}))).start()
        with patch('tb_agents.terminus_eval.random.uniform', return_value=0), \
                patch.object(Terminus2Eval, '_wait_gate', Terminus2Eval._wait_gate):
            a._gate_state['at'] = -1e9
            orig = asyncio.sleep

            async def fast_sleep(s):   # 5 s polls and 10 s file re-reads, compressed
                a._gate_state['at'] = -1e9
                await orig(min(s, 0.2))
            with patch('tb_agents.terminus_eval.asyncio.sleep', fast_sleep):
                asyncio.run(a._llm.call(prompt='p', message_history=[]))
        rec = json.loads((d / 'requests.jsonl').read_text())
        assert a._gate_state['holds'] == 1 and rec['gate_sec'] >= 0.8, (a._gate_state, rec)


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
