"""Endpoint checks: text answer, bash tool-call parsing, and Prometheus scraping."""
import json
import socket
import sys
import urllib.request

from .common import REPO

sys.path.insert(0, str(REPO / 'inference'))
from replay_bench import scrape, server_summary  # noqa: E402,F401

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
BASH = {'type': 'function', 'function': {'name': 'bash', 'description': 'Run shell command',
        'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}}}


def request(url, body=None, timeout=10):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with OPENER.open(req, timeout=timeout) as r:
        return json.load(r)


def free_port():
    with socket.socket() as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def probe(base, teacher, max_tokens=4096):
    """Greedy text and tool probes with the teacher's template kwargs."""
    overrides = teacher.get('request', {}).get('overrides', {})
    common = {'model': teacher['served_name'], 'temperature': 0, 'max_tokens': max_tokens,
              **{k: v for k, v in overrides.items() if k in ('chat_template_kwargs', 'reasoning_effort')}}
    text = request(base + '/chat/completions', {**common, 'messages': [
        {'role': 'user', 'content': 'What is 17 + 25? Answer briefly.'}]}, timeout=900)
    if '42' not in (text['choices'][0]['message'].get('content') or ''):
        raise RuntimeError(f'Text probe failed: {text}')
    tool = request(base + '/chat/completions', {**common, 'tools': [BASH], 'messages': [
        {'role': 'user', 'content': 'Use bash to run exactly: printf SYNTH_TOOL_OK'}]}, timeout=900)
    calls = tool['choices'][0]['message'].get('tool_calls') or []
    if not calls or calls[0]['function']['name'] != 'bash':
        raise RuntimeError(f'Tool parsing probe failed: {tool}')
    json.loads(calls[0]['function']['arguments'])
    return {'text': text, 'tool': tool}
