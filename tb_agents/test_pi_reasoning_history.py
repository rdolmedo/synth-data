"""Exercise the actual patched Pi 1.2.0 provider and CLI against a local SSE endpoint.

Build the immutable bundle first with gen/prepare_pi_cont16k.py --build.
No model server, task environment, or running campaign is involved.
"""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tb_agents.pi_continuation import (
    BUNDLE, CACHE, CONTINUE, LEGACY_CACHE, LEGACY_SHA256, PATCHES,
    REASONING_OLD, annotate_sessions, bundle_provenance,
    launch_cache, patch_source, verify_bundle,
)


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    cache = Path(os.environ.get('PI_TEST_CACHE', str(CACHE)))
    if not (cache / BUNDLE).is_file():
        pytest.skip('Build the fixed Pi bundle with gen/prepare_pi_cont16k.py --build')
    sha = verify_bundle(cache)
    assert bundle_provenance(cache, sha)['reasoning_history_patch']
    root = tmp_path_factory.mktemp('pi-runtime')
    with tarfile.open(cache / BUNDLE) as archive:
        # The bundled CLI imports regular files directly; npm's .bin symlinks
        # are unnecessary (and older Python tar filters reject those links).
        archive.extractall(root, members=[m for m in archive if m.isfile() or m.isdir()], filter='data')
    pi = root / 'opt/harbor-pi'
    return pi / 'node/bin/node', pi / 'pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle'


@contextlib.contextmanager
def endpoint(respond):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(payload)
            delta, finish, usage = respond(payload, len(requests))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for chunk in [
                {'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                {'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]},
                {'choices': [], 'usage': usage},
            ]:
                self.wfile.write(('data: ' + json.dumps(chunk) + '\n\n').encode())
            self.wfile.write(b'data: [DONE]\n\n')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_js(runtime, tmp_path, source, *args):
    node, bundle = runtime
    driver = tmp_path / 'driver.mjs'
    driver.write_text(source.replace('BUNDLE_URL', bundle.as_uri()))
    return subprocess.run([str(node), str(driver), *args], capture_output=True, text=True,
                          timeout=60, check=True)


def test_actual_provider_retains_repeated_capped_reasoning(runtime, tmp_path):
    def respond(payload, index):
        return ({'reasoning_content': f'thought-{index}'}, 'length',
                {'prompt_tokens': 100 * index, 'completion_tokens': 16384})

    with endpoint(respond) as (url, requests):
        result = run_js(runtime, tmp_path, """
import {stream} from 'BUNDLE_URL/chunks/openai-completions-QW2B4SML.js';
const model={id:'test',name:'test',provider:'test',api:'openai-completions',
 baseUrl:process.argv[2],reasoning:true,input:['text'],contextWindow:132000,maxTokens:16384,
 cost:{input:0,output:0,cacheRead:0,cacheWrite:0},compat:{supportsDeveloperRole:false}};
const context={messages:[{role:'user',content:'solve',timestamp:0}]};
const output=[];
for(let i=0;i<3;i++) {
 const message=await stream(model,context,{apiKey:'mock',maxTokens:16384}).result();
 if(message.stopReason!=='length') throw new Error(JSON.stringify(message));
 output.push(message);context.messages.push(message);
 context.messages.push({role:'user',content:process.argv[3],timestamp:i+1});
}
console.log(JSON.stringify(output));
""", url, CONTINUE)
    assert len(requests) == 3
    for index, request in enumerate(requests):
        assert [m['role'] for m in request['messages']] == ['user'] + ['assistant', 'user'] * index
        for previous in range(index):
            assistant = request['messages'][1 + previous * 2]
            assert assistant == {'role': 'assistant', 'content': None,
                                 'reasoning_content': f'thought-{previous + 1}'}
            assert request['messages'][2 + previous * 2]['content'] == CONTINUE
        assert request.get('max_tokens', request.get('max_completion_tokens')) == 16384
    native = json.loads(result.stdout)
    session = tmp_path / 'pi/sessions/test.jsonl'
    session.parent.mkdir(parents=True)
    session.write_text(''.join(json.dumps({'type': 'message', 'message': m}) + '\n' for m in native))
    annotated = annotate_sessions(tmp_path)['sessions'][0]['entries']
    assert all(e['message']['stopReason'] == 'length' and e['sft_loss_weight'] == 0 for e in annotated)


def test_converter_supported_fields_details_empty_and_tools(runtime, tmp_path):
    result = run_js(runtime, tmp_path, """
import {convertMessages} from 'BUNDLE_URL/chunks/openai-completions-QW2B4SML.js';
const model={id:'test',provider:'test',api:'openai-completions',input:['text'],reasoning:true};
const compat={requiresReasoningContentOnAssistantMessages:true};
const assistant=(content,stopReason='length')=>({role:'assistant',content,stopReason,
 api:model.api,provider:model.provider,model:model.id,timestamp:0});
const thinking=(thinking,thinkingSignature)=>({type:'thinking',thinking,thinkingSignature});
const convert=(messages,overrides={})=>convertMessages(model,{messages},{...compat,...overrides});
const details=[{type:'reasoning.encrypted',data:'opaque',id:'r1'}];
const cases={};
for(const field of ['reasoning','reasoning_content','reasoning_text'])
 cases[field]=convert([assistant([thinking('unfinished',field)])]);
cases.details=convert([assistant([thinking('',JSON.stringify(details))])]);
cases.empty=convert([assistant([]),assistant([thinking('  ','reasoning_content')])]);
cases.unsupported=convert([assistant([thinking('hidden','unsupported')])]);
cases.noSignature=convert([assistant([thinking('hidden')])]);
cases.errors=convert([assistant([thinking('hidden','reasoning_content')],'error'),
 assistant([thinking('hidden','reasoning_content')],'aborted')]);
cases.text=convert([assistant([{type:'text',text:'answer'}],'stop')]);
cases.tool=convert([assistant([{type:'toolCall',id:'t1',name:'bash',arguments:{command:'true'}},
 thinking('using a tool','reasoning_content')],'toolUse'),
 {role:'toolResult',toolCallId:'t1',toolName:'bash',content:[{type:'text',text:'ok'}],timestamp:1}]);
cases.thinkingAsText=convert([assistant([thinking('legacy','reasoning_content')])],{requiresThinkingAsText:true});
cases.crossModel=convert([{...assistant([thinking('other','reasoning_content')]),model:'different'}]);
cases.opencode=convertMessages({...model,provider:'opencode-go'},
 {messages:[{...assistant([thinking('alias','reasoning')]),provider:'opencode-go'}]},compat);
console.log(JSON.stringify(cases));
""")
    cases = json.loads(result.stdout)
    for field in ('reasoning', 'reasoning_content', 'reasoning_text'):
        assert cases[field][0][field] == 'unfinished'
        assert cases[field][0]['content'] is None
    assert cases['details'][0]['reasoning_details'] == [{'type': 'reasoning.encrypted', 'data': 'opaque', 'id': 'r1'}]
    for case in ('empty', 'unsupported', 'noSignature', 'errors'):
        assert cases[case] == []
    assert cases['text'][0]['content'] == 'answer'
    assert [m['role'] for m in cases['tool']] == ['assistant', 'tool']
    assert cases['tool'][0]['tool_calls'][0]['function']['arguments'] == '{"command":"true"}'
    assert cases['thinkingAsText'][0]['content'] == [{'type': 'text', 'text': 'legacy'}]
    assert cases['crossModel'][0]['content'] == 'other'
    assert cases['opencode'][0]['reasoning_content'] == 'alias'
    assert 'reasoning' not in cases['opencode'][0]


def test_cli_compacts_before_context_headroom_is_exhausted(runtime, tmp_path):
    node, bundle = runtime
    home = tmp_path / 'home'
    config = home / '.pi/agent'
    config.mkdir(parents=True)
    (config / 'settings.json').write_text(json.dumps({
        'compaction': {'enabled': True, 'reserveTokens': 16384, 'keepRecentTokens': 20000},
    }))
    summary_calls = []
    ordinary_calls = []

    def respond(payload, index):
        summary = any('You are a context summarization assistant.' in str(m.get('content', ''))
                      for m in payload['messages'] if m['role'] in ('system', 'developer'))
        # Deterministic mock token accounting includes serialized reasoning; no real tokenizer claim.
        chars = sum(len(str(m.get('content') or '')) + len(m.get('reasoning_content', ''))
                    for m in payload['messages'])
        input_tokens = (chars + 3) // 4
        if summary:
            summary_calls.append(payload)
            return {'content': '## Goal\nSolve the task.\n## Progress\nThinking done.'}, 'stop', {'prompt_tokens': input_tokens, 'completion_tokens': 20}
        ordinary_calls.append(payload)
        if summary_calls:
            return {'content': 'Done.'}, 'stop', {'prompt_tokens': input_tokens, 'completion_tokens': 2}
        if len(ordinary_calls) > 10:
            return {'content': 'Loop guard'}, 'stop', {'prompt_tokens': input_tokens, 'completion_tokens': 2}
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        return {'reasoning_content': 'r' * (cap * 4)}, 'length', {'prompt_tokens': input_tokens, 'completion_tokens': cap}

    with endpoint(respond) as (url, requests):
        (config / 'models.json').write_text(json.dumps({'providers': {'mock': {
            'baseUrl': url, 'api': 'openai-completions', 'apiKey': 'mock',
            'models': [{'id': 'test', 'name': 'test', 'reasoning': True, 'input': ['text'],
                        'contextWindow': 132000, 'maxTokens': 16384,
                        'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}],
            'compat': {'supportsDeveloperRole': False}}}}))
        process = subprocess.run([str(node), str(bundle / 'cli.js'), '--print', '--mode', 'json',
                                  '--provider', 'mock', '--model', 'test', '--thinking', 'high',
                                  '--no-extensions', '--no-skills', '--no-prompt-templates',
                                  '--session', str(tmp_path / 'session.jsonl'), '--', 'Solve this mock task.'],
                                 env={**{k: v for k, v in os.environ.items() if not k.lower().endswith('_proxy')},
                                      'HOME': str(home), 'PI_CODING_AGENT_DIR': str(config),
                                      'NO_PROXY': 'localhost,127.0.0.1'},
                                 cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stderr
    # Pi may separately summarize the completed history and a split turn prefix.
    assert summary_calls, process.stdout[-2000:]
    assert 2 < len(ordinary_calls) <= 9
    for i, payload in enumerate(ordinary_calls[:-1]):
        assistants = [m for m in payload['messages'] if m['role'] == 'assistant']
        assert len(assistants) == i
        for previous, assistant in zip(ordinary_calls[:i], assistants):
            previous_cap = previous.get('max_tokens', previous.get('max_completion_tokens'))
            assert assistant.get('reasoning_content') == 'r' * (4 * previous_cap)
        cap = payload.get('max_tokens', payload.get('max_completion_tokens'))
        # Pi can further reduce the cap as its own token estimate grows.
        assert 0 < cap <= 16384
        chars = sum(len(str(m.get('content') or '')) + len(m.get('reasoning_content', ''))
                    for m in payload['messages'])
        assert (chars + 3) // 4 + cap < 132000
    events = [json.loads(line) for line in (tmp_path / 'session.jsonl').read_text().splitlines()]
    assert sum(e['type'] == 'compaction' for e in events) == 1
    assert any(e.get('message', {}).get('stopReason') == 'length' for e in events)
    assert events[-1]['message']['stopReason'] == 'stop'


def test_patch_rejects_upstream_mismatch_and_already_patched():
    for suffix, old, new in PATCHES:
        assert patch_source('prefix' + old + 'suffix', old, new) == 'prefix' + new + 'suffix'
        for bad in ('different source', old + old, new):
            with pytest.raises(ValueError, match='exactly one stock'):
                patch_source(bad, old, new)


def test_exact_historical_bundle_remains_explicitly_pinned():
    if not (LEGACY_CACHE / BUNDLE).is_file():
        pytest.skip('Historical bundle not present')
    assert verify_bundle(LEGACY_CACHE) == LEGACY_SHA256
    assert bundle_provenance(LEGACY_CACHE, LEGACY_SHA256)['reasoning_history_patch'] is False


def test_launches_upgrade_legacy_override_and_preserve_custom_cache(tmp_path):
    assert launch_cache({}) == CACHE
    assert launch_cache({'PI_CONT16K_CACHE': str(CACHE)}) == CACHE
    assert launch_cache({'PI_CONT16K_CACHE': str(LEGACY_CACHE)}) == CACHE
    alias = tmp_path / 'legacy-alias'
    alias.symlink_to(LEGACY_CACHE, target_is_directory=True)
    assert launch_cache({'PI_CONT16K_CACHE': str(alias)}) == CACHE
    custom = tmp_path / 'custom-fixed-cache'
    assert launch_cache({'PI_CONT16K_CACHE': str(custom)}) == custom


def test_verifier_rejects_unpatched_adapter_even_with_valid_checksum(tmp_path):
    import hashlib
    import io
    archive = tmp_path / BUNDLE
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, 'w') as output:
        for suffix, old, new in PATCHES:
            content = (old if old == REASONING_OLD else new).encode()
            member = tarfile.TarInfo('opt/harbor-pi' + suffix)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    archive.with_suffix('.tar.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='patch missing'):
        verify_bundle(tmp_path)
