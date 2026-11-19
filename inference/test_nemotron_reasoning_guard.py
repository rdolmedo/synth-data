"""The production request adapter precedes exact tokenization and the wire log."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error

import pytest
from tb_agents.test_pi_context_limit import backend, guard, post
from inference.sticky_proxy import inline_reasoning_history
from synth.wire import load as load_wire


def test_adapter_retains_reasoning_and_is_idempotent():
    payload={'messages':[{'role':'assistant','content':'','reasoning_content':'retained thought'},
        {'role':'user','content':'Thinking exceeded the maximum thinking budget, please continue.'}]}
    inline_reasoning_history(payload)
    assert payload['messages'][0]['content']=='<think>\nretained thought</think>'
    assert payload['chat_template_kwargs']['truncate_history_thinking'] is False
    saved=json.dumps(payload)
    inline_reasoning_history(payload)
    assert json.dumps(payload)==saved


@pytest.mark.parametrize('stream',[False,True])
def test_actual_proxy_adapts_before_guard_and_wire_log(tmp_path,monkeypatch,stream):
    with backend(131998) as (url,requests,tokenizations), guard(url,tmp_path,monkeypatch) as guarded:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port=listener.getsockname()[1]
        script=Path(__file__).with_name('sticky_proxy.py')
        wire=tmp_path/'wire.jsonl'
        command=[sys.executable,str(script),'--port',str(port),'--backends',guarded,
            '--turn-max-tokens','128','--turn-retries','0','--turn-continue','0',
            '--inline-reasoning-history','--wire-log',str(wire)]
        proc=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            payload={'model':'test','messages':[{'role':'assistant','content':'','reasoning_content':'kept before note'},
                {'role':'user','content':'Thinking exceeded the maximum thinking budget, please continue.'}],
                'max_tokens':128,'stream':stream}
            deadline=time.monotonic()+15
            while True:
                try:
                    post(f'http://127.0.0.1:{port}',payload)
                    break
                except urllib.error.URLError:
                    assert proc.poll() is None
                    if time.monotonic()>deadline: raise
                    time.sleep(.1)
            assert tokenizations[-1]['messages'][0]['content']=='<think>\nkept before note</think>'
            assert tokenizations[-1]['chat_template_kwargs']['truncate_history_thinking'] is False
            assert requests[-1]['messages']==tokenizations[-1]['messages']
            assert requests[-1]['max_tokens']==2
            saved=load_wire(wire)[-1]['body']
            assert saved['messages']==requests[-1]['messages']
        finally:
            proc.terminate()
            proc.wait(timeout=10)
