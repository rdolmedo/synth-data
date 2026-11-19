"""Harness smoke without a model: a scripted client drives Stirrup through code_exec, finish and abandon in the
Apptainer sandbox.  python -m aa_evals.gdpval.smoke_fake [SIF]   (any SIF with bash + python works)."""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

from stirrup.core.models import AssistantMessage, LLMClient, TextBlock, TokenUsage, ToolCall

from aa_evals.gdpval import agent as A
from aa_evals.gdpval import tasks as T


class Scripted(LLMClient):
    def __init__(self, script):
        self.script = list(script)
        self.seen = []

    @property
    def max_tokens(self):
        return 1000

    @property
    def context_window_tokens(self):
        return 131072

    @property
    def model_slug(self):
        return 'scripted'

    async def generate(self, messages, tools):
        self.seen.append((len(messages), sorted(tools)))
        name, args = self.script.pop(0)
        return AssistantMessage(blocks=[TextBlock(text='ok'), ToolCall.from_provider(provider_id=f'c{len(self.seen)}',
                                name=name, arguments=json.dumps(args))], token_usage=TokenUsage(input=10, answer=5))

    async def aclose(self):
        pass


async def main(sif):
    task = next(t for t in T.load_tasks() if t.reference_files)
    ref = task.ref_targets()[0][1]
    results = {}
    for mode in ('finish', 'abandon', 'bad_path'):
        if mode == 'finish':
            script = [('code_exec', {'cmd': f'ls -la /home/user && pwd && echo $HOME && python3 -c "import sys;print(sys.version)"'}),
                      ('code_exec', {'cmd': 'cd /home/user && mkdir -p out && echo hello > out/report.txt && echo t > /tmp/x.txt'}),
                      ('code_exec', {'cmd': 'cat /etc/hostname /data/aai/aa_evals/README.md 2>&1 | head -3; env | grep -c KEY'}),
                      ('finish', {'reason': 'done', 'paths': ['/home/user/out/report.txt', '/tmp/x.txt', ref]})]
        elif mode == 'abandon':
            script = [('abandon_task_finish', {'reason': 'inputs missing'})]
        else:
            script = [('finish', {'reason': 'x', 'paths': ['/home/user/nope.docx']}),
                      ('finish', {'reason': 'x', 'paths': ['/home/user/' + Path(ref).name]})]
        client = Scripted(script)
        orig = A.CappedVLLMClient
        A.CappedVLLMClient = lambda **kw: client   # noqa: E731
        try:
            with tempfile.TemporaryDirectory(dir='/tmp') as d:
                cfg = {'model': {'spec': 'url:x@y', 'served_name': 'y', 'body': {}, 'max_output': 1000, 'vision': True},
                       'agent_network': dict(__import__('aa_evals.gdpval.network', fromlist=['POLICY']).POLICY),
                       'context': {'window': 131072}, 'sandbox': {'sif_local': sif}, 'limits': {'wall_sec': 600}}
                out = await A.run_attempt(task, 0, cfg=cfg, work=Path(d) / 'w', out_dir=Path(d) / 'o')
                tr = json.loads(__import__('gzip').open(Path(d) / 'o' / 'transcript.json.gz').read())
                tool_out = [m for g in tr['history'] for m in g if m.get('role') == 'tool' or 'tool_call_id' in m]
                results[mode] = {k: out[k] for k in ('status', 'end', 'error', 'submitted_files', 'turns', 'sandbox')}
                results[mode]['tools_seen'] = client.seen[0][1]
                results[mode]['tool_outputs'] = [str(m.get('content'))[:400] for m in tool_out]
        finally:
            A.CappedVLLMClient = orig
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else '/data/aai/aa_evals/sif/gdpval-v1.sif'))
