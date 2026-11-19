"""`claude -p` (Opus, the user's subscription) for low-volume brainstorming and labelling (hundreds of calls)."""
import json
import os
import subprocess


def opus(prompt: str, system: str | None = None, effort: str = 'high', timeout: int = 3600) -> str:
    cmd = ['claude', '-p', '--model', 'claude-opus-5-5', '--effort', effort, '--tools', '', '--strict-mcp-config',
           '--no-session-persistence', '--output-format', 'json']
    if system:
        cmd += ['--system-prompt', system]
    env = {k: v for k, v in os.environ.items()
           if k not in ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')
           and not k.startswith('CLAUDE_CODE_MESSAGING_')}
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=timeout,
                         env={**env, 'DISABLE_AUTOUPDATER': '1'})
    try:
        return json.loads(res.stdout)['result']
    except (ValueError, KeyError) as e:
        raise RuntimeError(f'claude -p failed: {res.stdout[:300]} {res.stderr[:300]}') from e


def json_lines(text: str) -> list[dict]:
    out = []
    for x in text.splitlines():
        x = x.strip().rstrip(',')
        if x.startswith('{') and x.endswith('}'):
            try:
                out.append(json.loads(x))
            except ValueError:
                pass
    return out
