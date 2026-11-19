"""Opus 5.5 through the headless Claude Code CLI (`claude -p`, the user's subscription), one plain call per prompt,
no tools. Every result is cached under WORK/calls/, keyed by (model, effort, system, user), so a rerun of a stage
reuses finished calls and only pays for new ones."""
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

WORK = Path(os.environ.get('VOCAB_WORK', '/data/aai/tbench/tasksynth/vocab/v2'))
MODEL = 'claude-opus-5-5'


class QuotaError(RuntimeError):
    pass


def call(system, user, tag, effort='high', timeout=3600):
    key = hashlib.sha256(json.dumps([MODEL, effort, system, user]).encode()).hexdigest()[:16]
    out = WORK / 'calls' / f'{tag}-{key}.json'
    if out.exists():
        return json.loads(out.read_text())['result']
    out.parent.mkdir(parents=True, exist_ok=True)
    sysf = WORK / 'calls' / f'{tag}-{key}.system.txt'
    sysf.write_text(system)
    env = {k: v for k, v in os.environ.items() if not k.startswith('CLAUDE')}
    env['DISABLE_AUTOUPDATER'] = '1'
    msg = ''
    for attempt in range(3):
        t0 = time.time()
        r = subprocess.run(['claude', '-p', '--model', MODEL, '--effort', effort, '--system-prompt-file', str(sysf),
                            '--tools', '', '--strict-mcp-config', '--no-session-persistence', '--output-format', 'json'],
                           input=user, capture_output=True, text=True, env=env, timeout=timeout)
        try:
            res = json.loads(r.stdout)
        except ValueError:
            res = {'is_error': True, 'result': (r.stderr or r.stdout)[-500:]}
        if not res.get('is_error') and res.get('result'):
            models = list(res.get('modelUsage') or {})
            if models and MODEL not in models[0]:
                print(f'WARNING {tag}: answered by {models}', flush=True)
            rec = {'tag': tag, 'models': models, 'effort': effort, 'user': user, 'result': res['result'],
                   'usage': res.get('usage'), 'seconds': round(time.time() - t0)}
            tmp = out.with_suffix('.partial')
            tmp.write_text(json.dumps(rec))
            tmp.replace(out)
            print(f'{time.strftime("%H:%M:%S")} {tag} ok {rec["seconds"]}s out={(rec["usage"] or {}).get("output_tokens")}',
                  flush=True)
            return res['result']
        msg = str(res.get('result'))
        if 'limit' in msg.lower() and 'usage' in msg.lower() or 'rate limit' in msg.lower():
            raise QuotaError(msg[:300])
        print(f'{tag} attempt {attempt} failed: {msg[:200]}', flush=True)
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f'{tag}: {msg[:300]}')


def parse_json(text):
    blocks = re.findall(r'```(?:json)?\s*\n(.*?)```', text, re.S) or [text]
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except ValueError:
            pass
    for o, c in (('[', ']'), ('{', '}')):
        i, j = text.find(o), text.rfind(c)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except ValueError:
                pass
    raise ValueError('no JSON in reply')


def ask_json(system, user, tag, effort='high'):
    """call() + parse; one retry with an explicit format reminder when the reply holds no parseable JSON."""
    try:
        return parse_json(call(system, user, tag, effort))
    except ValueError:
        return parse_json(call(system, user + '\n\nReply with the JSON only, in one ```json block.', tag + '-retry', effort))
