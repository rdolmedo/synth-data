"""Fingerprints of held-out test tasks written by the judge model (user 2026-10-31: DeepSeek, not Opus, for cost).

  python -m decontam.fingerprint --work W --out F.jsonl --bench "Terminal-Bench Lite (TB-Lite)" [--tasks a,b]

One single-turn request per test task: fingerprint_prompt.md (the Opus fingerprint brief and its audit lessons) as the
system prompt; the user message lists the opening of every instruction in the test set (siblings and base rates), then
the task's bundle (instruction, task.toml, tests, solution, Dockerfile, environment files) from W/bundles.jsonl. The
answer must be a fingerprint object (schema of variants/fp/fingerprints_v3.jsonl); invalid ones are retried. Resumable:
tasks already in --out are skipped. Raw answers go to <out>.raw.jsonl.

  python -m decontam.fingerprint ... --review F_OLD.jsonl --out F_NEW.jsonl
      Review pass (as the Opus fingerprints had): the same context plus the current fingerprint; the answer is the full
      revised fingerprint with a `changes` list, checked against the mistakes the prompt lists (a peer session found
      DeepSeek marking textbook techniques decisive, e.g. git-repo-forensics' "read the object store").
"""
import argparse
import asyncio
import json
from pathlib import Path
import re
import time

import aiohttp

from .judge import CONTEXT

PROMPT = Path(__file__).with_name('fingerprint_prompt.md')
KINDS = {'instance', 'crux', 'workaround', 'deliverable', 'interface', 'constraint', 'tooling', 'form', 'starter_files'}
OPENING = 400  # characters of each instruction in the sibling / base-rate list
REVIEW = ('Review this fingerprint against the criterion and every mistake listed in your instructions, especially '
          'mistake 2: a decisive item must be specific to this task (its instance, its crux, a given-away workaround '
          'that another author would not give); a textbook technique or the standard route for this kind of task '
          '(recovering unreachable commits from the git object store or reflog, repairing pip, querying a database, '
          'parsing a log format) is genre or at most minor, even when the instruction hints at it. Also check mistakes '
          '1, 3, 4 and 5. Answer with only the full revised fingerprint as one JSON object with the same fields plus '
          '"changes": a list of short notes on what you changed and why (empty if nothing).')


def parse(content):
    """The first JSON object in the answer that has fingerprint `choices`, or None (judge.parse wants a verdict label)."""
    if not content:
        return None
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', content):
        try:
            obj = decoder.raw_decode(content[m.start():])[0]
        except ValueError:
            continue
        if isinstance(obj, dict) and 'choices' in obj:
            return obj
    return None


def valid(fp):
    """The problem with a fingerprint object, or None."""
    if not isinstance(fp, dict):
        return 'not an object'
    for k, t in (('genre', str), ('conventions', list), ('choices', list), ('variant_test', str)):
        if not isinstance(fp.get(k), t):
            return f'{k} missing or not a {t.__name__}'
    if not fp['choices']:
        return 'no choices'
    for c in fp['choices']:
        if not isinstance(c, dict) or not all(isinstance(c.get(k), str) for k in ('id', 'kind', 'weight', 'choice')):
            return f'bad choice {str(c)[:80]}'
        if c['weight'] not in ('decisive', 'minor') or c['kind'] not in KINDS:
            return f'bad weight/kind in {c["id"]}'
    return None


def instruction(text):
    """The instruction.md block of a rendered bundle."""
    head = text.split('### instruction.md\n```\n', 1)
    return head[1].split('\n```', 1)[0] if len(head) == 2 else text


async def main(a):
    endpoint = json.loads(Path(a.endpoint).read_text())
    tb = {}
    with open(Path(a.work) / 'bundles.jsonl') as f:
        for line in f:
            row = json.loads(line)
            if row['side'] == 'tb':
                tb[row['name']] = row
    old = {r['tb']: {k: r[k] for k in ('tb', 'genre', 'conventions', 'choices', 'variant_test')}
           for r in map(json.loads, open(a.review))} if a.review else {}
    names = (sorted(old) if old else sorted(tb)) if not a.tasks else a.tasks.split(',')
    done = {json.loads(l)['tb'] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [n for n in names if n not in done]
    system = PROMPT.read_text().replace('{BENCH}', a.bench)
    siblings = '\n\n'.join(f'- `{n}`: {" ".join(instruction(tb[n]["text"]).split())[:OPENING]}' for n in sorted(tb))
    head = f'## Openings of every {a.bench} instruction (siblings and base rates)\n\n{siblings}\n\n'
    print(f'{len(todo)} tasks to fingerprint ({len(done)} done)', flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    t0 = time.time()

    async def one(session, name):
        user = head + f'## The task to fingerprint: `{name}`\n\n' + tb[name]['text']
        if old:
            user += ('\n\n## Its current fingerprint (to review)\n\n' + json.dumps(old[name], indent=1) + '\n\n' + REVIEW)
        else:
            user += '\n\nWrite the fingerprint of this task as the JSON object.'
        body = {'model': endpoint['served_name'], 'max_tokens': max(4096, min(a.max_tokens, CONTEXT - len(user) // 3 - 4000)),
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'reasoning_effort': a.effort, 'chat_template_kwargs': {'thinking': True, 'reasoning_effort': a.effort}}
        async with sem:
            for attempt in range(a.retries):
                try:
                    async with session.post(endpoint['url'] + '/chat/completions', json=body,
                                            timeout=aiohttp.ClientTimeout(total=3600)) as r:
                        data = await r.json(content_type=None)
                    msg = data['choices'][0]['message']
                except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, KeyError, IndexError) as exc:
                    problem, msg = f'request: {exc!r}', {}
                else:
                    fp = parse(msg.get('content'))
                    problem = valid(fp)
                with open(a.out + '.raw.jsonl', 'a') as f:
                    f.write(json.dumps({'tb': name, 'attempt': attempt, 'problem': problem,
                                        'content': msg.get('content'), 'reasoning': msg.get('reasoning')
                                        or msg.get('reasoning_content')}) + '\n')
                if problem is None:
                    fp = {'tb': name, **{k: fp[k] for k in ('genre', 'conventions', 'choices', 'variant_test')},
                          **({'changes': fp.get('changes', [])} if old else {}),
                          'author': f'{endpoint["served_name"]} (effort {a.effort}' + (', reviewed)' if old else ')')}
                    with open(a.out, 'a') as f:
                        f.write(json.dumps(fp) + '\n')
                    print(f'{time.time() - t0:7.0f}s {name}: {len(fp["choices"])} choices '
                          f'({sum(c["weight"] == "decisive" for c in fp["choices"])} decisive)', flush=True)
                    return True
                print(f'{name} attempt {attempt}: {problem}', flush=True)
        return False

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), trust_env=False) as session:
        ok = await asyncio.gather(*(one(session, n) for n in todo))
    print(json.dumps({'fingerprinted': sum(ok), 'failed': [n for n, o in zip(todo, ok) if not o],
                      'seconds': round(time.time() - t0)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bench', required=True, help='how the prompt names the test set, e.g. "Terminal-Bench 2.1 (TB2.1)"')
    ap.add_argument('--tasks', help='comma list (default: every test task in the bundles)')
    ap.add_argument('--endpoint', required=True)
    ap.add_argument('--effort', default='max')
    ap.add_argument('--max-tokens', type=int, default=32768)
    ap.add_argument('--concurrency', type=int, default=256)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--review', help='fingerprints to review (writes the revised ones to --out)')
    asyncio.run(main(ap.parse_args()))
