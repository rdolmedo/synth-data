"""Chat-template and reasoning-parser checks for a new judge server, before any re-grade (study s3, #41).

  python3 -m aa_evals.judges.s3_probe --arm gemma4-26b-a4b-pin    # prints a JSON verdict; nonzero exit on a failure

Checks, each with the arm's exact request settings:
  render     /tokenize of a system+user chat: the thinking switch reaches the prompt (Gemma: `<|think|>`), the turn
             markers are the model's own, and the generation prompt ends in the model turn
  think      a plain question: reasoning lands in `reasoning(_content)`, the answer in `content`, no channel or turn
             markers leak into either, finish_reason stop
  hle        HLE's strict json_schema judge output with thinking on: content parses and matches the schema
  gdp        GDP.pdf's per-criterion json_schema output: content parses
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request

from aa_evals.hle import prompts as hle_prompts
from aa_evals.judges import arms as arms_mod

LEAK = re.compile(r'<\|?(?:channel|turn|think|start_of_turn|end_of_turn)\|?>|<channel\|>|<turn\|>')


def post(url: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def endpoint(server: str) -> tuple[str, str]:
    from aa_evals.common import serving
    reg = serving.record(server)   # $TASKSYNTH_SERVING_ROOT, else the production registry
    return reg['urls'][0].rstrip('/'), reg['served_name']


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    a = ap.parse_args(argv)
    arm = arms_mod.ARMS[a.arm]
    base, model = endpoint(arm.server)
    base = base if base.endswith('/v1') else base + '/v1'
    root = base[:-3]
    extra = dict(arm.extra)
    res, fails = {'arm': a.arm, 'model': model}, []

    # render
    msgs = [{'role': 'system', 'content': 'You are a careful grader.'}, {'role': 'user', 'content': 'Is 2+2=4?'}]
    tok = post(root + '/tokenize', {'model': model, 'messages': msgs, 'add_generation_prompt': True,
                                    'chat_template_kwargs': extra.get('chat_template_kwargs') or {}})
    text = post(root + '/detokenize', {'model': model, 'tokens': tok['tokens']})['prompt']
    res['render'] = text
    thinking = (extra.get('chat_template_kwargs') or {}).get('enable_thinking')
    if thinking and '<|think|>' not in text:
        fails.append('render: thinking on but no <|think|> in the prompt')
    if not text.rstrip().endswith('model') and '<|turn>model' not in text[-30:]:
        fails.append('render: generation prompt does not end in the model turn')

    def chat(messages, **kw):
        body = {**extra, 'model': model, 'messages': messages, **kw}
        t0 = time.time()
        r = post(base + '/chat/completions', body)
        ch = r['choices'][0]
        msg = ch['message']
        return {'content': msg.get('content'), 'reasoning': msg.get('reasoning') or msg.get('reasoning_content'),
                'finish': ch.get('finish_reason'), 'usage': r.get('usage'), 's': round(time.time() - t0, 1)}

    # think
    out = chat([{'role': 'user', 'content': 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the '
                                            'ball. How much does the ball cost? Answer with just the amount.'}])
    res['think'] = {k: (v[:300] if isinstance(v, str) else v) for k, v in out.items()}
    if thinking and not out['reasoning']:
        fails.append('think: no reasoning returned with thinking on')
    if LEAK.search(out['content'] or '') or LEAK.search(out['reasoning'] or ''):
        fails.append('think: channel/turn markers leaked into content or reasoning')
    if out['finish'] != 'stop' or not out['content'] or '0.05' not in out['content']:
        fails.append(f'think: unexpected answer/finish {out["content"]!r} {out["finish"]}')

    # hle-style strict schema (synthetic item: no HLE data)
    item = {'question': 'What is the boiling point of water at sea level in degrees Celsius?', 'answer': '100'}
    for resp, want in (('Explanation: standard pressure.\nExact Answer: 100 °C\nConfidence: 95%', 'yes'),
                       ('Explanation: I recall.\nExact Answer: 90 °C\nConfidence: 60%', 'no')):
        out = chat(hle_prompts.judge_messages(item, resp), response_format=hle_prompts.JUDGE_RESPONSE_FORMAT)
        try:
            j = hle_prompts.parse_judgment(out['content'])
            ok = j.get('correct') == want
        except ValueError as e:
            j, ok = str(e), False
        res.setdefault('hle', []).append({'want': want, 'got': j, 'finish': out['finish'],
                                          'reasoning_chars': len(out['reasoning'] or ''), 'usage': out['usage']})
        if not ok:
            fails.append(f'hle: schema output wrong or unparseable ({want})')

    # gdp-style per-criterion schema
    from aa_evals.gdppdf import judge as gj
    prompt = gj.judge_prompt('Write a one-line summary of the report.', 'The report finds revenue grew 12% in 2025.',
                             'The summary states the revenue growth figure.')
    out = chat([{'role': 'user', 'content': prompt}], response_format=gj.RESPONSE_FORMAT)
    try:
        score, rationale = gj.parse_verdict(out['content'])
        res['gdp'] = {'score': str(score), 'pass': gj.passes(score), 'finish': out['finish'],
                      'reasoning_chars': len(out['reasoning'] or '')}
        if not gj.passes(score):
            fails.append('gdp: an obviously met criterion failed')
    except Exception as e:  # noqa: BLE001
        res['gdp'] = {'error': str(e)[:200], 'content': (out['content'] or '')[:200]}
        fails.append('gdp: unparseable')
    res['fails'] = fails
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
