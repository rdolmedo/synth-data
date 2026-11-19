#!/usr/bin/env python3
"""Chat-template / parser audit of a pi_eval run's streamed responses (M3.3 smoke, #176).

Per response (RUN/attempts/*/trial.tar.gz -> pi-cont16k/wire-responses/*.response, vLLM SSE): finish reason, parsed
tool calls, and the defects the smoke checks for:
  tool_markup_in_content: "<tool_call>" / "<function=" / "</parameter>" left in content (tool parser missed a call)
  think_in_content:       "<think>" / "</think>" in content (reasoning parser leaked)
  no_reasoning:           a reasoning model's response with an empty reasoning field
  stop_no_tool:           finish_reason stop with no tool call (Pi ends the episode on such a turn)
  length:                 finish_reason length (the per-request output cap)

  python parse_audit.py RUN [RUN ...]
"""
import collections
import glob
import json
import sys
import tarfile
from pathlib import Path


def responses(tar_path):
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            if m.isfile() and '/wire-responses/' in m.name and m.name.endswith('.response'):
                yield m.name, tf.extractfile(m).read().decode(errors='replace')


def parse(sse):
    reas, cont, tools, fin, usage = [], [], 0, None, None
    for line in sse.splitlines():
        if not line.startswith('data: {'):
            continue
        j = json.loads(line[6:])
        for c in j.get('choices', []):
            d = c.get('delta') or {}
            reas.append(d.get('reasoning') or d.get('reasoning_content') or '')
            cont.append(d.get('content') or '')
            tools += sum(1 for t in d.get('tool_calls') or [] if (t.get('function') or {}).get('name'))
            fin = c.get('finish_reason') or fin
        usage = j.get('usage') or usage
    return ''.join(reas), ''.join(cont), tools, fin, usage


def main():
    tot = collections.Counter()
    examples = collections.defaultdict(list)
    for run in sys.argv[1:]:
        for t in sorted(glob.glob(str(Path(run) / 'attempts' / '*' / 'trial.tar.gz'))):
            for name, sse in responses(t):
                reas, cont, tools, fin, usage = parse(sse)
                tot['responses'] += 1
                tot[f'finish_{fin}'] += 1
                tot['tool_calls'] += tools
                tot['completion_tokens'] += (usage or {}).get('completion_tokens') or 0
                tot['reasoning_tokens'] += ((usage or {}).get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
                flags = {'tool_markup_in_content': any(x in cont for x in ('<tool_call>', '<function=', '</parameter>')),
                         'think_in_content': '<think>' in cont or '</think>' in cont,
                         'no_reasoning': not reas.strip(),
                         'stop_no_tool': fin == 'stop' and tools == 0,
                         'length': fin == 'length'}
                for k, v in flags.items():
                    if v:
                        tot[k] += 1
                        if len(examples[k]) < 3:
                            examples[k].append(f'{Path(t).parent.name}:{Path(name).name}: {cont[-160:]!r}')
    print(json.dumps(dict(tot), indent=1))
    for k, v in examples.items():
        print(k)
        for e in v:
            print('   ', e)


if __name__ == '__main__':
    main()
