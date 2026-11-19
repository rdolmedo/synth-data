"""Difficulty proxy on DeepSeek (spirit check for prompt iteration; Super is never used to tune the generator).

DeepSeek V4.1 Flash answers our accepted questions closed-book exactly as in the eval run m35-ds-max-omniscience-full
(AA answer system prompt, bare question, effort max, T 1.0, top_p 0.95), and DeepSeek max grades the answer with AA's
grader template against the generator's check-only answer. Compare with that run's mix on the real test set:
CORRECT 40.1%, INCORRECT 38.4%, PARTIAL 3.4%, NOT_ATTEMPTED 18.1% (1,500 = 300 x 5).

  python -m m5.omniscience.analysis.proxy RUN [--n 300] [--concurrency 60]
Writes RUN/proxy.jsonl and prints the mix per domain and source.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import random
import re
import threading
from pathlib import Path

from aa_evals.omniscience.prompts import LETTER_TO_GRADE, OMNISCIENCE_ANSWER_PROMPT, OMNISCIENCE_GRADER_TEMPLATE
from m5.omniscience import fsio
from m5.omniscience.gen.pipeline import DS, Alloc

MAX = {'reasoning_effort': 'max', 'max_tokens': 129024,
       'chat_template_kwargs': {'thinking': True, 'reasoning_effort': 'max', 'drop_thinking': False}}
JUDGE = dict(MAX, max_tokens=32768)


def content(resp):
    if not resp or resp.get('_error'):
        return ''
    return (resp['choices'][0]['message'].get('content') or '').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--concurrency', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    run = Path(a.run)
    gen = [json.loads(l) for l in open(run / 'gen.jsonl') if '"accepted": true' in l]
    random.Random(a.seed).shuffle(gen)
    gen = gen[:a.n]
    ans, judge, stop = DS('m5-ds', Alloc('m5-ds')), DS('m5-ds', Alloc('m5-ds')), threading.Event()
    ans.overrides, judge.overrides = MAX, JUDGE

    def one(i, g):
        sysm = OMNISCIENCE_ANSWER_PROMPT.format(domain=g['domain'], subtopic=g['subtopic'])
        r = ans.chat([{'role': 'system', 'content': sysm}, {'role': 'user', 'content': g['question']}], i, stop, [])
        pred = content(r)
        if not pred:
            return g, pred, 'NOT_ATTEMPTED', r
        p = OMNISCIENCE_GRADER_TEMPLATE.format(question=g['question'], target=g['answer_check_only'],
                                               predicted_answer=pred)
        c = content(judge.chat([{'role': 'user', 'content': p}], i + 1, stop, []))
        m = re.search(r'\b([ABCD])\b', c[-20:])
        return g, pred, (LETTER_TO_GRADE.get(m.group(1), 'UNPARSED') if m else 'UNPARSED'), r

    rows = []
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for g, pred, grade, r in ex.map(lambda x: one(*x), enumerate(gen)):
            u = (r or {}).get('usage') or {}
            rows.append({'problem_id': g['problem_id'], 'domain': g['domain'], 'source': g['source'],
                         'question': g['question'], 'gold_check_only': g['answer_check_only'], 'pred': pred[:500],
                         'grade': grade, 'completion_tokens': u.get('completion_tokens')})
    fsio.write_atomic(run / 'proxy.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

    def mix(rs):
        c = collections.Counter(r['grade'] for r in rs)
        return {k: round(c[k] / len(rs), 3) for k in ('CORRECT', 'INCORRECT', 'PARTIAL_ANSWER', 'NOT_ATTEMPTED')}
    print('all', len(rows), mix(rows), 'mean tokens',
          round(sum(r['completion_tokens'] or 0 for r in rows) / len(rows)))
    for key in ('domain', 'source'):
        for v in sorted({r[key] for r in rows}):
            rs = [r for r in rows if r[key] == v]
            print(key, v, len(rs), mix(rs))


if __name__ == '__main__':
    main()
