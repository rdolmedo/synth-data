"""Spirit check only (never a filter): grade Super's pilot answers with AA's grader template against the generator's
check-only answer, on DeepSeek, to compare the grade mix with Super's on the real test set (M3.6: 22/25/49%).

  python -m m5.omniscience.analysis.grade RUN [--ds m5-ds] [--concurrency 32]
Writes RUN/grades.jsonl and prints the grade mix per domain.
"""
import argparse
import collections
import concurrent.futures as cf
import json
import re
import threading
from pathlib import Path

from aa_evals.omniscience.prompts import LETTER_TO_GRADE, OMNISCIENCE_GRADER_TEMPLATE
from m5.omniscience import fsio
from m5.omniscience.gen.pipeline import DS, Alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--ds', default='m5-ds')
    ap.add_argument('--concurrency', type=int, default=32)
    a = ap.parse_args()
    run = Path(a.run)
    gen = {json.loads(l)['problem_id']: json.loads(l) for l in open(run / 'gen.jsonl') if '"accepted": true' in l}
    tr = [json.loads(l) for l in open(run / 'traces.jsonl')]
    ds, stop = DS(a.ds, Alloc(a.ds)), threading.Event()

    def one(n, t):
        g = gen[t['problem_id']]
        if not t['answer'].strip():
            return t, 'NOT_ATTEMPTED'
        p = OMNISCIENCE_GRADER_TEMPLATE.format(question=g['question'], target=g['answer_check_only'],
                                               predicted_answer=t['answer'])
        resp = ds.chat([{'role': 'user', 'content': p}], n, stop, [])
        c = (resp or {}).get('choices', [{}])[0].get('message', {}).get('content', '') if resp else ''
        m = re.search(r'\b([ABCD])\b', c.strip()[-20:] or '')
        return t, LETTER_TO_GRADE.get(m.group(1), 'UNPARSED') if m else 'UNPARSED'

    out = []
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        for t, g in ex.map(lambda x: one(*x), enumerate(tr)):
            out.append({'problem_id': t['problem_id'], 'domain': t['domain'], 'grade': g})
    fsio.write_atomic(run / 'grades.jsonl', ''.join(json.dumps(r) + '\n' for r in out))
    mix = collections.Counter(r['grade'] for r in out)
    n = len(out)
    print('all', {k: round(v / n, 3) for k, v in mix.most_common()})
    for d in sorted({r['domain'] for r in out}):
        rs = [r for r in out if r['domain'] == d]
        c = collections.Counter(r['grade'] for r in rs)
        print(d, len(rs), {k: round(v / len(rs), 3) for k, v in c.most_common()})


if __name__ == '__main__':
    main()
