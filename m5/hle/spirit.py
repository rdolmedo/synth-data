"""Spirit match: Super's stated confidence and trace length on our exported problems vs on HLE's text items (the M3.6
Super run), per partition x answer type, plus question length (words) vs HLE-Diamond's quantiles (fit.json).
Also the exported set's partition x category x answer-type spread against the fitted targets (TV distance).
Only aggregates are written.

    python -m m5.hle.spirit --traces T/traces.jsonl --problems 'P/b*.jsonl' [--excluded W/excluded.json] --out S.json
"""
import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HLE_RUN = '/data/aai/aa_evals/runs/hle/m36-super-vl-hle-full'
CONF = re.compile(r'Confidence\s*:\s*\**\s*(\d+(?:\.\d+)?)\s*%')


def conf(text):
    m = CONF.findall(text or '')
    return min(100.0, float(m[-1])) if m else None


def q(xs, ps=(0.25, 0.5, 0.75)):
    xs = sorted(xs)
    return [round(xs[min(len(xs) - 1, int(p * len(xs)))], 1) for p in ps] if xs else None


def summ(cells):
    out = {}
    for k, v in sorted(cells.items()):
        c = [x for x in v['conf'] if x is not None]
        out[k] = {'n': len(v['tok']), 'conf_mean': round(st.mean(c), 1) if c else None, 'conf_q': q(c),
                  'conf_parsed': len(c), 'tokens_mean': round(st.mean(v['tok'])) if v['tok'] else None,
                  'tokens_q': q(v['tok'])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--problems', required=True)
    ap.add_argument('--excluded')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    excl = set(json.loads(Path(a.excluded).read_text())['excluded']) if a.excluded else set()
    probs = {}
    for f in sorted(glob.glob(a.problems)):
        for l in open(f):
            r = json.loads(l)
            probs[r['id']] = r
    from collections import Counter
    from .traces import cell_of, spread_targets
    ours, words, cells = defaultdict(lambda: {'conf': [], 'tok': []}), defaultdict(list), Counter()
    for l in open(a.traces):
        try:
            t = json.loads(l)
        except ValueError:
            continue
        if not t.get('kept') or t['id'] in excl:
            continue
        k = f"{t['spec']['partition']}/{t['spec']['answer_type']}"
        cells[cell_of(t)] += 1
        ours[k]['conf'].append(conf(t['content']))
        ours[k]['tok'].append(t['completion_tokens'])
        words[k].append(len((probs.get(t['id']) or {}).get('question', '').split()))
    hle = defaultdict(lambda: {'conf': [], 'tok': []})
    for l in open(f'{HLE_RUN}/items.jsonl'):
        r = json.loads(l)
        if r.get('has_image') or r.get('generation_status') != 'ok' or not r.get('transcript'):
            continue
        g = json.loads(Path(r['transcript']).read_text())['generation']
        k = f"{r['partition']}/{r['answer_type']}"
        hle[k]['conf'].append(conf(g.get('content')))
        hle[k]['tok'].append((g.get('usage') or {}).get('completion_tokens') or 0)
    fit = json.loads((HERE / 'data' / 'fit.json').read_text())
    tgt, n = spread_targets(), sum(cells.values())
    tv = 0.5 * sum(abs(cells.get(c, 0) / n - tgt.get(c, 0)) for c in set(tgt) | set(cells))
    worst = sorted(((cells.get(c, 0) / n - tgt.get(c, 0), '/'.join(c)) for c in set(tgt) | set(cells)), key=lambda x: -abs(x[0]))[:6]
    res = {'spread_tv_exported': round(tv, 4), 'spread_worst_cells': [[c, round(d, 4)] for d, c in worst],
           'ours': summ(ours), 'hle_text_items_m36_super': summ(hle),
           'question_words_ours_q': {k: q(v) for k, v in sorted(words.items())},
           'question_words_hle_diamond_q': fit.get('length_words') or fit.get('lengths'),
           'n_excluded_dropped': len(excl)}
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
