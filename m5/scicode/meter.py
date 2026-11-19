"""Pilot metrics for generated specs (and their Super traces).

    python -m m5.scicode.meter --gen DIR [--traces OUT] [--n-prod 9300]

- repeat meter: TF-IDF (word 1-2 grams) cosine between problems (name + main description + step descriptions);
  near-duplicate pairs above --dup-cos; the pairwise near-dup probability p projects to production as the share of
  problems with at least one near-duplicate among N: 1 - (1 - p)^(N - 1);
- 8-gram overlap of each spec with the generator prompt (its format examples) and with its own leaf text (the seed);
- taxonomy spread: realized field / subfield / method / step-count shares of accepted specs vs the targets;
- traces: submit share (every step has a ```python block), context-limit share, tokens per chain.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

from .common import read_jsonl, safe_write

HERE = Path(__file__).resolve().parent


def spec_text(s):
    return ' '.join([s['problem_name'].replace('_', ' '), s['problem_description_main']]
                    + [x['step_description_prompt'] for x in s['sub_steps']])


def ngrams(t, n=8):
    w = re.findall(r'\w+', t.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def repeat_meter(texts, dup_cos, n_prod):
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english').fit_transform(texts)
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0)
    n = len(texts)
    iu = np.triu_indices(n, 1)
    pairs = int((S[iu] > dup_cos).sum())
    p = pairs / max(1, len(iu[0]))
    nn = S.max(1)
    return {'n': n, 'dup_cos': dup_cos, 'pairs_above': pairs, 'share_with_dup': float((nn > dup_cos).mean()),
            'nn_cos_mean': float(nn.mean()), 'nn_cos_p90': float(np.quantile(nn, 0.9)),
            'pairwise_p': p, 'projected_share_with_dup': 1 - (1 - p) ** (n_prod - 1), 'n_prod': n_prod}, S


def spread(specs, vocab):
    out = {}
    tgt_cell = vocab['cells']
    real = collections.Counter(s['cell'] for s in specs)
    n = len(specs)
    for level, key in (('field', lambda c: c.split('/')[0]), ('subfield', lambda c: c.split('|')[0]),
                       ('method', lambda c: c.split('|')[1])):
        t, r = collections.Counter(), collections.Counter()
        for c, p in tgt_cell.items():
            t[key(c)] += p
        for c, k in real.items():
            r[key(c)] += k / n
        out[level] = {k: {'target': round(t[k], 4), 'real': round(r[k], 4)} for k in sorted(t)}
        out[level + '_tv'] = round(0.5 * sum(abs(t[k] - r[k]) for k in t), 4)
    st = collections.Counter(len(s['sub_steps']) for s in specs)
    out['steps'] = {k: {'target': round(vocab['p_steps'][k], 4), 'real': round(st[int(k)] / n, 4)}
                    for k in vocab['p_steps']}
    out['mean_steps'] = {'target': sum(int(k) * v for k, v in vocab['p_steps'].items()),
                         'real': sum(len(s['sub_steps']) for s in specs) / n}
    out['open_draw_share'] = sum(1 for s in specs if not s.get('leaf_id')) / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True)
    ap.add_argument('--traces', default=None)
    ap.add_argument('--n-prod', type=int, default=9300)
    ap.add_argument('--dup-cos', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    g = Path(a.gen)
    specs = read_jsonl(g / 'specs.jsonl')
    draws = {d['draw']: d for d in read_jsonl(g / 'draws.jsonl')}
    rej = read_jsonl(g / 'rejects.jsonl')
    vocab = json.loads((HERE / 'taxonomy' / 'vocab.json').read_text())
    rep = {'accepted': len(specs), 'rejected': len(rej), 'drawn': len(draws),
           'reject_reasons': collections.Counter(e.split(':')[0] for r in rej for e in r['errors']).most_common(15)}
    texts = [spec_text(s) for s in specs]
    rep['repeat'], S = repeat_meter(texts, a.dup_cos, a.n_prod)
    i, j = np.unravel_index(np.argsort(-S, axis=None)[:20:2], S.shape)
    rep['top_pairs'] = [(specs[x]['problem_id'], specs[y]['problem_id'], round(float(S[x, y]), 3),
                         specs[x]['problem_name'], specs[y]['problem_name']) for x, y in zip(i, j)]
    pg = ngrams((HERE / 'gen_prompt.md').read_text())
    ov_p, ov_l = [], []
    for s in specs:
        ng = ngrams(spec_text(s) + ' ' + ' '.join(x['function_header'] for x in s['sub_steps']))
        ov_p.append(len(ng & pg) / max(1, len(ng)))
        leaf = (draws.get(s['draw']) or {}).get('leaf')
        if leaf:
            ov_l.append(len(ng & ngrams(leaf['title'] + ' ' + leaf['gist'])) / max(1, len(ng)))
    rep['ngram8_overlap'] = {'prompt_mean': float(np.mean(ov_p)), 'prompt_max': float(np.max(ov_p)),
                             'prompt_share_any': float(np.mean([x > 0 for x in ov_p])),
                             'leaf_mean': float(np.mean(ov_l)) if ov_l else None,
                             'leaf_max': float(np.max(ov_l)) if ov_l else None}
    rep['spread'] = spread(specs, vocab)
    rep['spec_len'] = {'main_chars': float(np.mean([len(s['problem_description_main']) for s in specs])),
                       'step_chars': float(np.mean([len(x['step_description_prompt']) for s in specs for x in s['sub_steps']])),
                       'deps_lines': float(np.mean([len(s['required_dependencies'].splitlines()) for s in specs]))}
    gu = [s.get('gen_usage') or {} for s in specs]
    rep['gen_tokens'] = {'prompt_mean': float(np.mean([u.get('prompt_tokens') or 0 for u in gu])),
                         'completion_mean': float(np.mean([u.get('completion_tokens') or 0 for u in gu]))}
    if a.traces:
        ch = read_jsonl(Path(a.traces) / 'chains.jsonl')
        if ch:
            tot = [c['prompt_tokens'] + c['completion_tokens'] for c in ch]
            steps = read_jsonl(Path(a.traces) / 'steps.jsonl')
            rep['traces'] = {
                'chains': len(ch), 'tokens_mean': float(np.mean(tot)), 'tokens_median': float(np.median(tot)),
                'tokens_p90': float(np.quantile(tot, 0.9)), 'tokens_total': int(sum(tot)),
                'completion_mean': float(np.mean([c['completion_tokens'] for c in ch])),
                'submit_share_chain': float(np.mean([c['submitted_all'] for c in ch])),
                'cutoff_share_chain': float(np.mean([c['cutoff_any'] for c in ch])),
                'steps': len(steps),
                'submit_share_step': float(np.mean([bool(s['content']) and '```' in s['content'] for s in steps])),
                'cutoff_share_step': float(np.mean([s['finish_reason'] in ('length', 'prompt_over_context') for s in steps])),
                'step_completion_mean': float(np.mean([(s['usage'] or {}).get('completion_tokens') or 0 for s in steps])),
                'step_completion_max': int(max((s['usage'] or {}).get('completion_tokens') or 0 for s in steps)),
            }
    txt = json.dumps(rep, indent=1, default=str)
    if a.out:
        safe_write(Path(a.out), txt)
    print(txt)


if __name__ == '__main__' and not (len(sys.argv) > 1 and sys.argv[1] == 'spirit'):
    main()


def form_stats(probs):
    """Format / length / answer-type statistics of a list of SciCode-format problems (test or synthetic)."""
    import statistics as stt
    steps = [s for p in probs for s in p['sub_steps']]
    outs = collections.Counter()
    for s in steps:
        h = s.get('function_header') or ''
        hl = h.lower()
        cut = max(hl.rfind('output'), hl.rfind('return'))
        o = hl[cut:] if cut >= 0 else hl
        kind = ('array' if re.search(r'array|ndarray|matrix|vector|shape', o) else
                'float' if re.search(r'float|scalar|real', o) else
                'int' if re.search(r'\bint', o) else
                'complex' if 'complex' in o else
                'list/tuple/dict' if re.search(r'list|tuple|dict', o) else
                'bool' if 'bool' in o else 'other')
        outs[kind] += 1
    deps = [len([l for l in p['required_dependencies'].splitlines() if l.strip()]) for p in probs]
    scipy = sum('scipy' in p['required_dependencies'] for p in probs)
    return {
        'problems': len(probs), 'steps_mean': stt.mean(len(p['sub_steps']) for p in probs),
        'steps_median': stt.median(len(p['sub_steps']) for p in probs),
        'main_chars_mean': stt.mean(len(p['problem_description_main']) for p in probs),
        'step_chars_mean': stt.mean(len(s['step_description_prompt']) for s in steps),
        'step_chars_median': stt.median(len(s['step_description_prompt']) for s in steps),
        'header_chars_mean': stt.mean(len(s.get('function_header') or '') for s in steps),
        'class_step_share': sum((s.get('function_header') or '').lstrip().startswith('class') for s in steps) / len(steps),
        'latex_step_share': sum('$' in s['step_description_prompt'] for s in steps) / len(steps),
        'deps_lines_mean': stt.mean(deps), 'scipy_share': scipy / len(probs),
        'output_kind_share': {k: round(v / len(steps), 3) for k, v in outs.most_common()},
    }


def spirit_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', required=True)
    a = ap.parse_args(sys.argv[2:])
    from .common import TEST_PROBLEMS
    test = read_jsonl(TEST_PROBLEMS)
    syn = read_jsonl(Path(a.gen) / 'specs.jsonl')
    print(json.dumps({'test': form_stats(test), 'synthetic': form_stats(syn)}, indent=1))


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'spirit':
    spirit_main()
