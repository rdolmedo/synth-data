"""Class-level spirit check of generated CritPt problems vs the 70 test challenges, plus 8-gram overlap.

  python -m m5.critpt.spirit PROBLEMS.jsonl [PROBLEMS2.jsonl ...]

Prints only aggregate numbers and the public example's shared 8-grams: no test text is printed or written.
- mix: field / answer type / grounding / work mode of the generated set vs the taxonomy targets (fitted to the 70).
- length: words in the description, setup and main parts (generated vs the 70).
- 8-gram (word level, lowercased, LaTeX kept as tokens) of each description vs the public example challenge (the
  generator's format example) and vs each test challenge (early contamination signal; the full retrieval + judge
  decontamination runs once at the end).
- spread: distinct fields / subtopics / techniques / (subtopic, technique) leaves, max share of one subtopic.
"""
import collections
import json
import re
import statistics as S
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from aa_evals.critpt import data  # noqa: E402

TOK = re.compile(r'\\[A-Za-z]+|[A-Za-z]+|\d+|[^\sA-Za-z\d]')


def grams(text, n=8):
    t = [x.lower() for x in TOK.findall(text)]
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def parts(d):
    m = re.search(r'#\s*Main problem:?', d)
    return (d[:m.start()], d[m.end():]) if m else (d, '')


def words(s):
    return len(s.split())


def main(argv):
    runs = argv[argv.index('--runs') + 1:] if '--runs' in argv else []
    files = argv[:argv.index('--runs')] if runs else argv
    P = [json.loads(l) for f in files for l in open(f)]
    tax = json.loads((REPO / 'm5/critpt/taxonomy/taxonomy.json').read_text())
    tests = data.load_problems()
    ex = data.load_example()
    n = len(P)
    out = {'n_generated': n}

    def mix(key, target):
        c = collections.Counter(p['leaf'][key] for p in P)
        return {k: {'gen': round(c.get(k, 0) / n, 3), 'target': round(v, 3)} for k, v in target.items()}
    out['field'] = mix('field', {k: v['weight'] for k, v in tax['fields'].items()})
    out['answer_type'] = mix('answer_type', tax['answer_types'])
    out['grounding'] = mix('grounding', tax['grounding'])
    out['work_mode'] = {k: round(v / n, 3) for k, v in collections.Counter(p['leaf']['work_mode'] for p in P).most_common()}
    tv = lambda xs: {'median': S.median(xs), 'p10': sorted(xs)[len(xs) // 10], 'p90': sorted(xs)[9 * len(xs) // 10]}
    out['words'] = {
        'generated': {'total': tv([words(p['description']) for p in P]),
                      'setup': tv([words(parts(p['description'])[0]) for p in P]),
                      'main': tv([words(parts(p['description'])[1]) for p in P])},
        'test70': {'total': tv([words(t.description) for t in tests]),
                   'setup': tv([words(parts(t.description)[0]) for t in tests]),
                   'main': tv([words(parts(t.description)[1]) for t in tests])}}
    eg = grams(ex.description)
    tg = [grams(t.description) for t in tests]
    ex_hits, shared_ex, test_max, test_any = [], collections.Counter(), [], 0
    for p in P:
        g = grams(p['description'])
        h = g & eg
        ex_hits.append(len(h) / max(1, len(g)))
        shared_ex.update(' '.join(x) for x in h)
        m = max(len(g & t) for t in tg)
        test_max.append(m)
        test_any += m > 0
    out['8gram_vs_example'] = {'problems_with_any': sum(x > 0 for x in ex_hits), 'max_share': round(max(ex_hits), 4),
                               'mean_share': round(S.mean(ex_hits), 5),
                               'top_shared': shared_ex.most_common(8)}
    out['8gram_vs_test70'] = {'problems_with_any_shared_8gram': test_any,
                              'max_shared_with_one_test_item': max(test_max),
                              'problems_sharing_ge5': sum(m >= 5 for m in test_max)}
    sub = collections.Counter(p['leaf']['subtopic_id'] for p in P)
    out['spread'] = {'fields': len({p['leaf']['field'] for p in P}), 'subtopics': len(sub),
                     'subtopics_in_taxonomy': sum(len(v['subtopics']) for v in tax['fields'].values()),
                     'techniques': len({p['leaf']['technique'] for p in P}),
                     'leaves': len({(p['leaf']['subtopic_id'], p['leaf']['technique']) for p in P}),
                     'max_subtopic_share': round(max(sub.values()) / n, 3), 'papers': len({p['arxiv_id'] for p in P}),
                     'open_draws': sum(p['leaf']['subtopic_id'].endswith('::open') for p in P)}
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
