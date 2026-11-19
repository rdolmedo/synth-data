"""Fit the CritPt taxonomy: field x subtopic x technique x work mode x answer type (+ grounding).

  python -m m5.critpt.taxonomy.build --brainstorm DIR --testclass DIR --answer-types JSON --out taxonomy.json

- Top level (field): the 11 CritPt fields, weights = multi-label counts over the 70 test challenges (primary 0.65,
  secondary 0.35) + add-1 smoothing. No new top-level options.
- Below: subtopics and techniques brainstormed per field by Opus without seeing any test item (brainstorm.py),
  uniform within a field.
- Single-item leaves: each test challenge's class label (its generic subtopic, from the class-level classification)
  is mapped to the closest brainstormed subtopic of its primary field (Opus, labels only). A subtopic matched by
  exactly one test challenge is broadened (Opus rewrites name/scope/keywords wider, without seeing the item).
- Answer type: fitted over the 70 templates' classes + add-1 smoothing; work mode | answer type fitted from the joint
  with add-0.5 smoothing; grounding fitted (intermediate / main / extension).
- Open draws: OPEN_SHARE of calls take "any subtopic of the field" (<= 5%).
Only class-level statistics enter taxonomy.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

FIELD_ALIASES = {  # classification label -> taxonomy field
    'condensed matter': 'condensed matter',
    'quantum information and computing': 'quantum information and computing',
    'high energy and particle physics (incl. QFT, string theory, gravity)': 'high energy and particle physics',
    'atomic, molecular and optical (AMO)': 'atomic, molecular and optical physics',
    'mathematical physics': 'mathematical physics',
    'statistical physics': 'statistical physics',
    'astrophysics and cosmology': 'astrophysics and cosmology',
    'nuclear physics': 'nuclear physics',
    'biophysics': 'biophysics',
    'fluid dynamics': 'fluid dynamics',
    'nonlinear dynamics': 'nonlinear dynamics',
}
ANSWER_CLASS = {'numeric_scalar': 'numeric_scalar', 'numeric_multi': 'numeric_multi', 'discrete_labels': 'discrete',
                'symbolic_expr': 'symbolic_expr', 'symbolic_multi': 'symbolic_multi',
                'numeric_function': 'numeric_function'}
GROUNDING_CLASS = {'intermediate/derived result': 'intermediate', 'main result of a paper': 'main',
                   'textbook-extension': 'extension'}
OPEN_SHARE = 0.03


def claude_env() -> dict:
    """The environment for `claude -p` children, without this session's identity/messaging variables."""
    import os
    drop = ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')
    return {k: v for k, v in os.environ.items() if k not in drop and not k.startswith('CLAUDE_CODE_MESSAGING_')}


def opus(prompt: str) -> dict:
    for _ in range(3):
        r = subprocess.run(['claude', '-p', '--model', 'opus'], env=claude_env(), input=prompt, capture_output=True, text=True,
                           timeout=1800)
        t = r.stdout
        try:
            return json.loads(t[t.index('{'): t.rindex('}') + 1])
        except ValueError:
            print('retry opus', t[:200], flush=True)
    raise RuntimeError('opus failed')


MAP = """Below is a numbered list of subtopics of {field}, and a list of short subtopic labels. For each label, give
the number of the subtopic that covers it best (or -1 if none covers it at all).

Subtopics:
{subs}

Labels:
{labels}

Return ONLY a JSON object {{"map": [n_1, n_2, ...]}} with one number per label, in order.
"""

BROADEN = """In a taxonomy of {field} research subtopics (used to pick arXiv papers for generating physics problems),
the subtopic below is too narrow. Rewrite it to cover a clearly broader research area (about 3 times wider) that
still sits inside {field} and is distinct from these sibling subtopics: {siblings}.

Subtopic: {name}: {scope}

Return ONLY a JSON object {{"name": 3-7 words, "scope": one sentence, "keywords": 6-10 search keywords}}.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--brainstorm', required=True)
    ap.add_argument('--testclass', required=True)
    ap.add_argument('--answer-types', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    bs = {j['field']: j for j in (json.loads(p.read_text()) for p in Path(a.brainstorm).glob('*.json'))}
    assert set(bs) == set(FIELD_ALIASES.values()), set(FIELD_ALIASES.values()) - set(bs)
    tc = {p.stem: json.loads(p.read_text()) for p in Path(a.testclass).glob('*.json')}
    at = json.loads(Path(a.answer_types).read_text())
    assert len(tc) == 70 and len(at) == 70

    # fields
    w = collections.Counter()
    for r in tc.values():
        fs = [FIELD_ALIASES[f] for f in r['fields'][:2]]
        for i, f in enumerate(fs):
            w[f] += 1 if len(fs) == 1 else (0.65 if i == 0 else 0.35)
    fw = {f: w[f] + 1 for f in bs}
    tot = sum(fw.values())
    fw = {f: v / tot for f, v in fw.items()}

    # single-item subtopics: map class labels (not items) to subtopics, broaden the singly matched ones
    hits = collections.defaultdict(list)
    for f in bs:
        labels = [(pid, r['subtopic']) for pid, r in sorted(tc.items()) if FIELD_ALIASES[r['fields'][0]] == f]
        if not labels:
            continue
        subs = '\n'.join(f'{i}. {s["name"]}: {s["scope"]}' for i, s in enumerate(bs[f]['subtopics']))
        m = opus(MAP.format(field=f, subs=subs, labels='\n'.join(f'- {l}' for _, l in labels)))['map']
        for (pid, _), k in zip(labels, m):
            if isinstance(k, int) and 0 <= k < len(bs[f]['subtopics']):
                hits[(f, k)].append(pid)
    broadened = []
    for (f, k), pids in sorted(hits.items()):
        if len(pids) != 1:
            continue
        s = bs[f]['subtopics'][k]
        sib = ', '.join(x['name'] for j, x in enumerate(bs[f]['subtopics']) if j != k)
        new = opus(BROADEN.format(field=f, siblings=sib, name=s['name'], scope=s['scope']))
        broadened.append({'field': f, 'old': s['name'], 'new': new['name']})
        bs[f]['subtopics'][k] = {**new, 'broadened_from': s['name']}

    # answer type, work mode | answer type, grounding
    acount = collections.Counter(ANSWER_CLASS[v] for v in at.values())
    aw = {k: acount[k] + 1 for k in ANSWER_CLASS.values()}
    tot = sum(aw.values())
    aw = {k: v / tot for k, v in aw.items()}
    modes = sorted({r['work_mode'] for r in tc.values()})
    joint = collections.Counter((ANSWER_CLASS[at[pid]], r['work_mode']) for pid, r in tc.items())
    wm = {}
    for k in aw:
        row = {m: joint[(k, m)] + 0.5 for m in modes}
        s = sum(row.values())
        wm[k] = {m: v / s for m, v in row.items()}
    g = collections.Counter(GROUNDING_CLASS.get(r['paper_grounding'], 'intermediate') for r in tc.values())
    gw = {k: g[k] / sum(g.values()) for k in ('intermediate', 'main', 'extension')}

    out = {'version': 'critpt-tax-v1', 'open_share': OPEN_SHARE, 'fields': {}, 'answer_types': aw,
           'work_mode_given_answer_type': wm, 'grounding': gw, 'broadened': broadened,
           'single_item_check': {'n_subtopics_hit': len(hits), 'hits_per_subtopic':
                                 dict(collections.Counter(len(v) for v in hits.values()))},
           'fit': {'n_test': 70, 'field_counts_weighted': dict(w), 'answer_type_counts': dict(acount),
                   'grounding_counts': dict(g)}}
    for f, j in bs.items():
        subs = [{'id': f'{re.sub("[^a-z]+", "_", f)[:12]}.{i:02d}', **s} for i, s in enumerate(j['subtopics'])]
        out['fields'][f] = {'weight': fw[f], 'categories': j['categories'], 'subtopics': subs,
                            'techniques': j['techniques']}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({'fields': {f: round(v, 3) for f, v in fw.items()}, 'answer_types': aw, 'grounding': gw,
                      'broadened': broadened}, indent=1))


if __name__ == '__main__':
    main()
