"""Block taxonomy leaves that would lead the generator to a variant of a SciCode test problem (seed decontamination;
#19's lesson: one leaf drawn from a test item produced most flags). Candidate (leaf, test) pairs: each test problem's
top-20 leaves by TF-IDF (title + gist vs the test problem's name, main description and steps) plus every leaf whose
title matches a keyword of the test problem's core subject (KEYWORDS, hand-written from the test problems). A
DeepSeek judge (1 sample, high effort) answers BLOCK / KEEP; BLOCK leaves go to taxonomy/leaf_block.json.

    python -m m5.scicode.leaf_block run        # WORK/leaf_judge.jsonl
    python -m m5.scicode.leaf_block write      # taxonomy/leaf_block.json
"""
import asyncio
import json
import re
import sys

import numpy as np

from . import taxonomy as T
from .common import TEST_PROBLEMS, Alloc, append_line, read_jsonl, safe_write

KEYWORDS = {
 '5': r'\blanczos\b', '9': r'weighted jacobi|damped jacobi', '11': r'amplitude.damping', '71': r'amplitude.damping',
 '12': r'hartree|self-consistent.*(atom|radial)', '14': r'tweezer', '32': r'tweezer', '15': r'crank.?nicolson',
 '16': r'davidson', '17': r'tetrahedron', '18': r'nurbs|b-spline basis|cox.?de.?boor', '20': r'phonon angular momentum|chiral phonon',
 '21': r'algaas|gaalas|al_?xga|alga', '22': r'translation|re-?expansion|addition theorem|vector spherical', '23': r'blahut',
 '24': r'burgers', '25': r'consumer.resource|macarthur', '26': r'consumer.resource|serial (dilution|transfer)',
 '41': r'serial (dilution|transfer)|structural stability|feasibility', '56': r'serial (dilution|transfer)|temporal niche',
 '27': r'photodetector|p-i-n|pin diode', '28': r'gaussian beam', '30': r'helium|jastrow', '46': r'helium|jastrow',
 '68': r'helium|jastrow|diffusion monte carlo|\bdmc\b', '31': r'independent component|\bica\b|fastica', '33': r'haldane',
 '34': r'p-?n junction|pn diode|depletion', '35': r'quantum.dot', '36': r'quasi.fermi', '37': r'spherical aberration|doublet',
 '39': r'bragg reflector|\bdbr\b|quarter.wave', '40': r'strang|reaction.diffusion', '42': r'threshold current|quantum.well laser',
 '43': r'fiber laser|fibre laser', '45': r'heat equation', '48': r'eels|energy.loss', '50': r'sherrington|kirkpatrick|replica',
 '52': r'hydrogen.*shooting|shooting.*hydrogen', '57': r'harmonic oscillator.*(numerov|shooting)|(numerov|shooting).*harmonic',
 '53': r'lotka|predator', '54': r'supg|nitsche|streamline.upwind', '55': r'swift.hohenberg', '58': r'\btov\b|tolman|oppenheimer',
 '59': r'\bvqe\b|variational quantum eigensolver', '60': r'widom', '61': r'bragg peak|ub matrix|busing|four.circle|indexing',
 '73': r'bragg peak|ub matrix|busing|four.circle|indexing', '62': r'dmrg', '63': r'black.?scholes|option', '64': r'grand.canonical|gcmc',
 '65': r'\bghz\b', '66': r'kolmogorov.crespi|interlayer.*graphene', '67': r'layered electron gas|electron.gas.*(plasmon|layer)',
 '69': r'layered electron gas|raman', '72': r'\bising\b.*(metropolis|monte carlo|transition|critical)|(metropolis|monte carlo).*\bising\b',
 '74': r'householder', '75': r'bilayer graphene|aa.stacked', '76': r'position weight matrix|\bpwm\b|binding site|motif', '77': r'berendsen',
 '79': r'nos[eé].?hoover', '80': r'anders[eo]n thermostat',
}

RUBRIC = """You screen topic seeds for a synthetic training set of scientific programming problems. A held-out benchmark (SciCode) must not leak into training. A problem writer will receive only the seed (a one-line topic and gist) and write a multi-step coding problem on it. You see one benchmark test problem and one seed. Decide whether a problem written from this seed would very likely be a **variant of this specific test problem**: the same specific system or model with the same core computation or observable (in any wording, with any parameter values), or the test problem's computation contained as a main part. Sharing only the field, the method family or a textbook technique is not enough (e.g. another Monte Carlo simulation, another ODE shooting problem on a different system, another use of the same algorithm on a clearly different problem is KEEP).

Answer with only a JSON object: {"verdict": "BLOCK" | "KEEP", "reason": "<one sentence>"}"""


def test_text(p):
    return '\n'.join([f"Name: {p['problem_name']}", p['problem_description_main']]
                     + [f"Step {i + 1}: {s['step_description_prompt']}" for i, s in enumerate(p['sub_steps'])])


def pairs():
    from sklearn.feature_extraction.text import TfidfVectorizer
    rows = [x for _, x in T.all_leaf_rows()]
    tests = read_jsonl(TEST_PROBLEMS)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english')
    X = vec.fit_transform([f"{x['title']}. {x['gist']}" for x in rows] + [test_text(p) for p in tests])
    S = (X[:len(rows)] @ X[len(rows):].T).toarray()
    out = set()
    for j, p in enumerate(tests):
        for i in np.argsort(-S[:, j])[:20]:
            out.add((int(i), p['problem_id']))
        pat = KEYWORDS.get(p['problem_id'])
        if pat:
            for i, x in enumerate(rows):
                if re.search(pat, x['title'].lower()):
                    out.add((i, p['problem_id']))
    return rows, {p['problem_id']: p for p in tests}, sorted(out)


async def run():
    from .generate import DS
    rows, tests, ps = pairs()
    path = T.WORK / 'leaf_judge.jsonl'
    done = {(r['i'], r['test']) for r in read_jsonl(path)}
    todo = [p for p in ps if p not in done]
    print(f'{len(ps)} pairs, {len(todo)} to judge', flush=True)
    ds, alloc = DS(), Alloc()

    async def one(i, t):
        user = f"## Test problem\n{test_text(tests[t])}\n\n## Seed\n{rows[i]['title']}: {rows[i]['gist']}"
        for _ in range(3):
            g = await ds.chat(user, system=RUBRIC)
            m = re.search(r'\{.*\}', g['content'] or '', re.S)
            try:
                v = json.loads(m.group(0))
                if v.get('verdict') in ('BLOCK', 'KEEP'):
                    append_line(path, json.dumps({'i': i, 'test': t, 'title': rows[i]['title'], **v}))
                    return
            except (AttributeError, ValueError):
                pass

    pending = set()
    for i, t in todo:
        while len(pending) >= max(1, alloc.ds_cap()):
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        pending.add(asyncio.create_task(one(i, t)))
    await asyncio.gather(*pending)


# hand-added after reading the judge's KEEPs (2026-11-12): the test problems' crux on the same system, whatever the judge said
MANUAL = [r'haldane.*(chern|berry)', r'helium.*(vmc|dmc|jastrow|monte carlo|variational)', r'black.?scholes.*(finite|crank|grid|adi)',
          r'widom (insertion|chemical|excess)', r'(dbr|bragg reflector|quarter.wave).*(transfer|reflect|stopband)',
          r'nos[eé].?hoover.*harmonic', r'berendsen', r'anders[eo]n thermostat', r'dmrg.*(heisenberg|xxz)',
          r'consumer.resource.*(chemostat|serial|survivor)', r'(generalized|generalised) amplitude damping',
          r'coherent information.*amplitude', r'linear tetrahedron', r'blahut.arimoto (capacity|.*channel capacity)',
          r'tolman|\btov\b', r'swift.hohenberg.*(stripe|roll pattern)', r'stochastic lotka|lotka.volterra.*(gillespie|stochastic|extinction)',
          r'sherrington.kirkpatrick.*(monte carlo|metropolis|overlap|parallel tempering)', r'kolmogorov.crespi']


def write():
    js = read_jsonl(T.WORK / 'leaf_judge.jsonl')
    block = {r['title'] for r in js if r['verdict'] == 'BLOCK'}
    manual = {x['title'] for _, x in T.all_leaf_rows() if any(re.search(p, x['title'].lower()) for p in MANUAL)}
    print(len(manual - block), 'added by the manual patterns')
    block = sorted(block | manual)
    safe_write(T.HERE / 'leaf_block.json', json.dumps(
        {'note': 'Leaf titles blocked as seeds: a DeepSeek judge found a problem written from them would likely be a '
                 'variant of a SciCode test problem (m5/scicode/leaf_block.py).', 'n_judged': len(js),
         'block_titles': block}, indent=1, ensure_ascii=False))
    print(len(js), 'judged,', len(block), 'blocked')


if __name__ == '__main__':
    asyncio.run(run()) if sys.argv[1] == 'run' else write()
