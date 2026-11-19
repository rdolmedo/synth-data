"""Brainstorm subtopics and techniques per CritPt field with Opus (claude -p). No test item is shown.

  python brainstorm.py OUTDIR
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

# field -> (arXiv categories that hold its papers, number of subtopics, number of techniques)
FIELDS = {
    'condensed matter': ('cond-mat.str-el cond-mat.mes-hall cond-mat.supr-con cond-mat.mtrl-sci cond-mat.quant-gas cond-mat.other cond-mat.dis-nn', 55, 22),
    'quantum information and computing': ('quant-ph cs.IT math-ph', 40, 20),
    'high energy and particle physics': ('hep-th hep-ph hep-lat gr-qc', 40, 20),
    'atomic, molecular and optical physics': ('physics.atom-ph physics.optics quant-ph cond-mat.quant-gas physics.chem-ph', 32, 18),
    'mathematical physics': ('math-ph nlin.SI hep-th', 26, 16),
    'statistical physics': ('cond-mat.stat-mech cond-mat.dis-nn', 22, 16),
    'astrophysics and cosmology': ('astro-ph.CO astro-ph.HE astro-ph.SR astro-ph.GA gr-qc', 16, 14),
    'nuclear physics': ('nucl-th hep-ph', 14, 12),
    'biophysics': ('physics.bio-ph q-bio.PE q-bio.QM cond-mat.soft', 14, 12),
    'fluid dynamics': ('physics.flu-dyn nlin.CD', 14, 12),
    'nonlinear dynamics': ('nlin.CD nlin.PS nlin.AO', 14, 12),
}

PROMPT = """I am building a taxonomy for generating research-level theoretical physics problems from arXiv papers in the
field of **{field}** (arXiv categories: {cats}).

Each problem is a self-contained research-grade calculation (like a hard problem a PhD student would set from a paper): a
setup of 100-300 words defining a model or system, and one precise target quantity (a closed-form expression, a
coefficient, an exponent, a critical value, a spectrum, a count, a set of allowed values, a number to a few decimals)
that can be derived with analytic work plus optional light numerics in Python.

Give me:
1. "subtopics": {n} distinct subtopics of {field}. Each is a research area (not a single paper or famous result), broad
   enough that hundreds of arXiv papers up to 2023 study it, and narrow enough to steer the choice of a paper. Spread
   them over the whole breadth of the field (do not cluster on fashionable topics; include classic and less glamorous
   areas). Each: {{"name": 3-7 words, "scope": one sentence, "keywords": 6-10 search keywords likely to appear in such
   papers' titles/abstracts}}.
2. "techniques": {m} distinct calculation techniques typical of theory papers in {field} that end in a precise
   result. Each: {{"name": 3-7 words, "scope": one sentence}}.

Return ONLY a JSON object {{"subtopics": [...], "techniques": [...]}}.
"""


def claude_env() -> dict:
    """The environment for `claude -p` children, without this session's identity/messaging variables."""
    drop = ('CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_CHILD_SESSION', 'CLAUDECODE')
    return {k: v for k, v in os.environ.items() if k not in drop and not k.startswith('CLAUDE_CODE_MESSAGING_')}


def one(field, out):
    cats, n, m = FIELDS[field]
    p = out / (field.split(',')[0].replace(' ', '_') + '.json')
    if p.exists():
        return
    q = PROMPT.format(field=field, cats=cats, n=n, m=m)
    for _ in range(3):
        r = subprocess.run(['claude', '-p', '--model', 'opus'], env=claude_env(), input=q, capture_output=True, text=True, timeout=1800)
        t = r.stdout
        try:
            j = json.loads(t[t.index('{'): t.rindex('}') + 1])
            assert len(j['subtopics']) >= n * 0.8 and len(j['techniques']) >= m * 0.8
            j.update(field=field, categories=cats.split())
            p.write_text(json.dumps(j, indent=1))
            return
        except Exception as e:
            print('retry', field, e, t[:300], flush=True)


if __name__ == '__main__':
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda f: one(f, out), FIELDS))
    print('done', len(list(out.glob('*.json'))))
