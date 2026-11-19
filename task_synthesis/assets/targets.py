"""Sourcing targets from specs, and the routing of a target to catalogs.

A target is what the sourcing aims at: the anchor's class and formats (or language, or services) and the spec's
domain. The agent action stays out (BLUEPRINT.md), so a cell (class x formats x domain) is shared by every spec that
needs it and one asset can serve several tasks.
"""
import json
from collections import Counter
from pathlib import Path

from ..common import ROOT

STUDENT1_BUNDLES = ('r5', 's1', 's2')   # campaign bundles of student 1 (s2b reran s2's cut tasks: same task names)
STUDENT1_ROUNDS = ('r5', 'r6a', 'r6b', 'r6c', 'r6d', 'r6e', 'r6f', 'r6g', 'r6h')


def anchor(spec):
    return next(a for a in spec['assets'] if a['role'] == 'anchor')


def target(spec):
    a = anchor(spec)
    return {'cls': a['class'], 'formats': list(a.get('formats') or []), 'language': a.get('language'),
            'services': list(a.get('services') or []), 'domain': spec['domain'],
            'scale': spec.get('code_scale') or 'none'}


def cell(t):
    detail = t['language'] or ','.join(t['formats']) or ','.join(t['services']) or '-'
    return f'{t["cls"]}|{detail}|{t["domain"]}'


def student1_specs():
    """The best-generated specs of student 1's bundled tasks (the swap candidates), in spec-id order."""
    rounds = ROOT / 'rounds'
    specs = {}
    for r in STUDENT1_ROUNDS:
        for line in (rounds / r / 'specs.jsonl').read_text().splitlines():
            s = json.loads(line)
            specs[s['spec_id']] = {**s, 'round': r}
    used = set()
    for b in STUDENT1_BUNDLES:
        used |= {t.strip().replace('tasksynth-', '') for t in
                 (ROOT / 'campaigns' / b / 'tasks.tar.tasks.txt').read_text().splitlines() if t.strip()}
    return [specs[s] for s in sorted(used) if specs[s]['bucket'] == 'best-generated']


# Catalogs per class, and per format where the class is broad. Earlier entries are tried first; the planner writes
# queries for each catalog in the list. Formats not listed fall back to the class default.
ROUTES = {
    'python-code': ['github-repo'], 'c-cpp-code': ['github-repo'], 'other-code': ['github-repo'],
    'git-repo': ['github-repo'], 'running-service': ['github-repo'], 'upstream-project': ['github-repo'],
    'ml-model': ['hf-model'], 'config': ['github-file', 'github-repo'],
    'binary': {'elf-executable': ['debian', 'github-repo'], 'object-file': ['github-repo', 'debian'],
               'bytecode': ['github-repo'], 'firmware-rom': ['github-repo', 'zenodo'], '*': ['github-repo']},
    'structured-data': {'json': ['github-file', 'zenodo', 'hf-dataset'], 'jsonl': ['hf-dataset', 'zenodo', 'github-file'],
                        'rdf': ['github-file', 'zenodo'], '*': ['github-file', 'zenodo']},
    'tabular-data': {'csv': ['zenodo', 'hf-dataset', 'openml', 'github-file'], 'tsv': ['zenodo', 'github-file', 'hf-dataset'],
                     'excel': ['zenodo'], 'parquet': ['hf-dataset', 'zenodo'], 'feather': ['hf-dataset', 'zenodo'],
                     '*': ['zenodo', 'hf-dataset']},
    'database': {'sqlite': ['zenodo', 'github-file'], 'sql-dump': ['github-file', 'zenodo'],
                 '*': ['github-file', 'zenodo', 'hf-dataset']},
    'text-data': {'plain-text': ['gutenberg', 'wikipedia', 'zenodo'], 'log': ['zenodo', 'github-file'],
                  'gcode': ['github-file'], 'word-list': ['github-file'], 'hex-dump': ['github-file', 'debian'],
                  'calendar-ics': ['github-file'], '*': ['github-file', 'zenodo']},
    'media': {'video': ['commons'], 'audio': ['commons'], '*': ['commons']},
    'document': {'markdown': ['github-file', 'github-repo'], 'pdf': ['europepmc', 'arxiv', 'commons', 'zenodo'],
                 'latex': ['arxiv', 'github-file'], 'html': ['wikipedia', 'github-file'], 'office': ['zenodo'],
                 '*': ['github-file', 'zenodo']},
    'scientific-data': {'fasta': ['ncbi'], 'genbank': ['ncbi'], 'pdb': ['rcsb'], 'cif': ['rcsb', 'cod'],
                        'molecule': ['pubchem', 'github-file'], 'spectrum': ['massbank', 'github-file', 'zenodo'],
                        'numpy': ['zenodo'], 'robot-model': ['github-file', 'github-repo'],
                        'ontology': ['github-file'], '*': ['zenodo']},
    'archive': ['github-repo', 'zenodo'],
    # spec vocabulary v2 (#25) classes (#118): repositories hold these as files among others; the fit check picks the
    # paths to ship. crypto-material is not bound (production.BIND_EXCLUDED_CLASSES): scrub.py redacts private keys.
    'network-capture': ['github-repo', 'zenodo'],
    'deployment-manifest': ['github-repo'],
    'interface-schema': ['github-repo'],
    'software-package': {'debian-binary-package': ['debian', 'github-repo'], '*': ['github-repo']},
}


def routes(t, available):
    r = ROUTES.get(t['cls'], ['github-repo'])
    if isinstance(r, dict):
        seen = []
        for f in t['formats'] or ['*']:
            for s in r.get(f, r['*']):
                if s not in seen:
                    seen.append(s)
        r = seen
    return [s for s in r if s in available]


def summary(specs):
    ts = [target(s) for s in specs]
    return {'specs': len(specs), 'cells': len({cell(t) for t in ts}),
            'by_class': Counter(t['cls'] for t in ts).most_common()}


if __name__ == '__main__':
    print(json.dumps(summary(student1_specs()), indent=1))
