"""Brainstorm the v2 spec vocabulary with Opus 5.5 (#25). Stages, each cached and resumable:

  toplevel    new actions, domains, asset classes: a first list, up to two "what is still missing" passes, then a
              separate review call (keep / merge / drop; classes may move to formats of an existing class)
  subdomains  areas within every domain (current + kept new): a first list, extension passes until little new
              comes back, then a cleaning call
  formats     more formats / services / states per non-code class (current + new), each tagged with the domains
              where it is common; then a cleaning call
  languages   ~80 languages for other-code with a rough real-world share and offline installability
  compat      per asset class (and for no materials): how naturally each action and domain combines with it (0/1/2)
  pairs       per new asset class: how naturally each other class appears alongside it (0/1/2), the companion lift
  coverage    classify a sample of student 1's tasks into their domain's areas; propose areas for the uncovered

  python3 -m task_synthesis.vocab.brainstorm toplevel   (then subdomains, formats, languages, compat, coverage)
Outputs: WORK/<stage>/*.json (WORK = /data/aai/tbench/tasksynth/vocab/v2)."""
import argparse
import collections
import importlib.util
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..pipeline.prompts import ACTIONS
from .opus import WORK, ask_json

HERE = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('sample_assets', HERE / 'asset_sampler' / 'sample_assets.py')
SA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SA)
LABELS = json.loads((HERE / 'asset_sampler' / 'anchor_labels.json').read_text())

CLASS_DEFS = {   # TB_ASSET_CLASSES.md
    'python-code': 'Python codebase, package or scripts', 'c-cpp-code': 'C or C++ source',
    'other-code': 'source in another language', 'git-repo': 'a git repository whose history, branches or state matter',
    'upstream-project': 'a third-party open-source project fetched or vendored',
    'tabular-data': 'tables: CSV, TSV, Parquet, Excel', 'structured-data': 'JSON, JSONL, YAML, XML used as data',
    'text-data': 'plain-text corpora, logs, prose, puzzles', 'database': 'SQLite or server databases, dumps',
    'binary': 'executables, ELF, object files, bytecode, ROMs', 'media': 'images, video, audio',
    'document': 'PDF, LaTeX, HTML, Office files', 'ml-model': 'model weights, checkpoints, tokenizers',
    'archive': 'archives the agent must open, inspect or repair (zip, 7z, tar)',
    'config': 'system or service configuration as the object of work',
    'running-service': 'a live service or multi-service stack started in the environment',
    'scientific-data': 'domain formats such as FASTA, PDB, MJCF, spectra', 'vm-image': 'VM, disk, ISO or emulator images'}
# Details that name a task's defect or trick rather than a kind of file (#19: the only real leaks came from these).
MERGE_DETAILS = {'database': {'wal-journal': 'sqlite'},
                 'git-repo': {'lost-commit': 'rich-history', 'secret-in-history': 'rich-history'},
                 'archive': {'encrypted': None, 'polyglot': None}}
FIELD_WORDS = {'formats': 'formats', 'services': 'service types', 'state': 'repository states',
               'acquisition': 'acquisition modes'}

SYSTEM = """You help design the vocabulary for a generator of synthetic training tasks for AI agents that work in a Linux terminal.

Each task is a realistic, self-contained job that an engineer, analyst or scientist would do in a shell, in the spirit of Terminal-Bench: real technical work, checked by inspecting the end state. Every task runs in a sandbox: an Ubuntu 24.04 container with 2 CPUs, 8 GB RAM, 10 GB disk, no GPU and no internet; packages install from an offline mirror of Ubuntu apt, PyPI and npm. The files, data or services a task needs are created at build time by a setup script (with common tools and libraries) or shipped as small text files. An expert needs from 15 minutes to a few hours.

The generator receives a spec drawn from this vocabulary: what the agent mainly does (action), the field the task is set in (domain, and an area within it), and the materials in the environment (asset classes, each with formats). The current vocabulary was fitted to Terminal-Bench's own tasks. We are broadening it so that generated tasks cover more of real terminal work, while staying close to Terminal-Bench in spirit and difficulty.

Vocabulary entries are categories, never individual tasks: do not describe or name specific benchmark tasks."""

RULES = """Rules:
1. Broad: at least as broad as a typical current option; it must fit hundreds of different tasks.
2. Distinct: no overlap with a current option or with another proposal; a typical task gets exactly one {dim} without hesitation.
3. Feasible in the sandbox described above.
4. Not niche: a real, common kind of {what}."""


def share(key):
    c = collections.Counter(r[key] for r in LABELS if r[key])
    n = sum(c.values())
    return {k: v / n for k, v in c.most_common()}


def anchor_share():
    c = collections.Counter(r['anchor_class'] or 'none' for r in LABELS)
    n = sum(c.values())
    return {k: v / n for k, v in c.most_common() if k != 'none'}


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def pmap(fn, items, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------------------------------------------- top level
def current(kind):
    if kind == 'action':
        sh = share('agent_action')
        return [(a, ACTIONS[a], sh.get(a, 0)) for a in ACTIONS]
    if kind == 'domain':
        return [(d, '', s) for d, s in share('domain').items()]
    sh = anchor_share()
    return [(c, CLASS_DEFS[c], sh.get(c, 0)) for c in SA.CLASSES]


DIM = {'action': ('action (what the agent mainly does)', 'action', 'terminal work'),
       'domain': ('domain (the field, industry or subject area the task is set in)', 'domain', 'field'),
       'class': ('asset class (the kind of material in the environment that the agent works with)', 'class',
                 'material')}


def listing(items):
    return '\n'.join(f'- {n} ({s:.0%})' + (f': {d}' if d else '') for n, d, s in items)


def toplevel_one(kind):
    title, dim, what = DIM[kind]
    cur = current(kind)
    extra = ''
    if kind == 'class':
        extra = ('\nProduction uses only materials that a setup script can create offline (or small shipped text '
                 'files), so a class whose instances can only be downloaded does not work here. Many candidates are '
                 'better as a new format of a current class (e.g. a new image format belongs to media); propose '
                 'only kinds of material that no current class covers.\n')
    head = f'Dimension: {title}.\n\nCurrent options, with their share of Terminal-Bench tasks:\n{listing(cur)}\n{extra}'
    fmt = ('Reply with one ```json block: a list of {"name": "kebab-case", "description": "one line, in the style of '
           'the current descriptions", "examples": ["three short, varied task ideas"]}.')
    props = ask_json(SYSTEM, f'{head}\nPropose new {dim} options that the current list does not cover. '
                     f'{RULES.format(dim=dim, what=what)}\nPropose up to 15; fewer is fine if fewer qualify.\n\n{fmt}',
                     f'top-{kind}-p0')
    for p in (1, 2):
        so_far = '\n'.join(f'- {x["name"]}: {x["description"]}' for x in props)
        more = ask_json(SYSTEM, f'{head}\nProposed so far:\n{so_far}\n\nIs any broad, common kind of {what} still '
                        f'missing? Propose only further {dim} options that meet the same rules; reply with [] if none.\n'
                        f'{RULES.format(dim=dim, what=what)}\n\n{fmt}', f'top-{kind}-p{p}')
        names = {x['name'] for x in props}
        new = [x for x in more if x['name'] not in names]
        props += new
        if len(new) < 2:
            break
    cands = '\n'.join(f'- {x["name"]}: {x["description"]} (e.g. {"; ".join(x.get("examples", [])[:3])})' for x in props)
    into = 'merge (into a current option or another candidate; name it), or drop'
    if kind == 'class':
        into = ('merge (into a current class or another candidate; name it), format (better as a format of a current '
                'class; name the class), or drop')
    scopes = ('\nAlso give a one-line scope for every domain you keep and every current domain, so that areas within '
              'domains can be assigned without ambiguity: "scopes": {"domain-name": "scope"}.' if kind == 'domain' else '')
    review = ask_json(SYSTEM, f'{head}\nCandidate new {dim} options:\n{cands}\n\nReview each candidate against the '
                      f'current options and the other candidates: keep, {into}. Keep a candidate only if it meets every '
                      f'rule. Prefer fewer, cleaner options; when two candidates overlap, keep the broader one.\n'
                      f'{RULES.format(dim=dim, what=what)}{scopes}\n\nReply with one ```json block: {{"decisions": '
                      f'[{{"name": "...", "decision": "keep|merge|format|drop", "into": "name or null", "reason": '
                      f'"short"}}], "kept": [{{"name": "...", "description": "one line"}}]' +
                      (', "scopes": {...}' if kind == 'domain' else '') + '}', f'top-{kind}-review')
    save(WORK / 'toplevel' / f'{kind}.json', {'current': [n for n, _, _ in cur], 'proposals': props, **review})
    print(f'{kind}: {len(props)} proposed, {len(review["kept"])} kept', flush=True)


def toplevel(a):
    pmap(toplevel_one, ['action', 'domain', 'class'], 3)


def domains_all():
    t = load(WORK / 'toplevel' / 'domain.json')
    names = t['current'] + [x['name'] for x in t['kept']]
    return names, t.get('scopes', {})


def classes_all():
    t = load(WORK / 'toplevel' / 'class.json')
    return t['current'], t['kept'], [d for d in t['decisions'] if d['decision'] == 'format']


def actions_all():
    t = load(WORK / 'toplevel' / 'action.json')
    return [(a, ACTIONS[a]) for a in t['current']] + [(x['name'], x['description']) for x in t['kept']]


# ---------------------------------------------------------------------------------------------------- sub-domains
SUB_RULES = """Rules:
1. Broad enough to host at least ~20 clearly different terminal tasks (different actions, materials and tools).
2. Specific enough to change the scenario: for a domain like transportation, "airline crew scheduling" or "freight rail operations", not "logistics" and not "fixing a delayed-shipment report".
3. Distinct: no near-synonyms or heavy overlaps.
4. Feasible: the work happens on a computer in the sandbox (no lab equipment, GPU or live internet); the data, systems and documents of the area can be simulated.
5. Cover the whole domain, including less common areas, not only the most familiar ones."""
SUB_FMT = ('Reply with one ```json block: a list of {"name": "short noun phrase, 2 to 6 words", "note": "the kinds of '
           'terminal work that happen there, at most 15 words"}.')


def subdomains_one(dom):
    out = WORK / 'subdomains' / f'{dom}.json'
    if out.exists():
        return
    names, scopes = domains_all()
    others = '\n'.join(f'- {d}: {scopes.get(d, "")}' for d in names)
    head = (f'Domain: {dom}: {scopes.get(dom, "")}\n\nAll domains (an area must belong to this one, not to another):\n'
            f'{others}\n')
    areas = ask_json(SYSTEM, f'{head}\nList the areas within {dom} in which realistic terminal tasks are set.\n'
                     f'{SUB_RULES}\nAim for 40 to 80 areas.\n\n{SUB_FMT}', f'sub-{dom}-p0')
    passes = [len(areas)]
    for p in (1, 2, 3):
        so_far = '\n'.join(f'- {x["name"]}' for x in areas)
        more = ask_json(SYSTEM, f'{head}\nAreas listed so far:\n{so_far}\n\nList further areas within {dom} that this '
                        f'list does not cover yet. Same rules; reply with [] if the domain is well covered.\n'
                        f'{SUB_RULES}\n\n{SUB_FMT}', f'sub-{dom}-p{p}')
        seen = {x['name'].lower() for x in areas}
        new = [x for x in more if x['name'].lower() not in seen]
        areas += new
        passes.append(len(new))
        if len(new) < max(5, 0.1 * len(areas)):
            break
    cands = '\n'.join(f'- {x["name"]}: {x.get("note", "")}' for x in areas)
    clean = ask_json(SYSTEM, f'{head}\nCandidate areas within {dom}:\n{cands}\n\nClean this list: merge near-duplicates '
                     f'(keep the broader phrasing), and drop areas that are too narrow to host ~20 clearly different '
                     f'tasks, infeasible in the sandbox, or that belong to another domain. Do not add new areas.\n'
                     f'{SUB_RULES}\n\nReply with one ```json block: {{"kept": [{{"name": "...", "note": "..."}}], '
                     f'"dropped": [{{"name": "...", "reason": "short; for a merge: merged into <name>"}}]}}',
                     f'sub-{dom}-clean')
    save(out, {'domain': dom, 'passes': passes, 'candidates': areas, **clean})
    print(f'sub {dom}: passes {passes}, {len(areas)} candidates, {len(clean["kept"])} kept', flush=True)


def subdomains(a):
    names, _ = domains_all()
    pmap(subdomains_one, names, a.workers)


# ---------------------------------------------------------------------------------------------------- formats
FMT_RULES = """Rules:
1. It names a kind of file, service or repository state, never a defect, trick or what is wrong with it (nothing like "corrupted", "encrypted", "truncated", "leaked secret" or "lost commit"): which problem a task involves is left to the task generator.
2. A setup script can produce it offline in the sandbox with common tools or libraries (apt, PyPI, npm), and the agent can work with it there.
3. Common in real work; not a synonym or minor variant of a current option.
4. It belongs to this class rather than to another one."""


def formats_one(cls):
    out = WORK / 'formats' / f'{cls}.json'
    if out.exists():
        return
    cur_cls, new_cls, moved = classes_all()
    newdef = {x['name']: x['description'] for x in new_cls}
    field, opts = SA.DETAILS.get(cls, ('formats', {}))
    merge = MERGE_DETAILS.get(cls, {})
    cur = [o for o in opts if o not in merge] + sorted({v for v in merge.values() if v and v not in opts})
    hint = [f'{d["name"]} ({d.get("reason", "")})' for d in moved if d.get('into') == cls]
    doms, _ = domains_all()
    words = FIELD_WORDS[field]
    head = (f'Asset class: {cls}: {CLASS_DEFS.get(cls) or newdef.get(cls)}.\nOther classes: '
            f'{", ".join(c for c in cur_cls + list(newdef) if c != cls)}.\n\nCurrent {words}: '
            f'{", ".join(cur) if cur else "none yet"}.\n' +
            (f'Suggested when reviewing new classes (consider them): {"; ".join(hint)}.\n' if hint else '') +
            f'Domains: {", ".join(doms)}.\n')
    fmt = (f'Reply with one ```json block: a list of {{"name": "kebab-case", "label": "display name", "domains": '
           f'["the domains above where it is common"] or ["any"]}}.')
    props = ask_json(SYSTEM, f'{head}\nPropose further {words} for this class: common kinds an agent meets in real '
                     f'terminal work.\n{FMT_RULES}\nUp to 20; fewer is fine.\n\n{fmt}', f'fmt-{cls}-p0')
    cands = '\n'.join(f'- {x["name"]} ({x.get("label", "")}; domains: {", ".join(x.get("domains", []))})' for x in props)
    clean = ask_json(SYSTEM, f'{head}\nCandidate new {words}:\n{cands}\n\nReview them: drop any that break a rule or '
                     f'duplicate a current option or another candidate, and fix domain tags that are wrong. Do not add '
                     f'new ones.\n{FMT_RULES}\n\nReply with one ```json block: {{"kept": [{{"name": "...", "label": '
                     f'"...", "domains": [...]}}], "dropped": [{{"name": "...", "reason": "short"}}]}}',
                     f'fmt-{cls}-clean')
    save(out, {'class': cls, 'field': field, 'current': cur, 'proposals': props, **clean})
    print(f'fmt {cls}: {len(props)} proposed, {len(clean["kept"])} kept', flush=True)


def formats(a):
    cur_cls, new_cls, _ = classes_all()
    todo = [c for c in cur_cls if c not in ('python-code', 'c-cpp-code', 'other-code', 'vm-image', 'upstream-project')]
    pmap(formats_one, todo + [x['name'] for x in new_cls], a.workers)


# ---------------------------------------------------------------------------------------------------- languages
def languages(a):
    out = WORK / 'languages.json'
    if out.exists():
        return
    tb = sorted({l for r in json.loads((HERE / 'asset_sampler' / 'asset_labels.json').read_text())
                 if r['set'] in ('TB2.1', 'TB3.0') for l in r.get('languages', [])} | set(SA.EXTRA_LANGS))
    tb = [SA.LANG_ALIASES.get(l, l) for l in tb if l not in ('Python', 'C', 'C++', 'CUDA')]
    langs = ask_json(SYSTEM, f'Asset class other-code covers source code in languages other than Python, C, C++ and '
                     f'CUDA (those have their own classes).\n\nList the programming languages an engineer may meet in '
                     f'terminal work: general-purpose, scripting, functional, JVM and .NET, query, hardware-description, '
                     f'proof assistants, scientific and legacy languages, and esoteric or educational ones that real '
                     f'people use. Programming, scripting, query and hardware-description languages only: no markup or '
                     f'data formats. These must appear: {", ".join(tb)}. Aim for 70 to 90.\n\nFor each, give its rough '
                     f'share of public code in percent (e.g. of GitHub activity; an order of magnitude is enough, use '
                     f'small numbers such as 0.001 for rare languages), and whether a working compiler or interpreter '
                     f'installs offline on Ubuntu 24.04 from apt, PyPI or npm (yes / partial / no).\n\nReply with one '
                     f'```json block: a list of {{"name": "...", "share_pct": number, "offline": "yes|partial|no"}}.',
                     'languages')
    save(out, {'required': tb, 'languages': langs})
    print(f'languages: {len(langs)}', flush=True)


# ---------------------------------------------------------------------------------------------------- compat
def compat_one(cls):
    out = WORK / 'compat' / f'{cls}.json'
    if out.exists():
        return
    cur_cls, new_cls, _ = classes_all()
    desc = CLASS_DEFS.get(cls) or {x['name']: x['description'] for x in new_cls}.get(cls)
    subject = ('Tasks with no materials: the agent starts from a stock environment and builds everything itself.'
               if cls == 'none' else f'Tasks whose main material is {cls} ({desc}).')
    acts = '\n'.join(f'- {n}: {d}' for n, d in actions_all())
    doms, scopes = domains_all()
    dl = '\n'.join(f'- {d}: {scopes.get(d, "")}' for d in doms)
    res = ask_json(SYSTEM, f'{subject}\n\nRate how naturally each action and each domain combines with this in a '
                   f'realistic terminal task: 2 = natural and common, 1 = plausible, 0 = forced or unnatural.\n\n'
                   f'Actions:\n{acts}\n\nDomains:\n{dl}\n\nReply with one ```json block: {{"actions": {{"name": 0|1|2, '
                   f'...}}, "domains": {{"name": 0|1|2, ...}}}} covering every action and domain above.',
                   f'compat-{cls}', effort='medium')
    save(out, res)
    print(f'compat {cls}: ok', flush=True)


def compat(a):
    cur_cls, new_cls, _ = classes_all()
    pmap(compat_one, ['none'] + cur_cls + [x['name'] for x in new_cls], a.workers)


# ---------------------------------------------------------------------------------------------------- pairs
def pairs(a):
    """How naturally each new class appears alongside every other class in one task environment (0/1/2): the
    companion lift for pairs the TB fit has no data on."""
    out = WORK / 'pairs.json'
    if out.exists():
        return
    cur_cls, new_cls, _ = classes_all()
    defs = {**CLASS_DEFS, **{x['name']: x['description'] for x in new_cls}}
    allc = '\n'.join(f'- {c}: {defs[c]}' for c in cur_cls + [x['name'] for x in new_cls])
    res = ask_json(SYSTEM, f'Asset classes:\n{allc}\n\nA task environment often holds several materials. For each of '
                   f'these classes: {", ".join(x["name"] for x in new_cls)}, rate how naturally each other class '
                   f'appears alongside it in the same realistic terminal task: 2 = natural and common, 1 = plausible, '
                   f'0 = forced or unnatural.\n\nReply with one ```json block: {{"<class>": {{"<other class>": 0|1|2, '
                   f'...}}, ...}} with one entry per class listed, covering every other class.', 'pairs', effort='medium')
    save(out, res)
    print(f'pairs: {len(res)} classes', flush=True)


# ---------------------------------------------------------------------------------------------------- coverage
def student1_sample(per_domain, seed=0):
    R = Path('/data/aai/tbench/tasksynth')
    by = collections.defaultdict(list)
    for c in ('r5', 's1', 's2'):
        for t, v in json.loads((R / 'campaigns' / c / 'manifest.json').read_text()).items():
            d = R / 'rounds' / v['round'] / 'tasks' / v['spec_id']
            ts = json.loads((d / 'tasksynth.json').read_text())
            instr = ' '.join((d / 'instruction.md').read_text().split())[:300]
            by[ts['spec']['domain']].append({'id': v['spec_id'], 'title': ts['meta'].get('title', ''), 'text': instr})
    rng = random.Random(seed)
    return {d: rng.sample(v, min(per_domain, len(v))) for d, v in sorted(by.items())}


def coverage_one(item):
    dom, tasks = item
    out = WORK / 'coverage' / f'{dom}.json'
    if out.exists():
        return
    sub = load(WORK / 'subdomains' / f'{dom}.json')['kept']
    areas = '\n'.join(f'{i}. {x["name"]}' for i, x in enumerate(sub, 1))
    tl = '\n'.join(f'[{t["id"]}] {t["title"]}: {t["text"]}' for t in tasks)
    lab = ask_json(SYSTEM, f'Areas within the domain {dom}:\n{areas}\n\nTasks (id, title, start of the instruction):\n{tl}'
                   f'\n\nFor each task, give the number of the area it belongs to, or 0 if no area fits it well.\n\n'
                   f'Reply with one ```json block: {{"task id": number, ...}} for every task.',
                   f'cov-{dom}', effort='medium')
    miss = [t for t in tasks if not lab.get(t['id'])]
    add = []
    if miss:
        ml = '\n'.join(f'- {t["title"]}: {t["text"][:200]}' for t in miss)
        add = ask_json(SYSTEM, f'Areas within the domain {dom}:\n{areas}\n\nThese tasks from {dom} fit none of the '
                       f'areas:\n{ml}\n\nPropose areas that would cover them, following the rules below. Skip tasks '
                       f'too specific to warrant an area of their own (they stay reachable without one), and never '
                       f'duplicate a listed area. Reply with [] if none.\n{SUB_RULES}\n\n{SUB_FMT}',
                       f'cov-{dom}-add')
    save(out, {'domain': dom, 'n': len(tasks), 'uncovered': [t['id'] for t in miss], 'labels': lab, 'added': add})
    print(f'cov {dom}: {len(miss)}/{len(tasks)} uncovered, {len(add)} areas added', flush=True)


def coverage(a):
    pmap(coverage_one, list(student1_sample(a.per_domain).items()), a.workers)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('stage', choices=['toplevel', 'subdomains', 'formats', 'languages', 'compat', 'pairs', 'coverage'])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--per-domain', type=int, default=100)
    a = ap.parse_args()
    globals()[a.stage](a)


if __name__ == '__main__':
    main()
