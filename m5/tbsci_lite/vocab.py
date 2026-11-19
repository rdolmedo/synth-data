"""TB-Sci-lite spec vocabulary (#353): fields fitted to the lite pool, breadth below them from Opus brainstorms.

Top level (fitted, never broadened; #25 lesson 1):
  domain x field   lite's retained 135 tasks (manifest `domain`/`field`), fields smoothed within their domain
  archetype        what the agent produces (fit, predict, classify, reconstruct, simulate, design, pipeline, infer,
                   prove), fitted from Opus labels of the lite tasks (class-level counts only), smoothed
  modality         the main data modality, fitted the same way
Below it (brainstormed with Opus 5.5 per field; no lite task is ever in a brainstorm prompt):
  subtopics        ~60-120 per field: first list, "what is still missing" passes, a cleaning call
  formats          concrete file formats per field, tagged with modalities
Then `singles`: each lite task is matched to its nearest subtopics (Opus, lite text in context: this is the
checker, not the generator); a subtopic whose only lite match is one task and which is specific to it is broadened
or merged (#19 lesson 3).

  python -m m5.tbsci_lite.vocab label | brainstorm | formats | singles | build
Outputs: WORK/<stage>/*.json, vocab.json in this directory (built), WORK = /data/aai/m5/tbsci_lite/vocab.
"""
import argparse
import collections
import json
import os
import re
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('VOCAB_WORK', '/data/aai/m5/tbsci_lite/vocab')
from task_synthesis.vocab.opus import WORK, ask_json  # noqa: E402

from .common import LITE_MANIFEST, LITE_TASKS  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = HERE / 'vocab.json'
VALIDATION = Path(__file__).resolve().parents[2] / 'aa_evals' / 'tbsci_lite' / 'v2_1_validation.json'

FIELD_MERGE = {'formal mathematics': 'formal-mathematics'}
FIELD_DESC = {
    'physics': 'physics (classical and quantum mechanics, optics, plasma, condensed matter, nuclear and particle, '
               'acoustics, fluids, statistical physics)',
    'chemistry': 'chemistry (physical, analytical, organic, inorganic, computational and quantum chemistry, '
                 'spectroscopy, kinetics, electrochemistry)',
    'materials-science': 'materials science (crystallography, microstructure, mechanical and electronic properties, '
                         'polymers, alloys, batteries and energy materials, characterization)',
    'astronomy': 'astronomy and astrophysics (photometry, spectroscopy, astrometry, time-domain, cosmology, '
                 'planetary science, instrumentation)',
    'biology': 'biology (genomics, transcriptomics, proteomics, structural biology, evolution, cell and molecular '
               'biology, microbiology, systems biology, bioinformatics)',
    'medicine': 'medicine and health sciences (clinical data, epidemiology, medical imaging, physiology signals, '
                'pharmacology, biostatistics of trials)',
    'neuroscience': 'neuroscience (electrophysiology, spike trains, neuroimaging, behaviour, neural models)',
    'ecology': 'ecology and environmental biology (populations, communities, biodiversity surveys, movement, '
               'species distributions)',
    'mechanical-engineering': 'mechanical and aerospace engineering (dynamics, control, vibration, structures, '
                              'thermal and fluid systems, robotics, manufacturing)',
    'electrical-engineering': 'electrical and electronic engineering (circuits, signal processing, communications, '
                              'power systems, control, embedded and RF)',
    'civil-engineering': 'civil and environmental engineering (structures, geotechnics, hydraulics and hydrology, '
                         'transportation, construction materials)',
    'statistics': 'statistics and data science (inference, regression, Bayesian methods, time series, survival, '
                  'experimental design, causal inference)',
    'operations-research': 'operations research (optimization, scheduling, routing, queueing, inventory, '
                           'stochastic programming, network flows)',
    'applied-mathematics': 'applied mathematics (numerical analysis, ODE/PDE, dynamical systems, inverse problems, '
                           'approximation, signal and harmonic analysis)',
    'formal-mathematics': 'formal and computational mathematics (machine-checked proofs, combinatorics, number '
                          'theory and algebra computations, verified algorithms)',
    'geosciences': 'geosciences (seismology, geophysics, geochemistry, hydrogeology, remote sensing of land, '
                   'atmosphere and climate, geodesy)',
    'ocean-sciences': 'ocean sciences (physical oceanography, ocean biogeochemistry, waves and tides, marine '
                      'acoustics, ocean observing systems)',
}

ARCHETYPES = {
    'fit-estimate': 'fit a model to data and estimate its parameters (with uncertainties where natural)',
    'predict': 'build a predictive model from training data and predict held-out targets',
    'classify-detect': 'classify, detect, identify or flag items, events, signals or anomalies in data',
    'reconstruct-invert': 'reconstruct a hidden quantity, state, signal or structure from indirect or degraded measurements',
    'simulate-compute': 'implement a numerical model, simulation or solver and compute requested quantities',
    'design-optimize': 'design, tune or control a system, or solve an optimization problem, to meet stated targets',
    'process-pipeline': 'build a processing or reduction pipeline that turns raw data into specified derived products',
    'infer-test': 'statistical or causal inference: test hypotheses, estimate effects, quantify evidence',
    'prove-derive': 'prove, formally verify or exactly derive a mathematical result',
}
MODALITIES = {
    'tabular': 'tables of measurements or records (CSV, Parquet, Excel)',
    'time-series': 'time series or signals (sensor traces, recordings, light curves, logs over time)',
    'image': 'images, image stacks or video (microscopy, astronomy, medical, remote sensing)',
    'sequence': 'biological or symbolic sequences (DNA, RNA, protein, reads, alignments)',
    'structure': 'molecular, crystal or 3-D structures and meshes (PDB, CIF, XYZ, STL)',
    'gridded-field': 'gridded or spatial fields (NetCDF, HDF5 arrays, rasters, spatial point data)',
    'graph': 'graphs and networks (adjacency lists, interaction networks, road or power grids)',
    'spectrum': 'spectra (optical, mass, NMR, X-ray, frequency-domain measurements)',
    'model-spec': 'a model, system or problem specification without a dataset (equations, netlists, parameter files, '
                  'formal statements, instance files)',
}
MINUTES = [(240, 0.20), (360, 0.30), (480, 0.30), (720, 0.20)]   # v2: 4-12 h (pilot spirit check: Opus put the
# pilot's 2-6 h targets at a median 6 expert hours vs lite's 20; v1 was 120/180/240/360 at 35/30/20/15%)
REAL_SHARE = 0.25                                               # 20-30% real public data (#353)
OPEN_SUBTOPIC = 0.03                                            # <= 5% open draws (#25 lesson 2)


def lite_tasks():
    """{task: (domain, field, instruction)} of the retained v2.1 tasks (read by the labeler and the singles check
    only; never by a brainstorm prompt or the generator)."""
    m = json.loads(LITE_MANIFEST.read_text())['tasks']
    kept = set(json.loads(VALIDATION.read_text())['retained_tasks'])
    out = {}
    with tarfile.open(LITE_TASKS) as tf:
        for member in tf:
            parts = member.name.split('/')
            if len(parts) == 3 and parts[2] == 'instruction.md' and parts[1] in kept:
                text = tf.extractfile(member).read().decode(errors='replace')
                f = m[parts[1]]['field']
                out[parts[1]] = (m[parts[1]]['domain'], FIELD_MERGE.get(f, f), text)
    assert len(out) == len(kept), (len(out), len(kept))
    return out


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


# ------------------------------------------------------------------------------------------------ label (lite)
LABEL_SYSTEM = """You label scientific computing tasks with two categories each. Answer with JSON only."""


def label_prompt(batch):
    arch = '\n'.join(f'- {k}: {v}' for k, v in ARCHETYPES.items())
    mod = '\n'.join(f'- {k}: {v}' for k, v in MODALITIES.items())
    tasks = '\n\n'.join(f'## {t}\n{re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()[:3000]}' for t, text in batch)
    return f"""For each task below, choose the ONE archetype that best describes what the agent must mainly produce, and the ONE modality of the main data it works on.

Archetypes:
{arch}

Modalities:
{mod}

Also give expert_hours: your estimate of how long a domain expert would need (a number).

Reply with a JSON object {{"<task name>": {{"archetype": "...", "modality": "...", "expert_hours": n}}, ...}} covering every task.

{tasks}"""


def label(a):
    tasks = lite_tasks()
    names = sorted(tasks)
    batches = [names[i:i + 12] for i in range(0, len(names), 12)]

    def one(i):
        b = [(t, tasks[t][2]) for t in batches[i]]
        return ask_json(LABEL_SYSTEM, label_prompt(b), f'label-{i:02d}')

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, range(len(batches))))
    labels = {}
    for r in res:
        labels.update(r)
    missing = [t for t in names if t not in labels]
    bad = [t for t in names if t in labels and (labels[t].get('archetype') not in ARCHETYPES
                                                or labels[t].get('modality') not in MODALITIES)]
    out = {t: {**labels[t], 'domain': tasks[t][0], 'field': tasks[t][1]} for t in names if t in labels}
    save(WORK / 'label' / 'lite_labels.json', out)
    print(json.dumps({'labelled': len(out), 'missing': missing, 'bad': bad,
                      'archetype': collections.Counter(x['archetype'] for x in out.values()).most_common(),
                      'modality': collections.Counter(x['modality'] for x in out.values()).most_common()}, indent=1))


# ------------------------------------------------------------------------------------------------ brainstorm
BS_SYSTEM = """You help design the topic vocabulary for a generator of synthetic training tasks for AI agents that do scientific computing in a Linux terminal.

Each task is a realistic, self-contained piece of computational science that a researcher or engineer would do in a shell: analyse a dataset, fit or build a model, run a simulation, reconstruct a signal, design to a specification, and write precisely specified result files. Every task runs in a sandbox: Ubuntu 24.04 with Python 3.12 and the scientific Python stack (numpy, scipy, pandas, scikit-learn, statsmodels, astropy, biopython, rdkit, ase, xarray, netCDF4, h5py, cvxpy, ortools, ...), R, ngspice, 2 CPUs, 8 GB RAM, no GPU and no internet. Data is either produced at build time by a simulator the task author writes (with realistic noise and artifacts) or is a small (<= 50 MB) real public dataset. A domain expert needs 2-6 hours; the computation itself runs in minutes.

The generator receives a spec: a field, a subtopic within it, the kind of work (fit a model, predict, classify or detect, reconstruct or invert, simulate, design or optimize, build a processing pipeline, statistical inference, proof or exact derivation), a data modality and a format. Subtopics are categories, never individual tasks: each must fit many different tasks."""

BS_FIRST = """List subtopics of {desc} for such tasks.

Rules:
1. A subtopic is a recognizable research or engineering area, method family or object of study (e.g. for a different field: "glacier mass balance from altimetry", "Kalman filtering for tracking"), specific enough that two tasks drawn from it share a scientific setting, broad enough for dozens of different tasks.
2. Computational and terminal-feasible: data analysis, modelling, simulation, inversion, design or verification that runs on 2 CPUs in minutes. No wet-lab work, no GPU training, no manual GUI tools.
3. Cover the whole field, including less glamorous everyday work, not only famous showcase problems. Avoid famous benchmark datasets and textbook toy problems.
4. No duplicates or near-duplicates.

Give {n} subtopics. Reply with a JSON list of objects {{"name": "short name (3-8 words)", "gist": "one sentence: the scientific setting and typical computation"}}."""

BS_MORE = """Here are the subtopics of {desc} collected so far:

{have}

List {n} MORE subtopics of this field that are clearly not covered by any of these, under the same rules (recognizable area or method family, terminal-feasible computation, broad enough for dozens of tasks, no near-duplicates). Look especially at parts of the field the list neglects. Reply with a JSON list of {{"name", "gist"}} objects only for the new ones; an empty list if the field is covered."""

BS_CLEAN = """Below is a brainstormed list of subtopics of {desc}, used to give each synthetic computational task a specific scientific setting.

{have}

Clean it:
- merge near-duplicates (keep the broader name);
- drop entries that are not terminal-feasible (wet lab, GPU training, GUI-only tools, needs internet or huge data) or are single famous problems rather than areas;
- broaden entries so narrow that fewer than ~20 distinct tasks could come from them;
- keep everything else unchanged.

Reply with the cleaned JSON list of {{"name", "gist"}} objects."""

FMT_PROMPT = """For tasks in {desc}, list the concrete data and file formats that the data or materials of such tasks typically come in, beyond generic CSV/JSON (include those too, once). For each, give the modality it belongs to, from: {mods}.

Reply with a JSON list of {{"format": "short name (e.g. 'FITS image', 'FASTA', 'NetCDF-4 grid', 'SPICE netlist', 'mzML')", "modality": "...", "share": "common|occasional"}}; 10-25 entries."""


def fmt_list(items):
    return '\n'.join(f'- {x["name"]}: {x["gist"]}' for x in items)


def brainstorm(a):
    def field(f):
        desc = FIELD_DESC[f]
        items = ask_json(BS_SYSTEM, BS_FIRST.format(desc=desc, n=60), f'bs-{f}-first')
        for k in range(a.passes):
            more = ask_json(BS_SYSTEM, BS_MORE.format(desc=desc, have=fmt_list(items), n=30), f'bs-{f}-more{k}')
            print(f'{f}: pass {k}: +{len(more)}', flush=True)
            if len(more) < 5:
                items += more
                break
            items += more
        cleaned = ask_json(BS_SYSTEM, BS_CLEAN.format(desc=desc, have=fmt_list(items)), f'bs-{f}-clean')
        save(WORK / 'subtopics' / f'{f}.json', {'raw': items, 'clean': cleaned})
        return f, len(items), len(cleaned)

    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(field, a.fields or sorted(FIELD_DESC)):
            print(r, flush=True)


def formats(a):
    mods = ', '.join(MODALITIES)

    def field(f):
        r = ask_json(BS_SYSTEM, FMT_PROMPT.format(desc=FIELD_DESC[f], mods=mods), f'fmt-{f}')
        r = [x for x in r if x.get('modality') in MODALITIES]
        save(WORK / 'formats' / f'{f}.json', r)
        return f, len(r)

    with ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(field, sorted(FIELD_DESC)):
            print(r, flush=True)


# ------------------------------------------------------------------------------------------------ singles (#19)
SINGLES_SYSTEM = """You check a topic vocabulary against a set of held-out evaluation tasks. Answer with JSON only."""

SINGLES_PROMPT = """Below is one evaluation task from the field of {desc}, followed by the field's numbered subtopic list.

Which subtopics does this task fall under? Give the numbers of every subtopic whose setting the task clearly belongs to (usually 1-3), and for each say whether the subtopic is SPECIFIC to this task: so narrow that it essentially describes this task's particular problem (same object of study AND same method), rather than a broad area the task merely belongs to.

Reply with {{"matches": [{{"id": n, "specific": true|false}}]}}.

# Evaluation task
{task}

# Subtopics
{subs}"""


def singles(a):
    tasks = lite_tasks()
    subs = {f: json.loads((WORK / 'subtopics' / f'{f}.json').read_text())['clean'] for f in FIELD_DESC}

    def one(t):
        dom, f, text = tasks[t]
        lst = '\n'.join(f'{i}. {x["name"]}: {x["gist"]}' for i, x in enumerate(subs[f]))
        task = re.sub(r'<!--.*?-->', '', text, flags=re.S).strip()[:5000]
        r = ask_json(SINGLES_SYSTEM, SINGLES_PROMPT.format(desc=FIELD_DESC[f], task=task, subs=lst), f'single-{t}')
        return t, f, r.get('matches') or []

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, sorted(tasks)))
    hits = collections.defaultdict(list)
    for t, f, ms in res:
        for m in ms:
            if isinstance(m.get('id'), int) and 0 <= m['id'] < len(subs[f]):
                hits[(f, m['id'])].append((t, bool(m.get('specific'))))
    flagged = []
    for (f, i), lst in sorted(hits.items()):
        if len(lst) == 1 or any(s for _, s in lst):
            flagged.append({'field': f, 'id': i, 'name': subs[f][i]['name'], 'gist': subs[f][i]['gist'],
                            'n_tasks': len(lst), 'specific': [t for t, s in lst if s], 'tasks': [t for t, _ in lst]})
    save(WORK / 'singles' / 'matches.json', {'matches': {t: ms for t, _, ms in res}, 'flagged': flagged})
    print(json.dumps({'tasks': len(res), 'subtopics_hit': len(hits), 'flagged': len(flagged),
                      'single_task': sum(1 for x in flagged if x['n_tasks'] == 1),
                      'specific': sum(1 for x in flagged if x['specific'])}, indent=1))


BROADEN_PROMPT = """This subtopic of {desc} is too narrow for a synthetic-task vocabulary (it pins down one particular problem):

"{name}: {gist}"

Its siblings in the vocabulary are:
{siblings}

Either merge it into the sibling that covers it best (answer {{"merge": <sibling number>}}), or, if none does, broaden it into an area that covers dozens of distinct tasks with different objects of study or methods (answer {{"broaden": {{"name": "...", "gist": "..."}}}}). The broadened entry must not mention the particular problem's object of study and method together."""


def modality_by_archetype(labels, beta):
    """P(modality | archetype) from lite's labels, smoothed toward the pooled modality mix (pseudo-count beta). In
    lite the main data follows the kind of work (design 83% and simulate 61% model-spec, fit/infer/predict 0%); v2
    drew it from the domain alone, and the pilot and v2a turned most model-spec draws into fit-to-data tasks."""
    names = list(MODALITIES)
    pooled = collections.Counter(x['modality'] for x in labels.values())
    pooled = {k: (pooled.get(k, 0) + 0.5) / (sum(pooled.values()) + 0.5 * len(names)) for k in names}
    out = {}
    for arch in ARCHETYPES:
        c = collections.Counter(x['modality'] for x in labels.values() if x['archetype'] == arch)
        tot = sum(c.values()) + beta
        out[arch] = {k: round((c.get(k, 0) + beta * pooled[k]) / tot, 4) for k in names}
    return out


def joint(a):
    """Add modality_by_archetype to vocab.json in place (vocab v3; the rest of v2 unchanged)."""
    labels = json.loads((WORK / 'label' / 'lite_labels.json').read_text())
    v = json.loads(VOCAB.read_text())
    v['modality_by_archetype'] = modality_by_archetype(labels, a.beta)
    v['version'] = 3
    v['smoothing']['joint_beta'] = a.beta
    VOCAB.write_text(json.dumps(v, indent=1) + '\n')
    print(json.dumps({k: max(x, key=x.get) + f' {x["model-spec"]:.2f} model-spec' for k, x in v['modality_by_archetype'].items()}, indent=1))


def build(a):
    """vocab.json: fitted top level + brainstormed subtopics (flagged ones broadened or merged) + formats."""
    labels = json.loads((WORK / 'label' / 'lite_labels.json').read_text())
    single = json.loads((WORK / 'singles' / 'matches.json').read_text())
    # fields: lite counts within their domain, smoothed (+alpha each); domain shares = lite's
    dom_n = collections.Counter(x['domain'] for x in labels.values())
    field_n = collections.Counter((x['domain'], x['field']) for x in labels.values())
    alpha = a.alpha
    domains = {}
    for d, n in dom_n.items():
        fs = {f: c for (dd, f), c in field_n.items() if dd == d}
        tot = sum(fs.values()) + alpha * len(fs)
        domains[d] = {'weight': n / sum(dom_n.values()), 'fields': {f: (c + alpha) / tot for f, c in sorted(fs.items())}}
    # archetype and modality: per domain, lite counts smoothed toward the pooled mix (pseudo-count beta)
    def fitted(key, names):
        pooled = collections.Counter(x[key] for x in labels.values())
        pooled = {k: (pooled.get(k, 0) + 0.5) / (sum(pooled.values()) + 0.5 * len(names)) for k in names}
        out = {}
        for d in dom_n:
            c = collections.Counter(x[key] for x in labels.values() if x['domain'] == d)
            tot = sum(c.values()) + a.beta
            out[d] = {k: round((c.get(k, 0) + a.beta * pooled[k]) / tot, 4) for k in names}
        return pooled, out
    arch_pooled, arch = fitted('archetype', list(ARCHETYPES))
    mod_pooled, mod = fitted('modality', list(MODALITIES))
    # subtopics: flagged ones merged or broadened (Opus, cached)
    subs, log = {}, []
    flagged = {(x['field'], x['id']): x for x in single['flagged']
               if x['specific'] or (x['n_tasks'] == 1 and a.strict_singles)}
    for f in FIELD_DESC:
        items = json.loads((WORK / 'subtopics' / f'{f}.json').read_text())['clean']
        keep = []
        for i, x in enumerate(items):
            if (f, i) not in flagged:
                keep.append(x)
                continue
            sib = '\n'.join(f'{j}. {y["name"]}' for j, y in enumerate(items) if j != i)
            r = ask_json(BS_SYSTEM, BROADEN_PROMPT.format(desc=FIELD_DESC[f], name=x['name'], gist=x['gist'],
                                                          siblings=sib), f'broaden-{f}-{i}')
            if isinstance(r.get('merge'), int) and 0 <= r['merge'] < len(items) and r['merge'] != i:
                log.append({'field': f, 'from': x['name'], 'merged_into': items[r['merge']]['name']})
            elif isinstance(r.get('broaden'), dict) and r['broaden'].get('name'):
                keep.append({**r['broaden'], 'broadened_from': x['name']})
                log.append({'field': f, 'from': x['name'], 'broadened_to': r['broaden']['name']})
            else:
                log.append({'field': f, 'from': x['name'], 'dropped': True})
        names = set()
        subs[f] = [x for x in keep if not (x['name'].lower() in names or names.add(x['name'].lower()))]
    fmts = {f: json.loads((WORK / 'formats' / f'{f}.json').read_text()) for f in FIELD_DESC}
    vocab = {'version': 3, 'domains': domains, 'archetypes': ARCHETYPES, 'modalities': MODALITIES,
             'archetype_by_domain': arch, 'modality_by_domain': mod,
             'modality_by_archetype': modality_by_archetype(labels, a.beta), 'archetype_pooled': arch_pooled,
             'modality_pooled': mod_pooled, 'field_desc': FIELD_DESC, 'subtopics': subs, 'formats': fmts,
             'minutes': MINUTES, 'real_share': REAL_SHARE, 'open_subtopic': OPEN_SUBTOPIC,
             'singles_log': log, 'smoothing': {'field_alpha': alpha, 'mix_beta': a.beta, 'joint_beta': a.beta}}
    VOCAB.write_text(json.dumps(vocab, indent=1))
    print(json.dumps({'fields': {f: len(v) for f, v in subs.items()}, 'total_subtopics': sum(map(len, subs.values())),
                      'singles_changes': len(log)}, indent=1))


def label_generated(a):
    """Spirit check: the same Opus labels (archetype, modality, expert hours) on accepted generated tasks, for the
    comparison with lite's (label)."""
    from .common import ROOT
    tasks = {}
    for r in a.rounds:
        d = ROOT / 'rounds' / r
        for p in sorted((d / 'work').glob('*/record.json')):
            rec = json.loads(p.read_text())
            if rec.get('status') == 'valid' and (rec.get('judge') or {}).get('pass'):
                ins = d / 'tasks' / rec['spec_id'] / 'instruction.md'
                if ins.exists():
                    tasks[rec['spec_id']] = ins.read_text()
    names = sorted(tasks)[:a.limit]
    batches = [names[i:i + 12] for i in range(0, len(names), 12)]

    def one(i):
        try:
            return ask_json(LABEL_SYSTEM, label_prompt([(t, tasks[t]) for t in batches[i]]), f'glabel-{a.rounds[0]}-{i:03d}')
        except RuntimeError as e:   # e.g. a safeguard refusal on one batch (prod3 batch 2): label the rest
            print(f'batch {i} skipped: {str(e)[:160]}', flush=True)
            return {}

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, range(len(batches))))
    out = {}
    for r in res:
        out.update(r)
    save(WORK / 'label' / f'generated_{"+".join(a.rounds)}.json', out)
    lite = json.loads((WORK / 'label' / 'lite_labels.json').read_text())

    def dist(d, k):
        c = collections.Counter(x.get(k) for x in d.values())
        n = sum(c.values()) or 1
        return {kk: round(v / n, 3) for kk, v in c.most_common()}
    import statistics
    hours = lambda d: statistics.median(float(x.get('expert_hours') or 0) for x in d.values())  # noqa: E731
    print(json.dumps({'n': len(out), 'archetype': {'generated': dist(out, 'archetype'), 'lite': dist(lite, 'archetype')},
                      'modality': {'generated': dist(out, 'modality'), 'lite': dist(lite, 'modality')},
                      'expert_hours_median': {'generated': hours(out), 'lite': hours(lite)}}, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    sp.add_parser('label')
    b = sp.add_parser('brainstorm')
    b.add_argument('--passes', type=int, default=2)
    b.add_argument('--workers', type=int, default=6)
    b.add_argument('--fields', nargs='*')
    f = sp.add_parser('formats')
    f.add_argument('--workers', type=int, default=6)
    s = sp.add_parser('singles')
    s.add_argument('--workers', type=int, default=8)
    g = sp.add_parser('label-generated')
    g.add_argument('--rounds', nargs='+', required=True)
    g.add_argument('--limit', type=int, default=240)
    bu = sp.add_parser('build')
    bu.add_argument('--alpha', type=float, default=1.0)
    bu.add_argument('--beta', type=float, default=6.0)
    bu.add_argument('--strict-singles', action='store_true',
                    help='also broaden subtopics matched by exactly one lite task even if not specific to it')
    jo = sp.add_parser('joint')
    jo.add_argument('--beta', type=float, default=3.0)
    a = ap.parse_args(argv)
    {'label': label, 'label-generated': label_generated, 'brainstorm': brainstorm, 'formats': formats, 'singles': singles, 'build': build, 'joint': joint}[a.op](a)


if __name__ == '__main__':
    main()
