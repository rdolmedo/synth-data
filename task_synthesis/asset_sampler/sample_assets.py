#!/usr/bin/env python3
"""Sample anchor-first asset specs for Terminal-Bench-style environment synthesis.

Fits a small generative model to hand-checked labels of two releases (default TB 2.1 -> TB 3.0) and samples
specs that sit between them. Every spec with assets is built around one anchor: the artifact the rest of
the environment is built from or around.

   1. anchor bucket      none / best-generated / best-found-online / must-source   fitted mix, or --bucket-mix
   2. anchor class       given the bucket                                           tau_class
   3. companion classes  bundle size given the anchor; base weight x pairwise lift  tau_class (base only)
   4. asset roles        derived / authored / sourced / stock (+ derived_from)      untempered
   5. agent works on     the anchor or one of its companions                        untempered
   6. agent action       given what the agent works on                              tau_class
   7. domain             given the anchor class                                     tau_detail
   8. code scale         snippet / mini-package / real-project                      untempered
   9. per-class details  formats, languages, services, git state, ...               tau_detail
  10. origin             shipped / generated / downloaded, given the bucket         tau_class
  11. shipped size       file count and size bucket, borrowed from a similar task   untempered

Tasks of release A weigh (1 - lam) and tasks of release B weigh lam, so lam=0 reproduces A and lam=1
reproduces B. Temperatures flatten *which* anchors, classes, actions, domains and formats appear; pair
structure, bundle size and roles keep their fitted shape. Only class-level statistics leave this script:
no task names or descriptions are written to the specs.

Inputs (next to this file): asset_labels.json (asset classes, formats, sizes), anchor_buckets.json,
anchor_labels.json (anchor, roles, derivation edges, code scale, action, domain).

  python3 sample_assets.py --n 1000 --out specs.jsonl
  python3 sample_assets.py --bucket-mix none=0.1,best-generated=0.4,best-found-online=0.2,must-source=0.3
  python3 sample_assets.py --check
  python3 sample_assets.py --vocab vocab_v2.json --n 1000     # broadened vocabulary (#25), see extend()
"""
import argparse, collections, itertools, json, math, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['python-code', 'c-cpp-code', 'other-code', 'git-repo', 'upstream-project', 'tabular-data',
           'structured-data', 'text-data', 'database', 'binary', 'media', 'document', 'ml-model', 'archive',
           'config', 'running-service', 'scientific-data', 'vm-image']
BUCKETS = ['none', 'best-generated', 'best-found-online', 'must-source']
ROLES = ['derived', 'authored', 'sourced', 'stock']
SCALES = ['snippet', 'mini-package', 'real-project']
CODE_CLASSES = {'python-code', 'c-cpp-code', 'other-code', 'upstream-project'}
SIZE_BUCKETS = [(1e4, '<10KB'), (1e5, '10-100KB'), (1e6, '100KB-1MB'), (1e7, '1-10MB'), (1e8, '10-100MB'),
                (math.inf, '>100MB')]
CODE_LANGS = {'python-code': ['Python'], 'c-cpp-code': ['C', 'C++', 'CUDA'], 'other-code': None}
# Languages never seen in the two releases, so other-code can still reach them under temperature.
EXTRA_LANGS = ['Go', 'Ruby', 'Haskell', 'Julia', 'Kotlin', 'PHP', 'Lua', 'Zig', 'Elixir', 'Verilog', 'Fortran']
LANG_ALIASES = {'Bash': 'Shell'}
# class -> (spec field, {option: tokens}); a token starting with '.' matches a file
# extension, anything else a whole word in the task's asset description.
DETAILS = {
    'tabular-data': ('formats', {
        'csv': ['.csv', 'csv'], 'tsv': ['.tsv', 'tsv'], 'parquet': ['.parquet', 'parquet'],
        'excel': ['.xlsx', '.xls', 'excel', 'xlsx', 'xls'], 'feather': ['.feather', 'feather']}),
    'structured-data': ('formats', {
        'json': ['.json', 'json'], 'jsonl': ['.jsonl', 'jsonl'], 'yaml': ['.yaml', '.yml', 'yaml'],
        'xml': ['.xml', 'xml'], 'rdf': ['.ttl', '.owl', 'rdf', 'turtle', 'owl'], 'toml': ['.toml', 'toml']}),
    'text-data': ('formats', {
        'plain-text': ['.txt', 'text'], 'log': ['.log', 'log', 'logs'], 'calendar-ics': ['.ics', 'ics'],
        'hex-dump': ['.hex', 'hex', 'hexdump'], 'gcode': ['.gcode', 'gcode'], 'word-list': ['word list', 'dictionary']}),
    'document': ('formats', {
        'pdf': ['.pdf', 'pdf', 'pdfs'], 'markdown': ['.md', 'md', 'readme', 'markdown'], 'latex': ['.tex', 'latex'],
        'html': ['.html', 'html'], 'office': ['.docx', '.xlsm', 'docx', 'xlsm', 'workbook']}),
    'media': ('formats', {
        'png': ['.png', 'png', 'pngs'], 'jpeg': ['.jpg', '.jpeg', 'jpg', 'jpeg'], 'video': ['.mp4', 'mp4', 'video'],
        'audio': ['.mp3', '.wav', 'mp3', 'wav', 'audio'], 'ppm': ['.ppm', 'ppm'], 'svg': ['.svg', 'svg'],
        'gif': ['.gif', 'gif']}),
    'database': ('formats', {
        'sqlite': ['.db', '.sqlite', 'sqlite'], 'sql-dump': ['.sql', 'snapshot', 'dump'],
        'wal-journal': ['wal', 'journal'], 'server-db': ['postgres', 'postgresql', 'mysql', 'datadir']}),
    'archive': ('formats', {
        'zip': ['.zip', 'zip', 'zipped'], '7z': ['.7z', '7z'], 'tar': ['.tar', '.tar.gz', '.tgz', 'tar', 'tarball'],
        'encrypted': ['.enc', '.encrypted', 'encrypted'], 'zstd': ['.zst', 'zst'], 'polyglot': ['polyglot']}),
    'scientific-data': ('formats', {
        'fasta': ['.fasta', '.fa', 'fasta'], 'genbank': ['.gb', '.gbk', 'genbank'], 'pdb': ['.pdb', 'pdb'],
        'cif': ['.cif', 'cif'], 'molecule': ['.mol', '.sdf', 'mol', 'smiles'], 'spectrum': ['.dat', 'spectrum', 'spectra'],
        'numpy': ['.npy', '.npz', 'npy'], 'robot-model': ['.urdf', 'urdf', 'mjcf'], 'ontology': ['.ttl', '.owl', 'ontologies']}),
    'ml-model': ('formats', {
        'pytorch': ['.pt', '.pth', 'pytorch', 'state dict'], 'hf-download': ['hf', 'hugging', 'qwen', 'stablelm'],
        'tf-checkpoint': ['checkpoint', 'ckpt'], 'json-weights': ['json weights', 'json-weights'],
        'tokenizer': ['.model', 'tokenizer', 'vocab'], 'safetensors': ['.safetensors', 'safetensors']}),
    'binary': ('formats', {
        'elf-executable': ['elf', 'binary', 'binaries', 'stripped'], 'bytecode': ['.pyc', '.jar', 'pyc', 'jar'],
        'firmware-rom': ['.fd', '.rom', 'firmware', 'rom', 'ovmf'], 'object-file': ['.o', '.so']}),
    'vm-image': ('formats', {
        'qcow2': ['.qcow2', 'qcow2'], 'iso': ['.iso', 'iso'], 'raw-disk': ['.img', 'disk image', 'filesystem image'],
        'uefi-firmware': ['.fd', 'ovmf', 'uefi']}),
    'config': ('formats', {
        'nginx': ['nginx'], 'systemd': ['.service', 'systemd'], 'cron': ['cron'], 'supervisor': ['supervisor'],
        'ssh': ['ssh'], 'yaml': ['.yaml', '.yml', 'yaml'], 'ini-conf': ['.ini', '.conf', '.cfg', 'ini', 'conf'],
        'build-config': ['makefile', 'pom.xml', 'environment.yml', 'requirements', 'conda', 'sbt']}),
    'running-service': ('services', {
        'web-api': ['fastapi', 'flask', 'api', 'microservices', 'web app', 'portal', 'odoo'],
        'web-frontend': ['react', 'next.js', 'frontend', 'site'], 'sql-db': ['postgres', 'mysql', 'sqlite'],
        'cache-kv': ['redis', 'kv'], 'message-queue': ['kafka', 'queue'],
        'browser': ['vnc', 'playwright', 'chromium', 'browser'], 'mail': ['postfix', 'mailman'],
        'load-generator': ['loadgen', 'load-generating'], 'object-store': ['s3', 'minio', 'localstack'],
        'event-feed': ['event-feed', 'sidecar', 'sidecars']}),
    'git-repo': ('state', {
        'lost-commit': ['orphaned', 'detached', 'lost commit', 'dangling'],
        'secret-in-history': ['secret', 'api keys', 'leaked'], 'branches-to-merge': ['bundles', 'branch', 'merge'],
        'pinned-clone': ['pinned commit', 'cloned'], 'fresh-baseline': ['git init', 'baseline', 'git-initialized']}),
    'upstream-project': ('acquisition', {
        'git-clone': ['clone', 'cloned', 'repo'], 'release-tarball': ['tarball', 'release', '.tar.gz', 'downloaded'],
        'package': ['pip', 'package', 'vendored', 'pip-installed']}),
}
# Pairs the sandbox cannot host together, e.g. frozenset({'vm-image', 'running-service'}).
EXCLUDE_PAIRS = set()


def norm(counts):
    s = sum(counts.values())
    return {k: v / s for k, v in counts.items() if v > 0}


def temper(dist, tau):
    return norm({k: v ** (1 / tau) for k, v in dist.items()})


def draw(dist, rng):
    r = rng.random() * sum(dist.values())
    for k, v in dist.items():
        r -= v
        if r <= 0:
            return k
    return k


def draw_k(dist, k, rng):
    dist, out = dict(dist), []
    for _ in range(min(k, len(dist))):
        out.append(draw(dist, rng))
        del dist[out[-1]]
    return out


def size_bucket(x):
    return next(name for hi, name in SIZE_BUCKETS if x <= hi)


def matches(tokens, ext, desc):
    return any(t in ext if t.startswith('.') else re.search(r'(?<![a-z0-9])' + re.escape(t) + r'(?![a-z0-9])', desc)
               for t in tokens)


def tally(W, key):
    c = collections.Counter()
    for w, r in W:
        k = key(r)
        if k is not None:
            c[k] += w
    return c


def shrink(counts, prior, k):
    """Weighted counts pulled toward a prior distribution with k pseudo-tasks."""
    return norm({x: counts.get(x, 0) + k * p for x, p in prior.items()})


def conditional(W, cond, key, prior, k):
    by = collections.defaultdict(collections.Counter)
    for w, r in W:
        if key(r) is not None:
            by[cond(r)][key(r)] += w
    return {c: shrink(cnt, prior, k) for c, cnt in by.items()}


def load(labels, buckets, anchors):
    """One row per task: asset classes and roles from the anchor labels, formats and sizes from the asset labels."""
    base = {(r['set'], r['task']): r for r in json.load(open(labels))}
    bk = json.load(open(buckets))
    rows = []
    for a in json.load(open(anchors)):
        r = base[(a['set'], a['task'])]
        row = {'set': a['set'], 'bucket': bk[a['set']][a['task']], 'action': a['agent_action'],
               'domain': a['domain'], 'ext': r['ext'], 'description': r['description'].lower(),
               'languages': r['languages'], 'origin': '+'.join(sorted(r['origin'])),
               'n_files': r['n_files'], 'bytes': r['total_bytes']}
        if row['bucket'] != 'none':
            anchor = a['anchor_class']
            rank = {'anchor': 0, 'sourced': 1, 'derived': 2, 'authored': 3, 'stock': 4}
            roles, edges = {}, []
            for x in a['assets']:  # one role per class; the strongest wins when a class has several assets
                if x['class'] in CLASSES and rank[x['role']] < rank.get(roles.get(x['class']), 9):
                    roles[x['class']] = x['role']
            for x in a['assets']:
                if x['role'] == 'derived' and x['derived_from'] is not None:
                    edges.append((a['assets'][x['derived_from']]['class'], x['class']))
            extra = [c for c in CLASSES if c != anchor and (c in roles or c in r['classes'] or c == a['agent_works_on'])]
            row.update(anchor=anchor, classes=[anchor] + extra, works_on=a['agent_works_on'] or anchor,
                       roles={c: roles.get(c) for c in extra}, edges=edges, scale=a['code_scale'])
        rows.append(row)
    return rows


def weighted(rows, a, b, lam):
    na, nb = (sum(r['set'] == s for r in rows) for s in (a, b))
    nbar = (1 - lam) * na + lam * nb  # weights sum to the size of a release, so pseudo-counts stay in task units
    w = {a: (1 - lam) * nbar / na, b: lam * nbar / nb}
    return [(w[r['set']], r) for r in rows if r['set'] in w and w[r['set']] > 0]


def fit(rows, a, b, lam, rng, k_pair=10.0, k_back=3.0, a_class=0.2, a_detail=0.3):
    W = weighted(rows, a, b, lam)
    A = [(w, r) for w, r in W if r['bucket'] != 'none']
    N = [(w, r) for w, r in W if r['bucket'] == 'none']
    n_a = sum(w for w, _ in A)
    m = {'bucket': norm(tally(W, lambda r: r['bucket']))}

    # Anchor class given the bucket, backed off to the pooled anchor distribution.
    pooled = norm({c: tally(A, lambda r: r['anchor'])[c] + a_class for c in CLASSES})
    m['anchor'] = {bk: shrink(tally([(w, r) for w, r in A if r['bucket'] == bk], lambda r: r['anchor']), pooled, k_back)
                   for bk in BUCKETS[1:]}
    m['anchor_pooled'] = pooled

    # Companions: bundle size given the anchor, then a pairwise model moment-matched by simulation.
    size = norm(tally(A, lambda r: len(r['classes']) - 1))
    m['n_extra'] = conditional(A, lambda r: r['anchor'], lambda r: len(r['classes']) - 1, size, 2)
    m['n_extra_prior'] = size
    inc, sec, pair = collections.Counter(), collections.Counter(), collections.Counter()
    for w, r in A:
        for c in r['classes']:
            inc[c] += w
            sec[c] += w * (c != r['anchor'])
        for p in itertools.combinations(sorted(r['classes']), 2):
            pair[p] += w
    m['sec_target'] = {c: sec[c] / n_a for c in CLASSES}
    m['base'] = {c: (sec[c] + a_class) / n_a for c in CLASSES}
    m['lift'], m['pair_target'] = {}, {}
    for x, y in itertools.combinations(sorted(CLASSES), 2):
        q = inc[x] * inc[y] / n_a ** 2  # co-occurrence rate if the two were independent
        if q:
            # Observed rate plus k_pair pseudo-tasks at independence: strong pairs survive, and a pair
            # unseen in ~70 tasks keeps ~k_pair/(n+k_pair) of its independent rate instead of zero.
            m['pair_target'][(x, y)] = (pair[(x, y)] + k_pair * q) / (n_a + k_pair)
        m['lift'][(x, y)] = m['lift'][(y, x)] = m['pair_target'][(x, y)] / q if q else 1.0

    # Derivation edges child -> parent, then companion roles given (bucket, class, whether a class it is
    # derived from in some real task is in the bundle), backed off to (class, parent present), then pooled.
    m['edges'] = collections.defaultdict(collections.Counter)
    for w, r in A:
        for parent, child in r['edges']:
            m['edges'][child][parent] += w
    pp = lambda r, c: any(m['edges'][c][p] > 0 for p in r['classes'] if p != c)
    role_rows = [(w, (r['bucket'], c, pp(r, c), role)) for w, r in A for c, role in r['roles'].items() if role]
    role_counts = tally(role_rows, lambda t: t[3])
    role_prior = norm({ro: role_counts[ro] + a_class for ro in ROLES})
    m['role_prior'] = role_prior
    by_class = conditional(role_rows, lambda t: t[1:3], lambda t: t[3], role_prior, 1)
    m['role'] = {(bk, c, has): shrink(tally([(w, t) for w, t in role_rows if t[:3] == (bk, c, has)], lambda t: t[3]),
                                      by_class.get((c, has), role_prior), 1)
                 for bk in BUCKETS[1:] for c in CLASSES for has in (False, True)}

    # What the agent works on, given the anchor; when it is not the anchor it joins the bundle as a companion.
    same = sum(w for w, r in A if r['works_on'] == r['anchor']) / n_a
    other = norm(tally(A, lambda r: r['works_on'] if r['works_on'] != r['anchor'] else None))
    by_anchor = collections.defaultdict(collections.Counter)
    for w, r in A:
        by_anchor[r['anchor']][r['works_on']] += w
    m['works_on'] = {c: shrink(by_anchor[c], norm({x: (same if x == c else (1 - same) * other.get(x, 0)) for x in CLASSES}), 2)
                     for c in CLASSES}
    m['works_same'] = same

    # Code scale given (bucket, upstream project in bundle), backed off to the bucket, then pooled.
    code = [(w, r) for w, r in A if r['scale'] != 'none']
    sc_prior = norm({s: tally(code, lambda r: r['scale'])[s] + a_class for s in SCALES})
    sc_bucket = conditional(code, lambda r: r['bucket'], lambda r: r['scale'], sc_prior, 2)
    m['scale'] = {(bk, up): shrink(tally([(w, r) for w, r in code if r['bucket'] == bk and
                                          ('upstream-project' in r['classes']) == up], lambda r: r['scale']),
                                   sc_bucket.get(bk, sc_prior), 2)
                  for bk in BUCKETS[1:] for up in (False, True)}

    # Agent action given what it works on; domain given the anchor class; both backed off to pooled.
    act_prior = norm(tally(W, lambda r: r['action']))
    m['action'] = conditional(A, lambda r: r['works_on'], lambda r: r['action'], act_prior, 2)
    m['action_none'] = shrink(tally(N, lambda r: r['action']), act_prior, 2)
    m['action_prior'] = act_prior
    dom_prior = norm(tally(W, lambda r: r['domain']))
    m['domain'] = conditional(A, lambda r: r['anchor'], lambda r: r['domain'], dom_prior, 2)
    m['domain_none'] = shrink(tally(N, lambda r: r['domain']), dom_prior, 2)
    m['domain_prior'] = dom_prior

    # Per-class details, matched from extensions and asset descriptions.
    m['details'], m['details_by_domain'] = {}, {}
    for c, (field, opts) in DETAILS.items():
        counts, nopt = collections.Counter({o: a_detail for o in opts}), collections.Counter()
        for w, r in A:
            if c in r['classes']:
                hit = [o for o, toks in opts.items() if matches(toks, r['ext'], r['description'])]
                for o in hit:
                    counts[o] += w / len(hit)
                if hit:
                    nopt[min(len(hit), 4)] += w
        m['details'][c] = {'field': field, 'dist': norm(counts), 'n': norm(nopt) or {1: 1.0}}
        for dom in {r['domain'] for _, r in A}:  # formats follow the domain (FASTA for biology, CIF for materials)
            dc = collections.Counter()
            for w, r in A:
                if c in r['classes'] and r['domain'] == dom:
                    hit = [o for o, toks in opts.items() if matches(toks, r['ext'], r['description'])]
                    for o in hit:
                        dc[o] += w / len(hit)
            if dc:
                m['details_by_domain'][(c, dom)] = shrink(dc, m['details'][c]['dist'], 1)
    m['languages'] = {}
    for c, fixed in CODE_LANGS.items():
        langs = collections.Counter({l: a_detail for l in (fixed or EXTRA_LANGS)})
        keep = (lambda l: l in fixed) if fixed else (lambda l: l not in ('Python', 'C', 'C++', 'CUDA'))
        for w, r in A:
            seen = [l for l in (LANG_ALIASES.get(l, l) for l in r['languages']) if keep(l)] if c in r['classes'] else []
            for l in seen:
                langs[l] += w / len(seen)
        m['languages'][c] = norm(langs)

    org_prior = norm(tally(A, lambda r: r['origin'] or None))
    m['origin'] = {bk: shrink(tally([(w, r) for w, r in A if r['bucket'] == bk], lambda r: r['origin'] or None),
                              org_prior, k_back) for bk in BUCKETS[1:]}
    m['donors'] = [(w, r['bucket'], r['anchor'], r['scale'], len(r['classes']), r['n_files'], r['bytes']) for w, r in A]
    calibrate(m, rng)
    return m


def companions(m, anchor, base, rng, forced=()):
    chosen = [anchor, *forced]
    for _ in range(draw(m['n_extra'].get(anchor, m['n_extra_prior']), rng) - len(forced)):
        w = {}
        for c in m.get('classes', CLASSES):
            if c in chosen or any(frozenset({c, s}) in EXCLUDE_PAIRS for s in chosen):
                continue
            w[c] = base[c] * math.prod(m['lift'].get((c, s), 1.0) for s in chosen)  # v2 classes: independence
        if not w:
            break
        chosen.append(draw(w, rng))
    return chosen[1:]


def calibrate(m, rng, iters=20, n=8000, step=0.6):
    """Moment-match at tau=1: rescale base weights and pair lifts until each class appears as a companion,
    and each pair co-occurs, as often as fitted (a pairwise max-entropy fit, done by simulation)."""
    for _ in range(iters):
        got, got_pair = collections.Counter(), collections.Counter()
        for _ in range(n):
            a = draw(m['anchor_pooled'], rng)
            on = draw(m['works_on'][a], rng)
            sec = companions(m, a, m['base'], rng, () if on == a else (on,))
            got.update(sec)
            got_pair.update(itertools.combinations(sorted([a] + sec), 2))
        for c in CLASSES:
            if m['sec_target'][c] > 0:
                m['base'][c] *= (m['sec_target'][c] / max(got[c] / n, 1 / n)) ** step
        for (x, y), t in m['pair_target'].items():
            m['lift'][(x, y)] *= (t / max(got_pair[(x, y)] / n, 1 / n)) ** step
            m['lift'][(y, x)] = m['lift'][(x, y)]


W_COMPAT = {0: 0.0, 1: 1.0, 2: 3.0}  # Opus compatibility label -> weight (forced / plausible / natural)


def blend(tb, new, share):
    """TB-fitted options keep their shape and 1 - share of the mass; new options get share (nothing if none apply)."""
    if not new:
        return tb
    return {**{k: (1 - share) * p for k, p in tb.items()}, **{k: share * p for k, p in new.items()}}


def compat_dist(v, cls, kind, names):
    """Options `names` weighted by their compatibility with class `cls` ('none': no materials); {} if all are 0."""
    lab = v['compat'].get(cls, {}).get(kind, {})
    w = {n: W_COMPAT[int(lab.get(n, 1))] for n in names}
    return norm(w) if sum(w.values()) > 0 else {}


def equalize(new, pc, iters=60, clamp=5.0):
    """Compatibility decides where a new option appears, not how often overall: rescale the options of the new parts
    {condition: {option: p}} (iterative proportional fitting, multipliers clamped to [1/clamp, clamp]) so that every
    new option gets the same marginal under the condition distribution pc."""
    opts = sorted({o for d in new.values() for o in d})
    mult, adj = {o: 1.0 for o in opts}, new
    for _ in range(iters):
        adj, cur = {}, {o: 0.0 for o in opts}
        for c, d in new.items():
            if d:
                z = sum(mult[o] * p for o, p in d.items())
                adj[c] = {o: mult[o] * p / z for o, p in d.items()}
                for o, p in adj[c].items():
                    cur[o] += pc.get(c, 0) * p
            else:
                adj[c] = {}
        tot = sum(cur.values())
        for o in opts:
            if cur[o] > 0:
                mult[o] = min(clamp, max(1 / clamp, mult[o] * (tot / len(opts) / cur[o]) ** 0.5))
    return adj


def merge_dist(dist, mapping):
    """Fold options into others (a None target drops the option) and renormalize."""
    out = collections.Counter()
    for k, p in dist.items():
        t = mapping.get(k, k)
        if t is not None:
            out[t] += p
    return norm(out)


def extend(m, v):
    """Vocabulary v2 (#25; vocab_v2.json, built by task_synthesis.vocab.build from an Opus 5.5 brainstorm).

    - Details that name a task's defect or trick are merged into generic ones (v['merge_details']).
    - New asset classes: TB-fitted parts they lack are TB priors reweighted by Opus compatibility labels (domain
      given anchor, action given what the agent works on); companion pairs use Opus pairing labels as lifts; roles
      use the pooled prior.
    - sampler() then blends every dimension after tempering: TB options keep their fitted shape and 1 - cap of the
      mass, new options share cap (anchor and companion classes uniformly, actions and domains by compatibility).
      Formats blend at new_format_share, restricted to the formats tagged as common in the drawn domain.
      Languages of other-code are replaced by tempered real-world shares. A sub-domain is drawn uniformly within the
      domain except for an open_subdomain share of specs."""
    m['v'] = v
    new_cls = [c['name'] for c in v['classes']]
    m['classes'] = CLASSES + new_cls
    for c, mp in v['merge_details'].items():
        m['details'][c]['dist'] = merge_dist(m['details'][c]['dist'], mp)
        for key in [k for k in m['details_by_domain'] if k[0] == c]:
            m['details_by_domain'][key] = merge_dist(m['details_by_domain'][key], mp)
    # Companion lift of a pair with a new class: its Opus pairing label (forced pairs become rare, not impossible).
    # Normalized per class to a base-weighted mean of 1, so labels move a new class between partners without raising
    # how often it joins a bundle (its share stays at the cap).
    lift = {0: 0.05, 1: 1.0, 2: 3.0}
    for c, row in v.get('pairs', {}).items():
        if c not in new_cls:
            continue
        raw = {x: lift[int(lab)] for x, lab in row.items() if x in CLASSES + new_cls and x != c}
        bw = {x: m['base'].get(x, sum(m['base'].values()) / len(CLASSES)) for x in raw}
        mean = sum(bw[x] * raw[x] for x in raw) / sum(bw.values())
        for x, l in raw.items():
            m['lift'][(c, x)] = m['lift'][(x, c)] = l / mean
    other = norm(m['sec_target'])
    for c in new_cls:
        m['works_on'][c] = norm({c: m['works_same'], **{x: (1 - m['works_same']) * p for x, p in other.items()}})
        aw = compat_dist(v, c, 'actions', list(m['action_prior']))
        m['action'][c] = norm({a: m['action_prior'][a] * w for a, w in aw.items()}) if aw else m['action_prior']
        dw = compat_dist(v, c, 'domains', list(m['domain_prior']))
        m['domain'][c] = norm({d: m['domain_prior'][d] * dw[d] for d in dw}) if dw else m['domain_prior']
    n = collections.Counter()
    for d in m['details'].values():
        for k, p in d['n'].items():
            n[k] += p
    v['n_formats'] = norm(n)
    v['field'] = {c['name']: c.get('field', 'formats') for c in v['classes']}
    return m


def sampler(m, tau_class, tau_detail, bucket_mix=None):
    T = lambda d, t: {k: temper(v, t) for k, v in d.items()}
    mix = bucket_mix or m['bucket']
    anchor = T(m['anchor'], tau_class)
    base = temper(m['base'], tau_class)
    action, action_prior, action_none = T(m['action'], tau_class), temper(m['action_prior'], tau_class), \
        temper(m['action_none'], tau_class)
    domain, domain_prior, domain_none = T(m['domain'], tau_detail), temper(m['domain_prior'], tau_detail), \
        temper(m['domain_none'], tau_detail)
    details = {c: dict(d, dist=temper(d['dist'], tau_detail)) for c, d in m['details'].items()}
    details_dom = T(m['details_by_domain'], tau_detail)
    langs = T(m['languages'], tau_detail)
    v = m.get('v')
    origin = T(m['origin'], (v or {}).get('tau_origin', tau_class))  # build mechanics, not formatting
    if v:  # vocabulary v2: mix the tempered TB parts with the new options (see extend())
        new_cls = [c['name'] for c in v['classes']]
        # cap_new 'flat' (designs D2/D3): new options weigh like any other option, n_new / (n_tb + n_new) per dimension
        capof = lambda n_tb, n_new: (n_new / (n_tb + n_new) if v['cap_new'] == 'flat' else v['cap_new'])
        cap_cls = capof(len(CLASSES), len(new_cls))
        cap_act = capof(len(m['action_prior']), len(v['actions']))
        cap_dom = capof(len(m['domain_prior']), len(v['domains']))
        uniform = lambda xs: {x: 1 / len(xs) for x in xs} if xs else {}
        anchor = {bk: blend(d, uniform(new_cls), cap_cls) for bk, d in anchor.items()}
        base = {**base, **{c: cap_cls / (1 - cap_cls) / len(new_cls) for c in new_cls}}
        new_act = {c: compat_dist(v, c, 'actions', [x['name'] for x in v['actions']]) for c in m['classes'] + ['none']}
        new_dom = {c: compat_dist(v, c, 'domains', [x['name'] for x in v['domains']]) for c in m['classes'] + ['none']}
        # How often each class is the anchor / what the agent works on (and how often there are no materials).
        p_none = mix.get('none', 0)
        p_anchor, p_works = collections.Counter({'none': p_none}), collections.Counter({'none': p_none})
        for bk, pb in mix.items():
            for a_, pa in (anchor.get(bk) or {}).items():
                p_anchor[a_] += pb * pa
                for w_, pw in m['works_on'][a_].items():
                    p_works[w_] += pb * pa * pw
        new_act, new_dom = equalize(new_act, p_works), equalize(new_dom, p_anchor)
        action = {c: blend(action.get(c, action_prior), new_act[c], cap_act) for c in m['classes']}
        action_none = blend(action_none, new_act['none'], cap_act)
        domain = {c: blend(domain.get(c, domain_prior), new_dom[c], cap_dom) for c in m['classes']}
        domain_none = blend(domain_none, new_dom['none'], cap_dom)
        langs['other-code'] = norm(v['language_weights'])

    def subdomain(spec, rng):
        subs = v['subdomains'].get(spec['domain']) if v else None
        if subs and rng.random() >= v['open_subdomain']:
            spec['subdomain'] = rng.choice(subs)['name']
        return spec

    def detail_dist(c, dom):
        """TB-fitted details of class c (merged, tempered, domain-conditional) blended with the new formats of c that
        are common in the domain; a new class has only new formats."""
        tb = details_dom.get((c, dom), details[c]['dist']) if c in details else {}
        if not v or c not in v['formats']:
            return tb
        fs = v['formats'][c]
        new = uniform([f['name'] for f in fs if dom in f['domains'] or 'any' in f['domains']])
        if tb:  # no new format is common in this domain: the TB formats alone, rather than a forced pairing
            share = v['new_format_share']
            if share == 'flat':   # every eligible format weighs the same, old or new
                share = len(new) / (len(tb) + len(new)) if new else 0
            return blend(tb, new, share)
        return new or uniform([f['name'] for f in fs])

    def sample(rng):
        bk = draw(mix, rng)
        if bk == 'none':
            return subdomain({'bucket': 'none', 'has_assets': False, 'agent_action': draw(action_none, rng),
                              'domain': draw(domain_none, rng)}, rng)
        a = draw(anchor[bk], rng)
        works_on = draw(m['works_on'][a], rng)
        classes = [a] + companions(m, a, base, rng, () if works_on == a else (works_on,))
        # Roles once the bundle is complete: a companion can be derived only from a class it was derived from in
        # some real task; with no such class in the bundle it is authored instead.
        roles = {c: draw(m['role'].get((bk, c, any(m['edges'][c][p] > 0 for p in classes if p != c)), m['role_prior']),
                         rng)
                 for c in classes[1:]}
        parent = {}
        for c in classes[1:]:
            opts = {p: m['edges'][c][p] for p in classes if p != c and m['edges'][c][p] > 0}
            if roles[c] == 'derived' and opts:
                parent[c] = draw(opts, rng)
            elif roles[c] == 'derived':
                roles[c] = 'authored'
        for c in list(parent):  # break derivation cycles: the asset that closes a loop becomes authored
            seen, x = set(), c
            while x in parent and x not in seen:
                seen.add(x)
                x = parent[x]
            if x in parent:
                del parent[x]
                roles[x] = 'authored'
        order = [c for c in classes if c not in parent]  # anchor first, then parents before what derives from them
        while len(order) < len(classes):
            order += [c for c in parent if c not in order and parent[c] in order]
        dom = draw(domain.get(a, domain_prior), rng)
        assets = []
        for c in order:
            x = {'class': c, 'role': 'anchor' if c == a else roles[c]}
            if c in parent:
                x['derived_from'] = order.index(parent[c])
            if c in langs:
                x['language'] = draw(langs[c], rng)
            if c in details:
                d = details[c]
                x[d['field']] = draw_k(detail_dist(c, dom), draw(d['n'], rng), rng)
            elif v and c in v['formats']:  # a v2 class: as many formats as a typical class, new formats only
                x[v['field'][c]] = draw_k(detail_dist(c, dom), draw(v['n_formats'], rng), rng)
            assets.append(x)
        has_code = bool(CODE_CLASSES & set(classes))
        orig = draw(origin[bk], rng).split('+')
        spec = {'bucket': bk, 'has_assets': True, 'anchor_class': a, 'agent_works_on': works_on,
                'agent_action': draw(action.get(works_on, action_prior), rng),
                'domain': dom,
                'code_scale': draw(m['scale'][(bk, 'upstream-project' in classes)], rng) if has_code else 'none',
                'assets': assets, 'origin': orig}
        subdomain(spec, rng)
        # Borrow file count and size from a real task with the same shipping mode, favouring a similar bundle.
        shipped = 'shipped' in orig
        donors = {i: w * (2 if db == bk else 1) * (3 if da == a else 1) * (3 if ds == spec['code_scale'] else 1)
                     * (2 if min(k, 3) == min(len(classes), 3) else 1)
                  for i, (w, db, da, ds, k, nf, nb) in enumerate(m['donors']) if (nf > 0) == shipped}
        if shipped and donors:
            *_, nf, nb = m['donors'][draw(donors, rng)]
            spec['shipped_files'] = max(1, round(nf * math.exp(rng.gauss(0, 0.35))))
            spec['shipped_size'] = size_bucket(nb * math.exp(rng.gauss(0, 0.6)))
        return spec
    return sample


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def dist(xs):
    return norm(collections.Counter(xs))


def check(rows, a, b, rng, tau_class, tau_detail, n=20000):
    def summarize(specs):
        full = [s for s in specs if s['has_assets']]
        return {'bucket': dist(s['bucket'] for s in specs),
                'anchor': {bk: dist(s['anchor_class'] for s in full if s['bucket'] == bk) for bk in BUCKETS[1:]},
                'inc': {c: sum(c in s['classes'] for s in full) / len(full) for c in CLASSES},
                'k': sum(len(s['classes']) for s in full) / len(full),
                'sets': [frozenset(s['classes']) for s in full],
                'role': dist(r for s in full for r in s['roles']),
                'on_anchor': sum(s['works_on'] == s['anchor_class'] for s in full) / len(full),
                'scale': dist(s['code_scale'] for s in full), 'action': dist(s['agent_action'] for s in specs),
                'domain': dist(s['domain'] for s in specs)}

    def from_rows(rs):
        return [{'bucket': r['bucket'], 'has_assets': r['bucket'] != 'none', 'agent_action': r['action'],
                 'domain': r['domain'], **({'anchor_class': r['anchor'], 'classes': r['classes'], 'works_on': r['works_on'],
                                            'roles': [v for v in r['roles'].values() if v], 'code_scale': r['scale']}
                                           if r['bucket'] != 'none' else {})} for r in rs]

    def from_specs(specs):
        return [dict(s, classes=[x['class'] for x in s['assets']], works_on=s['agent_works_on'],
                     roles=[x['role'] for x in s['assets'][1:]]) if s['has_assets'] else s for s in specs]

    def lift(sets, x, y):
        nn, cx, cy = len(sets), sum(x in s for s in sets), sum(y in s for s in sets)
        return sum(x in s and y in s for s in sets) * nn / (cx * cy) if cx and cy else float('nan')

    for lam, s in ((0.0, a), (1.0, b)):
        m = fit(rows, a, b, lam, rng)
        sample = sampler(m, 1, 1)
        R, S = summarize(from_rows([r for r in rows if r['set'] == s])), summarize(from_specs([sample(rng) for _ in range(n)]))
        print(f'== lam={lam} vs {s}, tau=1')
        print('   bucket mix, real -> sampled: ' + ', '.join(f'{k} {R["bucket"].get(k, 0):.0%}->{S["bucket"].get(k, 0):.0%}' for k in BUCKETS))
        print('   anchor class per bucket, total-variation distance: ' + ', '.join(f'{bk} {tv(R["anchor"][bk], S["anchor"][bk]):.2f}' for bk in BUCKETS[1:]))
        gaps = sorted(((abs(S['inc'][c] - R['inc'][c]), c) for c in CLASSES), reverse=True)[:4]
        print(f'   classes per task {R["k"]:.2f}->{S["k"]:.2f}; largest class-share gaps: ' +
              ', '.join(f'{c} {R["inc"][c]:.0%}->{S["inc"][c]:.0%}' for _, c in gaps))
        top = collections.Counter(p for t in R['sets'] for p in itertools.combinations(sorted(t), 2)).most_common(5)
        print('   top pairs, lift real -> sampled: ' + ', '.join(f'{x}+{y} {lift(R["sets"], x, y):.1f}->{lift(S["sets"], x, y):.1f}' for (x, y), _ in top))
        print(f'   agent works on the anchor {R["on_anchor"]:.0%}->{S["on_anchor"]:.0%}; companion roles: ' +
              ', '.join(f'{k} {R["role"].get(k, 0):.0%}->{S["role"].get(k, 0):.0%}' for k in ROLES))
        print('   code scale: ' + ', '.join(f'{k} {R["scale"].get(k, 0):.0%}->{S["scale"].get(k, 0):.0%}' for k in ['none'] + SCALES))
        print(f'   action TV {tv(R["action"], S["action"]):.2f}, domain TV {tv(R["domain"], S["domain"]):.2f}; '
              f'distinct bundles: real {len(set(R["sets"]))} of {len(R["sets"])}, sampled {len(set(S["sets"][:1000]))} per 1000')
    m = fit(rows, a, b, 0.5, rng)
    for tc, td in ((1, 1), (tau_class, tau_detail)):
        S = summarize(from_specs([sampler(m, tc, td)(rng) for _ in range(n)]))
        print(f'== lam=0.5, tau_class={tc}, tau_detail={td}')
        print('   bucket mix: ' + ', '.join(f'{k} {S["bucket"].get(k, 0):.0%}' for k in BUCKETS))
        for bk in BUCKETS[1:]:
            print(f'   anchors in {bk}: ' + ', '.join(f'{c} {p:.0%}' for c, p in sorted(S['anchor'][bk].items(), key=lambda kv: -kv[1])[:6]))
        print('   code scale: ' + ', '.join(f'{k} {S["scale"].get(k, 0):.0%}' for k in ['none'] + SCALES))
        print('   actions: ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(S['action'].items(), key=lambda kv: -kv[1])[:7]))
        print('   domains: ' + ', '.join(f'{k} {v:.0%}' for k, v in sorted(S['domain'].items(), key=lambda kv: -kv[1])[:7]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--labels', default=os.path.join(HERE, 'asset_labels.json'))
    ap.add_argument('--buckets', default=os.path.join(HERE, 'anchor_buckets.json'))
    ap.add_argument('--anchors', default=os.path.join(HERE, 'anchor_labels.json'))
    ap.add_argument('--a', default='TB2.1', help='release at lam=0')
    ap.add_argument('--b', default='TB3.0', help='release at lam=1')
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--tau-class', type=float, default=1.5)
    ap.add_argument('--tau-detail', type=float, default=2.0)
    ap.add_argument('--bucket-mix', help='override, e.g. none=0.1,best-generated=0.5,best-found-online=0.15,must-source=0.25')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='-')
    ap.add_argument('--vocab', help='vocabulary v2 file (e.g. vocab_v2.json): broadened options blended with the TB fit')
    ap.add_argument('--check', action='store_true', help='compare fitted samples with the real releases')
    args = ap.parse_args()
    rows = load(args.labels, args.buckets, args.anchors)
    rng = random.Random(args.seed)
    if args.check:
        return check(rows, args.a, args.b, rng, args.tau_class, args.tau_detail)
    mix = None
    if args.bucket_mix:
        mix = norm({k: float(v) for k, v in (kv.split('=') for kv in args.bucket_mix.split(','))})
        assert set(mix) <= set(BUCKETS), f'buckets are {BUCKETS}'
    m = fit(rows, args.a, args.b, args.lam, rng)
    if args.vocab:
        m = extend(m, json.load(open(args.vocab)))
        # a vocabulary variant may set its own temperatures (designs D2/D3 flatten the TB-fitted parts)
        args.tau_class = m['v'].get('tau_class', args.tau_class)
        args.tau_detail = m['v'].get('tau_detail', args.tau_detail)
    sample = sampler(m, args.tau_class, args.tau_detail, mix)
    out = open(args.out, 'w') if args.out != '-' else None
    for i in range(args.n):
        spec = {'id': f'asset-spec-{args.seed}-{i:06d}', 'lam': args.lam, 'tau_class': args.tau_class,
                'tau_detail': args.tau_detail, **({'vocab': m['v']['version']} if args.vocab else {}), **sample(rng)}
        print(json.dumps(spec), file=out)


if __name__ == '__main__':
    main()
