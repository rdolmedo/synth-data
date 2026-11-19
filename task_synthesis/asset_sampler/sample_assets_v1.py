#!/usr/bin/env python3
"""Sample asset specs for Terminal-Bench-style environment synthesis.

Fits a small generative model to the per-task asset labels of two releases
(asset_labels.json) and samples specs that sit between them:

  1. has assets?            Bernoulli at the weighted rate              untempered
  2. primary class          smoothed categorical                        tau_class
  3. number of secondaries  weighted empirical                          untempered
  4. secondaries            base weight x pairwise lift, one at a time  tau_class (base only)
  5. per-class details      formats, languages, services, ...           tau_detail
  6. origin                 conditional on the primary class            tau_class
  7. shipped files and size borrowed from a similar real task           untempered

Tasks of release A weigh (1 - lam) and tasks of release B weigh lam, so lam=0
reproduces A and lam=1 reproduces B. Temperature only touches *which* classes and
formats appear (and, mildly, the origin); pairwise lifts (which classes go together), bundle size and file
sizes keep their fitted shape. Only class-level statistics leave this script: no
task names or descriptions are written to the specs.

  python3 sample_assets.py --n 1000 --out specs.jsonl
  python3 sample_assets.py --check
"""
import argparse, collections, itertools, json, math, os, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ['python-code', 'c-cpp-code', 'other-code', 'git-repo', 'upstream-project', 'tabular-data',
           'structured-data', 'text-data', 'database', 'binary', 'media', 'document', 'ml-model', 'archive',
           'config', 'running-service', 'scientific-data', 'vm-image']  # 'other' is a labelling remainder, not sampled
SIZE_BUCKETS = [(1e4, '<10KB'), (1e5, '10-100KB'), (1e6, '100KB-1MB'), (1e7, '1-10MB'), (1e8, '10-100MB'),
                (math.inf, '>100MB')]
SRC_BUCKETS = [(1, '1'), (5, '2-5'), (20, '6-20'), (math.inf, '21+')]
CODE_EXT = {'python-code': {'.py'},
            'c-cpp-code': {'.c', '.h', '.cc', '.cpp', '.hpp', '.cu'},
            'other-code': {'.js', '.ts', '.tsx', '.rs', '.go', '.java', '.kt', '.scala', '.ml', '.cbl', '.r', '.stan',
                           '.scm', '.pl', '.sh', '.v', '.lean', '.red', '.pov', '.sql', '.bas', '.f90', '.rb', '.hs'}}
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


def bucket(x, buckets):
    return next(name for hi, name in buckets if x <= hi)


def matches(tokens, ext, desc):
    return any(t in ext if t.startswith('.') else re.search(r'(?<![a-z0-9])' + re.escape(t) + r'(?![a-z0-9])', desc)
               for t in tokens)


def clean(rows):
    """Drop the catch-all 'other' class; a task whose only class was 'other' leaves the fit."""
    out = []
    for r in rows:
        classes = [c for c in r['classes'] if c in CLASSES or c == 'none']
        if classes:
            out.append(dict(r, classes=classes, primary=r['primary'] if r['primary'] in classes else classes[0]))
    return out


def weighted(rows, a, b, lam):
    na, nb = (sum(r['set'] == s for r in rows) for s in (a, b))
    nbar = (1 - lam) * na + lam * nb  # weights sum to the size of a release, so pseudo-counts stay in task units
    w = {a: (1 - lam) * nbar / na, b: lam * nbar / nb}
    return [(w[r['set']], r) for r in rows if r['set'] in w and w[r['set']] > 0]


def fit(rows, a, b, lam, rng, k_pair=10.0, a_class=0.2, a_detail=0.3):
    W = weighted(rows, a, b, lam)
    A = [(w, r) for w, r in W if r['primary'] != 'none']
    n_a = sum(w for w, _ in A)
    m = {'p_assets': n_a / sum(w for w, _ in W)}

    prim, nsec, inc, sec, pair = (collections.Counter() for _ in range(5))
    nsec_by = collections.defaultdict(collections.Counter)
    for w, r in A:
        prim[r['primary']] += w
        nsec[len(r['classes']) - 1] += w
        nsec_by[r['primary']][len(r['classes']) - 1] += w
        for c in r['classes']:
            inc[c] += w
            sec[c] += w * (c != r['primary'])
        for p in itertools.combinations(sorted(r['classes']), 2):
            pair[p] += w
    m['primary'] = norm({c: prim[c] + a_class for c in CLASSES})
    # Bundle size depends on the primary (binaries come with their C source, services with 3-6 classes),
    # backed off to the overall size distribution with 2 pseudo-tasks.
    m['n_secondary'] = {c: norm({k: nsec_by[c][k] + 2 * v for k, v in norm(nsec).items()}) for c in CLASSES}
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

    m['details'] = {}
    for c, (field, opts) in DETAILS.items():
        counts, nopt = collections.Counter({o: a_detail for o in opts}), collections.Counter()
        for w, r in A:
            if c in r['classes']:
                hit = [o for o, toks in opts.items() if matches(toks, r['ext'], r['description'].lower())]
                for o in hit:
                    counts[o] += w / len(hit)
                if hit:
                    nopt[min(len(hit), 4)] += w
        m['details'][c] = {'field': field, 'dist': norm(counts), 'n': norm(nopt) or {1: 1.0}}

    m['languages'], m['source_files'] = {}, {}
    for c, fixed in CODE_LANGS.items():
        langs = collections.Counter({l: a_detail for l in (fixed or EXTRA_LANGS)})
        src = collections.Counter()
        for w, r in A:
            if c not in r['classes']:
                continue
            keep = (lambda l: l in fixed) if fixed else (lambda l: l not in ('Python', 'C', 'C++', 'CUDA'))
            seen = [l for l in (LANG_ALIASES.get(l, l) for l in r['languages']) if keep(l)]
            for l in seen:
                langs[l] += w / len(seen)
            n_src = sum(v for e, v in r['ext'].items() if e in CODE_EXT[c])
            if n_src:
                src[bucket(n_src, SRC_BUCKETS)] += w
        m['languages'][c] = norm(langs)
        m['source_files'][c] = norm(src) or {'1': 1.0}

    org, org_by = collections.Counter(), collections.defaultdict(collections.Counter)
    for w, r in A:
        o = '+'.join(sorted(r['origin']))
        org[o] += w
        org_by[r['primary']][o] += w
    pg = norm(org)
    m['origin'] = {c: norm({o: org_by[c][o] + 3 * p for o, p in pg.items()}) for c in CLASSES}
    m['donors'] = [(w, r['primary'], len(r['classes']), r['n_files'], r['total_bytes']) for w, r in A]
    calibrate(m, rng)
    return m


def secondaries(m, primary, base, rng):
    chosen = [primary]
    for _ in range(draw(m['n_secondary'][primary], rng)):
        w = {}
        for c in CLASSES:
            if c in chosen or any(frozenset({c, s}) in EXCLUDE_PAIRS for s in chosen):
                continue
            w[c] = base[c] * math.prod(m['lift'][(c, s)] for s in chosen)
        if not w:
            break
        chosen.append(draw(w, rng))
    return chosen[1:]


def calibrate(m, rng, iters=20, n=8000, step=0.6):
    """Moment-match at tau=1: rescale base weights and pair lifts until each class appears as a secondary,
    and each pair co-occurs, as often as fitted (a pairwise max-entropy fit, done by simulation)."""
    for _ in range(iters):
        got, got_pair = collections.Counter(), collections.Counter()
        for _ in range(n):
            p = draw(m['primary'], rng)
            sec = secondaries(m, p, m['base'], rng)
            got.update(sec)
            got_pair.update(itertools.combinations(sorted([p] + sec), 2))
        for c in CLASSES:
            if m['sec_target'][c] > 0:
                m['base'][c] *= (m['sec_target'][c] / max(got[c] / n, 1 / n)) ** step
        for (x, y), t in m['pair_target'].items():
            m['lift'][(x, y)] *= (t / max(got_pair[(x, y)] / n, 1 / n)) ** step
            m['lift'][(y, x)] = m['lift'][(x, y)]


def sampler(m, tau_class, tau_detail):
    primary = temper(m['primary'], tau_class)
    base = temper(m['base'], tau_class)
    details = {c: dict(d, dist=temper(d['dist'], tau_detail)) for c, d in m['details'].items()}
    langs = {c: temper(d, tau_detail) for c, d in m['languages'].items()}
    origin = {c: temper(d, tau_class) for c, d in m['origin'].items()}  # build mechanics, not formatting

    def sample(rng):
        if rng.random() >= m['p_assets']:
            return {'has_assets': False}
        p = draw(primary, rng)
        classes = [p] + secondaries(m, p, base, rng)
        assets = []
        for c in classes:
            a = {'class': c}
            if c in langs:
                a['language'] = draw(langs[c], rng)
                a['source_files'] = draw(m['source_files'][c], rng)
            if c in details:
                d = details[c]
                a[d['field']] = draw_k(d['dist'], draw(d['n'], rng), rng)
            assets.append(a)
        orig = draw(origin[p], rng).split('+')
        shipped = 'shipped' in orig
        # Borrow file count and size from a real task with the same shipping mode, favouring a similar bundle.
        donors = {i: w * (3 if dp == p else 1) * (2 if min(k, 3) == min(len(classes), 3) else 1)
                  for i, (w, dp, k, nf, nb) in enumerate(m['donors']) if (nf > 0) == shipped}
        spec = {'has_assets': True, 'primary': p, 'classes': classes, 'assets': assets, 'origin': orig}
        if shipped and donors:
            _, _, _, nf, nb = m['donors'][draw(donors, rng)]
            spec['shipped_files'] = max(1, round(nf * math.exp(rng.gauss(0, 0.35))))
            spec['shipped_size'] = bucket(nb * math.exp(rng.gauss(0, 0.6)), SIZE_BUCKETS)
        return spec
    return sample


def stats(specs):
    n = len(specs)
    full = [s for s in specs if s['has_assets']]
    inc = collections.Counter(c for s in full for c in s['classes'])
    return {'none': 1 - len(full) / n, 'k': sum(len(s['classes']) for s in full) / len(full),
            'inc': {c: inc[c] / n for c in CLASSES}, 'sets': [frozenset(s['classes']) for s in full]}


def lift(sets, x, y):
    n, cx, cy = len(sets), sum(x in s for s in sets), sum(y in s for s in sets)
    return sum(x in s and y in s for s in sets) * n / (cx * cy) if cx and cy else float('nan')


def check(rows, a, b, rng, tau_class, tau_detail, n=20000):
    real = {s: [{'has_assets': r['primary'] != 'none', 'classes': r['classes']} for r in rows if r['set'] == s]
            for s in (a, b)}
    for lam, s in ((0.0, a), (1.0, b)):
        m = fit(rows, a, b, lam, rng)
        sim, emp = stats([sampler(m, 1, 1)(rng) for _ in range(n)]), stats(real[s])
        print(f'== lam={lam} vs {s}, tau=1: no assets {emp["none"]:.0%} -> {sim["none"]:.0%}; '
              f'classes per task with assets {emp["k"]:.2f} -> {sim["k"]:.2f}')
        diffs = sorted(((abs(sim['inc'][c] - emp['inc'][c]), c) for c in CLASSES), reverse=True)
        print('   share of tasks with class, real -> sampled (largest gaps): ' + ', '.join(
            f'{c} {emp["inc"][c]:.0%}->{sim["inc"][c]:.0%}' for _, c in diffs[:5]))
        top = collections.Counter(p for t in emp['sets'] for p in itertools.combinations(sorted(t), 2)).most_common(6)
        print('   top pairs, lift real -> sampled: ' + ', '.join(
            f'{x}+{y} {lift(emp["sets"], x, y):.1f}->{lift(sim["sets"], x, y):.1f}' for (x, y), _ in top))
        print(f'   distinct bundles: real {len(set(emp["sets"]))} of {len(emp["sets"])} tasks; '
              f'sampled {len(set(sim["sets"][:1000]))} per 1000')
    m = fit(rows, a, b, 0.5, rng)
    for tc, td in ((1, 1), (tau_class, tau_detail)):
        sample = sampler(m, tc, td)
        specs = [sample(rng) for _ in range(n)]
        st = stats(specs)
        fmts = collections.Counter(f for s in specs if s['has_assets'] for x in s['assets']
                                   if x['class'] == 'tabular-data' for f in x['formats'])
        langs = collections.Counter(x['language'] for s in specs if s['has_assets'] for x in s['assets']
                                    if x['class'] == 'other-code')
        tot = lambda c: sum(c.values())
        print(f'== lam=0.5, tau_class={tc}, tau_detail={td}: no assets {st["none"]:.0%}, classes per task '
              f'{st["k"]:.2f}, distinct bundles {len(set(st["sets"][:1000]))} per 1000')
        print('   share of tasks with class: ' + ', '.join(f'{c} {st["inc"][c]:.0%}' for c in CLASSES))
        print('   tabular formats: ' + ', '.join(f'{k} {v / tot(fmts):.0%}' for k, v in fmts.most_common()))
        print('   other-code languages (top 10 of %d): ' % len(langs) + ', '.join(
            f'{k} {v / tot(langs):.0%}' for k, v in langs.most_common(10)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--labels', default=os.path.join(HERE, 'asset_labels.json'))
    ap.add_argument('--a', default='TB2.1', help='release at lam=0')
    ap.add_argument('--b', default='TB3.0', help='release at lam=1')
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--tau-class', type=float, default=1.5)
    ap.add_argument('--tau-detail', type=float, default=2.0)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='-')
    ap.add_argument('--check', action='store_true', help='compare fitted samples with the real releases')
    args = ap.parse_args()
    rows = clean(json.load(open(args.labels)))
    rng = random.Random(args.seed)
    if args.check:
        return check(rows, args.a, args.b, rng, args.tau_class, args.tau_detail)
    sample = sampler(fit(rows, args.a, args.b, args.lam, rng), args.tau_class, args.tau_detail)
    out = open(args.out, 'w') if args.out != '-' else None
    for i in range(args.n):
        spec = {'id': f'asset-spec-{args.seed}-{i:06d}', 'lam': args.lam, 'tau_class': args.tau_class,
                'tau_detail': args.tau_detail, **sample(rng)}
        print(json.dumps(spec), file=out)


if __name__ == '__main__':
    main()
