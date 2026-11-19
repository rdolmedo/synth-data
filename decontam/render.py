"""Step 2: one text bundle per task, composed per pair as rubric -> TB2.1 bundle -> overlap signals -> candidate.

  python -m decontam.render --work /tmp/decontam/work

Per task: task.toml metadata, instruction, tests, solution and Dockerfile in full (a single file over FILE_CAP tokens
goes in as a head excerpt), the environment file tree with sizes, then environment text files within ENV_BUDGET
tokens (water-filled: small files whole, the largest as head excerpts). Binaries are listed by name and size only.
A bundle that cut anything is marked truncated. The pair's deterministic overlap signals (shared files, per-field
containment, shared identifiers, retrieval ranks) sit between the two bundles, so every prompt for one TB2.1 task
shares the rubric + TB2.1 prefix in the prefix cache.
Writes <work>/bundles.jsonl ({side, name, text, tokens, truncated, ...}) and <work>/render.json (size stats).
"""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

from . import tasks as T

TOKENIZER = ('/data/aai/tbench/synth/shared/checkpoints/deepseek-v41-flash-1017/'
             '1bae0e1088993b9b82de08f59de5be5a34e42170/tokenizer.json')
ENV_BUDGET = 16000
FILE_CAP = 24000
_TOK = None
# Labels the task_synthesis driver (decontam/synth.py) overrides: reference task -> benchmark, the candidate set.
REF_BENCH = {}
CANDIDATE_TITLE = 'CANDIDATE TASK (CalibForge, training data)'
CANDIDATE_POOL = '5,431 CalibForge tasks'


def encode(text):
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(TOKENIZER)
    return _TOK.encode(text, add_special_tokens=False)


def ntok(text):
    return len(encode(text).ids)


def head(text, budget):
    """The first ~budget tokens of text (cut on a line boundary when possible)."""
    enc = encode(text)
    if len(enc.ids) <= budget:
        return text, False
    cut = text[:enc.offsets[budget][0]]
    nl = cut.rfind('\n')
    return (cut[:nl] if nl > len(cut) // 2 else cut), True


def fence(path, text, note=''):
    tick = '````' if '```' in text else '```'
    return f'#### {path}{note}\n{tick}\n{text.rstrip()}\n{tick}\n'


def size(n):
    return f'{n} B' if n < 1024 else f'{n / 1024:.1f} KiB' if n < 1 << 20 else f'{n / (1 << 20):.1f} MiB'


def render(task_dir, side):
    return render_task(T.load(task_dir), side, side == 'cf' and (Path(task_dir) / 'tasksynth.json').exists())


def render_task(task, side, synth=False, extra=None, tree_cap=None):
    """extra: a markdown section placed after the instruction (decontam.pool: a real asset's provenance); tree_cap:
    (path prefix, n): list at most n of the environment tree's files under that prefix (a real asset's archive can
    hold thousands; other files are always listed)."""
    title = (f'TEST TASK ({REF_BENCH.get(task["name"], "Terminal-Bench 2.1")}, held out)' if side == 'tb'
             else CANDIDATE_TITLE)
    out = [f'## {title}: `{task["name"]}`\n']
    meta = [l for l in task['toml'].splitlines() if l.split('=')[0].strip() in
            ('difficulty', 'category', 'tags', 'expert_time_estimate_min')]
    if meta:
        out.append('### Metadata (task.toml)\n```\n' + '\n'.join(meta) + '\n```\n')
    out.append(fence('instruction.md', task['instruction']).replace('#### ', '### ', 1))
    if extra:
        out.append(extra)
    truncated, cut = False, []
    by_part = collections.defaultdict(list)
    for f in task['files']:
        by_part[f['part']].append(f)
    sections = (('Tests (tests/)', 'tests'), ('Reference solution (solution/)', 'solution'), ('Dockerfile', 'dockerfile'))
    if synth:
        sections = (('Tests (tests/)', 'tests'), ("Author's solution sketch (solution/sketch.md; private, never shown to "
                                                  'the agent)', 'solution'), ('Dockerfile', 'dockerfile'))
    for name, part in sections:
        files = by_part.get(part, [])
        if not files:
            out.append(f'### {name}\n' + ('(none: this training set ships no tests or verifier)\n'
                                          if synth and part == 'tests' else '(none)\n'))
            continue
        out.append(f'### {name}\n')
        for f in files:
            if 'text' not in f:
                out.append(f'#### {f["path"]} (binary, {size(f["size"])}; content not shown)\n')
                continue
            text, was_cut = head(f['text'], FILE_CAP)
            was_cut = was_cut or f.get('partial_read')
            out.append(fence(f['path'], text, f' (HEAD EXCERPT of {size(f["size"])})' if was_cut else ''))
            if was_cut:
                truncated = True
                cut.append(f['path'])
    env = by_part.get('environment', []) + by_part.get('top', [])
    tree, capped, under = [], [], 0
    for f in env:
        if tree_cap and f['path'].startswith(tree_cap[0]):
            under += 1
            if under > tree_cap[1]:
                capped.append(f)
                continue
        tree.append(f'{f["path"]}  {size(f["size"])}{"" if "text" in f else "  [binary]"}')
    if capped:
        tree.append(f'... and {len(capped)} more files under {tree_cap[0]} ({size(sum(f["size"] for f in capped))})')
    out.append('### Environment and other files: tree\n```\n' + '\n'.join(tree) + '\n```\n' if env else
               '### Environment and other files: tree\n(none besides the Dockerfile)\n')
    texts = [f for f in env if 'text' in f]
    if texts:
        tokens = {f['path']: ntok(f['text']) for f in texts}
        remaining, left = ENV_BUDGET, sorted(texts, key=lambda f: tokens[f['path']])
        alloc = {}
        while left:  # water-filling: equal share, small files keep what they need
            share = remaining / len(left)
            f = left.pop(0)
            alloc[f['path']] = min(tokens[f['path']], math.floor(share))
            remaining -= alloc[f['path']]
        out.append(f'### Environment text files (budget ~{ENV_BUDGET} tokens)\n')
        for f in texts:
            budget = alloc[f['path']]
            if budget < 50:
                out.append(f'#### {f["path"]} (omitted: {size(f["size"])}, over the budget)\n')
                truncated = True
                cut.append(f['path'])
                continue
            text, was_cut = head(f['text'], budget) if budget < tokens[f['path']] else (f['text'], False)
            was_cut = was_cut or f.get('partial_read')
            out.append(fence(f['path'], text, f' (HEAD EXCERPT of {size(f["size"])})' if was_cut else ''))
            if was_cut:
                truncated = True
                cut.append(f['path'])
    if truncated:
        out.insert(1, f'**Note: this bundle is TRUNCATED: {len(cut)} file(s) shown as head excerpts or omitted.**\n')
    text = '\n'.join(out)
    return {'side': side, 'name': task['name'], 'text': text, 'tokens': ntok(text), 'truncated': truncated,
            'cut': cut, 'canaries': T.canaries(task), 'ids': sorted(T.identifiers(task)),
            'files': [[f['path'], f['sha1'], f['size']] for f in task['files']]}


def _render(args):
    return render(*args)


def signals_text(pair, shared_ids, shared_files):
    lines = ['## Deterministic overlap signals for this pair (computed, not judged)']
    ranks = ', '.join(f'{s} #{r}' for s, r in pair['rank'].items())
    lines.append(f'- Retrieval rank of the candidate among {CANDIDATE_POOL} for this test task: {ranks}')
    c = pair.get('containment', {})
    lines.append('- Shingle containment |A∩B|/min(|A|,|B|) after dropping boilerplate shingles: ' +
                 (', '.join(f'{k} {v:.2f}' for k, v in c.items()) if c else 'none measurable'))
    lines.append('- Identical files (>= 300 bytes, rare): ' +
                 ('; '.join(f'{a} == {b}' for a, b in shared_files) if shared_files else 'none'))
    lines.append('- Shared rare identifiers (paths, filenames, snake/camel names in instruction+tests): ' +
                 (', '.join(f'`{x}`' for x in shared_ids[:30]) if shared_ids else 'none'))
    return '\n'.join(lines) + '\n'


def score_pairs(work, wanted):
    """Pair rows (ranks, containment, shared-file count) from retrieve's score matrices, for pairs it did not emit
    (propagation)."""
    import numpy as np
    with np.load(Path(work) / 'scores.npz') as npz:
        z = {k: npz[k] for k in npz.files}  # NpzFile decompresses on every access
    ti, ci = {n: i for i, n in enumerate(z['tb'])}, {n: j for j, n in enumerate(z['cf'])}
    signals = [k[5:] for k in z if k.startswith('rank_')]
    fields = [k[8:] for k in z if k.startswith('contain_')]
    for p in wanted:
        i, j = ti[p['tb']], ci[p['cf']]
        yield {**p, 'rank': {s: int(z[f'rank_{s}'][i, j]) for s in signals},
               'containment': {f: round(float(z[f'contain_{f}'][i, j]), 3) for f in fields if z[f'contain_{f}'][i, j]},
               'shared_files': int(z['shared_files'][i, j])}


def pair_signals(work, bundles, pairs=None, out_name='pair_signals.jsonl'):
    """<work>/<out_name>: the overlap-signal text of every retrieved pair (or of the given pair rows)."""
    cf = [b for b in bundles.values() if b['side'] == 'cf']
    id_df = collections.Counter(x for b in cf for x in b['ids'])
    blob_df = collections.Counter(h for b in bundles.values() for h in {f[1] for f in b['files'] if f[2] >= 300})
    n = 0
    if pairs is None:
        pairs = map(json.loads, open(Path(work) / 'pairs.jsonl'))
    with open(Path(work) / out_name, 'w') as out:
        for pair in pairs:
            a, b = bundles[('tb', pair['tb'])], bundles[('cf', pair['cf'])]
            ids = sorted(set(a['ids']) & set(b['ids']), key=lambda x: (id_df[x], x))
            mine = {f[1]: f[0] for f in a['files'] if f[2] >= 300 and blob_df[f[1]] <= 3}
            files = [(mine[f[1]], f[0]) for f in b['files'] if f[1] in mine]
            out.write(json.dumps({'tb': pair['tb'], 'cf': pair['cf'], 'text': signals_text(pair, ids, files)}) + '\n')
            n += 1
    return n


def main(work):
    work = Path(work)
    tb = T.extract(T.TB_TAR, work / 'tb')
    cf = T.extract(T.CF_TAR, work / 'cf')
    jobs = [(d, 'tb') for d in tb.values()] + [(d, 'cf') for d in cf.values()]
    bundles = {}
    with ProcessPoolExecutor() as pool, (work / 'bundles.jsonl').open('w') as f:
        rows = []
        for row in pool.map(_render, jobs, chunksize=16):
            f.write(json.dumps(row) + '\n')
            rows.append({k: row[k] for k in ('side', 'name', 'tokens', 'truncated')})
            bundles[(row['side'], row['name'])] = {k: row[k] for k in ('side', 'ids', 'files')}
    if (work / 'pairs.jsonl').exists():
        print(f'{pair_signals(work, bundles)} pair signal blocks')
    stats = {}
    for side in ('tb', 'cf'):
        toks = sorted(r['tokens'] for r in rows if r['side'] == side)
        q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))]
        stats[side] = {'n': len(toks), 'p50': q(.5), 'p90': q(.9), 'p99': q(.99), 'max': toks[-1],
                       'truncated': sum(r['truncated'] for r in rows if r['side'] == side)}
    (work / 'render.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def bundle_index(work):
    bundles = {}
    with open(Path(work) / 'bundles.jsonl') as f:
        for line in f:
            row = json.loads(line)
            bundles[(row['side'], row['name'])] = {k: row[k] for k in ('side', 'ids', 'files')}
    return bundles


def extra(work, pairs_file, out_name):
    """Signal blocks for pairs outside retrieval (e.g. propagation), in their own file."""
    wanted = [json.loads(l) for l in open(pairs_file)]
    print(f'{pair_signals(work, bundle_index(work), score_pairs(work, wanted), out_name)} pair signal blocks -> {out_name}')


def wide(work, pairs_file, out_name, env_budget, file_cap):
    """Step 4 (truncated bundles): re-render only the tasks of the given pairs with larger budgets."""
    global ENV_BUDGET, FILE_CAP
    ENV_BUDGET, FILE_CAP = env_budget, file_cap
    work = Path(work)
    pairs = [json.loads(l) for l in open(pairs_file)]
    tb, cf = T.extract(T.TB_TAR, work / 'tb'), T.extract(T.CF_TAR, work / 'cf')
    jobs = sorted({(str(tb[p['tb']]), 'tb') for p in pairs} | {(str(cf[p['cf']]), 'cf') for p in pairs})
    with ProcessPoolExecutor() as pool, (work / out_name).open('w') as f:
        rows = list(pool.map(_render, jobs))
        for row in rows:
            f.write(json.dumps(row) + '\n')
    print(f'{len(rows)} wide bundles ({sum(r["truncated"] for r in rows)} still truncated, max {max(r["tokens"] for r in rows)} '
          f'tokens) -> {out_name}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--wide-pairs', help='JSONL of {tb, cf}: re-render their tasks with --env-budget/--file-cap')
    ap.add_argument('--env-budget', type=int, default=ENV_BUDGET)
    ap.add_argument('--file-cap', type=int, default=FILE_CAP)
    ap.add_argument('--extra-pairs', help='JSONL of {tb, cf}: write signal blocks for these pairs only')
    ap.add_argument('--out-name', default='pair_signals.jsonl')
    ap.add_argument('--signals-only', action='store_true', help='pair signals of pairs.jsonl from existing bundles')
    a = ap.parse_args()
    if a.wide_pairs:
        wide(a.work, a.wide_pairs, a.out_name, a.env_budget, a.file_cap)
    elif a.extra_pairs:
        extra(a.work, a.extra_pairs, a.out_name)
    elif a.signals_only:
        print(f'{pair_signals(Path(a.work), bundle_index(a.work))} pair signal blocks')
    else:
        main(a.work)
