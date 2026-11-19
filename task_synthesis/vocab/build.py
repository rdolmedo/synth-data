"""Assemble asset_sampler/vocab_v2.json from the brainstorm outputs (WORK, see brainstorm.py) and print a summary.

  python3 -m task_synthesis.vocab.build [--alpha 0.4] [--out task_synthesis/asset_sampler/vocab_v2.json]

Settings agreed on #25 (2026-10-31): new top-level options get at most 25% of each dimension (cap_new), 20% of specs
draw no sub-domain (open_subdomain). Judgment calls made here: new formats get half of a class's format draws
(new_format_share) and other-code languages are weighted by real-world share ** alpha."""
import argparse
import json
import re
from pathlib import Path

from .brainstorm import CLASS_DEFS, MERGE_DETAILS, SA, WORK, load

OUT = Path(__file__).resolve().parents[1] / 'asset_sampler' / 'vocab_v2.json'
LABELS = {'rich-history': 'long, meaningful history'}
SHARE_FLOOR = 0.001   # percent of public code: rarer languages are treated as this rare


def kebab(s):
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')


def build(alpha, cap, open_sub, fmt_share, version='v2', taus=None):
    act, dom, cls = (load(WORK / 'toplevel' / f'{k}.json') for k in ('action', 'domain', 'class'))
    v = {'version': version, 'cap_new': cap, 'open_subdomain': open_sub, 'new_format_share': fmt_share,
         **{k: x for k, x in (taus or {}).items() if x is not None},
         'language_alpha': alpha, 'merge_details': MERGE_DETAILS, 'labels': dict(LABELS),
         'actions': [{'name': kebab(x['name']), 'description': x['description']} for x in act['kept']],
         'domains': [{'name': kebab(x['name']), 'scope': dom['scopes'].get(x['name'], '')} for x in dom['kept']],
         'domain_scopes': dom['scopes'],
         'classes': [{'name': kebab(x['name']), 'description': x['description'], 'field': 'formats'}
                     for x in cls['kept']]}
    domains = dom['current'] + [d['name'] for d in v['domains']]
    v['subdomains'] = {}
    for d in domains:
        if not (WORK / 'subdomains' / f'{d}.json').exists():
            print(f'WARNING: no sub-domains for {d} yet')
            continue
        s = load(WORK / 'subdomains' / f'{d}.json')
        kept = [{'name': x['name'].strip(), 'note': x.get('note', '')} for x in s['kept']]
        cov = WORK / 'coverage' / f'{d}.json'
        if cov.exists():
            kept += [{'name': x['name'].strip(), 'note': x.get('note', ''), 'from': 'coverage'} for x in load(cov)['added']]
        seen, uniq = set(), []
        for x in kept:
            if x['name'].lower() not in seen:
                seen.add(x['name'].lower())
                uniq.append(x)
        v['subdomains'][d] = uniq
    v['formats'] = {}
    for f in sorted((WORK / 'formats').glob('*.json')):
        s = load(f)
        c = s['class']
        current = set(SA.DETAILS.get(c, (None, {}))[1]) | set(s['current'])
        fs, seen = [], set()
        for x in s['kept']:
            n = kebab(x['name'])
            if n in current or n in seen:
                continue
            seen.add(n)
            ds = [d for d in x.get('domains', ['any']) if d in domains or d == 'any'] or ['any']
            fs.append({'name': n, 'label': x.get('label', n), 'domains': ds})
            v['labels'][n] = x.get('label', n)
        v['formats'][c] = fs
    lang = load(WORK / 'languages.json')
    # Languages without an offline toolchain are dropped, except the ones TB already uses (every TB option is kept):
    # their code can still be read, ported or analyzed, as with CUDA without a GPU (specs.describe_asset says so).
    rows = [x for x in lang['languages'] if x.get('offline') in ('yes', 'partial') or x['name'] in lang['required']]
    names = {x['name'] for x in rows}
    rows += [{'name': l, 'share_pct': SHARE_FLOOR, 'offline': 'unknown'} for l in lang['required'] if l not in names]
    v['languages'] = rows
    v['no_toolchain'] = sorted(x['name'] for x in rows if x.get('offline') == 'no')
    v['language_weights'] = {x['name']: max(float(x['share_pct']), SHARE_FLOOR) ** alpha for x in rows
                             if x['name'] not in ('Python', 'C', 'C++', 'CUDA')}
    v['compat'] = {f.stem: load(f) for f in sorted((WORK / 'compat').glob('*.json'))}
    v['pairs'] = load(WORK / 'pairs.json')
    return v


def summary(v):
    n_sub = {d: len(s) for d, s in v['subdomains'].items()}
    w = v['language_weights']
    tot = sum(w.values())
    top = sorted(w.items(), key=lambda kv: -kv[1])
    print(f"new actions {len(v['actions'])}, domains {len(v['domains'])}, classes {len(v['classes'])}")
    print(f"sub-domains {sum(n_sub.values())} total, per domain {min(n_sub.values())}-{max(n_sub.values())}: {n_sub}")
    print(f"new formats {sum(len(x) for x in v['formats'].values())}: " +
          ', '.join(f'{c} {len(x)}' for c, x in v['formats'].items()))
    print(f"languages {len(w)}: top {[(k, f'{x / tot:.1%}') for k, x in top[:5]]}, "
          f"bottom {[(k, f'{x / tot:.2%}') for k, x in top[-3:]]}")
    for l in ('Redcode', 'COBOL', 'POV-Ray SDL', 'Stan', 'VBA', 'JavaScript', 'Go'):
        if l in w:
            print(f'  {l}: {w[l] / tot:.2%}')
    missing = [c for c in ['none'] + SA.CLASSES + [x['name'] for x in v['classes']] if c not in v['compat']]
    print(f"compat classes {len(v['compat'])}, missing {missing}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--alpha', type=float, default=0.4)
    ap.add_argument('--cap', default='0.25', help="new options' share per dimension, or 'flat' (equal to any option)")
    ap.add_argument('--open-subdomain', type=float, default=0.2)
    ap.add_argument('--format-share', default='0.5', help="new formats' share within a class, or 'flat'")
    ap.add_argument('--version', default='v2', help='name recorded in each spec (specs.VOCABS maps it to its file)')
    ap.add_argument('--tau-class', type=float, help='override the sampler temperature for classes/actions/anchors')
    ap.add_argument('--tau-detail', type=float, help='override the sampler temperature for domains/details/languages')
    ap.add_argument('--tau-origin', type=float, help='origin temperature (build mechanics; default: tau-class)')
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    num = lambda x: x if x == 'flat' else float(x)
    v = build(a.alpha, num(a.cap), a.open_subdomain, num(a.format_share), a.version,
              {'tau_class': a.tau_class, 'tau_detail': a.tau_detail, 'tau_origin': a.tau_origin})
    summary(v)
    if not a.dry:
        Path(a.out).write_text(json.dumps(v, indent=1) + '\n')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
