"""Taxonomy fit and spec sampler (#359).

  PYTHONPATH=. venvs/automationbench/bin/python -m m5.automationbench.taxonomy fit --labels L.jsonl --out fit.json
  ... taxonomy sample --fit fit.json --leaves DIR --pools DIR --n N --seed S --start K --out specs.jsonl

Top level, fitted per domain on the 600 scored tasks (class-level counts only; the domains get equal shares):
workflow type, number and types of traps (Opus labels, label_test.py), apps (the app keys of the scored tasks'
starting states, 42 apps; no app outside them), number of apps, records before noise, trigger length (without the
benchmark's closing sentences), the closing sentences' rates, the number of target items and of rule sources, and
the rate at which Drive / Gmail are connected without seeded data (upstream's zapier_tools).
Below it: a brainstormed scenario leaf per (domain, workflow) (brainstorm.py, blind to test items), at most 5% open
draws, a sampled fictional company, people's names and a current date.
Merging: workflow and trap labels outside the fixed vocabulary ("other:...") are folded out (reallocated in
proportion); a type seen in at most one scored task overall is dropped. Smoothing: +0.5 per type and domain for
workflows and traps, +0.3 over all 42 apps for apps.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import random
import re
from pathlib import Path

SUFFIXES = [
    "When including values from the source data in your notifications or records, preserve them verbatim (don't paraphrase or round).",
    'Include the names of affected entities and the relevant amounts in your message(s).',
    'Include the relevant amounts from the source data in your message(s).',
    'Include the names of affected entities in your message(s).',
    'Use Gmail for all email sends.',
]
OPEN_DRAW = 0.04   # nominal; the realized share must stay <= 5% (#359)
# "other:" workflow labels merged into the broader fixed type (keyword on the label; first match wins)
OTHER_WORKFLOW = [(('outreach', 'notification', 'dispatch', 'broadcast', 'sending', 'reminder', 'enrollment',
                    'audience', 'shortlist', 'matching', 'signature', 'onboarding', 'publishing'), 'triage_route'),
                  (('scan and flag', 'audit', 'compliance', 'expiry', 'expiration', 'refresh'), 'bulk_cleanup'),
                  (('alert', 'monitor', 'scoring', 'clustering', 'strategy'), 'report'),
                  (('scheduling', 'generation', 'conversion', 'processing', 'renewal', 'transfer', 'payment'),
                   'update_on_trigger')]


def merge_workflow(label: str) -> str:
    if not label.startswith('other:'):
        return label
    low = label.lower()
    for keys, wf in OTHER_WORKFLOW:
        if any(k in low for k in keys):
            return wf
    return label


def strip_suffixes(trigger: str) -> tuple[str, list[str]]:
    t, found = trigger.strip(), []
    changed = True
    while changed:
        changed = False
        for s in SUFFIXES:
            if t.endswith(s):
                t, changed = t[: -len(s)].rstrip(), True
                found.append(s)
    return t, found[::-1]


def fit(labels_path: str, out: str):
    from m5.automationbench.check import count_records
    from m5.automationbench.prompt import TRAP_TEXT, WORKFLOW_TEXT
    from m5.automationbench.upstream import DOMAINS, load
    labels = {x['task']: x for x in (json.loads(l) for l in Path(labels_path).read_text().splitlines() if l.strip())}
    wf_all, trap_all = collections.Counter(), collections.Counter()
    for l in labels.values():
        l['workflow'] = merge_workflow(l['workflow'])
        wf_all[l['workflow']] += 1
        trap_all.update(set(l['traps']))
    keep_wf = [w for w in WORKFLOW_TEXT if wf_all[w] > 1]
    keep_tr = [t for t in TRAP_TEXT if trap_all[t] > 1]
    res = {'domains': {}, 'apps_all': [], 'labels_total': {'workflow': dict(wf_all), 'traps': dict(trap_all)},
           'kept': {'workflow': keep_wf, 'traps': keep_tr}, 'suffixes': SUFFIXES}
    apps_all = collections.Counter()
    for d in DOMAINS:
        rows = load(d, noise=False)
        wf, tr, ntr, tg, docs = (collections.Counter() for _ in range(5))
        apps, napps, recs, words, suf = collections.Counter(), collections.Counter(), [], [], collections.Counter()
        bpr = []
        drive_extra = gmail_extra = 0
        for r in rows:
            l = labels.get(r['task_name'])
            if l:
                if l['workflow'] in keep_wf:
                    wf[l['workflow']] += 1
                ts = [t for t in set(l['traps']) if t in keep_tr]
                tr.update(ts)
                ntr[min(max(len(ts), 1), 5)] += 1
                tg[min(max(int(l.get('n_targets') or 1), 1), 15)] += 1
                docs[min(max(int(l.get('n_policy_docs') or 0), 0), 5)] += 1
            st = r['info']['initial_state']
            a = sorted(k for k in st if k != 'meta')
            apps.update(a)
            napps[min(max(len(a), 1), 6)] += 1
            recs.append(count_records(st))
            bpr.append(len(json.dumps({k: v for k, v in st.items() if k != 'meta'})) / max(1, count_records(st)))
            body, found = strip_suffixes(r['trigger'])
            words.append(len(body.split()))
            suf['+'.join(str(SUFFIXES.index(s)) for s in found) or 'none'] += 1
            tools = set(r['info'].get('zapier_tools') or [])
            drive_extra += 'google_drive' not in st and any(t.startswith('google_drive') for t in tools)
            gmail_extra += 'gmail' not in st and any(t.startswith('gmail') for t in tools)
        apps_all.update(apps)
        n = len(rows)
        res['domains'][d] = {
            'n': n, 'workflow': dict(wf), 'traps': dict(tr), 'n_traps': dict(ntr), 'n_targets': dict(tg),
            'n_docs': dict(docs), 'bytes_per_record': round(sorted(bpr)[len(bpr) // 2]), 'apps': dict(apps), 'n_apps': dict(napps), 'records_pre_noise': sorted(recs),
            'trigger_words': sorted(words), 'suffix': dict(suf), 'drive_unseeded_rate': drive_extra / n,
            'gmail_unseeded_rate': gmail_extra / n}
    res['apps_all'] = sorted(apps_all)
    Path(out).write_text(json.dumps(res, indent=1))
    print(f'{len(res["apps_all"])} apps; workflows kept {keep_wf}; traps kept {keep_tr}')
    for d, x in res['domains'].items():
        print(d, x['workflow'], dict(collections.Counter(x['traps']).most_common(5)), x['suffix'])


def _wchoice(rng, weights: dict):
    ks = list(weights)
    return rng.choices(ks, [weights[k] for k in ks])[0]


def _sample_apps(rng, D, apps_all, n):
    w = {a: D['apps'].get(a, 0) + 0.3 for a in apps_all}
    out = []
    for _ in range(n):
        a = _wchoice(rng, {k: v for k, v in w.items() if k not in out})
        out.append(a)
    return sorted(out)


def sample(fit_path, leaves_dir, pools_dir, n, seed, start, out):
    F = json.loads(Path(fit_path).read_text())
    from m5.automationbench.upstream import DOMAINS
    leaves = collections.defaultdict(list)
    blocked_f = Path(leaves_dir) / 'blocked.txt'   # leafcheck.py: leaves resembling a single test task
    blocked = set(blocked_f.read_text().split()) if blocked_f.exists() else set()
    for f in Path(leaves_dir).glob('*.jsonl'):
        d, w = f.stem.split('.')
        leaves[(d, w)] = [x for x in (json.loads(l) for l in f.read_text().splitlines() if l.strip())
                          if f'{d}.{w}.{x["slug"]}' not in blocked]
    pools = Path(pools_dir)
    companies = [json.loads(x) for f in sorted((pools / 'companies').glob('*.jsonl')) for x in f.read_text().splitlines() if x.strip()]
    seen = set()
    companies = [c for c in companies if not (c['name'].lower() in seen or seen.add(c['name'].lower()))]
    firsts = sorted({json.loads(x)['name'] for f in (pools / 'people').glob('first_*.jsonl') for x in f.read_text().splitlines() if x.strip()})
    lasts = sorted({json.loads(x)['name'] for f in (pools / 'people').glob('surnames_*.jsonl') for x in f.read_text().splitlines() if x.strip()})
    rng = random.Random(seed)
    kw, kt = F['kept']['workflow'], F['kept']['traps']
    t0 = dt.datetime(2025, 6, 1, tzinfo=dt.timezone.utc)
    with open(out, 'w') as fo:
        for i in range(start, start + n):
            d = DOMAINS[i % len(DOMAINS)]   # equal shares
            D = F['domains'][d]
            wf = _wchoice(rng, {w: D['workflow'].get(w, 0) + 0.5 for w in kw})
            pool = leaves.get((d, wf), [])
            leaf = None if (rng.random() < OPEN_DRAW or not pool) else rng.choice(pool)
            n_apps = int(_wchoice(rng, D['n_apps']))
            apps = _sample_apps(rng, D, F['apps_all'], n_apps)
            k = int(_wchoice(rng, D['n_traps']))
            tw = {t: D['traps'].get(t, 0) + 0.5 for t in kt}
            traps = []
            for _ in range(k):
                traps.append(_wchoice(rng, {t: v for t, v in tw.items() if t not in traps}))
            suf = _wchoice(rng, D['suffix'])
            suffix = ' '.join(F['suffixes'][int(x)] for x in suf.split('+')) if suf != 'none' else ''
            if 'Gmail' in suffix and 'gmail' not in apps:
                suffix = suffix.replace(F['suffixes'][4], '').strip()
            tools = []
            if 'google_drive' not in apps and rng.random() < D['drive_unseeded_rate']:
                tools.append('google_drive_find_multiple_files')
            if 'gmail' not in apps and rng.random() < D['gmail_unseeded_rate']:
                tools.append('gmail_send_email')
            now = t0 + dt.timedelta(days=rng.randrange(0, 730), hours=rng.choice(range(7, 18)),
                                    minutes=rng.choice([0, 0, 15, 30, 45]))
            if now.weekday() >= 5:
                now += dt.timedelta(days=7 - now.weekday())
            people = set()
            while len(people) < 14:
                people.add(f'{rng.choice(firsts)} {rng.choice(lasts)}')
            spec = {'id': f'ab{i:06d}', 'example_id': 10_000_000 + i, 'seed': seed * 1_000_003 + i, 'domain': d,
                    'workflow': wf, 'leaf': leaf, 'leaf_key': f'{d}.{wf}.{leaf["slug"]}' if leaf else None,
                    'apps': apps, 'traps': traps,
                    'records': max(4, round(rng.choice(D['records_pre_noise']) * rng.uniform(0.85, 1.15))),
                    'trigger_words': max(12, round(rng.choice(D['trigger_words']) * rng.uniform(0.85, 1.15))),
                    'targets': int(_wchoice(rng, D['n_targets'])), 'docs': int(_wchoice(rng, D['n_docs'])),
                    'suffix': suffix, 'zapier_tools': tools, 'now': now.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'companies': rng.sample(companies, 3), 'people': sorted(people)}
            spec['bytes_per_record'] = D['bytes_per_record']   # deterministic: no draw, same specs as before
            fo.write(json.dumps(spec) + '\n')
    print(f'{n} specs -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fit', 'sample'])
    ap.add_argument('--labels')
    ap.add_argument('--fit')
    ap.add_argument('--leaves')
    ap.add_argument('--pools')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--seed', type=int, default=359)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.cmd == 'fit':
        fit(a.labels, a.out)
    else:
        sample(a.fit, a.leaves, a.pools, a.n, a.seed, a.start, a.out)


if __name__ == '__main__':
    main()
