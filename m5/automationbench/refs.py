"""Reference resolution in a validated world: every `<x>_id` / `<x>_ids` field that names a collection of the same
app must point at an existing record of it (problem-side check, #359).

Field -> collection is inferred by name (account_id -> accounts, channel_id -> channels, owner/assignee/... -> users),
restricted to the pairs that upstream's own scored tasks resolve (FK_FIELDS, fitted with `python -m
m5.automationbench.refs fit`): a pair upstream leaves dangling is not a rule of the simulator.
"""
from __future__ import annotations

import collections
import json
import sys

PERSON = ('owner', 'assignee', 'requester', 'author', 'creator', 'sender', 'responder', 'agent', 'manager',
          'reporter', 'host', 'organizer', 'submitter', 'approver', 'recruiter')


def collections_of(app: dict) -> dict[str, list[dict]]:
    return {k: v for k, v in app.items() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)}


def singular(coll: str) -> str:
    return coll[:-3] + 'y' if coll.endswith('ies') else coll[:-2] if coll.endswith('sses') else coll[:-1]


def ids_of(recs: list[dict], coll: str = '') -> set[str]:
    out = set()
    pk = singular(coll) + '_id'
    for r in recs:
        for k in ('id', 'row_id', 'key', 'ts', 'email', 'name', pk):
            if r.get(k) not in (None, ''):
                out.add(str(r[k]))
    return out


def targets(field: str, colls: dict) -> list[str]:
    base = field[:-4] if field.endswith('_ids') else field[:-3]
    last = base.split('_')[-1]
    cands = [base + 's', base + 'es', base[:-1] + 'ies', last + 's', last + 'es', last[:-1] + 'ies']
    if any(p in base for p in PERSON):
        cands += ['users', 'agents', 'members', 'employees', 'people']   # owners are never contacts
    return [c for c in dict.fromkeys(cands) if c in colls]


def walk(obj, path=()):
    """(path, field, value) of every *_id / *_ids field below obj."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and (k.endswith('_id') or k.endswith('_ids')) and v not in (None, '', []):
                yield path, k, v
            if isinstance(v, (dict, list)):
                yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for x in obj:
            yield from walk(x, path)


def check(world_dump: dict, skip: set | None = None) -> list[str]:
    """Dangling references in a world (WorldState.model_dump(mode='json')). skip: {(app, collection, field)} that
    upstream itself leaves dangling (SKIP_FIELDS); every other inferable reference is enforced."""
    bad = []
    for app, st in world_dump.items():
        if app == 'meta' or not isinstance(st, dict):
            continue
        colls = collections_of(st)
        known = {c: ids_of(v, c) for c, v in colls.items()}
        for path, field, v in walk(st):
            coll = path[0] if path else ''
            if len(path) == 1 and field == singular(coll) + '_id':
                continue   # the record's own key (xero contact_id in contacts)
            ts = targets(field, colls)
            if not ts or (skip is not None and (app, coll, field) in skip):
                continue
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                if isinstance(x, (str, int)) and not any(str(x) in known[t] for t in ts):
                    bad.append(f'{app}.{coll}.{field}={x!r} not in {app}.{"/".join(ts)}')
    return bad


def fit(out_path):
    """Per (app, collection, field): how often upstream's scored tasks resolve it. Enforce the ones >= 98%."""
    from automationbench.runner import strip_none_values
    from automationbench.schema.world import WorldState
    from m5.automationbench.upstream import scored
    tot, ok = collections.Counter(), collections.Counter()
    for r in scored(noise=False):
        w = WorldState(**strip_none_values(r['info']['initial_state'])).model_dump(mode='json')
        for app, st in w.items():
            if app == 'meta' or not isinstance(st, dict):
                continue
            colls = collections_of(st)
            known = {c: ids_of(v, c) for c, v in colls.items()}
            for path, field, v in walk(st):
                coll = path[0] if path else ''
                if len(path) == 1 and field == singular(coll) + '_id':
                    continue
                ts = targets(field, colls)
                if not ts:
                    continue
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, (str, int)):
                        key = (app, coll, field)
                        tot[key] += 1
                        ok[key] += any(str(x) in known[t] for t in ts)
    rows = sorted(({'app': k[0], 'collection': k[1], 'field': k[2], 'n': tot[k], 'resolved': ok[k] / tot[k]}
                   for k in tot), key=lambda r: -r['n'])
    for r in rows:
        print(f'{r["app"]:16s} {r["collection"]:22s} {r["field"]:22s} n={r["n"]:5d} resolved={r["resolved"]:.3f}')
    skip = [r for r in rows if r['resolved'] < 0.98 and r['n'] >= 3]
    json.dump(skip, open(out_path, 'w'), indent=1)
    print(len(skip), 'skipped (upstream leaves them dangling)')


def load_skip(path) -> set:
    return {(r['app'], r['collection'], r['field']) for r in json.load(open(path))}


if __name__ == '__main__':
    fit(sys.argv[2])
