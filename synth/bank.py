"""Trajectory bank: one growing corpus per teacher and dataset, fed by every campaign on it (user, 2026-10-25).

Campaigns keep their exports where they wrote them; the bank only indexes them. `deposit` reads a campaign's
train manifest and adds its usable, finalized trajectories with the campaign, the serving recipe it ran (teacher
tunables, admission, Pi bundle, commit), the verifier reward, and whether the verifier failed for infrastructure
reasons (proxy or download errors in its output: such a reward says nothing about the trajectory), and whether the
attempt was cut before it finished (`truncated`: tail cut, closed training; kept as data, but not counted toward a
task's target). A deposit replaces that campaign's earlier one, so depositing a running campaign again picks up its
new trajectories.

    <bank>/members/<campaign>.json   that campaign's records and recipe
    <bank>/manifest.json             union of the members in the train-manifest format: a campaign on the same
                                     teacher and dataset lists it in train.baseline_manifests and only tops each
                                     task up to its target
    <bank>/bank.json                 teacher, dataset, members, totals

    python -m synth.bank deposit /data/aai/tbench/synth/bank <campaign root>
    python -m synth.bank summary /data/aai/tbench/synth/bank/<teacher>/<dataset>
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import glob
import json
from pathlib import Path
import re
import time

from . import archive
from .common import atomic, read

INFRA = re.compile(r'port 8080|Connection reset by peer|Failed to connect|Could not resolve (proxy|host)|'
                   r'uvx: command not found|Temporary failure in name resolution')


def dataset_of(campaign):
    """Dataset name from the train bundle (…/datasets/<name>/…), else the bundle's file name, else (a campaign fed only
    by a task stream) the stream directory's name."""
    source = campaign['train'].get('bundle_source') or (campaign['train'].get('stream') or {}).get('dir') or ''
    m = re.search(r'/datasets/([^/]+)/', source)
    return m.group(1) if m else Path(source).stem or 'unknown'


def verdict(raw):
    """(reward, verifier_infra_failure, truncated) from an attempt's raw directory (synth.archive.verdict: the same
    rule as the archive lines; it used to take the last trial's reward and skip harbor/trial/<trial>/)."""
    return archive.verdict(raw)


def recipe(root, campaign, manifest):
    s = campaign['teacher'].get('serving', {})
    protocol = read(Path(root) / 'ops' / 'protocol.json', {})
    progress = read(Path(root) / 'ops' / 'progress.json', {})
    return {'campaign': campaign['id'], 'teacher': campaign['teacher']['slug'],
            'revision': campaign['teacher'].get('revision'), 'commit': progress.get('worktree_commit'),
            'tunables': s.get('tunables'), 'admission': campaign.get('admission'),
            'pi_bundle_sha256': (protocol.get('pi') or {}).get('bundle_sha256'),
            'trajectories_per_task': manifest.get('trajectories_per_task')}


def deposit(bank_root, root, workers=32):
    root = Path(root)
    campaign = read(root / 'ops' / 'campaign.json', None)
    manifest = read(root / 'train' / 'manifest.json', None)
    if not campaign or not manifest:
        raise SystemExit(f'{root}: no ops/campaign.json or train/manifest.json')
    teacher, dataset = campaign['teacher']['slug'], dataset_of(campaign)
    bank = Path(bank_root) / teacher / dataset
    info = read(bank / 'bank.json', {})
    if info and (info.get('teacher'), info.get('dataset')) != (teacher, dataset):
        raise SystemExit(f'{bank}: holds {info.get("teacher")}/{info.get("dataset")}, not {teacher}/{dataset}')
    records = [r for r in manifest.get('records', []) if r.get('usable') and r.get('finalized')]
    out = archive.folder(root, 'train')

    def pointer(index, uid):
        """(reward, infra, truncated, archive pointer) from the archive index once packed (synth/archive.py)."""
        number, line, reward, infra, truncated = index['attempts'][uid]
        shard = index['shards'][str(number)]
        return reward, infra, truncated, {'lines': str(out / shard['lines']), 'line': line, 'zip': str(out / shard['zip'])}

    def judged(raw):
        if not raw.is_dir():
            raise FileNotFoundError(f'{raw}: no attempt directory')
        return (*verdict(raw), None)
    # Consistent while the campaign is packed (#236): the index, then the other attempts' directories, then the index
    # again (archive.read_dirs); an attempt packed in between comes from its index entry.
    first = archive.load_index(root)
    verdicts, missing = {}, {}
    for r in records:
        if r['id'] in first['attempts']:
            verdicts[r['id']] = pointer(first, r['id'])
    rest = [r['id'] for r in records if r['id'] not in verdicts]
    for uid, value, error, index in archive.read_dirs(root, 'train', rest, judged, workers, chunk=max(1, len(rest))):
        if index is not None:
            verdicts[uid] = pointer(index, uid)
        elif error:
            missing[uid] = error
        else:
            verdicts[uid] = value
    if missing:   # a usable, finalized record in neither the archive nor its directory: never deposited silently
        raise archive.Incomplete(root, missing, len(records))
    verdicts = [verdicts[r['id']] for r in records]
    member = {'recipe': recipe(root, campaign, manifest), 'deposited_at': time.time(),
              'records': [{**r, 'campaign': campaign['id'], 'reward': v[0], 'verifier_infra_failure': v[1],
                           'truncated': v[2], **({'archive': v[3]} if v[3] else {})} for r, v in zip(records, verdicts)]}
    atomic(bank / 'members' / f'{campaign["id"]}.json', member)
    return rebuild(bank, teacher, dataset)


def rebuild(bank, teacher, dataset):
    members = {Path(p).stem: read(p, {}) for p in sorted(glob.glob(f'{bank}/members/*.json'))}
    seen, records = set(), []
    for m in members.values():
        for r in m.get('records', []):
            if r['sha256'] not in seen:
                seen.add(r['sha256'])
                records.append(r)
    per_task = collections.Counter(r['task'] for r in records if not r.get('truncated'))
    atomic(bank / 'manifest.json', {'kind': 'bank', 'teacher': teacher, 'dataset': dataset, 'records': records,
                                    'trajectory_count': len(records), 'updated_at': time.time()})
    summary = {'teacher': teacher, 'dataset': dataset, 'updated_at': time.time(), 'trajectories': len(records),
               'tokens': sum(int(r.get('tokens') or 0) for r in records), 'tasks': len(per_task),
               'tasks_at_8': sum(1 for n in per_task.values() if n >= 8),   # complete trajectories only
               'truncated': sum(1 for r in records if r.get('truncated')),
               'reward_1': sum(1 for r in records if r.get('reward') == 1.0),
               'verifier_infra_failures': sum(1 for r in records if r.get('verifier_infra_failure')),
               'members': {k: {'trajectories': len(m.get('records', [])), 'recipe': m.get('recipe')}
                           for k, m in members.items()}}
    atomic(bank / 'bank.json', summary)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('deposit')
    d.add_argument('bank_root')
    d.add_argument('campaign_roots', nargs='+')
    s = sub.add_parser('summary')
    s.add_argument('bank')
    a = ap.parse_args()
    if a.cmd == 'deposit':
        for root in a.campaign_roots:
            out = deposit(a.bank_root, root)
            print(json.dumps({k: v for k, v in out.items() if k != 'members'}))
    else:
        out = read(Path(a.bank) / 'bank.json', {})
        print(json.dumps({k: v for k, v in out.items() if k != 'members'}, indent=1))
        for k, m in out.get('members', {}).items():
            print(k, m['trajectories'], json.dumps((m.get('recipe') or {}).get('tunables')))
