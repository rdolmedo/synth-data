"""synth.bank: campaigns deposit into one corpus per teacher and dataset; its manifest drives top-ups."""
import json
from pathlib import Path
import tempfile
import unittest

from synth import bank
from synth.controller import baseline_counts


def campaign(root, cid, teacher, tasks, proxy_fail=(), cut=()):
    """A finished campaign on TMax: one usable trajectory per (task, attempt) plus one unusable record."""
    root = Path(root) / cid
    (root / 'ops').mkdir(parents=True)
    (root / 'ops' / 'campaign.json').write_text(json.dumps({
        'id': cid, 'teacher': {'slug': teacher, 'revision': 'r', 'serving': {'tunables': {'max_seqs': 768}}},
        'train': {'bundle_source': '/data/x/datasets/tmax/tmax-15k-harbor.tar'}, 'admission': {'mode': 'kv_fit'}}))
    (root / 'ops' / 'protocol.json').write_text(json.dumps({'pi': {'bundle_sha256': 'abc'}}))
    records = []
    for i, task in enumerate(tasks):
        raw = root / 'train' / 'raw' / f'{cid}-{task}-a{i}'
        verifier = raw / 'harbor' / f'{task}__x' / 'verifier'
        verifier.mkdir(parents=True)
        (verifier / 'reward.txt').write_text('0' if task in proxy_fail else '1')
        (verifier / 'test-stdout.txt').write_text('curl: (7) Failed to connect to 10.22.0.107 port 8080'
                                                  if task in proxy_fail else 'PASSED')
        (raw / 'worker-result.json').write_text(json.dumps({'interrupted': task in cut}))
        records.append({'id': f'{cid}-{task}-a{i}', 'task': task, 'usable': True, 'finalized': True,
                        'tokens': 100, 'sha256': f'{cid}{i}', 'raw': str(raw), 'path': str(raw) + '.json.gz'})
    records.append({'id': 'bad', 'task': tasks[0], 'usable': False, 'finalized': True, 'sha256': 'bad', 'raw': '/x'})
    (root / 'train').mkdir(exist_ok=True)
    (root / 'train' / 'manifest.json').write_text(json.dumps({'trajectories_per_task': 8, 'records': records}))
    return root


class Bank(unittest.TestCase):
    def test_deposits_union_feed_top_ups_and_flag_broken_verifiers(self):
        with tempfile.TemporaryDirectory() as d:
            a = campaign(d, 'c1', 'deepseek', ['t1', 't1', 't2'], proxy_fail={'t2'})
            b = campaign(d, 'c2', 'deepseek', ['t1', 't3'])
            out = bank.deposit(Path(d) / 'bank', a)
            self.assertEqual((out['dataset'], out['trajectories'], out['verifier_infra_failures']), ('tmax', 3, 1))
            bank.deposit(Path(d) / 'bank', a)                  # a re-deposit replaces, never doubles
            out = bank.deposit(Path(d) / 'bank', b)
            self.assertEqual((out['trajectories'], out['reward_1'], sorted(out['members'])), (5, 4, ['c1', 'c2']))
            counts = baseline_counts([Path(d) / 'bank' / 'deepseek' / 'tmax' / 'manifest.json'])
            self.assertEqual(dict(counts), {'t1': 3, 't2': 1, 't3': 1})
            cut = campaign(d, 'c4', 'deepseek', ['t4', 't5'], cut={'t5'})   # a tail-cut attempt
            out = bank.deposit(Path(d) / 'bank', cut)
            self.assertEqual((out['trajectories'], out['truncated']), (7, 1))
            counts = baseline_counts([Path(d) / 'bank' / 'deepseek' / 'tmax' / 'manifest.json'])
            self.assertEqual((counts['t4'], counts['t5']), (1, 0), 'a truncated trajectory does not count toward 8')
            other = campaign(d, 'c3', 'super', ['t1'])
            (Path(d) / 'bank' / 'super').mkdir()
            out = bank.deposit(Path(d) / 'bank', other)        # another teacher gets its own bank
            self.assertEqual(out['teacher'], 'super')
            self.assertEqual(json.loads((Path(d) / 'bank' / 'deepseek' / 'tmax' / 'bank.json').read_text())['trajectories'], 7)


if __name__ == '__main__':
    unittest.main()
