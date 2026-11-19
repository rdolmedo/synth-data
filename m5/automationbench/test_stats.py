"""Class-level statistics of AutomationBench's 600 scored tasks (and the 200 simple ones), for the taxonomy fit.

  /opt/aai/venvs/automationbench/bin/python -m m5.automationbench.test_stats OUT.json

Only aggregate numbers leave this script (per-domain distributions); test items never reach the generator.
"""
import collections
import json
import statistics
import sys

from automationbench.domains import get_domain_dataset
from automationbench.runner import strip_none_values

DOMAINS = ['sales', 'marketing', 'operations', 'support', 'finance', 'hr']


def count_records(state):
    n = 0
    for app, v in state.items():
        if app == 'meta' or not isinstance(v, dict):
            continue
        for coll, x in v.items():
            if isinstance(x, list):
                n += len(x)
                if app == 'google_sheets' and coll == 'spreadsheets':
                    for ss in x:
                        for ws in ss.get('worksheets', []):
                            n += len(ws.get('rows', []))
    return n


def rows(domain, noise=True):
    from m5.automationbench.upstream import load
    out = []
    for r in load(domain, noise):
        info = r['info']
        st = strip_none_values(info.get('initial_state', {}))
        user = [m['content'] for m in r['prompt'] if m['role'] == 'user'][0]
        apps = sorted(k for k in st if k != 'meta')
        out.append({'task': info['task_name'], 'apps': apps, 'n_apps': len(apps),
                    'state_bytes': len(json.dumps(st)), 'trigger_words': len(user.split()),
                    'n_records': count_records(st), 'n_tools': len(info.get('zapier_tools', [])),
                    'assertion_types': [a['type'] for a in info.get('assertions', [])],
                    'has_meta_time': 'meta' in st})
    return out


def q(xs):
    xs = sorted(xs)
    return {'p10': xs[len(xs) // 10], 'median': statistics.median(xs), 'p90': xs[9 * len(xs) // 10], 'max': xs[-1],
            'mean': round(statistics.mean(xs), 1)}


def main(out):
    res = {}
    for d in DOMAINS + ['simple']:
        rs = rows(d)
        pre = rows(d, noise=False)
        apps = collections.Counter(a for r in rs for a in r['apps'])
        nap = collections.Counter(r['n_apps'] for r in rs)
        res[d] = {'n': len(rs), 'apps': dict(apps.most_common()), 'n_apps_hist': dict(sorted(nap.items())),
                  'state_bytes': q([r['state_bytes'] for r in rs]), 'trigger_words': q([r['trigger_words'] for r in rs]),
                  'n_records': q([r['n_records'] for r in rs]), 'n_tools': q([r['n_tools'] for r in rs]),
                  'assertion_types': dict(collections.Counter(t for r in rs for t in r['assertion_types']).most_common(40)),
                  'state_bytes_pre_noise': q([r['state_bytes'] for r in pre]),
                  'n_records_pre_noise': q([r['n_records'] for r in pre]),
                  'app_sets': dict(collections.Counter('+'.join(r['apps']) for r in rs).most_common(15))}
    json.dump(res, open(out, 'w'), indent=1)
    allapps = collections.Counter()
    for d in DOMAINS:
        allapps.update(res[d]['apps'])
    print(len(allapps), 'apps in scored tasks:', dict(allapps.most_common()))
    for d in DOMAINS + ['simple']:
        r = res[d]
        print(d, 'bytes', r['state_bytes'], 'pre', r['state_bytes_pre_noise'], 'recs', r['n_records'], 'pre', r['n_records_pre_noise'])


if __name__ == '__main__':
    main(sys.argv[1])
