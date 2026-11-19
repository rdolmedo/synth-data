"""Production GPU-hours by generation-token share (#359). Super: our output tokens per engine / that engine's
generation-token rate (snapshots); engines without two snapshots use the mean rate of the others. DeepSeek: our
completion tokens (generator + judge) / m5-ds's rate in the window, x 4 GPUs."""
import collections, glob, gzip, json, sys
A = '/data/aai/m5/automationbench'; P = A + '/production'
cfg = {}
for d in glob.glob(A + '/pi_runs/m5_automationbench/prod-*'):
    try: cfg[d.rsplit('/', 1)[1]] = json.load(open(d + '/config.json'))['model'].split(':')[-1]
    except Exception: pass
out = collections.Counter(); n = collections.Counter()
for f in sorted(glob.glob(A + '/export/nemotron35-super/attempts/*.gz')):
    for line in gzip.open(f, 'rt'):
        x = json.loads(line); e = cfg[x['m5']['run']]; n[e] += 1
        for s in x['sessions']:
            for l in s['text'].splitlines():
                try: m = json.loads(l).get('message') or {}
                except ValueError: continue
                if m.get('role') == 'assistant': out[e] += int((m.get('usage') or {}).get('output') or 0)
snaps = [json.loads(l) for l in open(P + '/snaps.jsonl')]
by = collections.defaultdict(list)
for s in snaps: by[(s['name'], s['url'])].append(s)
rate = {}
for (name, url), v in by.items():
    v.sort(key=lambda s: s['t'])
    if len(v) >= 2 and v[-1]['generation_tokens'] > v[0]['generation_tokens']:
        rate[name] = ((v[-1]['generation_tokens'] - v[0]['generation_tokens']) / ((v[-1]['t'] - v[0]['t']) / 3600),
                      (v[-1]['t'] - v[0]['t']) / 3600, v[-1]['generation_tokens'] - v[0]['generation_tokens'])
sup = {k: r for k, r in rate.items() if k.startswith('m5-super')}
mean_rate = sum(r[0] for r in sup.values()) / len(sup)
R = {'super': {}, 'ds': {}}
tot = 0
for e in sorted(out):
    r = sup.get(e, (mean_rate,))[0]
    gh = out[e] / r; tot += gh
    R['super'][e] = {'attempts': n[e], 'our_output_tokens': out[e], 'engine_gen_tokens_per_h': round(r),
                     'rate_from': 'own snapshots' if e in sup else 'mean of snapshotted engines', 'gpu_hours': round(gh, 2)}
R['super']['total_gpu_hours'] = round(tot, 1); R['super']['our_output_tokens'] = sum(out.values())
gen = sum(u['completion'] for l in open(P + '/gen/accepted.jsonl') for u in (json.loads(l).get('usage') or []))
ds = rate['m5-ds']
R['ds'] = {'gen_completion_tokens': gen, 'engine_gen_tokens_per_h': round(ds[0]),
           'window_h': round(ds[1], 2), 'gpus': 4, 'gpu_hours_gen': round(4 * gen / ds[0], 1),
           'gpu_hours_judge_upper_bound': round(4 * 29.5 / 60, 1)}   # verdicts carry no usage: 29.5 min of the 4-GPU engine
json.dump(R, open(P + '/gpu_hours.json.tmp', 'w'), indent=1)
import os; os.replace(P + '/gpu_hours.json.tmp', P + '/gpu_hours.json')
print(json.dumps(R, indent=1))
