import json, csv, collections, os
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows=json.load(open(f'{S}/data/seed_candidates.json'))
docs=[d for d in json.load(open(f'{S}/data/decon_docs.json')) if d['role']=='cand']
op=json.load(open(f'{S}/data/decon_opus.json')); son={r['id']:r for r in json.load(open(f'{S}/data/decon_sonnet.json'))}
prop=json.load(open(f'{S}/data/decon_propagated.json'))
import glob
bkey={k['id']:k for k in json.load(open(f'{S}/data/decon_band_key.json'))}
bandx={}
for f in glob.glob(f'{S}/decon_band_ratings/opus_band_*.json'):
    for r in json.load(open(f)):
        if r['decision']=='EXCLUDE': bandx[bkey[r['id']]['cand']]=(r,bkey[r['id']]['ref'])
resub={}
if os.path.exists(f'{S}/data/openpr_final.json'):
    for x in json.load(open(f'{S}/data/openpr_final.json')): resub[x['seed']]=x
flag=json.load(open(f'{S}/data/decon_flagged.json'))
uid_of={}
for d in docs:
    k=(d['uid'].split(':')[1], d['uid'].split(':')[-1])
    uid_of[k]=d['uid']
def uid(r): return f"cand:{r['repo']}#{r['pr']}:{r['task']}" if r['pr'] else f"cand:tb1-extra:{r['task']}"
opx={son[i]['cand']:(r,son[i]) for i,r in op.items() if r['decision']=='EXCLUDE'}
nearest=collections.defaultdict(list)
for x in flag: nearest[x['cand']].append(x)
for r in rows:
    u=uid(r); st,why,ref='clean','',''
    if r['tier'].startswith('X'): st='junk'
    elif r['tier']=='Q overlaps OT-TB-Lite': st,why='contaminated','in OT-TB-Lite'
    elif r['tier']=='Q near-dup of test task': st,why,ref='contaminated','near-duplicate instruction of test task',r['most_similar_test_task']
    elif r['tier']=='Q was merged into a benchmark': st,why='contaminated','was merged into TB repo'
    elif u in opx: o,s=opx[u]; st,why,ref='contaminated',f"Opus: {o['relation']} - {o['rationale'][:160]}",s['ref']
    elif u in bandx: o,rf=bandx[u]; st,why,ref='contaminated',f"Opus (band): {o['relation']} - {o['rationale'][:160]}",rf
    elif u in prop: st,why,ref='contaminated',f"near-copy of excluded sibling {prop[u]['via'].split(':')[-1]}",prop[u]['via']
    elif r['tier']=='Q open (may enter test set)': st,why='quarantine','open PR in active TB repo; may become a test task'
    elif r['state']=='open': st,why='quarantine','open PR in frozen TB1/TB2 repo (dropped: no open PRs as seeds)'
    elif u in resub: st,why,ref='quarantine',f"resubmitted/sibling of open PR: {resub[u]['rationale'][:150]}",resub[u]['open']
    elif u in nearest: st='clean (flagged, cleared)'
    r['decon_status'],r['decon_reason'],r['decon_ref']=st,why,ref
    r['decon_max_e_instr']=max([x['e_instr'] for x in nearest.get(u,[])], default=None)
    r['usable_seed']= st.startswith('clean') and r['tier'][0] in 'AB'
c=collections.Counter((r['tier'][0], r['decon_status']) for r in rows)
for k,v in sorted(c.items()): print(v,k)
print('usable seeds', sum(r['usable_seed'] for r in rows), ' by tier', collections.Counter(r['tier'] for r in rows if r['usable_seed']))
print('contaminated total', sum(r['decon_status']=='contaminated' for r in rows), collections.Counter(r['decon_reason'].split(':')[0].split(' - ')[0][:40] for r in rows if r['decon_status']=='contaminated'))
with open(f'{S}/data/seed_candidates_decon.csv','w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(sorted(rows,key=lambda x:(not x['usable_seed'], x['tier'], x['task'])))
json.dump(rows, open(f'{S}/data/seed_candidates_decon.json','w'))
