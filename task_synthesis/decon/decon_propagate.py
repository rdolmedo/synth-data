# ~/mumwelt/.venv/bin/python -I ; propagate exclusions to near-copies among candidates
import json, re, os, sys, collections, zlib
import numpy as np
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
excluded=set(json.load(open(f'{S}/data/decon_excluded_uids.json')))
docs=[d for d in json.load(open(f'{S}/data/decon_docs.json'))]
Ei=np.load(f'{S}/data/decon_emb_instr.npy'); idx={d['uid']:i for i,d in enumerate(docs)}
cands=[d for d in docs if d['role']=='cand']
W=re.compile(r'[a-z0-9_]+'); C=re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]')
def sh(t,k): return {zlib.crc32(' '.join(t[i:i+k]).encode()) for i in range(max(0,len(t)-k+1))}
F={d['uid']:{'instr':sh(W.findall(d['instr'].lower()),5),'tests':sh([x.lower() for x in C.findall(d['tests'])],10),'sol':sh([x.lower() for x in C.findall(d['sol'])],10)} for d in cands}
fam=lambda u: u.split(':')[-1].lower()
for k in ['instr','tests','sol']:
    cnt=collections.Counter()
    for u,f in F.items():
        for x in f[k]: cnt[x]+=1
    common={x for x,v in cnt.items() if v>5}
    for f in F.values(): f[k]-=common
def cont(a,b): return len(a&b)/min(len(a),len(b)) if a and b and min(len(a),len(b))>=25 else 0
new=set(excluded); frontier=set(excluded); why={}
while frontier:
    nxt=set()
    for e in frontier:
        if e not in F: continue
        for c in cands:
            u=c['uid']
            if u in new: continue
            ct,cs,ci=cont(F[e]['tests'],F[u]['tests']),cont(F[e]['sol'],F[u]['sol']),cont(F[e]['instr'],F[u]['instr'])
            ei=float(Ei[idx[e]]@Ei[idx[u]])
            if ct>=0.5 or cs>=0.5 or ci>=0.5 or ei>=0.93:
                nxt.add(u); why[u]=dict(via=e,c_tests=round(ct,2),c_sol=round(cs,2),c_instr=round(ci,2),e_instr=round(ei,3))
    new|=nxt; frontier=nxt
json.dump(why, open(f'{S}/data/decon_propagated.json','w'), indent=1)
print('propagated', len(why))
for u,w in why.items(): print(' ',u,'<-',w)
