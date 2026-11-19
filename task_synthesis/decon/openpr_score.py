# ~/mumwelt/.venv/bin/python -I : usable seeds vs open PRs in live repos (resubmission check)
import json, re, os, collections, zlib
import numpy as np
from fastembed import TextEmbedding
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs=json.load(open(f'{S}/data/decon_docs.json')); idx={d['uid']:i for i,d in enumerate(docs)}
Ei=np.load(f'{S}/data/decon_emb_instr.npy')
rows=json.load(open(f'{S}/data/seed_candidates_decon.json'))
uid=lambda r: f"cand:{r['repo']}#{r['pr']}:{r['task']}" if r['pr'] else f"cand:tb1-extra:{r['task']}"
seeds=[docs[idx[uid(r)]] for r in rows if r['usable_seed'] and r['state']!='open']
opens=json.load(open(f'{S}/data/openref_docs.json'))
m=TextEmbedding('BAAI/bge-small-en-v1.5'); Eo=np.array(list(m.embed([d['instr'][:3000] or d['uid'] for d in opens],batch_size=16)),dtype=np.float32); Eo/=np.linalg.norm(Eo,axis=1,keepdims=True).clip(1e-9)
W=re.compile(r'[a-z0-9_]+'); C=re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]')
def sh(t,k): return {zlib.crc32(' '.join(t[i:i+k]).encode()) for i in range(max(0,len(t)-k+1))}
def F(d): return {'instr':sh(W.findall(d['instr'].lower()),5),'tests':sh([x.lower() for x in C.findall(d['tests'])],10),'sol':sh([x.lower() for x in C.findall(d['sol'])],10),'env':sh([x.lower() for x in C.findall(d['env'])],10)}
FS={d['uid']:F(d) for d in seeds}; FO={d['uid']:F(d) for d in opens}
for k in ['instr','tests','sol','env']:
    cnt=collections.Counter(x for f in list(FS.values())+list(FO.values()) for x in f[k]); common={x for x,v in cnt.items() if v>8}
    for f in list(FS.values())+list(FO.values()): f[k]-=common
def cont(a,b): return len(a&b)/min(len(a),len(b)) if a and b and min(len(a),len(b))>=25 else 0
blob=lambda d:{sha for p,sha,size in d['files'] if size>=300}
out=[]
for s in seeds:
    es=Ei[idx[s['uid']]]; sims=Eo@es; sa=set(s.get('authors_strong') or [a for a in s['authors'] if a not in ('scaleai','scaleai-bot','tbench@scale.com','snorkel ai','research@snorkel.ai','dpsnorkel')])
    for j,o in enumerate(opens):
        sig=dict(e_instr=round(float(sims[j]),3),**{f'c_{k}':round(cont(FS[s['uid']][k],FO[o['uid']][k]),3) for k in ['instr','tests','sol','env']},shared_blobs=len(blob(s)&blob(o)),same_author=bool(sa & set(o['authors'])))
        hit=sig['c_tests']>=0.15 or sig['c_sol']>=0.15 or sig['c_env']>=0.3 or sig['c_instr']>=0.1 or sig['shared_blobs']>0 or sig['e_instr']>=0.85 or (sig['same_author'] and sig['e_instr']>=0.72)
        if hit: out.append(dict(seed=s['uid'],open=o['uid'],**sig))
json.dump(out, open(f'{S}/data/openpr_flags.json','w'), indent=1)
print('seeds',len(seeds),'open refs',len(opens),'flagged pairs',len(out),'seeds flagged',len({x['seed'] for x in out}))
for x in sorted(out,key=lambda x:-max(x['c_tests'],x['c_sol'],x['c_instr'],x['e_instr']-0.5))[:40]: print(x)
