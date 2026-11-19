# run with ~/mumwelt/.venv/bin/python -I (needs numpy + fastembed)
import json, re, os, sys, collections, random, zlib
import numpy as np
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs=json.load(open(f'{S}/data/decon_docs.json'))
refs=[d for d in docs if d['role']=='ref']; cands=[d for d in docs if d['role']=='cand']
fam=lambda d: d['uid'].split(':')[-1].split('.')[0].lower()
print(len(refs), len(cands))
# ---- boilerplate blobs
bf=collections.defaultdict(set)
for d in docs:
    for p,sha,size in d['files']: bf[sha].add(fam(d))
def blobs(d): return {sha:size for p,sha,size in d['files'] if size>=300 and len(bf[sha])<=3}
# ---- shingles
W=re.compile(r'[a-z0-9_]+'); C=re.compile(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]')
def sh(tokens,k): return {zlib.crc32(' '.join(tokens[i:i+k]).encode()) for i in range(max(0,len(tokens)-k+1))}
def feats(d):
    return {'instr':sh(W.findall(d['instr'].lower()),5),'tests':sh([t.lower() for t in C.findall(d['tests'])],10),
            'sol':sh([t.lower() for t in C.findall(d['sol'])],10),'docker':sh([t.lower() for t in C.findall(d['docker'])],8),
            'env':sh([t.lower() for t in C.findall(d['env'])],10)}
F={d['uid']:feats(d) for d in docs}
B={d['uid']:blobs(d) for d in docs}
KEYS=['instr','tests','sol','docker','env']
# shingle document frequency over all docs (by family) -> drop very common shingles
dfc={k:collections.Counter() for k in KEYS}
famsets={k:collections.defaultdict(set) for k in KEYS}
for d in docs:
    for k in KEYS:
        for s in F[d['uid']][k]: famsets[k][s].add(fam(d))
common={k:{s for s,v in famsets[k].items() if len(v)>8} for k in KEYS}
for d in docs:
    for k in KEYS: F[d['uid']][k]-=common[k]
# ---- embeddings
from fastembed import TextEmbedding
def emb(model, texts):
    m=TextEmbedding(model); X=np.array(list(m.embed(texts, batch_size=16)),dtype=np.float32)
    return X/np.linalg.norm(X,axis=1,keepdims=True).clip(1e-9)
import time
def emb_cached(name, model, texts, bs):
    path=f'{S}/data/decon_emb_{name}.npy'
    if os.path.exists(path): return np.load(path)
    m=TextEmbedding(model); out=[]; t0=time.time()
    for i in range(0,len(texts),64):
        out+=list(m.embed(texts[i:i+64], batch_size=bs))
        print(f'{name}: {len(out)}/{len(texts)} {time.time()-t0:.0f}s', flush=True)
    X=np.array(out,dtype=np.float32); X/=np.linalg.norm(X,axis=1,keepdims=True).clip(1e-9); np.save(path,X); return X
Ei=emb_cached('instr','BAAI/bge-small-en-v1.5',[d['instr'][:3000] or d['uid'] for d in docs],16)
Ec=Ei  # code embeddings dropped: lexical/file-hash code checks cover reuse
idx={d['uid']:i for i,d in enumerate(docs)}
idfam=collections.defaultdict(set)
for d in docs:
    for a in d['authors']: idfam[a].add(fam(d))
for d in docs: d['authors_strong']=[a for a in d['authors'] if len(idfam[a])<=15]
def jac(a,b):
    if not a or not b: return 0.0,0.0
    i=len(a&b); return i/len(a|b), (i/min(len(a),len(b)) if min(len(a),len(b))>=25 else 0.0)
def slug(u): return set(re.split(r'[-_]+',u.split(':')[-1].lower()))
def pair(c,r):
    out={}
    for k in KEYS:
        j,cn=jac(F[c['uid']][k],F[r['uid']][k]); out['j_'+k]=round(j,3); out['c_'+k]=round(cn,3)
    sb=set(B[c['uid']])&set(B[r['uid']]); out['shared_blobs']=len(sb); out['shared_bytes']=sum(B[c['uid']][s] for s in sb)
    out['e_instr']=round(float(Ei[idx[c['uid']]]@Ei[idx[r['uid']]]),3); 
    a1,a2=set(c['authors']),set(r['authors']); out['same_org']=bool(a1&a2); out['same_author']=bool(set(c['authors_strong'])&set(r['authors_strong'])); out['author_overlap']=sorted(a1&a2)[:3]
    s1,s2=slug(c['uid']),slug(r['uid']); out['slug_j']=round(len(s1&s2)/len(s1|s2),2)
    return out
# ---- calibration: positives = closed/open PR versions of tasks later accepted vs final accepted version
finals={fam(r):r for r in refs if r['src'] in('tb2.0','tb3+','tb1-core')}
pos=[]
recs={(r['repo'],r['pr'],r['task']):r for r in json.load(open(f'{S}/data/records_joined.json'))}
for r in refs:
    if r['src']!='history': continue
    repo_pr,task=r['uid'][5:].rsplit(':',1); repo,pr=repo_pr.split('#')
    st=recs.get((repo,int(pr),task),{}).get('state')
    if st in('closed',) and fam(r) in finals: pos.append((r,finals[fam(r)]))
rng=random.Random(0); fl=list(finals.values()); neg=[]
while len(neg)<3000:
    a,b=rng.sample(fl,2)
    if fam(a)!=fam(b): neg.append((a,b))
cal={'pos':[dict(a=a['uid'],b=b['uid'],**pair(a,b)) for a,b in pos],'neg':[dict(a=a['uid'],b=b['uid'],**pair(a,b)) for a,b in neg]}
json.dump(cal,open(f'{S}/data/decon_calibration.json','w'))
print('calibration pos',len(cal['pos']),'neg',len(cal['neg']))
# ---- candidates x refs
inv={k:collections.defaultdict(list) for k in KEYS}
for j,r in enumerate(refs):
    for k in KEYS:
        for s in F[r['uid']][k]: inv[k][s].append(j)
RI=np.array([idx[r['uid']] for r in refs])
pairs=[]
for c in cands:
    cand_refs=set()
    for k in KEYS:
        cnt=collections.Counter(j for s in F[c['uid']][k] for j in inv[k].get(s,()))
        cand_refs|={j for j,v in cnt.items() if v>=3}
    ei=Ei[RI]@Ei[idx[c['uid']]]; ec=Ec[RI]@Ec[idx[c['uid']]]
    cand_refs|=set(np.argsort(-ei)[:5].tolist())
    ca=set(c['authors_strong'])
    cand_refs|={j for j,r in enumerate(refs) if ca & set(r['authors_strong'])}
    cb=set(B[c['uid']])
    cand_refs|={j for j,r in enumerate(refs) if cb & set(B[r['uid']])}
    for j in cand_refs: pairs.append(dict(cand=c['uid'],ref=refs[j]['uid'],ref_src=refs[j]['src'],tier=c.get('tier'),**pair(c,refs[j])))
json.dump(pairs,open(f'{S}/data/decon_pairs.json','w'))
print('pairs',len(pairs))
