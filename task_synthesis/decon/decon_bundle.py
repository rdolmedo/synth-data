import json, os, re, random
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs={d['uid']:d for d in json.load(open(f'{S}/data/decon_docs.json'))}
flag=[x for x in json.load(open(f'{S}/data/decon_flagged.json')) if x['tier'][0] in 'ABC']
os.makedirs(f'{S}/decon_bundles',exist_ok=True)
def cut(s,n): s=s or ''; return s if len(s)<=n else s[:n]+f'\n...[truncated {len(s)-n} chars]'
def side(label,d):
    files='\n'.join(f'{p}\t{size}' for p,sha,size in d['files'][:60])
    return (f'## {label}\n### File listing (first 60)\n{files}\n### Instruction\n{cut(d["instr"],5000)}\n### Environment (Dockerfile/compose)\n{cut(d["docker"],2000)}\n'
            f'### Tests (excerpt)\n{cut(d["tests"],4500)}\n### Reference solution (excerpt)\n{cut(d["sol"],3500)}\n### Other environment files (excerpt)\n{cut(d["env"],1500)}\n')
key=[]
random.Random(3).shuffle(flag)
for i,x in enumerate(flag):
    c,r=docs[x['cand']],docs[x['ref']]
    sig={k:x[k] for k in ['e_instr','c_instr','c_tests','c_sol','c_docker','c_env','shared_blobs','same_author']}
    txt=(f'# Pair P{i:03d}\nAutomated overlap signals (containment 0-1; e_instr = instruction embedding cosine; same_author = a shared contributor identity): {json.dumps(sig)}\n\n'
         + side('TASK A (candidate training seed)', c) + '\n' + side('TASK B (held-out evaluation task)', r))
    open(f'{S}/decon_bundles/P{i:03d}.md','w').write(txt)
    key.append(dict(id=f'P{i:03d}',cand=x['cand'],ref=x['ref'],ref_family=x['ref_family'],tier=x['tier'],reasons=x['reasons'],**sig))
json.dump(key,open(f'{S}/data/decon_pair_key.json','w'),indent=1)
print(len(key)); os.system(f'du -sh {S}/decon_bundles')
