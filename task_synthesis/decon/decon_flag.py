import json, collections, os
S=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pairs=json.load(open(f'{S}/data/decon_pairs.json'))
cal=json.load(open(f'{S}/data/decon_calibration.json'))
def reasons(x):
    r=[]
    if x['c_tests']>=0.15: r.append('tests')
    if x['c_sol']>=0.15: r.append('solution')
    if x['c_env']>=0.30: r.append('env-files')
    if x['c_docker']>=0.50: r.append('dockerfile')
    if x['c_instr']>=0.10 or x['j_instr']>=0.05: r.append('instr-text')
    if x['shared_blobs']>=1: r.append('shared-files')
    if x['e_instr']>=0.80: r.append('instr-emb')
    if x['same_author'] and x['e_instr']>=0.70: r.append('same-author+emb')
    if x['slug_j']>=0.5 and x['e_instr']>=0.65: r.append('name+emb')
    return r
print('positives caught', sum(bool(reasons(x)) for x in cal['pos']),'/',len(cal['pos']), ' random pairs flagged', sum(bool(reasons(x)) for x in cal['neg']),'/',len(cal['neg']))
fam=lambda u: u.split(':')[-1].split('.')[0].lower()
best={}
for x in pairs:
    rs=reasons(x)
    if not rs: continue
    k=(x['cand'], x['ref_src'] if x['ref_src'] in('ot-tblite','tb-science') else 'tb', fam(x['ref']))
    score=max(x['c_tests'],x['c_sol'],x['c_env'],x['c_instr'],x['e_instr']-0.5, min(1,x['shared_blobs']))
    if k not in best or score>best[k][0]: best[k]=(score,dict(x,reasons=rs,ref_family=k[1]+':'+k[2]))
flag=[v[1] for v in best.values()]
print('flagged (cand, ref-family) pairs', len(flag), 'unique cands', len({x['cand'] for x in flag}))
print('by ref family', collections.Counter(x['ref_family'].split(':')[0] for x in flag))
print('by reason', collections.Counter(r for x in flag for r in x['reasons']).most_common())
print('by cand tier', collections.Counter(x['tier'] for x in {x['cand']:x for x in flag}.values()).most_common())
json.dump(flag, open(f'{S}/data/decon_flagged.json','w'))
