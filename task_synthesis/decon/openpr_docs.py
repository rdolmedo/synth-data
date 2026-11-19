import json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decon_extract_lib import doc, S
P=f'{S}/prheads'
out=[]
rows=json.load(open(f'{S}/data/tbs_task_prs.json'))
for r in rows:
    if r['state']!='open': continue
    g=f'{P}/terminal-bench-science.git'; ref=f"refs/pr/{r['number']}"
    tree_out=__import__('subprocess').run(['git','-C',g,'ls-tree','-r','--name-only',ref,'tasks'],capture_output=True,text=True).stdout.split()
    dirs=sorted(set('/'.join(p.split('/')[:-1]) for p in tree_out if p.endswith('/task.toml') and p.split('/')[-2] in r['new']))
    for d in dirs:
        x=doc(f"openref:tbsci#{r['number']}:{d.split('/')[-1]}",'openref','tbsci-open',g,ref,d,r['author'])
        if x: out.append(x)
# TB repo open PRs are already extracted as candidates with tier Q open
for x in json.load(open(f'{S}/data/decon_docs.json')):
    if x['role']=='cand' and x.get('tier')=='Q open (may enter test set)':
        x=dict(x); x['role']='openref'; x['src']='tb-open'; x['uid']='openref:'+x['uid'][5:]; out.append(x)
json.dump(out, open(f'{S}/data/openref_docs.json','w'))
print(len(out), __import__('collections').Counter(x['src'] for x in out))
