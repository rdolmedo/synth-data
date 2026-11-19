import json, os, re, sys, subprocess, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract import tree, git, S
R=f'{S}/repos'; P=f'{S}/prheads'
sets=json.load(open(f'{S}/data/sets.json'))
recs=json.load(open(f'{S}/data/records_joined.json'))
cands=json.load(open(f'{S}/data/seed_candidates.json'))
norm=lambda t:t.split('.')[0].lower()
TEXT_EXT=re.compile(r'\.(py|sh|bash|md|txt|toml|ya?ml|json|js|ts|tsx|jsx|c|h|cc|cpp|hpp|rs|go|java|kt|rb|php|pl|r|R|jl|lua|sql|ini|cfg|conf|csv|tsv|xml|html|css|scss|tex|lean|v|ml|hs|scala|cs|swift|m|f90|asm|s|mk|cmake|proto|graphql|env|patch|diff|dockerfile|j2|tf|hcl|nix|el|vim|awk|sed|bat|ps1|ipynb|svg|log)$', re.I)
CANARY=re.compile(r'^.*(canary|BENCHMARK DATA SHOULD NEVER).*$', re.I|re.M)
def blobtext(gdir, sha):
    r=subprocess.run(['git','-C',gdir,'cat-file','-p',sha],capture_output=True)
    if r.returncode or b'\x00' in r.stdout[:8000]: return None
    return CANARY.sub('', r.stdout.decode('utf-8','replace'))
def authors_from(cfg):
    ids=set()
    for k in ['author_name','author_email','author_github','author','email','name']:
        for m in re.finditer(rf'^\s*{k}\s*[=:]\s*(.+)$', cfg or '', re.M):
            if k=='name' and 'author' not in (cfg[max(0,m.start()-200):m.start()]).lower(): continue
            for v in re.findall(r'"([^"]+)"|\'([^\']+)\'|([^\s,\[\]"\']+@[^\s,\[\]"\']+)', m.group(1)) or [(m.group(1),'','')]:
                v=next(x for x in v if x) if any(v) else ''
                v=v.strip().strip('"\'').lower()
                if v and not re.search(r'example\.com|anonymous|unknown|your[ _-]?name|todo|terminal-bench|harbor', v): ids.add(v)
    # authors = [{ name = "...", email = "..." }]
    for m in re.finditer(r'(?:name|email|github)\s*=\s*"([^"]+)"', cfg or ''):
        v=m.group(1).lower()
        if '@' in v or ' ' in v: 
            if not re.search(r'example\.com|anonymous|unknown|your[ _-]?name', v): ids.add(v)
    return ids
def doc(uid, role, src, gdir, ref, path, login=None, extra=None):
    f=tree(gdir, ref, path)
    if not f: return None
    d=dict(uid=uid, role=role, src=src, path=path, ref=ref, gdir=gdir.replace(S+'/',''), files=[], instr='', tests='', sol='', docker='', env='', cfg='')
    for p,(sha,size) in sorted(f.items()):
        d['files'].append([p,sha,size])
        if size>200_000: continue
        base=p.split('/')[-1]
        is_text = TEXT_EXT.search(p) or base in ('Dockerfile','Makefile','run-tests.sh') or base.startswith('Dockerfile')
        if not is_text: continue
        t=blobtext(gdir, sha)
        if t is None: continue
        if p=='instruction.md': d['instr']+=t
        elif p in ('task.toml','task.yaml','task.yml'): d['cfg']+=t
        elif p.startswith('tests/') or p=='run-tests.sh': d['tests']+=f'\n# FILE {p}\n'+t
        elif p.startswith('solution') or p in('solution.sh','solution.yaml'): d['sol']+=f'\n# FILE {p}\n'+t
        elif base.startswith('Dockerfile') or 'compose' in base: d['docker']+=f'\n# FILE {p}\n'+t
        elif p=='README.md': d['env']+=''  # metadata README, skip
        else: d['env']+=f'\n# FILE {p}\n'+t
    if not d['instr'] and d['cfg']:
        m=re.search(r'^(?:instruction|description)s?:\s*[|>]-?\s*\n((?:[ \t]+.*\n|\s*\n)+)', d['cfg'], re.M)
        if m: d['instr']=re.sub(r'^[ \t]{2,}','',m.group(1),flags=re.M)
    for k in ['tests','sol','docker','env']: d[k]=d[k][:150_000]
    d['authors']=sorted(authors_from(d['cfg']) | ({login.lower()} if login else set()))
    if extra: d.update(extra)
    return d
docs=[]
# PR author lookup for accepted tasks
login_by_name={}
for r in recs:
    if r['state']=='merged': login_by_name.setdefault(r['norm'], r['author'])
core=sorted(set(map(norm, sets['tb1-core-0.1.0']+sets['tb1-core-0.1.1'])))
for t in core: docs.append(doc(f'tb1core:{t}','ref','tb1-core',f'{R}/terminal-bench-1','HEAD',f'original-tasks/{t}',login_by_name.get(t)))
for t in sets['tb2.0']: docs.append(doc(f'tb2.0:{t}','ref','tb2.0',f'{R}/terminal-bench-2','HEAD',t,login_by_name.get(t)))
for t in sets['tb2.1']:
    if t.endswith(('.md','.toml')): continue
    docs.append(doc(f'tb2.1:{t}','ref','tb2.1',f'{R}/terminal-bench-2-1','HEAD',f'tasks/{t}',login_by_name.get(t)))
head=set(sets['tb-HEAD'])
for t in sorted(set(sets['tb-v3.0.0'])|head):
    docs.append(doc(f'tb3+:{t}','ref','tb3+',f'{R}/terminal-bench','HEAD' if t in head else 'v3.0.0',f'tasks/{t}',login_by_name.get(t)))
for t in open(f'{S}/data/ottblite/names.txt').read().split():
    docs.append(doc(f'ottb:{t}','ref','ot-tblite',f'{R}/ot-tblite','HEAD',t))
# pre-acceptance versions of test-set tasks (PR heads)
for r in recs:
    if not r['in_testset']: continue
    g=f"{P}/{r['repo']}.git"; ref=f"refs/pr/{r['pr']}"
    for base in (['tasks','original-tasks'] if r['repo']=='terminal-bench-1' else (['']if r['repo']=='terminal-bench-2' else ['tasks'])):
        pth=f'{base}/{r["task"]}' if base else r['task']
        d=doc(f"hist:{r['repo']}#{r['pr']}:{r['task']}",'ref','history',g,ref,pth,r['author'])
        if d: docs.append(d); break
# TB-science main
out=git(f'{R}/terminal-bench-science','ls-tree','-r','--name-only','HEAD','tasks')
for p in sorted(set('/'.join(l.split('/')[:-1]) for l in out.split() if l.endswith('/task.toml'))):
    docs.append(doc(f"tbsci:{p.split('/')[-1]}",'ref','tb-science',f'{R}/terminal-bench-science','HEAD',p))
# candidates
for c in cands:
    if c['pr']:
        g=f"{P}/{c['repo']}.git"; ref=f"refs/pr/{c['pr']}"; d=None
        for base in (['tasks','original-tasks'] if c['repo']=='terminal-bench-1' else ([''] if c['repo']=='terminal-bench-2' else ['tasks'])):
            pth=f"{base}/{c['task']}" if base else c['task']
            d=doc(f"cand:{c['repo']}#{c['pr']}:{c['task']}",'cand',c['pool'],g,ref,pth,c['author'],{'tier':c['tier']})
            if d: break
    else:
        d=doc(f"cand:tb1-extra:{c['task']}",'cand',c['pool'],f'{R}/terminal-bench-1','HEAD',f"original-tasks/{c['task']}",login_by_name.get(norm(c['task'])),{'tier':c['tier']})
    docs.append(d)
missing=sum(1 for d in docs if d is None); docs=[d for d in docs if d]
print('docs', len(docs), 'missing', missing, collections.Counter(d['src'] for d in docs))
print('with authors', collections.Counter((d['role'], bool(d['authors'])) for d in docs))
json.dump(docs, open(f'{S}/data/decon_docs.json','w'))
