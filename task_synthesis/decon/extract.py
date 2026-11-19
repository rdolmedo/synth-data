import json, subprocess, re, os, sys
S = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def git(repo, *a):
    r = subprocess.run(['git','-C',repo,*a], capture_output=True)
    return r.stdout.decode('utf-8','replace') if r.returncode==0 else None
def tree(repo, ref, d):
    out = git(repo,'ls-tree','-r','-l',ref,'--',d+'/')
    files = {}
    for line in (out or '').splitlines():
        meta, path = line.split('\t',1)
        _,typ,sha,size = meta.split()
        files[path[len(d)+1:]] = (sha, int(size) if size!='-' else 0)
    return files
def read(repo, files, name, maxsize=300_000):
    if name in files and files[name][1] <= maxsize:
        return git(repo,'cat-file','-p',files[name][0])
    return None
def task_record(repo, ref, d):
    f = tree(repo, ref, d)
    if not f: return None
    rec = {'n_files': len(f), 'total_bytes': sum(s for _,s in f.values()), 'paths': sorted(f)[:400]}
    instr = read(repo,f,'instruction.md')
    cfg = read(repo,f,'task.toml') or read(repo,f,'task.yaml') or read(repo,f,'task.yml')
    rec['format'] = 'harbor' if 'task.toml' in f else ('tb1' if ('task.yaml' in f or 'task.yml' in f) else 'unknown')
    if instr is None and cfg and rec['format']=='tb1':
        m = re.search(r'^(?:instruction|description)s?:\s*[|>]-?\s*\n((?:[ \t]+.*\n|\s*\n)+)', cfg, re.M)
        if m: instr = re.sub(r'^[ \t]{2,}', '', m.group(1), flags=re.M)
        else:
            m = re.search(r'description:\s*[|>]-?\s*\n((?:[ \t]+.*\n|\s*\n)+)', cfg)
            if m: instr = re.sub(r'^[ \t]{2,}', '', m.group(1), flags=re.M)
    rec['instruction'] = instr; rec['config'] = cfg
    rec['readme'] = read(repo,f,'README.md')
    tests = [p for p in f if re.search(r'(^|/)tests?/', p) or p in ('run-tests.sh',)]
    rec['has_tests'] = bool(tests)
    rec['has_solution'] = any(p.startswith('solution') or p in ('solution.sh','solution.yaml') for p in f)
    rec['has_dockerfile'] = any(p.endswith('Dockerfile') or 'docker-compose' in p for p in f)
    rec['multi_container'] = any('docker-compose' in p or 'compose.y' in p for p in f)
    tcode = ''.join(read(repo,f,p) or '' for p in tests if p.endswith(('.py','.sh')))
    rec['n_test_funcs'] = len(re.findall(r'^\s*(?:async\s+)?def test_', tcode, re.M))
    rec['test_bytes'] = len(tcode)
    sol = ''.join(read(repo,f,p) or '' for p in f if (p.startswith('solution') or p in ('solution.sh','solution.yaml')) and not p.endswith(('.png','.jpg','.bin','.gz','.zip','.tar')))
    rec['solution_bytes'] = len(sol)
    return rec
if __name__ == '__main__':
    res = json.load(open(f'{S}/data/pr_tasks.json'))
    out = []
    pat = {'terminal-bench-1': ['tasks','original-tasks'], 'terminal-bench-2': [''], 'terminal-bench': ['tasks']}
    for repo in pat:
        g = f'{S}/prheads/{repo}.git'
        for r in res[repo]:
            for t in r['new_tasks']:
                rec = None
                for base in pat[repo]:
                    d = f'{base}/{t}' if base else t
                    rec = task_record(g, f'refs/pr/{r["number"]}', d)
                    if rec: break
                row = dict(source=f'{repo}#PR', repo=repo, pr=r['number'], state=r['state'], task=t, author=r['author'], created=r['created'], title=r['title'], labels=r['labels'], pr_n_new_tasks=len(r['new_tasks']))
                row.update(rec or {'missing': True})
                out.append(row)
        print(repo, 'done', file=sys.stderr)
    json.dump(out, open(f'{S}/data/pr_task_records.json','w'))
    print(len(out), sum(1 for o in out if o.get('missing')))
