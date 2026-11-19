"""GitHub: repositories (code, projects, services, git history) and single files via code search (configs, logs,
data files, documents). Licenses come from GitHub's own detection (repository spdx_id); repositories without a
recognized license are excluded by the policy. Versions are commit SHAs; downloads go through codeload/raw at that
commit, so every asset is pinned.

Rate limits (one token): repository search 30/min, code search 10/min, core 5,000/h. The Http layer paces them.
"""
import json
import random
import shutil
import subprocess
from pathlib import Path

from ..http import TooLarge
from .base import MAX_FILES, Reject, Source, cap, count_files, git, unpack

API = 'https://api.github.com'
LANG = {'Python': 'Python', 'C': 'C', 'C++': 'C++', 'CUDA': 'Cuda', 'TypeScript': 'TypeScript', 'Rust': 'Rust',
        'JavaScript': 'JavaScript', 'PHP': 'PHP', 'Java': 'Java', 'Redcode': 'Redcode', 'OCaml': 'OCaml',
        'Perl': 'Perl', 'Lean': 'Lean', 'Shell': 'Shell', 'Go': 'Go', 'Ruby': 'Ruby', 'Haskell': 'Haskell',
        'Fortran': 'Fortran', 'COBOL': 'COBOL', 'R': 'R', 'Julia': 'Julia', 'Lua': 'Lua', 'Scala': 'Scala',
        'Kotlin': 'Kotlin', 'Elixir': 'Elixir', 'Erlang': 'Erlang', 'Coq': 'Coq', 'Prolog': 'Prolog',
        'Scheme': 'Scheme', 'Racket': 'Racket', 'Assembly': 'Assembly', 'Zig': 'Zig', 'Nim': 'Nim', 'Swift': 'Swift',
        'C#': 'C#', 'Verilog': 'Verilog', 'VHDL': 'VHDL', 'Ada': 'Ada', 'Pascal': 'Pascal', 'Tcl': 'Tcl',
        'Awk': 'Awk', 'SQL': 'PLpgSQL', 'Dart': 'Dart', 'Clojure': 'Clojure', 'F#': 'F#', 'Guile': 'Scheme',
        'Common Lisp': 'Common Lisp', 'Smalltalk': 'Smalltalk', 'Forth': 'Forth', 'Groovy': 'Groovy'}
# repository size (KB) by code scale
SIZE_KB = {'snippet': (5, 3000), 'mini-package': (20, 8000), 'real-project': (300, 40000), 'none': (5, 20000)}


def license_of(repo):
    lic = repo.get('license') or {}
    sid = lic.get('spdx_id')
    return None if sid in (None, 'NOASSERTION') else sid


def repo_cand(item, query):
    return {'source': 'github-repo', 'key': item['full_name'], 'url': item['html_url'], 'page_url': item['html_url'],
            'title': item['full_name'], 'description': (item.get('description') or '')[:400],
            'license_raw': license_of(item) or '', 'size': item.get('size', 0) * 1024,
            'formats': [], 'meta': {'stars': item.get('stargazers_count'), 'language': item.get('language'),
                                    'topics': item.get('topics', [])[:10], 'default_branch': item.get('default_branch'),
                                    'pushed_at': item.get('pushed_at')}, 'query': query}


class GithubRepos(Source):
    name = 'github-repo'
    classes = ('python-code', 'c-cpp-code', 'other-code', 'git-repo', 'running-service', 'binary', 'document',
               'config', 'upstream-project')

    async def search(self, target, query, n):
        lo, hi = SIZE_KB.get(target.get('scale') or 'none', SIZE_KB['none'])
        if target['cls'] == 'git-repo':
            lo, hi = 20, 30000
        q = [query, 'fork:false', 'archived:false', f'size:{lo}..{hi}']
        lang = target.get('language')
        if lang:
            q.append(f'language:"{LANG.get(lang, lang)}"')
        sort = random.choice(['stars', 'updated', 'best-match', 'best-match'])
        params = {'q': ' '.join(q), 'per_page': min(100, max(10, n * 3))}
        if sort != 'best-match':
            params['sort'] = sort
        j = await self.http.json(f'{API}/search/repositories', params=params, bucket='search')
        items = [it for it in j.get('items', []) if not it.get('fork') and not it.get('disabled')]
        return [repo_cand(it, query) for it in items][: n * 3]

    async def head_commit(self, full, branch):
        j = await self.http.json(f'{API}/repos/{full}/commits/{branch}')
        return j['sha'], j['commit']['committer']['date']

    async def fetch(self, cand, staging, target):
        full, branch = cand['key'], cand['meta'].get('default_branch') or 'HEAD'
        if target['cls'] == 'git-repo':
            return await self.fetch_git(cand, staging, target)
        sha, date = await self.head_commit(full, branch)
        tmp = Path(staging).with_name(Path(staging).name + '.tgz')
        try:
            digest, nbytes = await self.http.download(f'https://codeload.github.com/{full}/tar.gz/{sha}', tmp, cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        try:
            unpack(tmp, staging)
        finally:
            tmp.unlink(missing_ok=True)
        if count_files(staging) > MAX_FILES:
            raise Reject(f'{count_files(staging)} files')
        owner = full.split('/')[0]
        return {'version': sha, 'version_date': date, 'sha256': digest, 'download_bytes': nbytes,
                'url': f'https://github.com/{full}/tree/{sha}',
                'attribution': f'{full} by {owner} (GitHub), commit {sha[:12]}, {cand["license_raw"]}'}

    async def fetch_git(self, cand, staging, target):
        """Full history (git-repo tasks need it). Bounded by size and commit count."""
        full = cand['key']
        staging = Path(staging)
        loop_git = lambda *a, **k: git(*a, **k)  # noqa: E731
        import asyncio
        try:
            await asyncio.to_thread(loop_git, 'clone', '--no-tags', '--quiet', f'https://github.com/{full}.git',
                                    str(staging), timeout=900)
        except subprocess.TimeoutExpired:
            raise Reject('clone timed out')
        n_commits = int(git('rev-list', '--count', 'HEAD', cwd=staging).strip())
        if n_commits > 3000:
            raise Reject(f'{n_commits} commits')
        size = sum(p.stat().st_size for p in staging.rglob('*') if p.is_file())
        if size > cap(target):
            raise Reject(f'{size} bytes with history')
        sha = git('rev-parse', 'HEAD', cwd=staging).strip()
        branches = git('branch', '-a', cwd=staging).split('\n')
        first, last = git('log', '--reverse', '--format=%cI', cwd=staging).split('\n')[0], \
            git('log', '-1', '--format=%cI', cwd=staging).strip()
        authors = len(set(git('log', '--format=%an', cwd=staging).split('\n')))
        git('remote', 'remove', 'origin', cwd=staging)
        from ..scrub import history_has_secret, pseudonymize_git
        if history_has_secret(staging):
            raise Reject('a secret pattern appears in the git history')
        pseudonymize_git(staging)
        owner = full.split('/')[0]
        return {'version': sha, 'url': f'https://github.com/{full}/tree/{sha}', 'sha256': sha, 'download_bytes': size,
                'git': f'{n_commits} commits from {first[:10]} to {last[:10]}, {authors} authors, '
                       f'{len([b for b in branches if b.strip()])} branches',
                'attribution': f'{full} by {owner} (GitHub), history to commit {sha[:12]}, {cand["license_raw"]}',
                'scrub_in_tree': False}


# code search: what a format means as file qualifiers (any one of them per query)
FILE_QUALIFIERS = {
    'nginx': ['filename:nginx.conf', 'path:nginx extension:conf'], 'supervisor': ['filename:supervisord.conf',
                                                                                  'path:supervisor extension:conf'],
    'cron': ['filename:crontab'], 'systemd': ['extension:service', 'extension:timer'],
    'ssh': ['filename:sshd_config', 'filename:ssh_config'], 'ini-conf': ['extension:ini', 'extension:cfg',
                                                                       'extension:conf'],
    'yaml': ['extension:yaml', 'extension:yml'], 'json': ['extension:json'], 'jsonl': ['extension:jsonl'],
    'xml': ['extension:xml'], 'toml': ['extension:toml'], 'rdf': ['extension:ttl', 'extension:rdf', 'extension:nt'],
    'build-config': ['filename:CMakeLists.txt', 'filename:Makefile', 'filename:meson.build', 'filename:pom.xml',
                     'filename:build.gradle'],
    'log': ['extension:log'], 'gcode': ['extension:gcode'], 'calendar-ics': ['extension:ics'],
    'word-list': ['extension:dic', 'path:wordlist extension:txt', 'path:words extension:txt'],
    'hex-dump': ['extension:hex', 'extension:ihex'], 'markdown': ['extension:md'], 'latex': ['extension:tex'],
    'html': ['extension:html'], 'sql-dump': ['extension:sql'], 'csv': ['extension:csv'], 'tsv': ['extension:tsv'],
    'robot-model': ['extension:urdf', 'extension:xacro'], 'ontology': ['extension:obo', 'extension:owl'],
    'molecule': ['extension:sdf', 'extension:mol2', 'extension:mol'], 'cif': ['extension:cif'],
    'pdb': ['extension:pdb'], 'fasta': ['extension:fasta', 'extension:fa', 'extension:fna'],
    'genbank': ['extension:gb', 'extension:gbk'], 'spectrum': ['extension:jdx', 'extension:dx', 'extension:msp',
                                                               'extension:mgf'],
    'svg': ['extension:svg'], 'plain-text': ['extension:txt'], 'ppm': ['extension:ppm', 'extension:pgm'],
    'redcode': ['extension:red'], 'wal-journal': ['extension:sql'],
}
CLASS_QUALIFIERS = {'database': ['extension:sql'], 'config': ['extension:conf', 'extension:ini', 'extension:yaml'],
                    'structured-data': ['extension:json', 'extension:yaml', 'extension:xml'],
                    'text-data': ['extension:txt', 'extension:log'], 'document': ['extension:md'],
                    'tabular-data': ['extension:csv'], 'scientific-data': ['extension:csv', 'extension:txt']}


class GithubFiles(Source):
    """One file found by code search, pinned to the repository's head commit. Code search only indexes text files
    under ~384 KB, so this serves text formats."""
    name = 'github-file'
    classes = ('config', 'structured-data', 'text-data', 'document', 'scientific-data', 'tabular-data', 'database',
               'media')

    def __init__(self, http):
        super().__init__(http)
        self.repo_cache = {}

    async def repo(self, full):
        if full not in self.repo_cache:
            self.repo_cache[full] = await self.http.json(f'{API}/repos/{full}')
        return self.repo_cache[full]

    async def search(self, target, query, n):
        fmts = [f for f in target.get('formats') or [] if f in FILE_QUALIFIERS]
        quals = FILE_QUALIFIERS[random.choice(fmts)] if fmts else CLASS_QUALIFIERS.get(target['cls'])
        if not quals:
            raise Reject(f'no file qualifier for {target["cls"]} formats {target.get("formats")}')
        qual = random.choice(quals)
        params = {'q': f'{query} {qual}', 'per_page': min(100, max(10, n * 4))}
        j = await self.http.json(f'{API}/search/code', params=params, bucket='code_search')
        out = []
        for it in j.get('items', []):
            full = it['repository']['full_name']
            out.append({'source': self.name, 'key': f'{full}:{it["path"]}', 'url': it['html_url'],
                        'page_url': it['html_url'], 'title': f'{it["path"]} in {full}',
                        'description': (it['repository'].get('description') or '')[:300], 'license_raw': None,
                        'size': None, 'formats': [], 'meta': {'repo': full, 'path': it['path'], 'blob': it['sha']},
                        'query': f'{query} {qual}'})
        return out[: n * 4]

    async def resolve_license(self, cand):
        r = await self.repo(cand['meta']['repo'])
        cand['license_raw'] = license_of(r)
        cand['meta']['default_branch'] = r.get('default_branch')
        cand['meta']['stars'] = r.get('stargazers_count')
        return cand

    async def fetch(self, cand, staging, target):
        full, path = cand['meta']['repo'], cand['meta']['path']
        if cand.get('license_raw') is None:
            await self.resolve_license(cand)
        branch = cand['meta'].get('default_branch') or 'HEAD'
        j = await self.http.json(f'{API}/repos/{full}/commits', params={'path': path, 'sha': branch, 'per_page': 1})
        if not j:
            raise Reject('file not on the default branch')
        sha = j[0]['sha']
        dest = Path(staging) / Path(path).name
        try:
            digest, nbytes = await self.http.download(
                f'https://raw.githubusercontent.com/{full}/{sha}/{path}', dest, cap(target))
        except TooLarge as e:
            raise Reject(str(e))
        if nbytes < 200:
            raise Reject(f'file has only {nbytes} bytes')
        return {'version': sha, 'sha256': digest, 'download_bytes': nbytes,
                'url': f'https://github.com/{full}/blob/{sha}/{path}',
                'attribution': f'{path} from {full} (GitHub), commit {sha[:12]}, {cand["license_raw"]}'}
