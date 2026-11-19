"""CPU-only acquisition of public runtime packages in a benchmark's native image.

No model, reference, verifier or solution is run. APT downloads only; requested
packages remain absent until the agent explicitly installs them offline.
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import shlex
import tarfile
import tempfile
import urllib.request
import zipfile
import subprocess

from .runtime_cache import REMOTE, sha256

IMAGE_PROBE = 'cat /etc/os-release; printf "\\nAA_DPKG_STATUS_SHA256="; if test -f /var/lib/dpkg/status; then sha256sum /var/lib/dpkg/status | cut -d" " -f1; else echo none; fi'


async def build(a):
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import TrialPaths
    from aa_evals.common.harbor_mswea.ns_env import NsSidecarEnvironment
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    scratch = Path(os.environ.get('_CONDOR_SCRATCH_DIR') or tempfile.gettempdir()) / ('runtime-cache-' + a.task)
    taskdir = scratch / a.task
    taskdir.mkdir(parents=True)
    with tarfile.open(a.bundle) as tf:
        # Only public agent environment/instruction/config. Never stage a reference/verifier.
        prefix = 'tasks/' + a.task + '/'
        for m in tf.getmembers():
            if not m.name.startswith(prefix) or not m.isfile(): continue
            rel = Path(m.name[len(prefix):])
            if rel.parts[0] not in ('environment', 'instruction.md', 'task.toml'): continue
            if '..' in rel.parts or rel.is_absolute(): raise ValueError('unsafe public task path')
            p = taskdir / rel; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(tf.extractfile(m).read())
    task = Task(taskdir, disable_verification=True)
    trial = TrialPaths(scratch / 'trial')
    for name in ('agent', 'verifier', 'artifacts'): (trial.trial_dir / name).mkdir(parents=True)
    env = NsSidecarEnvironment(environment_dir=taskdir/'environment', environment_name=a.task,
            session_id='runtime-cache-build', trial_paths=trial, task_env_config=task.config.environment)
    commands = []
    async def run(command, timeout=1800):
        result = await env.exec(command, user='root', timeout_sec=timeout)
        commands.append({'command':command,'return_code':result.return_code,'stdout':result.stdout,'stderr':result.stderr})
        (out/'acquisition.json').write_text(json.dumps(commands, indent=2)+'\n')
        if result.return_code: raise RuntimeError(f'package acquisition failed: {command[:140]}: {result.stderr[-2000:]}')
        return result.stdout
    try:
        await env.start(force_build=False)
        probe = await run(IMAGE_PROBE)
        packages = [x for x in a.packages.split(',') if x]
        pip_packages = [x for x in a.pip.split(',') if x]
        cran_packages = [x for x in a.cran.split(',') if x]
        sources = json.loads(Path(a.sources_json).read_text()) if a.sources_json else []
        managers = []
        requested = []
        source_repositories = {}
        await run('mkdir -p /tmp/aa-runtime-payload')
        if packages:
            managers.append('apt'); requested += packages
            if any(not re.fullmatch(r'[a-z0-9][a-z0-9+.-]*(?:=[A-Za-z0-9:.+~_-]+)?', x) for x in packages):
                raise ValueError('invalid APT package selector')
            source_repositories['apt'] = await run('cat /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true')
            await run('mkdir -p /tmp/aa-runtime-download/partial /tmp/aa-runtime-payload/apt/pool; chmod -R a+rX /tmp/aa-runtime-download /tmp/aa-runtime-payload')
            await run('DEBIAN_FRONTEND=noninteractive apt-get update')
            selector = ' '.join(map(shlex.quote, packages))
            await run('DEBIAN_FRONTEND=noninteractive apt-get --yes --print-uris --download-only --reinstall --no-install-recommends -o Dir::Cache::archives=/tmp/aa-runtime-download install ' + selector)
            await run('DEBIAN_FRONTEND=noninteractive apt-get --yes --download-only --reinstall --no-install-recommends -o Dir::Cache::archives=/tmp/aa-runtime-download install ' + selector)
            await run('cp /tmp/aa-runtime-download/*.deb /tmp/aa-runtime-payload/apt/pool/; cd /tmp/aa-runtime-payload/apt; '
                      'for f in pool/*.deb; do dpkg-deb -f "$f"; '
                      'printf "Filename: %s\\nSize: %s\\nSHA256: %s\\n\\n" "$f" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d" " -f1)"; '
                      'done > Packages; gzip -n -c Packages > Packages.gz; '
                      'printf "Origin: AA public runtime cache\\nLabel: Public runtime packages\\nArchitectures: amd64 all\\nSHA256:\\n" > Release; '
                      'for f in Packages Packages.gz; do printf " %s %s %s\\n" "$(sha256sum "$f" | cut -d" " -f1)" "$(stat -c%s "$f")" "$f"; done >> Release; '
                      'printf "%s\\n" ' + shlex.quote('deb [trusted=yes] file:' + REMOTE + '/apt ./') + ' > sources.list; '
                      'printf "%s\\n" ' + ' '.join(map(shlex.quote, ['Dir::Etc::sourcelist "' + REMOTE + '/apt/sources.list";', 'Dir::Etc::sourceparts "-";', 'Acquire::Languages "none";']))
                      + ' > apt.conf; chmod -R a+rX /tmp/aa-runtime-payload')
        if pip_packages:
            managers.append('pip'); requested += pip_packages
            if any(not re.fullmatch(r'[A-Za-z0-9_.-]+(?:==[A-Za-z0-9_.+!-]+)?', x) for x in pip_packages):
                raise ValueError('invalid Python package selector')
            source_repositories['pip'] = 'https://pypi.org/simple'
            await run('mkdir -p /tmp/aa-runtime-payload/wheels; python3 -m pip download --no-cache-dir '
                      + ('--only-binary=:all: ' if a.binary_only else '')
                      + '--dest /tmp/aa-runtime-payload/wheels ' + ' '.join(map(shlex.quote,pip_packages)))
        if cran_packages:
            managers.append('cran'); requested += cran_packages
            source_repositories['cran'] = 'https://cloud.r-project.org'
            selectors = [x.split('==') for x in cran_packages]
            if any(len(x)!=2 or not re.fullmatch(r'[A-Za-z][A-Za-z0-9.]*',x[0]) or not re.fullmatch(r'[0-9.\-]+',x[1]) for x in selectors):
                raise ValueError('CRAN selectors require package==version')
            names = 'c('+','.join(json.dumps(x[0])for x in selectors)+')'
            versions = 'c('+','.join(json.dumps(x[1])for x in selectors)+')'
            code = ('repo<-"https://cloud.r-project.org"; ap<-available.packages(repos=repo); '
                    + 'targets<-'+names+'; versions<-'+versions+'; stopifnot(all(ap[targets,"Version"]==versions)); '
                    'deps<-unique(c(targets,unlist(tools::package_dependencies(targets,ap,which=c("Depends","Imports","LinkingTo"),recursive=TRUE)))); '
                    'nonavail<-setdiff(deps,rownames(ap)); stopifnot(all(nonavail %in% c("R",rownames(installed.packages())))); '
                    'deps<-intersect(deps,rownames(ap)); dir.create("/tmp/aa-runtime-payload/cran/src/contrib",recursive=TRUE); '
                    'download.packages(deps,destdir="/tmp/aa-runtime-payload/cran/src/contrib",repos=repo,type="source"); '
                    'tools::write_PACKAGES("/tmp/aa-runtime-payload/cran/src/contrib",type="source"); '
                    "writeLines('options(repos=c(CRAN=\"file:///opt/aa-runtime-cache/cran\"))',\"/tmp/aa-runtime-payload/cran/Rprofile\")")
            await run('Rscript -e '+shlex.quote(code))
        if not managers and not sources:
            raise ValueError('no reviewed public package acquisition selected')
        after = await run(IMAGE_PROBE)
        if probe != after: raise RuntimeError('download-only acquisition modified installed package inventory')
        await env.download_dir('/tmp/aa-runtime-payload', out/'payload')
        files = {str(p.relative_to(out/'payload')):sha256(p) for p in sorted((out/'payload').rglob('*')) if p.is_file()}
        package_info = await run('for f in /tmp/aa-runtime-payload/apt/pool/*.deb; do test -f "$f" || continue; dpkg-deb -f "$f" Package Version Architecture; done') if packages else ''
        upstream_sources = []
        if sources:
            managers.append('source')
            for spec in sources:
                filename = spec['filename']
                if not re.fullmatch(r'[A-Za-z0-9_.+-]+',filename) or not spec['url'].startswith('https://'):
                    raise ValueError('invalid public source archive descriptor')
                dst = out/'payload/sources'/filename; dst.parent.mkdir(parents=True,exist_ok=True)
                if spec.get('kind') == 'git':
                    command=['git','clone','--bare','--depth','1','--single-branch','--branch',spec['ref'],spec['url'],str(dst)]
                    git_env={**os.environ,'GIT_CONFIG_GLOBAL':'/dev/null','GIT_CONFIG_NOSYSTEM':'1','GIT_TERMINAL_PROMPT':'0'}
                    subprocess.run(command,env=git_env,check=True,timeout=600)
                    commit=subprocess.check_output(['git','-C',str(dst),'rev-parse','HEAD'],text=True).strip()
                    upstream_sources.append({**spec,'commit_sha':commit,'classification':'unmodified public tagged source and Git clone metadata; no built task artifacts'})
                    requested.append(spec['name']+'=='+spec['version'])
                    continue
                with urllib.request.urlopen(spec['url'],timeout=180) as response, dst.open('wb') as stream:
                    while block := response.read(1<<20):stream.write(block)
                if filename.lower().endswith('.zip') and not zipfile.is_zipfile(dst):
                    raise ValueError('upstream source response is not a ZIP archive')
                if any(filename.lower().endswith(ext) for ext in ('.tar.gz','.tar.xz','.tar.bz2','.tar')) and not tarfile.is_tarfile(dst):
                    raise ValueError('upstream source response is not a tar archive')
                upstream_sources.append({**spec,'sha256':sha256(dst),'bytes':dst.stat().st_size})
                requested.append(spec['name']+'=='+spec['version'])
            files = {str(p.relative_to(out/'payload')):sha256(p) for p in sorted((out/'payload').rglob('*')) if p.is_file()}
        files = {str(p.relative_to(out/'payload')):sha256(p) for p in sorted((out/'payload').rglob('*')) if p.is_file()}
        entry = {'managers':managers,'requested':list(dict.fromkeys(requested)),'payload_root':str(out/'payload'),
                 'instruction_sha256':sha256(taskdir/'instruction.md'),'image_probe':probe,
                 'docker_image':task.config.environment.docker_image,'native_rootfs_key':env._main.rootfs.parent.name,
                 'files':files,'directories':[str(p.relative_to(out/'payload')) for p in sorted((out/'payload').rglob('*')) if p.is_dir()],'package_inventory':package_info,
                 'requested_by_manager':{'apt':packages,'pip':pip_packages,'cran':cran_packages,'source':[x['name']+'=='+x['version'] for x in sources]},'source_repositories':source_repositories,
                 'upstream_sources':upstream_sources,
                 'classification':'unmodified public runtime/build packages, source archives and reduced repository metadata; no benchmark reference/test/solution/data; no requested software installed during setup'}
        manifest = {'schema_version':1,'cache_version':a.version,'tasks':{a.task:entry}}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps({'manifest':str(out/'manifest.json'),'sha256':sha256(out/'manifest.json'),'files':len(files),'unchanged_installed_inventory':True}),flush=True)
    finally:
        await env.stop(delete=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--task',required=True)
    p.add_argument('--packages',default='');p.add_argument('--pip',default='');p.add_argument('--cran',default='');p.add_argument('--sources-json',default='');p.add_argument('--binary-only',action='store_true');p.add_argument('--out',required=True);p.add_argument('--version',default='v1')
    asyncio.run(build(p.parse_args()))

if __name__=='__main__':main()
