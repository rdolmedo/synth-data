"""Evaluator-only online setup of generic, pinned offline dependencies (never run by an agent).

Run with GDPval's host venv and --scratch on local storage. Public archives/npm cache only are published;
resolution projects, node_modules, tests and task text are never copied into the payload.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from aa_evals.gdpval import tasks
from aa_evals.gdpval.runtime_cache import instruction_sha, prompt_sha, inventory, sha

NODE_VERSION = '18.20.8'
NODE_SHA256 = '5467ee62d6af1411d46b6a10e3fb5cacc92734dbcef465fea14e7b90993001c9'
NODE_URL = f'https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-linux-x64.tar.xz'
TERRAFORM_VERSION = '1.10.5'
AWS_PROVIDER_VERSION = '5.84.0'
GROUPS = {
 '7de33b48-5163-4f50-b5f3-8deea8185e57': [
  'react@18.3.1','react-dom@18.3.1','typescript@5.7.3','@testing-library/react@16.1.0',
  '@testing-library/dom@10.4.0','sinon@19.0.2','@types/react@18.3.18','@types/react-dom@18.3.5',
  '@types/sinon@17.0.3','tsx@4.19.2','jsdom@25.0.1'],
 '0e386e32-df20-4d1f-b536-7159bc409ad5': [
  'react@18.3.1','react-dom@18.3.1','typescript@5.7.3','ethers@6.13.5','solc@0.8.28',
  '@types/react@18.3.18','@types/react-dom@18.3.5'],
 '4122f866-01fa-400b-904d-fa171cdab7c7': ['@aws-sdk/client-ses@3.750.0'],
}


def download(url, path, expected=None):
 path.parent.mkdir(parents=True, exist_ok=True)
 if not path.exists():
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://proxy.internal:8080'}))
  with opener.open(url, timeout=180) as response, path.open('wb') as out:
   shutil.copyfileobj(response,out)
 if expected and sha(path)!=expected: raise ValueError(f'archive SHA256 mismatch: {path.name}')
 return sha(path)


def hashicorp(product, version, filename, target, scratch):
 base=f'https://releases.hashicorp.com/{product}/{version}/'
 checks=scratch/f'{product}-{version}.SHASUMS'
 download(base+f'{product}_{version}_SHA256SUMS',checks)
 expected=next(line.split()[0] for line in checks.read_text().splitlines() if line.split()[-1]==filename)
 download(base+filename,target,expected)
 return {'url':base+filename,'sha256':expected}


def copy_file(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 if source.stat().st_size>1024*1024:
  subprocess.run(['dd',f'if={source}',f'of={target}','bs=64M','oflag=direct','status=none'],check=True)
 else: shutil.copy2(source,target)


def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--scratch',type=Path,required=True)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--sif-sha256',required=True)
 args=ap.parse_args(argv)
 if (args.out/'manifest.json').exists():raise ValueError('manifest already published; use a new cache version/path')
 args.scratch.mkdir(parents=True,exist_ok=True)
 archive=args.scratch/'node.tar.xz';download(NODE_URL,archive,NODE_SHA256)
 subprocess.run(['tar','-C',str(args.scratch),'-xf',str(archive)],check=True)
 node=args.scratch/f'node-v{NODE_VERSION}-linux-x64/bin'
 env={**os.environ,'PATH':str(node)+':'+os.environ['PATH']}
 taskmap={t.task_id:t for t in tasks.load_tasks()}
 entries={}
 for task_id,requested in GROUPS.items():
  work=args.scratch/task_id;work.mkdir(exist_ok=True)
  cache=work/'npm';project=work/'resolution';project.mkdir(exist_ok=True)
  # This neutral dependency-resolution project remains in scratch; it is not a task implementation.
  with (work/'resolve.log').open('w') as log:
   subprocess.run([str(node/'npm'),'install','--prefix',str(project),'--cache',str(cache),
                   '--ignore-scripts','--no-audit','--no-fund','--save-exact',
                   '--registry=https://registry.npmjs.org','--https-proxy=http://proxy.internal:8080',*requested],
                  env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  root=args.out/'payloads'/task_id
  copy_file(archive,root/'runtime/node.tar.xz')
  # Only npm's original content-addressed public objects/index are included, not logs or build products.
  for p in sorted((cache/'_cacache').rglob('*')):
   if p.is_file():copy_file(p,root/'npm/_cacache'/p.relative_to(cache/'_cacache'))
  runtime_sources=[{'url':NODE_URL,'sha256':NODE_SHA256}]
  if task_id=='4122f866-01fa-400b-904d-fa171cdab7c7':
   name=f'terraform_{TERRAFORM_VERSION}_linux_amd64.zip';p=work/name
   runtime_sources.append(hashicorp('terraform',TERRAFORM_VERSION,name,p,args.scratch))
   copy_file(p,root/'runtime/terraform.zip')
   name=f'terraform-provider-aws_{AWS_PROVIDER_VERSION}_linux_amd64.zip';p=work/name
   runtime_sources.append(hashicorp('terraform-provider-aws',AWS_PROVIDER_VERSION,name,p,args.scratch))
   copy_file(p,root/'terraform-providers/registry.terraform.io/hashicorp/aws'/name)
  lock=json.loads((project/'package-lock.json').read_text())
  packages={name:{k:r[k] for k in ('version','resolved','integrity','engines') if k in r}
            for name,r in lock['packages'].items() if name}
  task=taskmap[task_id]
  entries[task_id]={'managers':['node','npm']+(['terraform'] if task_id.startswith('4122f866') else []),
   'requested':requested,'payload_root':str(root.absolute()),'instruction_sha256':instruction_sha(task),
   'task_prompt_sha256':prompt_sha(task),'sif_sha256':args.sif_sha256,
   'classification':'explicit-required-generic-dependencies','runtime_sources':runtime_sources,
   'support_choices':'Types, tsx and jsdom support neutral offline test execution; no task implementation supplied.',
   'package_lock_sha256':sha(project/'package-lock.json'),'package_inventory':packages,'files':inventory(root)}
  print(task_id,'generic packages',len(packages),'files',len(entries[task_id]['files']),flush=True)
 manifest={'schema_version':1,'cache_version':'gdpval-explicit-dependencies-v1','tasks':entries,
           'policy':'Evaluator setup networking only; agents install attempt-locally offline; no task code/data/solutions.'}
 args.out.mkdir(parents=True,exist_ok=True)
 p=args.out/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n')
 (args.out/'manifest.sha256').write_text(sha(p)+'  manifest.json\n')
 for item in args.out.rglob('*'):
  item.chmod(0o555 if item.is_dir() else 0o444)
 args.out.chmod(0o555)
 print('manifest',p,'sha256',sha(p))


if __name__=='__main__':main()
