"""CPU-only actual-SIF proof using neutral generic dependencies, never a benchmark solution/model.

Evaluator setup networking may populate the separately reviewed manifest. All commands below execute through
GDPval's unchanged owned network namespace. A fake localhost OpenAI model-list endpoint proves the host boundary.
"""
from __future__ import annotations
import argparse
import asyncio
import hashlib
import json
import os
import shlex
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import httpx
from openai import AsyncOpenAI
from aa_evals.gdpval import runtime_cache as C, tasks
from aa_evals.gdpval.sandbox import ApptainerCodeExecToolProvider


REACT_TEST = r'''
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const dom = new JSDOM('<!doctype html><html><body></body></html>');
global.window=dom.window; global.document=dom.window.document; global.navigator=dom.window.navigator;
const React=require('react'); const {render,screen,cleanup}=require('@testing-library/react');
const sinon=require('sinon'); const click=sinon.spy();
render(React.createElement('button',{onClick:click},'Neutral fixture'));
screen.getByRole('button').click(); assert.equal(click.callCount,1); cleanup();
console.log('neutral React/TestingLibrary/Sinon integration passed');
'''
SOLIDITY_TEST = r'''
const assert=require('node:assert/strict'); const {isAddress}=require('ethers');
assert(isAddress('0x0000000000000000000000000000000000000000'));
const solc=require('solc');
const input={language:'Solidity',sources:{'Neutral.sol':{content:'pragma solidity ^0.8.28; contract NeutralFixture {}'}},settings:{outputSelection:{'*':{'*':['abi']}}}};
const result=JSON.parse(solc.compile(JSON.stringify(input)));
assert(!(result.errors||[]).some(e=>e.severity==='error'));
console.log('neutral ethers/Solidity compiler integration passed');
'''
TERRAFORM_CONFIG = '''terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "= 5.84.0"
    }
  }
}
'''


async def proof(a):
    report={'proof':'actual-pinned-SIF-neutral-offline-install-v1','gpu_requests':0,'generation_requests':0,
            'manifest_sha256':C.sha(a.manifest),'sif_sha256':C.sha(a.sif),'cases':[]}
    calls=[]
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            data=json.dumps({'object':'list','data':[{'id':'neutral-host-fixture','object':'model','owned_by':'fixture'}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    port=server.server_address[1]
    client=AsyncOpenAI(base_url=f'http://127.0.0.1:{port}/v1',api_key='neutral-fixture',
                       http_client=httpx.AsyncClient(trust_env=False))
    env_before=dict(os.environ)
    selected=[t for t in tasks.load_tasks() if t.task_id in C.REQUIRED_TASKS]
    cfg={'sandbox':{'sif_local':str(a.sif),'sif_sha256':report['sif_sha256']}}
    cfg['sandbox']['runtime_packages']=await asyncio.to_thread(C.freeze,a.manifest,selected,report['sif_sha256'])
    try:
        assert (await client.models.list()).data[0].id=='neutral-host-fixture'
        with tempfile.TemporaryDirectory(prefix='gdpval-neutral-cache-',dir=os.environ.get('_CONDOR_SCRATCH_DIR')) as scratch:
            for task in selected:
                work=Path(scratch)/task.task_id;home=work/'home';tmp=work/'tmp';home.mkdir(parents=True);tmp.mkdir()
                cache=await asyncio.to_thread(C.prepare,cfg,task,home)
                before=await asyncio.to_thread(C.inventory,Path(cache['payload_root']))
                stats={};case={'task_id':task.task_id,'commands':[]}
                provider=ApptainerCodeExecToolProvider(sif=str(a.sif),home=home,tmp=tmp,stats=stats,runtime_cache=cache)
                async with provider:
                    async def command(label,text,success=True):
                        result=await provider.run_command('set -eu\n'+text,timeout=600)
                        case['commands'].append({'label':label,'exit_code':result.exit_code,
                          'stdout':result.stdout[-2000:],'stderr':result.stderr[-2000:]})
                        if success != (result.exit_code==0):raise AssertionError(f'{label}: unexpected exit {result.exit_code}')
                    await command('agent-local-node-install',"python - <<'NEUTRAL_PY'\nimport os,pathlib,tarfile\ndest=pathlib.Path('/home/user/.local'); dest.mkdir(parents=True)\nwith tarfile.open('/opt/aa-runtime-cache/runtime/node.tar.xz') as archive:\n for member in archive.getmembers():\n  name=member.name.partition('/')[2]\n  if not name: continue\n  member.name=name\n  member.uid=os.getuid(); member.gid=os.getgid(); member.uname=member.gname=None\n  if member.islnk(): member.linkname=member.linkname.partition('/')[2]\n  archive.extract(member,dest,filter='fully_trusted')\nNEUTRAL_PY\nnode --version\nnpm --version")
                    requested=cache['receipt']['requested']
                    await command('agent-local-npm-offline-install',shlex.join(['npm','install','--offline','--no-audit','--no-fund','--prefix','/home/user/neutral',*requested]))
                    def node_script(source):
                        return 'cd /home/user/neutral\nnode - <<\'NEUTRAL_JS\'\n'+source+'\nNEUTRAL_JS'
                    if task.task_id.startswith('7de33b48'):
                        await command('neutral-react-testing-library-sinon',node_script(REACT_TEST))
                        await command('neutral-typescript-execution',"cd /home/user/neutral\nprintf 'const value: number = 7; if (value !== 7) throw Error();\\n' > neutral.ts\nnode --import tsx neutral.ts\n./node_modules/.bin/tsc --version")
                    elif task.task_id.startswith('0e386e32'):
                        await command('neutral-ethers-solidity',node_script(SOLIDITY_TEST))
                    else:
                        await command('neutral-aws-sdk-load',node_script("const {SESClient}=require('@aws-sdk/client-ses'); const c=new SESClient({region:'us-east-1'}); c.destroy(); console.log('neutral SDK construction passed; no service request');"))
                        (home/'neutral/provider.tf').write_text(TERRAFORM_CONFIG)
                        await command('agent-local-terraform-install',"python - <<'NEUTRAL_PY'\nimport zipfile,os\nwith zipfile.ZipFile('/opt/aa-runtime-cache/runtime/terraform.zip') as z:z.extractall('/home/user/.local/bin')\nos.chmod('/home/user/.local/bin/terraform',0o755)\nNEUTRAL_PY\nterraform version -json")
                        await command('neutral-terraform-provider-offline-init','terraform -chdir=/home/user/neutral init -backend=false -input=false -no-color')
                    await command('readonly-payload','touch '+C.MOUNT+'/agent-write-forbidden',success=False)
                    await command('known-proxy-egress-denied',"https_proxy=http://proxy.internal:8080 curl --proxy http://proxy.internal:8080 --max-time 3 --fail https://example.com",success=False)
                    await command('raw-socket-egress-denied',"python -c \"import socket; socket.create_connection(('1.1.1.1',443),2)\"",success=False)
                    await command('host-loopback-isolated',f'curl --noproxy "*" --max-time 2 --fail http://127.0.0.1:{port}/v1/models',success=False)
                assert before==await asyncio.to_thread(C.inventory,Path(cache['payload_root']))
                case.update(stats=stats,payload_immutable=True,passed=True);report['cases'].append(case)
        assert (await client.models.list()).data[0].id=='neutral-host-fixture'
        assert dict(os.environ)==env_before
        report.update(passed=True,host_model_client_calls=len(calls),host_environment_unchanged=True,
                      no_egress_enforcement='owned-user-and-network-namespace-no-uplink')
    except Exception as exc:
        report.update(passed=False,error=f'{type(exc).__name__}: {exc}')
        if 'case' in locals() and case not in report['cases']:report['cases'].append(case)
        raise
    finally:
        a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
        await client.close();server.shutdown()
    print(json.dumps({'passed':report['passed'],'cases':len(report['cases']),'out':str(a.out)}))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--sif',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args(argv)
    asyncio.run(proof(a))


if __name__=='__main__':main()
