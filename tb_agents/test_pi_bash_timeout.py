"""The bash180_v1 Pi bundle: bash commands default to a 180 s timeout the model can raise.

Build the bundle first with
  gen/prepare_pi_cont16k.py --build --patch-set bash180_v1 --cache /data/aai/tbench/agent_cache_pi_cont_bash180_v1
"""
import json
import os
from pathlib import Path
import subprocess
import tarfile

import pytest

from tb_agents.pi_continuation import (
    BASH_CACHE, BASH_DEFAULT_TIMEOUT_SEC, BASH_PATCHES, BUNDLE, CACHE, PATCHES, bundle_provenance,
    patch_source, verify_bundle,
)


@pytest.fixture(scope='module')
def runtime(tmp_path_factory):
    cache = Path(os.environ.get('PI_BASH_TEST_CACHE', str(BASH_CACHE)))
    if not (cache / BUNDLE).is_file():
        pytest.skip('Build the bash180_v1 Pi bundle with gen/prepare_pi_cont16k.py')
    sha = verify_bundle(cache)
    assert bundle_provenance(cache, sha)['bash_default_timeout_sec'] == BASH_DEFAULT_TIMEOUT_SEC
    root = tmp_path_factory.mktemp('pi-bash-runtime')
    with tarfile.open(cache / BUNDLE) as archive:
        archive.extractall(root, members=[m for m in archive if m.isfile() or m.isdir()], filter='data')
    pi = root / 'opt/harbor-pi'
    return pi / 'node/bin/node', pi / 'pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle'


DRIVER = """
import {createBashTool} from 'BUNDLE_URL/chunks/chunk-TJXZSVTD.js';
// Record every timer the tool arms, so the 180 s default is visible without waiting for it.
const armed=[];const realSetTimeout=globalThis.setTimeout;
globalThis.setTimeout=(fn,ms,...rest)=>{armed.push(ms);return realSetTimeout(fn,ms,...rest)};
const tool=createBashTool(process.cwd());
const run=async(args,abortAfterMs)=>{
  const controller=new AbortController();
  if(abortAfterMs) realSetTimeout(()=>controller.abort(),abortAfterMs);
  const start=Date.now();armed.length=0;
  try{const r=await tool.execute('t',args,controller.signal,undefined,undefined);
      return {ok:true,text:r.content[0].text,armed:[...armed],seconds:(Date.now()-start)/1e3}}
  catch(e){return {ok:false,text:String(e.message),armed:[...armed],seconds:(Date.now()-start)/1e3}}
};
const out={description:tool.description,timeoutDescription:tool.parameters.properties.timeout.description};
out.quick=await run({command:'echo hi'});
out.defaulted=await run({command:'sleep 30'},1500);
out.explicit=await run({command:'sleep 30',timeout:1});
out.longer=await run({command:'sleep 30',timeout:600},1500);
console.log(JSON.stringify(out));
"""


def test_default_timeout_is_armed_stated_and_overridable(runtime, tmp_path):
    node, bundle = runtime
    driver = tmp_path / 'driver.mjs'
    driver.write_text(DRIVER.replace('BUNDLE_URL', bundle.as_uri()))
    result = subprocess.run([str(node), str(driver)], capture_output=True, text=True, timeout=60,
                            check=True, cwd=tmp_path)
    out = json.loads(result.stdout)
    # The model sees the default and how to raise it.
    assert 'killed after 180 seconds unless you provide a larger timeout' in out['description']
    assert out['timeoutDescription'].startswith('Timeout in seconds (default: 180).')
    assert out['quick']['ok'] and out['quick']['text'].strip() == 'hi'
    # No timeout argument: a 180 s kill timer is armed (the call is aborted here after 1.5 s).
    assert 180000 in out['defaulted']['armed'] and not out['defaulted']['ok']
    assert 'Command aborted' in out['defaulted']['text']
    # An explicit timeout wins, below or above the default, and the message says what to do next.
    assert not out['explicit']['ok'] and out['explicit']['seconds'] < 10
    assert 'Command timed out after 1 seconds and was killed.' in out['explicit']['text']
    assert 'run it again with a larger timeout' in out['explicit']['text']
    assert 600000 in out['longer']['armed'] and 180000 not in out['longer']['armed']


def test_bash_patches_apply_once_and_stock_bundle_stays_without_default():
    for suffix, old, new in BASH_PATCHES:
        assert patch_source('a' + old + 'b', old, new) == 'a' + new + 'b'
        with pytest.raises(ValueError):
            patch_source(new, old, new)
    if (CACHE / BUNDLE).is_file():
        sha = verify_bundle(CACHE)
        assert bundle_provenance(CACHE, sha)['bash_default_timeout_sec'] is None


def test_partially_patched_bundle_is_rejected(tmp_path):
    import hashlib
    import io
    archive = tmp_path / BUNDLE
    archive.parent.mkdir(parents=True)
    members = {}
    for suffix, old, new in PATCHES:
        members[suffix] = members.get(suffix, '') + new
    for i, (suffix, old, new) in enumerate(BASH_PATCHES):
        members[suffix] += new if i == 0 else old    # only the first bash patch applied
    with tarfile.open(archive, 'w') as output:
        for suffix, content in members.items():
            member = tarfile.TarInfo('opt/harbor-pi' + suffix)
            member.size = len(content.encode())
            output.addfile(member, io.BytesIO(content.encode()))
    archive.with_suffix('.tar.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='partially applied'):
        verify_bundle(tmp_path)
