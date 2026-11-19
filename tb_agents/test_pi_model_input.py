"""Real pinned Pi read-image → actual next HTTP payload, with a deterministic fake endpoint."""
import base64,binascii,json,os,struct,subprocess,zlib
from types import SimpleNamespace
from pathlib import Path
import pytest
from aa_evals.common.pi_eval import model_input as M
from tb_agents.harness_agents import PiCached
from tb_agents.test_pi_reasoning_history import runtime,endpoint

# A valid synthetic PNG, never benchmark content.
def png_chunk(kind,data):
 return struct.pack('>I',len(data))+kind+data+struct.pack('>I',binascii.crc32(kind+data)&0xffffffff)
PNG=(b'\x89PNG\r\n\x1a\n'+png_chunk(b'IHDR',struct.pack('>IIBBBBB',16,16,8,2,0,0,0))
     +png_chunk(b'IDAT',zlib.compress((b'\x00'+b'\xff\x00\x00'*16)*16))+png_chunk(b'IEND',b''))


@pytest.mark.parametrize('mode',['vision','text','legacy'])
def test_actual_pinned_pi_tool_image_payload_follows_frozen_input(runtime,tmp_path,monkeypatch,mode):
 node,bundle=runtime;image=tmp_path/'synthetic.png';image.write_bytes(PNG)
 if mode!='legacy':
  (tmp_path/'launch.json').write_text(json.dumps({'replicas':[{'argv': ['server','--served-model-name','opaque','--limit-mm-per-prompt','{"image":1}'] if mode=='vision' else ['server','--served-model-name','opaque','--language-model-only']}]}))
  ep={'kind':'serving','registry':str(tmp_path/'endpoints.json'),'served':'opaque'}
  record=M.resolve(ep);cfg={'model_input':record,'protocol':{'model_input':record['modalities']}}
  env={};M.apply_env(cfg,env);monkeypatch.setenv(M.ENV,env[M.ENV])
 else:monkeypatch.delenv(M.ENV,raising=False)
 monkeypatch.setenv('PI_CONTEXT_WINDOW','131072');monkeypatch.setenv('PI_MAX_OUTPUT','16384')
 def respond(payload,index):
  if index==1:
   return ({'role':'assistant','tool_calls':[{'index':0,'id':'read-synthetic','type':'function','function':{'name':'read','arguments':json.dumps({'path':str(image)})}}]},'tool_calls',{'prompt_tokens':20,'completion_tokens':5})
  return ({'role':'assistant','content':'done'},'stop',{'prompt_tokens':30,'completion_tokens':1})
 with endpoint(respond) as (url,requests):
  agent=PiCached(logs_dir=tmp_path/'agent-logs',model_api='openai-completions',thinking='high',version='1.2.0')
  access=SimpleNamespace(configured_base_url=url,api_key='mock',env={'LOCAL_FAKE_KEY':'mock'})
  models=agent._build_custom_models_json(access,'opaque')
  model=models['providers']['harbor-endpoint']['models'][0]
  if mode=='legacy':assert 'input' not in model
  else:assert model['input']==(['text','image'] if mode=='vision' else ['text'])
  config=tmp_path/'pi-config';config.mkdir();(config/'models.json').write_text(json.dumps(models))
  (config/'settings.json').write_text(json.dumps({'httpIdleTimeoutMs':60000,'compaction':{'enabled':False}}))
  home=tmp_path/'home';home.mkdir()
  env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
  env.update(HOME=str(home),PI_CODING_AGENT_DIR=str(config),LOCAL_FAKE_KEY='mock',NO_PROXY='127.0.0.1')
  result=subprocess.run([str(node),str(bundle/'cli.js'),'--print','--mode','json','--provider','harbor-endpoint','--model','opaque','--thinking','high','--session-dir',str(tmp_path/'sessions'),'--tools','read',f'Read {image} then say done.'],cwd=tmp_path,env=env,text=True,capture_output=True,timeout=90)
  assert result.returncode==0,result.stdout[-2000:]+result.stderr[-2000:]
  assert len(requests)==2,[(r.get('messages'),r.get('model')) for r in requests]
  blocks=[b for msg in requests[1]['messages'] if isinstance(msg.get('content'),list) for b in msg['content']]
  images=[b for b in blocks if b.get('type')=='image_url']
  if mode=='vision':
   assert len(images)==1 and images[0]['image_url']['url'].startswith('data:image/')
   assert base64.b64decode(images[0]['image_url']['url'].split(',',1)[1])
  else:
   assert not images
   assert 'does not support images' in json.dumps(requests[1])
