"""Pi modality is a recorded launch capability, not a model-name guess."""
import json
from pathlib import Path
import pytest
from . import model_input as M


def ep(tmp_path, argvs, name='opaque'):
 p=tmp_path/'endpoints.json';p.write_text('{}')
 (tmp_path/'launch.json').write_text(json.dumps({'replicas':[{'argv':[*a,'--served-model-name',name]} for a in argvs]}))
 return {'kind':'serving','registry':str(p),'served':name,'job_id':'fake.0'}


def config(record):return {'model_input':record,'protocol':{'model_input':record['modalities']}}


@pytest.mark.parametrize('argv',[
 ['server','--limit-mm-per-prompt','{"image":256,"video":0}'],
 ['server','--limit-mm-per-prompt={"image":1}'],
])
def test_actual_launch_image_allowance_and_frozen_hash(tmp_path,argv):
 r=M.resolve(ep(tmp_path,[argv,argv]));assert r['modalities']==['text','image']
 assert r['resolved_image_capability'] is True and r['source']['sha256']
 assert len(r['source']['replicas'])==2 and r['source']['kind']=='serving_launch'


@pytest.mark.parametrize('argv',[
 ['server','--language-model-only','--limit-mm-per-prompt','{"image":256}'],
 ['server','--limit-mm-per-prompt','{"image":0}'],
])
def test_text_launch_never_infers_vision_from_name(tmp_path,argv):
 r=M.resolve(ep(tmp_path,[argv],name='very-vl-vision-image-model'))
 assert r['modalities']==['text'] and r['resolved_image_capability'] is False
 with pytest.raises(ValueError,match='contradicts'):M.resolve(ep(tmp_path,[argv]),'on')


def test_mixed_replica_capabilities_rejected(tmp_path):
 with pytest.raises(ValueError,match='replicas disagree'):
  M.resolve(ep(tmp_path,[['s','--limit-mm-per-prompt','{"image":1}'],['s','--language-model-only']]))


def test_unknown_url_keeps_recorded_text_default_or_explicit_declaration():
 unknown={'kind':'url','served':'opaque'}
 auto=M.resolve(unknown);assert auto['modalities']==['text'] and auto['resolved_image_capability'] is None
 assert 'unknown' in auto['selection']
 assert M.resolve(unknown,'on')['modalities']==['text','image']
 assert M.resolve(unknown,'on')['selection']=='explicit declaration'
 assert M.resolve(unknown,'off')['modalities']==['text']


@pytest.mark.parametrize('argv',[
 ['s','--limit-mm-per-prompt'],['s','--limit-mm-per-prompt','bad'],
 ['s','--limit-mm-per-prompt','{"image":true}'],['s','--limit-mm-per-prompt','{"image":-1}'],
 ['s','--limit-mm-per-prompt','{}','--limit-mm-per-prompt','{}'],
 ['server'],['server','--limit-mm-per-prompt','{}'],
])
def test_unparsed_launch_cannot_implicitly_enable_images(tmp_path,argv):
 r=M.resolve(ep(tmp_path,[argv]));assert r['modalities']==['text']
 assert r['resolved_image_capability'] is None and r['source']['kind']=='unknown'


@pytest.mark.parametrize('bad',['null','[]','["image"]','["text","video"]','true','not json'])
def test_invalid_reserved_environment_rejected(bad):
 with pytest.raises(ValueError):M.from_env({M.ENV:bad})


def test_worker_uses_frozen_modalities_and_rejects_request_or_ambient_override(tmp_path,monkeypatch):
 from .worker import PiEvalWorker,q35
 monkeypatch.setattr(q35.Worker,'env',lambda *a:{})
 worker=object.__new__(PiEvalWorker)
 r=M.resolve(ep(tmp_path,[['server','--limit-mm-per-prompt','{"image":1}']]))
 worker.config={'benchmark':'tb21','agent':'pi',**config(r)}
 assert json.loads(worker.env('task__a0__r0','task',0)[M.ENV])==['text','image']
 worker.config['harbor_env']={M.ENV:'["text"]'}
 with pytest.raises(ValueError,match='conflicts'):worker.env('task__a0__r0','task',0)
 worker.config.pop('harbor_env');worker.config['request']={'overrides':{},'env':{M.ENV:'["text"]'}}
 with pytest.raises(ValueError,match='conflicts'):worker.env('task__a0__r0','task',0)
 monkeypatch.setattr(q35.Worker,'env',lambda *a:{M.ENV:'["text"]'})
 worker.config.pop('request')
 with pytest.raises(ValueError,match='conflicts'):worker.env('task__a0__r0','task',0)


def test_legacy_input_is_unchanged_and_existing_runs_cannot_auto_upgrade(tmp_path,monkeypatch):
 from . import run
 env={};M.apply_env({},env);assert M.ENV not in env
 with pytest.raises(ValueError,match='conflicts'):M.apply_env({}, {M.ENV:'["text","image"]'})
 monkeypatch.setattr(run,'RUNS_ROOT',tmp_path);monkeypatch.setattr(run,'status',lambda *a:{})
 monkeypatch.setattr(run,'workers_alive',lambda *a:True)
 monkeypatch.setattr(run.model_input,'resolve',lambda *a:pytest.fail('stored run re-resolved its modality'))
 p=tmp_path/'tb21/legacy';p.mkdir(parents=True)
 (p/'config.json').write_text(json.dumps({'agent':'pi','agent_internet':True,'worker':{'n':1}}))
 run.main(['--bench','tb21','--run-id','legacy','--agent-internet','--no-submit'])
 with pytest.raises(SystemExit,match='input policy differs'):
  run.main(['--bench','tb21','--run-id','legacy','--agent-internet','--no-submit','--model-images','on'])


def test_protocol_record_mismatch_rejected(tmp_path):
 r=M.resolve({'kind':'url'});cfg=config(r);cfg['protocol']['model_input']=['text','image']
 with pytest.raises(ValueError,match='disagree'):M.frozen_modalities(cfg)


def test_native_new_run_records_actual_input_without_model_or_submission(tmp_path,monkeypatch):
 from . import run
 import io,tarfile
 actual=ep(tmp_path,[['server','--limit-mm-per-prompt','{"image":1}']])
 actual.update(served='opaque',urls=['http://fake.invalid/v1'],request={'overrides':{'temperature':1,'top_p':.95}})
 monkeypatch.setattr(run.endpoint,'alive',lambda *a:(True,actual,'fake'))
 monkeypatch.setattr(run.endpoint,'max_model_len',lambda *a:131072)
 monkeypatch.setattr(run,'RUNS_ROOT',tmp_path/'runs')
 monkeypatch.setattr(run,'snapshot_code',lambda *a:str(Path(run.__file__).resolve().parents[3]))
 monkeypatch.setattr(run,'submit_workers',lambda *a:pytest.fail('submission'))
 monkeypatch.setattr(run.runtime_cache,'load_manifest',lambda *a:{})
 bundle=tmp_path/'bundle.tar'
 content=b'version = "1.0"\n[agent]\ntimeout_sec=21600\n'
 with tarfile.open(bundle,'w') as tf:
  m=tarfile.TarInfo('tasks/fake/task.toml');m.size=len(content);tf.addfile(m,io.BytesIO(content))
 run.main(['--bench','tb21','--run-id','fresh','--model','serving:fake','--seed-namespace','fake-seed','--bundle',str(bundle),'--no-submit'])
 cfg=json.loads((tmp_path/'runs/tb21/fresh/config.json').read_text())
 assert cfg['model_input']['modalities']==cfg['protocol']['model_input']==['text','image']
 assert cfg['model_input']['source']['kind']=='serving_launch'
 assert cfg['model_input']['source']['sha256']


@pytest.mark.parametrize('declaration',['auto','on'])
def test_worker_pauses_on_changed_or_missing_launch_instead_of_auto_changing_input(tmp_path,monkeypatch,declaration):
 from . import worker as W
 actual=ep(tmp_path,[['server','--limit-mm-per-prompt','{"image":1}']])
 actual.update(urls=['http://fake.invalid/v1'])
 record=M.resolve(actual,declaration)
 worker=object.__new__(W.PiEvalWorker);worker.run=tmp_path/'run';worker.run.mkdir();(worker.run/'queue').mkdir()
 worker.config={'agent':'pi','model_spec':'serving:fake','served':'opaque','max_in_flight':2,**config(record)}
 worker.ok=False;worker.gate_host=None;worker.urls=[];worker.name='fake'
 monkeypatch.setattr(W.registry,'alive',lambda *a:(True,actual,'fake'))
 monkeypatch.setattr(W,'alive',lambda *a:True);monkeypatch.setattr(W,'gate_queued_new',lambda *a:None)
 worker.refresh();assert worker.ok
 (tmp_path/'launch.json').unlink()
 worker.refresh();assert not worker.ok
 assert 'frozen' in json.loads((worker.run/'queue/admission.json').read_text())['reason']
 (tmp_path/'launch.json').write_text(json.dumps({'replicas':[{'argv':['server','--language-model-only']}]}))
 worker.refresh();assert not worker.ok
 admission=json.loads((worker.run/'queue/admission.json').read_text());assert admission['paused']
 assert ('frozen' in admission['reason']) if declaration=='auto' else ('contradicts' in admission['reason'])
 assert worker.config['model_input']['modalities']==['text','image']


def test_exact_launch_bytes_survive_rotation_and_tamper_is_rejected(tmp_path):
 import base64,hashlib
 actual=ep(tmp_path,[['server','--limit-mm-per-prompt','{"image":1}']])
 path=tmp_path/'launch.json';original=path.read_bytes()+b' \n\t'
 path.write_bytes(original);record=M.resolve(actual);cfg=config(record)
 assert base64.b64decode(record['source']['launch_json_base64'],validate=True)==original
 assert record['source']['sha256']==hashlib.sha256(original).hexdigest()
 path.write_text('{}')
 assert M.frozen_modalities(cfg)==['text','image']
 record['source']['launch_json_base64']=base64.b64encode(b'{}').decode()
 with pytest.raises(ValueError,match='hash mismatch'):M.frozen_modalities(cfg)
 record['source'].pop('launch_json_base64')
 with pytest.raises(ValueError,match='snapshot'):M.frozen_modalities(cfg)


@pytest.mark.parametrize('known',[['server','--language-model-only'],['server','--limit-mm-per-prompt','{"image":1}']])
def test_partial_unknown_replica_retains_known_evidence_and_blocks_contradiction(tmp_path,known):
 actual=ep(tmp_path,[known,['server','--limit-mm-per-prompt','malformed']])
 auto=M.resolve(actual)
 assert auto['modalities']==['text'] and auto['resolved_image_capability'] is None
 assert auto['source']['kind']=='unknown' and auto['source']['replicas'][1]['images'] is None
 images=auto['source']['replicas'][0]['images']
 assert images is ('--language-model-only' not in known)
 with pytest.raises(ValueError,match='contradicts'):
  M.resolve(actual,'off' if images else 'on')
 declared=M.resolve(actual,'on' if images else 'off')
 assert declared['resolved_image_capability'] is None
 assert declared['modalities']==(['text','image'] if images else ['text'])


def test_mixed_known_replicas_rejected_even_with_unknown_replica(tmp_path):
 actual=ep(tmp_path,[['server','--language-model-only'],['server'],['server','--limit-mm-per-prompt','{"image":1}']])
 for declaration in ['auto','on','off']:
  with pytest.raises(ValueError,match='replicas disagree'):M.resolve(actual,declaration)


@pytest.mark.parametrize('flags',[
 ['--served-model-name','alias1','opaque','--max-model-len','131072'],
 ['--served-model-name=opaque'],
 ['--served-model-name=alias1','opaque'],
])
def test_launch_identity_supports_actual_served_aliases(tmp_path,flags):
 actual={'kind':'serving','registry':str(tmp_path/'endpoints.json'),'served':'opaque','job_id':'registry-only'}
 launch={'teacher':'unrelated-teacher','time':123,'replicas':[{'argv':['server','--limit-mm-per-prompt','{"image":1}',*flags]}]}
 (tmp_path/'launch.json').write_text(json.dumps(launch))
 record=M.resolve(actual)
 assert record['resolved_image_capability'] is True
 assert 'opaque' in record['source']['replicas'][0]['served_names']
 assert record['source']['registry_job_id']=='registry-only' and 'job_id' not in record['source']
 assert record['source']['launch_time']==123


def test_missing_launch_identity_is_unknown_not_implicit_image_capability(tmp_path):
 actual={'kind':'serving','registry':str(tmp_path/'endpoints.json'),'served':'opaque'}
 (tmp_path/'launch.json').write_text(json.dumps({'teacher':'opaque','replicas':[{'argv':['server','--limit-mm-per-prompt','{"image":1}']}]}))
 record=M.resolve(actual)
 assert record['resolved_image_capability'] is None and record['modalities']==['text']
 assert record['source']['replicas'][0]['launch_flag_images'] is True
 assert M.resolve(actual,'on')['resolved_image_capability'] is None


@pytest.mark.parametrize('replicas',[
 [['server','--served-model-name','old-vision','--limit-mm-per-prompt','{"image":1}']],
 [['server','--served-model-name','opaque','--limit-mm-per-prompt','{"image":1}'],
  ['server','--served-model-name','old-vision','--limit-mm-per-prompt','{"image":1}']],
 [['server','--limit-mm-per-prompt','malformed'],['server','--served-model-name','old-vision','--language-model-only']],
])
def test_stale_or_mismatched_launch_identity_rejected_even_with_unknown_peer(tmp_path,replicas):
 actual={'kind':'serving','registry':str(tmp_path/'endpoints.json'),'served':'current-text' if len(replicas)==1 else 'opaque','job_id':'new-job'}
 (tmp_path/'launch.json').write_text(json.dumps({'replicas':[{'argv':a} for a in replicas]}))
 for declaration in ['auto','on','off']:
  with pytest.raises(ValueError,match='aliases do not match'):M.resolve(actual,declaration)


def test_repeated_served_name_is_ambiguous_and_rejected(tmp_path):
 actual=ep(tmp_path,[['server','--served-model-name=opaque','--limit-mm-per-prompt','{"image":1}']])
 with pytest.raises(ValueError,match='ambiguous repeated'):M.resolve(actual)
