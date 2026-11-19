"""Recorded per-task dispatch protects the other134 standard Pi paths and historical runs."""
import io
import json
from pathlib import Path
import tarfile

import pytest
from . import task_agents as T


def config(names):
    policy = T.selected(T.ALLOWED, names)
    return {'agent':'pi','benchmark':'tbsci_lite','envclock':True,'task_names':names,
            T.FIELD:policy,'protocol':{T.FIELD:dict(policy)}}


def test_actual_worker_dispatches_only_one_of_135_tasks(tmp_path, monkeypatch):
    from .worker import PiEvalWorker,q35
    monkeypatch.setattr(q35.Worker,'env',lambda *a:{'PI_CONT16K_CLASS':T.STANDARD})
    worker=object.__new__(PiEvalWorker)
    names=[f'ordinary-{i}' for i in range(134)]+[T.TASK]
    worker.config=config(names)
    for task in names:
        env=worker.env(f'{task}__a0__r0',task,0)
        assert env['PI_CONT16K_CLASS']==(T.MUSL if task==T.TASK else T.STANDARD)
        assert env.get(T.MARKER)==(task if task==T.TASK else None)


def test_subset_without_affected_task_records_empty_dispatch():
    cfg=config(['ordinary']);assert cfg[T.FIELD]=={}
    env={};T.apply_env(cfg,'ordinary',env)
    assert env['PI_CONT16K_CLASS']==T.STANDARD and T.MARKER not in env


def test_global_override_cannot_spread_task_adapter_to_other_tasks():
    cfg=config(['ordinary',T.TASK])
    with pytest.raises(ValueError,match='conflicts'):
        T.apply_env(cfg,'ordinary',{'PI_CONT16K_CLASS':T.MUSL})
    env={'PI_CONT16K_CLASS':T.MUSL};T.apply_env(cfg,T.TASK,env)
    assert env[T.MARKER]==T.TASK


def test_legacy_config_keeps_original_class_and_is_not_auto_upgraded():
    env={'PI_CONT16K_CLASS':T.STANDARD};T.apply_env({},T.TASK,env)
    assert env=={'PI_CONT16K_CLASS':T.STANDARD}


@pytest.mark.parametrize('policy',[{'ordinary':T.MUSL},{T.TASK:T.STANDARD},[],None])
def test_unsupported_new_declarations_rejected(policy):
    with pytest.raises(ValueError):T.selected(policy,[T.TASK,'ordinary'])


def test_changed_protocol_and_unselected_task_fail_closed():
    cfg=config([T.TASK]);cfg['protocol'][T.FIELD]={}
    with pytest.raises(ValueError,match='disagree'):T.frozen(cfg)
    cfg=config([T.TASK]);cfg['task_names']=[]
    with pytest.raises(ValueError,match='unselected'):T.frozen(cfg)
    cfg=config([T.TASK]);cfg['envclock']=False
    with pytest.raises(ValueError,match='environment clock'):T.frozen(cfg)
    cfg=config([T.TASK])
    with pytest.raises(ValueError,match='marker'):T.apply_env(cfg,T.TASK,{T.MARKER:'ordinary'})


def test_native_new_metadata_records_dispatch_without_model_or_submission(tmp_path,monkeypatch):
    from . import run
    ep={'kind':'url','served':'opaque','urls':['http://fake.invalid/v1'],
        'request':{'overrides':{'temperature':1,'top_p':.95}}}
    monkeypatch.setattr(run.endpoint,'alive',lambda *a:(True,ep,'fake'))
    monkeypatch.setattr(run.endpoint,'max_model_len',lambda *a:131072)
    monkeypatch.setattr(run,'RUNS_ROOT',tmp_path/'runs')
    monkeypatch.setattr(run,'snapshot_code',lambda *a:str(Path(run.__file__).resolve().parents[3]))
    monkeypatch.setattr(run,'submit_workers',lambda *a:pytest.fail('submission'))
    monkeypatch.setattr(run.runtime_cache,'load_manifest',lambda *a:{})
    bundle=tmp_path/'bundle.tar'
    with tarfile.open(bundle,'w') as tf:
        for task in ['ordinary',T.TASK]:
            content=b'version="1.0"\n[agent]\ntimeout_sec=21600\n'
            info=tarfile.TarInfo(f'tasks/{task}/task.toml');info.size=len(content);tf.addfile(info,io.BytesIO(content))
    defaults={T.FIELD:dict(T.ALLOWED)}
    args=['--bench','tbsci_lite','--run-id','fresh','--model','url:http://fake.invalid/v1@opaque',
          '--seed-namespace','fake','--bundle',str(bundle),'--no-submit']
    run.main(args,defaults)
    path=tmp_path/'runs/tbsci_lite/fresh/config.json';cfg=json.loads(path.read_text())
    assert cfg[T.FIELD]==cfg['protocol'][T.FIELD]==T.ALLOWED
    assert cfg['envclock'] and cfg['task_names']==['ordinary',T.TASK]
    # A stored run lacking dispatch remains legacy even when current defaults opt in.
    cfg.pop(T.FIELD);cfg['protocol'].pop(T.FIELD);path.write_text(json.dumps(cfg))
    monkeypatch.setattr(run,'status',lambda *a:{});monkeypatch.setattr(run,'workers_alive',lambda *a:True)
    before=path.read_bytes();run.main(args,defaults);assert path.read_bytes()==before
