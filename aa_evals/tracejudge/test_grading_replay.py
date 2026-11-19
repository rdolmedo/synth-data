"""Synthetic certificates only; no grader, network, or model execution."""
import copy
import io
import json
import tarfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest
from aa_evals.tracejudge import grading_replay as gr
from aa_evals.tracejudge.layouts import open_run
from aa_evals.tracejudge.digest import render
from aa_evals.tracejudge.rules import run_rules, decide
from aa_evals.tracejudge.core import build_row


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')


class Packet:
    def __init__(self, root, run=None):
        self.run=run or root/'source';self.d=self.run/'attempts'/gr.KEY
        (self.run/'queue').mkdir(parents=True,exist_ok=True);self.d.mkdir(parents=True,exist_ok=True)
        if not(self.run/'config.json').exists():write(self.run/'config.json',{'benchmark':'tb21','agent_internet':False})
        self.raw={'score':0.,'reward':0.,'infrastructure_failure':False,'exception_type':None,'exception_message':None,'stats':{'end_state':'finished'}}
        if not(self.d/'trial.tar.gz').exists():
            with tarfile.open(self.d/'trial.tar.gz','w:gz')as tf:
                content=b'historical fixture ValueError: cache configuration not found\n'
                m=tarfile.TarInfo(gr.KEY+'/T/verifier/test-stdout.txt');m.size=len(content);tf.addfile(m,io.BytesIO(content))
        if not(self.d/'attempt.json').exists():write(self.d/'attempt.json',{'key':gr.KEY})
        if not(self.d/'outcome.json').exists():write(self.d/'outcome.json',self.raw)
        self.raw=gr.read(self.d/'outcome.json');self.p=root/'proof';self.p.mkdir(parents=True)
        self.source={name:self.spec(path)for name,path in {'config':self.run/'config.json','archive':self.d/'trial.tar.gz','attempt':self.d/'attempt.json','outcome':self.d/'outcome.json'}.items()}
        self.plan={'key':gr.KEY,'schema_version':4,'runner_sha256':'a'*64,'protected_originals':{v['path']:v['sha256']for k,v in self.source.items()if k!='config'},'generation_calls':0,'original_writes':False,'passing_control_regrades':False,'restored_files':{'/app/fake.py':{'sha256':'f'*64,'mode':493}}}
        self.verifier=self.p/'trial/verifier'
        write(self.verifier/'ctrf.json',{'results':{'tests':[{'name':'test_outputs.py::test_compress_decompress_workflow','status':'passed'}],'summary':{'tests':1,'passed':1,'failed':0,'skipped':0,'pending':0,'other':0}}})
        write(self.verifier/'test-stdout.txt','PASSED test_compress_decompress_workflow\n');write(self.verifier/'reward.txt','1\n')
        data={'rows':3,'fingerprint':'fake','cache_files':[{'filename':'/grader-cache/fake.arrow','sha256':'b'*64}],'asset_sha256':'c'*64,'asset_bytes':12,'revision':'d'*40}
        call={k:True for k in ('original_loader_arguments','same_cache_files','same_fingerprint','packages_unchanged','no_arrow_rebuild','no_payload_redownload')}
        call.update({k:data[k]for k in ('rows','fingerprint','cache_files')});call.update(source_asset_sha256=data['asset_sha256'],official_heads=[{'revision':data['revision'],'etag':data['asset_sha256'],'bytes':data['asset_bytes']}]*2)
        self.receipt={'state':'NATIVE_GRADING_COMPLETE_PENDING_ROOT_ADOPTION','key':gr.KEY,'runner_sha256':self.plan['runner_sha256'],'generation_calls':0,'model_clients_constructed':0,'agent_constructions':0,'source_run_writes':False,'originals_unchanged':True,'grader_data_before_after_unchanged':True,'dataset':data,'restored_before':{'/app/fake.py':'f'*64},'restored_after':{'/app/fake.py':'f'*64},'original_call_before':copy.deepcopy(call),'original_call_after':copy.deepcopy(call),'grader_network':{'HF_HUB_OFFLINE':'0','HF_DATASETS_OFFLINE':'0'},'verifier_result':{'rewards':{'reward':1.}},'ctrf_sha256':gr.sha(self.verifier/'ctrf.json'),'verifier_log_sha256':gr.sha(self.verifier/'test-stdout.txt'),'reward_file':{'sha256':gr.sha(self.verifier/'reward.txt')}}
        self.audit={'state':'ROOT_REVIEWED_SCORING_ONLY_ADOPTION','schema_version':1,'kind':gr.KIND,'key':gr.KEY,'verdict':'adopt_native_grading','original_failure':'grader_fixture_before_test_call','generation_backend_failure':False,'originals_unchanged':True,'evidence':'Fake source review only.'}
        self.cert={'schema_version':1,'kind':gr.KIND,'key':gr.KEY,'verdict':'adopt_native_grading','source':self.source};self.path=self.d/'grading-replay-review.json';self.resign()

    @staticmethod
    def spec(path):return {'path':str(path),'sha256':gr.sha(path)}

    def resign(self):
        write(self.p/'plan.json',self.plan);self.receipt['plan_file_sha256']=gr.sha(self.p/'plan.json');write(self.p/'receipt.json',self.receipt)
        self.cert.update(plan=self.spec(self.p/'plan.json'),receipt=self.spec(self.p/'receipt.json'));self.cert.pop('root_audit',None)
        self.audit['certificate_payload_sha256']=gr.digest(self.cert);write(self.p/'root-audit.json',self.audit)
        self.cert['root_audit']=self.spec(self.p/'root-audit.json');write(self.path,self.cert)
        self.approved={'plan':self.cert['plan']['sha256'],'receipt':self.cert['receipt']['sha256'],'runner':self.plan['runner_sha256'],'source_config':self.source['config']['sha256']}

    def validate(self,outcome=None):
        with patch.dict(gr.APPROVED,self.approved,clear=True):return gr.validate(self.run,gr.KEY,self.path,outcome or self.raw)


def test_valid_original_and_derived_score_exact_generation(tmp_path):
    p=Packet(tmp_path);before={n:gr.sha(p.d/n)for n in ('attempt.json','trial.tar.gz')};e=p.validate();assert e['score']==1
    dest=tmp_path/'derived';shutil.copytree(p.run,dest);d=dest/'attempts'/gr.KEY
    o={**p.raw,'score':1.,'reward':1.,'grading_replay':e};write(d/'outcome.json',o)
    assert before=={n:gr.sha(p.d/n)for n in before}
    with patch.dict(gr.APPROVED,p.approved,clear=True):
        assert gr.validate(dest,gr.KEY,d/'grading-replay-review.json',o)==e
        t=open_run(dest).load(gr.KEY)
    text=render(t,run_rules(t));assert 'CURRENT validated native grade' in text and 'HISTORICAL evidence below' in text
    assert 'historical fixture ValueError' in text
    assert build_row(t,[],'clean',None,None,None)['recorded']['grading_replay']==e


def test_absent_byte_identical_input(tmp_path):
    p=Packet(tmp_path);p.path.unlink()
    with patch('aa_evals.tracejudge.grading_replay.trace_evidence',return_value={}):
        old=open_run(p.run).load(gr.KEY);expected=render(old,run_rules(old))
    actual=open_run(p.run).load(gr.KEY);assert actual==old and render(actual,run_rules(actual))==expected


@pytest.mark.parametrize('field,value',[('generation_calls',1),('generation_calls',False),('model_clients_constructed',1),('agent_constructions',1),('source_run_writes',True),('originals_unchanged',False),('grader_data_before_after_unchanged',False),('restored_after',{}),('state','PARTIAL')])
def test_receipt_scope_fail_closed_even_signed(tmp_path,field,value):
    p=Packet(tmp_path);p.receipt[field]=value;p.resign()
    with pytest.raises(ValueError):p.validate()


@pytest.mark.parametrize('field',['original_loader_arguments','same_cache_files','same_fingerprint','packages_unchanged','no_arrow_rebuild','no_payload_redownload'])
def test_original_fixture_pre_post_required(tmp_path,field):
    p=Packet(tmp_path);p.receipt['original_call_after'][field]=False;p.resign()
    with pytest.raises(ValueError):p.validate()


@pytest.mark.parametrize('mutation',['plan','receipt','config','archive','attempt','outcome','ctrf','log','reward','audit'])
def test_changed_proof_bytes(tmp_path,mutation):
    p=Packet(tmp_path);path={'plan':p.p/'plan.json','receipt':p.p/'receipt.json','audit':p.p/'root-audit.json','ctrf':p.verifier/'ctrf.json','log':p.verifier/'test-stdout.txt','reward':p.verifier/'reward.txt'}.get(mutation)
    if path is None:path=Path(p.source[mutation]['path'])
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):p.validate()


@pytest.mark.parametrize('field,value',[('infrastructure_failure',True),('exception_type','APIConnectionError'),('exception_message','backend down'),('stats',{'end_state':'error'})])
def test_primary_backend_infra_no_override(tmp_path,field,value):
    p=Packet(tmp_path);p.raw[field]=value;write(p.d/'outcome.json',p.raw);p.source['outcome']=p.spec(p.d/'outcome.json');p.plan['protected_originals'][str(p.d/'outcome.json')]=p.source['outcome']['sha256'];p.resign()
    with pytest.raises(ValueError,match='primary generation'):p.validate()


def test_invalid_stays_pending_flagged(tmp_path):
    p=Packet(tmp_path);write(p.path,{'schema_version':True});t=open_run(p.run).load(gr.KEY);hits=run_rules(t)
    assert t.recorded['grading_replay_review_pending'] is True and decide(hits)=='infra'
    assert build_row(t,hits,'infra',{'ok':True,'flags':[]},None,None)['flagged'] is True


@pytest.mark.parametrize('value', [None, {}, [], False, ''])
def test_present_empty_replay_metadata_requires_review(tmp_path, value):
    p=Packet(tmp_path);p.path.unlink()
    outcome={**p.raw,'grading_replay':value}
    assert gr.trace_evidence(p.run,gr.KEY,outcome)['grading_replay_review_pending'] is True


@pytest.mark.parametrize('status',['error','skipped','pending'])
def test_no_actual_test_call(tmp_path,status):
    p=Packet(tmp_path);c=gr.read(p.verifier/'ctrf.json');c['results']['tests'][0]['status']=status;write(p.verifier/'ctrf.json',c);p.receipt['ctrf_sha256']=gr.sha(p.verifier/'ctrf.json');p.resign()
    with pytest.raises(ValueError,match='test call'):p.validate()


def test_unsupported_missing_symlink_duplicate(tmp_path):
    p=Packet(tmp_path)
    with pytest.raises(ValueError,match='physical key'):gr.validate(p.run,'other',p.path,p.raw)
    raw=p.path.read_text();p.path.write_text(raw.replace('"schema_version": 1','"schema_version": 1,"schema_version": 1'))
    with pytest.raises(ValueError,match='duplicate'):p.validate()
    p.resign();p.path.unlink();p.path.symlink_to(p.p/'receipt.json')
    with pytest.raises(ValueError,match='regular file'):p.validate()
    p.path.unlink();write(p.d/'outcome.json',{**p.raw,'grading_replay':{'certificate_sha256':'f'*64}})
    assert gr.trace_evidence(p.run,gr.KEY,gr.read(p.d/'outcome.json'))['grading_replay_review_pending'] is True

@pytest.mark.parametrize('case',['unapproved_receipt','unapproved_config','unapproved_plan','unsigned_audit','derived_wrong_score','changed_official_revision','false_reward'])
def test_remaining_admission_failures(tmp_path,case):
    p=Packet(tmp_path)
    if case=='unapproved_receipt':p.approved['receipt']='0'*64
    elif case=='unapproved_plan':p.approved['plan']='0'*64
    elif case=='unapproved_config':p.approved['source_config']='0'*64
    elif case=='unsigned_audit':p.audit['state']='PROPOSED_ROOT_REVIEW_REQUIRED';p.resign()
    elif case=='changed_official_revision':p.receipt['original_call_before']['official_heads'][0]['revision']='0'*40;p.resign()
    elif case=='false_reward':p.receipt['verifier_result']['rewards']['reward']=False;p.resign()
    elif case=='derived_wrong_score':
        import shutil
        e=p.validate();dest=tmp_path/'derived';shutil.copytree(p.run,dest);p.run=dest
        p.raw={**p.raw,'score':0.,'reward':0.,'grading_replay':e};write(dest/'attempts'/gr.KEY/'outcome.json',p.raw)
    with pytest.raises(ValueError):p.validate()
