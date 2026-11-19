"""Physical session records, not Unicode text lines; no model/sandbox execution."""
import hashlib,importlib.util,json,os
from pathlib import Path
import pytest
from aa_evals.tracejudge.jsonl import physical_lines
from aa_evals.tracejudge.layouts import PiEvalLayout
from aa_evals.common.pi_eval.worker import q35
from tb_agents.pi_continuation import annotate_sessions

REPO=Path(__file__).resolve().parents[2]

def encoded(events,ending='\n'):
    return (ending.join(json.dumps(e,ensure_ascii=False) for e in events)+ending).encode()

def assistant(text='neutral',**fields):
    return {'type':'message','message':{'role':'assistant','content':[{'type':'text','text':text}],
        'stopReason':'stop',**fields}}

@pytest.mark.parametrize('separator',['\u2028','\u0085','\u2029','\v','\f'])
@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_valid_literal_unicode_stays_in_one_record_and_preserves_ids(tmp_path,separator,ending):
    # JSON forbids literal ASCII controls; the Unicode cases are written literally.
    text='page A'+separator+'page B'
    if separator in ('\v','\f'):text='page A'+separator+'page B'  # json.dumps escapes these validly.
    events=[assistant(),{'type':'message','message':{'role':'toolResult','toolName':'bash',
        'toolCallId':'tool-physical-1','content':[{'type':'text','text':text}]}},assistant('final '+text)]
    raw=encoded(events,ending);path=tmp_path/'events.jsonl';path.write_bytes(raw)
    lines=q35._pi_session_lines(path)
    assert lines==physical_lines(raw.decode()) and len(lines)==3
    decoded=[json.loads(line)for line in lines]
    assert decoded==events and decoded[1]['message']['toolCallId']=='tool-physical-1'
    assert q35._continued_agent_model_response(lines,1)is None  # no fabricated paid response.
    turns=PiEvalLayout._session(raw.decode(),0)
    assert len(turns)==3 and turns[1].text==text and turns[2].text=='final '+text
    assert turns[1].ref=='turn 2'

def test_physical_lf_preserves_blank_indices_and_rejects_cr_as_record_separator(tmp_path):
    text='{}\r\n\r\n{}\r\n';p=tmp_path/'x.jsonl';p.write_bytes(text.encode())
    assert physical_lines(text)==q35._pi_session_lines(p)==['{}','','{}']
    assert physical_lines('')==[] and physical_lines('\n')==['']
    assert physical_lines('{}\r{}')==['{}\r{}']
    with pytest.raises(ValueError):json.loads(physical_lines('{}\r{}')[0])

@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_worker_last_assistant_marker_hash_and_paid_line_index_are_native(tmp_path,ending):
    events=[assistant('one\u2028two\u0085three'),
        {'type':'message','message':{'role':'assistant','content':[{'type':'toolCall','id':'call-denied',
          'name':'bash','arguments':{'command':'cat /etc/apt/sources.list; timeout 5 apt-get update'}}]}},
        {'type':'message','message':{'role':'toolResult','toolName':'bash','toolCallId':'call-denied',
          'content':[{'type':'text','text':'ProxyError: Cannot connect to proxy'}]}},
        assistant('paid\u0085continuation',api='openai-completions',provider='harbor-endpoint',
          responseId='chatcmpl-paid',usage={'output':42})]
    path=tmp_path/'agent/pi/sessions/native.jsonl';path.parent.mkdir(parents=True);path.write_bytes(encoded(events,ending))
    lines=q35._pi_session_lines(path);markers=[]
    assert q35.last_assistant(tmp_path)==events[-1]['message']
    assert q35._offline_agent_proxy_failure(path,review=markers,bridge_url='http://10.22.2.196:45753/v1',bridge_config_sha256='a'*64)is None
    assert len(markers)==1 and markers[0]['tool_call_id']=='call-denied'
    assert markers[0]['tool_result_sha256']==hashlib.sha256(lines[2].encode()).hexdigest()
    assert markers[0]['continuation']=={'line_index':3,'response_id':'chatcmpl-paid','event_sha256':hashlib.sha256(lines[3].encode()).hexdigest()}
    assert physical_lines(path.read_bytes().decode())==lines

def test_postrun_annotations_keep_events_weights_and_physical_error_line(tmp_path):
    events=[assistant('one\u2028two'),{'type':'message','message':{'role':'toolResult','toolCallId':'tool-preserved',
        'content':[{'type':'text','text':'PDF\u0085content'}]}},assistant('final')]
    p=tmp_path/'pi/sessions/native.jsonl';p.parent.mkdir(parents=True);raw=encoded(events,'\r\n');p.write_bytes(raw)
    result=annotate_sessions(tmp_path);out=result['sessions'][0]
    assert not result['parse_errors']and result['valid_for_export']and p.read_bytes()==raw
    assert out['sha256']==hashlib.sha256(raw).hexdigest()and len(out['entries'])==3
    assert [e.get('sft_loss_weight')for e in out['entries']]==[1,0,1]
    p.write_bytes(raw+b'{malformed}\r\n');result=annotate_sessions(tmp_path)
    assert result['parse_errors']==[{'source':'pi/sessions/native.jsonl','line':4}]

def test_postrun_stats_counts_unicode_assistant_without_touching_source(tmp_path):
    spec=importlib.util.spec_from_file_location('physical_lf_score',REPO/'eval_q35/score.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
    trial=tmp_path/'trial';p=trial/'agent/pi/sessions/native.jsonl';p.parent.mkdir(parents=True)
    raw=encoded([assistant('first\u2028response'),assistant('last\u0085response')]);p.write_bytes(raw)
    stats=score.trial_stats(tmp_path/'run',{'key':'neutral__a0__r0','trial_dir':str(trial)})
    assert stats['assistant_messages']==2 and p.read_bytes()==raw

# The real fixture remains private/outside Git; portable regressions above always run.
REAL=Path(os.environ.get('AA_FINANCIAL_LF_FIXTURE',
    '/data/aai/aa_evals/audits/pi_jsonl_physical_lf_20261107/financial-session.jsonl'))
@pytest.mark.skipif(not REAL.is_file(),reason='private bounded financial fixture is unavailable')
def test_real_bounded_financial_tool_result_is_not_lost():
    raw=REAL.read_bytes();assert hashlib.sha256(raw).hexdigest()=='e4344ed8140477f022e31d15b2e5c41f8c8a442bdf8c706af4a23267eccc9ad6'
    lines=physical_lines(raw.decode());events=[json.loads(line)for line in lines]
    assert len(events)==134 and len(q35._pi_session_lines(REAL))==134
    assert hashlib.sha256(lines[24].encode()).hexdigest()=='0bc60f5b36942d5e5f321a0cba5c7d52b5b532dae772b9ca1124b41d5241b16e'
    assert events[24]['message']['role']=='toolResult'and events[24]['message']['toolCallId']=='chatcmpl-tool-ad153194a91d7881'
    turns=PiEvalLayout._session(raw.decode(),0);assert len(turns)==194
    assert [turn.idx for turn in turns]==list(range(194))
    legacy=[]
    for line in raw.decode().splitlines():
        try:legacy.append(json.loads(line))
        except ValueError:pass
    assert len(legacy)==133
    legacy_turns=PiEvalLayout._session('\n'.join(json.dumps(e)for e in legacy),0)
    assert len(legacy_turns)==193

@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_bound_tool_review_validator_uses_same_physical_indices(ending):
    import io,tarfile
    from aa_evals.tracejudge.test_tool_network_review import ToolNetworkReviewTests
    case=ToolNetworkReviewTests();case.setUp()
    try:
        f=case.f
        prefix={'type':'message','message':{'role':'toolResult','toolCallId':'neutral-document',
            'content':[{'type':'text','text':'PDF\u2028page\u0085tail'}]}}
        lines=[json.dumps(prefix,ensure_ascii=False),*case.lines]
        session=(ending.join(lines)+ending).encode()
        with tarfile.open(f.attempt/'trial.tar.gz','w:gz')as tf:
            for name,raw in [('trial/config.json',case.relay),('trial/agent/pi/sessions/actual.jsonl',session)]:
                member=tarfile.TarInfo(name);member.size=len(raw);tf.addfile(member,io.BytesIO(raw))
        marker=f.outcome['offline_proxy_review_evidence'][0];marker['continuation']['line_index']=3
        archive_sha=hashlib.sha256((f.attempt/'trial.tar.gz').read_bytes()).hexdigest()
        session_sha=hashlib.sha256(session).hexdigest()
        marker_sha=hashlib.sha256(json.dumps([marker],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        f.audit['bindings'].update(archive_sha256=archive_sha,session_sha256=session_sha,marker_evidence_sha256=marker_sha)
        f.receipt.update(archive_sha256=archive_sha,marker_evidence_sha256=marker_sha)
        f.receipt['session']['sha256']=session_sha
        case.publish();review=f.review()
        assert review['offline_proxy_review_pending']is False, review
        assert physical_lines(session.decode())==lines
        # A hash-consistent old Unicode-split index is still invalid evidence.
        marker['continuation']['line_index']=5
        stale=hashlib.sha256(json.dumps([marker],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        f.audit['bindings']['marker_evidence_sha256']=stale;f.receipt['marker_evidence_sha256']=stale
        case.publish();assert f.review()['offline_proxy_review_pending']is True
    finally:case.doCleanups()

def test_startup_guard_keeps_valid_header_and_no_assistant_evidence(tmp_path):
    msg="Command failed (exit 1): chmod 600 /tmp/harbor-pi-agent/models.json\nchmod: changing permissions of '/tmp/harbor-pi-agent/models.json': Operation not permitted"
    result={'exception_info':{'exception_type':'NonZeroAgentExitCodeError','exception_message':msg,
      'exception_traceback':'harbor/agents/installed/pi.py in run\n await self._write_custom_models_json(\nin _write_custom_models_json\n'+msg},
      'agent_result':{'n_input_tokens':0,'n_output_tokens':0}}
    path=tmp_path/'agent/pi/sessions/native.jsonl';path.parent.mkdir(parents=True)
    path.write_bytes(encoded([{'type':'session','id':'header\u2028retained\u0085intact'}],'\r\n'))
    assert q35._agent_startup_configuration_failure(tmp_path,result)is True
    path.write_bytes(path.read_bytes()+encoded([assistant('actual\u2028assistant')],'\r\n'))
    assert q35._agent_startup_configuration_failure(tmp_path,result)is False

def test_worker_reader_supports_readonly_archive_paths():
    # The coordinator's read-only classification adapter exposes read_bytes,
    # avoiding extraction or a filesystem stream for saved archive members.
    class ArchiveMember:
        def read_bytes(self):return encoded([assistant('saved\u2028context\u0085data')],'\r\n')
    lines=q35._pi_session_lines(ArchiveMember())
    assert len(lines)==1 and json.loads(lines[0])['message']['content'][0]['text']=='saved\u2028context\u0085data'
