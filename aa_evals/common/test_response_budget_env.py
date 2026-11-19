"""PI_RESPONSE_BUDGET into the single-turn suites' condor jobs: quoting, validation, preflight, and each submitter's
rendered submit file (nothing is submitted)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aa_evals.common import condor_cost
from aa_evals.common import response_budget_env as E
from tb_agents import response_budget as B

REPO = Path(__file__).resolve().parents[2]
OMNI = {'warn_after_response_tokens': 20192, 'finish_response_tokens': 10000, 'total_response_tokens': 30192,
        'token_counting': 'super', 'near_window': True, 'notice_extra': B.NOTICE_EXTRA['omniscience']}
OMNI_JSON = json.dumps(OMNI, separators=(',', ':'))


def parse_environment(sub: str) -> dict:
    """The `environment = "..."` line of a submit file, parsed as HTCondor does: "" is a literal " inside the outer
    quotes; entries are whitespace-separated name=value; '...' quotes a value, '' inside it is a literal '.
    (Checked once against condor_submit -dry-run 10.0.9: the job ad's Environment decodes to the same values.)"""
    line = next(x for x in sub.splitlines() if x.startswith('environment = '))
    s = line[len('environment = '):]
    assert s[0] == s[-1] == '"', line
    s = s[1:-1].replace('""', '"')
    toks, tok, q, i = [], [], False, 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            if q and s[i + 1:i + 2] == "'":
                tok.append("'")
                i += 2
                continue
            q = not q
        elif ch.isspace() and not q:
            if tok:
                toks.append(''.join(tok))
            tok = []
        else:
            tok.append(ch)
        i += 1
    assert not q, 'unbalanced single quote'
    if tok:
        toks.append(''.join(tok))
    return dict(t.split('=', 1) for t in toks)


def test_condor_quoting_round_trips_json_with_quotes_spaces_commas():
    v = json.dumps(dict(OMNI, notice_extra="say you don't know, \"really\""), separators=(',', ':'))
    env = {'HOME': '/home/x', 'HLE_JOB_ID': '$(Cluster).$(Process)', E.ENV: v}
    got = parse_environment(f'environment = "{E.condor_env(env)}"')
    assert got == env
    # values without quotes render as before (hle/submit.py's old f"{k}='{v}'")
    plain = {'PYTHONPATH': '/c', 'no_proxy': 'localhost,127.0.0.1'}
    assert E.condor_env(plain) == ' '.join(f"{k}='{v}'" for k, v in plain.items())


def test_value_validates_and_compacts():
    assert E.value({}) is None and E.value({E.ENV: ''}) is None
    assert E.value({E.ENV: json.dumps(OMNI, indent=2)}) == OMNI_JSON          # pretty JSON: compact, no newline
    with pytest.raises(AssertionError):
        E.value({E.ENV: json.dumps(dict(OMNI, total_response_tokens=1))})     # total != warn + finish
    with pytest.raises(SystemExit, match=r'\$\('):
        E.value({E.ENV: json.dumps(dict(OMNI, notice_extra='$(Cluster)'))})
    assert E.forwarded(env={}) == {} and E.forwarded(env={E.ENV: OMNI_JSON}) == {E.ENV: OMNI_JSON}


def test_preflight_refuses_an_interpreter_that_cannot_apply_the_budget(tmp_path):
    with pytest.raises(SystemExit, match='cannot apply it'):
        E.preflight(sys.executable, tmp_path, OMNI_JSON)                       # no tb_agents in this "frozen copy"
    with pytest.raises(SystemExit, match='could not be checked'):
        E.preflight(str(tmp_path / 'no-python'), tmp_path, OMNI_JSON)
    if not all(Path(p).exists() for p in (B.TOKENIZER, B.SUPER_TOKENIZER, B.TEMPLATE)):
        pytest.skip('tokenizer files not on this machine')
    E.preflight(sys.executable, REPO, OMNI_JSON)                               # this venv has tokenizers + jinja2


@pytest.fixture
def captured(monkeypatch):
    subs = []
    monkeypatch.setattr(condor_cost, 'submit', lambda path, bid=None: subs.append(Path(path).read_text())
                        or [{'cluster': '123'}])
    monkeypatch.setattr(E, 'preflight', lambda python, code, v: subs.append(('preflight', python, str(code))))
    return subs


@pytest.mark.parametrize('budget', [None, OMNI_JSON], ids=['off', 'on'])
def test_hle_submit_forwards_the_budget(tmp_path, monkeypatch, captured, budget):
    from aa_evals.hle import submit as S
    assert 'tb_agents' in S.FROZEN
    monkeypatch.setattr(S, 'frozen_copy', lambda: (tmp_path / 'code', 'f' * 40))
    monkeypatch.setattr(S, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.delenv(E.ENV, raising=False)
    monkeypatch.setattr(S, 'PYTHON', sys.executable)   # the job interpreter: any host's (the preflight runs it)
    (tmp_path / 'code').mkdir()   # the frozen copy: the submitter checks its tb_agents imports as the job runs (#451)
    (tmp_path / 'code' / 'tb_agents').symlink_to(Path(S.__file__).resolve().parents[2] / 'tb_agents')
    if budget:
        monkeypatch.setenv(E.ENV, budget)
    S.submit(argparse.Namespace(run_id='r1', force=False, cpus=1, memory_gib=4, disk_gib=5, hours=1, bid=None),
             ['--model', 'serving:x'])
    sub = captured[-1]
    env = parse_environment(sub)
    assert env['HLE_JOB_ID'] == '$(Cluster).$(Process)' and env['PYTHONPATH'] == str(tmp_path / 'code')
    if budget:
        assert env[E.ENV] == budget and ('preflight', S.PYTHON, str(tmp_path / 'code')) in captured
    else:
        assert E.ENV not in env and len(captured) == 1


def test_hle_submit_refuses_a_frozen_copy_without_tb_agents(tmp_path, monkeypatch, captured):
    # #449: the HLE job imports tb_agents for every item; the submitter checks the copy the way the job runs
    from aa_evals.hle import submit as S
    monkeypatch.setattr(S, 'frozen_copy', lambda: (tmp_path / 'code', 'f' * 40))
    monkeypatch.setattr(S, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(S, 'PYTHON', sys.executable)
    (tmp_path / 'code').mkdir()
    with pytest.raises(SystemExit, match=r"cannot import \['tb_agents.response_budget', 'tb_agents.response_budget_single'\]"):
        S.submit(argparse.Namespace(run_id='r1', force=False, cpus=1, memory_gib=4, disk_gib=5, hours=1, bid=None),
                 ['--model', 'serving:x'])


@pytest.mark.parametrize('budget', [None, OMNI_JSON], ids=['off', 'on'])
def test_gdppdf_submit_forwards_the_budget_and_freezes_tb_agents(tmp_path, monkeypatch, captured, budget):
    from aa_evals.gdppdf import submit as S
    texts = []
    monkeypatch.setattr(S, 'submit', lambda path, text, bid=None: texts.append(text))
    monkeypatch.setattr(S, 'RUNS', tmp_path / 'runs')
    monkeypatch.delenv(E.ENV, raising=False)
    monkeypatch.delenv('AA_GDPPDF_RUNS', raising=False)
    if budget:
        monkeypatch.setenv(E.ENV, budget)
    monkeypatch.setattr(sys, 'argv', ['submit', 'run', '--run-id', 'r1', '--', '--model', 'serving:x'])
    S.main()
    env = parse_environment(texts[0])
    code = Path(env['PYTHONPATH'])
    assert (code / 'tb_agents' / 'response_budget.py').exists() and (code / 'tb_agents' / 'pi_context_limit.py').exists()
    assert not list((code / 'tb_agents').glob('test_*'))
    assert env['no_proxy'].startswith('localhost')
    if budget:
        assert env[E.ENV] == budget and ('preflight', S.PYTHON, str(code)) in captured
    else:
        assert E.ENV not in env and not captured
        line = next(x for x in texts[0].splitlines() if x.startswith('environment = '))
        assert line.endswith(' NO_PROXY=localhost,127.0.0.1,localnet,.internal.cluster.internal"')   # as before


@pytest.mark.parametrize('budget', [None, OMNI_JSON], ids=['off', 'on'])
def test_scicode_submit_condor_forwards_the_budget(tmp_path, monkeypatch, captured, budget):
    from aa_evals.scicode import run as R
    assert 'tb_agents' in R.FROZEN
    sha = 'e' * 40
    monkeypatch.setattr(R, 'RUNS', tmp_path / 'runs')
    monkeypatch.setattr(R, 'git_dirty', lambda: False)
    monkeypatch.setattr(R, 'git_sha', lambda: sha)
    (tmp_path / 'runs' / '_code' / sha[:12]).mkdir(parents=True)               # frozen already: no git archive
    monkeypatch.delenv(E.ENV, raising=False)
    if budget:
        monkeypatch.setenv(E.ENV, budget)
    R.submit_condor(argparse.Namespace(run_id='r1', cpus=2, mem_gb=4, cpu_bid=None), ['--run-id', 'r1', '--condor'])
    env = parse_environment(captured[-1])
    assert env['SCICODE_CODE_SHA'] == sha and env['APPTAINER_CACHEDIR'] == '/data/aai/apptainer-cache'
    if budget:
        assert env[E.ENV] == budget and ('preflight', R.VENV_PY, str(tmp_path / 'runs' / '_code' / sha[:12])) in captured
    else:
        assert E.ENV not in env and len(captured) == 1


@pytest.mark.parametrize('budget', [None, OMNI_JSON], ids=['off', 'on'])
def test_omniscience_submit_sh_forwards_the_budget(tmp_path, budget):
    """The real script with DRY_RUN=1 (writes job.sub, submits nothing); with a budget, the preflight runs for real
    against the job's default interpreter (the harbor venv)."""
    if budget and not all(Path(p).exists() for p in (B.TOKENIZER, B.SUPER_TOKENIZER, B.TEMPLATE)):
        pytest.skip('tokenizer files not on this machine')
    env = {k: v for k, v in os.environ.items() if k not in (E.ENV, 'BID', 'PYTHONPATH')}
    env.update(DRY_RUN='1', RUNS_ROOT=str(tmp_path))
    if budget:
        env[E.ENV] = budget
    r = subprocess.run(['bash', str(REPO / 'aa_evals/omniscience/submit.sh'), 'r1', '--model', 'serving:x'],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert 'dry run' in r.stdout
    code = next((tmp_path / 'r1').glob('code-*'))
    assert (code / 'tb_agents' / 'response_budget_single.py').exists() and not list((code / 'tb_agents').glob('test_*'))
    sub = (code / 'job.sub').read_text()
    got = parse_environment(sub)
    assert got['NO_PROXY'].endswith('10.0.0.0/8') and got['PYTHONNOUSERSITE'] == '1'
    if budget:
        assert got[E.ENV] == budget
    else:
        assert E.ENV not in got and sub.count("NO_PROXY='localhost,") == 1 and "10.0.0.0/8'\"\n" in sub
