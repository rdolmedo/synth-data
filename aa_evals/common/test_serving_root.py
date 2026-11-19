"""$TASKSYNTH_SERVING_ROOT moves every aa_evals registry reader (tbshared/serving.py's registry_root, through its
aa_evals.common.serving copy), and every condor job whose client reads the registry gets it in its environment (the
jobs run with getenv = false); unset, each submit file is as before. Nothing is submitted."""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from aa_evals.common import condor_cost, serving
from aa_evals.common.test_response_budget_env import captured, parse_environment   # noqa: F401 (fixture)

REPO = Path(__file__).resolve().parents[2]
NAME = 'regroot-test-x'   # absent from the production registry: a reader that ignores the variable fails


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / 'serving'
    d = root / NAME
    d.mkdir(parents=True)
    now = time.time()
    (d / 'endpoints.json').write_text(json.dumps({
        'served_name': 'from-env', 'urls': ['http://h:1/v1'], 'updated_at': now, 'healthy_at': now, 'job_id': '7.0',
        'request': {'temperature': 1.0}}))
    (d / 'launch.json').write_text(json.dumps({'argv': ['vllm', 'serve', '--max-model-len', '4096'], 'env': {}}))
    (d / 'job.json').write_text(json.dumps({'job_id': '7.0'}))
    monkeypatch.setenv(serving.ROOT_ENV, str(root))
    return root


def test_no_aa_evals_module_hard_codes_the_registry():
    """Only serving.py names the production registry in code (comments and docstrings may)."""
    bad = []
    for p in sorted((REPO / 'aa_evals').rglob('*.py')):
        if p.name.startswith('test_') or p == REPO / 'aa_evals/common/serving.py':
            continue
        tree = ast.parse(p.read_text())
        docs = {id(n.body[0].value) for n in ast.walk(tree)
                if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
                and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
        bad += [f'{p.relative_to(REPO)}:{n.lineno}' for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs
                and str(serving.ROOT) in n.value]
    assert not bad, bad
    assert (REPO / 'aa_evals/common/serving.py').read_text() == (REPO / 'tbshared/serving.py').read_text()


def test_no_module_joins_a_path_onto_a_root_constant():
    """The modules' roots are None (serving.registry_root resolves them): `SERVING_ROOT / name` would raise TypeError
    on None, so every join goes through serving.registry_root(...) (the dispatcher's three did not, 11-18)."""
    names = {'SERVING_ROOT', 'SERVING', 'REG'}
    bad = []
    for p in sorted((REPO / 'aa_evals').rglob('*.py')):
        if p.name.startswith('test_'):
            continue
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
                x = n.left
                if (isinstance(x, ast.Name) and x.id in names) or (isinstance(x, ast.Attribute) and x.attr in names):
                    bad.append(f'{p.relative_to(REPO)}:{n.lineno}')
            elif isinstance(n, ast.ImportFrom) and any(a.name in names for a in n.names):
                bad.append(f'{p.relative_to(REPO)}:{n.lineno} (import of a root: read it from the module at call time)')
    assert not bad, bad


def test_every_reader_follows_the_variable(registry, monkeypatch):
    """(CritPt's reader: aa_evals/critpt/test_serving_root.py, in the stirrup venv.)"""
    from aa_evals.automationbench import run as ab
    from aa_evals.campaign import launch as L
    from aa_evals.common import protocol
    from aa_evals.common.harbor_mswea import endpoint as hm
    from aa_evals.gdppdf import client as gdppdf
    from aa_evals.gdpval import endpoint as gdpval
    from aa_evals.hle import client as hle
    from aa_evals.omniscience import client as omni
    spec = f'serving:{NAME}'
    assert hle.resolve(spec).served_name == 'from-env'
    assert gdppdf.resolve(spec).served_name == 'from-env'
    assert omni.resolve(spec).model == 'from-env'
    assert gdpval._resolve_one(spec)['served_name'] == 'from-env'
    assert ab.resolve_model(spec)['served_name'] == 'from-env'
    assert hm.resolve(spec)['registry'] == str(registry / NAME / 'endpoints.json')
    assert protocol.launch(spec)['argv'][-1] == '4096'
    assert gdppdf.serving_launch_limits(spec)[0] == 4096
    assert ab.serving_snapshot(spec)['job.json']['content']['job_id'] == '7.0'
    monkeypatch.setattr(L, 'live_server', lambda r, runner=None: ['7.0'])   # condor stubbed
    assert L.healthy(NAME)['served_name'] == 'from-env'
    from aa_evals.qwenserve import monitor
    import importlib
    assert importlib.reload(monitor).REGISTRY == str(registry / 'aa-qwen*' / 'endpoints.json')


def test_forwarded_and_condor_env(monkeypatch):
    assert serving.forwarded({}) == {} and serving.forwarded({serving.ROOT_ENV: ''}) == {}
    assert serving.forwarded({serving.ROOT_ENV: '/data/x/serving'}) == {serving.ROOT_ENV: '/data/x/serving'}
    for bad in ('rel/path', '/a b', "/a'b", '/a"b', '/a$(Cluster)'):
        with pytest.raises(SystemExit, match=serving.ROOT_ENV):
            serving.forwarded({serving.ROOT_ENV: bad})
    assert serving.condor_env({}) == ''
    assert serving.condor_env({serving.ROOT_ENV: '/r'}) == f' {serving.ROOT_ENV}=/r'
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    assert serving.condor_env() == ''
    monkeypatch.setenv(serving.ROOT_ENV, '/r')
    assert serving.condor_env() == f' {serving.ROOT_ENV}=/r'
    out = subprocess.run([sys.executable, '-m', 'aa_evals.common.serving', 'condor-env'], capture_output=True, text=True,
                         cwd=REPO, env={**os.environ, 'PYTHONPATH': str(REPO)})
    assert out.returncode == 0 and out.stdout == f' {serving.ROOT_ENV}=/r'


# ------------------------------------------------------------------------------------------------ submit files
ROOT = '/data/x/serving'


def env_of(sub: str) -> dict:
    return parse_environment(sub)


@pytest.mark.parametrize('root', [None, ROOT], ids=['off', 'on'])
def test_pi_and_mswea_workers_get_the_runs_root_not_the_submitters(tmp_path, monkeypatch, root):
    """The dispatcher submits more workers later, from its own process: the root is the run's (cfg['serving_env'],
    recorded at init), whatever this process has."""
    from aa_evals.common.harbor_mswea import run as hm
    from aa_evals.common.pi_eval import run as pi
    subs = []
    monkeypatch.setattr(condor_cost, 'submit', lambda path, bid=None, exclude=None: subs.append(Path(path).read_text())
                        or [{'cluster': '1', 'n': 1, 'submit_file': str(path), 'bid': 26}])
    monkeypatch.setenv(serving.ROOT_ENV, '/somewhere/else')
    cfg = {'code': str(tmp_path), 'benchmark': 'b', 'run_id': 'r', 'worker': {'slots': 2, 'cpus': 1, 'mem_gb': 2, 'disk_gb': 5},
           **({'serving_env': {serving.ROOT_ENV: root}} if root else {})}
    for mod in (pi, hm):
        run = tmp_path / mod.__name__
        run.mkdir()
        mod.submit_workers(run, cfg, 1)
        got = env_of(subs[-1])
        assert got.get(serving.ROOT_ENV) == root, mod.__name__
        assert got['CONDOR_JOB_ID'] == '$(Cluster).$(Process)'


@pytest.mark.parametrize('root', [None, ROOT], ids=['off', 'on'])
def test_single_job_submitters_forward_the_root(tmp_path, monkeypatch, captured, root):   # noqa: F811
    from aa_evals.common import response_budget_env as E
    from aa_evals.gdppdf import submit as gdppdf
    from aa_evals.gdpval import submit as gdpval
    from aa_evals.hle import submit as hle
    from aa_evals.scicode import run as scicode
    monkeypatch.delenv(E.ENV, raising=False)
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    if root:
        monkeypatch.setenv(serving.ROOT_ENV, root)
    got = {}
    # hle (as test_response_budget_env)
    monkeypatch.setattr(hle, 'frozen_copy', lambda: (tmp_path / 'code', 'f' * 40))
    monkeypatch.setattr(hle, 'RUNS_ROOT', tmp_path / 'runs')
    monkeypatch.setattr(hle, 'PYTHON', sys.executable)
    (tmp_path / 'code').mkdir()
    (tmp_path / 'code' / 'tb_agents').symlink_to(REPO / 'tb_agents')
    hle.submit(argparse.Namespace(run_id='r1', force=False, cpus=1, memory_gib=4, disk_gib=5, hours=1, bid=None),
               ['--model', 'serving:x'])
    got['hle'] = env_of(captured[-1])
    # gdppdf
    texts = []
    monkeypatch.setattr(gdppdf, 'submit', lambda path, text, bid=None: texts.append(text))
    monkeypatch.setattr(gdppdf, 'RUNS', tmp_path / 'gdppdf')
    monkeypatch.delenv('AA_GDPPDF_RUNS', raising=False)
    monkeypatch.setattr(sys, 'argv', ['submit', 'run', '--run-id', 'r1', '--', '--model', 'serving:x'])
    gdppdf.main()
    got['gdppdf'] = env_of(texts[0])
    # scicode --condor
    sha = 'e' * 40
    monkeypatch.setattr(scicode, 'RUNS', tmp_path / 'scicode')
    monkeypatch.setattr(scicode, 'git_dirty', lambda: False)
    monkeypatch.setattr(scicode, 'git_sha', lambda: sha)
    (tmp_path / 'scicode' / '_code' / sha[:12]).mkdir(parents=True)
    scicode.submit_condor(argparse.Namespace(run_id='r1', cpus=2, mem_gb=4, cpu_bid=None), ['--run-id', 'r1', '--condor'])
    got['scicode'] = env_of(captured[-1])
    # gdpval (its job template)
    got['gdpval'] = env_of(gdpval.job('run', tmp_path, '-m x', tmp_path, 1, 4, 5, 'b'))
    for k, env in got.items():
        assert env.get(serving.ROOT_ENV) == root, k
        assert env['HOME'], k


@pytest.mark.parametrize('root', [None, ROOT], ids=['off', 'on'])
def test_automationbench_forwards_the_root(tmp_path, monkeypatch, root):
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', serving.ROOT_ENV)}
    if root:
        env[serving.ROOT_ENV] = root
    r = subprocess.run([sys.executable, '-m', 'aa_evals.automationbench.submit', '--dry-run', '--run-id', 'r1',
                        '--model', 'serving:x', '--out-root', str(tmp_path / 'ab')], cwd=REPO, capture_output=True,
                       text=True, env={**env, 'PYTHONPATH': str(REPO)}, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    got = env_of(r.stdout)
    assert got.get(serving.ROOT_ENV) == root and got['PYTHONNOUSERSITE'] == '1'


@pytest.mark.parametrize('root', [None, ROOT], ids=['off', 'on'])
def test_omniscience_submit_sh_forwards_the_root(tmp_path, root):
    """The real script with DRY_RUN=1 (writes job.sub, submits nothing)."""
    env = {k: v for k, v in os.environ.items() if k not in ('PI_RESPONSE_BUDGET', 'BID', 'PYTHONPATH', serving.ROOT_ENV)}
    env.update(DRY_RUN='1', RUNS_ROOT=str(tmp_path))
    if root:
        env[serving.ROOT_ENV] = root
    r = subprocess.run(['bash', str(REPO / 'aa_evals/omniscience/submit.sh'), 'r1', '--model', 'serving:x'],
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    sub = (next((tmp_path / 'r1').glob('code-*')) / 'job.sub').read_text()
    got = parse_environment(sub)
    assert got.get(serving.ROOT_ENV) == root and got['NO_PROXY'].endswith('10.0.0.0/8')


@pytest.mark.parametrize('root', [None, ROOT], ids=['off', 'on'])
def test_grader_and_judge_sweep_jobs_forward_the_root(tmp_path, monkeypatch, root):
    """The campaign grader's jobs (GDPval's judge resolves serving:<arm.server> inside its job) and the judge sweep."""
    from aa_evals.campaign import grade
    from aa_evals.judges import sweep
    monkeypatch.delenv(serving.ROOT_ENV, raising=False)
    if root:
        monkeypatch.setenv(serving.ROOT_ENV, root)
    assert env_of(grade.own_sub(tmp_path, tmp_path / 'x.sh', 'b', 1, 2, 3)).get(serving.ROOT_ENV) == root
    monkeypatch.setattr(sweep, 'frozen_copy', lambda ref=None: tmp_path / 'code')
    monkeypatch.setattr(sweep, 'STUDY_ROOT', tmp_path / 'studies')
    subs = []
    monkeypatch.setattr(condor_cost, 'submit', lambda path, bid=None: subs.append(Path(path).read_text()))
    arm = sorted(sweep.arms_mod.ARMS)[0]
    sweep.main(['submit', '--study', 's', '--arms', arm])
    assert env_of(subs[-1]).get(serving.ROOT_ENV) == root


def test_dispatcher_reads_the_runs_recorded_root(tmp_path, monkeypatch):
    """dispatch.json records the root at init; a dispatcher restarted from a shell with another value (or none) keeps
    reading the run's registry (gate status, max_num_seqs, the server it releases). Older dirs: this process's."""
    from aa_evals.common.dispatch import dispatcher as dsp
    from aa_evals.common.dispatch.test_dispatch import table
    from aa_evals.common.harbor_mswea import endpoint
    monkeypatch.setattr(endpoint, 'SERVING_ROOT', None)
    run_root = tmp_path / 'serving'
    monkeypatch.setenv(serving.ROOT_ENV, str(run_root))
    d = tmp_path / 'disp'
    dsp.main(['init', str(d), '--budget', '3', '--durations', str(table(tmp_path)), '--registry', 'm'])
    assert read(d / 'dispatch.json')['serving_env'] == {serving.ROOT_ENV: str(run_root)}
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'elsewhere'))
    assert dsp.Dispatcher(d).registry_root() == run_root
    monkeypatch.delenv(serving.ROOT_ENV)
    assert dsp.Dispatcher(d).registry_root() == run_root
    d2 = tmp_path / 'disp2'
    dsp.main(['init', str(d2), '--budget', '3', '--durations', str(table(tmp_path)), '--registry', 'm'])
    assert read(d2 / 'dispatch.json')['serving_env'] == {}
    monkeypatch.setenv(serving.ROOT_ENV, str(tmp_path / 'elsewhere'))
    assert dsp.Dispatcher(d2).registry_root() == serving.ROOT                  # recorded: production
    cfg = read(d2 / 'dispatch.json')
    del cfg['serving_env']
    (d2 / 'dispatch.json').write_text(json.dumps(cfg))
    assert dsp.Dispatcher(d2).registry_root() == tmp_path / 'elsewhere'       # an older dir: this process's
    monkeypatch.setattr(endpoint, 'SERVING_ROOT', tmp_path / 'test-root')      # a test's root wins
    assert dsp.Dispatcher(d).registry_root() == tmp_path / 'test-root'


def read(p):
    return json.loads(Path(p).read_text())


def test_aa_serving_root_is_refused_not_ignored():
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', serving.ROOT_ENV)}
    r = subprocess.run([sys.executable, '-c', 'import aa_evals.omniscience.client'], cwd=REPO, capture_output=True,
                       text=True, env={**env, 'PYTHONPATH': str(REPO), 'AA_SERVING_ROOT': '/x'})
    assert r.returncode != 0 and 'set TASKSYNTH_SERVING_ROOT' in r.stderr
