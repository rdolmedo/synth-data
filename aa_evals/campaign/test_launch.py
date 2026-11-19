"""aa_evals.campaign.launch: config, commands, server waves and the canary gate (no condor needed)."""
import json
import types
from pathlib import Path

import pytest

from aa_evals.campaign import launch as L

EXAMPLE = Path(__file__).with_name('example_c30.json')


def campaign(tmp_path, **over):
    cfg = json.loads(EXAMPLE.read_text())
    plan = {'suites': {s: {'items': [{'id': f'{s}-{i}', 'repeats': 2} for i in range(5)]} for s in L.SUITES}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    cfg.update({'name': 'c99', 'plan': str(tmp_path / 'plan.json'), 'dispatch_root': str(tmp_path / 'dispatch'),
                'serving_workdir': str(tmp_path)}, **over)
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    return L.Campaign(tmp_path / 'c.json')


def wall(sec):
    """The example's suites block with suites.wall_cap_sec = sec (for campaign(..., suites=wall(...)))."""
    return {**json.loads(EXAMPLE.read_text())['suites'], 'wall_cap_sec': sec}


def test_config_substitutes_campaign_name(tmp_path):
    C = campaign(tmp_path)
    assert C.models['qwen35-4b']['registry'] == 'c99-qwen35-4b'
    assert C.s['capture_root'].endswith('/c99')
    assert C.run_id('gemma26', 'tb21', 'canary') == 'c99-gemma26-tb21-canary'


def test_pi_commands_leave_shape_bid_and_worker_count_to_defaults(tmp_path):
    C = campaign(tmp_path)
    b = L.build(C, 'qwen35-4b', 'full')
    for c in b['suites']:
        a = c['argv']
        for flag in ('--workers', '--slots', '--worker-cpus', '--worker-mem-gb', '--cpu-bid', '--bid'):
            assert flag not in a, (c['suite'], flag)
        assert not any(x.startswith('BID=') for x in a)
        if c['suite'] in L.PI_SUITES:
            assert a[a.index('--max-in-flight') + 1] == str(min(c['attempts'], 132))
            assert '--dispatch' in a
    init = b['dispatcher_init']
    assert init[init.index('--registry') + 1] == 'c99-qwen35-4b'
    assert init[init.index('--release-server-after') + 1] == '600'
    canary = L.build(C, 'qwen35-4b', 'canary')['dispatcher_init']
    assert '--release-server-after' not in canary                      # the full run follows on the same server
    assert all(c['repeats'] == 1 for c in L.build(C, 'qwen35-4b', 'canary')['suites'])


def test_shared_server_is_never_released(tmp_path):
    C = campaign(tmp_path)
    C.models['qwen35-4b']['shared_server'] = True
    assert '--release-server-after' not in L.build(C, 'qwen35-4b', 'full')['dispatcher_init']


def test_server_waves_largest_first_and_stagger(tmp_path, monkeypatch):
    C = campaign(tmp_path)
    assert L.server_order(C, ['qwen35-4b', 'ds-max', 'qwen38-xhigh', 'glm53-flash', 'gemma26']) == \
        [['ds-max', 'glm53-flash'], ['qwen38-xhigh'], ['qwen35-4b', 'gemma26']]
    calls, status = [], {'1': '1', '2': '1', '3': '1'}
    jobs = iter(['1', '2', '3'])

    def runner(argv, **kw):
        calls.append(argv)
        if argv[0] == 'condor_q' and '-constraint' in argv:          # live_server: none
            return types.SimpleNamespace(returncode=0, stdout='', stderr='')
        if argv[0] == 'condor_q':
            return types.SimpleNamespace(returncode=0, stdout=status[argv[1]], stderr='')
        return types.SimpleNamespace(returncode=0, stdout=f'job {next(jobs)}.0\n', stderr='')

    def sleep(s):
        status['1'] = '2'                                              # the 4-GPU server starts running
    L.servers(C, ['qwen35-4b', 'ds-max'], 10, False, runner=runner, sleep=sleep)
    submits = [c for c in calls if c[0] != 'condor_q']
    assert [c[c.index('--name') + 1] for c in submits] == ['c99-ds-max', 'c99-qwen35-4b']
    i_ds, i_4b = calls.index(submits[0]), calls.index(submits[1])
    assert any(c[0] == 'condor_q' and c[1] == '1' for c in calls[i_ds:i_4b])   # waited for the 4-GPU wave
    assert '--mem' not in ' '.join(submits[0])                          # serving RAM from the aa-serving spec


def test_decode_bins_and_ratio():
    lines = [f'(APIServer) INFO 11-11 21:{i:02d}:00 [loggers.py] Engine 000: Avg prompt throughput: 10.0 tokens/s, '
             f'Avg generation throughput: {tps} tokens/s, Running: {run} reqs, Waiting: 0 reqs'
             for i, (tps, run) in enumerate([(100, 3), (1500, 12), (1400, 10), (2000, 20), (2200, 22)])]
    meas = L.decode_by_bin(lines)
    assert meas == {'1-8': 100.0, '9-16': 1500.0, '17-24': 2200.0}
    ref = {'1-8': 900, '9-16': 1500, '17-24': 2200}
    assert L.decode_ratio(meas, ref) == pytest.approx(1.0)               # 1-8 skipped
    slow = {k: v * 0.45 for k, v in meas.items()}
    assert L.decode_ratio(slow, ref) == pytest.approx(0.45) and L.decode_ratio(slow, ref) < L.DECODE_MIN_RATIO
    assert L.decode_ratio(meas, {}) is None


def test_check_gates_on_canary_done_infra_and_decode(tmp_path, monkeypatch):
    C = campaign(tmp_path)
    d = C.dispatch_dir('qwen35-4b', 'canary')
    d.mkdir(parents=True)
    lanes = {f'{s}.x': {'done': True} for s in L.SUITES}
    (d / 'state.json').write_text(json.dumps({'lanes': lanes, 'in_flight': 0}))
    (d / 'events.jsonl').write_text(''.join(json.dumps({'ev': 'end', 'status': 'ok'}) + '\n' for _ in range(40)))
    reg = tmp_path / 'serving'
    monkeypatch.setattr(L, 'REG', reg)
    (reg / 'c99-qwen35-4b').mkdir(parents=True)
    ref = C.models['qwen35-4b']['ref_decode_per_gpu']
    log = [f'x Avg generation throughput: {ref["9-16"]} tokens/s, Running: 12 reqs' for _ in range(5)]
    (reg / 'c99-qwen35-4b' / 'vllm.log').write_text('\n'.join(log))
    r = L.check(C, 'qwen35-4b')
    assert r['ok'] and r['decode_ratio'] == pytest.approx(1.0)
    (reg / 'c99-qwen35-4b' / 'vllm.log').write_text('\n'.join(x.replace(str(ref['9-16']), str(ref['9-16'] * 0.4))
                                                              for x in log))
    r = L.check(C, 'qwen35-4b')
    assert not r['ok'] and 'decode' in r['reasons'][0]
    (d / 'events.jsonl').write_text(''.join(json.dumps({'ev': 'end', 'status': 'infra:stale_claim'}) + '\n'
                                            for _ in range(5)))
    lanes['tb21.x']['done'] = False
    (d / 'state.json').write_text(json.dumps({'lanes': lanes}))
    reasons = L.check(C, 'qwen35-4b')['reasons']
    assert any('not done' in x for x in reasons) and any('infra' in x for x in reasons)


def test_example_config_loads_and_every_model_builds():
    C = L.Campaign(EXAMPLE)
    if not C.plan.exists():
        pytest.skip('plan file not on this machine')
    for m in C.models:
        assert L.build(C, m, 'full')['attempts'] > 0


# ------------------------------------------------------------------------------------------------ capped protocol
# /data/aai/m5/ladder/eval/capped_protocol.json (capped-sft-eval-v1, 2026-11-15)
PROTOCOL = {'automationbench': {'warn': 36623, 'allowance': 7000, 'kind': 'pi'},
            'gdpval': {'warn': 75609, 'allowance': 17000, 'kind': 'pi'},
            'critpt': {'warn': 124142, 'allowance': 23000, 'kind': 'pi'},
            'tb21': {'warn': 111905, 'allowance': 17000, 'kind': 'pi'},
            'tbsci_lite': {'warn': 112698, 'allowance': 23000, 'kind': 'pi'},
            'gdppdf': {'warn': 26610, 'allowance': 5000, 'kind': 'single', 'near_window': True},
            'omniscience': {'warn': 20192, 'allowance': 10000, 'kind': 'single', 'near_window': True},
            'hle': {'warn': 130850, 'allowance': 32000, 'kind': 'single', 'near_window': True},
            'scicode': {'warn': 25775, 'allowance': 7000, 'kind': 'single', 'near_window': True, 'unit': 'per step'}}


def protocol_file(tmp_path, name='capped.json', **over):
    suites = {k: dict(v) for k, v in PROTOCOL.items()}
    for k, v in over.items():
        if v is None:
            suites.pop(k)
        else:
            suites[k].update(v)
    f = tmp_path / name
    f.write_text(json.dumps({'policy': 'test', 'suites': suites}))
    return f


def expected_json(suite):
    x = PROTOCOL[suite]
    b = {'warn_after_response_tokens': x['warn'], 'finish_response_tokens': x['allowance'],
         'total_response_tokens': x['warn'] + x['allowance'], 'token_counting': 'super'}
    if x['kind'] == 'single':
        b['near_window'] = True
    if suite == 'omniscience':
        b['notice_extra'] = ('If you are not confident in an answer, say that you do not know: that is a valid final '
                             'answer, and it is better than a wrong one.')
    return json.dumps(b, separators=(',', ':'))


def no_budget_anywhere(b):
    assert 'capped' not in b
    for c in b['suites']:
        assert 'env' not in c and 'response_budget' not in c, c['suite']
        assert '--harbor-env' not in c['argv'] and not any('PI_RESPONSE_BUDGET' in a for a in c['argv']), c['suite']


def test_capped_off_leaves_every_command_unchanged(tmp_path):
    """example_c30.json has no `capped`: no budget anywhere, and a campaign-level file that every model opts out of
    with `"capped": null` builds exactly the same plan."""
    C = campaign(tmp_path)
    assert 'capped' not in C.cfg and not any('capped' in m for m in C.models.values())
    off = {(m, st): L.build(C, m, st) for m in C.models for st in ('canary', 'full')}
    for b in off.values():
        no_budget_anywhere(b)
    cfg = json.loads((tmp_path / 'c.json').read_text())
    cfg['capped'] = str(protocol_file(tmp_path))
    for m in cfg['models'].values():
        m['capped'] = None
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    C2 = L.Campaign(tmp_path / 'c.json')
    assert {(m, st): L.build(C2, m, st) for m in C2.models for st in ('canary', 'full')} == off


def test_capped_pi_suites_get_the_exact_budget_as_harbor_env(tmp_path):
    from aa_evals.common.pi_eval import port, run as pi_run
    from aa_evals.tb21 import run as tb21
    from aa_evals.tbsci import pi as tbsci
    f = protocol_file(tmp_path)
    # the uncapped plan at the capped protocol's model-time caps (#531 B6), so only the budget args differ
    off = L.build(campaign(tmp_path, suites=wall(L.CAPPED_WALL_CAP_SEC)), 'qwen35-4b', 'full')
    b = L.build(campaign(tmp_path, capped=str(f)), 'qwen35-4b', 'full')
    offc = {c['suite']: c for c in off['suites']}
    for c in b['suites']:
        if c['suite'] not in L.PI_SUITES:
            continue
        a = c['argv']
        i = a.index('--harbor-env')
        assert a[i + 1] == 'PI_RESPONSE_BUDGET=' + expected_json(c['suite'])
        assert a[:i] + a[i + 2:] == offc[c['suite']]['argv']                 # only the two args added
        assert 'env' not in c and json.dumps(c['response_budget'], separators=(',', ':')) == expected_json(c['suite'])
        # what pi_eval parses (the wrapper's defaults, or port.base_args + everything after `--`): the budget joins
        # the wrapper's own harbor env (HARBOR_NS_MEMCAP) instead of replacing it
        if c['suite'] == 'tb21':
            d, pa = tb21.DEFAULTS, a[3:]
        elif c['suite'] == 'tbsci_lite':                                      # tbsci_lite.pi strips --pool first
            k = a.index('--pool')
            d, pa = tbsci.DEFAULTS, a[3:k] + a[k + 2:]
        else:
            assert i > a.index('--')                                          # in the pi_eval part
            d, pa = {}, port.base_args(c['suite'], 'r', 'serving:x', Path('/b.tar'), ['t'], 1, 'ns') + a[a.index('--') + 1:]
        p = pi_run.build_parser(d).parse_args(pa)
        env = {**(d.get('harbor_env') or {}), **dict(x.split('=', 1) for x in p.harbor_env)}   # pi_eval/run.py
        assert env['PI_RESPONSE_BUDGET'] == expected_json(c['suite']) and 'HARBOR_NS_MEMCAP' in env, c['suite']


def test_capped_single_turn_suites_get_the_budget_in_the_submitter_env(tmp_path):
    from tb_agents import response_budget as B
    f = protocol_file(tmp_path)
    off = {c['suite']: c for c in L.build(campaign(tmp_path, suites=wall(L.CAPPED_WALL_CAP_SEC)), 'qwen35-4b',
                                         'canary')['suites']}
    b = L.build(campaign(tmp_path, capped=str(f)), 'qwen35-4b', 'canary')
    for c in b['suites']:
        if c['suite'] in L.PI_SUITES:
            continue
        assert c['argv'] == off[c['suite']]['argv']           # the command is the uncapped one at the 12-h caps
        assert c['env'] == {'PI_RESPONSE_BUDGET': expected_json(c['suite'])}
        p = B.settings(c['env'])                                              # what the harness reads
        assert p['near_window'] and p['token_counting'] == 'super'
        assert ('notice_extra' in p) == (c['suite'] == 'omniscience')
    omni = next(c for c in b['suites'] if c['suite'] == 'omniscience')['response_budget']
    assert omni['notice_extra'] == B.NOTICE_EXTRA['omniscience']
    assert b['capped']['file'] == str(f) and len(b['capped']['sha256']) == 64
    assert set(b['capped']['budgets']) == set(L.SUITES)


def test_capped_model_key_overrides_the_campaign(tmp_path):
    f, g = protocol_file(tmp_path), protocol_file(tmp_path, 'other.json', hle={'warn': 1000, 'allowance': 100})
    C = campaign(tmp_path, capped=str(f))
    C.models['gemma26']['capped'] = None                                     # opts out
    C.models['qwen36']['capped'] = str(g)                                    # its own file
    no_budget_anywhere(L.build(C, 'gemma26', 'full'))
    hle = lambda m: next(c for c in L.build(C, m, 'full')['suites'] if c['suite'] == 'hle')['response_budget']
    assert hle('qwen36')['warn_after_response_tokens'] == 1000 and hle('qwen36')['total_response_tokens'] == 1100
    assert hle('qwen35-4b')['warn_after_response_tokens'] == 130850
    C = campaign(tmp_path)                                                   # model-level only
    C.models['qwen36']['capped'] = str(f)
    assert L.build(C, 'qwen36', 'full')['capped']['file'] == str(f)
    no_budget_anywhere(L.build(C, 'qwen35-4b', 'full'))


@pytest.mark.parametrize('over, msg', [({'hle': None}, 'no entry for suite hle'),
                                       ({'critpt': {'allowance': None}}, 'critpt needs positive integer'),
                                       ({'gdppdf': {'warn': 0}}, 'gdppdf needs positive integer'),
                                       ({'scicode': {'kind': 'pi'}}, 'scicode is a single suite')])
def test_capped_protocol_errors_are_loud(tmp_path, over, msg):
    C = campaign(tmp_path, capped=str(protocol_file(tmp_path, **over)))
    with pytest.raises(SystemExit, match=msg):
        L.build(C, 'qwen35-4b', 'full')


def test_launch_sets_the_budget_only_from_the_capped_protocol(tmp_path, monkeypatch):
    """launch() runs each single-turn submitter with its PI_RESPONSE_BUDGET; a value in the launcher's own shell
    never reaches any suite."""
    C = campaign(tmp_path, capped=str(protocol_file(tmp_path)))
    C.dispatch_dir('qwen35-4b', 'full').mkdir(parents=True)
    monkeypatch.setattr(L, 'healthy', lambda r: {'engine_urls': ['http://x:1/v1']})
    monkeypatch.setenv('PI_RESPONSE_BUDGET', '{"stray": 1}')
    runs = {}

    def run(argv, **kw):
        runs[next(s for s in L.SUITES if f'-{s}' in ' '.join(argv))] = kw['env']
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(L.subprocess, 'run', run)
    L.launch(C, 'qwen35-4b', 'full', only=['tb21', 'hle', 'omniscience'])
    assert 'PI_RESPONSE_BUDGET' not in runs['tb21']                          # Pi: in argv only
    assert runs['hle']['PI_RESPONSE_BUDGET'] == expected_json('hle')
    assert runs['omniscience']['PI_RESPONSE_BUDGET'] == expected_json('omniscience')
    ev = [json.loads(x) for x in (C.logdir / 'events.jsonl').read_text().splitlines()]
    assert {e['suite']: e['response_budget']['warn_after_response_tokens'] for e in ev} == \
        {'tb21': 111905, 'hle': 130850, 'omniscience': 20192}
    C = campaign(tmp_path)                                                   # off: the stray value is dropped too
    runs.clear()
    L.launch(C, 'qwen35-4b', 'full', only=['hle'])
    assert 'PI_RESPONSE_BUDGET' not in runs['hle']


def test_server_argv_checkpoint_and_extra(tmp_path):
    """Optional per-model checkpoint / serve_extra reach serve_teacher submit; absent: unchanged."""
    cfg = json.loads(EXAMPLE.read_text())
    model = next(iter(cfg['models']))
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    base = L.server_argv(L.Campaign(tmp_path / 'c.json'), model)
    assert '--checkpoint' not in base
    cfg['models'][model]['checkpoint'] = '/data/x/ckpt'
    cfg['models'][model]['serve_extra'] = ['--memory-gib', 128]
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert L.server_argv(L.Campaign(tmp_path / 'c.json'), model) == base + ['--checkpoint', '/data/x/ckpt', '--memory-gib', '128']


def flag(a, f):
    return a[a.index(f) + 1] if f in a else None


def test_wall_cap_sets_model_time_caps_apart_from_the_pi_environment_clock(tmp_path):
    C = campaign(tmp_path)
    cap = str(int(C.s.get('cap_sec', 7200)))
    unset = {c['suite']: c['argv'] for c in L.build(C, 'qwen35-4b', 'full')['suites']}
    model_time = {'scicode': '--chain-cap-sec', 'hle': '--request-cap-sec', 'omniscience': '--timeout',
                  'gdppdf': '--request-cap-sec'}
    for s, a in unset.items():   # unset: every cap is cap_sec, as before; HLE keeps its default request timeout
        if s in L.PI_SUITES:
            assert flag(a, '--agent-cap-sec') == flag(a, '--agent-wall-cap-sec') == cap, s
        else:
            assert flag(a, model_time[s]) == cap, s
    assert flag(unset['hle'], '--timeout') is None
    cfg = json.loads((tmp_path / 'c.json').read_text())
    cfg['suites']['wall_cap_sec'] = 43200
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    C2 = L.Campaign(tmp_path / 'c.json')
    for c in L.build(C2, 'qwen35-4b', 'full')['suites']:
        a, s = c['argv'], c['suite']
        if s in L.PI_SUITES:   # the environment clock (model time excluded) stays; only the wall clock moves
            assert flag(a, '--agent-cap-sec') == cap and flag(a, '--agent-wall-cap-sec') == '43200', s
        else:
            assert flag(a, model_time[s]) == '43200', s
        rest = [x for x in a if x not in (cap, '43200')]
        assert rest == [x for x in unset[s] if x not in (cap, '43200')] or s == 'hle', s
    hle = next(c['argv'] for c in L.build(C2, 'qwen35-4b', 'full')['suites'] if c['suite'] == 'hle')
    assert flag(hle, '--timeout') == '43200'   # one capped HLE call can outlast the default 7200 s client timeout
    cfg['suites']['wall_cap_sec'] = 3600
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match='below cap_sec'):
        L.build(L.Campaign(tmp_path / 'c.json'), 'qwen35-4b', 'full')


def test_wall_cap_default_depends_on_the_protocol(tmp_path):
    """#531 B6: unset suites.wall_cap_sec -> 12 h under the capped protocol (token budgets bind; time caps are a hang
    backstop), cap_sec (2 h, #344) without it; an explicit value wins in both; a model that opts out of `capped`
    keeps 2 h. The Pi environment clock is cap_sec throughout."""
    f = protocol_file(tmp_path)
    cap = str(int(campaign(tmp_path).s.get('cap_sec', 7200)))
    caps = {'scicode': '--chain-cap-sec', 'hle': '--request-cap-sec', 'omniscience': '--timeout',
            'gdppdf': '--request-cap-sec'}

    def walls(C, model='qwen35-4b'):
        out = {}
        for c in L.build(C, model, 'full')['suites']:
            a = c['argv']
            if c['suite'] in L.PI_SUITES:
                assert flag(a, '--agent-cap-sec') == cap, c['suite']
                out[c['suite']] = flag(a, '--agent-wall-cap-sec')
            else:
                out[c['suite']] = flag(a, caps[c['suite']])
        return set(out.values())
    assert L.CAPPED_WALL_CAP_SEC == 43200 == L.MAX_WALL_CAP_SEC
    assert walls(campaign(tmp_path)) == {cap}
    assert walls(campaign(tmp_path, capped=str(f))) == {'43200'}
    assert walls(campaign(tmp_path, capped=str(f), suites=wall(None))) == {'43200'}   # null = the default
    assert walls(campaign(tmp_path, suites=wall(None))) == {cap}
    with pytest.raises(SystemExit, match='above 43200'):
        walls(campaign(tmp_path, suites=wall(50000)))
    assert walls(campaign(tmp_path, capped=str(f), suites=wall(14400))) == {'14400'}
    assert walls(campaign(tmp_path, suites=wall(14400))) == {'14400'}
    cfg = json.loads((tmp_path / 'c.json').read_text())
    cfg.update(capped=str(f), suites=json.loads(EXAMPLE.read_text())['suites'])
    cfg['models']['qwen35-4b']['capped'] = None
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert walls(L.Campaign(tmp_path / 'c.json')) == {cap}
    cfg.pop('capped')                                       # capped through the model's own key only
    cfg['models']['qwen35-4b']['capped'] = str(f)
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert walls(L.Campaign(tmp_path / 'c.json')) == {'43200'}


def test_auto_budget_false_fixes_the_dispatcher_budget(tmp_path):
    # #531 B7: a fixed budget is the default (#451 decision 3; user 11-16: the auto budget pushed the 3B to 499)
    assert '--no-auto-budget' in L.build(campaign(tmp_path), 'qwen35-4b', 'full')['dispatcher_init']
    assert '--no-auto-budget' not in L.build(campaign(tmp_path, auto_budget=True), 'qwen35-4b',
                                             'full')['dispatcher_init']   # opt-in
    C = campaign(tmp_path, auto_budget=False)
    for st in ('canary', 'full'):
        assert '--no-auto-budget' in L.build(C, 'qwen35-4b', st)['dispatcher_init']
    C.models['qwen35-4b']['auto_budget'] = True   # a model key overrides the campaign
    assert '--no-auto-budget' not in L.build(C, 'qwen35-4b', 'full')['dispatcher_init']


def test_model_suites_subset_builds_only_those_lanes(tmp_path):
    C = campaign(tmp_path)
    assert L.model_suites(C, 'qwen35-4b') == L.SUITES
    C.models['qwen35-4b']['suites'] = ['hle', 'tb21', 'scicode']
    assert L.model_suites(C, 'qwen35-4b') == ['tb21', 'scicode', 'hle']   # SUITES order
    b = L.build(C, 'qwen35-4b', 'full')
    assert [c['suite'] for c in b['suites']] == ['tb21', 'scicode', 'hle']
    init = b['dispatcher_init']
    assert init[init.index('--expect-lanes') + 1] == '3'
    for bad in ([], ['tb21', 'nope']):
        C.models['qwen35-4b']['suites'] = bad
        with pytest.raises(SystemExit, match='non-empty subset'):
            L.build(C, 'qwen35-4b', 'full')


# ------------------------------------------------------------------------------------------------ request_overrides
SAMPLING_FLAGS = ('--request-overrides', '--temperature', '--top-p')


def with_overrides(tmp_path, model, ro):
    cfg = json.loads(EXAMPLE.read_text())
    over = {**cfg['models'], model: {**cfg['models'][model], 'request_overrides': ro}}
    return campaign(tmp_path, models=over)


class Parsed(Exception):
    pass


def client_args(module, argv, monkeypatch):
    """The namespace the suite's own client parses from `argv` (its main(), stopped right after parse_args)."""
    import argparse
    orig = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        raise Parsed(orig(self, args, namespace))
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', grab)
    try:
        module.main(argv)
    except Parsed as e:
        return e.args[0]
    finally:
        monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', orig)
    raise AssertionError(f'{module.__name__} never parsed its arguments')


def test_request_overrides_off_leaves_every_command_unchanged(tmp_path):
    C = campaign(tmp_path)
    assert not any('request_overrides' in m for m in C.models.values())
    for m in C.models:
        for st in ('canary', 'full'):
            b = L.build(C, m, st)
            assert 'request_overrides' not in b
            for c in b['suites']:
                assert 'request_overrides' not in c and not set(SAMPLING_FLAGS) & set(c['argv']), (m, c['suite'])
    assert L.build(with_overrides(tmp_path, 'qwen35-4b', {}), 'qwen35-4b', 'full') == L.build(C, 'qwen35-4b', 'full')


def test_request_overrides_reach_every_suite_through_its_own_client(tmp_path, monkeypatch):
    """A temperature-only eval needs no wrapper: Pi suites get pi_eval --request-overrides, single-turn suites
    --temperature / --top-p, each parsed by the suite's real client; nothing else in any command changes, and other
    models are untouched."""
    from aa_evals.common.pi_eval import port, run as pi_run
    from aa_evals.gdppdf import run as gdppdf
    from aa_evals.hle import run as hle
    from aa_evals.omniscience import run as omni
    from aa_evals.scicode import run as scicode
    from aa_evals.tb21 import run as tb21
    from aa_evals.tbsci import pi as tbsci
    ro = {'temperature': 0.6, 'top_p': 0.9}
    off = {c['suite']: c for c in L.build(campaign(tmp_path), 'qwen35-4b', 'full')['suites']}
    C = with_overrides(tmp_path, 'qwen35-4b', ro)
    b = L.build(C, 'qwen35-4b', 'full')
    assert b['request_overrides'] == ro and {c['suite'] for c in b['suites']} == set(L.SUITES)
    for c in b['suites']:
        a, s = c['argv'], c['suite']
        assert c['request_overrides'] == ro
        if s in L.PI_SUITES:
            i = a.index('--request-overrides')
            assert a[i + 1] == '{"temperature":0.6,"top_p":0.9}'
            assert a[:i] + a[i + 2:] == off[s]['argv'], s                      # only the two args added
            if s == 'tb21':
                d, pa = tb21.DEFAULTS, a[3:]
            elif s == 'tbsci_lite':
                k = a.index('--pool')
                d, pa = tbsci.DEFAULTS, a[3:k] + a[k + 2:]
            else:
                assert i > a.index('--'), s                                    # in the pi_eval part
                d, pa = {}, port.base_args(s, 'r', 'serving:x', Path('/b.tar'), ['t'], 1, 'ns') + a[a.index('--') + 1:]
            assert json.loads(pi_run.build_parser(d).parse_args(pa).request_overrides) == ro, s
        else:
            assert a == off[s]['argv'] + ['--temperature', '0.6', '--top-p', '0.9'], s
            # what each wrapper hands its run.py: scicode runs it directly; omniscience/submit.sh passes everything
            # after the run id; hle.submit and gdppdf.submit everything after `--`, each with --run-id
            run_id = ['--run-id', c['run_id']]
            if s == 'scicode':
                client, ca = scicode, a[3:]
            elif s == 'omniscience':
                client, ca = omni, run_id + a[3:]
            else:
                client, ca = {'hle': hle, 'gdppdf': gdppdf}[s], run_id + a[a.index('--') + 1:]
            got = client_args(client, ca, monkeypatch)
            assert (got.temperature, got.top_p) == (0.6, 0.9), s
    other = next(m for m in C.models if m != 'qwen35-4b')
    assert L.build(C, other, 'full') == L.build(campaign(tmp_path), other, 'full')


def test_launch_logs_the_request_overrides(tmp_path, monkeypatch):
    C = with_overrides(tmp_path, 'qwen35-4b', {'temperature': 0.6})
    C.dispatch_dir('qwen35-4b', 'full').mkdir(parents=True)
    monkeypatch.setattr(L, 'healthy', lambda r: {'engine_urls': ['http://x:1/v1']})
    argvs = {}

    def run(argv, **kw):
        argvs[next(s for s in L.SUITES if f'-{s}' in ' '.join(argv))] = argv
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(L.subprocess, 'run', run)
    L.launch(C, 'qwen35-4b', 'full', only=['tb21', 'hle'])
    assert argvs['tb21'][argvs['tb21'].index('--request-overrides') + 1] == '{"temperature":0.6}'
    assert argvs['hle'][-2:] == ['--temperature', '0.6']
    ev = [json.loads(x) for x in (C.logdir / 'events.jsonl').read_text().splitlines()]
    assert {e['suite']: e['request_overrides'] for e in ev} == {'tb21': {'temperature': 0.6}, 'hle': {'temperature': 0.6}}


@pytest.mark.parametrize('ro, msg', [({'top_k': 20}, r"\['top_k'\] not supported"),
                                     ({'temperature': '0.6'}, 'temperature must be a finite number'),
                                     ({'temperature': True}, 'temperature must be a finite number'),
                                     ({'temperature': -0.1}, 'temperature must be a finite number'),
                                     ({'temperature': float('nan')}, 'temperature must be a finite number'),
                                     ({'top_p': float('inf')}, 'top_p must be a finite number'),
                                     ({'top_p': 0}, r'0 < top_p <= 1'),
                                     ({'top_p': 1.5}, r'0 < top_p <= 1'),
                                     ([0.6], 'must be an object'), ('', 'must be an object'), (0, 'must be an object'),
                                     (False, 'must be an object'), ([], 'must be an object')])
def test_request_overrides_errors_are_loud(tmp_path, ro, msg):
    with pytest.raises(SystemExit, match=msg):
        L.build(with_overrides(tmp_path, 'qwen35-4b', ro), 'qwen35-4b', 'full')


def test_request_overrides_at_campaign_level_and_opt_out(tmp_path, capsys):
    """As `capped`: the campaign's key applies to every model, a model's own key wins, and null opts a model out."""
    cfg = json.loads(EXAMPLE.read_text())
    models = list(cfg['models'])
    over = {**cfg['models'], models[0]: {**cfg['models'][models[0]], 'request_overrides': {'top_p': 0.9}},
            models[1]: {**cfg['models'][models[1]], 'request_overrides': None}}
    C = campaign(tmp_path, models=over, request_overrides={'temperature': 0.6})
    assert L.request_overrides(C, models[0]) == {'top_p': 0.9}
    assert L.request_overrides(C, models[1]) == {}
    assert all(L.request_overrides(C, m) == {'temperature': 0.6} for m in models[2:])
    off = campaign(tmp_path)
    assert L.build(C, models[1], 'full') == L.build(off, models[1], 'full')
    L.show(C, [models[0], models[1]], 'full')
    out = capsys.readouterr().out
    assert out.count('  sampling: ') == 1 and '  sampling: {"top_p": 0.9}' in out


def test_tb_toolchain_bundle_reaches_tb21_only(tmp_path):
    """#531 B5: suites.tb_toolchain_bundle (example: tb21_6h for bundle-aptfix.tar) -> tb21's
    --verifier-toolchain-bundle, which pi_eval records as config verifier_toolchain_bundle; unset: as before."""
    from aa_evals.common.pi_eval import run as pi_run
    from aa_evals.tb21 import run as tb21
    C = campaign(tmp_path)
    assert C.s['tb_toolchain_bundle'] == ['tb21_6h', C.s['tb_bundle'][1]]   # pinned to the validated bundle
    for c in L.build(C, 'qwen35-4b', 'full')['suites']:
        a = c['argv']
        assert (flag(a, '--verifier-toolchain-bundle') == 'tb21_6h') is (c['suite'] == 'tb21'), c['suite']
        if c['suite'] == 'tb21':
            assert pi_run.build_parser(tb21.DEFAULTS).parse_args(a[3:]).verifier_toolchain_bundle == 'tb21_6h'
    cfg = json.loads((tmp_path / 'c.json').read_text())
    cfg['suites'].pop('tb_toolchain_bundle')
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    tb = next(c['argv'] for c in L.build(L.Campaign(tmp_path / 'c.json'), 'qwen35-4b', 'full')['suites']
              if c['suite'] == 'tb21')
    assert '--verifier-toolchain-bundle' not in tb
    assert pi_run.build_parser(tb21.DEFAULTS).parse_args(tb[3:]).verifier_toolchain_bundle is None


def test_tb_toolchain_bundle_pin_refuses_another_bundle(tmp_path):
    cfg_suites = json.loads(EXAMPLE.read_text())['suites']
    for bad, msg in ((dict(cfg_suites, tb_toolchain_bundle=['tb21_6h', '0' * 64]), 'validated for tb_bundle'),
                     (dict(cfg_suites, tb_toolchain_bundle='tb21_6h'), r'\[toolchain bundle stem')):
        with pytest.raises(SystemExit, match=msg):
            L.build(campaign(tmp_path, suites=bad), 'qwen35-4b', 'full')


def test_worker_profile_slow_puts_tb21_on_its_slow_shape(tmp_path):
    """#531 A/B: model worker_profile 'slow' -> tb21 gets --shape-key tb21_slow (6 CPU / 96 GB at 32 slots, pi_eval
    resolves it); other suites and models without the key are unchanged; an unknown profile is refused."""
    from aa_evals.common.pi_eval import run as pi_run
    from aa_evals.tb21 import run as tb21
    base = {c['suite']: c['argv'] for c in L.build(campaign(tmp_path), 'qwen35-4b', 'full')['suites']}
    cfg = json.loads((tmp_path / 'c.json').read_text())
    cfg['models']['qwen35-4b']['worker_profile'] = 'slow'
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    slow = {c['suite']: c['argv'] for c in L.build(L.Campaign(tmp_path / 'c.json'), 'qwen35-4b', 'full')['suites']}
    for suite, argv in slow.items():
        if suite == 'tb21':
            assert flag(argv, '--shape-key') == 'tb21_slow'
            assert [x for x in argv if x not in ('--shape-key', 'tb21_slow')] == base[suite]
            a = pi_run.build_parser(tb21.DEFAULTS).parse_args(argv[3:])
            assert a.shape_key == 'tb21_slow'
            assert pi_run.worker_shape(a.shape_key, pi_run.default_slots(a.shape_key, None), None, None) == (6, 64)
        else:
            assert argv == base[suite], suite
    other = next(m for m in cfg['models'] if m != 'qwen35-4b')
    assert all('--shape-key' not in c['argv']   # other models: no profile, no flag
               for c in L.build(L.Campaign(tmp_path / 'c.json'), other, 'full')['suites'])
    cfg['worker_profile'] = 'slow'                      # campaign-level, with a model-level null opt-out
    cfg['models']['qwen35-4b']['worker_profile'] = None
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    C2 = L.Campaign(tmp_path / 'c.json')
    assert all('--shape-key' not in c['argv'] for c in L.build(C2, 'qwen35-4b', 'full')['suites'])
    assert flag(next(c['argv'] for c in L.build(C2, other, 'full')['suites'] if c['suite'] == 'tb21'),
                '--shape-key') == 'tb21_slow'
    cfg['models']['qwen35-4b']['worker_profile'] = 'fast'
    cfg['models']['qwen35-4b']['suites'] = ['hle']       # validated even when the model runs no tb21
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match='worker_profile'):
        L.build(L.Campaign(tmp_path / 'c.json'), 'qwen35-4b', 'full')


# ------------------------------------------------------------------------------------------------ AAI-open version
REPO_CAPPED = Path(__file__).with_name('capped_protocol_v1.json')


def pinned(tmp_path, monkeypatch, **over):
    """The example campaign, with its test plan standing in for the pinned cheap_full_v1 (only the plan bytes differ)."""
    import hashlib
    C = campaign(tmp_path, **over)
    monkeypatch.setattr(L.aai_open, 'PLAN_SHA256', hashlib.sha256(C.plan.read_bytes()).hexdigest())
    return C


def test_aai_open_the_example_config_is_the_current_version(tmp_path, monkeypatch):
    """example_c30.json (the leaderboard template) builds 'v0.1' for every model and stage, every suite carries it,
    and its real plan is the pinned cheap_full_v1 when the file is reachable."""
    import hashlib
    plan_pin = L.aai_open.PLAN_SHA256
    C = pinned(tmp_path, monkeypatch)
    for m in C.models:
        for st in ('canary', 'full'):
            b = L.build(C, m, st)
            assert b['aai_open'] == {'version': 'v0.1', 'deviations': []}, (m, st, b['aai_open'])
            assert {c['aai_open'] for c in b['suites']} == {'v0.1'}
    plan = Path(json.loads(EXAMPLE.read_text())['plan'])
    if plan.exists():
        assert hashlib.sha256(plan.read_bytes()).hexdigest() == plan_pin
    monkeypatch.setattr(L.aai_open, 'PLAN_SHA256', plan_pin)
    assert L.version(campaign(tmp_path), 'qwen36', 'full')['version'] == 'v0.1+custom'   # another plan


def test_aai_open_capped_protocol_file_is_pinned(tmp_path, monkeypatch):
    """The repo copy of capped-sft-eval-v1 is the pinned file (and loads); another capped file, or a model-time cap
    other than the protocol's (12 h capped, cap_sec uncapped), makes the run custom."""
    import hashlib
    assert hashlib.sha256(REPO_CAPPED.read_bytes()).hexdigest() == L.aai_open.CAPPED_SHA256
    assert set(L.capped_budgets(REPO_CAPPED)) == set(L.SUITES)
    assert L.version(pinned(tmp_path, monkeypatch, capped=str(REPO_CAPPED)), 'qwen36', 'full')['version'] == 'v0.1'
    v = L.version(pinned(tmp_path, monkeypatch, capped=str(protocol_file(tmp_path))), 'qwen36', 'full')
    assert v['version'] == 'v0.1+custom' and v['deviations'][0].startswith('capped protocol')
    for capped, w, ok in ((None, 7200, True), (None, 43200, False), (str(REPO_CAPPED), 43200, True),
                          (str(REPO_CAPPED), 7200, False)):
        over = {'suites': wall(w), **({'capped': capped} if capped else {})}
        v = L.version(pinned(tmp_path, monkeypatch, **over), 'qwen36', 'full')
        assert (v['version'] == 'v0.1') == ok, (capped, w, v)


@pytest.mark.parametrize('key, value', [('context_window', 262144), ('cap_sec', 3600),
                                        ('tb_toolchain_bundle', None), ('tb_bundle', ['/x/other.tar', 'f' * 64]),
                                        ('tb_cache', ['/x/m.json', 'e' * 64]), ('lite_cache', ['/x/m.json', 'd' * 64]),
                                        ('scicode_dataset', 'original'), ('lite_pool', 'v2.0')])
def test_aai_open_suite_deviations_make_the_run_custom(tmp_path, monkeypatch, key, value):
    s = json.loads(EXAMPLE.read_text())['suites']
    if value is None:
        s.pop(key)
    else:
        s[key] = value
    v = L.version(pinned(tmp_path, monkeypatch, suites=s), 'qwen36', 'full')
    assert v['version'] == 'v0.1+custom' and len(v['deviations']) == 1 and key in v['deviations'][0], v


def test_aai_open_pins_apply_only_to_the_suites_a_model_runs(tmp_path, monkeypatch):
    s = {k: v for k, v in json.loads(EXAMPLE.read_text())['suites'].items() if k != 'tb_toolchain_bundle'}
    C = pinned(tmp_path, monkeypatch, suites=s)
    C.models['qwen36']['suites'] = ['hle', 'omniscience']          # a lean model without TB2.1
    assert L.version(C, 'qwen36', 'full')['version'] == 'v0.1'
    assert L.version(C, 'gemma26', 'full')['version'] == 'v0.1+custom'


def test_aai_open_request_overrides_make_the_run_custom(tmp_path, monkeypatch):
    C = pinned(tmp_path, monkeypatch, request_overrides={'temperature': 0.6})
    C.models['gemma26']['request_overrides'] = None                # opts out: the version's sampling
    v = L.version(C, 'qwen36', 'full')
    assert v['version'] == 'v0.1+custom' and 'request_overrides' in v['deviations'][0]
    assert L.version(C, 'gemma26', 'full')['version'] == 'v0.1'


def test_launch_runs_every_suite_with_the_label_never_the_shells(tmp_path, monkeypatch):
    C = pinned(tmp_path, monkeypatch)
    C.dispatch_dir('qwen35-4b', 'full').mkdir(parents=True)
    monkeypatch.setattr(L, 'healthy', lambda r: {'engine_urls': ['http://x:1/v1']})
    monkeypatch.setenv('AAI_OPEN_VERSION', 'v9')
    runs = {}

    def run(argv, **kw):
        runs[next(s for s in L.SUITES if f'-{s}' in ' '.join(argv))] = kw['env']
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(L.subprocess, 'run', run)
    L.launch(C, 'qwen35-4b', 'full', only=['tb21', 'hle'])
    assert runs['tb21']['AAI_OPEN_VERSION'] == runs['hle']['AAI_OPEN_VERSION'] == 'v0.1'
    ev = [json.loads(x) for x in (C.logdir / 'events.jsonl').read_text().splitlines()]
    assert {e['suite']: e['aai_open'] for e in ev} == {'tb21': 'v0.1', 'hle': 'v0.1'}


def test_aai_open_grading_judge_and_gdpval_refs_are_pinned(tmp_path, monkeypatch):
    C = pinned(tmp_path, monkeypatch, grading={'judge': {'arm': 'qwen38-27b-nothink'}})
    v = L.version(C, 'qwen36', 'full')
    assert v['version'] == 'v0.1+custom' and v['deviations'][0].startswith('grading.judge.arm')
    C = pinned(tmp_path, monkeypatch, grading={'gdpval_refs': ['c30-qwen38-xhigh-gdpval']})
    assert L.version(C, 'qwen36', 'full')['deviations'][0].startswith('grading.gdpval_refs')
    C.models['qwen36']['suites'] = ['hle']                         # refs only matter where GDPval runs
    assert L.version(C, 'qwen36', 'full')['version'] == 'v0.1'
    C = pinned(tmp_path, monkeypatch, grading={'judge': {'arm': L.aai_open.JUDGE_ARM}, 'gdpval_refs': L.aai_open.GDPVAL_REFS})
    assert L.version(C, 'qwen36', 'full')['version'] == 'v0.1'


def test_launch_logs_the_campaign_level_deviations(tmp_path, monkeypatch):
    C = pinned(tmp_path, monkeypatch, request_overrides={'temperature': 0.6})
    C.dispatch_dir('qwen35-4b', 'full').mkdir(parents=True)
    monkeypatch.setattr(L, 'healthy', lambda r: {'engine_urls': ['http://x:1/v1']})
    monkeypatch.setattr(L.subprocess, 'run', lambda argv, **kw: types.SimpleNamespace(returncode=0, stdout='', stderr=''))
    L.launch(C, 'qwen35-4b', 'full', only=['hle'])
    e = json.loads((C.logdir / 'events.jsonl').read_text().splitlines()[-1])
    assert e['aai_open'] == 'v0.1+custom' and e['aai_open_deviations'][0].startswith('request_overrides')


def test_aai_open_null_gdpval_refs_do_not_crash(tmp_path, monkeypatch):
    C = pinned(tmp_path, monkeypatch, grading={'gdpval_refs': None})
    assert L.version(C, 'qwen36', 'full')['version'] == 'v0.1+custom'


# ------------------------------------------------------------------------------------------------ SFT example, sets
SFT_EXAMPLE = Path(__file__).with_name('example_sft_snowball.json')
PROFILE = Path(__file__).with_name('expected') / 'sft_snowball.json'


def test_suites_may_name_a_set(tmp_path):
    C = campaign(tmp_path)
    C.models['qwen35-4b']['suites'] = 'lean'
    assert L.model_suites(C, 'qwen35-4b') == ['tb21', 'gdpval', 'automationbench', 'scicode', 'omniscience', 'gdppdf']
    b = L.build(C, 'qwen35-4b', 'full')
    assert flag(b['dispatcher_init'], '--expect-lanes') == '6'
    C.models['qwen35-4b']['suites'] = 'full'
    assert L.model_suites(C, 'qwen35-4b') == L.SUITES
    C.models['qwen35-4b']['suites'] = 'tiny'
    with pytest.raises(SystemExit, match='lean'):
        L.build(C, 'qwen35-4b', 'full')


def test_sft_example_renders_main_defaults(tmp_path):
    """example_sft_snowball.json: snowball-eval-main on 2xB200 at a fixed 256, capped protocol, TB2.1 offline verifier
    toolchain and slow worker shape, default sampling (no request overrides), lean = 6 lanes; the expected profile's
    suite sets are the launcher's."""
    cfg = json.loads(SFT_EXAMPLE.read_text())
    assert 'aai_open_version' not in cfg and cfg['tbsched_shadow'] is False   # the launcher computes the label (#481)
    assert cfg['capped'].endswith('/{campaign}-main/aa_evals/campaign/capped_protocol_v1.json')
    m = cfg['models']['sft']
    assert (m['teacher'], m['gpus'], m['gpu'], m['bid'], m['budget'], m['suites']) == \
        ('snowball-eval-main', 2, 'B200', 61, 256, 'lean')
    assert 'request_overrides' not in cfg and 'request_overrides' not in m
    assert json.loads(PROFILE.read_text())['suite_sets'] == L.SUITE_SETS
    plan = {'suites': {s: {'items': [{'id': f'{s}-{i}', 'repeats': 2} for i in range(5)]} for s in L.SUITES}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    cfg.update({'name': 'sft99', 'plan': str(tmp_path / 'plan.json'), 'dispatch_root': str(tmp_path / 'dispatch'),
                'capped': str(protocol_file(tmp_path))})
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    C = L.Campaign(tmp_path / 'c.json')
    assert C.cfg['code'].endswith('/sft99-main') and C.models['sft']['registry'] == 'sft99-sft'
    b = L.build(C, 'sft', 'full')
    assert [c['suite'] for c in b['suites']] == ['tb21', 'gdpval', 'automationbench', 'scicode', 'omniscience', 'gdppdf']
    assert flag(b['server'], '--teacher') == 'snowball-eval-main' and flag(b['server'], '--checkpoint')
    init = b['dispatcher_init']
    assert flag(init, '--budget') == '256' and '--no-auto-budget' in init and flag(init, '--release-server-after') == '600'
    tb = next(c['argv'] for c in b['suites'] if c['suite'] == 'tb21')
    assert flag(tb, '--shape-key') == 'tb21_slow' and flag(tb, '--verifier-toolchain-bundle') == 'tb21_6h'
    assert flag(tb, '--agent-wall-cap-sec') == '43200' and flag(tb, '--agent-cap-sec') == '7200'   # capped: 12-h backstop
    for c in b['suites']:
        assert not any(x in c['argv'] for x in SAMPLING_FLAGS), c['suite']
        assert c.get('response_budget'), c['suite']
    cfg['models']['sft']['suites'] = 'full'
    (tmp_path / 'c.json').write_text(json.dumps(cfg))
    assert len(L.build(L.Campaign(tmp_path / 'c.json'), 'sft', 'full')['suites']) == 9


def test_sft_example_on_this_machine():
    """The example itself with the real plan and this checkout's capped protocol: lean 2,217 attempts, full 2,805,
    and AAI-open v0.1 with no deviation for either set (the launcher's label, #481)."""
    C = L.Campaign(SFT_EXAMPLE)
    C.cfg['capped'] = str(Path(__file__).with_name('capped_protocol_v1.json'))   # the {campaign}-main checkout's copy
    if not C.plan.exists() or not L.capped_file(C, 'sft').exists():
        pytest.skip('plan or capped protocol not on this machine')
    assert L.capped(C, 'sft')['sha256'].startswith('403ad91eb358')
    for sset, n in (('lean', 2217), ('full', 2805)):
        C.models['sft']['suites'] = sset
        b = L.build(C, 'sft', 'full')
        assert b['attempts'] == n
        assert b['aai_open'] == {'version': 'v0.1', 'deviations': []}, b['aai_open']


def shadow_campaign(tmp_path, on=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    C = campaign(tmp_path, tbsched_shadow=on)
    d = C.dispatch_dir('qwen35-4b', 'full')
    for suite, kind in (('tb21', 'pi'), ('gdpval', 'pi'), ('hle', 'lane')):
        run = tmp_path / 'runs' / suite
        run.mkdir(parents=True)
        (run / 'config.json').write_text(json.dumps({'worker': {'slots': 32 if suite == 'tb21' else 16}}))
        lane = d / 'lanes' / f'{suite}.x'
        lane.mkdir(parents=True)
        (lane / 'registration.json').write_text(json.dumps({'suite': suite, 'kind': kind, 'run': str(run)}))
    return C


def test_tbsched_shadow_config_is_read_only(tmp_path):
    C = shadow_campaign(tmp_path)
    cfg = L.sched_config(C, 'qwen35-4b', 'full')
    assert cfg['mode'] == 'shadow' and set(cfg['runs']) == {'tb21', 'gdpval'}       # Pi lanes only
    assert all(r['hooked'] is False and r['kind'] == 'agentic' for r in cfg['runs'].values())
    assert cfg['runs']['tb21']['slots'] == 32 and cfg['runs']['tb21']['groups'] == {'c99-qwen35-4b': 1.0}
    g = cfg['groups']['c99-qwen35-4b']
    assert g['mixed'] is True and g['registry'].endswith('/c99-qwen35-4b')
    started = []

    class P:
        pid = 4242

    def popen(argv, **kw):
        started.append((argv, kw))
        return P()
    C.models['qwen35-4b']['suites'] = ['tb21', 'gdpval', 'hle']
    d = L.start_shadow(C, 'qwen35-4b', 'full', wait_sec=0, popen=popen)
    assert d == L.shadow_dir(C, 'qwen35-4b', 'full') and d.name == 'c99-qwen35-4b-sched'
    assert json.loads((d / 'sched' / 'config.json').read_text()) == cfg
    argv, kw = started[0]
    assert argv[:4] == ['timeout', str(16 * 3600), 'bash', '-c'] and 'tbsched.loop' in argv[4]
    assert argv[-2:] == [str(C.dispatch_dir('qwen35-4b', 'full')), str(d)]
    assert kw['start_new_session'] and (d / 'loop.pid').read_text().startswith('4242 ')
    (tmp_path / 'e').mkdir()
    empty = campaign(tmp_path / 'e', tbsched_shadow=True)
    assert L.start_shadow(empty, 'qwen35-4b', 'full', wait_sec=0, popen=popen) is None and len(started) == 1


def test_tbsched_shadow_starts_only_when_on_and_only_for_the_full_stage(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(L, 'start_shadow', lambda C, m, stage, **kw: calls.append(stage))
    monkeypatch.setattr(L, 'healthy', lambda r: {'engine_urls': ['http://x:1/v1']})
    def run(argv, **kw):   # the dispatcher's init makes its dir
        if 'init' in argv and '--budget' in argv:
            Path(argv[argv.index('init') + 1]).mkdir(parents=True)
        return types.SimpleNamespace(returncode=0, stdout='', stderr='')
    monkeypatch.setattr(L.subprocess, 'run', run)
    monkeypatch.setattr(L.subprocess, 'Popen', lambda argv, **kw: types.SimpleNamespace(pid=7, poll=lambda: None))
    monkeypatch.setattr(L.time, 'sleep', lambda s: None)
    for on in (False, True):
        C = shadow_campaign(tmp_path / str(on), on=on)
        L.launch(C, 'qwen35-4b', 'full', only=['hle'])             # a resubmit (--only): never
        assert calls == []
        L.show(C, ['qwen35-4b'], 'full')
        assert ('tbsched: shadow' in capsys.readouterr().out) is on
        (tmp_path / f'first{on}').mkdir()
        C = campaign(tmp_path / f'first{on}', tbsched_shadow=on)
        for stage in ('canary', 'full'):                          # a first launch: full stage only, when on
            L.launch(C, 'qwen35-4b', stage)
        assert calls == (['full'] if on else [])
        calls.clear()
    monkeypatch.setattr(L, 'start_shadow', lambda *a, **kw: (_ for _ in ()).throw(OSError('lanes unreadable')))
    (tmp_path / 'err').mkdir()
    C = campaign(tmp_path / 'err', tbsched_shadow=True)
    L.launch(C, 'qwen35-4b', 'full')                              # a failing shadow never fails the launch
    ev = [json.loads(x) for x in (C.logdir / 'events.jsonl').read_text().splitlines()]
    assert any(e['event'] == 'tbsched_shadow_failed' for e in ev)
    assert L.shadow_on(C, 'qwen35-4b')
    C.models['qwen35-4b']['tbsched_shadow'] = False                # the model's key wins
    assert not L.shadow_on(C, 'qwen35-4b')
