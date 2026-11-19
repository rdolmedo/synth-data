"""Worker job shape and bid defaults (#378): CPUs from slots x the benchmark's CPUs per slot, RAM 16 GB per CPU,
the CPU bid ladder unless --cpu-bid is given."""
from pathlib import Path

import pytest

pytest.importorskip('harbor')
from aa_evals.common.pi_eval import run as R   # noqa: E402


@pytest.mark.parametrize('bench,want', [('tb21', (32, 10, 128)), ('tbsci_lite', (4, 4, 64)),
                                        ('tbsci_lite_synth', (16, 8, 96)),
                                        ('critpt_pi', (24, 4, 32)), ('gdpval_pi', (16, 2, 24)),
                                        ('automationbench_pi', (32, 4, 32)), ('other', (24, 12, 192))])
def test_default_slots_and_shape_from_the_measurements(bench, want):
    slots = R.default_slots(bench, None)
    assert (slots, *R.worker_shape(bench, slots, None, None)) == want


def test_shape_scales_with_explicit_slots_and_explicit_values_win():
    assert R.default_slots('tb21', 16) == 16
    assert R.worker_shape('tb21', 16, None, None) == (7, 64)     # fixed overhead: more CPU per slot when small
    assert R.worker_shape('tb21', 16, 32, 256) == (32, 256)
    assert R.worker_shape('other', 24, 6, None) == (6, 96)       # unknown bench: 16 GB per CPU
    assert R.worker_shape('tb21', 32, None, 40) == (10, 40)


def test_parser_defaults_shape_unset_and_bid_ladder():
    a = R.build_parser({'bench': 'gdpval_pi'}).parse_args(['--run-id', 'x'])
    assert a.worker_cpus is None and a.worker_mem_gb is None and a.cpu_bid is None and a.slots is None


def test_runs_root_is_on_fast():
    assert str(R.RUNS_ROOT).startswith('/fast/') or 'AA_RUNS_ROOT' in __import__('os').environ


def test_submit_workers_uses_ladder_and_records_each_cluster(tmp_path, monkeypatch):
    calls = []

    def fake_submit(sub, bid=None, **kw):
        calls.append((Path(sub).read_text(), bid, kw))
        return [{'cluster': '11', 'bid': 26, 'n': 3, 'submit_file': str(sub)},
                {'cluster': '12', 'bid': 36, 'n': 2, 'submit_file': str(sub)}]
    monkeypatch.setattr(R.condor_cost, 'submit', fake_submit)
    run = tmp_path / 'tb21' / 'r1'
    run.mkdir(parents=True)
    cfg = {'code': '/code', 'benchmark': 'tb21', 'run_id': 'r1',
           'worker': {'n': 5, 'slots': 16, 'cpus': 12, 'mem_gb': 192, 'disk_gb': 300, 'bid': None}}
    assert R.submit_workers(run, cfg, 5) == '12'
    text, bid, kw = calls[0]
    assert bid is None and 'request_cpus = 12' in text and 'request_memory = 196608' in text and 'queue 5' in text
    assert kw['exclude'] == R.exclude_nodes(R.EXCLUDE_NODES)       # the shared list read now + this module's extras
    assert set(R.condor_cost.bad_nodes()) <= set(kw['exclude'])
    import json
    jobs = json.loads((run / 'jobs.json').read_text())
    assert [(j['cluster'], j['n'], j['bid']) for j in jobs] == [('11', 3, 26), ('12', 2, 36)]
    cfg['worker']['bid'] = 61                      # a run created with a fixed bid keeps it on --resubmit
    R.submit_workers(run, cfg, 1)
    assert calls[-1][1] == 61


def test_initial_workers_launch_small_only_with_dispatcher_growth(tmp_path):
    import json
    d = tmp_path / 'disp'
    d.mkdir()
    assert R.initial_workers(252, 80, 16) == 5                                  # no dispatcher: min(items, max in flight)
    (d / 'dispatch.json').write_text(json.dumps({'budget': 132}))
    assert R.initial_workers(252, 80, 16, str(d)) == 5                          # dispatcher without growth: unchanged
    (d / 'dispatch.json').write_text(json.dumps({'budget': 132, 'pi_autoscale': True}))
    assert R.initial_workers(252, 80, 16, str(d)) == 3                          # 25% of 132 = 33 slots
    assert R.initial_workers(10, 80, 16, str(d)) == 1


def test_shapes_come_from_pipeline_json_and_match_their_cpu_model():
    from tbshared import pipeline_config
    cfg = pipeline_config.load()
    assert cfg['pipeline'] == 'aa-evals' and pipeline_config.pipeline_name() == 'aa-evals'
    for bench, s in cfg['worker_shapes'].items():                # the stored CPUs are the model at the default slots
        assert R.worker_shape(bench, s['slots'], None, None) == (s['cpus'], s['mem_gb'])
        assert s['basis'] and s['cpu_model']['target'] == 0.8


def test_submitted_pi_workers_are_filled_where_free(tmp_path):
    """tbshared 1.5 fills the worker shape at submission (condor_cost.fill_plan): RAM to 16 GB per CPU, same units."""
    from aa_evals.common import condor_cost as C
    gb = 1024
    status = f'c01\t15\t400\t{8000 * gb}\t0\t0\t512\t{8100 * gb}\tundefined'   # a roomy floor-15 node
    want = {'tb21': (10, 160), 'tbsci_lite': (4, 64), 'critpt_pi': (4, 64), 'gdpval_pi': (2, 32),
            'automationbench_pi': (4, 64)}
    for bench, (c, g) in want.items():
        slots = R.default_slots(bench, None)
        cpus, mem = R.worker_shape(bench, slots, None, None)
        body = f'request_cpus = {cpus}\nrequest_memory = {mem * 1024}\n'
        out, plan, rec = C.fill_plan(body, 2, C.LADDER, status=status, where=str(tmp_path / f'{bench}.sub'),
                                     log=lambda m: None)
        assert C.parse_shape(out)[:2] == (c, float(g)) and plan == [(26, 2)], bench
        assert C.units(cpus, mem) == C.units(c, g)                                     # units unchanged


@pytest.mark.parametrize('bench,key', [
    ('m5l_tbsci_lite_t2', 'tbsci_lite'), ('m5l_tbsci_lite_finish', 'tbsci_lite'), ('m5l_gdpval_finish', 'gdpval_pi'),
    ('m5l_critpt_t2fin', 'critpt_pi'), ('m5_critpt', 'critpt_pi'), ('gdpval_m5', 'gdpval_pi'),
    ('m5_automationbench', 'automationbench_pi'), ('m5l_automationbench_t2', 'automationbench_pi'),
    ('critpt', 'critpt_pi'), ('tb21', 'tb21'), ('m5l_tb_finish', 'tb21'), ('m5l2_gdpval_t2', 'gdpval_pi'),
    ('tbsci', None), ('other', None)])
def test_derived_bench_names_take_their_base_suites_shape(bench, key):
    # #451 lesson 5: bench_shape() returned None for m5l_<suite>_t2 / _finish, so they got the 24-slot default
    from tbshared import pipeline_config
    shapes = pipeline_config.load()['worker_shapes']
    assert R.shape_key(bench, shapes) == key
    assert R.bench_shape(bench) == (shapes[key] if key else None)
    if key:
        slots = R.default_slots(bench, None)
        assert (slots, *R.worker_shape(bench, slots, None, None)) == (shapes[key]['slots'], shapes[key]['cpus'],
                                                                      shapes[key]['mem_gb'])


def test_explicit_shapes_far_from_the_measured_one_are_warned():
    # M5's launchers passed pre-#378 shapes: GDPval 16 CPUs / 96 GB for 16 slots (measured: 2 / 24)
    c = R.shape_check('gdpval_m5', 16, 16, 96)
    assert c['shape_key'] == 'gdpval_pi' and c['measured'] == {'cpus': 2, 'mem_gb': 24}
    assert len(c['warnings']) == 2 and '--worker-cpus 16 is 8.00x' in c['warnings'][0]
    assert R.shape_check('tbsci_lite', 16, 8, 96)['warnings'][0].startswith('--worker-cpus 8 is 0.53x')   # too small
    assert R.shape_check('gdpval_pi', 16, 3, 24)['warnings'] == []        # within 1.5x
    assert R.shape_check('gdpval_pi', 16, None, None)['warnings'] == []   # nothing explicit
    other = R.shape_check('other', 24, 64, 999)                           # no measured shape: said so, not compared
    assert other['shape_key'] is None and other['measured'] is None and 'no measured worker shape' in other['warnings'][0]


def test_launchers_pass_no_worker_cpus_or_ram():
    # resources come from pipeline.json; a launcher may still pass slots, disk and bid (#451 lesson 5)
    import re
    repo = Path(R.__file__).resolve().parents[3]
    literal = re.compile(r"""['"]--worker-(cpus|mem-gb)['"],\s*(['"]\d|str\((max|\d))""")
    default = re.compile(r"""add_argument\(['"]--worker-(cpus|mem-gb)['"], type=int, default=\d""")
    bad = []
    for top in ('m5', 'aa_evals'):
        for p in sorted((repo / top).rglob('*.py')):
            if {'pi_eval', 'harbor_mswea'} & set(p.parts) or p.name.startswith('test_'):   # the runners own defaults
                continue
            for i, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                if literal.search(line) or default.search(line):
                    bad.append(f'{p.relative_to(repo)}:{i}: {line.strip()}')
    assert not bad, '\n'.join(bad)


def test_shape_key_flag():
    a = R.build_parser({'bench': 'tbsci_lite'}).parse_args(['--run-id', 'x', '--shape-key', 'tbsci_lite_synth'])
    assert a.shape_key == 'tbsci_lite_synth'
    assert R.worker_shape(a.shape_key, R.default_slots(a.shape_key, None), None, None) == (8, 96)
    for key in ('nope', 'tbsci_lite_pi', 'm5_tbsci_lite'):   # exact entries only: no derived-name matching
        with pytest.raises(SystemExit, match='no such worker_shapes entry'):
            R.main(['--bench', 'tbsci_lite', '--run-id', 'x', '--shape-key', key, '--status'])
