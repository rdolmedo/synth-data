"""tbshared.pipeline_config: the pipeline-local tbshared/pipeline.json (schema, errors, defaults)."""
import json

import pytest

from tbshared import pipeline_config as pc

GOOD = {'pipeline': 'aa-evals',
        'worker_shapes': {'tb21': {'slots': 32, 'cpus': 10, 'mem_gb': 128, 'basis': 'c30 fit + #378 memrun',
                                   'cpu_model': {'p95_full_slot': 13.25, 'source': 'memrun/report.md'}},
                          'gdpval_pi': {'slots': 16, 'cpus': 2, 'mem_gb': 24, 'basis': '#378 memrun'}}}


def write(tmp_path, obj):
    p = tmp_path / 'pipeline.json'
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return p


def test_load_and_accessors(tmp_path):
    p = write(tmp_path, GOOD)
    assert pc.load(p) == GOOD
    assert pc.pipeline_name(p) == 'aa-evals'
    s = pc.worker_shape('tb21', path=p)
    assert (s['slots'], s['cpus'], s['mem_gb']) == (32, 10, 128) and s['cpu_model']['p95_full_slot'] == 13.25
    s['cpus'] = 99                                                      # a copy: callers cannot mutate the config
    assert pc.worker_shape('tb21', path=p)['cpus'] == 10
    assert pc.worker_shape('nope', path=p) is None
    assert pc.worker_shape('nope', default={'slots': 1}, path=p) == {'slots': 1}


def test_worker_shapes_are_optional(tmp_path):
    p = write(tmp_path, {'pipeline': 'eval'})
    assert pc.load(p) == {'pipeline': 'eval', 'worker_shapes': {}}
    assert pc.worker_shape('tb', default=7, path=p) == 7


def test_missing_file(tmp_path):
    p = tmp_path / 'pipeline.json'
    assert pc.worker_shape('tb21', default='fallback', path=p) == 'fallback'
    with pytest.raises(pc.PipelineConfigError, match='missing'):
        pc.pipeline_name(p)
    with pytest.raises(pc.PipelineConfigError, match='missing'):
        pc.load(p)


@pytest.mark.parametrize('obj,msg', [
    ('{"pipeline": "x",', 'not valid JSON'),
    ([], 'JSON object'),
    ({}, '"pipeline"'),
    ({'pipeline': ' '}, '"pipeline"'),
    ({'pipeline': 'x', 'worker_shape': {}}, 'unknown keys'),                     # typo of worker_shapes
    ({'pipeline': 'x', 'worker_shapes': []}, 'worker_shapes'),
    ({'pipeline': 'x', 'worker_shapes': {'a': 3}}, "worker_shapes\\['a'\\] must be an object"),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1}}}, 'basis'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1, 'basis': 'a\nb'}}}, 'basis'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1.5, 'mem_gb': 1, 'basis': 'b'}}}, 'cpus'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 0, 'cpus': 1, 'mem_gb': 1, 'basis': 'b'}}}, 'slots'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': True, 'mem_gb': 1, 'basis': 'b'}}}, 'cpus'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'basis': 'b'}}}, 'mem_gb'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1, 'basis': 'b', 'gpus': 1}}},
     'unknown keys'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1, 'basis': 'b', 'cpu_model': 3}}},
     'cpu_model'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1, 'basis': 'b',
                                               'cpu_model': {'p95_full_slot': -1}}}}, 'p95_full_slot'),
    ({'pipeline': 'x', 'worker_shapes': {'a': {'slots': 1, 'cpus': 1, 'mem_gb': 1, 'basis': 'b',
                                               'cpu_model': {'source': 3}}}}, 'source'),
])
def test_malformed_files_raise_a_clear_error(tmp_path, obj, msg):
    p = write(tmp_path, obj)
    with pytest.raises(pc.PipelineConfigError, match=msg) as e:
        pc.load(p)
    assert str(p) in str(e.value)                                       # the error names the file
    with pytest.raises(pc.PipelineConfigError):
        pc.worker_shape('a', default='never a silent fallback', path=p)


def test_default_path_is_next_to_the_module():
    assert pc.PATH.name == 'pipeline.json' and pc.PATH.parent.name == 'tbshared'


def test_find_and_load_follow_the_caller(tmp_path):
    """#480: the trunk's per-package files, found from the calling module (or `start`)."""
    r = tmp_path / 'trunk'
    (r / '.git').mkdir(parents=True)
    (r / 'aa_evals').mkdir()
    write(r / 'aa_evals', GOOD)
    (r / 'eval_q35').mkdir()
    write(r / 'eval_q35', {'pipeline': 'eval'})
    assert pc.find(start=r / 'aa_evals' / 'common' / 'x.py') == r / 'aa_evals' / 'pipeline.json'
    assert pc.load(start=r / 'eval_q35' / 'submit.py') == {'pipeline': 'eval', 'worker_shapes': {}}
    ns = {'pc': pc}
    exec(compile("NAME = pc.pipeline_name()\nSHAPE = pc.worker_shape('tb21')\n", str(r / 'aa_evals' / 'run.py'),
                 'exec'), ns)
    assert ns['NAME'] == 'aa-evals' and ns['SHAPE']['cpus'] == 10
    with pytest.raises(pc.PipelineConfigError, match='no pipeline.json governs'):
        pc.load(start=tmp_path / 'elsewhere' / 'x.py')
    assert pc.worker_shape('tb21', default='d', start=tmp_path / 'elsewhere' / 'x.py') == 'd'
