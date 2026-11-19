"""Numerics guard (numerics.py; README "Numerics guard").

In a pipeline branch whose tbshared/pipeline.json names one of numerics.PIPELINES, the NUMERICS fields extracted
from the repository must equal that pipeline's section of tbshared/numerics_record.json: a change to a weight, dtype,
kernel flag, template, parser, sampling default, output cap or runtime fails here, listing every field. On the trunk
(tbshared/TRUNK, #480) every pipeline a package's pipeline.json names is checked (numerics.trunk_pipelines), and the
differences main is known to have (numerics.TRUNK_KNOWN_GAPS, each with its reason) are reported apart: anything
else fails. Skipped on branch `shared`, in pipelines without pipeline.json (not opted in) or without a record
section, and outside a git worktree (frozen snapshots). The other tests check the extractor on small synthetic
fixtures."""
import json
import os
import shutil
import subprocess
import textwrap
import warnings
from pathlib import Path

import pytest

from tbshared import numerics, pipeline_config, sync

HERE = Path(__file__).resolve().parent


def _root():
    if shutil.which('git') is None:
        return None
    r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=HERE, capture_output=True, text=True)
    return Path(r.stdout.strip()) if r.returncode == 0 else None


@pytest.mark.parametrize('name', numerics.PIPELINES)
def test_pipeline_numerics_match_the_record(name):
    root = _root()
    if root is None or root != HERE.parent:
        pytest.skip('not inside a git worktree (a frozen code snapshot, or no git): checked where the copy was made')
    if sync.is_shared_tree(root):
        pytest.skip('branch `shared` itself: no pipeline to check')
    if sync.is_trunk(root):
        if name not in numerics.trunk_pipelines(root):
            pytest.skip(f'trunk: no package pipeline.json names {name!r}')
    else:
        if not pipeline_config.PATH.exists():
            pytest.skip('no tbshared/pipeline.json: this pipeline has not opted in to the numerics guard')
        here = pipeline_config.pipeline_name(pipeline_config.PATH)
        if here not in numerics.PIPELINES:
            pytest.skip(f'pipeline {here!r} has no numerics extractor (numerics.PIPELINES)')
        if name != here:
            pytest.skip(f'this branch is {here!r}')
    if numerics.tomllib is None and name != 'eval':
        pytest.skip('Python < 3.11 (no tomllib): the TOML specs cannot be read here')
    specs = {}
    d, host = numerics.check_full(name, root, extraction=specs)
    if sync.is_trunk(root):
        d, known = numerics.known_trunk_gaps(name, d, specs=specs)
        if known:
            reasons = sorted({why for _, why in known})
            warnings.warn(f'{name}: {len(known)} known trunk differences (numerics.TRUNK_KNOWN_GAPS):\n  '
                          + '\n  '.join(reasons), stacklevel=1)
    if host:   # a node without /data (or a moved venv) must never fail a pipeline: say what was not checked
        warnings.warn(f'{name}: {len(host)} host-derived numerics fields not checked (source unreadable here):\n  '
                      + '\n  '.join(host[:20]) + ('\n  ...' if len(host) > 20 else ''), stacklevel=1)
    assert not d, (f'{name}: serving numerics differ from tbshared/numerics_record.json:\n' + '\n'.join(d)
                   + f'\n{numerics.CHANGE_HINT}')


# ---------------------------------------------------------------------------------------------------------------------
# Fixtures: a minimal serve_teacher pipeline and a minimal eval_q35
# ---------------------------------------------------------------------------------------------------------------------
def toml_dumps(d: dict, prefix='') -> str:
    """Enough TOML for the fixtures: scalars and lists inline, dicts as [tables] (inline inside lists)."""
    def val(v):
        if isinstance(v, bool):
            return 'true' if v else 'false'
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return '[' + ', '.join(val(x) for x in v) + ']'
        return '{ ' + ', '.join(f'{json.dumps(k)} = {val(x)}' for k, x in v.items()) + ' }'
    lines = [f'{json.dumps(k)} = {val(v)}' for k, v in d.items() if not isinstance(v, dict)]
    for k, v in d.items():
        if isinstance(v, dict):
            name = f'{prefix}.{json.dumps(k)}' if prefix else json.dumps(k)
            lines += ['', f'[{name}]', toml_dumps(v, name)]
    return '\n'.join(lines)


BASE_ARGS = [
    '{model_dir}', '--served-model-name', '{served_name}', '--port', '{port}', '--max-model-len', '{max_model_len}',
    '--gpu-memory-utilization', '0.9', '--max-num-seqs', '{max_seqs}', '--enable-prefix-caching',
    '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder', '--enable-auto-tool-choice',
    '--generation-config', 'vllm', '--seed', '{seed}',
    '--override-generation-config', '{{"temperature":1.0,"top_p":0.95,"max_new_tokens":{max_output_tokens},"stop_token_ids":[7]}}',
    '--default-chat-template-kwargs', '{{"enable_thinking":true}}', '--chat-template', '{repo}/templates/t.jinja',
    '--trust-remote-code', '--some-new-flag', 'x', '--attention-config.indexer_kv_dtype=fp8',
]


def teacher(slug='foo', kv='fp8', mm=('--limit-mm-per-prompt', '{{"image":2}}'), args=(), tunables=None, env=None,
            runtime=None, resources=None, optional=None, entry=None, ckpt='/nonexistent/foo', venv='/nonexistent/venv'):
    return {'slug': slug, 'model': 'org/Foo', 'revision': 'abc123', 'checkpoint': ckpt, 'served_name': 'foo',
            'quantization': 'fp8/e4m3',
            'runtime': {'venv': venv, 'vllm_version': '0.1.0', 'cuda_home': 'cu13-wheel',
                        'env': {'HF_HUB_OFFLINE': '1', 'VLLM_USE_V2_MODEL_RUNNER': '1', **(env or {})}, **(runtime or {})},
            'serving': {'mode': 'replicas', 'entrypoint': entry or ['{venv}/bin/vllm', 'serve'],
                        'args': [*BASE_ARGS, '--kv-cache-dtype', kv, *mm, *args], 'seed': 3,
                        'tunables': {'mtp': 0, 'max_seqs': 64, **(tunables or {})},
                        'optional_args': {'mtp': ['--speculative-config',
                                                  '{{"method":"mtp","num_speculative_tokens":{mtp}}}'],
                                          **(optional or {})}},
            'request': {'pi_thinking': 'high', 'overrides': {'temperature': 1.0,
                                                             'chat_template_kwargs': {'enable_thinking': True}}},
            'serving_resources': {'cpus_per_gpu': 8, **(resources or {})}}


SERVE_TEACHER = """
# this serve_teacher honours: {features}
PROTOCOL = {{'context_tokens': 1000, 'max_output_tokens': 500}}
PROFILES = {{
    'foo': {{'protocol': {{'context_tokens': 2000, 'max_output_tokens': 2000}}, 'tunables': {profile_tunables},
            'gate': {{'policy': {{}}}}}},
}}
TEXT_ONLY_VARIANTS = ('foo',)
PROFILES.update({{f'{{slug}}-text': PROFILES[slug] for slug in TEXT_ONLY_VARIANTS}})
def serve():
    raise SystemExit('never run by the extractor')
"""

VLLM_LAUNCH = """
from pathlib import Path
def cuda_home_from_wheel(venvs, target):
    raise FileNotFoundError('the extractor must stub this (it links a venv into scratch)')
def replica_command(teacher, protocol, *, port, model_dir, tunables, rank_ports=None):
    serving, runtime = teacher['serving'], teacher['runtime']
    values = {'venv': runtime['venv'], 'model_dir': model_dir, 'served_name': teacher['served_name'], 'port': port,
              'seed': serving.get('seed', 0), 'max_model_len': protocol['context_tokens'],
              'max_output_tokens': protocol['max_output_tokens'], 'checkpoint': teacher['checkpoint'],
              **(rank_ports or {}), **serving.get('tunables', {}), **tunables}
    argv = [x.format(**values) for x in serving['entrypoint']] + [x.format(**values) for x in serving['args']]
    for key, extra in serving.get('optional_args', {}).items():
        if tunables.get(key):
            argv += [x.format(**values) for x in extra]
    return argv
def replica_env(teacher, *, gpus, scratch, tunables, base_env=None):
    runtime = teacher['runtime']
    env = dict(base_env or {})
    if runtime.get('cuda_visible_devices') == 'index':   # as aa-serving 89d85472: a lazy relative import
        from .gpu import indices
        env['CUDA_VISIBLE_DEVICES'] = ','.join(indices(gpus))
    if runtime.get('cuda_home') == 'cu13-wheel':
        env['CUDA_HOME'] = str(cuda_home_from_wheel([runtime['venv']], Path(scratch) / 'cuda_home')[0])
    env.update(runtime.get('env', {}))
    env.update({k: str(v).format(**tunables) for k, v in teacher['serving'].get('env', {}).items()})
    env.update(tunables.get('extra_env') or {})
    env['LAUNCHER_MARK'] = 'from the branch replica_env'
    return env
"""


def write(r, rel, text):
    p = r / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def toml_repo(tmp_path, specs=None, features='', profile_tunables='{"mtp": 2}', launch=True, campaigns=None):
    r = tmp_path / 'repo'
    write(r, 'templates/t.jinja', '{{ messages }}')
    write(r, 'task_synthesis/serve/serve_teacher.py',
          SERVE_TEACHER.format(features=features or 'nothing extra', profile_tunables=profile_tunables))
    write(r, 'synth/config.py', "PROTOCOL_DEFAULTS = {'context_tokens': 900, 'max_output_tokens': 100, "
                                "'finish_notice': 'v4'}\n")
    if launch:
        write(r, 'synth/vllm_launch.py', VLLM_LAUNCH)
    for t in specs or [teacher(), teacher('foo-text', mm=('--language-model-only',))]:
        write(r, f'synth/teachers/{t["slug"]}.toml', toml_dumps(t))
    for name, c in (campaigns or {}).items():
        write(r, f'synth/campaigns/{name}.toml', toml_dumps(c))
    return r


def git_commit(r):
    for cmd in (['init', '-q'], ['add', '-A'], ['-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x']):
        subprocess.run(['git', '-C', str(r), *cmd], check=True, capture_output=True)


needs_toml = pytest.mark.skipif(numerics.tomllib is None, reason='Python < 3.11: no tomllib')


@needs_toml
def test_toml_extraction_keeps_numerics_only(tmp_path):
    r = toml_repo(tmp_path, launch=False)
    full = numerics.extract_full('aa-serving', r)
    assert full['skipped'] == [] and set(full['specs']) == {'foo', 'foo-text'}
    assert any('vllm_launch.py missing' in w for w in full['warnings'])     # no synth/vllm_launch.py: built-in copy
    f = full['specs']['foo']
    assert f['kv_cache_dtype'] == 'fp8' and f['mamba_ssm_cache_dtype'] == 'auto' and f['quantization'] == 'none'
    assert f['max_model_len'] == '2000'                                      # the PROFILES protocol, not PROTOCOL
    assert (f['output_cap.tokens'], f['output_cap.kind'].split()[0]) == (2000, 'hard')
    assert f['stop_token_ids'] == [7] and f['sampling.temperature'] == 1.0 and 'sampling.max_new_tokens' not in f
    assert f['chat_template_kwargs.enable_thinking'] is True and f['seed'] == '3'
    assert f['adjacent.speculative.method'] == 'mtp' and f['adjacent.speculative.num_speculative_tokens'] == 2
    assert f['image_limits'] == {'image': 2} and f['trust_remote_code'] is True
    assert f['chat_template.path'] == '{repo}/templates/t.jinja'             # never the checkout's own path
    assert f['chat_template.path.sha256'] == numerics.hashlib.sha256(b'{{ messages }}').hexdigest()
    assert f['env.VLLM_USE_V2_MODEL_RUNNER'] == '1' and 'env.HF_HUB_OFFLINE' not in f
    assert f['args.--some-new-flag'] == 'x'                                  # unknown flag: counted as numerics
    assert f['attention_config.indexer_kv_dtype'] == 'fp8'
    assert (f['weights.model'], f['weights.revision'], f['weights.declared_quantization']) == (
        'org/Foo', 'abc123', 'fp8/e4m3')
    assert f['host.weights'].startswith('unreadable') and f['vllm.entrypoint'] == ['{venv}/bin/vllm', 'serve']
    assert f['request.pi_thinking'] == 'high' and f['request.overrides.chat_template_kwargs.enable_thinking'] is True
    for throughput in ('--gpu-memory-utilization', '--max-num-seqs', '--enable-prefix-caching', '--port'):
        assert f'args.{throughput}' not in f
    t = full['specs']['foo-text']
    assert t['image_limits'].startswith('text only') and t['max_model_len'] == '2000'   # the derived -text profile
    assert full['sources']['foo-text']['profile'].startswith('task_synthesis/serve/serve_teacher.py:')
    git_commit(r)                       # the same tree read from git (`record --ref COMMIT`) extracts the same
    assert numerics.extract('aa-serving', r, ref='HEAD') == full['specs']


@needs_toml
def test_pipeline_profiles_of_the_one_launcher(tmp_path):
    """#480: the trunk's serve_teacher holds every pipeline's profiles; a pipeline's own entry replaces the PROFILES
    entry for that pipeline only, and a launcher that stages stage_extra raises no placeholder warning."""
    r = toml_repo(tmp_path, specs=[teacher(), teacher('bar', resources={'cpus_per_gpu': 4})])
    st = r / 'task_synthesis/serve/serve_teacher.py'
    st.write_text(st.read_text() + """
PIPELINE_PROFILES = {
    'task-synthesis': {'foo': {'protocol': {'context_tokens': 3000, 'max_output_tokens': 3000}, 'tunables': {'mtp': 4}}},
}
""")
    aa, ts = numerics.extract_full('aa-serving', r), numerics.extract_full('task-synthesis', r)
    assert aa['specs']['foo']['max_model_len'] == '2000' and aa['specs']['foo']['adjacent.speculative.num_speculative_tokens'] == 2
    assert ts['specs']['foo']['max_model_len'] == '3000' and ts['specs']['foo']['adjacent.speculative.num_speculative_tokens'] == 4
    assert ts['specs']['bar'] == aa['specs']['bar']                       # not named by the pipeline: PROFILES / PROTOCOL
    assert "PIPELINE_PROFILES['task-synthesis']" in ts['sources']['foo']['profile']


@needs_toml
def test_stage_extra_warning_only_for_launchers_that_do_not_stage(tmp_path):
    spec = teacher('dr', args=('--speculative-config', '{{"model":"{stage_draft}"}}'))
    spec['serving']['stage_extra'] = [{'name': 'draft', 'source': '/nowhere/draft'}]
    r = toml_repo(tmp_path, specs=[spec])
    assert any('stage_extra' in w for w in numerics.extract_full('aa-serving', r)['warnings'])
    st = r / 'task_synthesis/serve/serve_teacher.py'
    st.write_text(st.read_text() + "\ndef serve2():\n    return stage_teacher(None, None)\n")
    full = numerics.extract_full('aa-serving', r)
    assert not any('stage_extra' in w for w in full['warnings'])
    assert full['specs']['dr']['adjacent.speculative.model'] == '{stage_draft}'


def test_fill_template_leaves_strings_and_comments():
    text = ('id = "@ID@"\nnote = """a "q" @X@\n@Y@ b"""\nesc = "x\\"@Z@"\nlit = \'@L@\'\n[train]\nn = @TRACES@   # @C@\n'
            '[smoke]\nattempts = [\n@SMOKE@\n]\n[serving]\njobs = @ENGINES@\n')
    out = numerics.fill_template(text)
    assert 'n = 0   # @C@' in out and 'jobs = 0' in out and '@SMOKE@' not in out
    assert '"""a "q" @X@\n@Y@ b"""' in out and '"x\\"@Z@"' in out and "'@L@'" in out
    if numerics.tomllib:
        raw = numerics.tomllib.loads(out)
        assert raw['id'] == '@ID@' and raw['train'] == {'n': 0} and raw['smoke'] == {'attempts': []}
    assert numerics.fill_template('a = 1\n') == 'a = 1\n'


@needs_toml
def test_campaign_templates_are_rendered_not_skipped(tmp_path):
    """#445: production's campaign template (nemotron35-super-tasksynth-m1.toml) is recorded, not skipped."""
    r = toml_repo(tmp_path)
    write(r, 'synth/campaigns/tmpl.toml', 'id = "@ID@"\nteacher = "foo"\n[train]\nn = @TRACES@\n[serving]\n'
                                          'jobs = @ENGINES@\nbid = @SERVING_BID@\n')
    full = numerics.extract_full('aa-serving', r)
    assert not [x for x in full['skipped'] if 'tmpl.toml' in x]
    f = full['specs']['campaign/tmpl']
    assert f['campaign.template'] is True and f['campaign.teacher'] == 'foo' and f['max_model_len'] == '900'


def test_known_trunk_gaps_explain_only_what_they_name(tmp_path):
    """A campaign's removed specs are explained only when its own file is skipped for tb_agents; an added spec only
    when the other section records it with the same fields, or when it matches its pinned digest; a pattern only what
    it matches; anything else stays unexplained."""
    rec = tmp_path / 'record.json'
    rec.write_text(json.dumps({'schema': 1, 'pipelines': {'aa-serving': {'specs': {'teach-a': {'x': 1}}}}}))
    rules = [('campaigns-unrenderable', None, 'tb'), ('union', 'aa-serving', 'union'), ('pinned', 'pin-me', 'pin'),
             ('re', r'^qwen38-27b: changed weights\.declared_quantization', 'label')]
    diff = ["skipped (could not extract): synth/campaigns/c1.toml (sha256 ab): ImportError(\"cannot import name "
            "'finish_notice_record' from 'x'\")",
            'removed spec campaign/c1', 'removed spec campaign/c1/candidate/k', 'removed spec campaign/c2',
            'added spec teach-a: {}', 'added spec teach-b: {}', 'added spec pin-me: {}',
            'qwen38-27b: changed weights.declared_quantization: null -> "modelopt/MIXED_PRECISION"',
            'deepseek: changed adjacent.speculative.num_speculative_tokens: 3 -> 5']
    pinned = {'pin-me': numerics.spec_digest({'a': 1, 'host.weights': 'other node'})}       # host fields left out
    specs = {'teach-a': {'x': 1}, 'teach-b': {}, 'pin-me': {'a': 1, 'host.weights': 'unreadable'}}
    left, known = numerics.known_trunk_gaps('task-synthesis', diff, rec, specs, rules, pinned)
    assert left == ['removed spec campaign/c2', 'added spec teach-b: {}',
                    'deepseek: changed adjacent.speculative.num_speculative_tokens: 3 -> 5']
    assert len(known) == 6
    left, _ = numerics.known_trunk_gaps('task-synthesis', diff, rec, {**specs, 'teach-a': {'x': 2},
                                                                     'pin-me': {'a': 2}}, rules, pinned)
    assert 'added spec teach-a: {}' in left and 'added spec pin-me: {}' in left
    assert numerics.known_trunk_gaps('eval', diff, rec) == (diff, [])                 # no rules: nothing explained


def test_main_has_no_known_gaps_left():
    """The serve_teacher sections were re-recorded from main (#480): every difference on main fails again."""
    assert numerics.TRUNK_KNOWN_GAPS == {'aa-serving': [], 'task-synthesis': []} and numerics.TRUNK_PINNED == {}


@needs_toml
def test_output_cap_kinds_and_a_dtype_change_are_reported(tmp_path):
    a = numerics.extract('task-synthesis', toml_repo(tmp_path / 'a'))
    t = teacher(kv='fp8_e5m2')
    t['serving']['args'] = [x.replace('max_new_tokens', 'max_tokens') for x in t['serving']['args']]
    b = numerics.extract('task-synthesis', toml_repo(tmp_path / 'b', specs=[t, teacher('foo-text')]))
    assert b['foo']['output_cap.kind'].startswith('default')
    d = numerics.compare(a, b)
    assert 'foo: changed kv_cache_dtype: "fp8" -> "fp8_e5m2"' in d
    assert any(x.startswith('foo: changed output_cap.kind') for x in d)
    assert numerics.compare(a, a) == []


def test_compare_lists_added_removed_and_adjacent_fields():
    want = {'s': {'kv_cache_dtype': 'fp8', 'adjacent.speculative.num_speculative_tokens': 1, 'seed': '0'},
            'gone': {'x': 1}}
    have = {'s': {'kv_cache_dtype': 'fp8', 'adjacent.speculative.num_speculative_tokens': 3, 'quantization': 'fp8'},
            'new': {'x': 1}}
    d = numerics.compare(want, have)
    assert 's: changed adjacent.speculative.num_speculative_tokens (numerics-adjacent): 1 -> 3' in d
    assert 's: removed seed (was "0")' in d and 's: added quantization = "fp8"' in d
    assert 'removed spec gone' in d and any(x.startswith('added spec new') for x in d)


@needs_toml
def test_record_updates_one_section_and_keeps_notes(tmp_path):
    rec = tmp_path / 'record.json'
    rec.write_text(json.dumps({'schema': 1, 'pipelines': {
        'eval': {'specs': {'e': {'x': 1}}},
        'aa-serving': {'specs': {}, 'notes': {'foo': ['MTP change, see #426'], 'gone': ['dropped']}}}}))
    r = toml_repo(tmp_path)
    sec = numerics.record('aa-serving', r, record_path=rec, force=True)
    data = json.loads(rec.read_text())
    assert data['pipelines']['eval'] == {'specs': {'e': {'x': 1}}}         # other pipelines untouched
    assert sec['notes'] == {'foo': ['MTP change, see #426']}               # notes of specs that still exist
    assert numerics.check('aa-serving', r, record_path=rec) == []
    (r / 'synth' / 'teachers' / 'foo-text.toml').unlink()
    assert numerics.check('aa-serving', r, record_path=rec) == ['removed spec foo-text']
    assert numerics.check('task-synthesis', r, record_path=rec) == [
        'pipeline task-synthesis has no section in record.json']


def test_toml_specs_are_skipped_without_tomllib(tmp_path, monkeypatch):
    monkeypatch.setattr(numerics, 'tomllib', None)
    full = numerics.extract_full('aa-serving', tmp_path)
    assert full['specs'] == {} and 'tomllib' in full['skipped'][0]


def test_serve_teacher_is_read_without_importing_it():
    src = SERVE_TEACHER.format(features='', profile_tunables='{}')
    st = numerics.literal_assignments(src, {'PROTOCOL', 'PROFILES', 'TEXT_ONLY_VARIANTS'}, 'st.py')
    assert st['PROFILES']['foo-text'] == st['PROFILES']['foo'] and st['__lines__']['PROFILES'] == 4
    with pytest.raises(numerics.ExtractError, match='cannot evaluate'):
        numerics.literal_assignments("PROFILES = {'a': open('/etc/passwd').read()}", {'PROFILES'}, 'st.py')


# --- review items (#439 review, 2026-11-14): each was a change that altered outputs but passed --------------------
CAMPAIGN = {'id': 'c1', 'teacher': 'foo',
            'teacher_overrides': {'serving': {'tunables': {'mtp': 3, 'extra_env': {
                'PYTHONPATH': '/data/plugins/cpu_supersede_v2:/data/plugins/eot_checkpoint_v4',
                'SYNTH_CPU_SUPERSEDE': '1', 'SYNTH_EOT_CHECKPOINT': '1'}}}},
            'protocol': {'context_tokens': 5000},
            'tuning': {'candidates': [{'name': 'mtp4', 'tunables': {'mtp': 4}},
                                      {'name': 'bs', 'tunables': {'bs': 32},
                                       'teacher': {'serving': {'optional_args': {'bs': ['--block-size', '{bs}']}}}}]}}


@needs_toml
def test_campaigns_record_the_effective_teacher_after_overrides(tmp_path):        # review item 3
    r = toml_repo(tmp_path, campaigns={'c1': CAMPAIGN})
    specs = numerics.extract('task-synthesis', r)
    c = specs['campaign/c1']
    assert c['adjacent.speculative.num_speculative_tokens'] == 3 and specs['foo']['adjacent.speculative.num_speculative_tokens'] == 2
    assert c['env.SYNTH_EOT_CHECKPOINT'] == '1' and 'eot_checkpoint_v4' in c['env.PYTHONPATH']
    assert c['adjacent.cpu_supersede'] == 'on' and specs['foo']['adjacent.cpu_supersede'] == 'off'   # item 10
    assert c['max_model_len'] == '5000' and c['output_cap.tokens'] == 100      # campaign [protocol] < PROTOCOL_DEFAULTS
    assert c['protocol.finish_notice'] == 'v4' and c['campaign.teacher'] == 'foo'
    assert specs['campaign/c1/candidate/mtp4']['adjacent.speculative.num_speculative_tokens'] == 4
    assert specs['campaign/c1/candidate/bs']['args.--block-size'] == '32'      # a candidate's own teacher overrides
    # stored as deltas against the teacher / campaign, decoded to the same full specs
    rec = tmp_path / 'record.json'
    numerics.record('task-synthesis', r, record_path=rec, force=True)
    stored = json.loads(rec.read_text())['pipelines']['task-synthesis']['specs']
    assert stored['campaign/c1']['$base'] == 'foo' and stored['campaign/c1/candidate/mtp4']['$base'] == 'campaign/c1'
    assert numerics.load_record(rec)['pipelines']['task-synthesis']['specs'] == numerics._jsonable(specs)
    c2 = json.loads(json.dumps(CAMPAIGN))
    c2['teacher_overrides']['serving']['tunables']['extra_env']['SYNTH_EOT_CHECKPOINT'] = '0'
    write(r, 'synth/campaigns/c1.toml', toml_dumps(c2))
    d = numerics.check('task-synthesis', r, record_path=rec)
    assert 'campaign/c1: changed env.SYNTH_EOT_CHECKPOINT: "1" -> "0"' in d


@needs_toml
def test_plugin_model_code_loaded_by_a_spec_is_hashed(tmp_path):                  # review item 4
    t = teacher(entry=['{venv}/bin/python', '{repo}/plugins/entry.py', 'serve'], env={'PYTHONPATH': '{repo}'},
                args=('--hf-overrides', '{{"architectures":["MyCached"]}}'))
    r = toml_repo(tmp_path, specs=[t])
    write(r, 'plugins/__init__.py', '')
    write(r, 'plugins/entry.py', "from vllm import ModelRegistry\n"
                                 "ModelRegistry.register_model('MyCached', 'plugins.model:MyCached')\n")
    write(r, 'plugins/model.py', 'from .helpers import K\nclass MyCached: pass\n')
    write(r, 'plugins/helpers.py', 'K = 1\n')
    write(r, 'plugins/unrelated.py', 'X = 1\n')
    f = numerics.extract('aa-serving', r)['foo']
    assert f['model_impl.repo_modules'] == ['plugins/__init__.py', 'plugins/entry.py', 'plugins/helpers.py',
                                            'plugins/model.py']
    write(r, 'plugins/helpers.py', 'K = 2\n')                                 # model code two imports away
    g = numerics.extract('aa-serving', r)['foo']
    assert g['model_impl.repo_modules.sha256'] != f['model_impl.repo_modules.sha256']
    write(r, 'plugins/unrelated.py', 'X = 2\n')                               # code the spec never loads
    assert numerics.extract('aa-serving', r)['foo']['model_impl.repo_modules.sha256'] == g['model_impl.repo_modules.sha256']


@needs_toml
def test_replica_tunables_arms_are_recorded_each(tmp_path):                        # review item 5
    tun = '{"mtp": 2, "replica_tunables": [{}, {"mtp": 4}]}'
    specs = numerics.extract('aa-serving', toml_repo(tmp_path / 'a', features="tunables.get('replica_tunables')",
                                                     profile_tunables=tun))
    assert specs['foo@r0']['adjacent.speculative.num_speculative_tokens'] == 2
    assert specs['foo@r1']['adjacent.speculative.num_speculative_tokens'] == 4 and 'foo' not in specs
    # a serve_teacher without per-replica arms ignores the tunable (task-synthesis's)
    assert 'foo' in numerics.extract('task-synthesis', toml_repo(tmp_path / 'b', profile_tunables=tun))


@needs_toml
def test_env_comes_from_the_branch_replica_env_and_runtime_keys(tmp_path):       # review item 6
    t = teacher(runtime={'nvcc_append_flags': '-DX', 'cu13_ld_library_path': True})
    f = numerics.extract('aa-serving', toml_repo(tmp_path, specs=[t]))['foo']
    assert f['env.LAUNCHER_MARK'] == 'from the branch replica_env'          # its own function, cu13 wheel stubbed
    assert f['vllm.nvcc_append_flags'] == '-DX' and f['vllm.cu13_ld_library_path'] is True


@needs_toml
def test_gpu_type_is_recorded(tmp_path):                                           # review item 7
    t = teacher(resources={'cuda_capability': '9.0'})
    assert numerics.extract('aa-serving', toml_repo(tmp_path / 'a', specs=[t], features="res.get('cuda_capability', '10.0')"))['foo']['gpu.cuda_capability'] == '9.0'
    # a serve_teacher that never reads the field submits to its default GPU type
    assert numerics.extract('aa-serving', toml_repo(tmp_path / 'b', specs=[t]))['foo']['gpu.cuda_capability'] == '10.0'


@needs_toml
@pytest.mark.parametrize('flag,value,recorded', [
    ('--load-format', 'auto', False), ('--load-format', 'dummy', True),                          # review item 9
    ('--model-loader-extra-config', '{{"num_threads":16}}', False),
    ('--model-loader-extra-config', '{{"num_threads":16,"revision":"x"}}', True),
    ('--engram-config', '{{"cpu_offload": true}}', False), ('--engram-config', '{{"bits": 4}}', True),
    ('--max-cudagraph-capture-size', '2048', False), ('--async-scheduling', None, False),        # false positives
    ('--block-size', '256', True),
])
def test_flags_skipped_only_at_safe_values(tmp_path, flag, value, recorded):
    t = teacher(args=(flag,) if value is None else (flag, value))
    f = numerics.extract('aa-serving', toml_repo(tmp_path, specs=[t]))['foo']
    assert (f'args.{flag}' in f) is recorded


@needs_toml
def test_cudagraph_capture_sizes_are_throughput(tmp_path):
    def comp(sizes):
        cfg = {'cudagraph_mode': 'FULL', 'cudagraph_capture_sizes': sizes}
        return ('--compilation-config', json.dumps(cfg).replace('{', '{{').replace('}', '}}'))
    a = numerics.extract('aa-serving', toml_repo(tmp_path / 'a', specs=[teacher(args=comp([1, 2]))]))['foo']
    b = numerics.extract('aa-serving', toml_repo(tmp_path / 'b', specs=[teacher(args=comp([1, 2, 4]))]))['foo']
    assert a == b and a['compilation_config.cudagraph_mode'] == 'FULL'


@needs_toml
def test_path_only_moves_do_not_fail(tmp_path):
    def node(tag, model_type='foo'):
        d = tmp_path / f'node-{tag}'
        write(d, 'venv/lib/python3.12/site-packages/vllm-0.1.0+g1.dist-info/METADATA', 'x')
        write(d, 'ckpt/config.json', f'{{"model_type": "{model_type}"}}')
        write(d, 'draft/config.json', '{"d": 1}')
        t = teacher(ckpt=str(d / 'ckpt'), venv=str(d / 'venv'))
        t['serving']['stage_extra'] = [{'name': 'draft', 'source': str(d / 'draft')}]
        t['serving']['kernel_wheels'] = [{'path': str(d / 'w' / 'k-1.0.whl'), 'sha256': 'abc'}]
        return numerics.extract('aa-serving', toml_repo(tmp_path / f'repo-{tag}', specs=[t]))['foo']
    a, b = node('a'), node('b')
    assert numerics.compare({'foo': a}, {'foo': b}) == []
    assert a['host.vllm.build'] == ['vllm-0.1.0+g1'] and a['vllm.kernel_wheels'] == [['k-1.0.whl', 'abc']]
    assert a['host.weights.files_sha256.config.json'] == numerics.hashlib.sha256(b'{"model_type": "foo"}').hexdigest()
    assert 'host.weights.stage_extra.draft.files_sha256.config.json' in a
    # the content, not the path, counts
    assert numerics.compare({'foo': a}, {'foo': node('b', 'bar')}) == [
        f'foo: changed host.weights.files_sha256.config.json: "{a["host.weights.files_sha256.config.json"]}" -> '
        f'"{numerics.hashlib.sha256(b"""{"model_type": "bar"}""").hexdigest()}"']


@needs_toml
def test_untracked_files_are_ignored(tmp_path):
    r = toml_repo(tmp_path)
    git_commit(r)
    before = numerics.extract('aa-serving', r)
    write(r, 'synth/teachers/scratch.toml', toml_dumps(teacher('scratch')))   # untracked: not part of the pipeline
    write(r, 'templates/untracked.jinja', 'x')
    assert numerics.extract('aa-serving', r) == before


# ---------------------------------------------------------------------------------------------------------------------
# eval: submit.py init (stubbed I/O) + serve_job.sh interpreter
# ---------------------------------------------------------------------------------------------------------------------
COMMON = """
from pathlib import Path
SIF = Path('/data/vllm.sif')
VLLM_VENVS = {}
TOOL_PARSERS = {}
REASONING_EFFORTS = {}
QWEN35_SERVING = {'fp8kv': {'kv_cache_dtype': 'fp8', 'serving_args': {'9.0': '--attention-backend FLASHINFER'}}}
PROTOCOL = {'agent': 'pi', 'context_window': 1000, 'max_output_tokens': 100, 'temperature': 1.0, 'top_p': 0.95,
            'pi_thinking': 'max'}
PROTOCOL_T2 = {'agent': 'terminus-2', 'parser': 'json', 'context_window': 1000, 'max_output_tokens': 100,
               'temperature': 0.7, 'top_p': 1.0}
PROTOCOLS = {'pi': PROTOCOL, 'terminus-2': PROTOCOL_T2}
DEFAULT_FINISH_NOTICE = '40:16'
def finish_notice(spec, harness='pi'):
    if harness != 'pi':
        return None
    from tb_agents import pi_context_limit as G       # the real one imports the repository's context guard
    warn, reserve = (int(x) for x in (spec or DEFAULT_FINISH_NOTICE).split(':'))
    return {'warn_at': warn, 'reserve': reserve, 'policy': G.POLICY}
"""

SUBMIT = """
import argparse
from common import *
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action')
    ap.add_argument('--name')
    ap.add_argument('--model')
    ap.add_argument('--family', default='qwen3_5')
    ap.add_argument('--serving', default='fp8kv')
    ap.add_argument('--serve-gpu', default='hgpu100')
    ap.add_argument('--harness', default=None)
    ap.add_argument('--finish-notice', default=None)
    a = ap.parse_args()
    run = ROOT / a.name
    if a.action == 'init':
        {extra}
        harness = harness_for(a, a.model)['harness']
        atomic(run / 'config.json', {{'model': a.model, 'served': 'q35', 'chat_template': a.model + '/t.jinja',
            'sif': str(SIF), 'max_model_len': PROTOCOL['context_window'], 'serve_gpu': a.serve_gpu,
            'tool_parser': '' if harness == 'terminus-2' else TOOL_PARSERS.get(a.family, 'qwen3_coder'),
            'vllm_venv': VLLM_VENVS.get(a.family, ''), 'reasoning_effort': REASONING_EFFORTS.get(a.family, ''),
            'harness': harness,
            'protocol': {{**PROTOCOLS[harness], 'finish_notice': finish_notice(a.finish_notice, harness)}},
            **QWEN35_SERVING[a.serving]}})
        return
    raise SystemExit('not init')
"""

SERVE_JOB = r"""#!/bin/bash
RUN=$1
cfg() { "$PY" -c "import json,sys; v=json.load(open('$RUN/config.json'))$1; print(v if isinstance(v,str) else json.dumps(v))"; }
export HOME=/home/u USER=u
MAX_LEN=$(cfg "['max_model_len']"); SIF=$(cfg "['sif']")
PROFILE=$(cfg ".get('serve_profile', 'qwen3_5')")
SERVE_ENV=()
if [ "$PROFILE" = snowball ]; then
  FAMILY_FLAGS=(--reasoning-parser snowball)
else   # qwen3_5
  FAMILY_FLAGS=(--limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3)
  KVD=$(cfg ".get('kv_cache_dtype') or ''"); [ -n "$KVD" ] && FAMILY_FLAGS+=(--kv-cache-dtype "$KVD")
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  SARGS=$(cfg ".get('serving_args', {}).get('$CC', '')")
  # shellcheck disable=SC2206
  [ -n "$SARGS" ] && FAMILY_FLAGS+=($SARGS)
  while IFS= read -r kv; do [ -n "$kv" ] && SERVE_ENV+=("$kv"); done < <(cfg ".get('serve_env', {})" |
    "$PY" -c 'import json, sys; [print(f"{k}={v}") for k, v in json.load(sys.stdin).items()]')
fi
TOOL_PARSER=$(cfg ".get('tool_parser', 'qwen3_coder')")
TOOL_FLAGS=(); [ -n "$TOOL_PARSER" ] && TOOL_FLAGS=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER")
python3 - <<'PYEOF'
print("it's a heredoc: an unbalanced quote here must not matter")
PYEOF
VENV=$(cfg ".get('vllm_venv', '')")
if [ -n "$VENV" ]; then
  LAUNCH=(env CUDA_HOME=/cuda HF_HUB_OFFLINE=1 "${SERVE_ENV[@]}" "$VENV/bin/python")
else
  LAUNCH=(apptainer exec --nv "$SIF" env HOME=/run/ev VLLM_USE_DEEP_GEMM=0 "${SERVE_ENV[@]}" python3)
fi
VARGS=($( [ "1" -gt 1 ] && echo "-m vllm.entrypoints.cli.main serve $M" \
      || echo "-m vllm.entrypoints.openai.api_server --model $M" ) --max-model-len "$MAX_LEN" \
    --max-num-seqs 256 --enable-prefix-caching "${FAMILY_FLAGS[@]}" \
    --generation-config vllm --override-generation-config '{"temperature":1.0,"top_p":0.95}' \
    --chat-template "$M/eval_chat_template.jinja" "${TOOL_FLAGS[@]}" --seed {seed})
if [ -n "$OLD" ] && ! curl -s "$OLD"; then
  echo "complex condition after the launch: ignored"
fi
setsid "${LAUNCH[@]}" "${VARGS[@]}" &
"""


def eval_repo(tmp_path, seed='0', extra='pass', insert=''):
    r = tmp_path / 'repo'
    write(r, 'eval_q35/common.py', COMMON)
    write(r, 'eval_q35/submit.py', SUBMIT.format(extra=extra))
    write(r, 'eval_q35/serve_job.sh', SERVE_JOB.replace('{seed}', seed).replace(
        'TOOL_PARSER=$(cfg', (insert + '\n' if insert else '') + 'TOOL_PARSER=$(cfg'))
    write(r, 'tb_agents/pi_context_limit.py', "POLICY = 'sticky-v4'\n")
    return r


def test_eval_extraction_runs_submit_init_and_serve_job(tmp_path):
    full = numerics.extract_full('eval', eval_repo(tmp_path))
    assert full['skipped'] == [] and full['warnings'] == []
    assert set(full['specs']) == {f'qwen3_5/fp8kv/{g}{h}' for g in ('hgpu100', 'h100x2', 'b200') for h in ('', '/terminus-2')}
    h, b = full['specs']['qwen3_5/fp8kv/h100'], full['specs']['qwen3_5/fp8kv/b200']
    assert h['kv_cache_dtype'] == 'fp8' and h['attention_backend'] == 'FLASHINFER'       # serving_args['9.0']
    assert b['attention_backend'] == 'vllm default'                                         # no '10.0' entry
    assert h['max_model_len'] == '1000' and h['seed'] == '0' and h['sampling.top_p'] == 0.95
    assert h['output_cap.kind'].startswith('client-side') and h['output_cap.tokens'] == 100
    assert h['image_limits'] == {'image': 0, 'video': 0} and h['reasoning_parser'] == 'qwen3'
    assert h['tool_parser'] == 'qwen3_coder' and h['auto_tool_choice'] is True
    assert h['vllm.sif'] == '/data/vllm.sif' and h['host.vllm.build'] is None and h['vllm.cuda_home'] == 'SIF'
    assert h['env.VLLM_USE_DEEP_GEMM'] == '0' and 'env.HOME' not in h and 'env.USER' not in h
    assert h['request.pi_thinking'] == 'max' and h['protocol.harness'] == 'pi'
    assert h['chat_template.path'].startswith('per run')
    assert numerics.compare(full['specs'], numerics.extract('eval', eval_repo(tmp_path / 'b', seed='1'))) == [
        f'qwen3_5/fp8kv/{g}{h}: changed seed: "0" -> "1"' for g in ('b200', 'hgpu100', 'h100x2')
        for h in ('', '/terminus-2')]


def test_eval_records_every_harness_and_the_real_finish_notice(tmp_path):        # review item 8
    specs = numerics.extract('eval', eval_repo(tmp_path))
    pi, t2 = specs['qwen3_5/fp8kv/h100'], specs['qwen3_5/fp8kv/h100/terminus-2']
    assert (pi['protocol.finish_notice.warn_at'], pi['protocol.finish_notice.policy']) == (40, 'sticky-v4')
    assert t2['protocol.finish_notice'] is None and t2['tool_parser'] == 'none' and t2['auto_tool_choice'] is False
    assert (t2['request.overrides.temperature'], t2['request.overrides.top_p']) == (0.7, 1.0)
    r = eval_repo(tmp_path / 'b')
    write(r, 'tb_agents/pi_context_limit.py', "POLICY = 'sticky-v5'\n")      # the guard's own code is recorded
    assert numerics.extract('eval', r)['qwen3_5/fp8kv/h100']['protocol.finish_notice.policy'] == 'sticky-v5'


def test_eval_init_runs_with_no_file_or_process_access(tmp_path):
    r = eval_repo(tmp_path, extra="open('/tmp/claude-numerics-must-not-exist', 'w')")
    full = numerics.extract_full('eval', r)
    assert full['specs'] == {} and all("name 'open' is not defined" in s for s in full['skipped'])
    assert not Path('/tmp/claude-numerics-must-not-exist').exists()


@pytest.mark.parametrize('insert,expect', [                                      # review item 1
    ('if [ -n "$Y" ] && something; then FAMILY_FLAGS+=(--seed 7); fi', 'new extractor warning'),
    ('case "$CC" in 9.0) FAMILY_FLAGS+=(--seed 7) ;; esac', 'new extractor warning'),
    ('case "$CC" in\n  9.0)\n    FAMILY_FLAGS+=(--seed 7)\n    ;;\nesac', 'new extractor warning'),
    ('add() { FAMILY_FLAGS+=(--seed 7); }\nadd', 'new extractor warning'),
    ('for x in 1; do FAMILY_FLAGS+=(--seed 7); done', 'new extractor warning'),
    ('if [ "$CC" = "9.0" ]; then FAMILY_FLAGS+=(--kv-cache-dtype fp8_e5m2); fi', 'changed kv_cache_dtype'),
    ('if [ "$CC" = "1.0" ]; then FAMILY_FLAGS+=(--seed 7); else FAMILY_FLAGS+=(--kv-cache-dtype fp8_e5m2); fi',
     'changed kv_cache_dtype'),
    # review item 2: exports reach vLLM (native `env` without -i; apptainer without --cleanenv)
    ('export VLLM_ATTENTION_BACKEND=FLASH_ATTN', 'added env.VLLM_ATTENTION_BACKEND'),
    # an unknown command substitution in a guard: unparsed, never "always false"
    ('[ "$(nvidia-smi --query-gpu=name --format=csv,noheader)" = "NVIDIA H100" ] && '
     'FAMILY_FLAGS+=(--kv-cache-dtype fp8_e5m2)', 'new extractor warning'),
    ('[ "$NODE_KIND" = hgpu100 ] && FAMILY_FLAGS+=(--kv-cache-dtype fp8_e5m2)', 'new extractor warning'),
])
def test_serve_job_changes_the_interpreter_cannot_follow_fail(tmp_path, insert, expect):
    rec = tmp_path / 'record.json'
    numerics.record('eval', eval_repo(tmp_path / 'a'), record_path=rec)
    d = numerics.check('eval', eval_repo(tmp_path / 'b', insert=insert), record_path=rec)
    assert any(expect in x for x in d), d


def test_warnings_compare_without_line_numbers():
    assert numerics.compare_warnings(['k: unparsed eval_q35/serve_job.sh:12: X'],
                                     ['k: unparsed eval_q35/serve_job.sh:40: X']) == []
    assert numerics.compare_warnings([], ['w']) == ['new extractor warning (it cannot follow this; it may change '
                                                    'numerics): w']


def test_apptainer_cleanenv_passes_only_apptainerenv_exports():                 # review item 2
    exports = {'VLLM_ATTENTION_BACKEND': 'X', 'APPTAINERENV_VLLM_USE_V1': '0'}
    env, interp, _ = numerics._launch_env(['apptainer', 'exec', '--cleanenv', 'a.sif', 'env', 'A=1', 'python3'], exports)
    assert env == {'VLLM_USE_V1': '0', 'A': '1'} and interp == 'python3'
    env, _, _ = numerics._launch_env(['env', 'A=1', 'python'], exports)
    assert env == {**exports, 'A': '1'}
    assert numerics._launch_env(['env', '-i', 'A=1', 'python'], exports)[0] == {'A': '1'}


def test_serve_job_interpreter_reports_what_it_cannot_follow():
    text = textwrap.dedent("""
        FAMILY_FLAGS=(--a 1)
        X=$(some_tool --flag)
        FAMILY_FLAGS+=(--b "$X")
        for f in a b; do FAMILY_FLAGS+=(--c "$f"); done
        if [ -n "$Y" ] && something; then
          FAMILY_FLAGS+=(--d 1)
        fi
        VARGS=("${FAMILY_FLAGS[@]}")
        """)
    sj = numerics.ServeJob(text, {}, '9.0').run()
    assert sj.arrays['VARGS'][:3] == ['--a', '1', '--b']
    assert any('a value it cannot know here' in u for u in sj.unparsed)
    assert any('FAMILY_FLAGS+=(--c' in u for u in sj.unparsed)              # a loop body
    assert any('FAMILY_FLAGS+=(--d' in u for u in sj.unparsed)              # under an if it cannot evaluate


def test_argv_fields_last_flag_wins_and_switches():
    f, warn = numerics.argv_fields(['vllm', 'serve', 'm', '--kv-cache-dtype', 'fp8', '--kv-cache-dtype=auto',
                                    '--mamba-backend', '--override-generation-config', 'not json'])
    assert f['kv_cache_dtype'] == 'auto' and f['args.--mamba-backend'] is True
    assert f['args.--override-generation-config'] == 'not json' and warn


def test_diff_rows_cover_models_in_several_pipelines():
    rec = {'pipelines': {
        'aa-serving': {'specs': {'a': {'weights.model': 'M', 'kv_cache_dtype': 'fp8', 'seed': '0'},
                                 'z': {'weights.model': 'Z', 'kv_cache_dtype': 'auto'}}},
        'task-synthesis': {'specs': {'a': {'weights.model': 'M', 'kv_cache_dtype': 'auto', 'seed': '0'}}}}}
    rows = numerics.diff_rows(rec)
    assert [(m, f) for m, f, _ in rows] == [('M', 'kv_cache_dtype')]        # Z is in one pipeline only


def test_the_record_covers_the_three_pipelines():
    rec = numerics.load_record()
    assert set(rec['pipelines']) == set(numerics.PIPELINES)
    for name, sec in rec['pipelines'].items():
        assert sec['specs'] and sec['recorded_from']['commit'], name
        for key, fields in sec['specs'].items():
            assert all(isinstance(k, str) and '{repo}' not in k for k in fields), (name, key)
            assert '/opt/aai/worktrees' not in json.dumps(fields), (name, key)


# --- second review round (#439 at 3dff2e8b) -------------------------------------------------------------------------
def _ckpt(d, shard=b'\x01' * (5 << 20), config='{"model_type": "foo"}', with_shards=True):
    write(d, 'config.json', config)
    write(d, 'model.safetensors.index.json', '{"weight_map": {"w": "model-00001-of-00001.safetensors"}}')
    if with_shards:
        (d / 'model-00001-of-00001.safetensors').write_bytes(shard)
    return d


@needs_toml
@pytest.mark.parametrize('swap,field', [
    (dict(with_shards=False), 'host.weights.shards'),                             # only the configs were copied
    (dict(shard=b'\x01' * ((5 << 20) - 1) + b'\x02'), 'host.weights.shards.sample_sha256'),  # another SFT step
    (dict(shard=b'\x01' * (6 << 20)), 'host.weights.shards'),                      # a re-quantized export
])
def test_swapping_the_weights_under_the_same_configs_fails(tmp_path, swap, field):   # review 2, item 1
    base = _ckpt(tmp_path / 'a' / 'ckpt')             # same name: only the content differs
    other = _ckpt(tmp_path / 'b' / 'ckpt', **swap)
    a = numerics.extract('aa-serving', toml_repo(tmp_path / 'ra', specs=[teacher(ckpt=str(base))]))
    b = numerics.extract('aa-serving', toml_repo(tmp_path / 'rb', specs=[teacher(ckpt=str(other))]))
    assert a['foo']['host.weights.shards'] == ['model-00001-of-00001.safetensors:5242880']
    assert 'host.weights.files_sha256.model.safetensors.index.json' in a['foo']
    d = numerics.compare(a, b)
    assert any(f' {field}:' in x or f' {field} ' in x for x in d), d
    if field.endswith('sample_sha256'):                    # same names and sizes: only the sampled bytes differ
        assert a['foo']['host.weights.shards'] == b['foo']['host.weights.shards'] and len(d) == 1
    # and the same weights at another path pass
    c = numerics.extract('aa-serving', toml_repo(tmp_path / 'rc', specs=[teacher(ckpt=str(_ckpt(tmp_path / 'c' / 'ckpt')))]))
    assert numerics.compare(a, c) == []


@pytest.mark.parametrize('insert', [                                             # review 2, item 2
    '[ "$(hostname -s)" = gpu101 ] && TOOL_PARSER=hermes',
    'if [ -n "$SLURM_JOB_ID" ]; then TOOL_PARSER=hermes; fi',
    'for p in hermes; do TOOL_PARSER=$p; done',
    '[ "$(hostname -s)" = gpu101 ] && export VLLM_ATTENTION_BACKEND=FLASH_ATTN',
    'if [ -n "$SLURM_JOB_ID" ]; then KVD=fp8_e5m2; fi\n[ -n "$KVD" ] && FAMILY_FLAGS+=(--kv-cache-dtype "$KVD")',
])
def test_assignments_under_unknown_guards_become_unknown(tmp_path, insert):
    rec = tmp_path / 'record.json'
    numerics.record('eval', eval_repo(tmp_path / 'a'), record_path=rec)
    r = eval_repo(tmp_path / 'b')
    p = r / 'eval_q35' / 'serve_job.sh'          # after the TOOL_PARSER assignment, before TOOL_FLAGS reads it
    p.write_text(p.read_text().replace('TOOL_FLAGS=(); ', insert + '\nTOOL_FLAGS=(); ', 1))
    d = numerics.check('eval', r, record_path=rec)
    assert any('new extractor warning' in x for x in d), d


def test_sourced_files_are_followed_or_reported(tmp_path):                       # review 2, item 3
    rec = tmp_path / 'record.json'
    numerics.record('eval', eval_repo(tmp_path / 'a'), record_path=rec)
    r = eval_repo(tmp_path / 'b', insert='source "$CODE/eval_q35/extra_flags.sh"')
    write(r, 'eval_q35/extra_flags.sh', 'FAMILY_FLAGS+=(--kv-cache-dtype fp8_e5m2)\n')
    assert 'qwen3_5/fp8kv/h100: changed kv_cache_dtype: "fp8" -> "fp8_e5m2"' in numerics.check('eval', r, record_path=rec)
    r = eval_repo(tmp_path / 'c', insert='. /etc/profile.d/site.sh')
    assert any('sources a file it cannot read' in x for x in numerics.check('eval', r, record_path=rec))


SYNTH_CONFIG = """
PROTOCOL_DEFAULTS = {'context_tokens': 900, 'max_output_tokens': 100, 'finish_notice': 'v4'}
FINISH_NOTICE_DEFAULT = (40, 16)
FINISH_OFF = ('off',)
def finish_notice_config(value):
    import sys
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tb_agents.pi_context_limit import finish_notice_record
    if value in FINISH_OFF or value is None:
        return None
    return finish_notice_record(*FINISH_NOTICE_DEFAULT)
PROTOCOL_DEFAULTS['finish_notice'] = finish_notice_config(PROTOCOL_DEFAULTS['finish_notice'])
"""
PI_CONTEXT_LIMIT = """
NOTICE_MARGIN = {margin}
def finish_notice_record(warn, reserve):
    return {{'warn_at': warn, 'reserve': reserve, 'notice_margin': NOTICE_MARGIN, 'sticky': True}}
"""


@needs_toml
def test_campaign_finish_notice_is_the_expanded_record(tmp_path):               # review 2, item 4
    def repo(tag, margin):
        r = toml_repo(tmp_path / tag, campaigns={'c1': CAMPAIGN})
        write(r, 'synth/config.py', SYNTH_CONFIG)
        write(r, 'tb_agents/pi_context_limit.py', PI_CONTEXT_LIMIT.format(margin=margin))
        return r
    a = numerics.extract('task-synthesis', repo('a', 256))['campaign/c1']
    assert (a['protocol.finish_notice.warn_at'], a['protocol.finish_notice.notice_margin']) == (40, 256)
    b = numerics.extract('task-synthesis', repo('b', 512))['campaign/c1']
    assert numerics.compare({'c': a}, {'c': b}) == ['c: changed protocol.finish_notice.notice_margin: 256 -> 512']


@needs_toml
def test_unreadable_host_sources_are_skipped_not_failed(tmp_path, monkeypatch):              # review 2, item 5
    ck = _ckpt(tmp_path / 'ckpt')
    venv = tmp_path / 'venv'
    for d in ('vllm-0.1.0', 'torch-2.9.0', 'triton-3.5.0', 'transformers-4.57.0', 'flashinfer_python-0.5.0',
              'torchvision-0.24.0', 'numpy-2.0.0'):
        (venv / 'lib/python3.12/site-packages' / f'{d}.dist-info').mkdir(parents=True)
    r = toml_repo(tmp_path, specs=[teacher(ckpt=str(ck), venv=str(venv))])
    rec = tmp_path / 'record.json'
    sec = numerics.record('aa-serving', r, record_path=rec)
    assert sec['specs']['foo']['host.vllm.build'] == ['flashinfer_python-0.5.0', 'torch-2.9.0', 'transformers-4.57.0',
                                                      'triton-3.5.0', 'vllm-0.1.0']
    shutil.rmtree(ck)                    # a node without /data: the filesystem itself cannot be read
    shutil.rmtree(venv)
    monkeypatch.setattr(numerics, 'host_mount_readable', lambda p: False)
    fails, host = numerics.check_full('aa-serving', r, record_path=rec)
    assert fails == [] and any('host.weights' in x for x in host) and any('host.vllm.build' in x for x in host)
    # repository-derived fields stay strict on the same node
    write(r, 'templates/t.jinja', '{{ changed }}')
    assert numerics.check_full('aa-serving', r, record_path=rec)[0] == [
        f'foo: changed chat_template.path.sha256: "{numerics.hashlib.sha256(b"{{ messages }}").hexdigest()}" -> '
        f'"{numerics.hashlib.sha256(b"{{ changed }}").hexdigest()}"']


def test_dir_identity_sees_a_same_size_rebuild(tmp_path):                        # review 2, item 5
    write(tmp_path / 'a', 'sm100/kernel.so', 'x' * 100)
    write(tmp_path / 'b', 'sm100/kernel.so', 'y' * 100)
    write(tmp_path / 'c', 'sm100/kernel.so', 'x' * 100)                         # a copy elsewhere
    big = tmp_path / 'd' / 'big.so'
    big.parent.mkdir()
    big.write_bytes(b'\0' * (numerics.SMALL + 1))
    before = numerics.dir_identity(str(tmp_path / 'd'))
    big.write_bytes(b'\0' * (numerics.SMALL // 2) + b'\1' + b'\0' * (numerics.SMALL // 2))   # same size, new middle
    assert numerics.dir_identity(str(tmp_path / 'a')) == numerics.dir_identity(str(tmp_path / 'c'))
    assert numerics.dir_identity(str(tmp_path / 'a')) != numerics.dir_identity(str(tmp_path / 'b'))
    assert numerics.dir_identity(str(tmp_path / 'd')) != before
    assert numerics.dir_identity(str(tmp_path / 'missing')).startswith('unreadable')


@needs_toml
def test_an_unparseable_campaign_is_skipped_with_its_sha256(tmp_path):         # review 2, item 6
    r = toml_repo(tmp_path, campaigns={'c1': CAMPAIGN})
    write(r, 'synth/campaigns/bad.toml', 'x = [1,\n')
    rec = tmp_path / 'record.json'
    sec = numerics.record('task-synthesis', r, record_path=rec, force=True)
    sha = numerics.hashlib.sha256(b'x = [1,\n').hexdigest()
    assert any(s.startswith(f'synth/campaigns/bad.toml (sha256 {sha}): TOMLDecodeError') for s in sec['skipped'])
    assert numerics.check('task-synthesis', r, record_path=rec) == []
    write(r, 'synth/campaigns/bad.toml', 'x = [2,\n')                             # still broken, but edited
    d = numerics.check('task-synthesis', r, record_path=rec)
    assert any(x.startswith('skipped (could not extract)') for x in d) and any(x.startswith('no longer skipped') for x in d)


@needs_toml
def test_entry_point_plugins_on_pythonpath_are_hashed(tmp_path):               # snowball-eval (aa-serving #441)
    t = teacher(env={'PYTHONPATH': '{repo}/serve_x/plugin'}, runtime={'cuda_visible_devices': 'index'},
                args=('--reasoning-parser-plugin', '{repo}/serve_x/parsers.py'))
    r = toml_repo(tmp_path, specs=[t])
    write(r, 'serve_x/parsers.py', 'import json\n')
    write(r, 'serve_x/plugin/grug-0.1.dist-info/entry_points.txt', '[vllm.general_plugins]\ngrug = grug:register\n')
    write(r, 'serve_x/plugin/grug/__init__.py', 'from .moe import M\ndef register(): pass\n')
    write(r, 'serve_x/plugin/grug/moe.py', 'M = 1\n')
    f = numerics.extract('aa-serving', r)['foo']
    assert f['model_impl.repo_modules'] == ['serve_x/parsers.py', 'serve_x/plugin/grug-0.1.dist-info/entry_points.txt',
                                            'serve_x/plugin/grug/__init__.py', 'serve_x/plugin/grug/moe.py']
    write(r, 'serve_x/plugin/grug/moe.py', 'M = 2\n')        # loaded only through the entry point, never imported
    g = numerics.extract('aa-serving', r)['foo']
    assert g['model_impl.repo_modules.sha256'] != f['model_impl.repo_modules.sha256']
    assert g['env.PYTHONPATH.sha256'] != f['env.PYTHONPATH.sha256'] and 'env.CUDA_VISIBLE_DEVICES' not in g


# --- third review round (#439 at 5ec70f4c) --------------------------------------------------------------------------
@needs_toml
def test_a_missing_checkpoint_on_a_readable_filesystem_fails(tmp_path, monkeypatch):   # review 3, item 1
    ck = _ckpt(tmp_path / 'models' / 'foo-step1000')
    rec = tmp_path / 'record.json'
    r = toml_repo(tmp_path / 'r', specs=[teacher(ckpt=str(ck))])
    sec = numerics.record('aa-serving', r, record_path=rec, force=True)
    assert sec['specs']['foo']['weights.checkpoint_name'] == 'foo-step1000'
    gone = tmp_path / 'models' / 'foo-step1000-typo'                          # /tmp is readable: really missing
    write(r, 'synth/teachers/foo.toml', toml_dumps(teacher(ckpt=str(gone))))
    d = numerics.check('aa-serving', r, record_path=rec)
    assert 'foo: changed weights.checkpoint_name: "foo-step1000" -> "foo-step1000-typo"' in d
    assert any(x.startswith('foo: added host.weights = "missing: no directory') for x in d), d
    # the same name on a node that cannot read the filesystem at all: host fields skipped, the name still checked
    write(r, 'synth/teachers/foo.toml', toml_dumps(teacher(ckpt=str(tmp_path / 'elsewhere' / 'foo-step1000'))))
    monkeypatch.setattr(numerics, 'host_mount_readable', lambda p: False)
    assert numerics.check_full('aa-serving', r, record_path=rec)[0] == []
    write(r, 'synth/teachers/foo.toml', toml_dumps(teacher(ckpt=str(tmp_path / 'elsewhere' / 'bar-step1000'))))
    assert numerics.check('aa-serving', r, record_path=rec) == [
        'foo: changed weights.checkpoint_name: "foo-step1000" -> "bar-step1000"']


@needs_toml
def test_record_refuses_unreadable_host_values_without_force(tmp_path, capsys):   # review 3, item 2
    r = toml_repo(tmp_path)                       # its checkpoint and venv are on a filesystem that does not exist
    rec = tmp_path / 'record.json'
    with pytest.raises(SystemExit, match='refusing to record'):
        numerics.record('aa-serving', r, record_path=rec)
    assert not rec.exists()
    assert 'host-derived fields unreadable on this node' in capsys.readouterr().err
    numerics.record('aa-serving', r, record_path=rec, force=True)
    assert rec.exists() and '(recorded anyway: --force)' in capsys.readouterr().err


@pytest.mark.parametrize('insert,flagged', [                                   # review 3, item 4
    ('sed -i "s/think/nothink/" "$M/eval_chat_template.jinja"', True),
    ('echo \'{"eos_token_id": 1}\' > "$M/generation_config.json"', True),
    ('cat extra.jinja >> "$M/eval_chat_template.jinja"', True),
    ('mv "$M/config.json" "$M/config.orig.json"', True),
    ('[ -n "$KVD" ] && cp /tmp/other.json "$M/config.json"', True),
    ('python3 patch_template.py "$M/eval_chat_template.jinja"', True),
    ('echo started > "$LOGS/started.txt"', False),
    ('cp "$M/config.json" "$LOGS/config.json"', False),            # reading the staged dir is fine
])
def test_writes_to_staged_inputs_are_warnings(tmp_path, insert, flagged):
    rec = tmp_path / 'record.json'
    numerics.record('eval', eval_repo(tmp_path / 'a'), record_path=rec)
    d = numerics.check('eval', eval_repo(tmp_path / 'b', insert=insert), record_path=rec)
    assert any('writes a staged input' in x for x in d) is flagged, d


def test_the_stub_package_has_a_matching_spec(tmp_path):                       # review 3, item 5
    r = toml_repo(tmp_path, specs=[teacher(runtime={'cuda_visible_devices': 'index'})])
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        numerics.extract('aa-serving', r)


def test_mount_of_follows_symlinks(tmp_path):
    """A venv linked from one filesystem into another lives where the link points (review 4): a dangling link into a
    missing /data is that mount's (unreadable), not the link's own readable one (missing)."""
    link = tmp_path / 'venv'
    link.symlink_to('/nonexistent-mount-xyz/venvs/vllm031')
    assert numerics.mount_of(link) == '/'          # its filesystem is not mounted: walks up to / (unreadable)
    assert not numerics.host_mount_readable(link)
    # this cluster's layout: /data is a symlink into a separate mount (/shared/fast) under an always-listable /lustre;
    # a path there belongs to that mount, never to /lustre
    if os.path.ismount('/shared/fast') and not os.path.ismount('/lustre'):
        fast = tmp_path / 'fastlink'
        fast.symlink_to('/shared/fast')
        assert numerics.mount_of(fast / 'no-such-dir-xyz') == '/shared/fast'


@needs_toml
def test_a_profile_venv_pin_declares_its_version(tmp_path):                       # rt-vllm #478
    pinned = '{"mtp": 2, "venv": "/nonexistent/other", "vllm_version": "0.0.9"}'
    f = numerics.extract('aa-serving', toml_repo(tmp_path / 'a', profile_tunables=pinned))['foo']
    assert f['vllm.version'] == '0.0.9'                                          # the pinned venv's, not the spec's
    f = numerics.extract('aa-serving', toml_repo(tmp_path / 'b'))['foo']
    assert f['vllm.version'] == '0.1.0'                                          # no pin: the spec's runtime
