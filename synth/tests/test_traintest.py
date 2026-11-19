"""#378 train/test audit, fixes 1 and 4 for new campaigns: the Pi finish notice (v4 sticky, aa-evals #382/#395) and the
generation sandbox limits (memory, stack, disk). Campaigns frozen before keep their recorded protocol and limits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from synth import archive, sandbox_limits
from synth.config import (PROTOCOL_DEFAULTS, Campaign, build, check_protocol, effective_protocol, finish_notice_config,
                          load_teacher)
from synth.worker import attempt_env
from tb_agents import pi_context_limit as G

V4 = G.finish_notice_record(40000, 16384)
# A protocol as campaigns frozen before 2026-11-14 recorded it (no finish_notice, 132,000 context).
LEGACY_PROTOCOL = {'context_tokens': 132000, 'max_output_tokens': 16384, 'agent_timeout_sec': 21600, 'pi_version': '1.2.0',
                   'pi_resamples': 0, 'compaction': False, 'context_limit_action': 'submit'}
LEGACY_HARBOR = {'resource_guard': True, 'package_proxy': True, 'persist_images': False}


def campaign(d, protocol=None, harbor=None):
    return Campaign({'id': 'c1', 'root': str(d), 'teacher': load_teacher('nemotron35-super'),
                     'protocol': copy.deepcopy(protocol if protocol is not None else PROTOCOL_DEFAULTS),
                     'harbor': copy.deepcopy(harbor if harbor is not None else sandbox_limits.new_campaign_harbor({})),
                     'train': {'task_list': ['t']}})


def item(task='t', **kw):
    return {'id': f'c1-{task}-a0', 'kind': 'train', 'task': task, 'seed': 7, **kw}


def toml(d, body=''):
    path = Path(d) / 'c.toml'
    path.write_text(f'id = "x"\nteacher = "nemotron35-super"\nroot = "{d}"\n[train]\nmode = "per_task"\n'
                    f'trajectories_per_task = 1\n[train.stream]\ndir = "{Path(d) / "stream"}"\n' + body)
    return path


class FinishNoticeConfig(unittest.TestCase):
    def test_new_campaign_defaults_to_the_eval_protocol(self):
        """v4 at 40000:16384 and a 131,072 context, frozen as pi_eval's record (the same strings and thresholds)."""
        with tempfile.TemporaryDirectory() as d:
            config = build(toml(d))
        p = config['protocol']
        self.assertEqual(p['context_tokens'], 131072)
        self.assertEqual(p['finish_notice'], V4)
        self.assertEqual((p['finish_notice']['policy'], p['finish_notice']['sticky']), (G.STICKY_POLICY, True))
        self.assertNotIn('overrides', p)
        self.assertEqual(json.loads(json.dumps(config))['protocol'], p, 'the frozen JSON keeps the record')

    def test_values(self):
        self.assertEqual(finish_notice_config('v4'), V4)
        self.assertEqual(finish_notice_config(True), V4)
        self.assertEqual(finish_notice_config(V4), V4, 'a frozen record passes through')
        self.assertEqual(finish_notice_config('30000:16384'), G.finish_notice_record(30000, 16384))
        self.assertEqual(finish_notice_config({'warn_at': 60000}), G.finish_notice_record(60000, 16384))
        for off in ('off', False, None, '0:0', 'none'):
            self.assertIsNone(finish_notice_config(off))
        with self.assertRaisesRegex(ValueError, 'v4 sticky'):   # v3: SFT cannot use its traces
            finish_notice_config({'warn_at': 40000, 'reserve': 16384, 'policy': G.FINISH_POLICY})
        with self.assertRaisesRegex(ValueError, 'differs'):
            finish_notice_config({**V4, 'text_sha256': 'x'})
        with self.assertRaises(ValueError):
            finish_notice_config('40000:100')                  # reserve below 4096

    def test_per_campaign_and_per_source_overrides(self):
        with tempfile.TemporaryDirectory() as d:
            config = build(toml(d, '[protocol]\ncontext_tokens = 120000\nfinish_notice = "50000:16384"\n'
                                   '[[protocol.overrides]]\ntasks = ["gdp-*"]\ncontext_tokens = 98304\n'
                                   '[[protocol.overrides]]\ntasks = ["sci-*", "x"]\nfinish_notice = "off"\n'))
        p = config['protocol']
        self.assertEqual((p['context_tokens'], p['finish_notice']), (120000, G.finish_notice_record(50000, 16384)))
        self.assertEqual(effective_protocol(p, 'gdp-1'),
                         {'context_tokens': 98304, 'finish_notice': p['finish_notice'], 'override': 0})
        self.assertEqual(effective_protocol(p, 'sci-2'), {'context_tokens': 120000, 'finish_notice': None, 'override': 1})
        self.assertEqual(effective_protocol(p, 'other')['override'], None)

    def test_override_checks(self):
        base = {**PROTOCOL_DEFAULTS}
        with self.assertRaisesRegex(ValueError, 'engines serve'):
            check_protocol({**base, 'overrides': [{'tasks': ['a'], 'context_tokens': 132000}]})
        with self.assertRaisesRegex(ValueError, 'unknown keys'):
            check_protocol({**base, 'overrides': [{'tasks': ['a'], 'max_output_tokens': 100}]})
        with self.assertRaisesRegex(ValueError, 'globs'):
            check_protocol({**base, 'overrides': [{'context_tokens': 100000}]})
        with self.assertRaisesRegex(ValueError, 'globs'):   # a bare string would fnmatch its characters ('*' = all)
            check_protocol({**base, 'overrides': [{'tasks': 'gdp-*', 'context_tokens': 100000}]})
        with self.assertRaisesRegex(ValueError, 'table'):
            check_protocol({**base, 'overrides': ['gdp-*']})
        with tempfile.TemporaryDirectory() as d:          # the same from a TOML: tasks = "gdp-*"
            with self.assertRaisesRegex(ValueError, 'globs'):
                build(toml(d, '[[protocol.overrides]]\ntasks = "gdp-*"\ncontext_tokens = 98304\n'))
        with self.assertRaisesRegex(ValueError, 'no room'):
            check_protocol({**base, 'overrides': [{'tasks': ['a'], 'context_tokens': 20000}]})
        self.assertIsNone(check_protocol({**base, 'context_tokens': 20000, 'finish_notice': 'off'})['finish_notice'])

    def test_frozen_campaign_without_the_key_runs_without_it(self):
        self.assertEqual(effective_protocol(LEGACY_PROTOCOL, 't'),
                         {'context_tokens': 132000, 'finish_notice': None, 'override': None})


class AttemptEnvironment(unittest.TestCase):
    def test_new_campaign_passes_v4_and_the_sandbox_limits(self):
        with tempfile.TemporaryDirectory() as d:
            c = campaign(d)
            env = attempt_env(c, item(), {}, Path(d) / 'raw' / 'u')
        self.assertEqual((env['PI_FINISH_WARN_AT'], env['PI_FINISH_RESERVE'], env['PI_FINISH_STICKY']),
                         ('40000', '16384', '1'))
        self.assertEqual({env[k] for k in ('MAX_LEN', 'PI_CONTEXT', 'PI_CONTEXT_WINDOW', 'PI_CONTEXT_PROXY')}, {'131072'})
        self.assertEqual(G.finish_settings(env), (40000, 16384))
        self.assertTrue(G.finish_sticky(env))
        self.assertEqual((env['PI_CAMPAIGN_MEMORY_FACTOR'], env['PI_CAMPAIGN_MEMORY_MIN_MB'],
                          env['PI_CAMPAIGN_MEMORY_SPARE']), ('1.5', '24576', 'pi'))
        self.assertEqual((env['HARBOR_NS_DISKCAP'], env['HARBOR_NS_DISKCAP_MIN_MB']), ('3', '30720'))
        self.assertEqual(env['HARBOR_NS_DISKCAP_LOG'], str(Path(d) / 'raw' / 'u' / 'diskcap.json'))
        self.assertEqual(env['HARBOR_NS_MEMCAP'], '0', "eval's memcap never runs next to synth's guard")
        # the shared Pi harness defaults to AA evals' choices (#480); synthesis keeps task-synthesis' explicitly
        self.assertEqual((env['PI_NATIVE_MCP'], env['PI_CLI_PROMPT_SEPARATOR'], env['PI_INSTRUCTION_ARGV']),
                         ('1', '1', '1'))
        self.assertEqual(env['PI_CONT16K_PROFILE'], 'synth')   # gen/pi_cont16k.py: synthesis' proxy defaults

    def test_frozen_campaign_keeps_its_protocol_and_limits(self):
        stale = {'PI_FINISH_WARN_AT': '1', 'PI_FINISH_RESERVE': '5000', 'PI_FINISH_STICKY': '1',
                 'PI_CAMPAIGN_MEMORY_FACTOR': '9', 'HARBOR_NS_DISKCAP_MIN_MB': '1'}
        with tempfile.TemporaryDirectory() as d:
            env = attempt_env(campaign(d, LEGACY_PROTOCOL, LEGACY_HARBOR), item(), stale)
        self.assertFalse(set(stale) & set(env), 'nothing from the worker environment turns a limit or notice on')
        self.assertEqual(env['HARBOR_NS_DISKCAP'], '0')
        self.assertEqual(env['HARBOR_NS_MEMCAP'], '0')
        self.assertEqual(env['MAX_LEN'], '132000')
        self.assertEqual(G.finish_settings(env), (0, 0))

    def test_source_override_and_smoke_window(self):
        protocol = check_protocol({**PROTOCOL_DEFAULTS, 'overrides': [
            {'tasks': ['short-*'], 'context_tokens': 65536, 'finish_notice': '20000:8192'},
            {'tasks': ['plain-*'], 'finish_notice': 'off'}]})
        with tempfile.TemporaryDirectory() as d:
            c = campaign(d, protocol)
            short = attempt_env(c, item('short-1'), {})
            plain = attempt_env(c, item('plain-1'), {})
            smoke = attempt_env(c, item('short-1', smoke_context_window=40000), {})
        self.assertEqual((short['PI_CONTEXT_WINDOW'], short['PI_FINISH_WARN_AT'], short['PI_FINISH_RESERVE']),
                         ('65536', '20000', '8192'))
        self.assertEqual(plain['PI_CONTEXT_WINDOW'], '131072')
        self.assertNotIn('PI_FINISH_WARN_AT', plain)
        self.assertEqual(smoke['PI_CONTEXT_WINDOW'], '40000')


class SandboxLimits(unittest.TestCase):
    def test_new_campaign_records_the_limits(self):
        with tempfile.TemporaryDirectory() as d:
            config = build(toml(d, '[harbor]\nresource_guard = true\n'))
        h = config['harbor']
        self.assertEqual((h['stack'], h['sandbox_memory'], h['diskcap'], h['resource_guard']),
                         ('unlimited (as eval)', {'factor': 1.5, 'min_mb': 24576, 'spare': ['pi']},
                          {'factor': 3.0, 'min_mb': 30720}, True))
        self.assertNotIn('stack_kb', h)
        self.assertIsNone(sandbox_limits.stack_kb(h), 'unlimited, as eval')
        # synth tasks declare storage_mb 10240 (task_synthesis/pipeline/package.py): max(3 x 10240, 30720) = 30 GiB
        cap = sandbox_limits.diskcap_config(h, 'any-task')
        self.assertEqual(max(10240 * cap['factor'], cap['min_mb']), 30720)

    def test_campaign_values_win(self):
        h = sandbox_limits.new_campaign_harbor({'stack_kb': 0, 'diskcap': False,
                                                'sandbox_memory': {'factor': 2.0, 'min_mb': 32768, 'spare': ['pi']}})
        self.assertEqual(sandbox_limits.stack_kb(h), 0)
        self.assertNotIn('stack', h, 'an explicit stack_kb replaces the unlimited default')
        with self.assertRaises(ValueError):
            sandbox_limits.new_campaign_harbor({'stack': '64 MiB'})
        self.assertIsNone(sandbox_limits.diskcap_config(h))
        self.assertEqual(sandbox_limits.sandbox_memory(h)['factor'], 2.0)
        with self.assertRaises(ValueError):
            sandbox_limits.new_campaign_harbor({'sandbox_memory': {'factor': -1}})
        with self.assertRaises(ValueError):
            sandbox_limits.new_campaign_harbor({'diskcap': {'factor': 1, 'tasks': {'a': 0}}})   # 0 is not off

    def test_partial_tables_merge_into_the_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            h = build(toml(d, '[harbor.diskcap]\nmin_mb = 40960\n[harbor.sandbox_memory]\nfactor = 2.0\n'))['harbor']
        self.assertEqual(h['diskcap'], {'factor': 3.0, 'min_mb': 40960})
        self.assertEqual(sandbox_limits.diskcap_config(h, 't'), {'factor': 3.0, 'min_mb': 40960}, 'still on')
        self.assertEqual(h['sandbox_memory'], {'factor': 2.0, 'min_mb': 24576, 'spare': ['pi']}, 'pi still spared')
        env = sandbox_limits.attempt_env({}, h, 't', '/x')
        self.assertEqual((env['HARBOR_NS_DISKCAP'], env['PI_CAMPAIGN_MEMORY_SPARE']), ('3', 'pi'))
        fam = sandbox_limits.new_campaign_harbor({'diskcap_family': 'tb2.1', 'diskcap': {'min_mb': 40000}})
        self.assertEqual((fam['diskcap']['factor'], fam['diskcap']['min_mb'], fam['diskcap']['tasks']['sam-cell-seg']),
                         (1.0, 40000, 40960), 'merged into the family caps')

    def test_tb_family_caps_match_eval(self):
        h = sandbox_limits.new_campaign_harbor({'diskcap_family': 'tb2.1'})
        self.assertEqual(h['diskcap'], {'factor': 1.0, 'min_mb': 32768,
                                        'tasks': {'rstan-to-pystan': 92160, 'sam-cell-seg': 40960},
                                        'one_per_worker': ['rstan-to-pystan']})
        self.assertEqual(sandbox_limits.diskcap_config(h, 'rstan-to-pystan'), {'factor': 1.0, 'min_mb': 92160})
        self.assertEqual(sandbox_limits.diskcap_config(h, 'fix-git'), {'factor': 1.0, 'min_mb': 32768})
        lite = sandbox_limits.new_campaign_harbor({'diskcap_family': 'tblite'})
        self.assertEqual(sandbox_limits.diskcap_config(lite, 'x'), {'factor': 1.0, 'min_mb': 20480})
        with self.assertRaises(ValueError):
            sandbox_limits.new_campaign_harbor({'diskcap_family': 'swe'})

    def test_frozen_campaign_without_limits_is_unchanged(self):
        self.assertEqual(sandbox_limits.stack_kb(LEGACY_HARBOR), 8192)
        self.assertIsNone(sandbox_limits.diskcap_config(LEGACY_HARBOR))
        self.assertIsNone(sandbox_limits.sandbox_memory(LEGACY_HARBOR))
        self.assertEqual(sandbox_limits.diskcap_config({'diskcap': True}), sandbox_limits.DISKCAP_DEFAULT)


class WorkerStack(unittest.TestCase):
    def test_unlimited_raises_the_soft_limit_to_the_hard_one(self):
        import resource
        from synth.worker import apply_stack_limit
        before = resource.getrlimit(resource.RLIMIT_STACK)
        try:
            soft, hard = before
            if hard == resource.RLIM_INFINITY or hard > 8 << 20:
                resource.setrlimit(resource.RLIMIT_STACK, (8 << 20, hard))
            got = apply_stack_limit(None)
            self.assertEqual(resource.getrlimit(resource.RLIMIT_STACK)[0], hard)
            self.assertEqual(got, 'unlimited' if hard == resource.RLIM_INFINITY else hard // 1024)
            self.assertEqual(apply_stack_limit(8192), 8192 if hard == resource.RLIM_INFINITY or hard >= 8 << 20 else hard // 1024)
        finally:
            resource.setrlimit(resource.RLIMIT_STACK, before)


class ArchiveFinishNotices(unittest.TestCase):
    def attempt(self, d, notices=None, protocol=None, logs='harbor/pi-cont16k'):
        raw = Path(d) / 'c1' / 'train' / 'raw' / 'u'
        sessions = raw / 'harbor' / 't__x' / 'agent' / 'pi' / 'sessions'
        sessions.mkdir(parents=True)
        (sessions / 's.jsonl').write_text('{"type": "message"}\n')
        (raw / logs).mkdir(parents=True, exist_ok=True)
        if notices is not None:
            (raw / logs / 'finish-notices.jsonl').write_text(notices)
        (raw / 'attempt.json').write_text(json.dumps({'task': 't', 'attempt': 0, 'retry': 0,
                                                      **({'protocol': protocol} if protocol else {})}))
        return raw

    def line(self, raw):
        return archive.summarize(raw, None, {'id': 'c1', 'teacher': {'slug': 's'}}, 'train')

    def test_line_ships_the_parsed_map(self):
        entry = {'policy': G.STICKY_POLICY, 'index': 3, 'role': 'user', 'base_sha256': 'h', 'text': 'N x',
                 'appended': False}
        with tempfile.TemporaryDirectory() as d:
            raw = self.attempt(d, json.dumps(entry) + '\nnot json\n', {'finish_notice': V4})
            self.assertEqual(self.line(raw)['finish_notices'], [entry, {'malformed': 'not json'}])

    def test_empty_map_is_an_attempt_without_notices(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.line(self.attempt(d, '', {'finish_notice': V4}))['finish_notices'], [])

    def test_unmoved_log_directory_is_found(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.attempt(d, '', {'finish_notice': V4}, logs='.pi-cont16k/harbor-abc123')
            self.assertEqual(self.line(raw)['finish_notices'], [])

    def test_lost_map_of_a_v4_attempt_is_unusable(self):
        """A v4 guard creates the map at startup: an attempt run with the notice and no map ships an entry without the
        map keys, which the SFT splice drops ("unmapped") rather than training on a trace missing its notices."""
        with tempfile.TemporaryDirectory() as d:
            notices = self.line(self.attempt(d, None, {'finish_notice': V4}))['finish_notices']
        self.assertEqual(notices, [archive.MISSING_NOTICE_MAP])
        self.assertTrue(any(k not in notices[0] for k in ('index', 'role', 'base_sha256', 'text', 'appended')))

    def test_attempt_without_notice_has_no_key(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotIn('finish_notices', self.line(self.attempt(d)))
        with tempfile.TemporaryDirectory() as d:   # notice off for this attempt (override)
            self.assertNotIn('finish_notices', self.line(self.attempt(d, None, {'finish_notice': None})))



class PiCont16kRecord(unittest.TestCase):
    def test_protocol_json_records_the_guard_policy(self):
        from gen.pi_cont16k import guard_policy
        self.assertEqual(guard_policy({}), {'context_policy': G.POLICY, 'finish_notice': None})
        with tempfile.TemporaryDirectory() as d:
            env = attempt_env(campaign(d), item(), {})
        self.assertEqual(guard_policy(env), {'context_policy': G.STICKY_POLICY,
                                             'finish_notice': {'warn_at': 40000, 'reserve': 16384, 'sticky': True}})


if __name__ == '__main__':
    unittest.main()
