"""synth.verifier_toolchain: test.sh parsing and rewriting, the verifier wrapper, lazy cached staging."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import threading
import unittest

from synth import verifier_toolchain as vt

PROLOGUE = '''#!/bin/bash

# harbor-reward-fallback: ensure failed verifiers still emit reward
mkdir -p /logs/verifier
_harbor_write_missing_reward() {
  if [ ! -f /logs/verifier/reward.txt ] && [ ! -f /logs/verifier/reward.json ]; then
    echo 0 > /logs/verifier/reward.txt
  fi
}
trap _harbor_write_missing_reward EXIT
'''
TAIL = '''
if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
'''
STANDARD = PROLOGUE + '''apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

curl -LsSf https://astral.sh/uv/0.9.7/install.sh | sh

source $HOME/.local/bin/env

uvx \\
  --with pytest==8.4.1 \\
  --with pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_state.py -rA
''' + TAIL
TEST_STATE = 'def test_ok():\n    assert open("/app/out.txt").read() == "42"\n'


def make_task(root, name='t1', script=STANDARD, files=None, timeout=900):
    task = Path(root) / name
    (task / 'tests').mkdir(parents=True)
    (task / 'tests' / 'test.sh').write_text(script)
    for rel, text in {'tests/test_state.py': TEST_STATE, 'instruction.md': 'Write 42 to /app/out.txt.\n',
                      'solution/solve.sh': '#!/bin/bash\necho -n 42 > /app/out.txt\n', **(files or {})}.items():
        (task / rel).parent.mkdir(parents=True, exist_ok=True)
        (task / rel).write_text(text)
    (task / 'task.toml').write_text(f'version = "1.0"\n[verifier]\ntimeout_sec = {timeout}\n[agent]\ntimeout_sec = 900.0\n')
    return task


class Parse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_standard_rts_bootstrap_is_offline_with_apt_dropped(self):
        r = vt.parse(make_task(self.root))
        self.assertEqual((r['kind'], r['reason'], r['uv_version'], r['strip_apt'], r['implicit_python']),
                         ('uv', None, '0.9.7', True, True))
        self.assertEqual(r['specs'], ['pytest', 'pytest-json-ctrf==0.3.5', 'pytest==8.4.1'])
        self.assertEqual((r['pythons'], r['packages']), ([], ['curl']))

    def test_python_request_and_short_options(self):
        script = STANDARD.replace('--with pytest==8.4.1', '-p 3.13 -w pytest==8.4.1 --with=pyyaml')
        r = vt.parse(make_task(self.root, script=script))
        self.assertEqual((r['pythons'], r['implicit_python'], r['reason']), (['3.13'], False, None))
        self.assertIn('pyyaml', r['specs'])

    def test_package_lines(self):
        cases = {
            'apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1 && rm -rf /var/lib/apt/lists/* || true':
                (True, True, ['curl']),
            'apt-get update': (True, True, []),
            'command -v curl >/dev/null 2>&1 || yum -y install curl &>/dev/null': (True, True, ['curl']),
            'apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*': (False, True, ['curl', 'git']),
            'if command -v apt-get; then apt-get install -y curl; fi': (False, False, []),
            'DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl ca-certificates':
                (True, True, ['ca-certificates', 'curl']),
        }
        for line, want in cases.items():
            pl = vt.package_line(line)
            self.assertEqual((pl['strippable'], pl['guardable'], pl['packages']), want, line)
        self.assertIsNone(vt.package_line('if command -v apt-get >/dev/null 2>&1; then'))
        self.assertIsNone(vt.package_line('# apt-get install -y curl'))

    def test_extra_packages_keep_apt_guarded(self):
        r = vt.parse(make_task(self.root, script=STANDARD.replace('install -y curl', 'install -y curl jq')))
        self.assertEqual((r['reason'], r['strip_apt'], r['packages']), (None, False, ['curl', 'jq']))

    def test_what_the_curl_install_could_change_keeps_apt(self):
        for files in ({'tests/test_tls.py': 'import subprocess\nsubprocess.run(["openssl", "version"])\n'},
                      {'tests/test_bashrc.py': 'assert "PATH" in open("/root/.bashrc").read()\n'},
                      {'instruction.md': 'Write a health check that uses curl.\n'},
                      {'solution/solve.sh': "cat > /app/check.sh <<'EOF'\ncurl -s localhost:8000\nEOF\n"}):
            with self.subTest(files=list(files)):
                d = Path(tempfile.mkdtemp(dir=self.root))
                r = vt.parse(make_task(d, files=files))
                self.assertEqual((r['reason'], r['strip_apt']), (None, False))
                self.assertTrue(r['apt_sensitive'])
        # curl run by the solution itself (agent phase) does not count
        r = vt.parse(make_task(Path(tempfile.mkdtemp(dir=self.root)),
                               files={'solution/solve.sh': 'curl -s http://localhost:8000/health\n'}))
        self.assertTrue(r['strip_apt'])

    def test_online_only_cases(self):
        cases = {
            'tests call uv/pip': (STANDARD, {'tests/test_x.py': 'subprocess.run(["pip", "install", "x"])\n'}),
            'uv pip': (STANDARD.replace('uvx \\', 'uv pip install --system pytest\nuvx \\'), {}),
            'non-index': (STANDARD.replace('--with pytest==8.4.1', '--with git+https://x/y.git'), {}),
            'network': (STANDARD.replace('source $HOME', 'pip install requests\nsource $HOME'), {}),
            'heavy': (STANDARD.replace('--with pytest==8.4.1', '--with torch --with pytest==8.4.1'), {}),
            'after the tests': (STANDARD + 'apt-get install -y jq\n', {}),
            'uvx python': (STANDARD.replace('pytest --ctrf', 'python@3.13 -m pytest --ctrf'), {}),
        }
        for name, (script, files) in cases.items():
            with self.subTest(name):
                r = vt.parse(make_task(Path(tempfile.mkdtemp(dir=self.root)), script=script, files=files))
                self.assertEqual(r['kind'], 'uv')
                self.assertTrue(r['reason'], r)
        r = vt.parse(make_task(self.root, script='#!/bin/bash\npython3 -m pytest /tests/test_state.py\n'))
        self.assertEqual((r['kind'], r['reason']), ('none', 'no pinned astral uv installer line'))

    def test_uv_mentions_that_are_not_commands(self):
        script = STANDARD.replace('curl -LsSf', 'if ! command -v uvx >/dev/null 2>&1; then\n  curl -LsSf') \
            .replace('install.sh | sh\n', 'install.sh | sh\nfi\n')
        script = script.replace('source $HOME', 'echo "running uvx now"\nsource $HOME')
        r = vt.parse(make_task(self.root, script=script))
        self.assertEqual((r['reason'], r['strip_apt']), (None, True))


class OfflineScript(unittest.TestCase):
    def rec(self, text, **kw):
        return {'offline': True, 'uv_version': '0.9.7', 'strip_apt': True,
                'test_sh_sha256': hashlib.sha256(text.encode()).hexdigest(), **kw}

    def test_drops_the_curl_bootstrap_and_installer(self):
        s = vt.offline_script(STANDARD, self.rec(STANDARD))
        self.assertTrue(s.startswith('#!/bin/bash\n# synth offline verifier'))
        self.assertNotIn('astral.sh', s.split('UV_PYTHON_PREFERENCE=system\n', 1)[1].replace(': # uv installer', ''))
        self.assertIn(': # offline: dropped', s)
        self.assertNotIn('\napt-get', s)
        self.assertIn('UV_DEFAULT_INDEX=file:///tests/.toolchain/simple', s)
        self.assertIn('trap _harbor_write_missing_reward EXIT', s)
        self.assertEqual(s.count('#!/bin/bash'), 1)
        self.assertEqual(subprocess.run(['bash', '-n'], input=s, text=True).returncode, 0)

    def test_kept_package_lines_are_guarded(self):
        text = STANDARD.replace('install -y curl && rm', 'install -y curl jq && rm').replace(
            'rm -rf /var/lib/apt/lists/*\n', 'rm -rf /var/lib/apt/lists/* || true\n', 1)
        s = vt.offline_script(text, self.rec(text, strip_apt=False))
        line = next(l for l in s.splitlines() if l.startswith('apt-get'))
        self.assertTrue(line.endswith(vt.PKG_GUARD) and '|| true' not in line, line)
        self.assertEqual(subprocess.run(['bash', '-n'], input=s, text=True).returncode, 0)

    def test_heredoc_bodies_are_never_rewritten(self):
        text = STANDARD.replace('source $HOME', "cat > /tmp/x.sh <<'EOF'\napt-get install -y curl\nEOF\nsource $HOME")
        s = vt.offline_script(text, self.rec(text))
        self.assertIn("<<'EOF'\napt-get install -y curl\nEOF\n", s)

    def test_refuses_a_changed_script(self):
        self.assertIsNone(vt.offline_script(STANDARD + '\n', self.rec(STANDARD)))
        self.assertIsNone(vt.offline_script(STANDARD, self.rec(STANDARD, offline=False)))


def fake_toolchain(root, task_recs):
    """A toolchain root with dummy uv files, one CPython tarball and two wheels, and a manifest for task_recs."""
    root = Path(root)
    for f in ('uv', 'uvx', 'env'):
        (root / 'uv' / '0.9.7').mkdir(parents=True, exist_ok=True)
        (root / 'uv' / '0.9.7' / f).write_text(f'#!/bin/sh\necho {f}\n')
    (root / 'pool').mkdir()
    pool = {}
    for w, proj in (('pytest-8.4.1-py3-none-any.whl', 'pytest'), ('pluggy-1.7.0-py3-none-any.whl', 'pluggy')):
        (root / 'pool' / w).write_bytes(w.encode())
        pool[w] = {'bytes': len(w), 'sha256': hashlib.sha256(w.encode()).hexdigest(), 'project': proj,
                   'requires_python': '>=3.9'}
    src = root / 'src' / 'cpython-3.14.0-linux-x86_64-gnu'
    (src / 'bin').mkdir(parents=True)
    (src / 'bin' / 'python3.14').write_text('#!/bin/sh\n')
    os.symlink('python3.14', src / 'bin' / 'python3')
    (root / 'python-tar' / '0.9.7').mkdir(parents=True)
    tar = root / 'python-tar' / '0.9.7' / 'cpython-3.14.0-linux-x86_64-gnu.tar'
    with tarfile.open(tar, 'w') as tf:
        tf.add(src, arcname=src.name)
    group = {'uv_version': '0.9.7', 'wheels': sorted(pool), 'runtime_pythons': ['3.14'], 'offline': True}
    manifest = {'built_at': 'now', 'groups': {'g1': group}, 'pythons': {'0.9.7': {'3.14': src.name}},
                'pool_files': pool, 'tasks': {f'rts/{n}': {**r, 'group': 'g1', 'offline': True} for n, r in task_recs.items()},
                'uv_files': {f'uv/0.9.7/{f}': vt.sha256(root / 'uv' / '0.9.7' / f) for f in ('uv', 'uvx', 'env')},
                'python_tars': {'python-tar/0.9.7/cpython-3.14.0-linux-x86_64-gnu.tar': vt.sha256(tar)}}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


class Staging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.task = make_task(self.root / 'tasks')
        rec = vt.parse(self.task)
        self.tc = fake_toolchain(self.root / 'tc', {'t1': rec})
        self.local = self.root / 'cache'

    def tearDown(self):
        self.tmp.cleanup()

    def test_prepare_stages_once_and_is_idempotent(self):
        a = vt.prepare_verifier(self.task, self.tc, self.local, holder='a1')
        self.assertEqual((a['mode'], a['wrapped'], a['staged_now'], a['strip_apt']), ('offline', True, True, True))
        tests = self.task / 'tests'
        self.assertIn(vt.WRAPPER_MARK, (tests / 'test.sh').read_text())
        self.assertEqual((tests / 'test_task.sh').read_text(), STANDARD)
        self.assertIn('__BUDGET__', vt.VERIFIER_WRAPPER)
        self.assertIn('-gt 450 ]', (tests / 'test.sh').read_text())   # half the 900 s verifier timeout
        tc = tests / '.toolchain'
        self.assertEqual(sorted(p.name for p in (tc / 'wheels').iterdir()),
                         ['pluggy-1.7.0-py3-none-any.whl', 'pytest-8.4.1-py3-none-any.whl'])
        self.assertTrue((tc / 'python' / 'cpython-3.14.0-linux-x86_64-gnu' / 'bin' / 'python3').is_symlink())
        self.assertIn('data-requires-python="&gt;=3.9"', (tc / 'simple' / 'pytest' / 'index.html').read_text())
        self.assertEqual(os.stat(tc / 'uv' / 'uv').st_ino, os.stat(self.local / 'uv' / '0.9.7' / 'uv').st_ino)
        b = vt.prepare_verifier(self.task, self.tc, self.local, holder='a2')
        self.assertEqual((b['mode'], b['staged_now']), ('offline', False))
        self.assertEqual((tests / 'test_task.sh').read_text(), STANDARD)   # never double-wrapped

    def test_concurrent_attempts_of_one_task(self):
        results, errors = [], []

        def go(i):
            try:
                results.append(vt.prepare_verifier(self.task, self.tc, self.local, holder=f'a{i}'))
            except Exception as e:   # noqa: BLE001
                errors.append(e)
        threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(r['mode'] for r in results), ['offline'] * 8)
        self.assertEqual(sum(r['staged_now'] for r in results), 1)
        self.assertEqual([p.name for p in (self.local / '.staging').iterdir()], [])
        # the toolchain stays while any attempt holds it
        for i in range(7):
            self.assertFalse(vt.release_verifier(self.task, self.local, holder=f'a{i}'))
            self.assertTrue((self.task / 'tests' / 'test_offline.sh').exists())
        self.assertTrue(vt.release_verifier(self.task, self.local, holder='a7'))
        self.assertFalse((self.task / 'tests' / '.toolchain').exists())
        self.assertFalse((self.task / 'tests' / 'test_offline.sh').exists())
        self.assertIn(vt.WRAPPER_MARK, (self.task / 'tests' / 'test.sh').read_text())
        again = vt.prepare_verifier(self.task, self.tc, self.local, holder='b')
        self.assertEqual((again['mode'], again['staged_now']), ('offline', True))

    def test_online_when_the_task_or_manifest_disagree(self):
        (self.task / 'tests' / 'test.sh').write_text(STANDARD.replace('-rA', '-rA -x'))
        r = vt.prepare_verifier(self.task, self.tc, self.local)
        self.assertEqual((r['mode'], r['reason'], r['wrapped']), ('online', 'test.sh changed since the build', True))
        self.assertFalse((self.task / 'tests' / 'test_offline.sh').exists())
        other = make_task(self.root / 'tasks', name='t2')
        self.assertEqual(vt.prepare_verifier(other, self.tc, self.local)['reason'], 'task not in the toolchain manifest')
        plain = make_task(self.root / 'tasks', name='t3', script='#!/bin/bash\npython3 -m pytest /tests\n')
        r = vt.prepare_verifier(plain, self.tc, self.local)
        self.assertEqual((r['mode'], r['wrapped']), ('unchanged', False))
        self.assertFalse((plain / 'tests' / 'test_task.sh').exists())

    def test_corrupt_toolchain_piece_stays_online(self):
        (self.tc / 'pool' / 'pluggy-1.7.0-py3-none-any.whl').write_bytes(b'corrupt')
        r = vt.prepare_verifier(self.task, self.tc, self.local)
        self.assertEqual(r['mode'], 'online')
        self.assertIn('sha256 mismatch', r['reason'])
        self.assertFalse((self.task / 'tests' / '.toolchain').exists())


class Wrapper(unittest.TestCase):
    """The wrapper with /tests and /logs/verifier moved into a temp dir and no backoff sleeps."""

    def run_wrapper(self, offline, online, budget=450):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / 'tests').mkdir()
            (d / 'logs').mkdir()
            calls = d / 'calls'
            script = vt.VERIFIER_WRAPPER.replace('__BUDGET__', str(budget)).replace('__NET_FAIL__', vt.NET_FAIL) \
                .replace('/tests/', f'{d}/tests/').replace('/logs/verifier', f'{d}/logs').replace('15 * 2 ** n', '0')
            (d / 'tests' / 'test.sh').write_text(script)
            if offline is not None:
                (d / 'tests' / 'test_offline.sh').write_text(f'echo offline >> {calls}\n' + offline.replace('LOGS', f'{d}/logs'))
            (d / 'tests' / 'test_task.sh').write_text(
                f'echo online >> {calls}\nn=$(grep -c online {calls})\n' + online.replace('LOGS', f'{d}/logs'))
            r = subprocess.run(['bash', str(d / 'tests' / 'test.sh')], capture_output=True, text=True, timeout=60)
            reward = (d / 'logs' / 'reward.txt').read_text().strip() if (d / 'logs' / 'reward.txt').exists() else None
            return r.returncode, calls.read_text().split(), reward, r.stdout

    def test_offline_session_is_final(self):
        rc, calls, reward, _ = self.run_wrapper('echo "== test session starts =="; echo 1 > LOGS/reward.txt; exit 0',
                                                'exit 9')
        self.assertEqual((rc, calls, reward), (0, ['offline'], '1'))
        rc, calls, reward, _ = self.run_wrapper('echo "1 failed in 0.1s"; echo 0 > LOGS/reward.txt; exit 1', 'exit 9')
        self.assertEqual((rc, calls, reward), (1, ['offline'], '0'))

    def test_offline_bootstrap_failure_falls_back_online_with_a_fresh_reward(self):
        rc, calls, reward, out = self.run_wrapper(
            'echo 0 > LOGS/reward.txt; echo "No solution found"; exit 1',
            'echo "== test session starts =="; touch LOGS/reward.json; exit 0')
        self.assertEqual((rc, calls, reward), (0, ['offline', 'online'], None))   # the trap's stale 0 is gone
        self.assertIn('running the task\'s own script online', out)

    def test_online_retries_only_network_failures(self):
        net = 'if [ $n -lt 3 ]; then echo "curl: (7) Failed to connect to proxy port 8080"; exit 1; fi\n' \
              'echo "== test session starts =="; exit 0'
        rc, calls, _, _ = self.run_wrapper(None, net)
        self.assertEqual((rc, calls), (0, ['online'] * 3))
        rc, calls, _, _ = self.run_wrapper(None, 'echo "precheck failed: /app missing"; exit 1')
        self.assertEqual((rc, calls), (1, ['online']))   # not a network failure: no retry
        rc, calls, _, _ = self.run_wrapper(None, 'echo "Could not resolve host: astral.sh"; exit 1')
        self.assertEqual((rc, calls), (1, ['online'] * 5))   # bounded


if __name__ == '__main__':
    unittest.main()
