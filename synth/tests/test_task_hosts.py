"""Generated tasks whose setup writes /etc/hosts (task_synthesis.pipeline.package): the image keeps setup's copy, the
entrypoint puts it back at container start, and `round rehost` migrates tasks made before."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from task_synthesis.pipeline import package as P

OLD_SERVICES_DOCKERFILE = '''FROM local/tasksynth-base:2
COPY tasksynth/ /tasksynth-meta/
RUN bash /tasksynth-meta/setup.sh && rm -rf /tasksynth /tasksynth-meta/setup.sh
RUN mkdir -p /etc/tasksynth && mv /tasksynth-meta/services.sh /etc/tasksynth/services.sh \
&& mv /tasksynth-meta/entrypoint /usr/local/bin/tasksynth-entrypoint \
&& chmod +x /usr/local/bin/tasksynth-entrypoint /etc/tasksynth/services.sh
ENTRYPOINT ["/usr/local/bin/tasksynth-entrypoint"]
RUN rm -rf /tasksynth-meta
WORKDIR /app
'''
HOSTS_SETUP = '#!/bin/bash\nset -euo pipefail\nprintf "10.99.99.99 build.internal\\n" >> /etc/hosts\n'


def pkg(setup='#!/bin/bash\nset -euo pipefail\n', services=None):
    return {'meta': {'slug': 'demo-task', 'apt': [], 'pip': []}, 'files': {}, 'setup': setup, 'services': services,
            'instruction': 'x\n', 'sketch': 'x\n'}


def make_task(task_dir, p):
    return P.materialize(p, {'spec_id': 's-1'}, {'model': 'm'}, Path(task_dir))


class Template(unittest.TestCase):
    def test_tasks_that_leave_etc_hosts_alone_keep_the_old_dockerfile(self):
        self.assertEqual(P.dockerfile(pkg(services='mysqld &')), OLD_SERVICES_DOCKERFILE)
        self.assertNotIn('ENTRYPOINT', P.dockerfile(pkg()))

    def test_a_setup_that_writes_etc_hosts_gets_the_entrypoint(self):
        for services in (None, 'mysqld &'):
            text = P.dockerfile(pkg(HOSTS_SETUP, services))
            self.assertIn('RUN ' + P.HOSTS_SETUP_RUN + '\n', text)
            self.assertEqual(text.count(P.ENTRYPOINT_LINE), 1)
            self.assertEqual(P.ENTRYPOINT_RUN in text, services is None)
        with tempfile.TemporaryDirectory() as d:
            task = make_task(Path(d) / 'task', pkg(HOSTS_SETUP))
            self.assertEqual((task / 'environment/tasksynth/entrypoint').read_text(), P.ENTRYPOINT)
            self.assertFalse((task / 'environment/tasksynth/services.sh').exists())


class Rehost(unittest.TestCase):
    def old_task(self, task_dir, setup, services=None):
        """A task as the template before HOSTS wrote it (setup did not change the Dockerfile then)."""
        task = make_task(task_dir, pkg(services=services))
        (task / 'environment/tasksynth/setup.sh').write_text(setup)
        if services:
            (task / 'environment/tasksynth/entrypoint').write_text('#!/bin/bash\nbash /etc/tasksynth/services.sh\n')
        return task

    def test_an_older_task_gets_what_the_template_writes_now(self):
        for services in (None, 'mysqld &'):
            with tempfile.TemporaryDirectory() as d:
                task = self.old_task(Path(d) / 'task', HOSTS_SETUP, services)
                self.assertTrue(P.rehost(task))
                env = task / 'environment'
                self.assertEqual((env / 'Dockerfile').read_text(), P.dockerfile(pkg(HOSTS_SETUP, services)))
                self.assertEqual((env / 'tasksynth/entrypoint').read_text(), P.ENTRYPOINT)
                self.assertFalse(P.rehost(task), 'a second run changes nothing')

    def test_tasks_without_etc_hosts_and_foreign_dockerfiles(self):
        with tempfile.TemporaryDirectory() as d:
            task = self.old_task(Path(d) / 'task', '#!/bin/bash\ntrue\n')
            before = (task / 'environment/Dockerfile').read_text()
            self.assertFalse(P.rehost(task))
            self.assertEqual((task / 'environment/Dockerfile').read_text(), before)
        with tempfile.TemporaryDirectory() as d:
            task = self.old_task(Path(d) / 'task', HOSTS_SETUP)
            df = task / 'environment/Dockerfile'
            df.write_text(df.read_text() + 'ENTRYPOINT ["/bin/other"]\n')
            with self.assertRaises(P.PackageError):
                P.rehost(task)

    def test_rounds_and_cohorts_are_migrated_past_a_task_still_being_written(self):
        from task_synthesis.pipeline import cohort, round as round_mod
        with tempfile.TemporaryDirectory() as d:
            rounds, cohorts = Path(d) / 'rounds', Path(d) / 'cohorts'
            self.old_task(rounds / 'r1/tasks/a', HOSTS_SETUP)
            (rounds / 'r1/tasks/b/environment/tasksynth').mkdir(parents=True)  # materialize() in progress
            self.old_task(rounds / 'r2/tasks/c', HOSTS_SETUP)
            self.old_task(cohorts / 'c1/tasks/d', HOSTS_SETUP)
            with mock.patch.object(round_mod, 'ROUNDS', rounds), mock.patch.object(cohort, 'COHORTS', cohorts), \
                    mock.patch('builtins.print') as out:
                round_mod.rehost_rounds([])
            lines = [c.args[0] for c in out.call_args_list]
            self.assertEqual(lines[-1], '3 tasks rehosted, 1 left alone, in 3 rounds and cohorts')
            for t in (rounds / 'r1/tasks/a', rounds / 'r2/tasks/c', cohorts / 'c1/tasks/d'):
                self.assertIn(P.HOSTS_SETUP_RUN, (t / 'environment/Dockerfile').read_text())

    def test_a_setup_line_with_a_prefix_keeps_it(self):
        """The real-asset rounds unpack the asset in the same RUN, before setup."""
        with tempfile.TemporaryDirectory() as d:
            task = self.old_task(Path(d) / 'task', HOSTS_SETUP)
            df = task / 'environment/Dockerfile'
            prefix = 'mkdir -p /tasksynth/asset && tar --zstd -xf /tasksynth-meta/asset.tar.zst -C /tasksynth/asset && '
            df.write_text(df.read_text().replace('RUN ' + P.SETUP_RUN, 'RUN ' + prefix + P.SETUP_RUN))
            self.assertTrue(P.rehost(task))
            self.assertIn('RUN ' + prefix + P.HOSTS_SETUP_RUN + '\n', df.read_text())


class Entrypoint(unittest.TestCase):
    """The entrypoint, run with /etc under a temporary directory and `hostname` stubbed."""

    def run_entrypoint(self, runtime, saved, services=None):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'etc/tasksynth').mkdir(parents=True)
            (root / 'var/log').mkdir(parents=True)
            (root / 'etc/hosts').write_bytes(runtime)
            if saved is not None:
                (root / 'etc/tasksynth/hosts').write_bytes(saved)
            if services:
                (root / 'etc/tasksynth/services.sh').write_text(services.replace('ROOT', d))
            (root / 'bin').mkdir()
            (root / 'bin/hostname').write_text('#!/bin/sh\necho 3f9a1c\n')
            (root / 'bin/hostname').chmod(0o755)
            script = root / 'entrypoint'
            script.write_text(P.ENTRYPOINT.replace('/etc/', f'{d}/etc/').replace('/var/log/', f'{d}/var/log/'))
            out = subprocess.run(['bash', str(script), 'echo', 'command ran'], capture_output=True, text=True,
                                 env={**os.environ, 'PATH': f'{root / "bin"}:{os.environ["PATH"]}'}, check=True)
            self.assertEqual(out.stdout, 'command ran\n')
            return (root / 'etc/hosts').read_bytes(), (root / 'ran').exists()

    RUNTIME = b'127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n127.0.1.1\t3f9a1c\n'

    def test_setups_hosts_come_back_byte_for_byte_with_this_hosts_line(self):
        saved = (b'127.0.0.1\tlocalhost\n# lab sidecars, 3f9a1c\n10.42.0.10\tapi.lab.internal\n'
                 b'127.0.0.11\tdb.lab.internal\r\n10.99.99.99 build.internal')
        hosts, _ = self.run_entrypoint(self.RUNTIME, saved)
        self.assertEqual(hosts, saved + b'\n127.0.1.1\t3f9a1c\n')

    def test_a_saved_file_naming_this_host_is_kept_as_is(self):
        saved = b'127.0.0.1\tlocalhost\n10.0.0.5\t3f9a1c alias\n'
        self.assertEqual(self.run_entrypoint(self.RUNTIME, saved)[0], saved)

    def test_without_a_saved_file_hosts_stay_and_services_run(self):
        hosts, ran = self.run_entrypoint(self.RUNTIME, None, services='touch ROOT/ran\n')
        self.assertEqual(hosts, self.RUNTIME)
        self.assertTrue(ran)


if __name__ == '__main__':
    unittest.main()
