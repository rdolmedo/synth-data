"""#234 R2: a DeepSeek server released in the middle of a judge pass. The judge must move the server's test tasks to
the kept server with no sample lost (written without a verdict) and none counted twice, even while the released
server's endpoints.json still looks healthy (serve_teacher removes it only after vLLM has stopped).

  PYTHONPATH=. python -m unittest decontam.tests.test_judge_release      (CPU, ~30 s; mock servers on localhost)
"""
import asyncio
import collections
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock

from aiohttp import web

TB = re.compile(r'TB:(\S+)')
CF = re.compile(r'CF:(\S+)')


def label(tb, cf):
    h = int(hashlib.sha1(f'{tb}|{cf}'.encode()).hexdigest(), 16) % 10
    return 'REUSE' if h == 0 else 'WEAK_REUSE' if h == 1 else 'NO_REUSE'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class MockServer:
    """An OpenAI-compatible judge on its own event loop and thread; kill() closes it like a removed job."""

    def __init__(self, name, log):
        self.name, self.log, self.port = name, log, free_port()
        self.served = 0
        self.dead = False
        self.loop = asyncio.new_event_loop()
        ready = threading.Event()
        threading.Thread(target=self._run, args=(ready,), daemon=True).start()
        ready.wait(10)

    def _run(self, ready):
        asyncio.set_event_loop(self.loop)
        app = web.Application(client_max_size=16 << 20)
        app.add_routes([web.post('/v1/chat/completions', self.chat)])
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        self.loop.run_until_complete(web.TCPSite(self.runner, '127.0.0.1', self.port).start())
        ready.set()
        self.loop.run_forever()

    async def chat(self, request):
        body = await request.json()
        user = body['messages'][1]['content']
        tb, cf = TB.search(user).group(1), CF.search(user).group(1)
        await asyncio.sleep(0.05 + (hash(cf) % 10) / 100)
        if self.dead:   # killed while this request ran: the connection drops without an answer
            request.transport.close()
            raise ConnectionResetError('killed')
        self.served += 1
        self.log.append((self.name, request.headers.get('X-data-parallel-rank'), tb, cf, time.time()))
        verdict = {'label': label(tb, cf), 'reason': 'mock', 'matches': []}
        return web.json_response({'choices': [{'message': {'content': json.dumps(verdict)}, 'finish_reason': 'stop'}],
                                  'usage': {'prompt_tokens': 100, 'completion_tokens': 10}})

    def kill(self):
        self.dead = True
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(30)

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}/v1'


class ServerReleasedMidPass(unittest.TestCase):
    def test_no_lost_or_double_counted_verdicts_and_groups_repin_to_the_kept_server(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = mock.patch.dict(os.environ, {'TASKSYNTH_SERVING_ROOT': str(root / 'serving')})
            env.start()
            self.addCleanup(env.stop)
            os.environ.pop('DECONTAM_SERVING_ROOT', None)   # restored by env.stop
            from decontam import judge as J
            J = importlib.reload(J)
            J.LABELS = ('REUSE', 'WEAK_REUSE', 'NO_REUSE')   # as decontam.pool runs it (--labels)
            log = []
            kept, gone = MockServer('ds1', log), MockServer('ds2', log)
            fresh = {'ds1': True, 'ds2': True}

            def heartbeat():
                while True:
                    for name, srv in (('ds1', kept), ('ds2', gone)):
                        home = root / 'serving' / name
                        home.mkdir(parents=True, exist_ok=True)
                        (home / 'launch.json').write_text(json.dumps({'argv': ['vllm', '--data-parallel-size', '2']}))
                        if fresh[name]:
                            tmp = home / 'endpoints.json.partial'
                            tmp.write_text(json.dumps({'served_name': 'mock', 'urls': [srv.url], 'healthy_at': 1.0,
                                                       'updated_at': time.time()}))
                            os.replace(tmp, home / 'endpoints.json')
                    time.sleep(1)
            threading.Thread(target=heartbeat, daemon=True).start()
            time.sleep(1.5)
            tbs = [f'tb{i:02d}' for i in range(40)]
            pairs = [{'tb': t, 'cf': f'cf{t}-{j}', 'text': 'signals\n'} for t in tbs for j in range(10)]
            bundles = {('tb', t): {'name': t, 'text': f'TB:{t}', 'tokens': 10, 'truncated': False} for t in tbs}
            bundles.update({('cf', p['cf']): {'name': p['cf'], 'text': f'CF:{p["cf"]}', 'tokens': 10, 'truncated': False} for p in pairs})
            model, targets = J.serving_targets('ds1+ds2')
            self.assertEqual(len(targets), 4, 'two servers x two DP ranks')
            judge = J.Judge(None, model, 'max', 1024, 64, 32, serving='ds1+ds2')
            judge.targets = targets
            judge.load_file = root / 'load.json'
            out = root / 'verdicts.jsonl'

            def release_ds2():   # the driver releases ds2 once it has served part of its share
                while gone.served < 60:
                    time.sleep(0.05)
                self.killed_at = time.time()
                gone.kill()
                time.sleep(20)   # its serving file outlives the server (serve_teacher stops vLLM first)
                fresh['ds2'] = False
                (root / 'serving' / 'ds2' / 'endpoints.json').unlink(missing_ok=True)
            threading.Thread(target=release_ds2, daemon=True).start()
            summary = asyncio.run(judge.run(pairs, bundles, 2, out))

            rows = [json.loads(l) for l in out.read_text().splitlines()]
            keys = collections.Counter((r['tb'], r['cf'], r['sample']) for r in rows)
            self.assertEqual(len(keys), 800, 'every sample written')
            self.assertEqual(max(keys.values()), 1, 'no sample written twice')
            self.assertTrue(all(r.get('verdict') for r in rows), [r for r in rows if not r.get('verdict')][:3])
            self.assertTrue(all(r['verdict']['label'] == label(r['tb'], r['cf']) for r in rows))
            # groups that were on ds2 continue on ds1, each on one DP rank there
            on_ds2 = {tb for srv, _, tb, _, _ in log if srv == 'ds2'}
            self.assertTrue(on_ds2, 'ds2 served some groups before it left')
            after = [(srv, rank, tb) for srv, rank, tb, _, t in log if t > self.killed_at]
            self.assertTrue(all(srv == 'ds1' for srv, _, _ in after), 'nothing reaches ds2 after its release')
            ranks = collections.defaultdict(set)
            for srv, rank, tb in after:
                ranks[tb].add(rank)
            self.assertTrue(all(len(r) == 1 for r in ranks.values()), 'a re-pinned group stays on one rank')
            self.assertTrue(on_ds2 & set(ranks), 'ds2 groups finished on ds1')
            self.assertGreaterEqual(summary.get('targets_set_aside', 0), 1, 'set aside while its file looked healthy')
            load = json.loads((root / 'load.json').read_text())
            self.assertEqual(load['serving'], 'ds1+ds2')
            self.assertEqual(sum(w for *_, w in load['targets']), sum(judge.pinned.values()))


class ServingRoot(unittest.TestCase):
    """TASKSYNTH_SERVING_ROOT (tbshared.serving: every reader) moves the judge's registry; the old
    DECONTAM_SERVING_ROOT still works (running campaigns set it); both set to different roots is refused."""

    def setUp(self):
        saved = {k: os.environ.pop(k, None) for k in ('TASKSYNTH_SERVING_ROOT', 'DECONTAM_SERVING_ROOT')}

        def restore():
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
        self.addCleanup(restore)

    def test_root(self):
        from tbshared import serving
        from decontam import judge as J
        self.assertEqual(J.serving_root(), serving.ROOT)
        os.environ['DECONTAM_SERVING_ROOT'] = '/x/old'
        J._warned.clear()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(J.serving_root(), Path('/x/old'))
            J.serving_root()
        self.assertEqual(err.getvalue().count('DECONTAM_SERVING_ROOT is deprecated'), 1)   # once per process
        os.environ['TASKSYNTH_SERVING_ROOT'] = '/x/old'
        self.assertEqual(J.serving_root(), Path('/x/old'))
        os.environ['TASKSYNTH_SERVING_ROOT'] = '/x/new'
        with self.assertRaisesRegex(SystemExit, 'disagree'):
            J.serving_root()
        del os.environ['DECONTAM_SERVING_ROOT']
        self.assertEqual(J.serving_root(), Path('/x/new'))

    def test_targets_read_the_variables_registry(self):
        import time
        from decontam import judge as J
        with tempfile.TemporaryDirectory() as d:
            reg = Path(d) / 'm'
            reg.mkdir()
            now = time.time()
            (reg / 'endpoints.json').write_text(json.dumps({'served_name': 'x', 'urls': ['http://h:1/v1'],
                                                            'updated_at': now, 'healthy_at': now}))
            os.environ['TASKSYNTH_SERVING_ROOT'] = d
            self.assertEqual(J.serving_targets('m')[0], 'x')


if __name__ == '__main__':
    unittest.main()
