"""Harbor environment for synth campaigns (HARBOR_ENVIRONMENT=synth.environment:CampaignEnvironment).

* Tasks without network still need the model: Pi's endpoint is rewritten to a
  loopback relay inside the sandbox netns (Unix socket to the host proxy); the
  task itself gets no route, NAT or internet.
* Optional verifier dependency caches (campaign harbor.verifier_caches) are
  injected only after the agent finished, for verifiers whose pip downloads fail.
* Optional verifier infrastructure retries (campaign harbor.verifier_infra_retries = N): a /tests/test.sh run whose
  output shows the proxy or network failing (curl to the proxy refused or reset, so uv never installs) is run again
  on the same sandbox after a backoff, up to N times. Earlier outputs stay as test-stdout.infra-<i>.txt and each
  retry is logged in verifier/infra-retries.jsonl. 2026-10-25: 31% of DeepSeek CalibForge verifiers failed this way.
"""
import asyncio
import json
import os
from pathlib import Path
import re
import tarfile
import time

from harbor_ns.environment import NsEnvironment

from .common import atomic, read, sha
from .endpoint_relay import EndpointRelay

VERIFIER_UV_CACHE = '/.synth-verifier-uv-cache'
INFRA_FAILURE = re.compile(r'port 8080|Connection reset by peer|Failed to connect to|Could not resolve (proxy|host)|'
                           r'uvx: command not found|Temporary failure in name resolution')


# pytest ran: its tests are never run a second time, whatever infrastructure noise the output also shows (the synth
# verifier wrapper, synth/verifier_toolchain.py, logs failed bootstrap tries before the one that started pytest)
TESTS_STARTED = re.compile(r'test session starts|^(=+ )?[0-9]+ (passed|failed|errors?|skipped|xfailed|xpassed|deselected)\b|'
                           r'no tests ran', re.M)


async def verify_with_retries(run, stdout_path, reward_paths, retries, log_path, sleep=asyncio.sleep, base_delay=60):
    """Run the verifier; while its output shows an infrastructure failure and its tests never started, keep that output
    aside, clear the reward and run it again (up to `retries` more times, backing off). Returns the last run's result."""
    stdout_path = Path(stdout_path)
    for attempt in range(retries + 1):
        result = await run()
        try:
            text = stdout_path.read_text(errors='replace')
        except OSError:
            text = ''
        match = INFRA_FAILURE.search(text)
        if not match or attempt == retries or TESTS_STARTED.search(text) or \
                (stdout_path.parent / 'ctrf.json').exists():
            return result
        stdout_path.rename(stdout_path.with_name(f'test-stdout.infra-{attempt}.txt'))
        for path in reward_paths:
            Path(path).unlink(missing_ok=True)
        with open(log_path, 'a') as log:
            log.write(json.dumps({'time': time.time(), 'attempt': attempt, 'marker': match.group(0)}) + '\n')
        await sleep(base_delay * (attempt + 1))


def campaign_config():
    path = os.environ.get('SYNTH_CAMPAIGN')
    return json.loads(Path(path).read_text()) if path else None


class CampaignEnvironment(NsEnvironment):
    async def start(self, force_build):
        await super().start(force_build)
        harbor = (campaign_config() or {}).get('harbor') or {}
        if harbor.get('package_proxy') and not self._box.internet:
            await self._start_package_proxy(harbor)

    async def _start_package_proxy(self, harbor):
        """Registries-only proxy for an offline sandbox (synth.package_proxy): host Unix socket, exposed in the
        sandbox's network namespace at 127.0.0.1:3128, where its proxy variables point (HARBOR_NS_OFFLINE_PROXY_URL)."""
        import tempfile
        from .package_proxy import DEFAULT_ALLOW, SANDBOX_PORT, PackageProxy
        log_dir = self.trial_paths.agent_dir / 'package-proxy'
        log_dir.mkdir(parents=True, exist_ok=True)
        relay = EndpointRelay(log_dir)
        relay.scratch = tempfile.TemporaryDirectory(prefix='synth-pkg-', dir='/tmp')
        sock = Path(relay.scratch.name) / 'p.sock'
        from harbor_ns.sandbox import package_cache
        # harbor.package_cache (default on): allowed requests go through the worker's package cache (synth.package_cache)
        cache = package_cache if harbor.get('package_cache', True) else None
        proxy = await PackageProxy(sock, log_dir / 'requests.jsonl', os.environ.get('http_proxy') or 'http://proxy.internal:8080',
                                   harbor.get('package_proxy_allow') or DEFAULT_ALLOW, cache=cache,
                                   pip=bool(cache and harbor.get('package_cache_pip', False)),
                                   tag=re.sub(r'[^A-Za-z0-9_.-]', '_', self.trial_paths.trial_dir.name)[:100]).start()
        self.__dict__['_package_proxy'] = (proxy, relay)
        ns = await relay.spawn(['nsenter', '-t', str(self._box.init_pid), '-U', '-n', '--preserve-credentials', '--',
                                '/usr/bin/socat', '-d', '-d', f'TCP4-LISTEN:{SANDBOX_PORT},bind=127.0.0.1,reuseaddr,fork',
                                f'UNIX-CONNECT:{sock}'], 'namespace-relay.log')
        # Its log is node-local and the deadline is in seconds (synth.endpoint_relay; m1-hero #244: 3,328 attempts
        # failed a ~10 s loop over a log on /data).
        await relay.wait_for('listening on', 'namespace-relay.log', (ns,), 'Package proxy relay')

    async def upload_file(self, source_path, target_path):
        if str(target_path) == '/tmp/harbor-pi-agent/models.json' and not self._box.internet:
            models = json.loads(Path(source_path).read_text())
            provider = models.get('providers', {}).get('harbor-endpoint')
            if provider:
                relay = EndpointRelay(self.trial_paths.agent_dir / 'model-endpoint-relay')
                original = provider['baseUrl']
                provider['baseUrl'] = await relay.start(original, self._box.init_pid)
                self.__dict__.setdefault('_model_relays', []).append(relay)
                local = Path(source_path).with_name('models-local-endpoint.json')
                atomic(local, models)
                atomic(self.trial_paths.agent_dir / 'model-endpoint-transport.json', {
                    'original_endpoint': original, 'sandbox_endpoint': provider['baseUrl'],
                    'transport': 'loopback TCP to host Unix socket to the attempt proxy',
                    'task_internet_enabled': False, 'model_and_context_guard_unchanged': True})
                source_path = local
        await super().upload_file(source_path, target_path)

    async def stop(self, delete):
        try:
            for relay in self.__dict__.get('_model_relays', []):
                await relay.close()
            self.__dict__.pop('_model_relays', None)
            package = self.__dict__.pop('_package_proxy', None)
            if package:
                await package[1].close()
                await package[0].close()
        finally:
            await super().stop(delete)

    async def upload_dir(self, source_dir, target_dir):
        await super().upload_dir(source_dir, target_dir)
        # Harbor uploads /tests only after the agent has ended.
        if str(target_dir).rstrip('/') != '/tests':
            return
        caches = ((campaign_config() or {}).get('harbor') or {}).get('verifier_caches', {})
        task = self.environment_dir.parent.name
        config = read(caches[task], {}) if task in caches else None
        if not config:
            return
        started = time.time()
        archive = Path(config['archive'])
        if sha(archive) != config['sha256']:
            raise ValueError('Verifier dependency cache checksum mismatch')
        root = await self._mkdirs(VERIFIER_UV_CACHE)

        def unpack():
            with tarfile.open(archive) as source:
                source.extractall(root, filter='data')
        await asyncio.to_thread(unpack)
        self._verifier_uv_cache = VERIFIER_UV_CACHE
        atomic(self.trial_paths.verifier_dir / 'dependency-cache.json', {
            'time': time.time(), 'archive': str(archive), 'sha256': config['sha256'], 'task': task,
            'injected_after_agent': True, 'elapsed_sec': time.time() - started,
            'requirements': config.get('requirements'), 'verifier_command_unchanged': True,
            'verifier_timeout_unchanged': True})

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if '/tests/test.sh' not in command or 'chmod' in command:
            return await super().exec(command, cwd, env, timeout_sec, user)
        if getattr(self, '_verifier_uv_cache', None):
            env = {**(env or {}), 'UV_CACHE_DIR': self._verifier_uv_cache}
        retries = int(((campaign_config() or {}).get('harbor') or {}).get('verifier_infra_retries', 0))
        paths = self.trial_paths
        return await verify_with_retries(lambda: super(CampaignEnvironment, self).exec(command, cwd, env, timeout_sec, user),
                                         paths.test_stdout_path, (paths.reward_text_path, paths.reward_json_path),
                                         retries, paths.verifier_dir / 'infra-retries.jsonl')
