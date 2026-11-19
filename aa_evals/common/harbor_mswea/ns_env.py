"""harbor_ns environment with Harbor's compose-sidecar operations.

    harbor run --env aa_evals.common.harbor_mswea.ns_env:NsSidecarEnvironment ...

harbor_ns (`harbor_ns.environment.NsEnvironment`) runs every compose service in its own sandbox but only
implements exec/upload/download for `main`.  Harbor 0.24 also needs, for tasks whose artifacts name a
sidecar (`artifacts = [{ source = "/state/x", service = "submission" }]`) or that define sidecar collect
hooks, the per-service operations below.  Without them the sidecar artifacts are silently "failed" in
artifacts/manifest.json, the separate verifier never receives them, and the trial scores 0 even for the
reference solution (TB-Science v0.1.0: longitudinal-clinical-agent, protein-active-learning,
tamp-skill-planning; TB 4.0: 8 tasks with sidecar artifacts).  From aa-tbsci (aa_evals/tbsci/ns_env.py, 9332590).

  service_exec            nsenter into the service's sandbox (as root; Harbor collects as root)
  service_download_file   copy out of /proc/<service init>/root
  service_download_dir    same, recursively
  [environment.env]       for compose tasks, added to main's exec env as Docker does (harbor_ns only
                          interpolated it into the compose file)
  network_mode            compose `network_mode: "service:X"`: a relay (netns_relay.py) makes X's localhost
                          ports reachable on the service's localhost (harbor_ns gives each service its own
                          netns; TB 4.0 ctr-optimization and nextjs-performance; added by aa-tb4)
  /etc/pip.conf proxy     every service gets pip's global `proxy` (when its image's pip config sets none):
                          sandboxes reach the internet only through the cluster proxy, and scripts that run
                          pip under `env -i` (TB 4.0 payments-pipeline-fix's verifier) drop the proxy
                          variables that harbor_ns sets (added by aa-tb4)
  no_proxy <local>        every service's no_proxy gets `<local>` (Chromium: hostnames without a dot bypass the
                          proxy) and the task's compose service names (curl/Python/pip match names exactly).  harbor_ns adds /etc/hosts names when an exec starts; a name a script maps
                          later (TB 4.0 cumulative-layout-shift's verifier: `127.0.0.1 barber-shop-data-backend`,
                          then a headless browser fetches it) would otherwise go to the cluster proxy and fail
                          (added by aa-tb4; payments-pipeline-fix's verifier: `127.0.0.1 kafka customer`)
  stop_service            kill the service's sandbox (Harbor stops `main` before collecting sidecar
                          evidence when the verifier is separate, as `docker compose stop main` would)

service_download_dir_with_exclusions and service_is_dir come from Harbor's base class (built on
service_exec + service_download_file).  This wraps harbor_ns without modifying it.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
import re
import shutil
import signal
import sys
from pathlib import Path

import yaml

from harbor.environments.base import ExecResult
from harbor.models.task.config import NetworkMode, NetworkPolicy

from harbor_ns.builder import resolve_in_root
from harbor_ns.environment import NsEnvironment


class NsSidecarEnvironment(NsEnvironment):
    def __init__(self, *args, **kwargs):
        capture_plan = kwargs.pop('required_output_capture_plan', None)
        capture_sha = kwargs.pop('required_output_capture_sha256', None)
        super().__init__(*args, **kwargs)
        self._required_outputs = None
        if capture_plan is not None or capture_sha is not None:
            if not capture_plan or not capture_sha:
                raise ValueError('required-output capture needs both plan and hash')
            from aa_evals.common.pi_eval.required_outputs import Capture, load_plan
            self._required_outputs = Capture(load_plan(
                capture_plan, capture_sha, self.environment_dir, self.trial_paths.trial_dir))
        self._offline = None
        # Harbor's base env adds task.toml [environment.env] to every exec only for non-compose tasks;
        # for compose tasks the provider must (Docker puts it in main's env).  harbor_ns only uses it to
        # interpolate the compose file, so main never saw it (TB-Science tamp-skill-planning:
        # TAMP_SIM_URL=http://sim:8788 -> the reference solution fell back to 127.0.0.1 and failed).
        if self._uses_compose and self.task_env_config.env:
            from harbor.utils.env import resolve_env_vars  # noqa: PLC0415

            self._persistent_env = {**resolve_env_vars(self.task_env_config.env), **self._persistent_env}

    def _shared_net(self) -> list[tuple[str, str]]:
        """(service, target) pairs of compose `network_mode: "service:<target>"`."""
        if not self._uses_compose:
            return []
        try:
            data = yaml.safe_load(self._compose_file.read_text()) or {}
        except (OSError, yaml.YAMLError):
            return []
        out = []
        for name, spec in (data.get('services') or {}).items():
            mode = str((spec or {}).get('network_mode') or '')
            if mode.startswith('service:'):
                out.append((name, mode.split(':', 1)[1]))
        return out

    def _seed_pip_proxy(self) -> None:
        # Explicit task-scoped compatibility opt-out. Existing image config is
        # never removed/rewritten; other tasks retain env -i evaluator pip support.
        if os.environ.get('HARBOR_NS_SKIP_PIP_PROXY_SEED') == '1':
            return
        for s in (getattr(self, '_services', None) or {}).values():
            box = s.box
            proxy = (box.base_env.get('https_proxy') or box.base_env.get('HTTPS_PROXY')) if box else None
            if not (box and box.init_pid and proxy):
                continue
            try:
                f = resolve_in_root(Path(f'/proc/{box.init_pid}/root'), '/etc/pip.conf')
                if not f.parent.is_dir():
                    continue
                old = f.read_text(errors='replace') if f.is_file() else ''
                if re.search(r'^\s*proxy\s*=', old, re.M):
                    continue
                if re.search(r'^\s*\[global\]', old, re.M):
                    new = re.sub(r'^(\s*\[global\][^\n]*\n?)', lambda m: m.group(1).rstrip('\n') + f'\nproxy = {proxy}\n',
                                 old, count=1, flags=re.M)
                else:
                    new = old + ('\n' if old and not old.endswith('\n') else '') + f'[global]\nproxy = {proxy}\n'
                f.write_text(new)
            except OSError:
                pass

    def _task_service_names(self) -> list[str]:
        """Service names of the task's agent-side compose file (for a separate verifier, environment_dir is the
        task's tests/: its scripts often alias those names to 127.0.0.1 and run the services themselves)."""
        names: list[str] = []
        for d in {self.environment_dir, self.environment_dir.parent / 'environment'}:
            for f in ('docker-compose.yaml', 'docker-compose.yml'):
                try:
                    data = yaml.safe_load((Path(d) / f).read_text()) or {}
                except (OSError, yaml.YAMLError):
                    continue
                names += [n for n in (data.get('services') or {}) if n not in names]
        return names

    def _local_no_proxy(self) -> None:
        extra = ['<local>', *self._task_service_names()]
        for s in (getattr(self, '_services', None) or {}).values():
            if s.box is None:
                continue
            for k in ('no_proxy', 'NO_PROXY'):
                v = s.box.base_env.get(k)
                if v is not None:
                    cur = v.split(',')
                    s.box.base_env[k] = ','.join(cur + [x for x in extra if x not in cur])

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        if getattr(self, '_required_outputs', None) is not None:
            rootfs = getattr(getattr(self, '_main', None), 'rootfs', None)
            self._required_outputs.bind_root(self._root(), {
                'environment_name': self.environment_name, 'session_id': self.session_id,
                'trial_dir': str(self.trial_paths.trial_dir),
                'native_rootfs': str(rootfs) if rootfs is not None else None,
                'native_rootfs_key': Path(rootfs).parent.name if rootfs is not None else None},
                sandbox=self._box)
        self._seed_pip_proxy()
        self._local_no_proxy()
        self._relays = []
        for name, target in self._shared_net():
            a, b = (getattr(self, '_services', {}) or {}).get(name), (getattr(self, '_services', {}) or {}).get(target)
            if not (a and b and a.box and b.box and a.box.init_pid and b.box.init_pid and self._pod is not None):
                self.logger.warning(f'network_mode service:{target} for {name}: services not running, no relay')
                continue
            log = open(Path(self._pod_dir or '/tmp') / f'relay-{name}-{target}.log', 'ab')
            proc = await asyncio.create_subprocess_exec(
                'nsenter', '-t', str(self._pod.holder_pid), '-U', '--preserve-credentials', '--',
                sys.executable, '-I', str(Path(__file__).with_name('netns_relay.py')),
                str(a.box.init_pid), str(b.box.init_pid),
                stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            self._relays.append(proc)
            # the target is healthy (depends_on) but give the relay a moment to bind its listeners
            for _ in range(20):
                await asyncio.sleep(0.25)
                if proc.returncode is not None or log.tell() or os.path.getsize(log.name):
                    break

    async def stop(self, delete: bool) -> None:
        if getattr(self, '_required_outputs', None) is not None:
            self._required_outputs.close()
        if self._offline is not None:
            await self._offline.close()
            self._offline = None
        for proc in getattr(self, '_relays', []) or []:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
        self._relays = []
        await super().stop(delete)

    @asynccontextmanager
    async def offline_agent_model(self, backend):
        """Disable agent egress permanently, keeping only a fixed, local model API during the agent phase."""
        from .offline import ModelRelay, OfflineNetworks
        if self._offline is not None:
            raise RuntimeError('offline agent network already initialized')
        previous = self.network_policy
        self._offline = OfflineNetworks(self)
        relay = ModelRelay(self._box, self._pod, backend)
        try:
            await self._offline.isolate()
            await self.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
            yield await relay.start()
        finally:
            await relay.close()
            self._offline.agent_active = False
            # This restores evaluator permissions, never the original agent namespaces' NAT.
            await self.set_network_policy(previous)

    async def _apply_network_policy(self, network_policy):
        if self._offline is None:
            return await super()._apply_network_policy(network_policy)
        self._offline.evaluator_internet = network_policy.network_mode != NetworkMode.NO_NETWORK
        await self._offline.isolate()

    async def _select_exec_network(self, box):
        box.exec_network_pid = await self._offline.evaluator_network(box) if self._offline is not None else None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        await self._select_exec_network(self._box)
        return await super().exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)

    def _service(self, service: str):
        s = (getattr(self, "_services", None) or {}).get(service)
        if s is None:
            raise RuntimeError(f"compose service {service!r} is not defined for this environment")
        if s.box is None or s.box.init_pid is None:
            raise RuntimeError(f"compose service {service!r} is not running")
        return s

    def _service_root(self, service: str) -> Path:
        return Path(f"/proc/{self._service(service).box.init_pid}/root")

    async def service_exec(self, command: str, *, service: str | None = None, cwd: str | None = None,
                           env: dict[str, str] | None = None, timeout_sec: int | None = None,
                           user: str | int | None = None) -> ExecResult:
        if self.is_main_service(service):
            return await self.exec(command, cwd=cwd, env=env, timeout_sec=timeout_sec, user=user)
        s = self._service(service)
        await self._select_exec_network(s.box)
        workdir = cwd or s.working_dir or (getattr(s.meta, "workdir", None) if s.meta else None) or "/"
        r = await s.box.exec(command, cwd=workdir, env=self._merge_env(env) or {}, timeout=timeout_sec)
        return ExecResult(stdout=r.stdout, stderr=r.stderr, return_code=r.return_code)

    async def service_is_dir(self, path: str, *, service: str | None = None,
                             user: str | int | None = None) -> bool:
        capture = getattr(self, '_required_outputs', None)
        if capture is not None and capture.selected(path, service):
            return await asyncio.wait_for(asyncio.to_thread(capture.is_dir, self._root(), path), 10)
        return await super().service_is_dir(path, service=service, user=user)

    async def service_download_file(self, source_path: str, target_path: Path | str, *,
                                    service: str | None = None) -> None:
        capture = getattr(self, '_required_outputs', None)
        if capture is not None and capture.selected(source_path, service):
            await asyncio.wait_for(asyncio.to_thread(
                capture.materialize, self._root(), source_path, target_path), 30)
            return
        if self.is_main_service(service):
            return await self.download_file(source_path, target_path)
        src = resolve_in_root(self._service_root(service), source_path)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_running_loop().run_in_executor(None, shutil.copy2, src, target_path)

    async def service_download_dir(self, source_dir: str, target_dir: Path | str, *,
                                   service: str | None = None) -> None:
        capture = getattr(self, '_required_outputs', None)
        if capture is not None and capture.selected(source_dir, service):
            # Required filenames must never trigger recursive workspace copies.
            await asyncio.wait_for(asyncio.to_thread(
                capture.materialize, self._root(), source_dir, target_dir), 30)
            return
        if self.is_main_service(service):
            return await self.download_dir(source_dir, target_dir)
        src = resolve_in_root(self._service_root(service), source_dir)
        tgt = Path(target_dir)

        def _copy():
            tgt.mkdir(parents=True, exist_ok=True)
            for r, _dirs, files in os.walk(src):
                rel = Path(r).relative_to(src)
                (tgt / rel).mkdir(parents=True, exist_ok=True)
                for f in files:
                    sp, dp = Path(r) / f, tgt / rel / f
                    if dp.is_symlink() or dp.exists():
                        dp.unlink()
                    if sp.is_symlink():
                        os.symlink(os.readlink(sp), dp)
                    elif sp.is_file():
                        shutil.copy2(sp, dp)

        await asyncio.get_running_loop().run_in_executor(None, _copy)

    async def stop_service(self, service: str) -> None:
        s = (getattr(self, "_services", None) or {}).get(service)
        if s is None or s.box is None:
            return
        # delete=False: NsEnvironment.stop() still owns the sandbox directory and removes it.
        await s.box.stop(delete=False)
