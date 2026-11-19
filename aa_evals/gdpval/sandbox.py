"""The task sandbox: Stirrup's code_exec in an Apptainer container (AA: an E2B sandbox).

Each `code_exec` command runs as its own `apptainer exec --containall --cleanenv` of the GDPval SIF (AA's pinned
Debian trixie + Python 3.13 environment, `sandbox/gdpval.def`), with two host dirs of the attempt bound in:
`home/` at `/home/user` (the default working directory AA's prompt names, where the reference files are) and `tmp/`
at `/tmp`. Everything else is the read-only image. As in AA's prompt, no shell state carries over between commands;
unlike E2B, a background process does not outlive its command (each exec has its own PID namespace).

The agent process and model client stay on the host. Each sandbox command runs in an owned user+network
namespace with no uplink, so changing proxy variables or opening raw sockets cannot restore egress. Namespace
creation failure never falls back to host networking. Setup, rendering and grading are outside this wrapper.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
from pathlib import Path

import anyio

from stirrup.tools.code_backends import LocalCodeExecToolProvider
from stirrup.tools.code_backends.base import CommandResult

HOME = '/home/user'
SHELL_TIMEOUT = 600   # AA: "Commands are terminated after 10 minutes"
# Per-command memory cap (resident set of the command's whole process tree). An E2B sandbox has a fixed RAM size,
# so a runaway command is OOM-killed and the model sees the error. Here, with no cap, one command reached 130 GB
# and took down a 22-attempt worker (Phase B, 2026-11-05). AA doesn't state its sandbox size; 16 GB is our
# default (Stirrup's E2B backend takes a template name; the template's RAM, AA's, is unpublished). Rootless apptainer
# can't set cgroup limits on this cluster (cgroup v1), so this is a watchdog. Scope: only the code_exec command's
# process tree, rooted at the `apptainer exec` this call spawns in its own session. Stirrup, the endpoint client and
# the worker are its ancestors, never counted or killed (the #145 lesson: TB's memcap once killed the agent).
MEM_LIMIT_BYTES = int(os.environ.get('GDPVAL_SANDBOX_MEM_GB', '16')) * 1024 ** 3
MEM_POLL_SEC = 2.0
_PAGE = os.sysconf('SC_PAGE_SIZE')
PROXY = 'http://proxy.internal:8080'
NO_PROXY = 'localhost,127.0.0.1'


class ApptainerCodeExecToolProvider(LocalCodeExecToolProvider):
    def __init__(self, *, sif: str, home: Path, tmp: Path, shell_timeout: int = SHELL_TIMEOUT,
                 proxy: str | None = PROXY, apptainer: str = 'apptainer', stats: dict | None = None,
                 offline: bool = True, unshare: str = '/usr/bin/unshare', runtime_cache: dict | None = None):
        super().__init__(shell_timeout=shell_timeout)
        self.sif = str(sif)
        self.home = Path(home)
        self.tmp = Path(tmp)
        self.proxy = proxy
        self.offline, self.unshare = offline, unshare
        self.apptainer = apptainer
        self.stats = stats if stats is not None else {}
        self.stats['network_mode'] = 'offline' if offline else 'legacy-online'
        if offline:
            self.stats['network_enforcement'] = 'owned-user-and-network-namespace-no-uplink'
        self.runtime_cache = runtime_cache
        if runtime_cache:
            self.stats['runtime_cache'] = runtime_cache['receipt']
        self.pid_ns = True   # --containall; dropped (with a stat) where the host forbids mounting /proc

    async def __aenter__(self):
        self.home.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self._temp_dir = self.home
        # Stirrup's default description (AA ran Stirrup's E2B backend, which uses the same default)
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._temp_dir = None   # the runner packs home/ after the session; nothing is deleted here

    def _check_absolute_paths(self, argv, cmd):
        return None   # the container is the sandbox: absolute paths are its own

    def _resolve_and_validate_path(self, path: str) -> Path:
        if self._temp_dir is None:
            raise RuntimeError('ExecutionEnvironment not started.')
        p = str(path)
        if p == HOME or p.startswith(HOME + '/'):
            host, root = self.home / p[len(HOME):].lstrip('/'), self.home
        elif p == '/tmp' or p.startswith('/tmp/'):
            host, root = self.tmp / p[len('/tmp'):].lstrip('/'), self.tmp
        elif not p.startswith('/'):
            host, root = self.home / p, self.home
        else:
            raise ValueError(f'Path is outside the writable sandbox (/home/user, /tmp): {path}')
        try:
            host.resolve().relative_to(root.resolve())
        except ValueError as e:
            raise ValueError(f'Path is outside execution environment: {path}') from e
        return host

    def output_source_roots(self):
        """Submitted files may live under /home/user or /tmp (both are host dirs of the attempt)."""
        from stirrup.tools.code_backends.base import OutputSourceRoot
        if self._temp_dir is None:
            return ()
        # container paths (the model's spelling); _resolve_and_validate_path maps them to the host dirs
        return (OutputSourceRoot(path=HOME), OutputSourceRoot(path='/tmp'))

    def argv(self, cmd: str) -> list[str]:
        env = ['--env', f'HOME={HOME}']
        if self.proxy and not self.offline:
            for k in ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY'):
                env += ['--env', f'{k}={self.proxy}']
            env += ['--env', f'no_proxy={NO_PROXY}', '--env', f'NO_PROXY={NO_PROXY}']
        binds = []
        if self.runtime_cache:
            for name, value in self.runtime_cache['environment'].items():
                env += ['--env', f'{name}={value}']
            binds = ['-B', f"{self.runtime_cache['payload_root']}:/opt/aa-runtime-cache:ro"]
        iso = ['--containall'] if self.pid_ns else ['--contain', '--ipc', '--cleanenv']
        prefix = [self.unshare, '--user', '--map-root-user', '--net', '--'] if self.offline else []
        return [*prefix, self.apptainer, '--silent', 'exec', *iso, '--cleanenv', '--no-eval', '--no-nv',
                '-B', f'{self.home}:{HOME}', '-B', f'{self.tmp}:/tmp', *binds, '--pwd', HOME, *env,
                self.sif, 'bash', '-c', cmd]

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        if self._temp_dir is None:
            raise RuntimeError('ExecutionEnvironment not started.')
        timeout = timeout or self._shell_timeout
        _, rejection = self._prepare_command(cmd)
        if rejection is not None:
            return rejection
        self.stats['commands'] = self.stats.get('commands', 0) + 1
        process = None
        try:
            with anyio.fail_after(timeout):
                process = await anyio.open_process(self.argv(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                   cwd=self.home, start_new_session=True)
                out: list[bytes] = []
                err: list[bytes] = []

                async def rd(stream, buf):
                    if stream:
                        buf.extend([c async for c in stream])

                over = {}

                async def watch():
                    while process.returncode is None:
                        rss, pids = await anyio.to_thread.run_sync(_tree_rss, process.pid)
                        self.stats['max_command_rss_gb'] = max(self.stats.get('max_command_rss_gb', 0),
                                                               round(rss / 1024 ** 3, 2))
                        if rss > MEM_LIMIT_BYTES:
                            over['rss'] = rss
                            with contextlib.suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                            for p in pids:
                                with contextlib.suppress(ProcessLookupError, PermissionError):
                                    os.kill(p, signal.SIGKILL)
                            return
                        await anyio.sleep(MEM_POLL_SEC)

                async with anyio.create_task_group() as tg:   # as before: wait for EOF on both streams
                    tg.start_soon(rd, process.stdout, out)
                    tg.start_soon(rd, process.stderr, err)
                    tg.start_soon(watch)                           # ends when the process exits (or is killed)
                await process.wait()
                stderr = b''.join(err).decode('utf-8', errors='replace')
                if over:
                    self.stats['memory_kills'] = self.stats.get('memory_kills', 0) + 1
                    lim = MEM_LIMIT_BYTES / 1024 ** 3
                    return CommandResult(exit_code=137, stdout=b''.join(out).decode('utf-8', errors='replace'),
                                         stderr=stderr + f'\nKilled: the command exceeded the sandbox memory limit '
                                                         f'({lim:.0f} GB)', error_kind='execution_error')
                if process.returncode == 255 and self.pid_ns and "can't mount proc" in stderr:
                    self.pid_ns = False
                    self.stats['no_pid_namespace'] = True
                    return await self.run_command(cmd, timeout=timeout)
                if (process.returncode == 255 and _apptainer_failed(stderr)) or (
                        self.offline and process.returncode and stderr.lstrip().startswith('unshare:')):
                    self.stats['sandbox_errors'] = self.stats.get('sandbox_errors', 0) + 1
                    self.stats['last_sandbox_error'] = stderr[-500:]
                return CommandResult(exit_code=process.returncode or 0,
                                     stdout=b''.join(out).decode('utf-8', errors='replace'), stderr=stderr)
        except TimeoutError:
            self.stats['timeouts'] = self.stats.get('timeouts', 0) + 1
            if process:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                with anyio.move_on_after(5):
                    await process.wait()
            return CommandResult(exit_code=1, stdout='', stderr=f'Command timed out after {timeout} seconds',
                                 error_kind='timeout')
        except Exception as exc:  # noqa: BLE001
            self.stats['sandbox_errors'] = self.stats.get('sandbox_errors', 0) + 1
            self.stats['last_sandbox_error'] = repr(exc)[:500]
            return CommandResult(exit_code=1, stdout='', stderr=str(exc), error_kind='execution_error')


def _tree_rss(root: int) -> tuple[int, list[int]]:
    """(resident bytes, pids) of `root` and all its descendants, from /proc."""
    children: dict[int, list[int]] = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat', 'rb') as f:
                st = f.read()
            ppid = int(st[st.rindex(b')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    pids, todo = [], [root]
    while todo:
        p = todo.pop()
        pids.append(p)
        todo.extend(children.get(p, []))
    rss = 0
    for p in pids:
        try:
            with open(f'/proc/{p}/statm') as f:
                rss += int(f.read().split()[1]) * _PAGE
        except (OSError, ValueError, IndexError):
            pass
    return rss, pids


def _apptainer_failed(stderr: str) -> bool:
    """The container itself failed to start (not the model's command)."""
    s = stderr.lower()
    return 'fatal:' in s and ('container' in s or 'apptainer' in s or 'image' in s)
