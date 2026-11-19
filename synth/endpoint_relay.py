"""Expose only the existing model proxy inside a sandbox with no network egress."""
import asyncio
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

# How long a relay's socat may take to report "listening on" (SYNTH_RELAY_READY_SEC). m1-hero (#244): 5,664 attempts
# (4.6%) failed before the agent's first request because a relay missed a ~10 s readiness loop on a host whose /data
# client was saturated (0.4% of attempts with one campaign worker on the host, 13% with five).
READY_SEC = 60.0


def ready_seconds():
    return float(os.environ.get('SYNTH_RELAY_READY_SEC', READY_SEC))


class EndpointRelay:
    """socat processes of one attempt. Their logs are written to node-local scratch and copied to log_dir when the
    relay closes: socat's readiness line is read from its log, and a write to the attempt directory on /data could
    block it past the deadline (m1-hero: the first log line arrived, "listening on" never did)."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.processes = []
        self.handles = []
        self.logs = []
        self.scratch = None

    def local_dir(self):
        if self.scratch is None:
            self.scratch = tempfile.TemporaryDirectory(prefix='synth-relay-', dir='/tmp')
        return Path(self.scratch.name)

    def log_text(self, name):
        try:
            return (self.local_dir() / name).read_text(errors='replace')
        except OSError:
            return ''

    async def spawn(self, args, name):
        handle = (self.local_dir() / name).open('ab')
        self.handles.append(handle)
        if name not in self.logs:
            self.logs.append(name)
        process = await asyncio.create_subprocess_exec(*args, stdout=handle,
            stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        self.processes.append(process)
        return process

    async def wait_for(self, pattern, name, processes, what, timeout=None, ready=None):
        """The first match of `pattern` in log `name` once `ready(match)` (default: always) holds; raises
        RuntimeError when one of `processes` exits first and TimeoutError after `timeout` seconds."""
        deadline = time.monotonic() + (ready_seconds() if timeout is None else timeout)
        while True:
            if any(p.returncode is not None for p in processes):
                raise RuntimeError(f'{what} exited; inspect {self.log_dir}')
            match = re.search(pattern, self.log_text(name))
            if match and (ready is None or ready(match)):
                return match
            if time.monotonic() >= deadline:
                raise TimeoutError(f'{what} did not become ready; inspect {self.log_dir}')
            await asyncio.sleep(.1)

    async def start(self, endpoint, init_pid):
        parsed = urlsplit(endpoint)
        if parsed.scheme != 'http' or not parsed.hostname or not parsed.port or parsed.username:
            raise ValueError('Namespace model relay requires the local HTTP proxy endpoint')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Unix sockets connect across network namespaces without adding a route,
        # slirp/NAT, or internet access to the task. Keep the socket path short.
        self.scratch = tempfile.TemporaryDirectory(prefix='synth-model-', dir='/tmp')
        sock = Path(self.scratch.name) / 'proxy.sock'
        try:
            host = await self.spawn(['/usr/bin/socat',
                f'UNIX-LISTEN:{sock},fork,mode=0600',
                f'TCP4:{parsed.hostname}:{parsed.port}'], 'host-relay.log')
            ns = await self.spawn(['nsenter', '-t', str(init_pid), '-U', '-n',
                '--preserve-credentials', '--', '/usr/bin/socat', '-d', '-d',
                'TCP4-LISTEN:0,bind=127.0.0.1,reuseaddr,fork',
                f'UNIX-CONNECT:{sock}'], 'namespace-relay.log')
            match = await self.wait_for(r'listening on AF=2 127\.0\.0\.1:(\d+)', 'namespace-relay.log', (host, ns),
                                        'Model relay', ready=lambda _: sock.exists())
            return urlunsplit(parsed._replace(netloc=f'127.0.0.1:{match[1]}'))
        except BaseException:
            await self.close()
            raise

    def copy_logs(self):
        if self.scratch is None:
            return
        for name in self.logs:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(self.scratch.name) / name, self.log_dir / name)
            except OSError:
                pass

    async def close(self):
        for process in reversed(self.processes):
            # socat forks per connection. Its owned process group includes all
            # children, even when the listener has already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 3)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
        self.processes.clear()
        for handle in self.handles:
            handle.close()
        self.handles.clear()
        if self.scratch is not None:
            await asyncio.to_thread(self.copy_logs)
            self.scratch.cleanup()
            self.scratch = None
        self.logs.clear()
