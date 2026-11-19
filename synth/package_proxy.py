"""Package-registry proxy for sandboxes without network (campaign key harbor.package_proxy).

The agent's sandbox has no route out. This forward proxy listens on a host Unix socket; a socat in the sandbox's
network namespace exposes it at 127.0.0.1:3128 (the same relay as the model endpoint, synth.endpoint_relay), and the
sandbox's proxy variables point there (HARBOR_NS_OFFLINE_PROXY_URL). Requests to the allowed registries are chained to
the cluster proxy; everything else gets 403. One request per connection for plain HTTP (so each one is checked);
HTTPS is a CONNECT tunnel to an allowed host on port 443. Every request is logged as a JSON line (time, method, host,
target, allowed, status) so a task's fetches can be pinned later.

With the job's package cache (synth.package_cache, campaign harbor.package_cache, on by default) allowed plain-HTTP
requests (apt) go to the cache instead, which serves repeats from node-local disk; the log then also has the cache's
verdict (hit, miss, ...) and the file's SHA-256. If the cache is down, the request goes to the cluster proxy as before.
harbor.package_cache_pip (off by default) also gives the sandbox PIP_INDEX_URL=http://127.0.0.1:3128/pypi/simple/,
served from the cache (or straight from PyPI when the cache is down): off because the variable overrides any index a
task's own pip.conf names (pip-configuration tasks exist) and pip then prints "Looking in indexes", visible in traces.

  python -m synth.package_proxy selftest      # through the cluster proxy, from this host
  python -m synth.package_proxy sandboxtest   # in an offline harbor_ns sandbox (CPU job)
"""
import asyncio
import ipaddress
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

SANDBOX_PORT = 3128
SANDBOX_URL = f'http://127.0.0.1:{SANDBOX_PORT}'
DEFAULT_ALLOW = ('archive.ubuntu.com', 'security.ubuntu.com', 'pypi.org', 'files.pythonhosted.org',
                 'registry.npmjs.org')
HEAD_LIMIT = 64 << 10


def allowed(host, allow):
    host = (host or '').lower().rstrip('.')
    return any(host == a or host.endswith('.' + a) for a in allow)


def local_host(host):
    """A name or address only the sandbox itself can reach: loopback/private/link-local IPs, unqualified names, and
    special-use or private domains (harbor_ns.sandbox: the offline no_proxy list)."""
    from harbor_ns.sandbox import OFFLINE_DOMAINS, SPECIAL_DOMAINS
    host = (host or '').lower().rstrip('.').strip('[]')
    try:
        ip = ipaddress.ip_address(host)
        return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified
    except ValueError:
        return '.' not in host or host == 'localhost' or host.endswith((*SPECIAL_DOMAINS, *OFFLINE_DOMAINS))


def refusal(host):
    """The 403 reason. S1 traces (2026-10-31): agents reached 127.0.0.1 through this proxy with malformed URLs
    (`http://127.0.0.1:18083http://...`, `:8710v1/...`: no numeric port, so no_proxy never matched) or with
    `curl --noproxy "'*'"` (a quoted pattern from an unquoted variable, which replaces no_proxy and matches nothing),
    and saw only "only the registries are reachable"."""
    if local_host(host):
        shown = re.sub(r'[^\x21-\x7e]', '?', host or '')[:80]
        return (f'Forbidden: {shown} is local to the sandbox; connect to it directly, not through this proxy '
                f'(check the URL and no_proxy)')
    return 'Forbidden: only the Ubuntu, PyPI and npm registries are reachable'


async def attach(box, log_dir, upstream, allow=DEFAULT_ALLOW, cache=None, pip=False, tag=None, log_path=None):
    """A PackageProxy for a running harbor_ns sandbox, at 127.0.0.1:3128 in its network namespace: the proxy listens
    on a host Unix socket and a socat in the sandbox's namespace relays to it. -> (proxy, relay); close both (relay
    first). Raises if the relay does not come up (nothing is left running then)."""
    import tempfile
    from .endpoint_relay import EndpointRelay
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    relay = EndpointRelay(log_dir)
    relay.scratch = tempfile.TemporaryDirectory(prefix='synth-pkg-', dir='/tmp')
    sock = Path(relay.scratch.name) / 'p.sock'
    proxy = await PackageProxy(sock, log_path or log_dir / 'requests.jsonl', upstream, allow, cache=cache, pip=pip,
                               tag=tag).start()
    try:
        ns = await relay.spawn(['nsenter', '-t', str(box.init_pid), '-U', '-n', '--preserve-credentials', '--',
                                '/usr/bin/socat', '-d', '-d', f'TCP4-LISTEN:{SANDBOX_PORT},bind=127.0.0.1,reuseaddr,fork',
                                f'UNIX-CONNECT:{sock}'], 'namespace-relay.log')
        for _ in range(100):
            if ns.returncode is not None:
                raise RuntimeError(f'package proxy relay exited; inspect {log_dir}')
            if 'listening on' in (log_dir / 'namespace-relay.log').read_text(errors='replace'):
                return proxy, relay
            await asyncio.sleep(.1)
        raise TimeoutError(f'package proxy relay did not become ready; inspect {log_dir}')
    except BaseException:
        await relay.close()
        await proxy.close()
        raise


async def read_head(reader):
    head = await reader.readuntil(b'\r\n\r\n')
    if len(head) > HEAD_LIMIT:
        raise ValueError('request head too large')
    return head


async def pipe(reader, writer):
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


PIP_INDEX = f'{SANDBOX_URL}/pypi/simple/'


def cache_headers(head):
    """(status, X-Pkgcache, X-Pkgcache-Sha256) of a response head from the package cache."""
    lines = head.decode('latin-1').split('\r\n')
    try:
        status = int(lines[0].split(' ', 2)[1])
    except (IndexError, ValueError):
        status = None
    h = {k.strip().lower(): v.strip() for k, _, v in (l.partition(':') for l in lines[1:]) if k}
    return status, h.get('x-pkgcache'), h.get('x-pkgcache-sha256')


class PackageProxy:
    def __init__(self, socket_path, log_path, upstream, allow=DEFAULT_ALLOW, cache=None, pip=False, tag=None):
        """allow: the hosts this sandbox may reach, or None for any host that is not local to the sandbox (a sandbox
        with internet, e.g. a task author's: its plain HTTP comes here only to reach the cache); cache: a callable
        returning the package cache's registry (harbor_ns.sandbox.package_cache) or None while it is down; pip: also
        serve the PyPI index at /pypi/ (origin-form requests to 127.0.0.1:3128); tag: label of this sandbox's requests
        in the cache's log."""
        self.socket_path, self.log_path = str(socket_path), Path(log_path)
        self.upstream = upstream
        up = urlsplit(upstream)
        self.up_host, self.up_port = up.hostname, up.port or 8080
        self.allow = None if allow is None else tuple(allow)
        self.cache, self.pip, self.tag = cache, pip, tag
        self.direct = None   # storeless synth.package_cache.PackageCache: PyPI while the cache is down
        self.server = None

    async def start(self):
        self.server = await asyncio.start_unix_server(self.handle, path=self.socket_path)
        os.chmod(self.socket_path, 0o600)
        return self

    async def close(self):
        if self.server is not None:
            self.server.close()
            try:
                await asyncio.wait_for(self.server.wait_closed(), 5)
            except asyncio.TimeoutError:
                pass
            self.server = None

    def log(self, **entry):
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({'time': time.time(), **entry}) + '\n')
        except OSError:
            pass

    @staticmethod
    async def refuse(writer, code, reason):
        body = f'{reason}\n'.encode()
        writer.write(f'HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n'
                     f'Connection: close\r\n\r\n'.encode() + body)
        try:
            await writer.drain()
        finally:
            writer.close()

    async def handle(self, reader, writer):
        try:
            head = await asyncio.wait_for(read_head(reader), 60)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError,
                ConnectionError):
            writer.close()
            return
        line, *headers = head.decode('latin-1').split('\r\n')
        try:
            method, target, version = line.split(' ', 2)
        except ValueError:
            return await self.refuse(writer, 400, 'Bad Request')
        if self.pip and target.startswith('/pypi/'):
            return await self.pypi(head, method, target, version, headers, reader, writer)
        if method == 'CONNECT':
            host, _, port = target.rpartition(':')
            ok = not local_host(host) if self.allow is None else allowed(host, self.allow) and port == '443'
            cacheable = False
        else:
            url = urlsplit(target)
            host = url.hostname
            try:
                port = url.port or 80
            except ValueError:   # a port that is not a number
                port = None
            cacheable = url.scheme == 'http' and port == 80
            ok = url.scheme == 'http' and not local_host(host) if self.allow is None else \
                cacheable and allowed(host, self.allow)
        if not ok:
            self.log(method=method, host=host, target=target[:300], allowed=False, status=403)
            return await self.refuse(writer, 403, refusal(host))
        if cacheable and await self.via_cache(method, target, version, headers, host, reader, writer):
            return
        try:
            up_reader, up_writer = await asyncio.wait_for(asyncio.open_connection(self.up_host, self.up_port), 30)
        except (OSError, asyncio.TimeoutError):
            self.log(method=method, host=host, target=target[:300], allowed=True, status=502)
            return await self.refuse(writer, 502, 'Bad Gateway')
        if method == 'CONNECT':
            up_writer.write(f'CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n'.encode())
            await up_writer.drain()
            try:
                reply = await asyncio.wait_for(read_head(up_reader), 60)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError, ConnectionError):
                up_writer.close()
                self.log(method=method, host=host, target=target, allowed=True, status=504)
                return await self.refuse(writer, 504, 'Gateway Timeout')
            status = int(reply.split(b' ', 2)[1]) if reply.count(b' ') >= 1 else 502
            self.log(method=method, host=host, target=target, allowed=True, status=status)
            if status != 200:
                up_writer.close()
                return await self.refuse(writer, 502, 'Bad Gateway')
            writer.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
            await writer.drain()
            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
            return
        # plain HTTP: forward this one request with Connection: close, then relay the response
        kept = [h for h in headers if h and h.split(':', 1)[0].strip().lower()
                not in ('connection', 'proxy-connection', 'keep-alive')]
        length = next((int(h.split(':', 1)[1]) for h in kept if h.lower().startswith('content-length:')), 0)
        body = await reader.readexactly(length) if length else b''
        up_writer.write(('\r\n'.join([f'{method} {target} {version}', *kept, 'Connection: close', '', ''])).encode('latin-1')
                        + body)
        await up_writer.drain()
        self.log(method=method, host=host, target=target[:300], allowed=True, status=None)
        await pipe(up_reader, writer)
        up_writer.close()


    def cache_port(self):
        """Where the job's package cache listens: its unix socket (no ephemeral ports: every request is a new
        connection, and 2,400 sandboxes at 1,400 requests/s ran a loopback port range dry), else its TCP port."""
        reg = self.cache() if self.cache else None
        if not reg:
            return None
        sock = reg.get('sock')
        return sock if sock and os.path.exists(sock) else int(reg['port'])

    async def forward_to_cache(self, port, request_head, body, reader, writer):
        """Send one request to the package cache and relay its response. -> (status, cache verdict, sha256) or None
        when the cache cannot be reached (nothing was sent to the client then). `port`: TCP port or unix socket."""
        try:
            conn = asyncio.open_unix_connection(port) if isinstance(port, str) else \
                asyncio.open_connection('127.0.0.1', port)
            c_reader, c_writer = await asyncio.wait_for(conn, 5)
        except (OSError, asyncio.TimeoutError):
            return None
        try:
            c_writer.write(request_head + body)
            await c_writer.drain()
            try:
                resp = await asyncio.wait_for(read_head(c_reader), 900)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, ValueError, ConnectionError):
                return None
            writer.write(resp)
            await writer.drain()
            await pipe(c_reader, writer)
            return cache_headers(resp)
        finally:
            c_writer.close()

    def cache_head(self, method, target, version, headers):
        kept = [h for h in headers if h and h.split(':', 1)[0].strip().lower()
                not in ('connection', 'proxy-connection', 'keep-alive', 'x-pkgcache-tag')]
        tag = [f'X-Pkgcache-Tag: {self.tag}'] if self.tag else []
        return '\r\n'.join([f'{method} {target} {version}', *kept, *tag, 'Connection: close', '', '']).encode('latin-1')

    async def via_cache(self, method, target, version, headers, host, reader, writer):
        """An allowed plain-HTTP request through the package cache. -> True if answered, False to use the proxy."""
        port = self.cache_port()
        if port is None:
            return False
        length = next((int(h.split(':', 1)[1]) for h in headers if h.lower().startswith('content-length:')), 0)
        if length:
            return False   # requests with a body (none from apt) keep the direct path
        got = await self.forward_to_cache(port, self.cache_head(method, target, version, headers), b'', reader, writer)
        if got is None:
            return False
        status, verdict, sha = got
        from .package_cache import describe
        pkg, ver = describe(target.split('?', 1)[0])
        self.log(method=method, host=host, target=target[:300], allowed=True, status=status, cache=verdict,
                 **({'sha256': sha} if sha else {}), **({'package': pkg, 'version': ver} if pkg else {}))
        return True

    async def pypi(self, head, method, target, version, headers, reader, writer):
        """PyPI index and files (PIP_INDEX_URL points here): from the package cache, or straight from PyPI through
        the cluster proxy while the cache is down. Logged with the upstream URL."""
        from .package_cache import PYPI_FILES, PYPI_PAGES, PackageCache, Request, describe
        path = target.split('?', 1)[0]
        url = (PYPI_PAGES + path[len('/pypi/simple/'):] if path.startswith('/pypi/simple/') else
               PYPI_FILES + path[len('/pypi/packages/'):] if path.startswith('/pypi/packages/') else path)
        host = urlsplit(url).hostname or '127.0.0.1'
        port = self.cache_port()
        if port is not None:
            got = await self.forward_to_cache(port, self.cache_head(method, target, version, headers), b'',
                                              reader, writer)
            if got is not None:
                status, verdict, sha = got
                pkg, ver = describe(url)
                self.log(method=method, host=host, target=url[:300], allowed=True, status=status, cache=verdict,
                         **({'sha256': sha} if sha else {}), **({'package': pkg, 'version': ver} if pkg else {}))
                writer.close()
                return
        if self.direct is None:
            self.direct = PackageCache(None, upstream=self.upstream, store=False, log=False)
        pairs = [tuple(x.strip() for x in h.split(':', 1)) for h in headers if ':' in h]
        req = Request(method, target, version, pairs, head)
        spec, _, _ = self.direct.route(req)
        if isinstance(spec, str):
            self.log(method=method, host=host, target=url[:300], allowed=False, status=404)
            return await self.refuse(writer, 404, 'Not Found')
        await self.direct.serve_spec(req, writer, spec, None, False)
        self.log(method=method, host=host, target=url[:300], allowed=True, status=None, cache='bypass')
        writer.close()


async def selftest():
    import subprocess
    import tempfile
    upstream = os.environ.get('http_proxy') or 'http://proxy.internal:8080'
    with tempfile.TemporaryDirectory(prefix='pkgproxy-', dir='/tmp') as tmp:
        sock, log = Path(tmp) / 'p.sock', Path(tmp) / 'log.jsonl'
        proxy = await PackageProxy(sock, log, upstream).start()
        relay = await asyncio.create_subprocess_exec('socat', 'TCP4-LISTEN:0,bind=127.0.0.1,reuseaddr,fork',
                                                     f'UNIX-CONNECT:{sock}', stderr=asyncio.subprocess.PIPE)
        port = None
        try:
            # socat -d -d prints the port; with plain socat find it via ss
            for _ in range(50):
                out = subprocess.run(['ss', '-ltnp'], capture_output=True, text=True).stdout
                for row in out.splitlines():
                    if f'pid={relay.pid},' in row:
                        port = row.split()[3].rsplit(':', 1)[1]
                if port:
                    break
                await asyncio.sleep(.1)
            px = f'http://127.0.0.1:{port}'
            env = {**os.environ, 'http_proxy': px, 'https_proxy': px, 'no_proxy': ''}
            checks = [('https pypi', ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'https://pypi.org/simple/six/'], '200'),
                      ('http ubuntu', ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}',
                                       'http://archive.ubuntu.com/ubuntu/dists/noble/Release'], '200'),
                      ('https npm', ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'https://registry.npmjs.org/left-pad'], '200'),
                      ('github blocked', ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'https://github.com/'], '000'),
                      ('http blocked', ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://example.com/'], '403')]
            for name, cmd, want in checks:
                r = await asyncio.create_subprocess_exec(*cmd, env=env, stdout=asyncio.subprocess.PIPE)
                got = (await r.communicate())[0].decode().strip()
                print(f'{name}: {got} {"OK" if got == want else "FAIL (want " + want + ")"}')
            print(log.read_text())
        finally:
            relay.terminate()
            await relay.wait()
            await proxy.close()


# A local service as tasks run it: started in the sandbox, reached by 127.0.0.1, by another loopback address and by
# names the task gives in /etc/hosts. The "agent" shell gets its environment before the names exist (Pi starts before
# the agent edits /etc/hosts); legacy.example.org is in the image's /etc/hosts only (the sandbox's is fresh).
LOCAL_URLS = ('http://127.0.0.1:8080/', 'http://127.0.0.10:8080/', 'http://api.lab.internal:8080/',
              'http://admin.corp:8080/', 'http://legacy.example.org:8080/')
AGENT_SHELL = ('setsid nohup bash -c \'while [ ! -f /tmp/go ]; do sleep 0.2; done; for u in ' + ' '.join(LOCAL_URLS)
               + '; do curl -s -m 10 -o /dev/null -w "$u %{http_code}\\n" $u; done > /tmp/agent.out 2>&1\' '
               '>/dev/null 2>&1 & echo agent shell waiting')


async def sandboxtest():
    """An offline harbor_ns sandbox on local/tasksynth-base:2 with the proxy relayed in at 127.0.0.1:3128, as
    synth.environment does it: pip, apt and npm-registry fetches work, other hosts fail, and a local service is
    reached directly (LOCAL_URLS). Run in a CPU job."""
    import tempfile
    from harbor_ns.cli import default_store
    from harbor_ns.environment import offline_bypass_names
    from harbor_ns.sandbox import Sandbox, proxy_env
    from .endpoint_relay import EndpointRelay
    os.environ.setdefault('HARBOR_NS_PERSIST_IMAGES', '0')
    store = default_store()
    with tempfile.TemporaryDirectory(prefix='pkgproxy-ctx-') as ctx:
        Path(ctx, 'Dockerfile').write_text('FROM local/tasksynth-base:2\n'
                                           'RUN echo "127.0.0.1 legacy.example.org" >> /etc/hosts\n')
        _, rootfs, meta = await store.build(Path(ctx), label='package-proxy-test')
    # as harbor_ns.environment does for an offline sandbox
    box = Sandbox(cfg=store.rt, lower=[rootfs], hostname='task', internet=False,
                  base_env={**meta.env, 'HOME': '/root',
                            **proxy_env(offline_bypass_names(rootfs) or None, url=SANDBOX_URL, offline=True)})
    await box.start()
    log_dir = Path(tempfile.mkdtemp(prefix='pkgproxy-log-'))
    relay = EndpointRelay(log_dir)
    relay.scratch = tempfile.TemporaryDirectory(prefix='synth-pkg-', dir='/tmp')
    sock = Path(relay.scratch.name) / 'p.sock'
    proxy = await PackageProxy(sock, log_dir / 'requests.jsonl', os.environ.get('http_proxy') or 'http://proxy.internal:8080').start()
    try:
        await relay.spawn(['nsenter', '-t', str(box.init_pid), '-U', '-n', '--preserve-credentials', '--',
                           '/usr/bin/socat', '-d', '-d', f'TCP4-LISTEN:{SANDBOX_PORT},bind=127.0.0.1,reuseaddr,fork',
                           f'UNIX-CONNECT:{sock}'], 'namespace-relay.log')
        await asyncio.sleep(1)
        for cmd in ['env | grep -i _proxy | sort',
                    'python3 -m pip download --no-deps -d /tmp/dl six 2>&1 | tail -2',
                    'apt-get update 2>&1 | tail -3 && apt-get install -y --no-install-recommends cowsay 2>&1 | tail -2',
                    'curl -s -o /dev/null -w "npm %{http_code}\\n" https://registry.npmjs.org/left-pad',
                    'curl -s -o /dev/null -w "github %{http_code}\\n" https://github.com/ ; echo rc=$?',
                    'curl -s --noproxy "*" -m 5 -o /dev/null -w "direct %{http_code}\\n" http://archive.ubuntu.com/ ; echo rc=$?',
                    'setsid nohup python3 -m http.server 8080 --bind 0.0.0.0 >/tmp/http.log 2>&1 & sleep 2; echo served',
                    AGENT_SHELL,
                    'printf "127.0.0.1 api.lab.internal admin.corp legacy.example.org\\n" >> /etc/hosts; touch /tmp/go; '
                    'for i in $(seq 60); do [ $(cat /tmp/agent.out 2>/dev/null | wc -l) -ge 5 ] && break; sleep 0.5; done; '
                    'cat /tmp/agent.out',
                    # an unquoted variable holding --noproxy '*' passes the quotes: no_proxy is replaced, nothing matches
                    'NP="--noproxy \'*\'"; curl -s -D - -o /dev/null $NP http://127.0.0.1:8080/ | head -1']:
            r = await box.exec(cmd, timeout=300)
            print(f'$ {cmd}\n{r.stdout or ""}{r.stderr or ""}'.rstrip(), flush=True)
        out = (await box.exec('cat /tmp/agent.out', timeout=30)).stdout
        codes = dict(line.split() for line in out.splitlines() if line.strip())
        bad = {u: codes.get(u) for u in LOCAL_URLS if codes.get(u) != '200'}
        print(f'local service from the agent shell: {"OK" if not bad else f"FAIL {bad}"}', flush=True)
        print((log_dir / 'requests.jsonl').read_text())
    finally:
        await relay.close()
        await proxy.close()
        await box.stop()


if __name__ == '__main__':
    import sys
    asyncio.run({'selftest': selftest, 'sandboxtest': sandboxtest}[sys.argv[1]]())
