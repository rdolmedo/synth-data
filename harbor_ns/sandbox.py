"""Unprivileged namespace sandboxes (no Docker, no Apptainer at runtime).

A sandbox is an overlayfs (read-only image rootfs + per-sandbox upper dir) inside
fresh user/mount/pid/net/ipc/uts namespaces, created with ``unshare`` and
entered with ``nsenter``.  ``launch.sh`` prepares the mounts and execs the
static ``nsinit`` as PID 1, which ``pivot_root``s into the overlay.  Networking
is a private netns with ``slirp4netns`` user-mode NAT (outbound only), so
concurrent sandboxes never share ports and cannot reach host loopback services.

Only uid/gid 0 are mapped (no subuid range on this cluster).  Inside the box the
user is root; ``nsfake.so`` (via /etc/ld.so.preload, see builder) papers over
chown/setuid calls that need other ids.

Multi-uid mode (opt-in, ``Pod(multiuid=True)``; NsEnvironment ``ns_multiuid`` or HARBOR_NS_MULTIUID=1):
the pod's user namespace maps uid/gid 0 to the caller and 1..N to the caller's /etc/subuid and
/etc/subgid range (newuidmap/newgidmap), so privilege drops (setpriv, su, setuid) are real.  The
sandboxes of such a pod run without nsfake, and their files owned by subordinate ids are removed (or,
with delete=False, chowned back to the caller) from inside the namespace at teardown.  Image builds are
unchanged (single uid).  See MULTIUID.md.
"""

from __future__ import annotations

import asyncio
import itertools
import contextlib
import contextvars
import datetime
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
LAUNCH_SH = PKG_DIR / "launch.sh"

# Outbound HTTP(S) goes through the cluster proxy; resolve its IP once on the host
# so sandboxes do not depend on the cluster search domain.
DEFAULT_PROXY_HOST = os.environ.get("HARBOR_NS_PROXY_HOST", "proxy")
DEFAULT_PROXY_PORT = int(os.environ.get("HARBOR_NS_PROXY_PORT", "8080"))
SPECIAL_DOMAINS = (".local", ".localhost", ".test", ".example", ".invalid")
# Offline sandboxes only: private-use names (ICANN's .internal, RFC 8375 .home.arpa, the undelegated .lan and .corp)
# and the loopback and RFC 1918 ranges, which an offline sandbox's package proxy (synth.package_proxy) can only refuse.
# Tasks name their own services (api.lab.internal on 127.0.0.10) in /etc/hosts, often from the agent's shell after the
# agent started: Sandbox._env adds /etc/hosts names per exec, so the long-running agent's environment never had them
# and its requests got 403 from the proxy (tasksynth S1 traces).
OFFLINE_DOMAINS = (".internal", ".lan", ".corp", ".home.arpa")
OFFLINE_NETS = ("127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
OFFLINE_JVM_NETS = ("10.*", "192.168.*", *(f"172.{n}.*" for n in range(16, 32)))  # nonProxyHosts has no CIDR


def _resolve_proxy() -> str | None:
    import socket

    host = DEFAULT_PROXY_HOST
    if not host or host.lower() == "none":
        return None
    try:
        ip = socket.gethostbyname(host)
    except OSError:
        return None
    return f"http://{ip}:{DEFAULT_PROXY_PORT}"


_PROXY_URL: str | None | bool = False


def proxy_url() -> str | None:
    global _PROXY_URL
    if _PROXY_URL is False:
        _PROXY_URL = _resolve_proxy()
    return _PROXY_URL  # type: ignore[return-value]


def proxy_env(extra_no_proxy: list[str] | None = None, url: str | None = None,
              offline: bool = False) -> dict[str, str]:
    """Proxy settings for sandboxes; `extra_no_proxy` = compose service names etc.
    HARBOR_NS_NO_PROXY (comma-separated hosts/IPs) is added too: cluster-network
    endpoints an agent inside the sandbox talks to directly, e.g. the LLM server.
    `url` overrides the cluster proxy (an offline sandbox's local package proxy).
    `offline`: a sandbox without internet; private names and ranges bypass the proxy too."""
    url = url or proxy_url()
    if not url:
        return {}
    extra_no_proxy = [*(extra_no_proxy or []),
                      *(h.strip() for h in os.environ.get("HARBOR_NS_NO_PROXY", "").split(",") if h.strip())]
    # Reserved/special-use names (RFC 6761/6762) never resolve on the internet, so the
    # proxy can only fail for them; tasks put e.g. `homeserver.local` in /etc/hosts.
    domains = (*SPECIAL_DOMAINS, *(OFFLINE_DOMAINS if offline else ()))
    no_proxy = ",".join(["localhost", "127.0.0.1", "::1", "0.0.0.0", *domains, *(OFFLINE_NETS if offline else ()),
                         *(extra_no_proxy or [])])
    host, _, port = url.rsplit("/", 1)[-1].partition(":")
    # JVM tools (maven, gradle) ignore http_proxy; they read system properties.
    nph = "|".join(["localhost", "127.*", *(f"*{d}" for d in domains), *(OFFLINE_JVM_NETS if offline else ()),
                    *(extra_no_proxy or [])])
    jvm = (f"-Dhttp.proxyHost={host} -Dhttp.proxyPort={port} -Dhttps.proxyHost={host} "
           f"-Dhttps.proxyPort={port} -Dhttp.nonProxyHosts={nph}")
    return {
        "http_proxy": url,
        "https_proxy": url,
        "HTTP_PROXY": url,
        "HTTPS_PROXY": url,
        "no_proxy": no_proxy,
        "NO_PROXY": no_proxy,
        # Maven >=3.9's default (native) transport ignores these JVM properties; wagon honours them.
        "MAVEN_OPTS": f"{jvm} -Dmaven.resolver.transport=wagon",
        # gradlew (Gradle <7) does `eval set -- ... $GRADLE_OPTS`: an unescaped `|` becomes a
        # shell pipe ("eval: 127.*: not found"); `\|` survives both gradlew generations.
        "GRADLE_OPTS": jvm.replace("|", "\\|"),
        # Berkeley UPC's upcc does HTTP netcompile (upc-translator.lbl.gov) with its own socket
        # code and only honours a proxy from upcc.conf/-http-proxy; UPCC_FLAGS is prepended to argv.
        "UPCC_FLAGS": f"-http-proxy={url}",
    }


# The job's package cache (synth.package_cache, issue #60): HARBOR_NS_PACKAGE_CACHE names the registry file the cache
# process writes ({"url", "port", "pid", "root", ...}); it is removed while the cache is down.
PACKAGE_CACHE_ENV = "HARBOR_NS_PACKAGE_CACHE"
_PACKAGE_CACHE: dict = {"t": -1e9, "path": None, "value": None}


def package_cache() -> dict | None:
    """The package cache's registry if the cache is up (pid alive, port accepting), else None. Checked at most every
    2 s per process, so a build step started after the cache died falls back to the cluster proxy."""
    path = os.environ.get(PACKAGE_CACHE_ENV)
    if not path:
        return None
    t = time.monotonic()
    if _PACKAGE_CACHE["path"] == path and t - _PACKAGE_CACHE["t"] < 2:
        return _PACKAGE_CACHE["value"]
    value = None
    try:
        import socket

        reg = json.loads(Path(path).read_text())
        os.kill(int(reg["pid"]), 0)
        with socket.create_connection(("127.0.0.1", int(reg["port"])), timeout=1):
            pass
        value = reg
    except (OSError, ValueError, KeyError, TypeError):
        value = None
    _PACKAGE_CACHE.update(t=t, path=path, value=value)
    return value


# pip keeps downloads in its HTTP cache (~/.cache/pip) only for https and trusted hosts: without this an index on
# http://127.0.0.1 leaves the cache empty (images and sandboxes would differ from cache-off ones on disk). A task's
# own --trusted-host, image ENV or inline export still wins (then pip just skips its cache, as before).
PIP_CACHE_TRUST = {"PIP_TRUSTED_HOST": "127.0.0.1"}


def package_cache_env(tag: str = "", pip: bool = True) -> dict[str, str]:
    """Environment overrides for a build step through the package cache: apt (plain HTTP) and other http:// downloads
    use it as their proxy (https:// keeps going to the cluster proxy), pip uses its PyPI pull-through index. `tag`
    labels the step's requests in the cache's fetch log (proxy user name / index path). {} when the cache is down."""
    c = package_cache()
    if not c:
        return {}
    port = int(c["port"])
    url = f"http://{tag}@127.0.0.1:{port}" if tag else f"http://127.0.0.1:{port}"
    env = {"http_proxy": url, "HTTP_PROXY": url}
    if pip:
        env["PIP_INDEX_URL"] = f"http://127.0.0.1:{port}/t/{tag}/pypi/simple/" if tag else \
            f"http://127.0.0.1:{port}/pypi/simple/"
        env.update(PIP_CACHE_TRUST)
    return env


def hosts_names(text: str) -> list[str]:
    """Host names of an /etc/hosts file, in order, once each."""
    names: list[str] = []
    for line in text.splitlines():
        parts = line.split("#", 1)[0].split()
        names += [n for n in parts[1:] if n not in names]
    return names


# Variables that carry the proxy (see proxy_env).  sudo's env_reset and PAM logins
# (`su -`, sshd, cron) drop them, and sandboxes have no direct egress (Docker does).
PROXY_KEYS = ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")
SUDOERS_PROXY = f'Defaults env_keep += "{" ".join(PROXY_KEYS)} MAVEN_OPTS GRADLE_OPTS"\n'


@dataclass
class RuntimeConfig:
    """Host-side paths/binaries needed by the runtime."""

    nsinit: Path
    work_root: Path  # node-local scratch for overlays (must support overlay upper)
    slirp4netns: str = "slirp4netns"
    host_resolv: str = "/etc/resolv.conf"
    # Current nsfake.so: copied over the image's (build-time) copy at sandbox start, so
    # shim fixes reach already-built (cached) images without a BUILDER_VERSION bump.
    nsfake: Path | None = None


def find_runtime_bins(cache_dir: Path) -> Path:
    """Compile nsinit (static) into cache_dir if needed; return its path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "nsinit"
    src = PKG_DIR / "nsinit.c"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        tmp = cache_dir / f".nsinit.{os.getpid()}"
        subprocess.run(["gcc", "-static", "-O2", "-o", str(tmp), str(src)], check=True)
        os.replace(tmp, out)
    return out


def build_nsfake(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "nsfake.so"
    src = PKG_DIR / "nsfake.c"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        tmp = cache_dir / f".nsfake.{os.getpid()}.so"
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-fno-delete-null-pointer-checks", "-U_FORTIFY_SOURCE",
             "-fno-stack-protector", "-o", str(tmp), str(src)],
            check=True,
        )
        os.replace(tmp, out)
    return out


def build_nsdev(cache_dir: Path) -> Path:
    """The id-independent preload of multi-uid sandboxes (nsdev.c): CPU view, affinity remap, RLIMIT_CORE clamp,
    emulated device-node mknod.  Replaces nsfake.so in the sandbox's view (no id emulation: real ids)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "nsdev.so"
    src = PKG_DIR / "nsdev.c"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        tmp = cache_dir / f".nsdev.{os.getpid()}.so"
        subprocess.run(["gcc", "-shared", "-fPIC", "-O2", "-o", str(tmp), str(src)], check=True)
        os.replace(tmp, out)
    return out


_PRLIMIT: str | None | bool = False


def _fsize_argv(limit: int | None) -> list[str]:
    """argv prefix that runs a command with RLIMIT_FSIZE = limit bytes (soft = hard), inherited by everything it starts:
    the per-file part of the disk cap (diskcap.py).  [] without a limit, or when util-linux prlimit is missing."""
    global _PRLIMIT
    if not limit:
        return []
    if _PRLIMIT is False:
        _PRLIMIT = shutil.which("prlimit")
    return [_PRLIMIT, f"--fsize={limit}:{limit}", "--"] if _PRLIMIT else []


def fsize_supported() -> bool:
    return bool(_fsize_argv(1))


_SIGRESET: list[str] | None = None


def _sigreset() -> list[str]:
    """argv prefix that starts the sandboxed process with default signal dispositions.

    A command started in the background by a non-interactive shell (worker.sh runs
    harbor with `&`) has SIGINT/SIGQUIT ignored, and ignored dispositions are inherited
    across fork/exec into every sandbox process: a task's `proc.send_signal(SIGINT)`
    then does nothing (TB2.1 cancel-async-tasks hangs).  `docker exec` starts from
    default dispositions."""
    global _SIGRESET
    if _SIGRESET is None:
        try:
            ok = subprocess.run(["env", "--default-signal", "true"], capture_output=True).returncode == 0
        except OSError:
            ok = False
        _SIGRESET = ["env", "--default-signal"] if ok else []
    return _SIGRESET


def _unshare_argv(private_net: bool) -> list[str]:
    argv = ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork",
            "--ipc", "--uts", "--propagation", "private"]
    if private_net:
        argv.insert(-2, "--net")
    return argv


def _child_pid(parent: int, timeout: float = 10.0) -> int:
    deadline = time.monotonic() + timeout
    path = f"/proc/{parent}/task/{parent}/children"
    while time.monotonic() < deadline:
        try:
            kids = open(path).read().split()
        except FileNotFoundError:
            raise RuntimeError(f"unshare process {parent} exited") from None
        if kids:
            return int(kids[0])
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for sandbox init pid")


# With exactly 65,536 subordinate ids not every id a task uses can be mapped, so the map has bands (namespace ids that
# no task uses stay unmapped; setuid() into one fails).  Scan of the TB 2.1, TB-Science 0.1 and TB-Science-lite tasks
# (2026-11-05) for the run uids they setuid/setpriv/chown into: 1000/1001, 6000, 10001, 20000, 21001-3, 50001,
# 60000+index (up to ~60100), 61000-61042, 61500, 62000, 65010, nobody 65534 and 65531-65537 (tess-transit-vetting),
# and, for weighted-vaccine-inference, a fresh 100000+n per study (n < ~9000, so up to 108997 here).  A single hole at
# 60000 (the first version) broke every `60000 + index` verifier on its index-0 case.  Bands: 1..11000, 20000..65537,
# 100000..108997 = 11000 + 45538 + 8998 = 65536 ids.  Unmapped: 11001..19999 and 65538..99999 (no task uses them).
MULTIUID_BANDS = ((1, 11000), (20000, 65537), (100000, 108997))


def _subid_extents(start: int, count: int) -> list[str]:
    """newuidmap/newgidmap triples (inside outside count) mapping the namespace id bands onto [start, start+count)."""
    total = sum(hi - lo + 1 for lo, hi in MULTIUID_BANDS)
    if count < total:
        return ["1", str(start), str(count)]  # a smaller range: one contiguous band from id 1
    out, off = [], 0
    for lo, hi in MULTIUID_BANDS:
        out += [str(lo), str(start + off), str(hi - lo + 1)]
        off += hi - lo + 1
    return out


def _subid_range(path: str) -> tuple[int, int]:
    """First (start, count) range of the calling user in /etc/subuid or /etc/subgid."""
    import pwd

    uid = os.getuid()
    names = {str(uid)}
    try:
        names.add(pwd.getpwuid(uid).pw_name)
    except KeyError:
        pass
    with open(path) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 3 and parts[0] in names:
                return int(parts[1]), int(parts[2])
    raise RuntimeError(f"multi-uid sandboxes need a {path} range for uid {uid}")


@dataclass
class ExecOutput:
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def _rmtree_force(path: Path) -> None:
    if not path.exists():
        return
    # Agents may leave dirs without write/exec permission; make them removable.
    subprocess.run(["chmod", "-R", "u+rwX", str(path)], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        subprocess.run(["rm", "-rf", str(path)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


_BUILD_CPU_COUNTER = itertools.count()


def _tree_pids(root: int) -> list[int]:
    """A process and all its descendants, from /proc/<pid>/task/<tid>/children: the tree only, not every process of
    the node (48 build guards each listing all of /proc every 2 s cost a round job a lot of CPU). Processes orphaned
    inside a sandbox are re-parented to its init, which is in the tree."""
    out, stack, seen = [], [root], set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{pid}/task/{tid}/children") as f:
                    stack.extend(int(c) for c in f.read().split())
            except (OSError, ValueError):
                pass
    return out


def _tree_rss_kib(root: int) -> int:
    """Resident memory (KiB) of a process and all its descendants, from /proc."""
    total = 0
    page_kib = os.sysconf("SC_PAGE_SIZE") // 1024
    for pid in _tree_pids(root):
        try:
            with open(f"/proc/{pid}/statm") as f:
                total += int(f.read().split()[1]) * page_kib
        except (OSError, IndexError, ValueError):
            pass
    return total


# Memory cap of the build steps of the image being built (MiB), set by ImageStore.build(memory_mb=...) for its steps;
# None: HARBOR_NS_BUILD_MEM_GB (or no cap).
BUILD_MEMORY_MB: contextvars.ContextVar = contextvars.ContextVar("harbor_ns_build_memory_mb", default=None)
# exit status build_run returns for a step its memory guard killed (as a SIGKILL'd process: 128 + 9)
MEMORY_KILLED = 137
# set by build_run when its memory guard killed the step (ImageStore._run_step does not retry such a step)
BUILD_MEMORY_KILLED: contextvars.ContextVar = contextvars.ContextVar("harbor_ns_build_memory_killed", default=None)


def apply_stack_limit(kib: int | None = None) -> int | None:
    """Soft RLIMIT_STACK of this process (inherited by every build step and sandbox it starts) -> the limit set (KiB),
    or None. HTCondor jobs run with an unlimited stack, where Docker gives containers 8 MiB: a task whose program
    recurses without end expects a SIGSEGV (Docker) but grows until the node's memory is gone (m1-hero round job
    17841191.0, 2026-11-04: a WSDL compiler crash reproducer in a task's setup; the job was held at 392 GB). An
    unlimited stack also selects the legacy mmap layout for children. HARBOR_NS_STACK_KB, or `kib`; 0: unchanged."""
    import resource
    if kib is None:
        kib = int(os.environ.get("HARBOR_NS_STACK_KB") or 0)
    if not kib:
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    want = kib * 1024 if hard == resource.RLIM_INFINITY else min(kib * 1024, hard)
    if soft != want:
        resource.setrlimit(resource.RLIMIT_STACK, (want, hard))
    return want // 1024


def _kill_tree(root: int) -> None:
    """SIGKILL a process tree: its process group at once (build steps start in their own session), then every process
    left in the tree. Fast on purpose: a runaway recursion grows by GBs per second (#171: listing all of /proc first let
    a step the guard had caught at 26 GB push its job past 59 GB)."""
    try:
        os.killpg(root, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        pids = _tree_pids(root)
    except OSError:
        pids = [root]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


async def build_run(
    cfg: RuntimeConfig,
    rootfs: Path,
    argv: list[str],
    env: dict[str, str],
    cwd: str,
    log_file,
    timeout: float | None = None,
    binds: list[tuple[str, str]] | None = None,
) -> int:
    """Run one command with ``rootfs`` mounted writable as /, host network.

    Used for Dockerfile RUN steps.  Returns the exit status; output goes to
    ``log_file`` (a binary file object).
    """
    tmp = Path(tempfile.mkdtemp(prefix="build-", dir=cfg.work_root))
    try:
        merged = tmp / "merged"
        etc = tmp / "etc"
        merged.mkdir()
        etc.mkdir()
        shutil.copy(cfg.host_resolv, etc / "resolv.conf")
        (etc / "hosts").write_text("127.0.0.1\tlocalhost buildkitsandbox\n::1\tlocalhost ip6-localhost ip6-loopback\n")
        lenv = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "NS_DIRECT": str(rootfs),
            "NS_MERGED": str(merged),
            "NS_HOSTNAME": "buildkitsandbox",
            "NS_NSINIT": str(cfg.nsinit),
            "NS_ETC_DIR": str(etc),
            "NS_BINDS": "\n".join(f"{s}\t{d}" for s, d in (binds or [])),
            "NS_PRIVATE_NET": "0",
        }
        # nsinit execs the shell with the env we give to `env -i` below.
        inner = ["/usr/bin/env", "-i", *[f"{k}={v}" for k, v in env.items()]]
        # `cd` happens inside the sandbox via the shell wrapper.
        wrapper = ["/bin/sh", "-c", 'cd "$1" 2>/dev/null || { mkdir -p "$1" && cd "$1"; }; shift; exec "$@"', "sh", cwd]
        full = [*_sigreset(), *_unshare_argv(False), "bash", str(LAUNCH_SH), "--", *inner, *wrapper, *argv]
        # HARBOR_NS_BUILD_CPUS: pin each build step to that many of the job's cores, so `make -j$(nproc)` and the
        # like run a few jobs instead of one per host core. HARBOR_NS_BUILD_MEM_GB: kill a step whose process tree
        # outgrows it (a build has no PSS guard: RTS 2026-10-26, nvm compiling node from source took 90+ GB and the
        # whole 48-slot worker with it).
        ncpu = int(os.environ.get("HARBOR_NS_BUILD_CPUS") or 0)
        if ncpu > 0:
            allowed = sorted(os.sched_getaffinity(0))
            if len(allowed) > ncpu:
                start = (next(_BUILD_CPU_COUNTER) * ncpu) % len(allowed)
                full = ["taskset", "-c", ",".join(str(allowed[(start + k) % len(allowed)]) for k in range(ncpu)), *full]
        mem_mb = BUILD_MEMORY_MB.get()
        mem_kib = mem_mb * 1024 if mem_mb else float(os.environ.get("HARBOR_NS_BUILD_MEM_GB") or 0) * 2**20
        proc = await asyncio.create_subprocess_exec(
            *full, env=lenv, stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
            start_new_session=True,
        )
        over = []

        async def watchdog():
            # every 0.5 s, inline (a walk of the step's own tree takes well under a millisecond; a busy default
            # executor would delay it): an unbounded recursion fills ~10 GB/s
            while proc.returncode is None:
                await asyncio.sleep(0.5)
                rss = _tree_rss_kib(proc.pid)
                if rss > mem_kib:
                    over.append(rss)
                    _kill_tree(proc.pid)
                    return
        guard = asyncio.ensure_future(watchdog()) if mem_kib > 0 else None
        try:
            rc = await asyncio.wait_for(proc.wait(), timeout)
            if over:
                log_file.write(f"\n[harbor_ns] build step killed: its processes used {over[0] / 2**20:.1f} GB of "
                               f"memory, over the build's limit of {mem_kib / 2**20:.0f} GB (error: out of memory)\n".encode())
                BUILD_MEMORY_KILLED.set(True)
                return MEMORY_KILLED
            return rc
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            log_file.write(f"\n[harbor_ns] build step timed out after {timeout}s\n".encode())
            return 124
        except asyncio.CancelledError:
            try:  # don't leave the step running against a rootfs that is about to be discarded
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise
        finally:
            if guard is not None:
                guard.cancel()
    finally:
        # The bind mounts lived only in the dead mount namespace; tmp is plain dirs.
        _rmtree_force(tmp)


def memory_victim(rows: list[dict], spare=()) -> dict:
    """The process the RAM guard kills: the largest by PSS, a process whose comm is in `spare` (the agent) only when
    nothing else is left (harbor_ns/memcap.MemCap.check's rule)."""
    others = [r for r in rows if r.get('comm') not in set(spare)]
    return max(others or rows, key=lambda r: r['pss_kib'])


@dataclass
class Sandbox:
    cfg: RuntimeConfig
    lower: list[Path]  # read-only layers, top first
    hostname: str = "sandbox"
    binds: list[tuple[str, str]] = field(default_factory=list)
    private_net: bool = True
    cpus: list[int] | None = None  # CPU affinity for everything in the box
    cpu_view: bool = False  # multi-uid only: the nsdev preload reports len(cpus) as the CPU count (marker file)
    entrypoint: list[str] | None = None  # started in background after boot (Docker ENTRYPOINT)
    base_env: dict[str, str] = field(default_factory=dict)
    pod: "Pod | None" = None  # launch inside this pod's user namespace
    extra_hosts: dict[str, str] = field(default_factory=dict)  # name -> IP for /etc/hosts
    internet: bool = True  # slirp4netns NAT (only with private_net)
    exec_network_pid: int | None = None  # evaluator-only sibling netns; existing agent processes never move
    fsize: int | None = None  # RLIMIT_FSIZE (bytes) for every process in the box (diskcap.py); None: unlimited
    memory_limit_mb: int | None = None
    memory_event_log: Path | None = None
    # False: a host without a private procfs runs the sandbox without the RAM guard instead of failing to start
    memory_guard_required: bool = True
    memory_kills: list = field(default_factory=list)   # processes the RAM guard killed ({'pid', 'pss_kib', 'comm'})
    memory_poll_s: float = 1.0                          # RAM guard sampling interval
    # comm names the RAM guard kills only when nothing else is left (as harbor_ns/memcap.SPARE; synth campaigns since
    # 2026-11-14: ('pi',), the agent). Default (): the largest process, whatever it is, as before.
    memory_spare: tuple = ()

    init_pid: int | None = None
    proc_mode: str = "?"
    _unshare: asyncio.subprocess.Process | None = None
    _slirp: asyncio.subprocess.Process | None = None
    _dir: Path | None = None
    _memory_task: asyncio.Task | None = field(default=None, init=False, repr=False)

    @property
    def dir(self) -> Path:
        assert self._dir is not None
        return self._dir

    async def start(self, timeout: float = 60.0) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix=f"box-{self.hostname[:40]}-", dir=self.cfg.work_root))
        upper, work, merged, etc = (self._dir / n for n in ("upper", "work", "merged", "etc"))
        for p in (upper, work, merged, etc):
            p.mkdir()
        (etc / "resolv.conf").write_text(
            "nameserver 10.0.2.3\n" if self.private_net else open(self.cfg.host_resolv).read()
        )
        hosts = f"127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n"
        if self.hostname not in self.extra_hosts:
            hosts += f"127.0.1.1\t{self.hostname}\n"
        for name, ip in self.extra_hosts.items():
            hosts += f"{ip}\t{name}\n"
        (etc / "hosts").write_text(hosts)
        (etc / "hostname").write_text(self.hostname + "\n")
        lenv = {
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "NS_LOWER": ":".join(str(p) for p in self.lower),
            "NS_UPPER": str(upper),
            "NS_WORK": str(work),
            "NS_MERGED": str(merged),
            "NS_HOSTNAME": self.hostname,
            "NS_NSINIT": str(self.cfg.nsinit),
            "NS_ETC_DIR": str(etc),
            "NS_BINDS": "\n".join(f"{s}\t{d}" for s, d in self.binds),
            "NS_PRIVATE_NET": "1" if self.private_net else "0",
        }
        if self.memory_limit_mb:  # /dev/shm is RAM the PSS guard cannot see: cap it at the same budget
            lenv["NS_SHM_SIZE"] = f"{int(self.memory_limit_mb)}m"
        if self.pod is not None:
            # Join the pod's user namespace (shared by all of the trial's services, so
            # their network namespaces can be wired together), then unshare the rest.
            argv = ["nsenter", "-t", str(self.pod.holder_pid), "-U", "--preserve-credentials", "--",
                    *[a for a in _unshare_argv(self.private_net) if a not in ("--user", "--map-root-user")],
                    "bash", str(LAUNCH_SH)]
        else:
            argv = [*_unshare_argv(self.private_net), "bash", str(LAUNCH_SH)]
        if self.cpus:
            argv = ["taskset", "-c", ",".join(map(str, self.cpus)), *argv]
        argv = [*_fsize_argv(self.fsize), *argv]
        self._unshare = await asyncio.create_subprocess_exec(
            *argv, env=lenv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=open(self._dir / "launch.log", "wb"), start_new_session=True,
        )
        try:
            line = await asyncio.wait_for(self._unshare.stdout.readline(), timeout)
        except asyncio.TimeoutError:
            await self.stop()
            raise RuntimeError("sandbox did not become ready") from None
        if b"NSINIT_READY" not in line:
            log = (self._dir / "launch.log").read_text(errors="replace")[-2000:]
            await self.stop()
            raise RuntimeError(f"sandbox launch failed: {log}")
        self.init_pid = _child_pid(self._unshare.pid)
        log = (self._dir / "launch.log").read_text(errors="replace")
        self.proc_mode = "private" if "NSLAUNCH_PROC=private" in log else "host"
        await asyncio.get_running_loop().run_in_executor(None, self._seed_runtime_files)
        if self.private_net and self.internet:
            await self._start_slirp(timeout)
        if self.entrypoint:
            await self.spawn_background(self.entrypoint)
        if self.memory_limit_mb and self.proc_mode != 'private' and not self.memory_guard_required:
            self.memory_limit_mb = None
        if self.memory_limit_mb:
            if self.proc_mode != 'private':
                await self.stop()
                raise RuntimeError('Sandbox RAM guard requires a private PID namespace procfs')
            try:
                sample=await asyncio.to_thread(self._memory_snapshot)
                if not sample:raise RuntimeError('Sandbox RAM guard cannot read process PSS')
            except BaseException:
                await self.stop()
                raise
            self._memory_task=asyncio.create_task(self._watch_memory())

    def _memory_snapshot(self):
        """Read only this sandbox's private procfs; returned PIDs are namespace PIDs."""
        proc=Path(self.host_path('/proc'))
        rows=[]
        for path in proc.iterdir():
            if not path.name.isdecimal():continue
            try:
                values=(path/'smaps_rollup').read_text()
                pss=next(int(line.split()[1]) for line in values.split('\n') if line.startswith('Pss:'))
                name=(path/'comm').read_text().strip()
            except (OSError,StopIteration,ValueError):continue  # Concurrent process exit.
            rows.append({'pid':int(path.name),'pss_kib':pss,'comm':name})
        return rows

    def _memory_event(self,event):
        path=self.memory_event_log or self.dir/'memory-events.jsonl'
        event={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'scope':'sandbox-private-pid-namespace','method':f'sum process PSS, sampled every {self.memory_poll_s:g} s',
               'limit_mb':self.memory_limit_mb,**event}
        for target in (path,self.dir/'memory-events-local.jsonl'):
            try:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('a') as stream:stream.write(json.dumps(event)+'\n')
                return
            except OSError:continue

    async def _watch_memory(self):
        self._memory_event({'event':'guard_started','init_host_pid':self.init_pid,'cpu_affinity':self.cpus})
        while self.init_pid is not None:
            try:
                rows=await asyncio.to_thread(self._memory_snapshot)
                total=sum(row['pss_kib'] for row in rows)
                candidates=[row for row in rows if row['pid']>1]
                if candidates and total>self.memory_limit_mb*1024:
                    victim=memory_victim(candidates,self.memory_spare)
                    self.memory_kills.append({**victim, 'total_pss_kib': total})
                    self._memory_event({'event':'memory_budget_exceeded','total_pss_kib':total,
                        'victim':victim,'action':'SIGKILL largest process inside this namespace; keep PID1 alive'
                        +(f'; spare {",".join(self.memory_spare)} unless nothing else is left' if self.memory_spare else '')})
                    # Execute kill *inside* the recorded namespace. Never use a
                    # namespace PID as a host PID or signal unrelated processes.
                    await self.exec(f'kill -KILL {victim["pid"]}',shell='sh',timeout=10)
            except asyncio.CancelledError:raise
            except OSError as exc:
                self._memory_event({'event':'guard_observation_error','error':repr(exc)})
            await asyncio.sleep(self.memory_poll_s)

    async def update_env(self, env: dict[str, str]) -> None:
        """Change the environment of later exec()s; proxy variables also go to /etc/environment (PAM sessions)."""
        self.base_env.update(env)
        if self.init_pid:
            await asyncio.get_running_loop().run_in_executor(None, self._seed_runtime_files)

    def _seed_runtime_files(self) -> None:
        """Runtime files every sandbox gets on top of its image (written through the
        overlay before anything but the static nsinit runs in it)."""
        from .builder import resolve_in_root  # noqa: PLC0415  (builder imports this module)

        root = Path(f"/proc/{self.init_pid}/root")
        # 1. The current nsfake.so over the copy baked into (possibly older, cached) images,
        #    so nsfake fixes apply without rebuilding every image.  Only where the image has it.
        nsf = self.cfg.nsfake
        multi = self.pod is not None and self.pod.multiuid
        if multi:  # real ids: nsdev.so (no id emulation) takes nsfake.so's place in this sandbox's view
            nsf = build_nsdev(self.cfg.nsinit.parent)
            if self.cpu_view and self.cpus:  # sysconf(_SC_NPROCESSORS_*) = affinity size
                try:
                    shim_dir = resolve_in_root(root, "/usr/lib/harbor-ns")
                    if shim_dir.is_dir():
                        (shim_dir / "cpu_view").write_text(f"{len(self.cpus)}\n")
                except OSError:
                    pass
        try:
            dst = resolve_in_root(root, "/usr/lib/harbor-ns/nsfake.so")
            if nsf is not None and nsf.is_file() and dst.is_file():
                tmp = dst.with_name(f".nsfake.{os.getpid()}.so")
                shutil.copyfile(nsf, tmp)
                os.chmod(tmp, 0o755)
                os.replace(tmp, dst)
        except OSError:
            pass
        if multi:
            _disable_nsfake(root)
            self._replay_owners(root)
        px = {k: self.base_env[k] for k in PROXY_KEYS if self.base_env.get(k)}
        if not px:
            return
        try:
            etc = resolve_in_root(root, "/etc")
            if not etc.is_dir():
                return
            # 2. sudo's env_reset drops the proxy (`sudo apt-get update`: "no Release file").
            sd = resolve_in_root(root, "/etc/sudoers.d")
            sd.mkdir(mode=0o755, exist_ok=True)
            f = sd / "zz-harbor-ns-proxy"
            if not f.is_file() or f.read_text(errors="replace") != SUDOERS_PROXY:
                f.unlink(missing_ok=True)   # 0440 once written: replace, never rewrite (update_env runs this again)
                f.write_text(SUDOERS_PROXY)
                os.chmod(f, 0o440)
            # 3. PAM sessions (`su -`, sshd, cron) start from /etc/environment (pam_env).
            envf = resolve_in_root(root, "/etc/environment")
            old = envf.read_text(errors="replace") if envf.is_file() else ""
            lines = [ln for ln in old.splitlines() if ln.split("=", 1)[0].strip() not in PROXY_KEYS]
            lines += [f'{k}="{v}"' for k, v in px.items()]
            envf.write_text("\n".join(lines) + "\n")
        except OSError:
            pass

    def _replay_owners(self, root: Path) -> None:
        """Multi-uid: apply the build-time owners nsfake recorded (owners.py), as Docker's image would have them."""
        from . import owners  # noqa: PLC0415

        rec = owners.recorded(self.lower, self.cfg.work_root.parent / "owners-cache")
        if rec is None:
            note = "no owner records (unpacked image without owners.json)"
        else:
            note = owners.replay(self.pod.holder_pid, str(root), rec) if rec else "none recorded"
        try:
            with open(self.dir / "launch.log", "a") as f:
                f.write(f"MULTIUID owners: {note}\n")
        except OSError:
            pass

    async def set_internet(self, on: bool) -> None:
        """Toggle outbound internet (slirp4netns) at runtime."""
        if not self.private_net or on == (self._slirp is not None and self._slirp.returncode is None):
            self.internet = on
            return
        if on:
            await self._start_slirp(30)
        else:
            assert self._slirp is not None
            try:
                os.killpg(self._slirp.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._slirp.wait()
            self._slirp = None
        self.internet = on

    async def _start_slirp(self, timeout: float) -> None:
        r, w = os.pipe()
        try:
            self._slirp = await asyncio.create_subprocess_exec(
                self.cfg.slirp4netns, "--configure", "--mtu=65520", "--disable-host-loopback",
                f"--ready-fd={w}", str(self.init_pid), "tap0",
                stdin=subprocess.DEVNULL, stdout=open(self.dir / "slirp.log", "wb"),
                stderr=subprocess.STDOUT, pass_fds=(w,), start_new_session=True,
            )
        finally:
            os.close(w)
        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(loop.run_in_executor(None, os.read, r, 1), timeout)
        finally:
            os.close(r)
        if data != b"1":
            raise RuntimeError("slirp4netns failed: " + (self.dir / "slirp.log").read_text(errors="replace")[-1000:])

    def _nsenter(self, ids: tuple[int, int] | None = None) -> list[str]:
        assert self.init_pid is not None
        # -w = target's cwd ("/"); `nsenter --wd=DIR` (util-linux 2.37) resolves DIR
        # on the host, so callers cd inside the sandbox instead.
        argv = ["nsenter", "-t", str(self.init_pid), "-U", "-m", "-p", "-i", "-u", "-r", "-w",
                "--preserve-credentials"]
        if ids is not None:  # multi-uid pods: enter as a real (mapped) uid/gid instead of root
            argv[-1:] = ["-S", str(ids[0]), "-G", str(ids[1])]
        if self.private_net:
            argv.insert(4, f"--net=/proc/{self.exec_network_pid}/ns/net" if self.exec_network_pid else "-n")
        argv = [*_sigreset(), *argv]
        if self.cpus:
            argv = ["taskset", "-c", ",".join(map(str, self.cpus)), *argv]
        return [*_fsize_argv(self.fsize), *argv]

    def _env(self, env: dict[str, str] | None) -> dict[str, str]:
        e = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME": "/root"}
        e.update(self.base_env)
        if env:
            e.update(env)
        # Names added to the sandbox's /etc/hosts (e.g. `127.0.0.1 myapp.lan` by a task or
        # its oracle) must not go through the proxy; Docker has no proxy, so they resolve locally.
        if "no_proxy" in e or "NO_PROXY" in e:
            names = self._hosts_names()
            for k in ("no_proxy", "NO_PROXY"):
                if k in e and names:
                    cur = [x for x in e[k].split(",") if x]
                    e[k] = ",".join(cur + [n for n in names if n not in cur])
        return e

    def _hosts_names(self) -> list[str]:
        if self._dir is None:
            return []
        try:  # launch.sh copies /etc/hosts into the overlay, so the live copy is in upper/
            text = (self._dir / "upper" / "etc" / "hosts").read_text(errors="replace")
        except OSError:
            return []
        return hosts_names(text)


    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        shell: str = "bash",
        ids: tuple[int, int] | None = None,
    ) -> ExecOutput:
        if self.init_pid is None:
            raise RuntimeError("sandbox not started")
        # Fall back to /bin/sh when the image has no bash.
        sh = "/bin/bash" if shell == "bash" and await self._has("/bin/bash") else "/bin/sh"
        if cwd:
            command = f"cd -- {shlex.quote(cwd)} || exit 1\n{command}"
        argv = [*self._nsenter(ids), "--", sh, "-c", command]
        out_f = tempfile.TemporaryFile(dir=self.dir)
        err_f = tempfile.TemporaryFile(dir=self.dir)
        stdin_f = None
        if stdin is not None:
            stdin_f = tempfile.TemporaryFile(dir=self.dir)
            stdin_f.write(stdin)
            stdin_f.seek(0)
        timed_out = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, env=self._env(env), stdin=stdin_f or subprocess.DEVNULL,
                stdout=out_f, stderr=err_f, start_new_session=True,
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout)
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                rc = await proc.wait()
                rc = 124
            except asyncio.CancelledError:
                # Harbor enforces verifier/agent timeouts by cancelling the task.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise
            out_f.seek(0)
            err_f.seek(0)
            return ExecOutput(
                return_code=rc,
                stdout=out_f.read().decode(errors="replace"),
                stderr=err_f.read().decode(errors="replace"),
                timed_out=timed_out,
            )
        finally:
            out_f.close()
            err_f.close()
            if stdin_f:
                stdin_f.close()

    _has_cache: dict[str, bool] = field(default_factory=dict)

    async def _has(self, path: str) -> bool:
        if path not in self._has_cache:
            # /proc/<pid>/root resolves inside the sandbox root; lexists so a
            # symlinked /bin -> usr/bin still counts.
            self._has_cache[path] = os.path.lexists(f"/proc/{self.init_pid}/root{path}")
        return self._has_cache[path]

    async def spawn_background(self, argv: list[str], cwd: str | None = None,
                               env: dict[str, str] | None = None,
                               ids: tuple[int, int] | None = None) -> asyncio.subprocess.Process:
        """Start a long-lived process in the box (e.g. the image ENTRYPOINT)."""
        if cwd:
            argv = ["/bin/sh", "-c", 'cd -- "$0" && exec "$@"', cwd, *argv]
        full = [*self._nsenter(ids), "--", *argv]
        log = open(self.dir / f"bg-{uuid.uuid4().hex[:6]}.log", "wb")
        return await asyncio.create_subprocess_exec(
            *full, env=self._env(env), stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True,
        )

    def host_path(self, path: str) -> str:
        """Host-visible path of a sandbox path (resolved by the kernel inside the box root)."""
        return f"/proc/{self.init_pid}/root{path}"

    async def stop(self, delete: bool = True) -> None:
        if self._memory_task is not None:
            self._memory_task.cancel()
            # a guard that died of an error must not keep stop() from killing the box (the error is recorded)
            try:
                await self._memory_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                self._memory_event({'event':'guard_error','error':repr(exc)})
            self._memory_task=None
        for pid in (self.init_pid,):
            if pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for p in (self._slirp, self._unshare):
            if p and p.returncode is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(p.wait(), 10)
                except asyncio.TimeoutError:
                    pass
        self.init_pid = None
        if self._dir is not None and self.pod is not None and self.pod.multiuid:
            # Files of subordinate ids (setpriv'd processes) are not ours on the host.
            await (self.pod.purge if delete else self.pod.normalize_owner)([self._dir])
        if delete and self._dir is not None:
            await asyncio.get_running_loop().run_in_executor(None, _rmtree_force, self._dir)


def _disable_nsfake(root: Path) -> None:
    """Multi-uid sandboxes have real ids: drop the single-uid emulation from this sandbox's view.  The real gosu is
    restored (see builder.install_shims); the /etc/ld.so.preload entry stays, because _seed_runtime_files has put
    nsdev.so (no id emulation) at nsfake.so's path.  The image is untouched."""
    from .builder import resolve_in_root  # noqa: PLC0415

    try:
        for gp in ("/usr/local/bin/gosu", "/usr/bin/gosu", "/usr/sbin/gosu"):
            orig = resolve_in_root(root, gp + ".harbor-ns-orig")
            if orig.is_file():
                os.replace(orig, resolve_in_root(root, gp))
    except OSError:
        pass


async def _run_quiet(argv: list[str]) -> tuple[int, str]:
    p = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT)
    out, _ = await p.communicate()
    return p.returncode or 0, out.decode(errors="replace")


@dataclass
class Pod:
    """A user namespace (plus a hub network namespace) shared by a trial's services.

    Multi-service (docker-compose) tasks get one sandbox per service, each with
    its own network namespace attached by a veth pair to a bridge in the hub, so
    services reach each other by name (via /etc/hosts) and may listen on the
    same ports, like a compose network.  Each service also has its own
    slirp4netns uplink for internet access.
    """

    bridge: bool = False
    subnet_prefix: str = "172.31.0"
    multiuid: bool = False  # map the caller's subordinate id range (see module doc)
    holder: asyncio.subprocess.Process | None = None
    _n_links: int = 0

    @property
    def holder_pid(self) -> int:
        assert self.holder is not None
        return self.holder.pid

    def ip(self, index: int) -> str:
        return f"{self.subnet_prefix}.{index + 2}"

    async def start(self) -> None:
        if self.multiuid:
            # No --map-root-user: the full mapping is written below (newuidmap/newgidmap).
            argv = ["unshare", "--user", "--net", "sleep", "infinity"]
        else:
            argv = ["unshare", "--user", "--map-root-user", "--net", "sleep", "infinity"]
        self.holder = await asyncio.create_subprocess_exec(
            *argv,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # unshare execs sleep in place; wait until the namespace exists.
        for _ in range(500):
            try:
                if os.readlink(f"/proc/{self.holder.pid}/ns/user") != os.readlink("/proc/self/ns/user"):
                    break
            except FileNotFoundError:
                pass
            await asyncio.sleep(0.01)
        else:
            raise RuntimeError("pod user namespace did not appear")
        if self.multiuid:
            await self._map_ids()
        if self.bridge:
            rc, out = await _run_quiet(["nsenter", "-t", str(self.holder_pid), "-U", "-n", "--preserve-credentials",
                                        "sh", "-c", "ip link add br0 type bridge && ip link set br0 up && ip link set lo up"])
            if rc != 0:
                raise RuntimeError(f"pod bridge setup failed: {out}")

    async def _map_ids(self) -> None:
        """uid/gid 0 -> the caller, 1..N -> its subordinate range (newuidmap/newgidmap are setuid
        helpers that only grant /etc/subuid|subgid ranges; setgroups stays allowed)."""
        pid = str(self.holder_pid)
        for tool, own, table in (("newuidmap", os.getuid(), "/etc/subuid"), ("newgidmap", os.getgid(), "/etc/subgid")):
            start, count = _subid_range(table)
            rc, out = await _run_quiet([tool, pid, "0", str(own), "1", *_subid_extents(start, count)])
            if rc != 0:
                await self.stop()
                raise RuntimeError(f"{tool} failed: {out.strip()}")

    def enter_argv(self) -> list[str]:
        """argv prefix that runs a command as root of the pod's user namespace (host mount namespace), with every
        capability over the mapped ids (the disk cap's multi-uid reads, diskcap.NsView); [] once it is gone."""
        if self.holder is None or self.holder.returncode is not None:
            return []
        return ["nsenter", "-t", str(self.holder_pid), "-U", "--preserve-credentials", "--"]

    async def _in_userns(self, argv: list[str]) -> None:
        """Run argv as root of the pod's user namespace (host mount namespace): it has
        CAP_DAC_OVERRIDE/CAP_CHOWN over every mapped id, which the caller lacks on the host."""
        if self.holder is None or self.holder.returncode is not None:
            raise RuntimeError("pod user namespace is gone")
        rc, out = await _run_quiet(["nsenter", "-t", str(self.holder_pid), "-U", "--preserve-credentials", "--",
                                    *argv])
        if rc != 0:
            raise RuntimeError(f"{argv[0]} in the pod user namespace failed: {out[-500:]}")

    async def purge(self, paths: list[Path]) -> None:
        """Delete host paths that may hold files of subordinate ids."""
        paths = [str(p) for p in paths if os.path.lexists(p)]
        if paths:
            await self._in_userns(["rm", "-rf", "--one-file-system", "--", *paths])

    async def normalize_owner(self, paths: list[Path]) -> None:
        """chown host paths back to the caller (namespace uid/gid 0), recursively."""
        paths = [str(p) for p in paths if os.path.lexists(p)]
        if paths:
            await self._in_userns(["chown", "-R", "-h", "0:0", "--", *paths])

    async def attach(self, box: Sandbox, ip: str) -> None:
        """Connect a started sandbox to the pod bridge as eth1 with the given IP."""
        self._n_links += 1
        veth = f"vh{self._n_links}"
        rc, out = await _run_quiet([
            "nsenter", "-t", str(self.holder_pid), "-U", "-n", "--preserve-credentials", "sh", "-c",
            f"ip link add {veth} type veth peer name eth1 netns {box.init_pid} && "
            f"ip link set {veth} master br0 && ip link set {veth} up"])
        if rc != 0:
            raise RuntimeError(f"veth setup failed: {out}")
        rc, out = await _run_quiet([
            "nsenter", "-t", str(box.init_pid), "-U", "-n", "--preserve-credentials", "sh", "-c",
            f"ip addr add {ip}/24 dev eth1 && ip link set eth1 up"])
        if rc != 0:
            raise RuntimeError(f"eth1 setup failed: {out}")

    async def stop(self) -> None:
        if self.holder and self.holder.returncode is None:
            try:
                os.killpg(self.holder.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self.holder.wait()
