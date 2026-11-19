"""Build a task's Dockerfile into a flat rootfs directory, without Docker.

* Base images are pulled with ``apptainer build --sandbox <dir> docker://<ref>``
  (Docker Hub credentials from ~/.docker/config.json are passed through to
  Apptainer to avoid anonymous rate limits) and cached as SIF files on shared
  storage so other nodes do not re-pull.
* Each stage starts from an XFS-reflink copy of its base rootfs (instant), then
  RUN steps execute in a rootless build sandbox (see ``sandbox.build_run``)
  with the stage's ENV/ARG values, SHELL and WORKDIR, and COPY/ADD are applied
  by a path resolver that keeps every write inside the rootfs.
* The result is ``<images>/<key>/rootfs`` plus ``meta.json`` (env, workdir,
  user, entrypoint/cmd); ``key`` is a content hash of the build context.

Every rootfs also gets the cluster compatibility shims: apt runs its sandbox as
root, and glibc images preload nsfake.so (see nsfake.c).
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import fcntl
import fnmatch
import functools
import glob
import hashlib
import json
import os
import random
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from . import dockerfile as dfp
from .sandbox import (BUILD_MEMORY_KILLED, BUILD_MEMORY_MB, RuntimeConfig, build_run, package_cache, package_cache_env,
                      proxy_env, proxy_url)

BUILDER_VERSION = "4"
# Base refs that are never pulled from a registry: only <shared>/bases/<safe name>.sqfs (synth.prefix_bases).
LOCAL_REF_PREFIX = "local/"
SHIM_DIR = "/usr/lib/harbor-ns"
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


# BuildKit's automatic platform ARGs (Docker predefines them; `ARG TARGETARCH` without a default gets the build
# platform).  Our builder used to leave them empty, so e.g. `case "${TARGETARCH}"` hit "unsupported TARGETARCH".
PLATFORM_ARGS = {
    "TARGETPLATFORM": "linux/amd64", "TARGETOS": "linux", "TARGETARCH": "amd64", "TARGETVARIANT": "",
    "BUILDPLATFORM": "linux/amd64", "BUILDOS": "linux", "BUILDARCH": "amd64", "BUILDVARIANT": "",
}
_PLATFORM_ARG_RE = re.compile(r"(?mi)^\s*ARG\s+(" + "|".join(PLATFORM_ARGS) + r")\s*$")


class BuildError(RuntimeError):
    pass


def _resolve_user(rootfs: Path, spec: str) -> tuple[int, int, str] | None:
    """USER spec (name|uid[:group|gid]) -> (uid, gid, home) from the image's passwd/group."""
    u, _, g = spec.partition(":")
    pw = {}
    try:
        for ln in resolve_in_root(rootfs, "/etc/passwd").read_text(errors="replace").splitlines():
            f = ln.split(":")
            if len(f) >= 7 and f[2].isdigit() and f[3].isdigit():
                pw[f[0]] = (int(f[2]), int(f[3]), f[5])
    except OSError:
        pass
    if u in pw:
        uid, gid, home = pw[u]
    elif u.isdigit():
        uid = int(u)
        ent = next((v for v in pw.values() if v[0] == uid), None)
        gid, home = (ent[1], ent[2]) if ent else (uid, "/")
    else:
        raise BuildError(f"USER {spec}: unknown user")
    if g:
        if g.isdigit():
            gid = int(g)
        else:
            try:
                for ln in resolve_in_root(rootfs, "/etc/group").read_text(errors="replace").splitlines():
                    f = ln.split(":")
                    if len(f) >= 3 and f[0] == g and f[2].isdigit():
                        gid = int(f[2])
            except OSError:
                pass
    return uid, gid, home


GOSU_SHIM = r"""#!/bin/sh
# harbor_ns replacement for gosu (see harbor_ns/nsfake.c): the sandbox maps only
# uid 0, so the identity switch is emulated via NSFAKE_ID for the exec'd command.
set -e
if [ "$#" -lt 2 ]; then echo "Usage: gosu user-spec command [args]" >&2; exit 1; fi
spec=$1; shift
u=${spec%%:*}; g=; case "$spec" in *:*) g=${spec#*:};; esac
uid=$(id -u "$u" 2>/dev/null || echo "$u")
if [ -n "$g" ]; then gid=$(getent group "$g" | cut -d: -f3); gid=${gid:-$g}
else gid=$(id -g "$u" 2>/dev/null || echo 0); fi
export NSFAKE_ID="$uid:$uid:$uid:$gid:$gid:$gid"
home=$(getent passwd "$u" | cut -d: -f6) || true
[ -n "$home" ] && export HOME="$home"
exec "$@"
"""


# Installs tmux the way terminus-2 would at trial start, but once per image.
# apt lists are removed afterwards only if the image had none before.
TMUX_INSTALL = r"""
command -v tmux >/dev/null 2>&1 && exit 0
if command -v apt-get >/dev/null 2>&1; then
  had_lists=$(ls /var/lib/apt/lists 2>/dev/null | grep -v -e '^lock$' -e '^partial$' | head -1)
  for i in 1 2 3; do apt-get update -qq && apt-get install -y -qq --no-install-recommends tmux && break; sleep 5; done
  [ -z "$had_lists" ] && rm -rf /var/lib/apt/lists/* 2>/dev/null
elif command -v apk >/dev/null 2>&1; then apk add --no-cache tmux
elif command -v dnf >/dev/null 2>&1; then dnf install -y tmux && dnf clean all
elif command -v yum >/dev/null 2>&1; then yum install -y tmux && yum clean all
elif command -v microdnf >/dev/null 2>&1; then microdnf install -y tmux
fi
command -v tmux >/dev/null 2>&1
"""


# apt settings for Dockerfile RUN steps only (APT_CONFIG in _run_step; the image's runtime apt is
# unchanged). Under load the cluster proxy refuses or resets connections for minutes at a time
# (2026-10-23: 920 of 929 failed TMax builds). apt >= 2.3 waits 1, 2, 4, ... s between retries, up
# to Delay::Maximum, so 10 retries ride out about 5 minutes per file; older apt ignores the delay keys.
APT_BUILD_CONF = """\
// harbor_ns: build-time only (APT_CONFIG), see harbor_ns/builder.py
Acquire::Retries "10";
Acquire::Retries::Delay "true";
Acquire::Retries::Delay::Maximum "60";
"""
APT_BUILD_CONF_PATH = f"{SHIM_DIR}/apt-build.conf"

# Downloads that failed to reach the network through the proxy, as the tools report them once
# their own retries are spent. Infrastructure failures by definition: a 404, a missing package or
# a refused localhost probe is not matched (curl/wget only count when they name the proxy port).
_APT_FETCH_FAILED = re.compile(
    r"^[WE]: Failed to fetch \S+\s+(Unable to connect to|Could not connect to|Connection failed"
    r"|Connection timed out|Could not resolve|Temporary failure resolving|5\d\d\s)", re.M)
# "too many 503 error responses": the package cache (synth.package_cache) answers 503 when the upstream proxy is down
_PIP_FAILED = re.compile(r"^ERROR: .*(ProxyError|Cannot connect to proxy|Tunnel connection failed"
                         r"|too many 50\d error responses)", re.M)
_PIP_NO_DIST = re.compile(r"^ERROR: (No matching distribution found|Could not find a version)", re.M)
# pip could not reach the package cache's index (the cache died mid-step; the step's retry falls back to the proxy)
_PIP_CACHE_DOWN = re.compile(r"after connection broken by .*: /(t/[^/\s]+/)?pypi/(simple|packages)/")
# npm and git through the cluster proxy (https): the proxy refused, reset or timed out the tunnel. A registry or host
# on loopback is a task's own service, not matched.
_NPM_FAILED = re.compile(r"^npm (?:ERR!|error) (?:network )?request to https?://(?!localhost|127\.)\S+ failed, reason: "
                         r"(?:Proxy connection ended|tunneling socket could not be established|connect ECONNREFUSED"
                         r"|connect ETIMEDOUT|socket hang up|read ECONNRESET|getaddrinfo EAI_AGAIN)", re.M)
_GIT_FAILED = re.compile(r"^fatal: unable to access '[^']+': (?:Proxy CONNECT aborted|Received HTTP code 5\d\d from "
                         r"proxy after CONNECT|Could not resolve proxy)", re.M)
# written by the builder when the package cache answered a build's request with an error (upstream unreachable and
# nothing stored: 503) or cut a download: pip then often says only "from versions: none", like a missing package
_CACHE_FAILED = re.compile(r"^\[harbor_ns\] package cache: (\d+) requests? of this attempt failed upstream", re.M)


def proxy_failures(text: str, proxy_port: int | str | None = None) -> int:
    """Number of downloads in a build log that failed at the proxy (apt, pip, curl, wget)."""
    n = len(_APT_FETCH_FAILED.findall(text)) + len(_PIP_FAILED.findall(text)) + len(_NPM_FAILED.findall(text)) \
        + len(_GIT_FAILED.findall(text)) + sum(int(m) for m in _CACHE_FAILED.findall(text))
    if "Cannot connect to proxy" in text or "ProxyError" in text or "pkgcache: upstream unreachable" in text \
            or re.search(r"too many 50\d error responses", text) or _PIP_CACHE_DOWN.search(text):
        # pip's final message is often only "No matching distribution" after proxy retries.
        n += len(_PIP_NO_DIST.findall(text))
    if proxy_port:
        n += len(re.findall(rf"curl: \(\d+\) [^\n]*(port {proxy_port}\b|from proxy)", text))
        n += len(re.findall(rf"Connecting to \S+:{proxy_port}\.\.\. failed", text))
        n += len(re.findall(rf"^fatal: unable to access '[^']+': Failed to connect to \S+ port {proxy_port}\b", text,
                            re.M))
    return n


# ---------------------------------------------------------------- package cache (synth.package_cache)
# (tag, pip) of the build running in this task: RUN steps go through the job's package cache when it is up.
_BUILD_CACHE: contextvars.ContextVar = contextvars.ContextVar("harbor_ns_build_cache", default=None)
# Build contexts that configure pip through a configuration file keep pip on its own settings. pip's precedence is
# command line > environment > configuration files, so PIP_INDEX_URL only overrides an index (or find-links/no-index)
# a pip.conf / pip.ini / PIP_CONFIG_FILE / `pip config set` names. A command-line --index-url or -i (r6d s64-00394:
# torch from download.pytorch.org in setup.sh), a requirements-file option, an image ENV PIP_INDEX_URL (image ENV is
# applied over the cache's) or an inline export still win; the rest of such a build's pip installs use the cache.
_PIP_CONFIG = re.compile(rb"pip\.conf|pip\.ini|PIP_CONFIG_FILE|pip3? config|-m pip config")
_PIP_CONFIG_NAMES = ("pip.conf", "pip.ini")


def _pip_configured(ctx: Path, max_files: int = 4000, max_bytes: int = 2 << 20, budget: int = 32 << 20) -> bool:
    """True if the build context has a pip configuration file or a text file mentions one (binary and large files are
    skipped). A
    context too big to read within `budget` bytes or `max_files` files counts as configured: pip keeps its own settings
    (uncached), as before the cache."""
    if not os.environ.get("HARBOR_NS_PACKAGE_CACHE"):
        return False
    n = read = 0
    for root, _, files in os.walk(ctx):
        for name in files:
            n += 1
            if n > max_files or read > budget:
                return True
            if name in _PIP_CONFIG_NAMES:   # shipped as a file (COPY files/ /), whatever its content says
                return True
            p = Path(root) / name
            try:
                if p.is_symlink() or p.stat().st_size > max_bytes:
                    continue
                data = p.read_bytes()
            except OSError:
                continue
            read += len(data)
            if b"\0" in data[:8192]:
                continue
            if _PIP_CONFIG.search(data):
                return True
    return False


def _build_tag(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", label or "build")[:80] + "." + secrets.token_hex(3)


def _cache_root() -> Path | None:
    """The package cache's store directory (also while it is down: the last registry, if any)."""
    c = package_cache()
    if c and c.get("root"):
        return Path(c["root"])
    reg = os.environ.get("HARBOR_NS_PACKAGE_CACHE")
    try:
        return Path(json.loads(Path(reg).read_text())["root"]) if reg else None
    except (OSError, ValueError, KeyError):
        return None


def _flush_cache_log() -> None:
    """(blocking) Ask the package cache to write out its queued log lines (it writes them from a thread)."""
    c = package_cache()
    if not c or not c.get("url"):
        return
    import urllib.request
    try:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(c["url"] + "/_pkgcache/flush", timeout=15).read()
    except (OSError, ValueError):
        pass


def _take_fetch_log(tag: str) -> list[dict]:
    """(blocking) The package cache's log lines for one build (its tag file), removed from the cache's root."""
    root = _cache_root()
    if root is None:
        return []
    _flush_cache_log()
    path = root / "tags" / f"{tag}.jsonl"
    try:
        lines = path.read_text().splitlines()
        path.unlink()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _cache_failures(tag: str, since: float) -> int:
    """Requests of this build the package cache answered with an error (503: upstream unreachable, nothing stored)
    or cut mid-download, since `since` (the tag file stays for _take_fetch_log)."""
    root = _cache_root()
    if root is None:
        return 0
    _flush_cache_log()
    try:
        lines = (root / "tags" / f"{tag}.jsonl").read_text().splitlines()
    except OSError:
        return 0
    n = 0
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if (e.get("t") or 0) >= since and (e.get("cache") == "error" or e.get("status") == 0):
            n += 1
    return n


def fetch_summary(lines: list[dict]) -> dict:
    """Counts of a build's package-cache requests: served from the store vs fetched upstream."""
    s = {"requests": len(lines), "hit": 0, "miss": 0, "coalesced": 0, "revalidated": 0, "stale": 0, "bypass": 0,
         "pass": 0, "error": 0, "relay": 0, "bytes": 0}
    for e in lines:
        c = e.get("cache") or "error"
        s[c if c in s else "error"] += 1
        s["bytes"] += int(e.get("bytes") or 0)
    s["upstream"] = s["miss"] + s["revalidated"] + s["bypass"] + s["pass"] + s["relay"]
    return s


# ---------------------------------------------------------------- paths inside a rootfs


def resolve_in_root(root: Path, path: str, follow_last: bool = True, max_links: int = 64) -> Path:
    """Resolve ``path`` as the kernel would with ``root`` as /, without escaping it."""
    root = Path(root)
    parts = [p for p in path.split("/") if p not in ("", ".")]
    cur: list[str] = []
    links = 0
    while parts:
        comp = parts.pop(0)
        if comp == "..":
            if cur:
                cur.pop()
            continue
        cand = root.joinpath(*cur, comp)
        if (parts or follow_last) and cand.is_symlink():
            links += 1
            if links > max_links:
                raise BuildError(f"too many symlinks resolving {path}")
            target = os.readlink(cand)
            if target.startswith("/"):
                cur = []
            parts = [p for p in target.split("/") if p not in ("", ".")] + parts
            continue
        cur.append(comp)
    return root.joinpath(*cur)


def mkdirs_in_root(root: Path, path: str) -> Path:
    p = resolve_in_root(root, path)
    if p.exists() and not p.is_dir():
        raise BuildError(f"{path} exists and is not a directory")
    if not p.exists():
        parent = mkdirs_in_root(root, os.path.dirname(path.rstrip("/")) or "/")
        p = parent / os.path.basename(path.rstrip("/"))
        p.mkdir(exist_ok=True)
    return p


def _copy_entry(src: Path, root: Path, dst_in_root: str, chmod: int | None) -> None:
    """Copy file/dir/symlink ``src`` to ``dst_in_root`` (a path inside root)."""
    if src.is_symlink():
        parent = mkdirs_in_root(root, os.path.dirname(dst_in_root) or "/")
        target = parent / os.path.basename(dst_in_root)
        if target.is_symlink() or target.is_file():
            target.unlink()
        os.symlink(os.readlink(src), target)
        return
    if src.is_dir():
        d = mkdirs_in_root(root, dst_in_root)
        if chmod is not None:
            os.chmod(d, chmod)
        for child in sorted(os.listdir(src)):
            _copy_entry(src / child, root, dst_in_root.rstrip("/") + "/" + child, chmod)
        return
    parent = mkdirs_in_root(root, os.path.dirname(dst_in_root.rstrip("/")) or "/")
    target = parent / os.path.basename(dst_in_root.rstrip("/"))
    if target.is_symlink():
        target = resolve_in_root(root, dst_in_root)
    if target.is_dir():
        raise BuildError(f"cannot overwrite directory {dst_in_root} with a file")
    if target.exists():
        os.chmod(target, stat.S_IMODE(target.stat().st_mode) | stat.S_IWUSR)
        target.unlink()
    shutil.copy2(src, target, follow_symlinks=False)
    if chmod is not None:
        os.chmod(target, chmod)


# ---------------------------------------------------------------- .dockerignore


def _load_dockerignore(ctx: Path) -> list[tuple[bool, str]]:
    f = ctx / ".dockerignore"
    if not f.exists():
        return []
    pats = []
    for ln in f.read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        neg = ln.startswith("!")
        if neg:
            ln = ln[1:].strip()
        pats.append((neg, os.path.normpath(ln.lstrip("/"))))
    return pats


_IGNORE_RE_CACHE: dict[str, re.Pattern] = {}


def _ignore_regex(pat: str) -> re.Pattern:
    """Docker's .dockerignore matching (moby/patternmatcher): `*` and `?` stay within one
    path component, `**` spans any number of them, `[...]` is a character class."""
    r = _IGNORE_RE_CACHE.get(pat)
    if r is not None:
        return r
    out, i, n = ["^"], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                i += 2
                if i < n and pat[i] == "/":  # `**/` = zero or more leading directories
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = pat.find("]", i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pat[i + 1:j].replace("\\", "\\\\")
                if body.startswith("^"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = j
        elif c == "\\" and i + 1 < n:
            i += 1
            out.append(re.escape(pat[i]))
        else:
            out.append(re.escape(c))
        i += 1
    out.append("$")
    r = _IGNORE_RE_CACHE[pat] = re.compile("".join(out))
    return r


def _ignored(rel: str, pats: list[tuple[bool, str]]) -> bool:
    """True if `rel` (context-relative) is excluded: the last matching pattern wins, and a
    pattern that matches a parent directory matches everything below it (Docker)."""
    parts = rel.split("/")
    prefixes = ["/".join(parts[:k]) for k in range(1, len(parts) + 1)]
    res = False
    for neg, p in pats:
        rx = _ignore_regex(p)
        if any(rx.match(x) for x in prefixes):
            res = not neg
    return res


# ---------------------------------------------------------------- base images


def _docker_creds() -> tuple[str, str] | None:
    try:
        d = json.load(open(os.path.expanduser("~/.docker/config.json")))
        a = d["auths"]["https://index.docker.io/v1/"]["auth"]
        u, p = base64.b64decode(a).decode().split(":", 1)
        return u, p
    except Exception:  # noqa: BLE001
        return None


def _is_docker_hub(ref: str) -> bool:
    first = ref.split("/", 1)[0]
    has_registry = "/" in ref and ("." in first or ":" in first or first == "localhost")
    return not has_registry or first in ("docker.io", "index.docker.io", "registry-1.docker.io")


def _pullable_ref(ref: str) -> str:
    """Apptainer rejects `name:tag@sha256:...`; the digest alone pins the image."""
    if "@" in ref:
        name, digest = ref.split("@", 1)
        last = name.rsplit("/", 1)[-1]
        if ":" in last:
            name = name[: len(name) - len(last)] + last.split(":", 1)[0]
        return f"{name}@{digest}"
    return ref


_ALPINE_OFFICIAL = re.compile(
    r"^(?P<name>(?:docker\.io/)?(?:library/)?(?:postgres|redis|nginx|httpd|mariadb|mysql|mongo|memcached|rabbitmq))"
    r"(?::(?P<tag>[^@]*))?(?:@.*)?$")


def debianize_ref(ref: str) -> str:
    """Official service images: `X:tag-alpine` -> `X:tag` (Debian variant, same version).

    musl ignores /etc/ld.so.preload, so nsfake can't emulate the service user
    that these images drop to (su-exec/setuid); the Debian variants use glibc
    and gosu, which harbor_ns handles.  Digest pins are dropped with the swap.
    """
    m = _ALPINE_OFFICIAL.match(ref)
    if not m or "alpine" not in (m.group("tag") or ""):
        return ref
    tag = re.sub(r"-?alpine[0-9.]*", "", m.group("tag")) or "latest"
    return f"{m.group('name')}:{tag}"


def _exists(p: Path | None) -> bool:
    """exists() that treats shared-storage errors (Lustre ESHUTDOWN etc.) as absent."""
    if p is None:
        return False
    try:
        return p.exists()
    except OSError:
        return False


def _safe_name(ref: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", ref)
    if len(s) > 120:
        s = s[:80] + "_" + hashlib.sha1(ref.encode()).hexdigest()[:16]
    return s


# Below this share of free scratch each build warns on stderr (evict acts below 25%, EVICT_TRIGGER_FRACTION).
SCRATCH_WARN_FRACTION = 0.15


def _tree_bytes(root: Path) -> int:
    """Allocated bytes of a directory tree (symlinks not followed); 0 if it is gone."""
    total, stack = 0, [str(root)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_blocks * 512
                    except OSError:
                        pass
        except OSError:
            pass
    return total


@contextmanager
def file_lock(path: Path):
    """flock on node-local disk (Lustre has no kernel locks; keep lock files on /tmp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _parse_apptainer_env(rootfs: Path) -> dict[str, str]:
    """Environment of a docker:// image, from Apptainer's 10-docker2singularity.sh."""
    env: dict[str, str] = {}
    f = rootfs / ".singularity.d" / "env" / "10-docker2singularity.sh"
    if not f.exists():
        return env
    for ln in f.read_text(errors="replace").splitlines():
        m = re.match(r'^export ([A-Za-z_][A-Za-z0-9_]*)="(.*)"$', ln)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        # Apptainer writes `export K="${K:-"value"}"` for non-PATH vars.
        m2 = re.match(r'^\$\{' + re.escape(k) + r':-"(.*)"\}$', v)
        if m2:
            v = m2.group(1)
        env[k] = v.replace('\\"', '"').replace("\\$", "$").replace("\\`", "`").replace("\\\\", "\\")
    return env


def _parse_apptainer_runscript(rootfs: Path) -> tuple[list[str] | None, list[str] | None]:
    f = rootfs / ".singularity.d" / "runscript"
    if not f.exists():
        return None, None
    txt = f.read_text(errors="replace")

    def grab(name: str) -> list[str] | None:
        m = re.search(rf"^{name}='(.*)'$", txt, re.M)
        if not m or not m.group(1).strip():
            return None
        try:
            return shlex.split(m.group(1))
        except ValueError:
            return None

    return grab("OCI_ENTRYPOINT"), grab("OCI_CMD")


@dataclass
class ImageMeta:
    env: dict[str, str] = field(default_factory=dict)
    workdir: str = "/"
    user: str | None = None
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None
    shell: list[str] = field(default_factory=lambda: ["/bin/sh", "-c"])
    libc: str = "unknown"
    base: str | None = None
    tmux: bool | None = None
    oci: bool = False  # env/workdir/entrypoint/cmd/user taken from the registry config
    healthcheck: dict | None = None  # Dockerfile HEALTHCHECK, in compose `healthcheck:` form

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, d: dict) -> "ImageMeta":
        m = cls()
        for k, v in d.items():
            if hasattr(m, k):
                setattr(m, k, v)
        return m


def parse_healthcheck(raw: str) -> dict | None:
    """`HEALTHCHECK [--interval=..] [--timeout=..] [--start-period=..] [--retries=N] CMD ...`
    (or NONE) -> compose-style healthcheck dict (used for depends_on: service_healthy)."""
    rest = raw.strip()
    if rest.upper() == "NONE":
        return {"test": ["NONE"]}
    hc: dict = {}
    while rest.startswith("--"):
        opt, _, rest = rest.partition(" ")
        k, _, v = opt[2:].partition("=")
        k = k.replace("-", "_")
        if k in ("interval", "timeout", "start_period", "start_interval"):
            hc[k] = v
        elif k == "retries" and v.isdigit():
            hc[k] = int(v)
        rest = rest.lstrip()
    kw, _, body = rest.partition(" ")
    if kw.upper() != "CMD" or not body.strip():
        return None
    try:
        argv = json.loads(body)
    except ValueError:
        argv = None
    hc["test"] = ["CMD", *argv] if isinstance(argv, list) else ["CMD-SHELL", body.strip()]
    return hc


_JVM_OPTS_RE = re.compile(r"(?mi)^\s*(ENV|ARG)\b.*\b(MAVEN_OPTS|GRADLE_OPTS)\b")


def _sets_jvm_opts(dockerfile: Path) -> bool:
    """Dockerfile sets MAVEN_OPTS/GRADLE_OPTS itself.  Builds before the proxy-props merge
    in _run_step lost the JVM proxy (e.g. `RUN ./gradlew dependencies || true` cached
    nothing), so such images get their own cache key and are rebuilt."""
    try:
        return bool(_JVM_OPTS_RE.search(dockerfile.read_text(errors="replace")))
    except OSError:
        return False


def _detect_libc(rootfs: Path) -> str:
    for pat in ("lib/x86_64-linux-gnu/libc.so.6", "lib64/libc.so.6", "usr/lib/x86_64-linux-gnu/libc.so.6",
                "lib/libc.so.6", "usr/lib64/libc.so.6", "usr/lib/libc.so.6"):
        if os.path.lexists(rootfs / pat):
            return "glibc"
    if glob.glob(str(rootfs / "lib/ld-musl-*.so.1")):
        return "musl"
    return "unknown"


def apply_oci_config(meta: "ImageMeta", ref: str) -> None:
    """Fill workdir/entrypoint/cmd/user/env from the registry's image config."""
    from .registry import image_config  # noqa: PLC0415

    cfg = image_config(ref)
    env = {}
    for kv in cfg.get("Env") or []:
        k, _, v = kv.partition("=")
        env[k] = v
    if env:
        meta.env = env
    meta.workdir = cfg.get("WorkingDir") or "/"
    meta.entrypoint = cfg.get("Entrypoint") or None
    meta.cmd = cfg.get("Cmd") or None
    meta.user = cfg.get("User") or None
    meta.oci = True


@dataclass
class ImageStore:
    """Local (node) image cache + shared SIF cache for base images."""

    local_root: Path  # node-local, reflink-capable (XFS /tmp)
    shared_root: Path | None  # e.g. /data/aai/tbench/harbor_ns (Lustre)
    rt: RuntimeConfig
    nsfake: Path
    build_timeout: float = 3600.0
    run_step_timeout: float = 3600.0
    # Rebuilds after downloads failed at the proxy (see _build_network_retry); 0 disables.
    network_rebuilds: int = int(os.environ.get("HARBOR_NS_NETWORK_REBUILDS", "2"))
    # seconds before rebuild 1, 2, ... (x0.5-1.5 jitter); HARBOR_NS_NETWORK_BACKOFF="60,180,300"
    network_backoff: tuple = tuple(float(x) for x in os.environ.get("HARBOR_NS_NETWORK_BACKOFF", "30,120").split(","))
    # HARBOR_NS_PERSIST_IMAGES=0: task images stay in node-local scratch (shared bases are still read and
    # published). Datasets whose tasks share prefix bases (synth.prefix_bases) rebuild a task's thin layer in
    # seconds, so copying one flat squashfs per task to /data (TMax: 14.4k images, 7.9 TB) buys nothing.
    persist_images: bool = os.environ.get("HARBOR_NS_PERSIST_IMAGES", "1") != "0"

    def __post_init__(self):
        for d in ("bases", "images", "locks", "tmp", "logs"):
            (self.local_root / d).mkdir(parents=True, exist_ok=True)
        if self.shared_root:
            for d in ("bases", "images", "logs"):
                (self.shared_root / d).mkdir(parents=True, exist_ok=True)

    # ---------------- bases
    def _apptainer_env(self, ref: str = "") -> dict[str, str]:
        e = dict(os.environ)
        e["APPTAINER_CACHEDIR"] = str(self.local_root / "apptainer-cache")
        e["APPTAINER_TMPDIR"] = str(self.local_root / "tmp")
        for k in ("SINGULARITY_CACHEDIR", "SINGULARITY_TMPDIR"):
            e.pop(k, None)
        # Docker Hub credentials only for Docker Hub refs (ghcr.io etc. 403 on them).
        creds = _docker_creds()
        if creds and _is_docker_hub(ref):
            e["APPTAINER_DOCKER_USERNAME"], e["APPTAINER_DOCKER_PASSWORD"] = creds
        return e

    async def ensure_base(self, ref: str) -> tuple[Path, ImageMeta]:
        name = _safe_name(ref)
        dest = self.local_root / "bases" / name
        rootfs = dest / "rootfs"
        metaf = dest / "meta.json"
        loop = asyncio.get_running_loop()
        if metaf.exists():
            meta = ImageMeta.from_json(json.loads(metaf.read_text()))
            if not meta.oci:
                meta = await self._refresh_oci(ref, meta, metaf)
            self.touch(dest)
            return rootfs, meta
        await loop.run_in_executor(None, self.maybe_evict)
        lock = self.local_root / "locks" / f"base-{name}.lock"
        await loop.run_in_executor(None, self._lock_acquire, lock)
        try:
            if metaf.exists():
                meta = ImageMeta.from_json(json.loads(metaf.read_text()))
                if not meta.oci:
                    meta = await self._refresh_oci(ref, meta, metaf)
                return rootfs, meta
            if dest.exists():
                subprocess.run(["chmod", "-R", "u+rwX", str(dest)], check=False)
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True)
            shared = self.shared_root / "bases" / f"{name}.sqfs" if self.shared_root else None
            unpack_error = None
            if _exists(shared) and _exists(shared.with_suffix(".json")):
                # Another node already pulled this base: unpack the shared squashfs. A scratch that ran out of space
                # is evicted (force) and the unpack tried once more (#451 lesson 7: TB-Sci-lite workers whose
                # base unpack hit ENOSPC fell through to a registry pull of local/tasksynth-sci, "denied").
                for space_try in (0, 1):
                    try:
                        await self._run(["unsquashfs", "-q", "-n", "-no-xattrs", "-d", str(rootfs), str(shared)],
                                        dest / "pull.log")
                        meta = ImageMeta.from_json(json.loads(shared.with_suffix(".json").read_text()))
                        _write_meta(metaf, meta.to_json())
                        if not meta.oci:
                            meta = await self._refresh_oci(ref, meta, metaf)
                        return rootfs, meta
                    except (BuildError, OSError, ValueError) as e:
                        unpack_error = e
                        metaf.unlink(missing_ok=True)
                        subprocess.run(["chmod", "-R", "u+rwX", str(dest)], check=False)
                        shutil.rmtree(rootfs, ignore_errors=True)
                        free0 = _free_gb(self.local_root)   # after this attempt's own cleanup
                        if space_try or not out_of_space(e, dest / "pull.log"):
                            break
                        freed = await self._recover_space(free0)
                        with open(dest / "pull.log", "a") as f:
                            f.write(f"\nharbor_ns: unpack ran out of scratch space; evicted {freed}\n")
                        if not freed["retry"]:
                            break
            if ref.startswith(LOCAL_REF_PREFIX):
                # Shared bases built here (synth.prefix_bases) exist only as the shared squashfs.
                if unpack_error is not None:
                    raise BuildError(f"local base {ref}: unpacking {shared} failed: {unpack_error}")
                raise BuildError(f"local base {ref} is not published at {shared}")
            # Unprivileged `apptainer build --sandbox` from docker:// only unpacks
            # layers (building from a local SIF would need a /proc mount, which
            # LXCfs blocks on this cluster).
            await self._run_apptainer(["apptainer", "build", "--force", "--sandbox", str(rootfs),
                                       f"docker://{_pullable_ref(ref)}"], dest / "pull.log", ref=ref)
            meta = ImageMeta(env=_parse_apptainer_env(rootfs), base=ref)
            meta.entrypoint, meta.cmd = _parse_apptainer_runscript(rootfs)
            try:
                await loop.run_in_executor(None, apply_oci_config, meta, _pullable_ref(ref))
            except Exception as e:  # noqa: BLE001  (fall back to what Apptainer kept)
                with open(dest / "pull.log", "a") as f:
                    f.write(f"\nharbor_ns: registry config fetch failed: {e!r}\n")
            self._sanitize_rootfs(rootfs)
            meta.libc = _detect_libc(rootfs)
            _write_meta(metaf, meta.to_json())
            if shared is not None:
                # Awaited (not fire-and-forget) so a process exit can't cut it off.
                await loop.run_in_executor(None, self._persist_squashfs, rootfs, shared, meta.to_json())
            return rootfs, meta
        finally:
            self._lock_release(lock)

    _locks: dict = field(default_factory=dict)

    async def _refresh_oci(self, ref: str, meta: "ImageMeta", metaf: Path) -> "ImageMeta":
        """Add the registry config to a base cached before harbor_ns read it."""
        try:
            await asyncio.get_running_loop().run_in_executor(None, apply_oci_config, meta, _pullable_ref(ref))
        except Exception:  # noqa: BLE001
            return meta
        _write_meta(metaf, meta.to_json())
        if self.shared_root:
            sj = self.shared_root / "bases" / f"{_safe_name(ref)}.json"
            if sj.exists():
                try:
                    tmp = sj.with_suffix(f".json.{os.getpid()}")
                    tmp.write_text(json.dumps(meta.to_json(), indent=1))
                    os.replace(tmp, sj)
                except OSError:
                    pass
        return meta

    def _persist_squashfs(self, rootfs: Path, dest: Path, meta: dict) -> None:
        """Pack a rootfs to squashfs on local disk, then copy it to shared storage."""
        tmp = self.local_root / "tmp" / f"{dest.name}.{os.getpid()}.{time.time_ns()}"
        try:
            if _exists(dest):
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["mksquashfs", str(rootfs), str(tmp), "-comp", "zstd", "-quiet",
                                "-no-progress", "-processors", "8"], capture_output=True)
            if r.returncode != 0:
                return
            part = dest.with_name(dest.name + f".partial{os.getpid()}")
            shutil.copyfile(tmp, part)
            os.replace(part, dest)
            dest.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        except OSError:
            pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    # ---------------- local cache bookkeeping (LRU eviction, in-use markers)
    def touch(self, image_dir: Path) -> None:
        try:
            (image_dir / ".last_used").write_text(f"{time.time():.0f}\n")
        except OSError:
            pass

    def mark_in_use(self, image_dir: Path, token: str) -> None:
        d = image_dir / ".inuse"
        d.mkdir(exist_ok=True)
        tmp = d / f".{token}.tmp"   # atomic: a scan never reads a marker without its pid (and drops it as dead)
        tmp.write_text(str(os.getpid()))
        os.replace(tmp, d / token)
        self.touch(image_dir)

    def claim(self, image_dir: Path, token: str) -> bool:
        """Pin an image (or base) for a trial or a build: the in-use marker first, then meta.json must still be there.
        evict() moves meta.json aside first and then re-checks the markers (moving it back for a pinned image), so an
        image is either pinned or reported gone (False: load or build it again), never deleted under its user. A marker
        that cannot be written for lack of space evicts (force) and is tried once more. Blocking (may evict)."""
        try:
            self.mark_in_use(image_dir, token)
        except OSError as e:
            if not out_of_space(e):
                return False
            try:
                self.maybe_evict(force=True)
                self.mark_in_use(image_dir, token)
            except OSError:
                return False
        if (image_dir / "meta.json").exists():
            return True
        self.unmark_in_use(image_dir, token)
        return False

    async def build_pinned(self, ctx: Path, dockerfile: str = "Dockerfile", label: str = "", *, token: str,
                           attempts: int = 3, **kw) -> tuple[str, Path, ImageMeta]:
        """build() and claim() the image for `token`, for callers that use the rootfs outside a harbor trial (a check
        sandbox, an author sandbox, a squash of the image): an image evicted between build() returning and the pin is
        loaded or built again. The caller unpins with unmark_in_use(image_dir(key), token) (or `pinned`); a marker of
        a live process otherwise keeps the image for that process's lifetime."""
        loop = asyncio.get_running_loop()
        for _ in range(attempts):
            key, rootfs, meta = await self.build(ctx, dockerfile, label, **kw)
            fut = loop.run_in_executor(None, self.claim, rootfs.parent, token)
            try:
                pinned = await asyncio.shield(fut)
            except asyncio.CancelledError:
                # a cancelled caller (wait_for timeout) never sees a pin the thread still writes: drop it then
                fut.add_done_callback(lambda f, d=rootfs.parent: (not f.cancelled() and f.exception() is None
                                                                  and f.result() and self.unmark_in_use(d, token)))
                raise
            if pinned:
                return key, rootfs, meta
        free = _free_gb(self.local_root)
        why = ("No space left on device: the scratch is full" if free is not None and free < 1.0
               else "evicted each time")
        raise BuildError(f"{label or ctx}: image could not be pinned ({why})")

    @asynccontextmanager
    async def pinned(self, ctx: Path, dockerfile: str = "Dockerfile", label: str = "", **kw):
        """`async with store.pinned(ctx, label=...) as (key, rootfs, meta):` the image is pinned inside the block."""
        token = f"pin-{os.getpid()}-{secrets.token_hex(6)}"
        key, rootfs, meta = await self.build_pinned(ctx, dockerfile, label, token=token, **kw)
        try:
            yield key, rootfs, meta
        finally:
            self.unmark_in_use(rootfs.parent, token)

    def unmark_in_use(self, image_dir: Path, token: str) -> None:
        try:
            (image_dir / ".inuse" / token).unlink()
        except OSError:
            pass

    @staticmethod
    def _in_use(image_dir: Path) -> bool:
        d = image_dir / ".inuse"
        if not d.is_dir():
            return False
        live = False
        for m in d.iterdir():
            if m.name.startswith("."):
                continue   # a marker being written (mark_in_use)
            try:
                os.kill(int(m.read_text().strip()), 0)
                live = True
            except PermissionError:
                live = True   # a live process of another user
            except (ValueError, ProcessLookupError, OSError):
                try:
                    m.unlink()  # owner process is gone
                except OSError:
                    pass
        return live

    def free_gb(self) -> float:
        st = os.statvfs(self.local_root)
        return st.f_bavail * st.f_frsize / 1e9

    def total_gb(self) -> float:
        st = os.statvfs(self.local_root)
        return st.f_blocks * st.f_frsize / 1e9

    def maybe_evict(self, min_free_gb: float = 100, target_free_gb: float = 160,
                    min_idle_hours: float | None = None, force: bool = False) -> dict:
        """evict() under this store's scratch (one rule for every pipeline)."""
        return evict(self.local_root, min_free_gb, target_free_gb, min_idle_hours, force=force)

    async def _recover_space(self, free_at_failure: float | None = None) -> dict:
        """A forced eviction after a write ran out of space. "retry": this eviction freed something, or the scratch has
        more free space than when the write failed (another process's forced eviction got there first); never raises."""
        loop = asyncio.get_running_loop()
        try:
            out = await loop.run_in_executor(None, functools.partial(self.maybe_evict, force=True))
        except Exception as e:  # noqa: BLE001
            out = {"error": repr(e)}
        try:
            grew = free_at_failure is not None and self.free_gb() > free_at_failure + 1.0
        except OSError:
            grew = False
        out["retry"] = _freed(out) or grew
        return out

    def _lock_acquire(self, path: Path) -> None:
        f = open(path, "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        self._locks[str(path)] = f

    def _lock_release(self, path: Path) -> None:
        f = self._locks.pop(str(path), None)
        if f:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    async def _run(self, argv: list[str], log: Path, env: dict | None = None, timeout: float = 3600) -> None:
        with open(log, "ab") as lf:
            lf.write(f"\n$ {' '.join(argv)}\n".encode())
            lf.flush()
            p = await asyncio.create_subprocess_exec(*argv, stdout=lf, stderr=lf, env=env)
            try:
                rc = await asyncio.wait_for(p.wait(), timeout)
            except asyncio.TimeoutError:
                p.kill()
                await p.wait()
                rc = -9
        if rc != 0:
            tail = log.read_text(errors="replace")[-1500:]
            raise BuildError(f"{argv[0]} {argv[1]} failed ({rc}): {tail}")

    async def _run_apptainer(self, argv: list[str], log: Path, ref: str = "") -> None:
        last: Exception | None = None
        for attempt in range(4):
            try:
                await self._run(argv, log, env=self._apptainer_env(ref), timeout=3600)
                return
            except BuildError as e:  # registry hiccups through the proxy are common
                last = e
                await asyncio.sleep(10 * (attempt + 1))
        raise last  # type: ignore[misc]

    def _sanitize_rootfs(self, rootfs: Path) -> None:
        """Remove Apptainer artifacts and make every path writable by the owner."""
        for p in ("environment", "singularity", ".singularity.d", ".exec", ".run", ".shell", ".test"):
            q = rootfs / p
            if q.is_symlink() or q.is_file():
                q.unlink()
            elif q.is_dir():
                shutil.rmtree(q, ignore_errors=True)
        subprocess.run(["chmod", "-R", "u+rwX", str(rootfs)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for d in ("proc", "dev", "sys", "tmp", "etc", "root", "run"):
            (rootfs / d).mkdir(exist_ok=True)
        os.chmod(rootfs / "tmp", 0o1777)

    @staticmethod
    def _debian_archive_sources(rootfs: Path) -> None:
        """Debian suites whose files left deb.debian.org: point apt at archive.debian.org (same frozen indices).

        bullseye-security froze on 2026-09-12 and its pool moved to archive.debian.org while deb.debian.org still
        serves the index, so `apt-get install` of an updated package 404s (RTS: 3 of the first shared prefix bases,
        ~370 tasks). buster and older left entirely. Idempotent; leaves other distributions alone."""
        try:
            osr = resolve_in_root(rootfs, "/etc/os-release")
            m = re.search(r"^VERSION_CODENAME=\"?(\w+)", osr.read_text(errors="replace"), re.M) if osr.is_file() else None
        except (BuildError, OSError):
            return
        codename = m.group(1) if m else ""
        if codename == "bullseye":
            pattern = r"https?://(?:deb|security)\.debian\.org/debian-security"
        elif codename in ("buster", "stretch", "jessie"):
            pattern = r"https?://(?:deb|security)\.debian\.org/(debian-security|debian)"
        else:
            return
        changed = False
        for p in ("/etc/apt/sources.list", "/etc/apt/sources.list.d/debian.sources"):
            try:
                f = resolve_in_root(rootfs, p)
                if not f.is_file():
                    continue
                text = f.read_text(errors="replace")
                new = re.sub(pattern, lambda mm: "http://archive.debian.org/" + (
                    mm.group(1) if mm.lastindex else "debian-security"), text)
                if new != text:
                    f.write_text(new)
                    changed = True
            except (BuildError, OSError):
                continue
        if changed and codename != "bullseye":
            (rootfs / "etc/apt/apt.conf.d/99harbor-archive").write_text('Acquire::Check-Valid-Until "false";\n')

    def install_shims(self, rootfs: Path, libc: str) -> None:
        """Cluster compatibility shims (single-uid user namespace).

        Idempotent; run on every stage base and again on the final rootfs.
        """
        apt_dir = rootfs / "etc/apt/apt.conf.d"
        if apt_dir.is_dir():
            (apt_dir / "00harbor-ns").write_text('APT::Sandbox::User "root";\n')
            (mkdirs_in_root(rootfs, SHIM_DIR) / Path(APT_BUILD_CONF_PATH).name).write_text(APT_BUILD_CONF)
            self._debian_archive_sources(rootfs)
        if libc == "glibc":
            d = mkdirs_in_root(rootfs, SHIM_DIR)
            shutil.copy(self.nsfake, d / "nsfake.so")
            pre = resolve_in_root(rootfs, "/etc/ld.so.preload")
            lines = pre.read_text().splitlines() if pre.exists() else []
            entry = f"{SHIM_DIR}/nsfake.so"
            if entry not in lines:
                pre.write_text("\n".join([entry, *lines]) + "\n")
            # gosu is a static Go binary (no libc, so nsfake can't reach its setuid
            # calls); swap in a script that switches identity through NSFAKE_ID.
            for gp in ("/usr/local/bin/gosu", "/usr/bin/gosu", "/usr/sbin/gosu"):
                g = resolve_in_root(rootfs, gp)
                if g.is_file() and not g.read_bytes()[:2] == b"#!":
                    g.rename(g.with_name("gosu.harbor-ns-orig"))
                    g.write_text(GOSU_SHIM)
                    os.chmod(g, 0o755)

    # ---------------- task images
    @staticmethod
    def context_key(ctx: Path, dockerfile: str = "Dockerfile") -> str:
        h = hashlib.sha256(f"harbor_ns-builder-{BUILDER_VERSION}\n{dockerfile}\n".encode())
        # New keys only for the contexts whose build changes (no other cached image is invalidated):
        if (ctx / ".dockerignore").exists():  # Docker .dockerignore semantics (moby/patternmatcher)
            h.update(b"dockerignore-v2\n")
        try:  # images built before PLATFORM_ARGS saw these ARGs empty
            if _PLATFORM_ARG_RE.search((ctx / dockerfile).read_text(errors="replace")):
                h.update(b"platform-args-1\n")
        except OSError:
            pass
        for dirpath, dirnames, filenames in os.walk(ctx):
            dirnames.sort()
            for fn in sorted(filenames):
                p = Path(dirpath) / fn
                rel = p.relative_to(ctx).as_posix()
                st = p.lstat()
                h.update(f"{rel}\0{stat.S_IMODE(st.st_mode)}\0".encode())
                if p.is_symlink():
                    h.update(os.readlink(p).encode())
                else:
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 20), b""):
                            h.update(chunk)
                h.update(b"\0")
        return h.hexdigest()[:24]

    def image_dir(self, key: str) -> Path:
        return self.local_root / "images" / key

    async def _ensure_owners(self, key: str) -> None:
        """An image unpacked from the shared cache before owners.json existed: write it now (multi-uid only)."""
        from . import owners  # noqa: PLC0415

        d = self.image_dir(key)
        if not owners.wanted() or (d / "owners.json").exists() or not (d / "unpack.log").exists():
            return
        shared = self.shared_root / "images" / f"{key}.sqfs" if self.shared_root else None
        if _exists(shared):
            await asyncio.get_running_loop().run_in_executor(
                None, owners.owners_from_sqfs, shared, d / "owners.json", d / "unpack.log")

    def load(self, key: str) -> tuple[Path, ImageMeta] | None:
        d = self.image_dir(key)
        if (d / "meta.json").exists():
            return d / "rootfs", ImageMeta.from_json(json.loads((d / "meta.json").read_text()))
        return None

    async def build(self, ctx: Path, dockerfile: str = "Dockerfile", label: str = "",
                    ensure_tmux: bool = False, memory_mb: int | None = None) -> tuple[str, Path, ImageMeta]:
        """memory_mb: kill a RUN step whose process tree holds more resident memory than this (the step fails; default:
        HARBOR_NS_BUILD_MEM_GB, or no cap)."""
        if memory_mb:
            token = BUILD_MEMORY_MB.set(memory_mb)
            try:
                return await self.build(ctx, dockerfile, label, ensure_tmux)
            finally:
                BUILD_MEMORY_MB.reset(token)
        key = self.context_key(ctx, dockerfile + ("+tmux" if ensure_tmux else "")
                               + ("+jvmproxy" if _sets_jvm_opts(ctx / dockerfile) else ""))
        got = self.load(key)
        if got:
            self.touch(self.image_dir(key))
            await self._ensure_owners(key)
            return key, *got
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.maybe_evict)
        lock = self.local_root / "locks" / f"img-{key}.lock"
        await loop.run_in_executor(None, self._lock_acquire, lock)
        try:
            got = self.load(key)
            if got:
                await self._ensure_owners(key)
                return key, *got
            d = self.image_dir(key)
            if d.exists():
                subprocess.run(["chmod", "-R", "u+rwX", str(d)], check=False)
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            shared = self.shared_root / "images" / f"{key}.sqfs" if self.shared_root and self.persist_images else None
            if _exists(shared) and _exists(shared.with_suffix(".json")):
                # Built before (maybe on another node): unpack instead of rebuilding. -no-xattrs: a rootless
                # unpack cannot restore security.capability (unsquashfs then exits 2, which used to trigger a
                # full rebuild that depends on package mirrors being up); a rootless build has none either.
                for space_try in (0, 1):
                    try:
                        await self._run(["unsquashfs", "-q", "-n", "-no-xattrs", "-d", str(d / "rootfs"),
                                         str(shared)], d / "unpack.log")
                        # Nor can it set user xattrs on read-only files, so nsfake's owner records are read from a
                        # squashfuse mount instead (owners.json; multi-uid sandboxes replay them, see owners.py).
                        from . import owners  # noqa: PLC0415
                        if owners.wanted():
                            await asyncio.get_running_loop().run_in_executor(
                                None, owners.owners_from_sqfs, shared, d / "owners.json", d / "unpack.log")
                        meta_d = json.loads(shared.with_suffix(".json").read_text())
                        _write_meta(d / "meta.json", meta_d)
                        return key, d / "rootfs", ImageMeta.from_json(meta_d)
                    except (BuildError, OSError, ValueError) as e:
                        nospace = out_of_space(e, d / "unpack.log")
                        subprocess.run(["chmod", "-R", "u+rwX", str(d)], check=False)
                        shutil.rmtree(d, ignore_errors=True)
                        d.mkdir(parents=True, exist_ok=True)
                        free0 = _free_gb(self.local_root)   # after this attempt's own cleanup
                        if not nospace:
                            break   # a damaged squashfs etc.: rebuild from the Dockerfile, as before
                        # the scratch ran out of space: evict (force) and unpack once more; never a rebuild (it can
                        # differ from the persisted image, and needs the same space)
                        if space_try or not (await self._recover_space(free0))["retry"]:
                            raise BuildError(f"unpacking {shared} ran out of scratch space") from e
            log = d / "build.log"
            t0 = time.time()
            tag = _build_tag(label or key)
            pip = not await loop.run_in_executor(None, _pip_configured, ctx)
            token = _BUILD_CACHE.set((tag, pip))
            try:
                for space_try in (0, 1):
                    try:
                        meta, failures, attempt = await self._build_network_retry(ctx, dockerfile, d, log,
                                                                                  ensure_tmux, label or key, key)
                        break
                    except BaseException as e:  # BuildError, timeouts, cancellation (the running step is killed first)
                        await loop.run_in_executor(None, _take_fetch_log, tag)
                        nospace = not space_try and isinstance(e, Exception) and out_of_space(e, log)
                        await loop.run_in_executor(None, self._drop_failed, d, log, label or key, key, t0)
                        free0 = _free_gb(self.local_root)   # after this attempt's own cleanup
                        if not nospace:
                            raise
                        # the scratch ran out of space (#451 lesson 7): evict (force) and build once more, if space
                        # was freed (by this eviction or a concurrent one)
                        freed = await self._recover_space(free0)
                        if not freed["retry"]:
                            raise
                        d.mkdir(parents=True, exist_ok=True)
                        with open(log, "ab") as f:
                            f.write(f"[harbor_ns] the previous build ran out of scratch space; evicted {freed}; "
                                    f"building again\n".encode())
            finally:
                _BUILD_CACHE.reset(token)
            fetches = await loop.run_in_executor(None, _take_fetch_log, tag)
            if fetches:
                # what this image fetched through the package cache (URL, version, SHA-256: pinning, licensing)
                with open(d / "fetch.jsonl", "w") as f:
                    f.writelines(json.dumps(e) + "\n" for e in fetches)
            (d / "fetch.json").write_text(json.dumps({"tag": tag, "pip_cache": pip, "cache": bool(fetches),
                                                      **fetch_summary(fetches)}))
            meta_d = meta.to_json()
            meta_d["build_seconds"] = round(time.time() - t0, 1)
            meta_d["label"] = label
            if attempt or failures:
                meta_d["build_attempts"] = attempt + 1
                meta_d["proxy_failures_last_attempt"] = failures
            _write_meta(d / "meta.json", meta_d)
            await loop.run_in_executor(None, lambda: self._log_scratch(label or key, key, True, t0, _tree_bytes(d)))
            if shared is not None:
                await loop.run_in_executor(None, self._persist_squashfs, d / "rootfs", shared, meta_d)
            return key, d / "rootfs", meta
        finally:
            self._lock_release(lock)

    def _save_failed_log(self, log: Path, name: str, key: str) -> None:
        if self.shared_root and log.exists():
            shutil.copy(log, self.shared_root / "logs" / f"{name}.{key}.failed.log")

    def _drop_failed(self, d: Path, log: Path, name: str, key: str, t0: float) -> None:
        """Delete a failed build's image directory. Without meta.json nothing reuses it and eviction skipped it: at
        1,000 builds per job (task synthesis r6) failed builds filled the 200 GB job scratch, and the out-of-disk
        errors read as build failures. Its log is kept in <shared>/logs (and, without a shared root, the last
        256 KiB in <local>/logs)."""
        size = _tree_bytes(d)
        try:
            self._save_failed_log(log, name, key)
            if not self.shared_root and log.exists():
                with open(log, "rb") as f:
                    f.seek(max(0, log.stat().st_size - (256 << 10)))
                    (self.local_root / "logs" / f"{_safe_name(name)}.{key}.failed.log").write_bytes(f.read())
        except OSError:
            pass
        subprocess.run(["chmod", "-R", "u+rwX", str(d)], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(d, ignore_errors=True)
        self._log_scratch(name, key, False, t0, size)

    def _log_scratch(self, name: str, key: str, ok: bool, t0: float, image_bytes: int) -> None:
        """One line per build in <local>/logs/builds.jsonl (image size, scratch left); a warning on stderr once
        free space is below 15% of the filesystem, so growth shows before builds fail."""
        try:
            free, total = self.free_gb(), self.total_gb()
            with open(self.local_root / "logs" / "builds.jsonl", "a") as f:
                f.write(json.dumps({"time": time.time(), "label": name, "key": key, "ok": ok,
                                    "seconds": round(time.time() - t0, 1), "image_gb": round(image_bytes / 1e9, 3),
                                    "free_gb": round(free, 1), "total_gb": round(total, 1)}) + "\n")
        except OSError:
            return
        if free < SCRATCH_WARN_FRACTION * total:
            print(f"[harbor_ns] scratch low: {free:.0f} of {total:.0f} GB free in {self.local_root} after building "
                  f"{name} ({image_bytes / 1e9:.1f} GB); see {self.local_root / 'logs' / 'builds.jsonl'}",
                  file=sys.stderr, flush=True)

    async def _build_network_retry(self, ctx: Path, dockerfile: str, d: Path, log: Path, ensure_tmux: bool,
                                   name: str, key: str) -> tuple[ImageMeta, int, int]:
        """``_build``, redone from scratch while downloads fail at the proxy.

        Task setup scripts commonly run ``apt-get update && apt-get install ...`` without
        ``set -e`` (all 14,601 TMax post_install.sh), so a download refused by the congested proxy
        surfaces steps later (``pip3: not found``, exit 127; retrying that step cannot help) or not at
        all (the image builds without the package). A build whose log shows proxy failures is
        therefore rebuilt after a backoff, whether it failed or not; the last attempt's outcome
        stands. Returns (meta, proxy failures in the last attempt, attempt index)."""
        port = (proxy_url() or "").rsplit(":", 1)[-1] or None
        attempts = 1 + max(0, self.network_rebuilds)
        build_cache = _BUILD_CACHE.get()
        for attempt in range(attempts):
            offset = log.stat().st_size if log.exists() else 0
            started = time.time()
            error = None
            try:
                meta = await asyncio.wait_for(self._build(ctx, dockerfile, d, log, ensure_tmux), self.build_timeout)
            except asyncio.TimeoutError:
                raise BuildError(f"build exceeded {self.build_timeout}s") from None
            except BuildError as e:
                error = e
            except Exception:
                self._save_failed_log(log, name, key)
                raise
            cache_failed = await asyncio.get_running_loop().run_in_executor(
                None, _cache_failures, build_cache[0], started) if build_cache else 0
            if cache_failed:
                with open(log, "ab") as f:
                    f.write(f"\n[harbor_ns] package cache: {cache_failed} requests of this attempt failed upstream "
                            f"(503 or cut download; see the cache's log for tag {build_cache[0]})\n".encode())
            with open(log, "rb") as f:
                f.seek(offset)
                failures = proxy_failures(f.read().decode(errors="replace"), port)
            if not failures or attempt == attempts - 1:
                break
            delay = self.network_backoff[min(attempt, len(self.network_backoff) - 1)] * random.uniform(0.5, 1.5)
            outcome = f"build failed: {error}" if error else "build succeeded"
            with open(log, "ab") as f:
                f.write(f"\n[harbor_ns] {failures} downloads failed at the proxy ({outcome}); rebuilding from "
                        f"scratch in {delay:.0f}s (attempt {attempt + 2}/{attempts})\n".encode())
            await asyncio.sleep(delay)
            for p in list(d.iterdir()):
                if p == log:
                    continue
                subprocess.run(["chmod", "-R", "u+rwX", str(p)], check=False)
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
        if error is not None:
            self._save_failed_log(log, name, key)
            raise error
        return meta, failures, attempt

    async def _build(self, ctx: Path, dockerfile: str, d: Path, log: Path, ensure_tmux: bool = False) -> ImageMeta:
        text = (ctx / dockerfile).read_text()
        stages, global_args = dfp.parse(text)
        # Only build stages the final stage depends on.
        needed = {len(stages) - 1}
        names = {s.name: s.index for s in stages if s.name}
        changed = True
        while changed:
            changed = False
            for i in list(needed):
                s = stages[i]
                deps = []
                if s.base.lower() in names:
                    deps.append(names[s.base.lower()])
                for ins in s.instructions:
                    fr = ins.flags.get("from")
                    if fr is not None:
                        if fr.lower() in names:
                            deps.append(names[fr.lower()])
                        elif fr.isdigit():
                            deps.append(int(fr))
                for j in deps:
                    if j not in needed:
                        needed.add(j)
                        changed = True
        ignore = _load_dockerignore(ctx)
        built: dict[int, tuple[Path, ImageMeta]] = {}
        with open(log, "ab") as lf:
            for s in stages:
                if s.index not in needed:
                    continue
                sroot = d / ("rootfs" if s.index == len(stages) - 1 else f"stage{s.index}")
                meta = await self._build_stage(s, stages, names, built, global_args, ctx, ignore, sroot, lf)
                built[s.index] = (sroot, meta)
        final_root, final_meta = built[len(stages) - 1]
        self.install_shims(final_root, _detect_libc(final_root))
        if ensure_tmux:
            with open(log, "ab") as lf:
                lf.write(b"\n===== harbor_ns: ensure tmux (terminus-2 would install it at trial start) =====\n")
                lf.flush()
                cache = _BUILD_CACHE.get()
                env = {**proxy_env(), **(package_cache_env(*cache) if cache else {}), **final_meta.env,
                       "DEBIAN_FRONTEND": "noninteractive"}
                env.setdefault("HOME", "/root")
                if resolve_in_root(final_root, APT_BUILD_CONF_PATH).is_file():
                    env.setdefault("APT_CONFIG", APT_BUILD_CONF_PATH)
                rc = await build_run(self.rt, final_root, ["/bin/sh", "-c", TMUX_INSTALL], env, "/", lf,
                                     timeout=900)
            final_meta.tmux = rc == 0
        # Drop intermediate stages.
        for i, (p, _) in built.items():
            if i != len(stages) - 1:
                subprocess.run(["chmod", "-R", "u+rwX", str(p)], check=False)
                shutil.rmtree(p, ignore_errors=True)
        return built[len(stages) - 1][1]

    async def _build_stage(self, s, stages, names, built, global_args, ctx: Path, ignore, sroot: Path, lf) -> ImageMeta:
        lf.write(f"\n===== stage {s.index} FROM {s.base} =====\n".encode())
        lf.flush()
        base_l = s.base.lower()
        token = f"build-{os.getpid()}-{time.time_ns()}"
        pinned = False
        if base_l in names and names[base_l] in built:
            src_root, bmeta = built[names[base_l]]
        elif base_l == "scratch":
            src_root, bmeta = None, ImageMeta()
        else:
            base = s.base
            uses_apk = any(i.cmd == "RUN" and re.search(r"\bapk\b", i.raw + "".join(h.content for h in i.heredocs))
                           for i in s.instructions)
            if not uses_apk and debianize_ref(base) != base:
                lf.write(f"    harbor_ns: base {base} -> {debianize_ref(base)} (musl can't run nsfake)\n".encode())
                base = debianize_ref(base)
            src_root, bmeta = await self.ensure_base(base)
            # pinned like a trial's image (claim): evicted between ensure_base and the pin -> unpack/pull it again
            loop = asyncio.get_running_loop()
            for attempt in range(3):
                if attempt:
                    src_root, bmeta = await self.ensure_base(base)
                if await loop.run_in_executor(None, self.claim, src_root.parent, token):
                    pinned = True
                    break
            else:
                raise BuildError(f"base {base} could not be pinned (evicted, or the scratch is full)")
        if src_root is not None:
            if not pinned:
                self.mark_in_use(src_root.parent, token)
            try:
                r = subprocess.run(["cp", "-a", "--reflink=auto", str(src_root), str(sroot)],
                                   capture_output=True)
            finally:
                self.unmark_in_use(src_root.parent, token)
            if r.returncode != 0:
                raise BuildError(f"copy base rootfs failed: {r.stderr.decode()[-500:]}")
        else:
            sroot.mkdir(parents=True)
            self._sanitize_rootfs(sroot)
        meta = ImageMeta.from_json(json.loads(json.dumps(bmeta.to_json())))
        meta.base = s.base
        meta.libc = _detect_libc(sroot)
        self.install_shims(sroot, meta.libc)
        env = dict(meta.env)
        env.setdefault("PATH", DEFAULT_PATH)
        args: dict[str, str] = {}
        workdir = meta.workdir or "/"
        shell = list(meta.shell or ["/bin/sh", "-c"])

        def subst(v: str) -> str:
            return dfp.substitute(v, {**args, **env})

        for ins in s.instructions:
            c = ins.cmd
            lf.write(f"\n--> [{ins.lineno}] {c} {ins.raw[:500]}\n".encode())
            lf.flush()
            if c == "ARG":
                for k, v in dfp.parse_env_pairs(ins.raw):
                    if k in global_args and not v:
                        v = global_args[k]
                    if k in PLATFORM_ARGS and not v:
                        v = PLATFORM_ARGS[k]
                    args[k] = subst(v)
            elif c == "ENV":
                for k, v in dfp.parse_env_pairs(ins.raw):
                    env[k] = subst(v)
            elif c == "WORKDIR":
                w = subst(ins.raw.strip())
                workdir = os.path.normpath(os.path.join(workdir, w))
                mkdirs_in_root(sroot, workdir)
            elif c == "USER":
                meta.user = subst(ins.raw.strip())
            elif c == "HEALTHCHECK":
                meta.healthcheck = parse_healthcheck(ins.raw)
            elif c == "SHELL":
                if ins.json_form:
                    shell = ins.args
            elif c in ("ENTRYPOINT", "CMD"):
                val = ins.args if ins.json_form else (["/bin/sh", "-c", ins.raw] if ins.raw.strip() else None)
                if c == "ENTRYPOINT":
                    meta.entrypoint = val
                    meta.cmd = None  # Docker resets CMD when ENTRYPOINT is set
                else:
                    meta.cmd = val
            elif c == "RUN":
                await self._run_step(ins, sroot, env, args, workdir, shell, lf, user=meta.user)
            elif c in ("COPY", "ADD"):
                await self._copy_step(ins, sroot, ctx, ignore, built, names, stages, subst, workdir, lf)
            else:
                lf.write(f"    (ignored {c})\n".encode())
        meta.env = env
        meta.workdir = workdir
        meta.shell = shell
        return meta

    async def _run_step(self, ins, sroot: Path, env, args, workdir, shell, lf, user: str | None = None) -> None:
        if ins.json_form:
            argv = ins.args
        elif ins.heredocs and re.fullmatch(r"\s*<<-?\s*[\"']?\w+[\"']?\s*", ins.raw):
            body = ins.heredocs[0].content
            if body.startswith("#!"):
                # Script with its own interpreter: write it and execute it.
                sp = "/tmp/.harbor-ns-run.sh"
                f = resolve_in_root(sroot, sp)
                f.write_text(body)
                os.chmod(f, 0o755)
                argv = [sp]
            else:
                argv = [*shell, body]
        else:
            cmd = ins.raw
            for h in ins.heredocs:
                cmd += "\n" + h.content + h.name
            argv = [*shell, cmd]
        ident = _resolve_user(sroot, user) if user and user not in ("root", "0", "0:0") else None

        def step_env():
            px = proxy_env()
            # through the job's package cache while it is up (computed per try: a cache that died since falls back)
            cache = _BUILD_CACHE.get()
            if cache is not None:
                px.update(package_cache_env(*cache))
            run_env = {**px, **{k: v for k, v in args.items()}, **env}
            # An image ENV/ARG MAVEN_OPTS/GRADLE_OPTS (e.g. "-Xmx2g") must not drop the JVM proxy
            # properties: the JVM ignores http_proxy, so gradlew/mvn would have no internet.
            for k in ("MAVEN_OPTS", "GRADLE_OPTS"):
                if k in px and run_env.get(k) != px[k]:
                    run_env[k] = f"{run_env.get(k, '')} {px[k]}".strip()
            if ident is not None:
                # Docker runs RUN as the USER in effect; emulate the identity via nsfake.
                uid, gid, home = ident
                if uid or gid:
                    run_env["NSFAKE_ID"] = f"{uid}:{uid}:{uid}:{gid}:{gid}:{gid}"
                run_env.setdefault("HOME", home or "/")
            run_env.setdefault("HOME", "/root")
            if resolve_in_root(sroot, APT_BUILD_CONF_PATH).is_file():
                run_env.setdefault("APT_CONFIG", APT_BUILD_CONF_PATH)
            return run_env

        killed = BUILD_MEMORY_KILLED.set(None)
        rc = await build_run(self.rt, sroot, argv, step_env(), workdir, lf, timeout=self.run_step_timeout)
        memory_killed = BUILD_MEMORY_KILLED.get()
        BUILD_MEMORY_KILLED.reset(killed)
        if rc not in (0, 124) and not memory_killed:   # a step over its memory cap would only blow it again
            # Most build failures here are transient proxy/registry hiccups under load.
            lf.write(f"\n[harbor_ns] RUN failed with {rc}; retrying once in 20s\n".encode())
            lf.flush()
            await asyncio.sleep(20)
            killed = BUILD_MEMORY_KILLED.set(None)
            rc = await build_run(self.rt, sroot, argv, step_env(), workdir, lf, timeout=self.run_step_timeout)
            memory_killed = BUILD_MEMORY_KILLED.get()   # the retry may itself go over the memory cap
            BUILD_MEMORY_KILLED.reset(killed)
        try:
            os.unlink(resolve_in_root(sroot, "/tmp/.harbor-ns-run.sh"))
        except OSError:
            pass
        if rc != 0:
            lf.flush()
            if memory_killed:
                mb = BUILD_MEMORY_MB.get() or float(os.environ.get("HARBOR_NS_BUILD_MEM_GB") or 0) * 1024
                raise BuildError(f"RUN (line {ins.lineno}) was killed: out of memory (its processes used more than the "
                                 f"build's {mb / 1024:.0f} GB)")
            raise BuildError(f"RUN (line {ins.lineno}) failed with exit code {rc}")

    async def _copy_step(self, ins, sroot: Path, ctx: Path, ignore, built, names, stages, subst, workdir, lf) -> None:
        chmod = int(ins.flags["chmod"], 8) if ins.flags.get("chmod", "").isdigit() else None
        fr = ins.flags.get("from")
        if ins.heredocs:
            # COPY <<EOF /dst
            toks = ins.raw.split()
            dst = subst(toks[-1])
            for h in ins.heredocs:
                target = dst if not dst.endswith("/") else dst + h.name
                if not target.startswith("/"):
                    target = os.path.join(workdir, target)
                parent = mkdirs_in_root(sroot, os.path.dirname(target))
                p = parent / os.path.basename(target)
                p.write_text(h.content)
                os.chmod(p, chmod if chmod is not None else 0o644)
            return
        items = [subst(a) for a in ins.args]
        if len(items) < 2:
            raise BuildError(f"{ins.cmd} needs a source and a destination (line {ins.lineno})")
        srcs, dst = items[:-1], items[-1]
        if not dst.startswith("/"):
            dst = os.path.join(workdir, dst) + ("/" if dst.endswith("/") else "")
        if fr is not None:
            fl = fr.lower()
            if fl in names:
                src_root = built[names[fl]][0]
            elif fl.isdigit():
                src_root = built[int(fl)][0]
            else:
                src_root, _ = await self.ensure_base(subst(fr))
            matches: list[Path] = []
            for s in srcs:
                ms = glob.glob(str(resolve_in_root(src_root, os.path.dirname(s) or "/")) + "/" + os.path.basename(s)) \
                    if any(ch in s for ch in "*?[") else [str(resolve_in_root(src_root, s))]
                ms = [m for m in ms if os.path.lexists(m)]
                if not ms:
                    raise BuildError(f"COPY --from={fr}: {s} not found")
                matches += [Path(m) for m in ms]
        else:
            matches = []
            for s in srcs:
                if ins.cmd == "ADD" and re.match(r"^https?://", s):
                    tmpd = Path(tempfile.mkdtemp(dir=self.local_root / "tmp"))
                    fn = tmpd / (os.path.basename(s.split("?")[0]) or "download")
                    await asyncio.get_running_loop().run_in_executor(None, self._download, s, fn)
                    matches.append(fn)
                    continue
                pat = os.path.normpath(s.lstrip("/"))
                if pat in (".", ""):
                    found = [str(ctx)]
                elif any(ch in pat for ch in "*?["):
                    found = sorted(glob.glob(str(ctx / pat)))
                else:
                    found = [str(ctx / pat)] if os.path.lexists(ctx / pat) else []
                if not found:
                    raise BuildError(f"{ins.cmd}: source {s} not found in build context")
                matches += [Path(m) for m in found]
        multi = len(matches) > 1 or dst.endswith("/")
        for m in matches:
            if ins.cmd == "ADD" and fr is None and m.is_file() and tarfile.is_tarfile(m) and not re.match(r"^https?://", str(m)):
                dd = mkdirs_in_root(sroot, dst)
                with tarfile.open(m) as tf:
                    tf.extractall(dd, filter="tar")
                continue
            if m.is_dir() and not m.is_symlink():
                # Directory: copy its contents into dst.
                mkdirs_in_root(sroot, dst)
                for child in sorted(os.listdir(m)):
                    rel = (m / child).relative_to(ctx).as_posix() if fr is None and str(m).startswith(str(ctx)) else None
                    if rel is not None and _ignored(rel, ignore):
                        continue
                    self._copy_tree_filtered(m / child, sroot, dst.rstrip("/") + "/" + child, chmod, ctx if fr is None else None, ignore)
            else:
                target = dst.rstrip("/") + "/" + m.name if multi or self._is_dir_in_root(sroot, dst) else dst
                _copy_entry(m, sroot, target, chmod)

    def _copy_tree_filtered(self, src: Path, root: Path, dst: str, chmod, ctx: Path | None, ignore) -> None:
        if src.is_dir() and not src.is_symlink():
            d = mkdirs_in_root(root, dst)
            if chmod is not None:
                os.chmod(d, chmod)
            for child in sorted(os.listdir(src)):
                if ctx is not None and _ignored((src / child).relative_to(ctx).as_posix(), ignore):
                    continue
                self._copy_tree_filtered(src / child, root, dst + "/" + child, chmod, ctx, ignore)
        else:
            _copy_entry(src, root, dst, chmod)

    @staticmethod
    def _is_dir_in_root(root: Path, path: str) -> bool:
        try:
            return resolve_in_root(root, path).is_dir()
        except BuildError:
            return False

    @staticmethod
    def _download(url: str, dest: Path) -> None:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler())
        with opener.open(url, timeout=300) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)


@contextmanager
def _try_lock(local_root: Path, d: Path):
    """The build/pull lock of an image or base directory, non-blocking: yields False while a builder holds it (or
    the lock file cannot be opened)."""
    prefix = "img" if d.parent.name == "images" else "base"
    try:
        lf = open(local_root / "locks" / f"{prefix}-{d.name}.lock", "a")
    except OSError:
        yield False
        return
    try:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    finally:
        lf.close()


def _drop_orphan(local_root: Path, d: Path) -> bool:
    """Delete an image or base directory without meta.json unless a build or pull holds its lock (True: deleted)."""
    prefix = "img" if d.parent.name == "images" else "base"
    with open(local_root / "locks" / f"{prefix}-{d.name}.lock", "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False  # its builder is running
        try:
            if not (d / "meta.json").exists() and not ImageStore._in_use(d):
                subprocess.run(["chmod", "-R", "u+rwX", str(d)], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                shutil.rmtree(d, ignore_errors=True)
                return True
            return False
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


# Eviction (one rule for every pipeline, #451 lesson 7). Condor job scratch is an XFS project quota, which charges every
# reflinked copy in full: each task image built FROM a base costs the base's whole size (TB-Sci-lite: 2.8 GB per task,
# so ~70 per-task images fill a 200 GB worker), however little it adds.
EVICT_TRIGGER_FRACTION = 0.25   # evict when less than min(min_free_gb, 25% of the scratch) is free ...
EVICT_TARGET_FRACTION = 0.40    # ... down to min(target_free_gb, 40%) free
EVICT_IDLE_H = 1 / 6            # never an image or base used in the last 10 min (HARBOR_NS_EVICT_IDLE_H)


def evict(local_root: Path, min_free_gb: float = 100, target_free_gb: float = 160,
          min_idle_hours: float | None = None, *, force: bool = False) -> dict:
    """Delete least-recently-used unpacked images/bases (not in use) when disk runs low; returns what it did.

    When less than min(min_free_gb, 25% of the scratch filesystem) is free (statvfs: on condor scratch, the job's
    request_disk quota), delete until min(target_free_gb, 40%) is free: first orphans (a directory without meta.json
    whose build lock is free: its builder died mid-build), then images and bases that no running trial uses (in-use
    markers), least recently used first, never one used within min_idle_hours (default HARBOR_NS_EVICT_IDLE_H, else
    10 min). Images used in the last hours go last, so a pool that reuses its images (TB2.1, TB-Lite) keeps them, and
    one that uses each image once (synthesized tasks) loses the old ones instead of filling the scratch.
    force (recovery after ENOSPC): evict to the target whatever is free (same idle floor: some users of a rootfs, such
    as task_synthesis' check sandbox or synth.prefix_bases' squash, do not mark it in use).
    Each candidate is checked again when its turn comes (a trial may have pinned it since the listing): meta.json is
    unlinked first, then the in-use markers are read, and a pinned image gets its meta.json back (ImageStore.claim
    marks first and then checks meta.json).

    Until #451 lesson 7, aa-evals/eval evicted only below an absolute 100 GB free and only images idle >= 5 h: a 200 GB
    TB-Sci-lite worker filled in 2-4 h with younger per-task images (ENOSPC, then base unpacks failing). task-synthesis
    capped the thresholds at 15% / 30% and dropped the idle floor to 10 min only under 10% free."""
    local_root = Path(local_root)
    if min_idle_hours is None:
        min_idle_hours = float(os.environ.get("HARBOR_NS_EVICT_IDLE_H") or EVICT_IDLE_H)

    def stat():
        st = os.statvfs(local_root)
        return st.f_bavail * st.f_frsize / 1e9, st.f_blocks * st.f_frsize / 1e9
    free, total = stat()
    trigger = min(min_free_gb, EVICT_TRIGGER_FRACTION * total)
    target = min(target_free_gb, EVICT_TARGET_FRACTION * total)
    out = {"free_gb": round(free, 1), "total_gb": round(total, 1), "removed": 0, "orphans": 0, "force": force}
    if free >= (target if force else trigger):
        return out
    (local_root / "locks").mkdir(parents=True, exist_ok=True)
    with open(local_root / "locks" / "evict.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            cands = []
            now = time.time()
            for sub in ("images", "bases"):
                if not (local_root / sub).is_dir():
                    continue
                for d in (local_root / sub).iterdir():
                    if not (d / "meta.json").exists():
                        if not ImageStore._in_use(d):
                            cands.append((0, d))  # an orphan, if its lock is free (checked when deleting)
                        continue
                    if ImageStore._in_use(d):
                        continue
                    lu = d / ".last_used"
                    try:
                        t = max(lu.stat().st_mtime if lu.exists() else 0, (d / "meta.json").stat().st_mtime)
                    except OSError:
                        continue
                    if now - t < min_idle_hours * 3600:
                        continue  # may still be about to back a trial (built, not yet marked in use)
                    cands.append((t, d))
            idle = []
            for t, d in sorted(cands):
                if stat()[0] >= target:
                    break
                if t == 0:
                    if _drop_orphan(local_root, d):
                        out["orphans"] += 1
                    continue
                # Checked again now (a long pass deletes many trees): used or pinned since the listing -> keep.
                lu = d / ".last_used"
                try:
                    t = max(lu.stat().st_mtime if lu.exists() else 0, (d / "meta.json").stat().st_mtime)
                except OSError:
                    continue
                if time.time() - t < min_idle_hours * 3600 or ImageStore._in_use(d):
                    continue
                # Under its build/pull lock (skipped while a build or unpack holds it, so they never see the window
                # below): meta.json moved aside first (a rename needs no space on a full disk), so concurrent readers
                # see a miss rather than a half-deleted rootfs; then the markers once more: a trial or build that
                # pinned it in between (ImageStore.claim) keeps it, and meta.json moves back.
                with _try_lock(local_root, d) as locked:
                    if not locked:
                        continue
                    aside = d / ".meta.json.evicting"
                    try:
                        os.replace(d / "meta.json", aside)
                    except OSError:
                        continue
                    if ImageStore._in_use(d):
                        try:
                            os.replace(aside, d / "meta.json")
                        except OSError:
                            pass
                        continue
                    subprocess.run(["chmod", "-R", "u+rwX", str(d)], check=False,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    shutil.rmtree(d, ignore_errors=True)
                out["removed"] += 1
                idle.append((now - t) / 3600)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    out["free_after_gb"] = round(stat()[0], 1)
    if idle:
        out["idle_h_max"], out["idle_h_min"] = round(max(idle), 2), round(min(idle), 2)
    if out["removed"] or out["orphans"]:
        try:   # one line per eviction that deleted something (scratch accounting, #451)
            with open(local_root / "logs" / "evict.jsonl", "a") as f:
                f.write(json.dumps({"time": round(time.time(), 1), **out}) + "\n")
        except OSError:
            pass
    return out


_NO_SPACE = ("No space left on device", "Disk quota exceeded")


def _freed(ev: dict) -> bool:
    return bool(ev.get("removed") or ev.get("orphans"))


def _free_gb(path: Path) -> float | None:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return None


def _write_meta(path: Path, data: dict) -> None:
    """meta.json written whole or not at all (a 0-byte file from a full disk made every later load fail)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def out_of_space(exc: BaseException | None = None, log: Path | None = None) -> bool:
    """True if a failure was the scratch filling up: ENOSPC/EDQUOT in the exception chain, or in the last 64 KiB of
    a build/unpack log."""
    e = exc
    while e is not None:
        if isinstance(e, OSError) and e.errno in (28, 122):
            return True
        if any(m in str(e) for m in _NO_SPACE):
            return True
        e = e.__cause__ or e.__context__
    if log is not None:
        try:
            with open(log, "rb") as f:
                f.seek(max(0, os.path.getsize(log) - (64 << 10)))
                tail = f.read().decode(errors="replace")
            return any(m in tail for m in _NO_SPACE)
        except OSError:
            return False
    return False
