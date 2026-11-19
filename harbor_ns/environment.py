"""Harbor environment backed by harbor_ns rootless sandboxes.

Use with Harbor >= 0.24:

    harbor run --env harbor_ns.environment:NsEnvironment ...

(with this repo on PYTHONPATH).  Optional ``--ek`` kwargs:
    ns_local_root=/tmp/$USER/harbor_ns      node-local image/overlay scratch
    ns_shared_root=/data/.../harbor_ns      shared cache of base/task images (squashfs)
    ns_cpu_affinity=true                    pin each trial to `cpus` host cores
    ns_multiuid=true                        map the subuid/subgid range (real users, no nsfake;
                                            default HARBOR_NS_MULTIUID=1, else off; see MULTIUID.md)
    ns_memcap=<factor>                      kill the main sandbox's largest process while its memory is above
                                            memory_mb x factor (default HARBOR_NS_MEMCAP, else off; memcap.py)
    ns_diskcap=<factor>                     cap every file and the trial's overlay upper dirs at storage_mb x factor
                                            (RLIMIT_FSIZE + a watchdog that kills the writer above it; default
                                            HARBOR_NS_DISKCAP, else off; diskcap.py)

Synth campaigns (PI_CAMPAIGN_RESOURCE_GUARD=1) use the sandbox's own PSS guard on the main service instead of
memcap.py, at campaign_memory_limit_mb(): PI_CAMPAIGN_MEMORY_FACTOR / PI_CAMPAIGN_MEMORY_MIN_MB /
PI_CAMPAIGN_MEMORY_SPARE (synth/worker.py, from the campaign's harbor.sandbox_memory); unset: memory_mb x 1.0, 4096 MB
for a task without memory_mb, nothing spared (campaigns frozen before 2026-11-14).

Each trial (and each separate verifier) is a ``Pod``: one user namespace in
which every docker-compose service runs in its own sandbox, wired together by
a bridge.  Single-container tasks are a one-service pod.  The main service's
exec/upload/download go through ``nsenter`` and ``/proc/<pid>/root``.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy

from . import diskcap
from . import dockerfile as dfp
from . import memcap
from .builder import BuildError, ImageMeta, ImageStore, debianize_ref, resolve_in_root
from .sandbox import (PIP_CACHE_TRUST, Pod, RuntimeConfig, Sandbox, build_nsfake, find_runtime_bins, fsize_supported,
                      hosts_names, proxy_env)


def campaign_memory_limit_mb(memory_mb, env=None) -> int:
    """The synth PSS guard's budget (MB) for a main sandbox whose task declares memory_mb (None/0: undeclared).
    PI_CAMPAIGN_MEMORY_FACTOR (default 1.0) x memory_mb, raised to PI_CAMPAIGN_MEMORY_MIN_MB when that is set (and
    exactly it for an undeclared task); without PI_CAMPAIGN_MEMORY_MIN_MB an undeclared task gets 4096 MB (the rule of
    campaigns frozen before 2026-11-14)."""
    env = os.environ if env is None else env
    factor = float(env.get("PI_CAMPAIGN_MEMORY_FACTOR") or 1.0)
    if factor <= 0:
        raise ValueError(f"PI_CAMPAIGN_MEMORY_FACTOR must be > 0: {factor}")
    floor = int(float(env.get("PI_CAMPAIGN_MEMORY_MIN_MB") or 0))
    if floor:
        return int(max((memory_mb or 0) * factor, floor))
    return int(memory_mb * factor) if memory_mb else 4096


def campaign_memory_spare(env=None) -> tuple:
    """PI_CAMPAIGN_MEMORY_SPARE: comm names the synth PSS guard kills only when nothing else is left (comma list)."""
    env = os.environ if env is None else env
    return tuple(c for c in (env.get("PI_CAMPAIGN_MEMORY_SPARE") or "").split(",") if c)


def image_hosts_names(rootfs: Path) -> list[str]:
    """Host names in an image's /etc/hosts."""
    try:
        return hosts_names(resolve_in_root(Path(rootfs), "/etc/hosts").read_text(errors="replace"))
    except (OSError, BuildError):
        return []


def offline_bypass_names(rootfs: Path) -> list[str]:
    """The image's own /etc/hosts names, for an offline sandbox's no_proxy: a task's setup wrote them, the sandbox gets
    a fresh /etc/hosts (as in Docker), and the task may have the agent restore them. Registries the offline package proxy
    serves (HARBOR_NS_OFFLINE_PROXY_HOSTS) are left out: a build-time `127.0.0.1 pypi.org` must not cut pip off."""
    served = [h.strip().lower() for h in os.environ.get("HARBOR_NS_OFFLINE_PROXY_HOSTS", "").split(",") if h.strip()]
    return [n for n in image_hosts_names(rootfs)
            if not any(n.lower() == h or n.lower().endswith("." + h) for h in served)]


def service_proxy_env(hosts, rootfs: Path, internet: bool, env=None) -> dict[str, str]:
    """Proxy variables of a compose service (sandbox.proxy_env; `hosts`: the pod's service names).
    Services started without internet: HARBOR_NS_OFFLINE_PROXY_URL points their proxy variables at a proxy inside the
    sandbox (synth.package_proxy, package registries only); HARBOR_NS_OFFLINE_NO_PROXY=1 drops them (they would only
    point the agent at a cluster proxy it cannot reach). With HARBOR_NS_OFFLINE_BYPASS=1 (synth workers) or a package
    proxy, their no_proxy also covers private names and ranges and the image's /etc/hosts names (sandbox.OFFLINE_DOMAINS);
    otherwise (aa-evals, eval) it is the online no_proxy. HARBOR_NS_OFFLINE_PIP_INDEX: pip's index for an offline
    sandbox, served by its package proxy (synth campaign harbor.package_cache_pip; the proxy serves PyPI at /pypi/)."""
    env = os.environ if env is None else env
    offline_url = None if internet else env.get("HARBOR_NS_OFFLINE_PROXY_URL")
    offline_bare = not internet and not offline_url and env.get("HARBOR_NS_OFFLINE_NO_PROXY") == "1"
    offline = not internet and (bool(offline_url) or env.get("HARBOR_NS_OFFLINE_BYPASS") == "1")
    bypass = sorted(hosts)
    if offline:
        bypass += [n for n in offline_bypass_names(rootfs) if n not in bypass]
    px = {} if offline_bare else proxy_env(bypass or None, url=offline_url, offline=offline)
    if offline_url and env.get("HARBOR_NS_OFFLINE_PIP_INDEX"):
        px["PIP_INDEX_URL"] = env["HARBOR_NS_OFFLINE_PIP_INDEX"]
        px.update(PIP_CACHE_TRUST)
    return px


_STORES: dict[tuple[str, str], ImageStore] = {}
_CPU_COUNTER = itertools.count()


def _cpu_registry() -> Path:
    """Node-wide CPU pin registry of this user's harbor_ns processes (HARBOR_NS_CPU_VIEW).  Every Pi attempt is its
    own Harbor process, so a per-process count would pin all concurrent attempts on a worker to the same CPUs."""
    return Path(os.environ.get("HARBOR_NS_CPU_REGISTRY", f"/dev/shm/harbor_ns_cpus.{os.getuid()}.json"))


def _cpu_registry_update(fn):
    """Run fn(entries) -> result under an exclusive flock of the registry (local tmpfs: flock works); entries of
    dead processes are pruned first."""
    import fcntl  # noqa: PLC0415
    import json  # noqa: PLC0415

    reg = _cpu_registry()
    with open(str(reg) + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            entries = json.loads(reg.read_text()) if reg.exists() else {}
        except (OSError, ValueError):
            entries = {}
        alive = {}
        for tok, e in entries.items():
            try:
                os.kill(int(e["pid"]), 0)
                alive[tok] = e
            except (OSError, KeyError, ValueError):
                pass
        out = fn(alive)
        tmp = reg.with_name(reg.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(alive))
        os.replace(tmp, reg)
        return out


def _cpu_acquire(allowed: list[int], k: int) -> tuple[list[int], str]:
    tok = f"{os.getpid()}-{secrets.token_hex(4)}"

    def pick(entries: dict) -> list[int]:
        use: dict[int, int] = {}
        for e in entries.values():
            for c in e["cpus"]:
                use[c] = use.get(c, 0) + 1
        got = sorted(sorted(allowed, key=lambda c: (use.get(c, 0), c))[:k])
        entries[tok] = {"pid": os.getpid(), "cpus": got}
        return got

    return _cpu_registry_update(pick), tok


def _cpu_release(tok: str) -> None:
    _cpu_registry_update(lambda entries: entries.pop(tok, None))


def get_store(local_root: str | None, shared_root: str | None) -> ImageStore:
    local = local_root or os.environ.get("HARBOR_NS_LOCAL") or f"/tmp/{os.environ.get('USER', 'user')}/harbor_ns"
    shared = shared_root if shared_root is not None else os.environ.get("HARBOR_NS_SHARED", "")
    key = (local, shared)
    if key not in _STORES:
        lp = Path(local)
        bins = lp / "bin"
        rt = RuntimeConfig(nsinit=find_runtime_bins(bins), work_root=lp / "boxes",
                           nsfake=build_nsfake(bins))
        rt.work_root.mkdir(parents=True, exist_ok=True)
        _STORES[key] = ImageStore(local_root=lp, shared_root=Path(shared) if shared else None, rt=rt,
                                  nsfake=rt.nsfake)
    return _STORES[key]


# ----------------------------------------------------------------------------- compose model


@dataclass
class Service:
    name: str
    build_ctx: Path | None = None
    dockerfile: str = "Dockerfile"
    image: str | None = None
    environment: dict[str, str] = field(default_factory=dict)
    command: list[str] | None = None
    entrypoint: list[str] | None = None
    working_dir: str | None = None
    healthcheck: dict[str, Any] | None = None
    depends_on: dict[str, str] = field(default_factory=dict)  # name -> condition
    volumes: list[tuple[str, str, bool]] = field(default_factory=list)  # (src spec, target, ro)
    aliases: list[str] = field(default_factory=list)
    extra_hosts: dict[str, str] = field(default_factory=dict)
    # resolved at start
    rootfs: Path | None = None
    meta: ImageMeta | None = None
    box: Sandbox | None = None
    ip: str | None = None
    proc: Any = None  # the service's main process (ENTRYPOINT + CMD), if any


def _interp(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return dfp.substitute(value.replace("$$", "\x00"), env).replace("\x00", "$")
    if isinstance(value, list):
        return [_interp(v, env) for v in value]
    if isinstance(value, dict):
        return {k: _interp(v, env) for k, v in value.items()}
    return value


def _as_argv(v: Any) -> list[str] | None:
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v]
    return shlex.split(str(v))


def _as_env(v: Any) -> dict[str, str]:
    if not v:
        return {}
    if isinstance(v, dict):
        return {str(k): "" if x is None else str(x) for k, x in v.items()}
    out = {}
    for item in v:
        k, _, x = str(item).partition("=")
        out[k] = x
    return out


def _duration(v: Any, default: float) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    total, s = 0.0, str(v).strip()
    for num, unit in re.findall(r"([\d.]+)\s*(ms|us|h|m|s)?", s):
        mult = {"ms": 0.001, "us": 1e-6, "h": 3600, "m": 60, "s": 1, "": 1}[unit or ""]
        total += float(num) * mult
    return total or default


def _dockerfile_healthcheck(path: Path) -> dict | None:
    """The final stage's HEALTHCHECK, in compose `healthcheck:` form (None if absent)."""
    from .builder import parse_healthcheck  # noqa: PLC0415

    try:
        stages, _ = dfp.parse(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    hcs = [i for i in stages[-1].instructions if i.cmd == "HEALTHCHECK"] if stages else []
    return parse_healthcheck(hcs[-1].raw) if hcs else None


async def _shielded(coro):
    """Await an image build so that cancelling the trial (Harbor's environment-start
    timeout) does not kill it: the build finishes in the background and lands in the
    cache for the next trial of the same image instead of starting over each time."""
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        raise


# ----------------------------------------------------------------------------- environment


class NsEnvironment(BaseEnvironment):
    def __init__(
        self,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths,
        task_env_config,
        *args,
        ns_local_root: str | None = None,
        ns_shared_root: str | None = None,
        ns_cpu_affinity: bool | str = False,
        ns_start_timeout: float = 120.0,
        ns_multiuid: bool | str | None = None,
        ns_memcap: float | str | None = None,
        ns_diskcap: float | str | None = None,
        **kwargs,
    ):
        self._compose_file = next(
            (p for p in (environment_dir / "docker-compose.yaml", environment_dir / "docker-compose.yml") if p.exists()),
            None,
        )
        self._ns_local_root = ns_local_root
        self._ns_shared_root = ns_shared_root
        self._ns_cpu_affinity = str(ns_cpu_affinity).lower() in ("1", "true", "yes")
        # Each campaign trial runs in its own Harbor process. A process-local
        # counter alone would pin every trial to the same first host cores.
        self._campaign_cpu_offset = (int.from_bytes(hashlib.sha256(environment_name.encode()).digest()[:4],'big')
            if os.environ.get('PI_CAMPAIGN_RESOURCE_GUARD')=='1' else 0)
        self._ns_start_timeout = float(ns_start_timeout)
        if ns_multiuid is None:
            ns_multiuid = os.environ.get("HARBOR_NS_MULTIUID", "")
        self._ns_multiuid = str(ns_multiuid).lower() in ("1", "true", "yes")
        self._ns_memcap = memcap.parse(os.environ.get("HARBOR_NS_MEMCAP") if ns_memcap is None else ns_memcap)
        self._memcap: memcap.MemCap | None = None
        self._memcap_task: asyncio.Task | None = None
        self._ns_diskcap = diskcap.parse(os.environ.get("HARBOR_NS_DISKCAP") if ns_diskcap is None else ns_diskcap)
        self._diskcap_mb: float | None = None
        self._diskcap: diskcap.DiskCap | None = None
        self._diskcap_task: asyncio.Task | None = None
        self._pod: Pod | None = None
        self._services: dict[str, Service] = {}
        self._main: Service | None = None
        self._pod_dir: Path | None = None
        super().__init__(environment_dir, environment_name, session_id, trial_paths, task_env_config,
                         *args, **kwargs)

    # ---- identity / capabilities
    @staticmethod
    def type() -> str:
        return "harbor-ns"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            mounted=True,
            disable_internet=True,
            dynamic_network_policy=True,
            docker_compose=True,
            gpus=False,
        )

    @property
    def _uses_compose(self) -> bool:
        return self._compose_file is not None

    def _validate_definition(self):
        if self._compose_file is None and not (self.environment_dir / "Dockerfile").exists() \
                and not self.task_env_config.docker_image:
            raise FileNotFoundError(f"{self.environment_dir} has no Dockerfile, docker-compose.yaml or docker_image")

    # ---- service planning
    def _compose_vars(self) -> dict[str, str]:
        v = dict(os.environ)
        if self.task_env_config.env:
            from harbor.utils.env import resolve_env_vars  # noqa: PLC0415

            v.update(resolve_env_vars(self.task_env_config.env))
        v.update(self._persistent_env)
        v["CONTEXT_DIR"] = str(self.environment_dir.resolve())
        v["MAIN_IMAGE_NAME"] = f"hb__{self.environment_name}"
        if self.task_env_config.docker_image:
            v["PREBUILT_IMAGE_NAME"] = self.task_env_config.docker_image
        if self._effective_cpus:
            v["CPUS"] = str(self._effective_cpus)
        if self._effective_memory_mb:
            v["MEMORY"] = f"{self._effective_memory_mb}M"
        for m in self._mounts:
            tgt = m.get("target", "")
            for key, name in (("/logs/verifier", "VERIFIER_LOGS"), ("/logs/agent", "AGENT_LOGS"),
                              ("/logs/artifacts", "ARTIFACTS")):
                if tgt == key:
                    v[f"HOST_{name}_PATH"] = m.get("source", "")
                    v[f"ENV_{name}_PATH"] = tgt
        return v

    def _plan_services(self) -> dict[str, Service]:
        main_default = Service(name="main", command=["sh", "-c", "sleep infinity"])
        # Harbor's Docker env (should_use_prebuilt_docker_image): a task's prebuilt
        # `docker_image` is used unless --force-build, even when a Dockerfile exists.
        # Rebuilding from the Dockerfile picks up upstream drift (apt/pip/python updates,
        # deleted repos); start() falls back to the Dockerfile if the image can't be pulled.
        if self.task_env_config.docker_image and (
                not getattr(self, "_force_build", False) or not (self.environment_dir / "Dockerfile").exists()):
            main_default.image = self.task_env_config.docker_image
        else:
            main_default.build_ctx = self.environment_dir
        if self._compose_file is None:
            return {"main": main_default}
        data = yaml.safe_load(self._compose_file.read_text()) or {}
        data = _interp(data, self._compose_vars())
        services: dict[str, Service] = {}
        for name, spec in (data.get("services") or {}).items():
            spec = spec or {}
            s = Service(name=name) if name != "main" else Service(name="main", build_ctx=main_default.build_ctx,
                                                                  image=main_default.image,
                                                                  command=main_default.command)
            b = spec.get("build")
            if b is not None:
                if isinstance(b, str):
                    ctx, df = b, "Dockerfile"
                else:
                    ctx, df = b.get("context", "."), b.get("dockerfile", "Dockerfile")
                cp = Path(ctx)
                s.build_ctx = (cp if cp.is_absolute() else (self.environment_dir / cp)).resolve()
                s.dockerfile = df
                s.image = None
            elif spec.get("image") and name != "main":
                if spec["image"] == f"hb__{self.environment_name}" and main_default.build_ctx is not None:
                    # `image: ${MAIN_IMAGE_NAME}` (from tbsci-lite): Docker reuses the image built for main, there
                    # is nothing to pull (bosch-drie-control's 4-service verifier)
                    s.build_ctx = main_default.build_ctx
                    s.image = None
                else:
                    s.image = spec["image"]
            if "command" in spec:
                s.command = _as_argv(spec["command"])
            if "entrypoint" in spec:
                s.entrypoint = _as_argv(spec["entrypoint"]) or []
            s.environment = _as_env(spec.get("environment"))
            for ef in ([spec["env_file"]] if isinstance(spec.get("env_file"), str) else spec.get("env_file") or []):
                p = Path(ef if isinstance(ef, str) else ef.get("path", ""))
                p = p if p.is_absolute() else self.environment_dir / p
                if p.exists():
                    for ln in p.read_text().splitlines():
                        if ln.strip() and not ln.lstrip().startswith("#") and "=" in ln:
                            k, _, x = ln.partition("=")
                            s.environment.setdefault(k.strip(), x.strip().strip("'\""))
            s.working_dir = spec.get("working_dir")
            hc = spec.get("healthcheck")
            if hc and not hc.get("disable"):
                s.healthcheck = hc
            dep = spec.get("depends_on") or {}
            if isinstance(dep, list):
                s.depends_on = {d: "service_started" for d in dep}
            else:
                s.depends_on = {d: (c or {}).get("condition", "service_started") for d, c in dep.items()}
            for vol in spec.get("volumes") or []:
                if isinstance(vol, str):
                    parts = vol.split(":")
                    if len(parts) == 1:
                        continue  # anonymous volume
                    src, tgt = parts[0], parts[1]
                    ro = len(parts) > 2 and "ro" in parts[2].split(",")
                else:
                    src, tgt, ro = vol.get("source", ""), vol.get("target", ""), bool(vol.get("read_only"))
                if src and tgt:
                    s.volumes.append((src, tgt, ro))
            aliases = [spec.get("container_name"), spec.get("hostname")]
            nets = spec.get("networks")
            if isinstance(nets, dict):
                for n in nets.values():
                    aliases += (n or {}).get("aliases", []) if isinstance(n, dict) else []
            s.aliases = [a for a in aliases if a and isinstance(a, str) and a != name]
            for eh in spec.get("extra_hosts") or []:
                h, _, ip = str(eh).partition(":")
                if h and ip and ip != "host-gateway":
                    s.extra_hosts[h] = ip
            services[name] = s
        if "main" not in services:
            services["main"] = main_default
        return services

    # ---- lifecycle
    async def start(self, force_build: bool) -> None:
        store = get_store(self._ns_local_root, self._ns_shared_root)
        self._force_build = bool(force_build) or os.environ.get("HARBOR_NS_FORCE_BUILD") == "1"
        self._services = self._plan_services()
        self._main = self._services["main"]
        self._diskcap_mb = diskcap.limit_mb(self.task_env_config.storage_mb, self._ns_diskcap)
        label = f"{self.environment_name}:{self.environment_dir.name}"
        # Build / pull every service image (the store dedupes and locks per image), each pinned against eviction
        # (ImageStore.claim) as soon as it is there, for the lifetime of this trial.
        self._inuse_token = f"{os.getpid()}-{secrets.token_hex(6)}"

        async def image(s: Service, tmux: bool) -> None:
            if s.build_ctx is not None:
                _, s.rootfs, s.meta = await _shielded(store.build(
                    s.build_ctx, s.dockerfile, label=f"{label}:{s.name}", ensure_tmux=tmux))
            elif s.image:
                # Build `FROM <image>` so image-only services also get the compat shims
                # (nsfake preload, gosu shim): their entrypoints chown/gosu as root.
                ctxd = store.local_root / "svc-ctx" / re.sub(r"[^A-Za-z0-9_.-]", "_", s.image)
                ctxd.mkdir(parents=True, exist_ok=True)
                df = ctxd / "Dockerfile"
                if not df.exists():  # concurrent trials: write atomically
                    tmpf = ctxd / f".Dockerfile.{os.getpid()}.{secrets.token_hex(3)}"
                    tmpf.write_text(f"FROM {s.image}\n")
                    os.replace(tmpf, df)
                try:
                    _, s.rootfs, s.meta = await _shielded(store.build(
                        ctxd, "Dockerfile", label=f"{label}:{s.name}", ensure_tmux=tmux))
                except BuildError:
                    # Prebuilt task image unavailable (deleted tag, registry auth): build the
                    # task's own Dockerfile instead.
                    if not (s.name == "main" and s.image == self.task_env_config.docker_image
                            and (self.environment_dir / "Dockerfile").exists()):
                        raise
                    s.image, s.build_ctx = None, self.environment_dir
                    _, s.rootfs, s.meta = await _shielded(store.build(
                        s.build_ctx, s.dockerfile, label=f"{label}:{s.name}", ensure_tmux=tmux))
            else:
                raise BuildError(f"service {s.name} has neither build nor image")

        for s in self._services.values():
            # The agent's tmux lives in `main`; verifier images don't need it.
            tmux = s.name == "main" and "__verifier__" not in self.session_id
            for _ in range(3):   # evicted between build and pin (another trial's eviction): load or build again
                await image(s, tmux)
                # claim may evict (a marker that cannot be written for lack of space): off the event loop
                if await asyncio.get_running_loop().run_in_executor(
                        None, store.claim, s.rootfs.parent, self._inuse_token):
                    break
            else:
                raise BuildError(f"service {s.name}: image {s.rootfs.parent.name} could not be pinned "
                                 f"(evicted each time, or the scratch is full)")
            if s.healthcheck is None and s.build_ctx is not None:
                # Docker gates `depends_on: service_healthy` on the image's HEALTHCHECK when
                # compose sets none (parsed from the Dockerfile: cached image metas predate it).
                hc = getattr(s.meta, "healthcheck", None) or _dockerfile_healthcheck(s.build_ctx / s.dockerfile)
                if hc and hc.get("test") != ["NONE"]:
                    s.healthcheck = hc
        # Every trial start, with its own images pinned (#451 lesson 7: cache hits never evicted, so a worker whose
        # tasks were all built filled up): one statvfs unless the scratch is low; never fails the trial.
        try:
            await asyncio.get_running_loop().run_in_executor(None, store.maybe_evict)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"harbor_ns: eviction at trial start failed: {e!r}")
        multi = len(self._services) > 1
        self._pod = Pod(bridge=multi, multiuid=True) if self._ns_multiuid else Pod(bridge=multi)
        await self._pod.start()
        self._pod_dir = Path(tempfile.mkdtemp(prefix="pod-", dir=store.rt.work_root))
        names = list(self._services)
        hosts: dict[str, str] = {}
        if multi:
            for i, n in enumerate(names):
                s = self._services[n]
                s.ip = self._pod.ip(i)
                for h in [n, *s.aliases]:
                    hosts[h] = s.ip
        self._seed_named_volumes()
        cpus = self._cpu_set()
        internet = self.network_policy.network_mode != NetworkMode.NO_NETWORK
        healthy: set[str] = set()
        order = self._topo_order()
        try:
            for n in order:
                s = self._services[n]
                for dep, cond in s.depends_on.items():
                    if dep not in self._services:
                        continue
                    if cond == "service_healthy" and dep not in healthy:
                        await self._wait_healthy(self._services[dep])
                        healthy.add(dep)
                    elif cond == "service_completed_successfully":
                        p = self._services[dep].proc
                        if p is not None:
                            rc = await asyncio.wait_for(p.wait(), self._ns_start_timeout * 10)
                            if rc != 0:
                                raise RuntimeError(f"service {dep} exited with {rc}")
                await self._start_service(s, hosts, cpus, internet)
            self._start_diskcap()
        except BaseException:
            await self.stop(delete=True)
            raise
        # Docker creates the bind-mount targets; make sure /logs/* exist & are writable.
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    def _seed_named_volumes(self) -> None:
        """compose creates every container before starting any, and creating a container
        copies the image's content at the mount target into an empty named volume."""
        assert self._pod_dir is not None
        for n in self._topo_order():
            s = self._services[n]
            for src, tgt, _ro in s.volumes:
                if src.startswith((".", "/")) or s.rootfs is None:
                    continue
                vol = self._pod_dir / "volumes" / re.sub(r"[^A-Za-z0-9_.-]", "_", src)
                vol.mkdir(parents=True, exist_ok=True)
                img = Path(s.rootfs) / tgt.lstrip("/")
                if any(vol.iterdir()) or img.is_symlink() or not img.is_dir():
                    continue
                subprocess.run(["cp", "-a", "--reflink=auto", f"{img}/.", str(vol)], check=False)

    def _topo_order(self) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str, stack: tuple[str, ...] = ()) -> None:
            if n in seen or n in stack or n not in self._services:
                return
            for d in self._services[n].depends_on:
                visit(d, (*stack, n))
            seen.add(n)
            order.append(n)

        for n in self._services:
            if n != "main":
                visit(n)
        visit("main")
        return order

    def _cpu_set(self) -> list[int] | None:
        # Multi-uid + HARBOR_NS_CPU_VIEW=1: pin every box to its task's CPU count and show only those CPUs to
        # glibc get_nprocs / os.cpu_count (nsdev.so), like a small Docker host; otherwise code sizes its thread
        # pools to the whole node (RLIMIT_NPROC aborts) and `taskset -c 0-3` fails on a cpuset without CPUs 0-3
        # (the preload remaps it).  Single-uid sandboxes are unchanged.
        view = self._ns_multiuid and os.environ.get("HARBOR_NS_CPU_VIEW") == "1"
        if not (self._ns_cpu_affinity or view):
            return None
        allowed = sorted(os.sched_getaffinity(0))
        k = max(1, min(int(self._effective_cpus or 1), len(allowed)))
        if view and not self._effective_cpus:
            # No `cpus` (e.g. a verifier environment): Docker sets no limit, so upstream these run with the CI
            # runner's CPUs (4 on GitHub's standard runners), not 1 and not a 64+-core node.
            k = max(1, min(int(os.environ.get("HARBOR_NS_CPU_DEFAULT", "4")), len(allowed)))
        if view:  # least-used CPUs of this user's harbor_ns processes on the node, released in stop()
            try:
                pick, tok = _cpu_acquire(allowed, k)
            except OSError:  # no usable registry: fall back to a per-process round robin
                start = (next(_CPU_COUNTER) + self._campaign_cpu_offset) * k
                return [allowed[(start + i) % len(allowed)] for i in range(k)]
            self._cpu_tokens = [*getattr(self, "_cpu_tokens", []), tok]
            return pick
        start = (next(_CPU_COUNTER) + self._campaign_cpu_offset) * k
        return [allowed[(start + i) % len(allowed)] for i in range(k)]

    async def _start_service(self, s: Service, hosts: dict[str, str], cpus, internet: bool) -> None:
        assert self._pod is not None and self._pod_dir is not None and s.meta is not None
        binds: list[tuple[str, str]] = []
        if s.name == "main":
            for m in self._mounts:
                src, tgt = m.get("source"), m.get("target")
                if m.get("type", "bind") == "bind" and src and tgt:
                    Path(src).mkdir(parents=True, exist_ok=True)
                    if self._ns_multiuid and tgt.startswith("/logs/"):
                        # Docker's bind mounts of the trial dirs are world-writable for the task's agent and
                        # verifier users (e.g. 65532:65532); here the dir belongs to the caller (= namespace root),
                        # so other uids need the mode bits.  Teardown chowns them back (normalize_owner).
                        os.chmod(src, 0o777)
                    binds.append((src, tgt))
        for src, tgt, _ro in s.volumes:
            if src.startswith((".", "/")):
                hp = Path(src) if src.startswith("/") else (self.environment_dir / src)
                hp = hp.resolve()
                if not hp.exists():
                    continue
                if not (hp.is_dir() or hp.is_file()):
                    # e.g. /var/run/docker.sock: no daemon to share (and copying it raises EACCES).
                    continue
                # Private copy: several rollouts of the same task run concurrently.
                dst = self._pod_dir / "binds" / f"{s.name}-{len(binds)}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if hp.is_dir():
                    shutil.copytree(hp, dst, symlinks=True)
                else:
                    shutil.copy2(hp, dst)
                binds.append((str(dst), tgt))
            else:
                vol = self._pod_dir / "volumes" / re.sub(r"[^A-Za-z0-9_.-]", "_", src)
                vol.mkdir(parents=True, exist_ok=True)
                binds.append((str(vol), tgt))
        binds.sort(key=lambda b: len(Path(b[1]).parts))  # Docker mounts parents before nested targets
        px = service_proxy_env(hosts, s.rootfs, internet)
        for k in ("MAVEN_OPTS", "GRADLE_OPTS"):
            if k in px and (k in s.meta.env or k in s.environment):
                px[k] = f"{s.environment.get(k, s.meta.env.get(k, ''))} {px[k]}".strip()
        env = {**s.meta.env, **s.environment, **px}
        env.setdefault("HOME", "/root")
        hostname = s.name if len(self._services) > 1 else secrets.token_hex(6)
        env.setdefault("HOSTNAME", hostname)
        # Docker semantics: process = ENTRYPOINT + CMD; compose `command` replaces CMD,
        # `entrypoint` replaces ENTRYPOINT (and resets CMD).  For `main`, Harbor's
        # base compose sets command `sh -c "sleep infinity"`.
        ep = s.entrypoint if s.entrypoint is not None else (s.meta.entrypoint or [])
        cmd = s.command if s.command is not None else (None if s.entrypoint is not None else s.meta.cmd)
        proc = [*ep, *(cmd or [])]
        if proc == ["sh", "-c", "sleep infinity"]:
            proc = []  # nsinit already keeps the box alive
        box = Sandbox(
            cfg=get_store(self._ns_local_root, self._ns_shared_root).rt,
            lower=[s.rootfs],
            hostname=hostname,
            binds=binds,
            private_net=True,
            cpus=cpus,
            base_env=env,
            pod=self._pod,
            cpu_view=bool(cpus) and self._ns_multiuid and os.environ.get("HARBOR_NS_CPU_VIEW") == "1",
            extra_hosts={**hosts, **s.extra_hosts},
            internet=internet,
            fsize=diskcap.fsize_bytes(self._diskcap_mb),
        )
        if os.environ.get('PI_CAMPAIGN_RESOURCE_GUARD')=='1' and s.name=='main':
            box.memory_limit_mb=campaign_memory_limit_mb(self._effective_memory_mb)
            box.memory_spare=campaign_memory_spare()
            log_dir=next((Path(m['source']) for m in self._mounts if m.get('target')=='/logs/agent'),None)
            if log_dir is not None:box.memory_event_log=log_dir/'sandbox-memory-events.jsonl'
        await box.start(timeout=self._ns_start_timeout)
        s.box = box
        self._fix_eol_apt_sources(box)
        if s.ip:
            await self._pod.attach(box, s.ip)
        if proc and self._ns_multiuid:
            # Docker runs the entrypoint as the image USER (HOME from its passwd entry unless set).
            ids, home = self._image_user(box, s.meta.user)
            s.proc = await box.spawn_background(proc, cwd=s.working_dir or s.meta.workdir or "/",
                                                env=None if home is None else {"HOME": home},
                                                ids=ids if ids != (0, 0) else None)
        elif proc:
            s.proc = await box.spawn_background(proc, cwd=s.working_dir or s.meta.workdir or "/")
        if s.name == "main":
            self._start_memcap(box)

    def _start_memcap(self, box: Sandbox) -> None:
        mb = memcap.limit_mb(self.task_env_config.memory_mb, self._ns_memcap)
        if not mb or box.init_pid is None:
            return
        self._memcap = memcap.MemCap(box.init_pid, mb, log=self.logger.warning)
        self._memcap_task = asyncio.create_task(self._memcap.run())
        self.logger.info(f"harbor_ns memcap: main capped at {mb:.0f} MB")

    def _start_diskcap(self) -> None:
        """The disk cap's watchdog over every service box of the trial (the per-file RLIMIT_FSIZE is set at launch).
        Multi-uid pods walk and sample writers as root of the pod's user namespace (diskcap.NsView), so private
        directories and processes of subordinate ids are seen."""
        mb = self._diskcap_mb
        if not mb:
            return

        def roots():
            out = [(n, s.box.dir / "upper") for n, s in self._services.items() if s.box is not None and s.box._dir]
            if self._pod_dir is not None:
                out.append(("pod", self._pod_dir))   # compose named volumes and private bind copies
            return out

        def inits():
            return [s.box.init_pid for s in self._services.values() if s.box is not None and s.box.init_pid]

        def enter():
            return self._pod.enter_argv() if self._pod is not None else []

        ns = diskcap.NsView(enter, log=self.logger.warning) if self._ns_multiuid else None
        self._diskcap = diskcap.DiskCap(roots, mb, inits=inits, fsize=diskcap.fsize_bytes(mb), log=self.logger.warning,
                                        ns=ns)
        self._diskcap_task = asyncio.create_task(self._diskcap.run())
        self.logger.info(f"harbor_ns diskcap: sandbox capped at {mb:.0f} MB in total and per file")
        if not fsize_supported():
            self.logger.warning("harbor_ns diskcap: prlimit not found; no per-file limit, only the watchdog")

    def _diskcap_stop(self) -> None:
        """Final walk and summary (HARBOR_NS_DISKCAP_LOG) of the disk cap; best effort, never fails the trial."""
        cap = self._diskcap
        if cap is None:
            return
        try:
            cap.final()
        except Exception:  # noqa: BLE001
            pass
        if cap.kills:
            self.logger.warning(f"harbor_ns diskcap: {len(cap.kills)} kill(s) in this sandbox")
        out = os.environ.get("HARBOR_NS_DISKCAP_LOG")
        if not out:
            return
        try:
            p = Path(out)
            if p.is_dir():   # many trials in one Harbor process (e.g. verifier_toolchain.py validate)
                p = p / f"{self.session_id}.json"
            elif "__verifier__" in self.session_id:   # a separate verifier environment of the same trial
                p = p.with_name(f"{p.stem}.verifier{p.suffix}")
            p.write_text(json.dumps({**cap.summary(), "session": self.session_id, "multiuid": self._ns_multiuid}))
        except Exception:  # noqa: BLE001
            pass

    # Debian releases whose security suite left deb.debian.org (packages now only on
    # archive.debian.org; the dists/ index is still served, so `apt-get install` 404s).
    _EOL_SECURITY = ("bullseye",)

    def _fix_eol_apt_sources(self, box: Sandbox) -> None:
        """Point EOL Debian security suites at archive.debian.org: mirror drift since the
        task's image was built (e.g. TB2.1 qemu verifiers `apt-get install curl` at test time)."""
        root = Path(f"/proc/{box.init_pid}/root")
        try:
            osr = resolve_in_root(root, "/etc/os-release")
            text = osr.read_text(errors="replace") if osr.is_file() else ""
            m = re.search(r"^VERSION_CODENAME=\"?([a-z]+)", text, re.M)
            if not m or m.group(1) not in self._EOL_SECURITY or not re.search(r"^ID=\"?debian", text, re.M):
                return
            files = [resolve_in_root(root, "/etc/apt/sources.list")]
            d = resolve_in_root(root, "/etc/apt/sources.list.d")
            if d.is_dir():
                files += [f for f in d.iterdir() if f.suffix in (".list", ".sources") and not f.is_symlink()]
            for f in files:
                if not f.is_file():
                    continue
                t = f.read_text(errors="replace")
                n = re.sub(r"https?://(deb|security)\.debian\.org/debian-security",
                           "http://archive.debian.org/debian-security", t)
                if n != t:
                    f.write_text(n)
        except OSError:
            pass

    async def _wait_healthy(self, s: Service) -> None:
        hc = s.healthcheck
        if not hc or s.box is None:
            await asyncio.sleep(2)
            return
        test = hc.get("test")
        if isinstance(test, list):
            if test and test[0] == "NONE":
                return
            if test and test[0] == "CMD-SHELL":
                cmd = " ".join(test[1:])
            elif test and test[0] == "CMD":
                cmd = shlex.join(test[1:])
            else:
                cmd = shlex.join(test)
        else:
            cmd = str(test)
        interval = _duration(hc.get("interval"), 5.0)
        timeout = _duration(hc.get("timeout"), 30.0)
        retries = int(hc.get("retries", 3))
        start_period = _duration(hc.get("start_period"), 0.0)
        deadline = time.monotonic() + start_period + max(retries, 1) * (interval + timeout) + 60
        await asyncio.sleep(min(1.0, interval))
        while time.monotonic() < deadline:
            r = await s.box.exec(cmd, cwd=s.working_dir or s.meta.workdir, timeout=timeout, shell="sh")
            if r.return_code == 0:
                return
            await asyncio.sleep(interval)
        tail = ""
        try:  # the service's own output (bg-*.log in its box dir) is gone after stop(); surface it
            logs = sorted(s.box.dir.glob("bg-*.log"), key=lambda p: p.stat().st_mtime)
            if logs:
                tail = "\n--- service output (tail) ---\n" + logs[-1].read_text(errors="replace")[-3000:]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"service {s.name} never became healthy ({cmd}){tail}")

    async def stop(self, delete: bool) -> None:
        for tok in getattr(self, "_cpu_tokens", None) or []:
            try:
                _cpu_release(tok)
            except OSError:
                pass
        self._cpu_tokens = []
        if self._memcap_task is not None:
            self._memcap_task.cancel()
            self._memcap_task = None
            if self._memcap is not None and self._memcap.kills:
                self.logger.warning(f"harbor_ns memcap: {len(self._memcap.kills)} kill(s) in this sandbox")
            out = os.environ.get("HARBOR_NS_MEMCAP_LOG")
            if self._memcap is not None and out:
                try:   # best effort: the summary must never fail the trial
                    Path(out).write_text(json.dumps(self._memcap.summary()))
                except Exception:  # noqa: BLE001
                    pass
        if self._diskcap_task is not None:
            self._diskcap_task.cancel()
            self._diskcap_task = None
            await asyncio.get_running_loop().run_in_executor(None, self._diskcap_stop)
            self._diskcap = None
        for s in reversed(list(self._services.values())):
            if s.box is not None:
                await s.box.stop(delete=delete)
                s.box = None
        if self._pod is not None and self._pod.multiuid and self._pod.holder is not None:
            # Before the namespace goes: no file of a subordinate id may outlive the trial (the
            # trial's bind-mounted log dirs are chowned back; pod volumes/copies are removed).
            srcs = [Path(m["source"]) for m in self._mounts
                    if m.get("type", "bind") == "bind" and m.get("source")]
            try:
                await self._pod.normalize_owner(srcs)
                if self._pod_dir is not None:
                    await (self._pod.purge if delete else self._pod.normalize_owner)([self._pod_dir])
            except RuntimeError as e:
                self.logger.warning(f"harbor_ns multi-uid teardown: {e}")
        token = getattr(self, "_inuse_token", None)
        if token:
            store = get_store(self._ns_local_root, self._ns_shared_root)
            for s in self._services.values():
                if s.rootfs is not None:
                    store.unmark_in_use(s.rootfs.parent, token)
        if self._pod is not None:
            await self._pod.stop()
            self._pod = None
        if self._pod_dir is not None and delete:
            from .sandbox import _rmtree_force  # noqa: PLC0415

            await asyncio.get_running_loop().run_in_executor(None, _rmtree_force, self._pod_dir)
            self._pod_dir = None

    # ---- network policy
    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        on = network_policy.network_mode != NetworkMode.NO_NETWORK
        for s in self._services.values():
            if s.box is not None:
                await s.box.set_internet(on)

    # ---- exec & files
    @property
    def _box(self) -> Sandbox:
        if self._main is None or self._main.box is None:
            raise RuntimeError("environment not started")
        return self._main.box

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        assert self._main is not None and self._main.meta is not None
        # Docker semantics: exec runs as `user`, else the orchestrator's default user (the
        # task's agent/verifier user), else the image USER.  Only uid 0 exists in the
        # sandbox, so a non-root identity is emulated by nsfake (NSFAKE_ID).
        u = self._resolve_user(user)
        if u is None:
            u = self._main.meta.user
        if self._ns_multiuid:
            # Real ids: enter as the user (nsenter -S/-G) instead of nsfake's NSFAKE_ID.
            ids = self._user_ids(u)
            r = await self._box.exec(
                command,
                cwd=cwd or self._main.working_dir or self._main.meta.workdir or "/",
                env={**self._user_env(u), **(self._merge_env(env) or {})},
                timeout=timeout_sec,
                ids=ids if ids != (0, 0) else None,
            )
            return ExecResult(stdout=r.stdout, stderr=r.stderr, return_code=r.return_code)
        r = await self._box.exec(
            command,
            cwd=cwd or self._main.working_dir or self._main.meta.workdir or "/",
            env={**self._user_env(u), **(self._merge_env(env) or {})},
            timeout=timeout_sec,
        )
        return ExecResult(stdout=r.stdout, stderr=r.stderr, return_code=r.return_code)

    @staticmethod
    def _image_user(box: Sandbox, user: str | int | None) -> tuple[tuple[int, int], str | None]:
        """((uid, gid), HOME override or None) of a service's image USER, read from its own rootfs.
        HOME comes from the passwd entry only when neither the image nor compose set it."""
        if user is None or str(user) in ("", "root", "0", "0:0"):
            return (0, 0), None
        from .builder import _resolve_user  # noqa: PLC0415

        try:
            ident = _resolve_user(Path(f"/proc/{box.init_pid}/root"), str(user))
        except Exception:  # noqa: BLE001
            return (0, 0), None
        if not ident:
            return (0, 0), None
        uid, gid, home = ident
        if box.base_env.get("HOME") not in (None, "/root"):
            home = None  # set by the image or compose (env.setdefault put /root only when unset)
        return (uid, gid), home

    def _user_ids(self, user: str | int | None) -> tuple[int, int]:
        """(uid, gid) of `user` in the main image (root if unknown), for multi-uid exec."""
        if user is None or str(user) in ("", "root", "0", "0:0"):
            return 0, 0
        from .builder import _resolve_user  # noqa: PLC0415

        try:
            ident = _resolve_user(self._root(), str(user))
        except Exception:  # noqa: BLE001
            return 0, 0
        return (ident[0], ident[1]) if ident else (0, 0)

    def _user_env(self, user: str | int | None) -> dict[str, str]:
        """Env for exec'ing as `user`: NSFAKE_ID, and HOME from the user's passwd entry
        (like `docker exec -u`) unless the image or compose env sets HOME."""
        if user is None or str(user) in ("", "root", "0", "0:0"):
            return {}
        from .builder import _resolve_user  # noqa: PLC0415

        try:
            ident = _resolve_user(self._root(), str(user))
        except Exception:  # noqa: BLE001  (unknown user: Docker would refuse; keep root)
            return {}
        if ident is None:
            return {}
        uid, gid, home = ident
        out: dict[str, str] = {}
        if uid or gid:
            out["NSFAKE_ID"] = f"{uid}:{uid}:{uid}:{gid}:{gid}:{gid}"
        m = self._main
        if not (m and ("HOME" in (m.meta.env or {}) or "HOME" in (m.environment or {}))):
            out["HOME"] = home or "/"
        return out

    def _root(self) -> Path:
        return Path(f"/proc/{self._box.init_pid}/root")

    def _host_path(self, path: str, follow_last: bool = True) -> Path:
        return resolve_in_root(self._root(), path, follow_last=follow_last)

    async def _mkdirs(self, path: str) -> Path:
        r = await self._box.exec(f"mkdir -p {shlex.quote(path)}", timeout=60, shell="sh")
        if r.return_code != 0:
            raise RuntimeError(f"mkdir {path} failed: {r.stderr}")
        return self._host_path(path)

    def _stage_dir(self) -> Path:
        """Host dir of the trial's /logs/agent bind mount (world-writable in multi-uid mode): where a copy that
        the host user cannot do (files of subordinate uids with restrictive modes) is staged."""
        for m in self._mounts:
            if m.get("type", "bind") == "bind" and m.get("target") == "/logs/agent" and m.get("source"):
                return Path(m["source"])
        raise PermissionError("no /logs/agent bind mount to stage through")

    async def _ns_stage_in(self, src: Path, target: str, is_dir: bool = False) -> None:
        """Multi-uid fallback of upload_*: Docker's `cp` runs as root; the host user is only namespace root's
        owner and cannot traverse e.g. a 0700 directory made by the agent user.  Copy through the bind mount
        and let namespace root place it."""
        import uuid  # noqa: PLC0415

        tag = f".harbor-ns-stage-{uuid.uuid4().hex[:12]}"
        host = self._stage_dir() / tag
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, (shutil.copytree if is_dir else shutil.copy2), src, host)
            q = shlex.quote
            if is_dir:
                cmd = f"mkdir -p {q(target)} && cp -a /logs/agent/{tag}/. {q(target)}/"
            else:
                cmd = (f"mkdir -p {q(os.path.dirname(target) or '/')} && "
                       f"if [ -d {q(target)} ]; then cp -f /logs/agent/{tag} {q(target)}/{q(src.name)}; "
                       f"else cp -f /logs/agent/{tag} {q(target)}; fi")
            r = await self._box.exec(cmd, timeout=600, shell="sh")
            if r.return_code != 0:
                raise RuntimeError(f"upload to {target} failed: {r.stderr}")
        finally:
            await self._box.exec(f"rm -rf /logs/agent/{tag}", timeout=60, shell="sh")

    async def _ns_stage_out(self, source: str, target: Path, is_dir: bool = False) -> None:
        """Multi-uid fallback of download_*: namespace root copies into the bind mount, the host user reads it."""
        import uuid  # noqa: PLC0415

        tag = f".harbor-ns-stage-{uuid.uuid4().hex[:12]}"
        host = self._stage_dir() / tag
        q = shlex.quote
        try:
            r = await self._box.exec(f"cp -a {q(source)} /logs/agent/{tag} && chmod -R a+rX /logs/agent/{tag}",
                                     timeout=600, shell="sh")
            if r.return_code != 0:
                raise RuntimeError(f"download of {source} failed: {r.stderr}")
            tgt = Path(target)

            def _copy():
                tgt.parent.mkdir(parents=True, exist_ok=True)
                if is_dir:
                    shutil.copytree(host, tgt, dirs_exist_ok=True)
                else:
                    shutil.copy2(host, tgt)

            await asyncio.get_running_loop().run_in_executor(None, _copy)
        finally:
            await self._box.exec(f"rm -rf /logs/agent/{tag}", timeout=60, shell="sh")

    async def upload_file(self, source_path: Path | str, target_path: str):
        try:
            await self._upload_file_host(source_path, target_path)
        except PermissionError:
            if not self._ns_multiuid:
                raise
            await self._ns_stage_in(Path(source_path), target_path)

    async def _upload_file_host(self, source_path: Path | str, target_path: str):
        await self._mkdirs(os.path.dirname(target_path) or "/")
        dst = self._host_path(target_path)
        if dst.is_dir():
            dst = dst / Path(source_path).name
        await asyncio.get_running_loop().run_in_executor(None, shutil.copy2, source_path, dst)

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        try:
            await self._upload_dir_host(source_dir, target_dir)
        except PermissionError:
            if not self._ns_multiuid:
                raise
            await self._ns_stage_in(Path(source_dir), target_dir, is_dir=True)

    async def _upload_dir_host(self, source_dir: Path | str, target_dir: str):
        await self._mkdirs(target_dir)
        loop = asyncio.get_running_loop()
        src = Path(source_dir)

        def _copy():
            for root, dirs, files in os.walk(src):
                rel = Path(root).relative_to(src)
                base = target_dir.rstrip("/") + ("/" + rel.as_posix() if str(rel) != "." else "")
                d = resolve_in_root(self._root(), base)
                d.mkdir(parents=True, exist_ok=True)
                for f in files:
                    sp = Path(root) / f
                    dp = resolve_in_root(self._root(), base + "/" + f, follow_last=False)
                    if dp.is_symlink() or dp.exists():
                        dp.unlink()
                    if sp.is_symlink():
                        os.symlink(os.readlink(sp), dp)
                    else:
                        shutil.copy2(sp, dp)

        await loop.run_in_executor(None, _copy)

    async def download_file(self, source_path: str, target_path: Path | str):
        try:
            await self._download_file_host(source_path, target_path)
        except PermissionError:
            if not self._ns_multiuid:
                raise
            await self._ns_stage_out(source_path, Path(target_path))

    async def _download_file_host(self, source_path: str, target_path: Path | str):
        src = self._host_path(source_path)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_running_loop().run_in_executor(None, shutil.copy2, src, target_path)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        try:
            await self._download_dir_host(source_dir, target_dir)
        except PermissionError:
            if not self._ns_multiuid:
                raise
            await self._ns_stage_out(source_dir, Path(target_dir), is_dir=True)

    async def _download_dir_host(self, source_dir: str, target_dir: Path | str):
        root = self._root()
        src = resolve_in_root(root, source_dir)
        tgt = Path(target_dir)

        def _copy():
            tgt.mkdir(parents=True, exist_ok=True)
            for r, dirs, files in os.walk(src):
                rel = Path(r).relative_to(src)
                (tgt / rel).mkdir(parents=True, exist_ok=True)
                for f in files:
                    sp = Path(r) / f
                    dp = tgt / rel / f
                    if dp.is_symlink() or dp.exists():
                        dp.unlink()
                    if sp.is_symlink():
                        os.symlink(os.readlink(sp), dp)
                    elif sp.is_file():
                        shutil.copy2(sp, dp)

        await asyncio.get_running_loop().run_in_executor(None, _copy)
