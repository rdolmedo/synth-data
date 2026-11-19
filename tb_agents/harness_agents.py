"""Harbor's claude-code, opencode and pi agents with the pinned CLI installed from a local
cache instead of downloaded inside every sandbox.

  -a tb_agents.harness_agents:ClaudeCodeCached --ak version=2.1.207
  -a tb_agents.harness_agents:OpenCodeCached   --ak version=1.18.35
  -a tb_agents.harness_agents:PiCached         --ak version=1.2.0

Stock install per trial: apt-get nodejs/npm (+curl, procps), then for Claude Code the
bootstrap script (downloads the latest native binary, ~260 MB, which then downloads the
requested version, another ~260 MB), for OpenCode nvm + Node 22 + `npm i -g opencode-ai`.
Here the task image only gets what the CLI needs at run time (bash, procps for Claude Code;
bash, coreutils/stdbuf for OpenCode; both checked first and installed only if missing) and
the binary is copied in from the host:

  $HARNESS_AGENT_CACHE/<claude-code|opencode>/<version>/<platform>/<claude|opencode>
  $HARNESS_AGENT_CACHE/pi/<version>/linux-x64/harbor-pi.tar   (Node.js + Pi under /opt/harbor-pi,
      extracted in the sandbox; /usr/local/bin/pi runs Pi on that Node, which stays off PATH)
      (default /data/aai/tbench/agent_cache; filled by gen/fetch_agent_cli.sh, which
       records each file's sha256 next to it)

staged once per job into $HARNESS_AGENT_CACHE_LOCAL (default $TMPDIR/agent_cache) and
checked against the sha256.  Everything after install (the CLI invocation, env, trajectory
conversion, token accounting) is Harbor's own agent code.  A missing cache entry or a
non-x86_64 / musl sandbox falls back to Harbor's stock install of the same version.

OpenCode clamps every response to 32000 output tokens unless OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX
is set.  OpenCodeCached sets it on its own commands from the host's OC_OUTPUT_TOKEN_MAX (set by
gen/run_harbor.sh to MAX_TOKENS), not through `--ae`: Harbor treats any --ae name matching
KEY|SECRET|TOKEN|... as a secret and replaces its value ("65536") in every file of the trial dir,
which corrupts result.json.

Token accounting: Harbor's OpenCode converter counts OpenCode's `output` tokens, which exclude
reasoning, while Claude Code (Anthropic output_tokens) and Pi (completion_tokens) include it.
OpenCodeCached adds the reasoning tokens back into agent_result.n_output_tokens so the three
harnesses report the same quantity, and keeps the split in agent_result.metadata
(output_tokens_excl_reasoning, reasoning_tokens).  trajectory.json is Harbor's own (per-step
completion_tokens without reasoning; reasoning in metrics.extra.reasoning_tokens).

PiCached also completes Harbor's custom-endpoint models.json (Harbor writes only the model id
and reasoning flags; Pi's defaults for an unknown model are contextWindow 128000 and
maxTokens 16384): contextWindow = $PI_CONTEXT_WINDOW, maxTokens = $PI_MAX_OUTPUT, compat =
$PI_COMPAT_JSON (default {"supportsDeveloperRole": false}: GLM's template has no developer
role), and writes Pi's settings.json next to it with httpIdleTimeoutMs = $PI_HTTP_IDLE_TIMEOUT_MS
(default 1800000; Pi's own default of 5 min would abort requests queued on a loaded server).
PI_SETTINGS_JSON (JSON object) is merged into that settings.json; for pi-cont16k the
context-limit settings (compaction disabled) are applied last.

Consolidated opt-in Pi options (all default to the original behavior unless noted):
  PI_RUNTIME_TMPFS=1            install the Pi runtime into the sandbox's private /dev/shm
                                (falls back to the disk path when unavailable)
  PI_CAPTURE_REQUESTS=1         keep the generated models.json as pi-models.json (nemotron35-super)
  PI_CLI_PROMPT_SEPARATOR=0|1   pi-cont16k: pass the task instruction after `--` so a leading '-'
                                is not parsed as an option (default 1, as aa-evals/eval always did;
                                task-synthesis' copy defaulted to 0 and synth sets 1).
  PI_RUNTIME_COMPAT_ARCHIVE     pi-cont16k: private glibc loader tarball (+ .tar.sha256) used only
                                when the pinned Node fails with "GLIBC_... not found" on an old
                                task image (alias PI_GLIBC_COMPAT_ARCHIVE).
  PI_NATIVE_MCP=0|1             pi-cont16k: 1 uses pinned Pi 1.2.0's native MCP (direct tool exposure);
                                Harbor's adapter extension is not installed by the cached installer.
                                Default 0 (Harbor's adapter config/flag unchanged, as AA evals and
                                eval_q35 run); synth/worker.py sets 1 (task-synthesis' default).
  PI_MCP_AUTO_CODEMODE=0|1      write autoEnableCodemode into mcp.json (unset: omit, Pi default).
  PI_INSTRUCTION_ARGV=0|1       pi-cont16k: 1 passes the task instruction as Pi's last argument; default 0
                                pipes it through stdin (aa-evals #203, as AA evals run; Pi trims stdin).
                                eval_q35's Worker and synth/worker.py set 1 (eval's and task-synthesis'
                                launch).
  environment.install_pi_extensions(agent)  optional environment hook run after install.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import shutil
import time
import uuid
from pathlib import Path

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.opencode import OpenCode
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.pi import (
    _CUSTOM_PROVIDER, _REMOTE_PI_CONFIG_DIR, _REMOTE_PI_MCP_ADAPTER_EXTENSION, Pi,
)
from harbor.environments.base import BaseEnvironment
from aa_evals.common.pi_eval.model_input import from_env as model_input_from_env

_PLATFORM_CMD = (
    "m=$(uname -m); "
    "if ldd --version 2>&1 | grep -qi musl || [ -f /etc/alpine-release ]; then l=-musl; else l=; fi; "
    'echo "$m$l"'
)


def _host_has_avx2() -> bool:
    try:
        return " avx2 " in Path("/proc/cpuinfo").read_text()
    except OSError:
        return True


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def cached_binary(tool: str, version: str | None, platform: str, name: str, *, shared_root=None) -> Path | None:
    """Local (node) copy of a cached CLI binary, staged from the shared cache on first use."""
    if not version:
        return None
    shared = Path(shared_root or os.environ.get("HARNESS_AGENT_CACHE", "/data/aai/tbench/agent_cache"))
    src = shared / tool / version / platform / name
    if not src.is_file():
        return None
    want = (src.parent / f"{name}.sha256").read_text().split()[0] if (src.parent / f"{name}.sha256").is_file() else None
    local_root = Path(os.environ.get("HARNESS_AGENT_CACHE_LOCAL")
                      or Path(os.environ.get("TMPDIR") or f"/tmp/{os.environ.get('USER', 'u')}") / "agent_cache")
    # Stock and patched Pi both call themselves 1.2.0. A version-only cache can
    # silently substitute the wrong bundle (including same-size modifications).
    identity = want or hashlib.sha256(str(src).encode()).hexdigest()
    dst = local_root / identity / tool / version / platform / name
    if dst.is_file() and dst.stat().st_size == src.stat().st_size:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    shutil.copy2(src, tmp)
    if want and _sha256(tmp) != want:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for cached {src}")
    os.chmod(tmp, 0o755)
    os.replace(tmp, dst)  # atomic: concurrent trials on this node may stage at once
    return dst


async def _sandbox_platform(environment: BaseEnvironment) -> str:
    r = await environment.exec(command=_PLATFORM_CMD, user="root")
    return (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""


class ClaudeCodeCached(ClaudeCode):
    async def install(self, environment: BaseEnvironment) -> None:
        if await self._installed_claude_satisfies_version(environment):
            return
        plat = await _sandbox_platform(environment)
        src = cached_binary("claude-code", self._version, "linux-x64", "claude") if plat == "x86_64" else None
        if src is None:
            self.logger.info(f"claude-code {self._version}: no cached binary for {plat!r}; stock install")
            await super().install(environment)
            return
        await self.ensure_system_dependencies(environment, ("bash", "procps"))
        await environment.upload_file(src, "/usr/local/bin/claude")
        await self.exec_as_root(environment, command="chmod 755 /usr/local/bin/claude")
        await self.exec_as_agent(
            environment,
            command=('mkdir -p "$HOME/.local/bin" && ln -sf /usr/local/bin/claude "$HOME/.local/bin/claude" && '
                     'export PATH="$HOME/.local/bin:$PATH" && claude --version'),
        )


class OpenCodeCached(OpenCode):
    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        out_max = os.environ.get("OC_OUTPUT_TOKEN_MAX")
        if out_max:
            env = {**(env or {}), "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": out_max}
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    def populate_context_post_run(self, context) -> None:
        super().populate_context_post_run(context)
        reasoning = 0
        for event in self._parse_stdout():
            if event.get("type") == "step_finish":
                tokens = (event.get("part") or {}).get("tokens") or {}
                reasoning += int(tokens.get("reasoning") or 0)
        if reasoning and context.n_output_tokens is not None:
            context.metadata = {**(context.metadata or {}),
                                "output_tokens_excl_reasoning": context.n_output_tokens,
                                "reasoning_tokens": reasoning}
            context.n_output_tokens += reasoning

    async def install(self, environment: BaseEnvironment) -> None:
        plat = await _sandbox_platform(environment)
        variant = "linux-x64" if _host_has_avx2() else "linux-x64-baseline"
        src = cached_binary("opencode", self._version, variant, "opencode") if plat == "x86_64" else None
        if src is None:
            self.logger.info(f"opencode {self._version}: no cached binary for {plat!r}; stock install")
            await super().install(environment)
            return
        await self.ensure_system_dependencies(environment, ("bash", "coreutils"))
        await environment.upload_file(src, "/usr/local/bin/opencode")
        await self.exec_as_root(environment, command="chmod 755 /usr/local/bin/opencode")
        await self.exec_as_agent(environment, command="opencode --version")


PI_LAUNCH = "pi --print --mode json "
PI_LAUNCH_STDIN = " 2>&1 </dev/null | "   # Harbor's run(): `pi ... <quoted instruction> 2>&1 </dev/null | grep ...`


def instruction_via_stdin(command: str, instruction_file: str) -> tuple[str, str]:
    """(command, instruction): Harbor's Pi launch with the positional instruction replaced by `< instruction_file`.

    Pi runs inside the sandbox, so an instruction passed as an argument sits in the cmdline of the agent's shell and
    of Pi itself, and a model that kills processes by pattern (`pkill -f win311`) on a word from it kills its own
    agent (#203). Pi 1.2.0 takes piped stdin as the initial message (trimmed: the instruction's final newline goes).
    """
    i = command.find(PI_LAUNCH_STDIN)
    head = command[:i] if i >= 0 else ""
    instruction = shlex.split(head)[-1] if head else ""
    quoted = shlex.quote(instruction)
    if i < 0 or not instruction or not head.endswith(" " + quoted):
        raise ValueError("unexpected Harbor Pi launch command: cannot locate the instruction argument")
    return (head[: -len(quoted)].rstrip() + " 2>&1 <" + shlex.quote(instruction_file) + " | "
            + command[i + len(PI_LAUNCH_STDIN):]), instruction


REPO = Path(__file__).resolve().parents[1]
_PI_CLI = '/opt/harbor-pi/pi/lib/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js'


def pi_runtime_tmpfs_enabled():
    return os.environ.get('PI_RUNTIME_TMPFS') == '1'


def prompt_separator_enabled():
    # Default on (aa-evals / eval and the 10-16 base: the separator was unconditional); task-synthesis made it opt-in
    # and synth/worker.py sets 1. '0' turns it off (#480: one harness, per-pipeline flags).
    return os.environ.get('PI_CLI_PROMPT_SEPARATOR', '1') != '0'


def pi_prompt_separator(command):
    """Keep Harbor's exact task prompt positional, including a leading dash (idempotent)."""
    prefix, marker, suffix = command.rpartition(' 2>&1 </dev/null | grep ')
    if not marker:
        raise ValueError('Unrecognized Pi launch redirection; cannot protect positional prompt')
    prompt = shlex.split(prefix)[-1]
    quoted = shlex.quote(prompt)
    if not prefix.endswith(quoted):
        raise ValueError('Unrecognized Pi positional prompt quoting')
    head = prefix[:-len(quoted)]
    if head.endswith(' -- '):
        return command
    return head + '-- ' + quoted + marker + suffix


def _compat_archive():
    """(archive, expected sha256, source) for the private glibc loader, or None."""
    for key in ('PI_RUNTIME_COMPAT_ARCHIVE', 'PI_GLIBC_COMPAT_ARCHIVE'):
        if os.environ.get(key):
            archive = Path(os.environ[key])
            return archive, archive.with_suffix('.tar.sha256').read_text().split()[0], key
    return None


class PiCached(Pi):
    # PI_INSTRUCTION_ARGV=1: the instruction as Pi's last argument (eval_q35's and synth's launch; AA evals since M3.5
    # pipe it through stdin, #203)
    INSTRUCTION_FILE = (_REMOTE_PI_CONFIG_DIR / "instruction.md").as_posix()

    async def run(self, instruction, environment, context):
        if os.environ.get('PI_AGENT_INTERNET') != '0':
            return await super().run(instruction, environment, context)
        backend = self.model_connection.configured_base_url
        if not backend or not hasattr(environment, 'offline_agent_model'):
            raise RuntimeError('offline Pi requires a dedicated model endpoint and the AA namespace environment')
        from aa_evals.common.pi_eval.runtime_cache import prepare as prepare_runtime_cache
        cache = await prepare_runtime_cache(environment, os.environ.get('PI_RUNTIME_CACHE_MANIFEST', ''),
                                            os.environ.get('PI_RUNTIME_CACHE_MANIFEST_SHA256', ''))
        self._runtime_cache_env = cache['agent_env'] if cache else {}
        if cache:
            instruction += '\n\n' + cache['notice']
            context.metadata = {**(context.metadata or {}), 'runtime_package_cache': cache}
        from aa_evals.common.harbor_mswea.offline import capdrop_binary
        binary = await asyncio.to_thread(capdrop_binary)
        await environment.upload_file(binary, '/opt/harbor-pi/offline-capdrop')
        await self.exec_as_root(environment, command='chmod 755 /opt/harbor-pi/offline-capdrop')
        async with environment.offline_agent_model(backend) as url:
            self._offline_model_url = url
            try:
                instruction += '\n\nEnvironment: internet access is disabled during agent execution.'
                context.metadata = {**(context.metadata or {}), 'agent_network': {
                    'internet': False, 'model_relay': 'fixed_endpoint_only',
                    'setup_and_grading_network': 'separate_evaluator_namespace',
                    'agent_capabilities_dropped': ['SYS_ADMIN', 'SYS_PTRACE'], 'no_new_privs': True,
                    'background_agent_processes_remain_offline': True}}
                await super().run(instruction, environment, context)
            finally:
                self._offline_model_url = None
                self._runtime_cache_env = {}

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if PI_LAUNCH in command and getattr(self, '_runtime_cache_env', None):
            env = {**(env or {}), **self._runtime_cache_env}
        if PI_LAUNCH in command and os.environ.get("PI_INSTRUCTION_ARGV") != "1":
            command, instruction = instruction_via_stdin(command, self.INSTRUCTION_FILE)
            await self._upload_config_text(environment, content=instruction, remote_path=self.INSTRUCTION_FILE,
                                           filename="instruction.md")
        if PI_LAUNCH in command and os.environ.get('PI_AGENT_INTERNET') == '0':
            # Replace the launch shell too: every pipeline peer must share the agent's reduced capabilities.
            command = 'exec /opt/harbor-pi/offline-capdrop /bin/bash -c ' + shlex.quote(command)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    def _cached_bundle(self, plat):
        return cached_binary("pi", self._version, "linux-x64", "harbor-pi.tar") if plat == "x86_64" else None

    async def install(self, environment: BaseEnvironment) -> None:
        plat = await _sandbox_platform(environment)
        src = self._cached_bundle(plat)
        if src is None:
            self.logger.info(f"pi {self._version}: no cached bundle for {plat!r}; stock install")
            await super().install(environment)
            return
        # run() pipes Pi's output through grep and `stdbuf -oL tee`
        await self.ensure_system_dependencies(environment, ("bash", "tar", "coreutils"))
        requested = pi_runtime_tmpfs_enabled()
        memory_install = False
        if requested:
            probe = await environment.exec(command=(
                'test "$(stat -f -c %T /dev/shm)" = tmpfs && '
                'test ! -e /opt/harbor-pi && test ! -L /opt/harbor-pi'), user='root')
            memory_install = probe.return_code == 0
        archive = '/dev/shm/harbor-pi.tar' if memory_install else '/tmp/harbor-pi.tar'
        self._pi_runtime_in_tmpfs = memory_install
        started = time.monotonic()
        await environment.upload_file(src, archive)
        uploaded = time.monotonic()
        if memory_install:
            command = ('mkdir -p /dev/shm/harbor-pi-runtime && '
                'tar -xf /dev/shm/harbor-pi.tar -C /dev/shm/harbor-pi-runtime && '
                'rm -f /dev/shm/harbor-pi.tar && mkdir -p /opt /usr/local/bin && '
                'ln -s /dev/shm/harbor-pi-runtime/opt/harbor-pi /opt/harbor-pi && '
                'ln -sf /opt/harbor-pi/bin-pi /usr/local/bin/pi')
        else:
            command = ("tar -xf /tmp/harbor-pi.tar -C / && rm -f /tmp/harbor-pi.tar && "
                       "mkdir -p /usr/local/bin && ln -sf /opt/harbor-pi/bin-pi /usr/local/bin/pi")
        await self.exec_as_root(environment, command=command)
        await self.exec_as_agent(environment, command="pi --version")
        extracted = time.monotonic()
        if requested:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / 'pi-runtime-install.json').write_text(json.dumps({
                'mode': 'private-tmpfs-symlink-v1' if memory_install else 'original-disk-fallback',
                'archive': str(src), 'archive_sha256': _sha256(src),
                'upload_seconds': uploaded-started, 'extract_seconds': extracted-uploaded,
                'installed_path': '/opt/harbor-pi',
                'memory_accounting': 'Private namespace tmpfs; charged to owning job cgroup, retained in working-set admission. Released with sandbox.',
                'raw_artifact_storage_unchanged': True}, indent=2)+'\n')

    def _build_custom_models_json(self, access, model_id):
        models_json = super()._build_custom_models_json(access, model_id)
        if models_json is None:
            return None
        provider = models_json["providers"][_CUSTOM_PROVIDER]
        modalities = model_input_from_env(os.environ)
        for model in provider["models"]:
            if modalities is not None:
                model['input'] = modalities
            if os.environ.get("PI_CONTEXT_WINDOW"):
                model["contextWindow"] = int(os.environ["PI_CONTEXT_WINDOW"])
            if os.environ.get("PI_MAX_OUTPUT"):
                model["maxTokens"] = int(os.environ["PI_MAX_OUTPUT"])
        provider["compat"] = json.loads(os.environ.get("PI_COMPAT_JSON") or '{"supportsDeveloperRole": false}')
        if os.environ.get("PI_CAPTURE_REQUESTS") == "1":
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / "pi-models.json").write_text(json.dumps(models_json, indent=2) + "\n")
        return models_json

    async def _write_custom_models_json(self, environment: BaseEnvironment, models_json) -> None:
        if getattr(self, '_offline_model_url', None):
            models_json['providers'][_CUSTOM_PROVIDER]['baseUrl'] = self._offline_model_url
        await super()._write_custom_models_json(environment, models_json)
        settings = {"httpIdleTimeoutMs": int(os.environ.get("PI_HTTP_IDLE_TIMEOUT_MS") or 1800000)}
        settings.update(json.loads(os.environ.get('PI_SETTINGS_JSON') or '{}'))
        if isinstance(self, PiCont16kCached) and os.environ.get('PI_CONTEXT_POLICY', 'submit') != 'compact':
            from tb_agents.pi_context_limit import settings as context_settings
            settings.update(context_settings(settings['httpIdleTimeoutMs']))
        await self._upload_config_text(
            environment,
            content=json.dumps(settings, indent=2) + "\n",
            remote_path=(_REMOTE_PI_CONFIG_DIR / "settings.json").as_posix(),
            filename="settings.json",
        )


class PiCont16kCached(PiCached):
    """Continuation is performed by the pinned Pi bundle, never hidden in a proxy."""

    def build_cli_flags(self) -> str:
        flags = super().build_cli_flags()
        if prompt_separator_enabled():
            # Harbor appends the literal task instruction right after these flags.
            # Pi otherwise parses an instruction beginning with '-' as an option and
            # exits before any model request (TB2.1 pytorch-model-recovery).
            return (flags + " --").strip()
        return flags

    def _native_mcp(self) -> bool:
        # Default off: AA evals and eval_q35 keep Harbor's adapter config/flag as they ran it. Synthesis sets
        # PI_NATIVE_MCP=1 (synth/worker.py; task-synthesis' copy defaulted to on). #480 "Defaults that differ".
        return os.environ.get('PI_NATIVE_MCP', '0') == '1'

    def _build_mcp_config(self):
        config = super()._build_mcp_config()
        if not self._native_mcp():
            return config
        # Harbor's adapter targets Pi <0.86 and is never installed by the cached
        # installer; pinned Pi 1.2.0 implements MCP natively. Its adapter settings
        # are dropped; task endpoints are kept with direct tool exposure.
        config.pop('settings', None)
        for server in config['mcpServers'].values():
            server['exposure'] = 'direct'
        if os.environ.get('PI_MCP_AUTO_CODEMODE') in ('0', '1'):
            config['autoEnableCodemode'] = os.environ['PI_MCP_AUTO_CODEMODE'] == '1'
        return config

    async def _write_mcp_config(self, environment):
        await super()._write_mcp_config(environment)
        if not self._native_mcp():
            return
        # Harbor still passes the adapter path in run(); exec_as_agent removes it, and
        # a no-op file at that path keeps the command valid if it is ever kept.
        stub = ('// MCP is supplied by pinned Pi 1.2.0, not this compatibility file.\n'
                'export default function () {}\n')
        await self.exec_as_agent(environment, command='mkdir -p ' + shlex.quote(
            _REMOTE_PI_MCP_ADAPTER_EXTENSION.parent.as_posix()))
        await self._upload_config_text(environment, content=stub,
            remote_path=_REMOTE_PI_MCP_ADAPTER_EXTENSION.as_posix(),
            filename='native-mcp-compat.ts')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / 'pi-native-mcp.json').write_text(json.dumps({
            'implementation': 'pinned-pi-1.2.0-native', 'tool_exposure': 'direct',
            'servers': [server.name for server in self.mcp_servers],
            'legacy_extension': _REMOTE_PI_MCP_ADAPTER_EXTENSION.as_posix(),
            'compatibility_stub_sha256': hashlib.sha256(stub.encode()).hexdigest(),
            'pi_bundle_unchanged': True}, indent=2) + '\n')

    async def exec_as_agent(self, environment, command, env=None, cwd=None, timeout_sec=None):
        if self._native_mcp() and getattr(self, 'mcp_servers', None) and '--print --mode json' in command:
            # Harbor puts this flag before the instruction. Remove that first
            # occurrence only; literal text inside the task prompt is preserved.
            legacy = '--extension ' + shlex.quote(_REMOTE_PI_MCP_ADAPTER_EXTENSION.as_posix()) + ' '
            command = command.replace(legacy, '', 1)
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    async def install(self, environment: BaseEnvironment) -> None:
        try:
            await super().install(environment)
        except NonZeroAgentExitCodeError as exc:
            # Some task images predate the pinned Node binary's glibc. Keep the Pi
            # bundle and task runtime intact: run only Pi's Node through a
            # separately checksummed private loader.
            if 'GLIBC_' not in str(exc) or 'not found' not in str(exc):
                raise
            found = _compat_archive()
            if found is None:
                raise
            archive, expected, source = found
            if _sha256(archive) != expected:
                raise ValueError('Pi private runtime checksum mismatch')
            await environment.upload_file(archive, '/tmp/harbor-pi-glibc-compat.tar')
            wrapper = ('#!/bin/sh\nexec /opt/harbor-pi/glibc-compat/ld-linux-x86-64.so.2 '
                       '--library-path /opt/harbor-pi/glibc-compat '
                       '/opt/harbor-pi/node/bin/node ' + _PI_CLI + ' "$@"\n')
            target = '/dev/shm/harbor-pi-runtime' if getattr(self, '_pi_runtime_in_tmpfs', False) else '/'
            await self.exec_as_root(environment, command=(
                'tar -xf /tmp/harbor-pi-glibc-compat.tar -C ' + target + ' && '
                'rm -f /tmp/harbor-pi-glibc-compat.tar && printf %s ' + shlex.quote(wrapper) +
                ' > /opt/harbor-pi/bin-pi && chmod 755 /opt/harbor-pi/bin-pi'))
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.logs_dir / 'pi-runtime-compat.json').write_text(json.dumps({
                'archive': str(archive), 'sha256': expected, 'source': source,
                'reason': 'Pinned Node binary requires a newer glibc than this task image',
                'pinned_pi_bundle_unchanged': True, 'task_runtime_unchanged': True,
                'wrapper': wrapper}, indent=2) + '\n')
            await self.exec_as_agent(environment, command='pi --version')
        install_extensions = getattr(environment, 'install_pi_extensions', None)
        if install_extensions is not None:
            await install_extensions(self)

    def _cached_bundle(self, plat):
        from tb_agents.pi_continuation import VERSION, bundle_provenance, digest, launch_cache, verify_bundle
        if plat != "x86_64" or self._version != VERSION:
            raise ValueError("pi-cont16k requires the patched Pi 1.2.0 linux-x64 bundle")
        cache = launch_cache()
        self._continuation_sha256 = verify_bundle(cache)
        self._continuation_provenance = bundle_provenance(cache, self._continuation_sha256)
        src = cached_binary("pi", VERSION, "linux-x64", "harbor-pi.tar", shared_root=cache)
        if src is None or digest(src) != self._continuation_sha256:
            raise ValueError("Local Pi continuation bundle checksum mismatch")
        return src

    def _build_custom_models_json(self, access, model_id):
        cap = int(os.environ.get("PI_CONT16K_CAP", "16384"))
        if not 1 <= cap <= 16384 or int(os.environ.get("PI_MAX_OUTPUT", "0")) != cap:
            raise ValueError("Pi maxTokens must equal the continuation proxy cap")
        if os.environ.get("PI_CONT16K_PROXY_READY") != "1":
            raise ValueError("pi-cont16k must run through gen/run_harbor.sh's continuation proxy")
        return super()._build_custom_models_json(access, model_id)

    def populate_context_post_run(self, context):
        super().populate_context_post_run(context)
        from tb_agents.pi_continuation import annotate_sessions
        annotations = annotate_sessions(self.logs_dir)
        messages = [e for s in annotations['sessions'] for e in s['entries'] if e.get('type') == 'message']
        context.metadata = {**(context.metadata or {}), 'pi_continuation': {
            'variant': 'pi-cont16k', 'max_output_tokens': int(os.environ.get('PI_CONT16K_CAP', '16384')),
            'proxy_resamples': int(os.environ.get('PI_CONT16K_RESAMPLES', '0')), 'proxy_continuations': 0,
            'bundle_sha256': getattr(self, '_continuation_sha256', None),
            **getattr(self, '_continuation_provenance', {}),
            'length_stops': sum(e['message'].get('stopReason') == 'length' for e in messages),
            'sft_events': 'pi/sft-events.json'}}


class PiCont16kEnvClock(PiCont16kCached):
    """pi-cont16k whose time limit counts environment time only, not model time.

    Same rule as tb_agents/terminus_envclock.py (branch eval-envclock): the task's [agent]
    timeout_sec (x $PI_ENVCLOCK_MULT, default 1) is charged only while no model request is in
    flight.  Pi runs inside the sandbox, so model time is read from the rollout's own
    continuation proxy ($PI_MODEL_CLOCK_URL, set by gen/pi_cont16k.py): seconds with at least
    one generation request in flight (queueing, prefill, decode).  When the budget is spent the
    run is cancelled as Harbor's own timeout would cancel it (AgentTimeoutError; the verifier
    then scores the environment).  Harbor's wall-clock timeout stays as a backstop
    (--agent-timeout-multiplier).  Accounting: <agent logs>/env_clock.json, whose state says
    which limit ended the run (env_budget_spent vs wall_clock_backstop).
    """

    async def _model_sec(self, url):
        import urllib.request

        def fetch():
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=10) as response:
                return json.loads(response.read())

        return await asyncio.to_thread(fetch)

    async def run(self, instruction, environment, context) -> None:
        import time
        from harbor.models.task.config import TaskConfig
        from harbor.trial.errors import AgentTimeoutError

        toml = Path(environment.environment_dir).parent / "task.toml"
        limit = TaskConfig.model_validate_toml(toml.read_text()).agent.timeout_sec
        budget = None if limit is None else limit * float(os.environ.get("PI_ENVCLOCK_MULT") or 1)
        url = os.environ["PI_MODEL_CLOCK_URL"]
        cut_file = os.environ.get("PI_TAIL_CUT_FILE")   # eval_q35 tail rule: its existence ends the run like a timeout
        t0 = time.monotonic()
        base = (await self._model_sec(url))["model_sec"]
        model = {"sec": 0.0, "at": t0, "in_flight": 0, "requests": 0, "errors": 0}

        async def poll():
            try:
                clock = await self._model_sec(url)
            except Exception:  # noqa: BLE001 -- proxy busy: extrapolate from the last reading
                model["errors"] += 1
                return
            model.update(sec=clock["model_sec"] - base, at=time.monotonic(),
                         in_flight=clock["in_flight"], requests=clock["requests"],
                         gate_held_sec=clock.get("gate_held_sec", 0.0))

        def model_now():
            return model["sec"] + (time.monotonic() - model["at"] if model["in_flight"] else 0.0)

        state = "finished"
        task = asyncio.create_task(super().run(instruction, environment, context))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=5)
                if done:
                    return task.result()
                await poll()
                if budget is not None and time.monotonic() - t0 - model_now() >= budget:
                    state = "env_budget_spent"
                    raise AgentTimeoutError(
                        f"environment-time budget of {budget:.0f} s spent (model time excluded)")
                if cut_file and os.path.exists(cut_file):   # eval tail cut (eval_q35): stop as the time limit would
                    state = "tail_cut"
                    raise AgentTimeoutError("eval tail cut: agent stopped by the operator's tail rule")
        except asyncio.CancelledError:
            state = "wall_clock_backstop"
            raise
        except AgentTimeoutError:
            raise
        except Exception:
            state = "error"
            raise
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            try:
                await poll()
            except Exception:  # noqa: BLE001
                pass
            wall = time.monotonic() - t0
            try:
                self.logs_dir.mkdir(parents=True, exist_ok=True)
                (Path(self.logs_dir) / "env_clock.json").write_text(json.dumps({
                    "state": state, "budget_sec": budget, "wall_sec": round(wall, 1),
                    "model_sec": round(model_now(), 1), "env_sec": round(wall - model_now(), 1),
                    "model_requests": model["requests"], "kv_gate_held_sec": model.get("gate_held_sec", 0.0),
                    "clock_read_errors": model["errors"]}) + "\n")
            except OSError:
                pass


class PiCont16kEnvClockMuslCompat(PiCont16kEnvClock):
    """Explicit task-only private runtime; the standard Pi protocol remains inherited."""

    async def install(self, environment: BaseEnvironment) -> None:
        from tb_agents.pi_musl_compat import install
        await install(self, environment)

    async def _upload_config_text(self, environment: BaseEnvironment, *, content, remote_path, filename) -> None:
        from tb_agents.pi_musl_compat import upload_config
        await upload_config(self, environment, super()._upload_config_text,
                            content=content, remote_path=remote_path, filename=filename)
