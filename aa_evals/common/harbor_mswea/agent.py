"""Harbor 0.24's stock mini-swe-agent agent, installed from a pinned bundle instead of uv/PyPI in every sandbox.

    -a aa_evals.common.harbor_mswea.agent:MiniSweAgentAA -m hosted_vllm/<served name>
       --ak version=2.4.6 --ak config='<mini config JSON, see run_config.mini_config>'

Everything after install is Harbor's own `MiniSweAgent.run`: `mini-swe-agent --yolo --exit-immediately
-c mini -c model.model_kwargs.timeout=3600 -c <our config>` with the task instruction, the trajectory in
/logs/agent/mini-swe-agent.trajectory.json and Harbor's ATIF conversion and token accounting.

The install differs from stock in two ways only:
- The bundle (build_bundle.sh) is a relocatable CPython 3.12 with stock mini-swe-agent==<version> from PyPI under
  /opt/aa-mswea, uploaded and unpacked (sha256-checked, staged once per node). Stock Harbor runs `curl astral.sh |
  sh; uv python install; uv tool install mini-swe-agent` through the cluster proxy in every sandbox.
- Stock install also apt-installs curl, git and build tools into the task image when missing. Those would change
  the task environment the agent sees, so only `tar` (to unpack) is ensured here. Images without glibc x86_64 (musl,
  arm) fall back to Harbor's stock install of the same version, and say so in the trial log.
"""
from __future__ import annotations

import os
from pathlib import Path

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from harbor.environments.base import BaseEnvironment

from tb_agents.harness_agents import _sandbox_platform, cached_binary

BUNDLE_ROOT = Path(os.environ.get("AA_MSWEA_CACHE", "/data/aai/aa_evals/agent_cache"))
BUNDLE_NAME = "aa-mswea.tar"
PREFIX = "/opt/aa-mswea"
DEFAULT_VERSION = "2.4.6"


def bundle_path(version: str = DEFAULT_VERSION, root: Path = BUNDLE_ROOT) -> Path:
    return root / "mini-swe-agent" / version / "linux-x64" / BUNDLE_NAME


class MiniSweAgentAA(MiniSweAgent):
    def __init__(self, *args, version: str | None = None, **kwargs):
        super().__init__(*args, version=version or DEFAULT_VERSION, **kwargs)

    def get_version_command(self) -> str | None:
        return (f"if [ -x {PREFIX}/bin/python3.12 ]; then MSWEA_SILENT_STARTUP=1 {PREFIX}/bin/python3.12 -c "
                "'import minisweagent; print(\"mini-swe-agent \" + minisweagent.__version__)'; "
                "else " + (super().get_version_command() or "true") + "; fi")

    async def install(self, environment: BaseEnvironment) -> None:
        platform = await _sandbox_platform(environment)
        src = cached_binary("mini-swe-agent", self._version, "linux-x64", BUNDLE_NAME,
                            shared_root=BUNDLE_ROOT) if platform == "x86_64" else None
        if src is None:
            self.logger.warning(f"aa-mswea: no bundle for platform {platform!r} (version {self._version}); "
                                "Harbor's stock mini-swe-agent install")
            await super().install(environment)
            return
        await self.ensure_system_dependencies(environment, ("tar",))
        await environment.upload_file(src, "/tmp/aa-mswea.tar")
        await self.exec_as_root(environment, command=(
            f"set -e; tar -xf /tmp/aa-mswea.tar -C / && rm -f /tmp/aa-mswea.tar && "
            f"ln -sf {PREFIX}/mini-swe-agent /usr/local/bin/mini-swe-agent"))
        # Harbor's run() puts $HOME/.local/bin first on PATH (where `uv tool install` puts the CLI)
        await self.exec_as_agent(environment, command=(
            f'mkdir -p "$HOME/.local/bin" && ln -sf {PREFIX}/mini-swe-agent "$HOME/.local/bin/mini-swe-agent" && '
            f"MSWEA_SILENT_STARTUP=1 {PREFIX}/bin/python3.12 -c "
            "'import minisweagent, sys; v = minisweagent.__version__; print(\"mini-swe-agent\", v); "
            f"sys.exit(v != \"{self._version}\")'"))
