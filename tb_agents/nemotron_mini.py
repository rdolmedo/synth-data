"""Harbor mini v2 integration using the pinned Lightning fork for both tool sets."""
import os
from pathlib import Path

from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from tb_agents.harness_agents import cached_binary, _sandbox_platform


class MiniSweAgentCached(MiniSweAgent):
    def get_version_command(self):
        return "/opt/harbor-mini/bin/python3.12 -c 'import minisweagent; print(minisweagent.__version__)'"

    async def install(self, environment):
        platform = await _sandbox_platform(environment)
        if platform != "x86_64":
            raise RuntimeError(f"Pinned mini v2 bundle requires glibc x86_64; got {platform}")
        cache = Path(os.environ.get("MINI_AGENT_CACHE", "/data/aai/tbench/nemotron-super/agent_cache"))
        revision = (cache / "mini-revision.txt").read_text().strip()
        # cached_binary supports a per-tool root through the common cache variable.
        old = os.environ.get("HARNESS_AGENT_CACHE")
        os.environ["HARNESS_AGENT_CACHE"] = str(cache)
        try:
            src = cached_binary("mini-swe-agent", revision, "linux-x64", "harbor-mini.tar")
        finally:
            if old is None:
                os.environ.pop("HARNESS_AGENT_CACHE", None)
            else:
                os.environ["HARNESS_AGENT_CACHE"] = old
        if src is None:
            raise RuntimeError("Pinned mini v2 cache missing; run gen/prepare_nemotron_mini.sh")
        await self.ensure_system_dependencies(environment, ("bash", "tar", "coreutils"))
        await environment.upload_file(src, "/tmp/harbor-mini.tar")
        await self.exec_as_root(environment, command=(
            "tar -xf /tmp/harbor-mini.tar -C / && rm /tmp/harbor-mini.tar && "
            "ln -sf /opt/harbor-mini/mini-swe-agent /usr/local/bin/mini-swe-agent && "
            "(command -v python3 >/dev/null || ln -s /opt/harbor-mini/bin/python3.12 /usr/local/bin/python3)"))
        await self.exec_as_agent(environment, command="/opt/harbor-mini/bin/python3.12 -c "
            "'import minisweagent; print(minisweagent.__version__)'")
