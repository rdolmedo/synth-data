"""terminus-2 whose time limit counts environment time only, not model time.

Harbor enforces a task's agent timeout on wall-clock time, which includes waiting for the
model.  Served locally (vLLM on an H100 shared by ~24 rollouts, ~60-90 tok/s per rollout)
a model decodes far slower than a hosted API, so a model that thinks at length times out on
decode speed rather than on what it does in the terminal.  This agent charges the task's
[agent] timeout_sec (x env_budget_multiplier) only for time spent outside LLM calls:
running commands (terminus-2 caps each command's wait at 60 s), reading the terminal,
bookkeeping.  Time inside LLM calls (queueing, prefill, decode, retries) is free; commands
left running in the background keep running meanwhile.  When the budget is spent the run
is cancelled just as Harbor's own timeout would cancel it (AgentTimeoutError, then the
verifier scores the environment).  The episode is still bounded by the context cap
(auto-submit, eval/eval_job.sh), and Harbor's wall-clock timeout stays as a backstop:
raise it with --agent-timeout-multiplier.

Per-trial accounting goes to <agent logs>/env_clock.json.

  harbor run ... -a tb_agents.terminus_envclock:Terminus2EnvClock --agent-timeout-multiplier 40

result.json then reports a timeout as "timed out after <backstop> seconds" (Harbor re-wraps the
error); env_clock.json says which limit ended the run (state env_budget_spent vs wall_clock_backstop).
"""
import asyncio
import json
import time
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.models.task.config import TaskConfig
from harbor.trial.errors import AgentTimeoutError


class Terminus2EnvClock(Terminus2):
    def __init__(self, *args, env_budget_multiplier: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._env_mult = float(env_budget_multiplier)
        self._llm_sec = 0.0
        self._llm_depth = 0      # overlapping LLM calls (subagents) count once
        self._llm_since = 0.0
        call = self._llm.call    # Chat, summarization and subagents all go through it

        async def timed_call(*a, **kw):
            if self._llm_depth == 0:
                self._llm_since = time.monotonic()
            self._llm_depth += 1
            try:
                return await call(*a, **kw)
            finally:
                self._llm_depth -= 1
                if self._llm_depth == 0:
                    self._llm_sec += time.monotonic() - self._llm_since

        self._llm.call = timed_call

    def _llm_now(self) -> float:
        return self._llm_sec + (time.monotonic() - self._llm_since if self._llm_depth else 0.0)

    def _budget(self, environment) -> float | None:
        toml = Path(environment.environment_dir).parent / "task.toml"
        t = TaskConfig.model_validate_toml(toml.read_text()).agent.timeout_sec
        return None if t is None else t * self._env_mult

    async def run(self, instruction, environment, context) -> None:
        budget = self._budget(environment)
        self._llm_sec, self._llm_depth = 0.0, 0
        t0 = time.monotonic()
        state = "finished"
        task = asyncio.create_task(super().run(instruction, environment, context))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=2)
                if done:
                    return task.result()
                if budget is not None and time.monotonic() - t0 - self._llm_now() >= budget:
                    state = "env_budget_spent"
                    raise AgentTimeoutError(
                        f"environment-time budget of {budget:.0f} s spent (model time excluded)")
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
            wall = time.monotonic() - t0
            llm = self._llm_now()
            try:
                (Path(self.logs_dir) / "env_clock.json").write_text(json.dumps({
                    "state": state, "budget_sec": budget, "wall_sec": round(wall, 1),
                    "llm_sec": round(llm, 1), "env_sec": round(wall - llm, 1)}) + "\n")
            except OSError:
                pass
