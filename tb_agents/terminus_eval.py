"""Harbor's terminus-2 under the eval rules of the token-matched campaign (eval_q35, harness "terminus-2").

The stock agent (Harbor 0.24.0, JSON parser), run with interleaved_thinking=True, enable_summarize=False and
proactive_summarization_threshold=0, with four additions:

1. Environment clock: the task's [agent] timeout_sec (x env_budget_multiplier; <= 0 turns it off) is charged only
   for time outside LLM calls, the rule of tb_agents/terminus_envclock.py (branch eval-envclock) and of Pi's
   PiCont16kEnvClock.  When it is spent the run is cancelled as Harbor's timeout would cancel it
   (AgentTimeoutError; the verifier then scores the environment).  Harbor's wall-clock timeout stays as a backstop
   (--agent-timeout-multiplier).
2. The context limit ends the episode.  Without summarization terminus-2 raises ContextLengthExceededError when
   vLLM rejects a request (prompt + max_tokens > --max-model-len).  Harbor would record that as a trial error and
   skip the verifier; here run() returns normally instead, so the verifier scores the final state (as Pi's
   submit-at-limit does).
3. Per-request seeds: request i of the trajectory gets seed (sampling_seed + i) mod 2^31-1, as the Pi proxy does.
   Harbor's own retries of a failed request resend the same seed.
4. KV gate (kv_gate "HIGH,LOW", kv_gate_file {t, kv} written by the serving job): a request waits while the
   published KV usage is >= HIGH, until it is <= LOW (readings older than 120 s leave the gate open; at most
   20 min per request).  Waiting counts as model time, like queueing inside vLLM.

5. Tail cut (tail_cut_file, eval_q35 tail rule): when that file exists the run is cancelled as the time limit would
   cancel it (AgentTimeoutError), so the verifier scores the environment.

Accounting, in the agent log directory: env_clock.json (state finished | context_limit | env_budget_spent | tail_cut |
wall_clock_backstop | error, budget/wall/llm/env seconds, requests, output-cap hits, gate holds) and requests.jsonl
(per request: index, history length, seed, prompt/completion tokens, error type; no content), which
eval_q35/check_history.py uses with the trial's stored messages.
"""
import asyncio
import json
import random
import time
from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2, Terminus2Options
from harbor.llms.base import ContextLengthExceededError, OutputLengthExceededError
from harbor.models.task.config import TaskConfig
from harbor.trial.errors import AgentTimeoutError

SEED_MOD = 2147483647


class Terminus2EvalOptions(Terminus2Options):
    """Harbor validates --ak against the agent's options model (extra=forbid), also in preflight."""
    env_budget_multiplier: float = 1.0
    sampling_seed: int | None = None
    kv_gate: str = ""
    kv_gate_file: str = ""
    tail_cut_file: str = ""


class Terminus2Eval(Terminus2):
    options_model = Terminus2EvalOptions

    def __init__(self, *args, env_budget_multiplier: float = 1.0, sampling_seed: int | None = None,
                 kv_gate: str = "", kv_gate_file: str = "", tail_cut_file: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._cut_file = str(tail_cut_file)
        self._env_mult = float(env_budget_multiplier)
        self._seed = None if sampling_seed is None else int(sampling_seed)
        gate = kv_gate if isinstance(kv_gate, (list, tuple)) else str(kv_gate or "").split(",")
        self._gate = tuple(float(x) for x in gate) if kv_gate else None
        self._gate_file = str(kv_gate_file)
        self._gate_state = {"open": True, "kv": 0.0, "at": -1e9, "holds": 0, "held_sec": 0.0}
        self._llm_sec, self._llm_depth, self._llm_since = 0.0, 0, 0.0
        self._n_requests, self._output_cap_hits = 0, 0
        call = self._llm.call    # the agent loop and Harbor's output-cap retry both go through it

        async def eval_call(*a, **kw):
            if self._llm_depth == 0:
                self._llm_since = time.monotonic()
            self._llm_depth += 1
            index = self._n_requests
            self._n_requests += 1
            rec = {"i": index, "history": len(kw.get("message_history") or []), "t": round(time.time(), 1)}
            if self._seed is not None:
                kw["seed"] = rec["seed"] = (self._seed + index) % SEED_MOD
            t0 = time.monotonic()
            try:
                rec["gate_sec"] = round(await self._wait_gate(), 1)
                response = await call(*a, **kw)
                usage = response.usage
                if usage is not None:
                    rec.update(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)
                return response
            except BaseException as e:
                rec["error"] = type(e).__name__
                if isinstance(e, OutputLengthExceededError):
                    self._output_cap_hits += 1
                raise
            finally:
                rec["sec"] = round(time.monotonic() - t0, 1)
                self._llm_depth -= 1
                if self._llm_depth == 0:
                    self._llm_sec += time.monotonic() - self._llm_since
                try:
                    with open(Path(self.logs_dir) / "requests.jsonl", "a") as f:
                        f.write(json.dumps(rec) + "\n")
                except OSError:
                    pass

        self._llm.call = eval_call

    async def _wait_gate(self) -> float:
        if not self._gate:
            return 0.0
        high, low = self._gate
        g, t0, held = self._gate_state, time.monotonic(), False
        while time.monotonic() - t0 < 1200:
            if time.monotonic() - g["at"] >= 10:
                g["at"] = time.monotonic()
                try:
                    j = json.loads(Path(self._gate_file).read_text())
                    g["kv"] = float(j["kv"]) if time.time() - float(j["t"]) < 120 else 0.0
                except (OSError, ValueError, KeyError, TypeError):
                    g["kv"] = 0.0
            if g["open"] and g["kv"] >= high:
                g["open"] = False
            elif not g["open"] and g["kv"] <= low:
                g["open"] = True
                if held:   # spread the requests released together (they would refill KV at once)
                    await asyncio.sleep(random.uniform(0, 30))
            if g["open"]:
                break
            held = True
            await asyncio.sleep(5)
        if held:
            g["holds"] += 1
            g["held_sec"] += time.monotonic() - t0
        return time.monotonic() - t0

    def _llm_now(self) -> float:
        return self._llm_sec + (time.monotonic() - self._llm_since if self._llm_depth else 0.0)

    def _budget(self, environment) -> float | None:
        if self._env_mult <= 0:
            return None
        toml = Path(environment.environment_dir).parent / "task.toml"
        t = TaskConfig.model_validate_toml(toml.read_text()).agent.timeout_sec
        return None if t is None else t * self._env_mult

    async def run(self, instruction, environment, context) -> None:
        budget = self._budget(environment)
        self._llm_sec, self._llm_depth = 0.0, 0
        t0 = time.monotonic()
        state, error = "finished", None
        task = asyncio.create_task(super().run(instruction, environment, context))
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=2)
                if done:
                    try:
                        return task.result()
                    except ContextLengthExceededError:
                        state = "context_limit"   # the episode ends here; Harbor's verifier scores the final state
                        return None
                if budget is not None and time.monotonic() - t0 - self._llm_now() >= budget:
                    state = "env_budget_spent"
                    raise AgentTimeoutError(
                        f"environment-time budget of {budget:.0f} s spent (model time excluded)")
                if self._cut_file and Path(self._cut_file).exists():   # eval tail cut: stop as the time limit would
                    state = "tail_cut"
                    raise AgentTimeoutError("eval tail cut: agent stopped by the operator's tail rule")
        except asyncio.CancelledError:
            state = "wall_clock_backstop"
            raise
        except AgentTimeoutError:
            raise
        except Exception as e:
            state, error = "error", type(e).__name__
            raise
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            wall = time.monotonic() - t0
            llm = self._llm_now()
            g = self._gate_state
            try:
                (Path(self.logs_dir) / "env_clock.json").write_text(json.dumps({
                    "harness": "terminus-2", "state": state, "error": error, "budget_sec": budget,
                    "wall_sec": round(wall, 1), "llm_sec": round(llm, 1), "env_sec": round(wall - llm, 1),
                    "requests": self._n_requests, "output_cap_hits": self._output_cap_hits,
                    "episodes": self._n_episodes, "gate_holds": g["holds"],
                    "gate_held_sec": round(g["held_sec"], 1)}) + "\n")
            except OSError:
                pass
