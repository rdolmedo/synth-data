"""AutomationBench-AA on our cluster: upstream zapier/AutomationBench v1.0.6, AA's scoring.

    python -m aa_evals.automationbench.run --model serving:aa-super --run-id smoke-super --limit 12

The upstream harness needs Python >= 3.13 and its own pinned venv (see setup.sh); when started
under another interpreter this module re-executes itself in that venv.

The upstream environment, task set, tools, assertions and retrying chat-completions client are
used unmodified. This module adds: model-spec resolution, the endpoint's request profile
(sampling overrides plus reasoning-history adapters), AA's objective/guardrail score
(scoring.py), infra-vs-model error classification, resume, and the aa_evals output contract.
"""

from __future__ import annotations

import os
import sys

VENV = os.environ.get("AA_AUTOMATIONBENCH_VENV", "/opt/aai/venvs/automationbench")


def _ensure_venv() -> None:
    try:
        import automationbench  # noqa: F401
        import verifiers  # noqa: F401
        return
    except ImportError:
        pass
    py = os.path.join(VENV, "bin", "python")
    if os.path.realpath(sys.executable) == os.path.realpath(py) or os.environ.get("_AA_AB_REEXEC"):
        raise SystemExit(f"automationbench is not importable from {sys.executable}; run setup.sh")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, _AA_AB_REEXEC="1", PYTHONNOUSERSITE="1")
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    os.execve(py, [py, "-m", "aa_evals.automationbench.run", *sys.argv[1:]], env)


if __name__ == "__main__":
    _ensure_venv()

import argparse  # noqa: E402
import asyncio  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from aa_evals.automationbench.scoring import aa_task_score  # noqa: E402
from aa_evals.tracejudge import fold_summary  # noqa: E402
from aa_evals.common import jsonl  # noqa: E402
from aa_evals.common import serving  # noqa: E402  (tbshared/serving.py, an ALIASES copy: frozen zips carry it)

BENCHMARK = "AutomationBench-AA"
METRIC = "AutomationBench-AA score (share of objectives completed; any guardrail violation zeroes the task)"
UPSTREAM_REPO = "https://github.com/zapier/AutomationBench"
UPSTREAM_COMMIT = "4a8e1061254004d9dac807054eed33fad7d1ff14"
UPSTREAM_VERSION = "1.0.6"
PUBLIC_DOMAINS = ["sales", "marketing", "operations", "support", "finance", "hr"]
SERVING_ROOT = None   # None: serving.registry_root(), $TASKSYNTH_SERVING_ROOT else the production registry
RUNS_ROOT = Path("/data/aai/aa_evals/runs/automationbench")
LEADERBOARD = Path("/data/aai/tbench/index_audit/aa_leaderboard_2026-11-02.json")
ENDPOINT_MAX_AGE_S = 120
MAX_INFRA_RATE = 0.02

# History adapters added on top of the endpoint's request profile, by served name (see README:
# "Null tool arguments" and "Request profile and reasoning history"). `--adapters` replaces the whole list.
EXTRA_ADAPTERS: dict[str, list[str]] = {
    "nemotron-3.5-super-vl": ["xml_null_args"],
}
# vLLM >= 0.31 drops `reasoning_content` from request messages and maps only `reasoning` into the template, so every
# model whose profile has no reasoning-history adapter gets mirror_reasoning_content (Qwen's template with
# preserve_thinking would otherwise replay every earlier turn as <think></think>; M3.2: the same default for all).
HISTORY_ADAPTERS = ("inline_reasoning_history", "mirror_reasoning_content")
CONTEXT_CAP = 131072   # M3.0 protocol (#173): regular evals cap the context at 128k

# served_name -> AA leaderboard name; a dict value is keyed by reasoning effort.
AA_NAMES: dict[str, Any] = {
    "nemotron-3.5-super-vl": "Nemotron 3.5 Super VL",
    "deepseek-v41-flash-1017": {
        "high": "DeepSeek V4.1 Flash 1017 (High)",
        "max": "DeepSeek V4.1 Flash 1017 (Max)",
        "none": "DeepSeek V4.1 Flash 1017 (Non-reasoning)",
    },
    "qwen38-27b": {
        "xhigh": "Qwen3.8 27B (Xhigh)",
        "medium": "Qwen3.8 27B (Medium)",
        "low": "Qwen3.8 27B (Low)",
        "none": "Qwen3.8 27B (Non-reasoning)",
    },
    "glm-5.3-flash": "GLM-5.3-Flash",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
}


# ----------------------------------------------------------------------------- model resolution


def resolve_model(spec: str) -> dict[str, Any]:
    """`serving:<name>` (endpoints.json written by serve_teacher) or `url:<base_url>@<served_name>`."""
    if spec.startswith("serving:"):
        name = spec.split(":", 1)[1]
        path = serving.path(name, SERVING_ROOT)
        if not path.exists():
            raise SystemExit(f"{path} does not exist: endpoint {name!r} is not up")
        ep = serving.record(name, SERVING_ROOT)
        age = serving.age(ep)
        if age > ENDPOINT_MAX_AGE_S:
            raise SystemExit(f"{path} is {age:.0f}s old (> {ENDPOINT_MAX_AGE_S}s): server is dead")
        return {
            "spec": spec,
            "served_name": ep["served_name"],
            "urls": list(ep["urls"]),
            "request_profile": ep.get("request") or {},
            "endpoint_job_id": ep.get("job_id"),
            "teacher": ep.get("teacher"),
        }
    if spec.startswith("url:"):
        base, _, served = spec[4:].rpartition("@")
        if not base or not served:
            raise SystemExit("url spec must be url:<base_url>@<served_name>")
        return {"spec": spec, "served_name": served, "urls": [base], "request_profile": {}}
    raise SystemExit(f"unknown model spec {spec!r}")


def serving_snapshot(spec: str) -> dict[str, Any] | None:
    """For `serving:<name>`: the engine launch record (argv, GPUs, host, gate), the job record, the GPU
    model, and the teacher spec (path, sha256, text), so config.json pins the serving configuration
    (#30 audit finding 3)."""
    import hashlib

    if not spec.startswith("serving:"):
        return None
    d = serving.registry_root(SERVING_ROOT) / spec.split(":", 1)[1]
    snap: dict[str, Any] = {}
    for name in ("launch.json", "job.json"):
        f = d / name
        if f.exists():
            raw = f.read_bytes()
            snap[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "content": json.loads(raw)}
    try:
        smi = json.loads((d / "preflight.json").read_text())["nvidia_smi"]["stdout"].strip()
        snap["gpus"] = [line.split(", ")[2] for line in smi.splitlines() if line.count(", ") >= 2]
    except Exception:
        pass
    job = (snap.get("job.json") or {}).get("content") or {}
    if job.get("code") and job.get("teacher"):
        hits = list(Path(job["code"]).glob(f"**/teachers/{job['teacher']}.toml"))
        if hits:
            raw = hits[0].read_bytes()
            snap["teacher_spec"] = {"path": str(hits[0]), "sha256": hashlib.sha256(raw).hexdigest(), "text": raw.decode()}
    return snap or None


async def wait_for_endpoint(spec: str, served_name: str, max_wait_s: float) -> dict[str, Any] | None:
    """Re-read a `serving:` endpoint before an infra retry round: it may have been restarted on another
    node (new URL) since the run started. Polls up to `max_wait_s` for a fresh heartbeat. Returns the new
    resolution, or None for `url:` specs or when the endpoint stays dead."""
    if not spec.startswith("serving:"):
        return None
    deadline = time.time() + max_wait_s
    while True:
        try:
            ep = resolve_model(spec)
        except SystemExit as e:
            if time.time() > deadline:
                print(f"[ab] endpoint still down after {max_wait_s:.0f}s: {e}", flush=True)
                return None
            await asyncio.sleep(60)
            continue
        if ep["served_name"] != served_name:
            raise SystemExit(f"{spec} now serves {ep['served_name']!r}, not {served_name!r}; refusing to mix models in one run")
        return ep


def served_max_model_len(url: str, served_name: str) -> int | None:
    """The endpoint's context length from vLLM's /v1/models, for config.json (None if unavailable)."""
    import urllib.request

    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url.rstrip("/") + "/models", timeout=10) as r:
            for m in json.load(r).get("data", []):
                if m.get("id") == served_name:
                    return m.get("max_model_len")
    except Exception:
        pass
    return None


def cutoff_row(r: dict, k: int) -> dict[str, Any]:
    return {"item_id": r["_task_name"], "domain": r["_domain"], "repeat": k, "status": "cutoff",
            "score": 0.0, "turns": 0, "output_tokens": 0, "input_tokens": 0,
            "error": "no result at the --deadline cutoff; scored 0"}


def parse_deadline(spec: str | None) -> float | None:
    """`--deadline`: epoch seconds, local ISO `YYYY-MM-DDTHH:MM`, or local `HH:MM` (the next such time;
    a resumed run reuses the absolute deadline stored in its config.json instead)."""
    if not spec:
        return None
    import datetime as dt

    if "T" in spec:
        return dt.datetime.fromisoformat(spec).timestamp()
    if ":" not in spec:
        return float(spec)
    hh, mm = (int(x) for x in spec.split(":"))
    now = dt.datetime.now()
    t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if t <= now:
        t += dt.timedelta(days=1)
    return t.timestamp()


def build_sampling(profile: dict, cli_json: str | None, use_profile: bool) -> dict[str, Any]:
    """Endpoint request overrides (the model's serving profile), then CLI overrides (null deletes)."""
    sampling: dict[str, Any] = dict(profile.get("overrides") or {}) if use_profile else {}
    if cli_json:
        for k, v in json.loads(cli_json).items():
            if v is None:
                sampling.pop(k, None)
            elif isinstance(v, dict) and isinstance(sampling.get(k), dict):
                sampling[k] = {**sampling[k], **v}
            else:
                sampling[k] = v
    return sampling


def apply_effort(sampling: dict, effort: str | None) -> dict:
    """--reasoning-effort: the effort both top level and in chat_template_kwargs (M3.2: per-model profiles pin their own
    effort; this overrides it, e.g. Qwen3.8 medium on a profile pinned to xhigh)."""
    if effort:
        sampling = dict(sampling)
        sampling["reasoning_effort"] = effort
        sampling["chat_template_kwargs"] = {**(sampling.get("chat_template_kwargs") or {}), "reasoning_effort": effort}
    return sampling


def reasoning_effort_of(sampling: dict) -> str | None:
    eff = sampling.get("reasoning_effort") or (sampling.get("chat_template_kwargs") or {}).get("reasoning_effort")
    ctk = sampling.get("chat_template_kwargs") or {}
    if ctk.get("thinking") is False or ctk.get("enable_thinking") is False:
        return "none"
    return eff


def aa_reference(served_name: str, sampling: dict, override: str | None) -> dict[str, Any] | None:
    name = override
    if name is None:
        entry = AA_NAMES.get(served_name)
        if isinstance(entry, dict):
            entry = entry.get(reasoning_effort_of(sampling) or "")
        name = entry
    if not name or not LEADERBOARD.exists():
        return None
    for m in json.loads(LEADERBOARD.read_text())["models"]:
        if m.get("name") == name:
            v = (m.get("scores") or {}).get("automationbench-aa")
            if v is not None:
                return {"model": name, "score": round(100 * v, 2), "source": str(LEADERBOARD),
                        "note": "AA's private 657-task split; ours is the public 600"}
    return None


# ----------------------------------------------------------------------------- client


def make_client_class():
    from automationbench.clients import RetryingOpenAIChatCompletionsClient

    class ProfiledChatClient(RetryingOpenAIChatCompletionsClient):
        """Upstream retrying client + the endpoint's reasoning-history adapters.

        Upstream replays prior reasoning as `reasoning_content`. Our vLLM builds read it from
        `reasoning` (mirror_reasoning_content), and the Nemotron template wants it inline as
        <think>...</think> (inline_reasoning_history); same adapters as inference/sticky_proxy.py
        on the task-synthesis branch, applied to the served copy only.
        """

        adapters: tuple[str, ...] = ()
        context_window: int | None = None

        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
            """The context cap (M3.0, 128k): max_tokens = the room left under `context_window`, estimated from above
            (the previous response's exact prompt + completion, plus ~1 token per 2.5 characters of the messages
            added since). With no room left the rollout ends as a context overflow (OverlongPromptError ->
            model_error), as it would against a server with that context length."""
            from verifiers.errors import OverlongPromptError

            cap = self.context_window
            state = kwargs.get("state")
            if not cap:
                return await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
            last = state.get("_ctx_last") if state is not None else None
            first = json.dumps(prompt[0], default=str)[:4000] if prompt else ""
            if last and len(prompt) > last[0] and last[1] == first:
                est = last[2] + int(len(json.dumps(prompt[last[0] + 1:], default=str)) / 2.5) + 8 * (len(prompt) - last[0])
            else:
                est = int((len(json.dumps(prompt, default=str)) + len(json.dumps(tools or [], default=str))) / 2.5) + 8 * len(prompt)
            room = cap - est - 16
            if room < 256:
                raise OverlongPromptError(f"prompt estimate {est} leaves {room} tokens under the {cap}-token context cap")
            sa = dict(sampling_args or {})
            cur = sa.pop("max_completion_tokens", None) or sa.get("max_tokens")
            sa["max_tokens"] = min(cur, room) if cur else room
            resp = await super().get_native_response(prompt, model, sa, tools, **kwargs)
            u = getattr(resp, "usage", None)
            if u is not None and state is not None:
                pt, ct = getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0
                if pt > cap:
                    raise OverlongPromptError(f"prompt of {pt} tokens is over the {cap}-token context cap")
                if pt + ct > cap:
                    state["_ctx_overshoot"] = state.get("_ctx_overshoot", 0) + 1
                state["_ctx_last"] = (len(prompt), first, pt + ct)
                state["_ctx_max"] = max(state.get("_ctx_max", 0), pt + ct)
            return resp

        async def to_native_prompt(self, messages):
            native, extra = await super().to_native_prompt(messages)
            for m in native:
                if m.get("role") != "assistant":
                    continue
                if "xml_null_args" in self.adapters and m.get("tool_calls"):
                    # qwen3_coder-style XML templates render each argument with `| string`, so a
                    # JSON-null argument comes back as the Python text `None`; the model then copies
                    # `None` into later calls, which the parser passes through as the *string* "None".
                    # The parser maps the model's literal `null` to JSON null, so replaying the
                    # text "null" re-renders exactly what the model wrote.
                    calls = []
                    for tc in m["tool_calls"]:
                        fn = dict(tc.get("function") or {})
                        try:
                            args = json.loads(fn.get("arguments") or "{}")
                        except (TypeError, ValueError):
                            args = None
                        if isinstance(args, dict) and any(v is None for v in args.values()):
                            fn["arguments"] = json.dumps({k: ("null" if v is None else v) for k, v in args.items()})
                            tc = {**tc, "function": fn}
                        calls.append(tc)
                    m["tool_calls"] = calls
                if "mirror_reasoning_content" in self.adapters:
                    if m.get("reasoning_content") is not None and "reasoning" not in m:
                        m["reasoning"] = m["reasoning_content"]
                if "inline_reasoning_history" in self.adapters:
                    r = m.get("reasoning_content") or m.get("reasoning")
                    if isinstance(r, str) and r.strip():
                        content = m.get("content") or ""
                        prefix = "<think>\n" + r + "</think>"
                        if isinstance(content, list):
                            m["content"] = [{"type": "text", "text": prefix}] + list(content)
                        else:
                            m["content"] = prefix + content
                    m.pop("reasoning_content", None)
                    m.pop("reasoning", None)
            return native, extra

    return ProfiledChatClient


# Per-request HTTP timeout. Long on purpose (#66 audit rule): time spent queued at an admission gate
# must not count against a request, and a long generation from a healthy server is a model outcome
# bounded by max_tokens and the 50-turn cap, not an infra timeout.
REQUEST_TIMEOUT_S = 86400.0


class Limiter:
    """An in-flight cap that can change while the run is going (see `--concurrency-file`)."""

    def __init__(self, n: int):
        self.n, self.active = n, 0
        self.cond = asyncio.Condition()

    async def __aenter__(self):
        async with self.cond:
            await self.cond.wait_for(lambda: self.active < self.n)
            self.active += 1

    async def __aexit__(self, *exc):
        async with self.cond:
            self.active -= 1
            self.cond.notify_all()

    async def set(self, n: int):
        async with self.cond:
            self.n = n
            self.cond.notify_all()


async def watch_concurrency_file(path: Path, limiters: list, default: int):
    """Poll `path` every 30 s; an integer in it is the per-URL in-flight cap."""
    last = default
    while True:
        await asyncio.sleep(30)
        try:
            n = int(path.read_text().strip())
        except Exception:
            continue
        if n > 0 and n != last:
            print(f"[ab] concurrency {last} -> {n} (from {path})", flush=True)
            for lim in limiters:
                await lim.set(n)
            last = n


def make_clients(urls: list[str], adapters: list[str], context_window: int | None = None) -> list[Any]:
    from verifiers.types import ClientConfig

    from urllib.parse import urlparse

    cls = make_client_class()
    cls.adapters = tuple(adapters)
    cls.context_window = context_window
    os.environ.setdefault("AA_AB_DUMMY_KEY", "EMPTY")
    # Model endpoints are cluster-internal: never route them through the shared web proxy.
    hosts = {urlparse(u).hostname for u in urls} - {None}
    for var in ("no_proxy", "NO_PROXY"):
        cur = [h for h in os.environ.get(var, "").split(",") if h]
        os.environ[var] = ",".join(cur + sorted(hosts - set(cur)))
    clients = []
    for u in urls:
        # max_retries=0: the upstream wrapper already retries 40x with backoff (AA: 30).
        cfg = ClientConfig(api_key_var="AA_AB_DUMMY_KEY", api_base_url=u, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        clients.append(cls(cfg))
    return clients


# ----------------------------------------------------------------------------- data


def load_tasks(domains: list[str]) -> list[dict]:
    from automationbench.domains import get_combined_dataset

    rows = get_combined_dataset(domains).to_list()
    for r in rows:
        info = json.loads(r["info"]) if isinstance(r["info"], str) else r["info"]
        r["_task_name"] = info.get("task_name") or r.get("task")
        r["_domain"] = r["_task_name"].split(".")[0]
    names = [r["_task_name"] for r in rows]
    dup = [n for n, c in collections.Counter(names).items() if c > 1]
    if dup:
        raise SystemExit(f"duplicate task names: {dup[:5]}")
    return rows


def stratified(rows: list[dict], limit: int, seed: int | None = None) -> list[dict]:
    """Deterministic subset, round-robin over domains. Without a seed it takes each domain's first tasks
    in upstream file order (the earliest-written ones; rounds 1 used this); with a seed, a seeded random
    draw within each domain."""
    import random

    by_dom: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_dom[r["_domain"]].append(r)
    if seed is not None:
        for d in by_dom:
            random.Random(f"{seed}:{d}").shuffle(by_dom[d])
    order, i = [], 0
    while len(order) < min(limit, len(rows)):
        for d in by_dom:
            if i < len(by_dom[d]):
                order.append(by_dom[d][i])
        i += 1
    return order[:limit]


def contract_digest(rows: list[dict]) -> str:
    from automationbench.task_contract import task_contract_sha256

    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: r["_task_name"]):
        info = json.loads(r["info"]) if isinstance(r["info"], str) else r["info"]
        h.update(task_contract_sha256(example_id=r["example_id"], prompt=r["prompt"], info=info).encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- error classification

_INFRA_NAMES = {
    "APIConnectionError", "APITimeoutError", "InternalServerError", "RateLimitError", "ConnectError",
    "ConnectTimeout", "ReadTimeout", "ReadError", "RemoteProtocolError", "PoolTimeout", "TimeoutError",
    "ConnectionError", "ConnectionResetError", "ConnectionRefusedError", "ServiceUnavailableError",
    "AuthenticationError", "PermissionDeniedError", "NotFoundError", "InfraError",
}


def _chain(e: BaseException | None) -> list[BaseException]:
    out, seen = [], set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def classify_error(err: BaseException) -> str:
    """infra_error when the endpoint (not the model) failed; model_error otherwise.

    Connection/timeout/5xx/429 survive the upstream client's 40 retries only when the server is
    down. 401/403/404 mean a wrong URL or served name. Everything else (400s such as invalid
    tool-call history, empty responses after retries) is attributed to the model, as AA scores
    errored tasks 0.
    """
    chain = _chain(err)
    for e in chain:
        if type(e).__name__ in _INFRA_NAMES:
            return "infra_error"
        status = getattr(e, "status_code", None)
        if isinstance(status, int) and (status >= 500 or status in (401, 403, 404, 408, 429)):
            return "infra_error"
    return "model_error"


def err_text(err: BaseException) -> str:
    return " <- ".join(f"{type(e).__name__}: {str(e)[:400]}" for e in _chain(err))


# ----------------------------------------------------------------------------- one rollout


def _jsonable(x: Any) -> Any:
    if hasattr(x, "model_dump"):
        return _jsonable(x.model_dump(mode="json", exclude_none=True))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


async def run_item(env, row: dict, repeat: int, client, served_name: str, sampling: dict,
                   max_turns: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from automationbench.rubric.registry import AssertionRegistry
    from automationbench.runner import strip_none_values
    from automationbench.schema.world import WorldState

    item_id = row["_task_name"]
    t0 = time.time()
    rec: dict[str, Any] = {"item_id": item_id, "domain": row["_domain"], "repeat": repeat}
    state = None
    try:
        inp = {k: v for k, v in row.items() if not k.startswith("_")}
        state = await env._run_rollout_state(inp, client, served_name, dict(sampling))
        info = state["info"]
        assertions = [strip_none_values(a) for a in info["assertions"]]
        initial = WorldState(**state["initial_state"])
        final = state["world"]
        grading_exc: list[str] = []

        def safe_check(world, which):
            # Upstream's AssertionRegistry.check maps AttributeError/TypeError/KeyError from a handler to
            # False. Any other exception from a handler (e.g. a ValueError parsing model-written state) is
            # treated the same way: the assertion fails. That makes it a model/task outcome, never infra, and
            # it is recorded in `grading_exceptions` (#30 audit finding 6).
            def check(a):
                try:
                    return AssertionRegistry.check(world, a)
                except Exception as e:
                    grading_exc.append(f"{which}:{a.get('type')}: {type(e).__name__}: {str(e)[:200]}")
                    return False
            return check

        aa = aa_task_score(assertions, safe_check(initial, "initial"), safe_check(final, "final"))
        if grading_exc:
            rec["grading_exceptions"] = grading_exc
        metrics = state.get("metrics") or {}
        rec.update(
            final_state_score=aa["score"],
            n_objectives=aa["n_objectives"], n_objectives_met=aa["n_objectives_met"],
            n_guardrails=aa["n_guardrails"], n_guardrails_broken=aa["n_guardrails_broken"],
            task_completed=aa["task_completed"],
            upstream_partial_credit=metrics.get("partial_credit"),
            upstream_strict_pass=metrics.get("task_completed_correctly"),
        )
        err = state.get("error")
        if err is not None:
            rec["status"] = classify_error(err)
            rec["error"] = err_text(err)
        elif state.get("prompt_too_long"):
            rec["status"] = "model_error"
            rec["error"] = "context overflow (OverlongPromptError)"
        else:
            rec["status"] = "ok"
    except Exception as e:  # escaped the rollout (auth/404 from the endpoint) or a harness crash (not an assertion handler)
        rec["status"] = "infra_error"
        rec["error"] = "escaped: " + err_text(e)
        rec["traceback"] = traceback.format_exc()[-4000:]

    # AA: errored tasks score 0. infra_error rows are excluded from the score (score None).
    if rec["status"] == "ok":
        rec["score"] = rec["final_state_score"]
    elif rec["status"] == "model_error":
        rec["score"] = 0.0
    else:
        rec["score"] = None

    traj = (state or {}).get("trajectory") or []
    rec["turns"] = len(traj)
    rec["stop_condition"] = (state or {}).get("stop_condition")
    rec["truncated_turns"] = sum(1 for s in traj if s.get("is_truncated"))
    rec["max_context_tokens"] = (state or {}).get("_ctx_max")
    rec["n_cap_overshoot"] = (state or {}).get("_ctx_overshoot", 0)
    usage = env.get_state_usage(state) if state is not None else None
    rec["input_tokens"] = int(usage["input_tokens"]) if usage else None
    rec["output_tokens"] = int(usage["output_tokens"]) if usage else None
    rec["elapsed_s"] = round(time.time() - t0, 1)
    rec["finished_at"] = time.time()

    transcript = {
        "item_id": item_id,
        "repeat": repeat,
        "status": rec["status"],
        "messages": _jsonable(list((state or {}).get("prompt") or []) + list((state or {}).get("completion") or [])),
        "assertion_results": _jsonable((state or {}).get("_assertion_results")),
        "debug": _jsonable((state or {}).get("_debug")),
        "perf": _jsonable((state or {}).get("_perf")),
    }
    return rec, transcript


# ----------------------------------------------------------------------------- summary


def summarize(rows: list[dict], cfg: dict) -> dict[str, Any]:
    n_infra = sum(r["status"] == "infra_error" for r in rows)
    scored = [r for r in rows if r["status"] != "infra_error"]
    per_item: dict[str, list[float]] = collections.defaultdict(list)
    for r in scored:
        per_item[r["item_id"]].append(r["score"])
    means = [sum(v) / len(v) for v in per_item.values()]
    n = len(means)
    mean = sum(means) / n if n else float("nan")
    sd = math.sqrt(sum((m - mean) ** 2 for m in means) / (n - 1)) if n > 1 else float("nan")
    half = 1.96 * sd / math.sqrt(n) if n > 1 else float("nan")

    def pct(xs):
        xs = list(xs)
        return round(100 * sum(xs) / len(xs), 2) if xs else None

    by_dom = collections.defaultdict(list)
    for r in scored:
        by_dom[r["domain"]].append(r)
    obj_total = sum(r.get("n_objectives") or 0 for r in scored if r["status"] == "ok")
    obj_met = sum(r.get("n_objectives_met") or 0 for r in scored if r["status"] == "ok")
    infra_rate = n_infra / len(rows) if rows else 0.0
    gen = cfg.get("generation") or {}
    return {
        "benchmark": BENCHMARK,
        "metric": METRIC,
        "score": round(100 * mean, 2) if n else None,
        "n_items": n,
        "n_repeats": cfg["repeats"],
        "ci95": [round(100 * (mean - half), 2), round(100 * (mean + half), 2)] if n > 1 else None,
        # M3.2 protocol fields (#175)
        "model": (cfg.get("model") or {}).get("spec"), "served_name": (cfg.get("model") or {}).get("served_name"),
        "reasoning_effort": gen.get("reasoning_effort"),
        "context_window": gen.get("context_window"), "context_cap": gen.get("context_cap"),
        "judge": None, "code_sha": cfg.get("git_sha"), "code_dirty": cfg.get("git_dirty"),
        "n_infra": n_infra,
        "n_context_limit": sum(r["status"] == "model_error" and "context" in (r.get("error") or "") for r in rows),
        "n_cap_overshoot": sum(r.get("n_cap_overshoot") or 0 for r in rows),
        "n_infra_errors": n_infra,
        "n_model_errors": sum(r["status"] == "model_error" for r in rows),
        "n_cutoff": sum(r["status"] == "cutoff" for r in rows),
        "n_items_with_grading_exceptions": sum(bool(r.get("grading_exceptions")) for r in rows),
        "score_finished_only": (lambda f: round(100 * sum(r["score"] for r in f) / len(f), 2) if f else None)(
            [r for r in scored if r["status"] != "cutoff"]),
        "n_rows": len(rows),
        "infra_error_rate": round(infra_rate, 4),
        "valid": infra_rate <= MAX_INFRA_RATE,
        "aa_reference": cfg.get("aa_reference"),
        "secondary": {
            "tasks_completed_pct": pct(bool(r.get("task_completed")) and r["status"] == "ok" for r in scored),
            "zapier_pass_rate_pct": pct((r.get("upstream_strict_pass") or 0) == 1.0 and r["status"] == "ok" for r in scored),
            "upstream_partial_credit_pct": pct((r.get("upstream_partial_credit") or 0) if r["status"] == "ok" else 0 for r in scored),
            "objectives_completed_pct": round(100 * obj_met / obj_total, 2) if obj_total else None,
            "guardrail_violations_per_task": round(sum(r.get("n_guardrails_broken") or 0 for r in scored) / len(scored), 3) if scored else None,
            "tasks_with_violation_pct": pct((r.get("n_guardrails_broken") or 0) > 0 for r in scored),
            "turn_cap_hit_pct": pct(r.get("stop_condition") == "max_turns_reached" for r in scored),
            "by_domain": {d: {"score": pct(r["score"] for r in rs), "n": len(rs)} for d, rs in sorted(by_dom.items())},
            "output_tokens_total": sum(r.get("output_tokens") or 0 for r in rows),
            "input_tokens_total": sum(r.get("input_tokens") or 0 for r in rows),
        },
        "status_counts": dict(collections.Counter(r["status"] for r in rows)),
    }


# ----------------------------------------------------------------------------- main


def git_info() -> dict[str, Any]:
    import pkgutil

    try:  # set by submit.py's frozen zip
        frozen = pkgutil.get_data("aa_evals.automationbench", "FROZEN_FROM.json")
    except OSError:
        frozen = None
    if frozen:
        return json.loads(frozen)
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", "."],
                                    capture_output=True, text=True).stdout.strip())
        return {"git_sha": sha, "git_dirty": dirty}
    except OSError:
        return {"git_sha": None}


def harness_versions() -> dict[str, Any]:
    from importlib.metadata import version

    out = {"automation_bench": version("automation-bench"), "verifiers": version("verifiers"),
           "openai": version("openai"), "python": platform.python_version()}
    commit_file = Path(VENV) / "AUTOMATIONBENCH_COMMIT"
    out["automationbench_commit"] = commit_file.read_text().strip() if commit_file.exists() else None
    return out


async def main_async(a: argparse.Namespace) -> int:
    from automationbench.rubric import create_rubric
    from automationbench.runner import AutomationBenchEnv
    from datasets import Dataset

    model = resolve_model(a.model)
    if a.serving_record and not model["request_profile"]:   # url: spec standing in for a registry entry
        rec = serving.registry_root(SERVING_ROOT) / a.serving_record / "endpoints.json"
        if rec.exists():
            model["request_profile"] = json.loads(rec.read_text()).get("request") or {}
    sampling = apply_effort(build_sampling(model["request_profile"], a.sampling, not a.no_profile), a.reasoning_effort)
    adapters = [] if a.no_profile else list(model["request_profile"].get("adapters") or [])
    adapters += [x for x in EXTRA_ADAPTERS.get(model["served_name"], []) if x not in adapters]
    if not any(x in adapters for x in HISTORY_ADAPTERS):
        adapters.append("mirror_reasoning_content")
    if a.adapters is not None:
        adapters = [x for x in a.adapters.split(",") if x]
    served_len = served_max_model_len(model["urls"][0], model["served_name"])
    # the client-side cap; with --context-window 0 the server's own max_model_len applies, as before M3.2
    context_window = min(a.context_window, served_len or a.context_window) if a.context_window else None
    domains = [d.strip() for d in a.domains.split(",")] if a.domains != "all" else list(PUBLIC_DOMAINS)

    rows = load_tasks(domains)
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",")}
        rows = [r for r in rows if r["_task_name"] in want]
        missing = want - {r["_task_name"] for r in rows}
        if missing:
            raise SystemExit(f"unknown tasks: {sorted(missing)[:5]}")
    if a.limit:
        rows = stratified(rows, a.limit, a.subset_seed)

    run_dir = Path(a.out_root) / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # /data enforces a file-count quota: all transcripts of a run go in one append-only JSONL;
    # each item row points at its line by byte offset (retried items append a new line).
    transcripts_path = run_dir / "transcripts.jsonl"

    versions = harness_versions()
    if versions["automation_bench"] != UPSTREAM_VERSION or versions["automationbench_commit"] not in (None, UPSTREAM_COMMIT):
        raise SystemExit(f"venv harness {versions} does not match pinned {UPSTREAM_VERSION}@{UPSTREAM_COMMIT[:12]}")

    cfg = {
        "benchmark": BENCHMARK,
        "run_dir": str(run_dir),
        **git_info(),
        "model": {k: model[k] for k in ("spec", "served_name", "urls") if k in model},
        "endpoint": {**{k: model.get(k) for k in ("teacher", "endpoint_job_id")},
                     "max_model_len": served_len},
        "note": a.note,
        "serving": serving_snapshot(model["spec"] if not a.serving_record else f"serving:{a.serving_record}"),
        "endpoint_request_profile": model["request_profile"],
        "generation": {"sampling_args": sampling, "adapters": adapters,
                       "max_tokens": sampling.get("max_tokens", "the room left under the context window (estimated per request)"),
                       "context_window": context_window, "context_cap": a.context_window or None,
                       "reasoning_effort": reasoning_effort_of(sampling)},
        "dataset": {"source": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT, "version": UPSTREAM_VERSION,
                    "split": "public", "domains": domains, "n_tasks": len(rows),
                    "task_contract_digest": contract_digest(rows),
                    "subset": ((f"seeded random draw per domain (seed {a.subset_seed})" if a.subset_seed is not None
                                else "first tasks per domain in upstream file order") if a.limit
                               else ("--tasks" if a.tasks else "all"))},
        "harness": {"env": "automationbench.runner.AutomationBenchEnv (verifiers StatefulToolEnv)", "toolset": "api",
                    "max_turns": a.max_turns, "client": f"automationbench.clients.RetryingOpenAIChatCompletionsClient ({a.client_retries} attempts)",
                    **versions},
        "judge": None,
        "scoring": "aa_evals.automationbench.scoring.aa_task_score; model_error -> 0; infra_error excluded",
        "limits": {"limit": a.limit, "tasks": a.tasks, "repeats": a.repeats, "concurrency_per_url": a.concurrency},
        "repeats": a.repeats,
        "aa_reference": aa_reference(model["served_name"], sampling, a.aa_name),
        "host": socket.gethostname(),
        "condor_job": os.environ.get("CONDOR_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cfg_path = run_dir / "config.json"
    old_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    if old_cfg is not None and not a.force:
        old = old_cfg
        for key in ("model.served_name", "generation", "dataset.commit", "harness.max_turns"):
            k1, _, k2 = key.partition(".")
            ov, nv = old.get(k1), cfg.get(k1)
            if k2:
                ov, nv = (ov or {}).get(k2), (nv or {}).get(k2)
            if ov != nv:
                raise SystemExit(f"{cfg_path}: {key} differs from this invocation ({ov!r} vs {nv!r}); use a new --run-id or --force")
        cfg["resumed_from"] = old.get("started_at")
    if old_cfg is not None:
        # Keep a record of every earlier attempt on this run id (endpoint switches, code SHAs, deadlines).
        prev = {k: old_cfg.get(k) for k in ("started_at", "git_sha", "condor_job", "model", "endpoint",
                                             "serving", "deadline", "note", "endpoint_moves") if k in old_cfg}
        cfg["previous_attempts"] = list(old_cfg.get("previous_attempts") or []) + [prev]
    cfg_path.write_text(json.dumps(cfg, indent=1))

    items_path = run_dir / "items.jsonl"
    done: dict[tuple[str, int], dict] = {}
    if items_path.exists():
        for line in items_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["item_id"], r["repeat"])] = r
    keep = {k: r for k, r in done.items() if r["status"] in ("ok", "model_error")}

    env = AutomationBenchEnv(dataset=Dataset.from_list([{k: v for k, v in rows[0].items() if not k.startswith("_")}]),
                             rubric=create_rubric(), max_turns=a.max_turns, toolset="api")
    import automationbench.clients as ab_clients

    ab_clients._RETRY_MAX_ATTEMPTS = a.client_retries
    clients = make_clients(model["urls"], adapters, context_window)
    sems = [Limiter(a.concurrency) for _ in clients]
    if a.concurrency_file:
        asyncio.create_task(watch_concurrency_file(Path(a.concurrency_file), sems, a.concurrency))
    lock = asyncio.Lock()
    todo = [(r, k) for k in range(a.repeats) for r in rows if (r["_task_name"], k) not in keep]
    print(f"[ab] {model['served_name']} x {len(model['urls'])} url(s); {len(rows)} tasks x {a.repeats} repeats; "
          f"{len(keep)} done, {len(todo)} to run; sampling={json.dumps(sampling)} adapters={adapters}", flush=True)

    results = dict(keep)
    t_start = time.time()
    counter = {"n": 0}

    async def one(i: int, row: dict, k: int):
        ci = i % len(clients)
        async with sems[ci]:
            rec, transcript = await run_item(env, row, k, clients[ci], model["served_name"], sampling, a.max_turns)
        async with lock:
            with transcripts_path.open("ab") as f:
                f.seek(0, 2)
                rec["transcript"] = str(transcripts_path)
                rec["transcript_offset"] = f.tell()
                f.write((json.dumps(transcript) + "\n").encode())
            results[(rec["item_id"], k)] = rec
            with items_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            counter["n"] += 1
            sc = "-" if rec["score"] is None else f"{rec['score']:.2f}"
            print(f"[ab] {counter['n']}/{total} {rec['item_id']} r{k} {rec['status']} score={sc} turns={rec['turns']} "
                  f"out_tok={rec['output_tokens']} {rec['elapsed_s']}s"
                  + (f" err={rec.get('error', '')[:200]}" if rec["status"] != "ok" else ""), flush=True)

    # The absolute deadline is stored in config.json and reused on resume, so re-running the same command
    # after the cutoff neither drops the cutoff rows nor rolls `HH:MM` over to the next day.
    deadline = parse_deadline(a.deadline)
    stored = (old_cfg or {}).get("deadline")
    if stored and a.extend_deadline and deadline and deadline > stored:
        print(f"[ab] extending the stored deadline {stored} -> {deadline}", flush=True)
    elif stored and not a.force:
        if deadline and abs(deadline - stored) > 1:
            print(f"[ab] keeping the run's stored deadline {stored} (ignoring --deadline {a.deadline}; use --force to change it)", flush=True)
        deadline = float(stored)
    cfg["request_timeout_s"] = REQUEST_TIMEOUT_S
    if deadline:
        cfg["deadline"] = deadline
        cfg_path.write_text(json.dumps(cfg, indent=1))
        print(f"[ab] cutoff at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))}: unfinished items score 0", flush=True)

    for rnd in range(1 + a.infra_retry_rounds):
        if deadline and time.time() >= deadline:
            break
        if rnd:
            todo = [(r, k) for (iid, k), rec in results.items() if rec["status"] == "infra_error"
                    for r in rows if r["_task_name"] == iid]
            if not todo:
                break
            print(f"[ab] infra retry round {rnd}: {len(todo)} item(s)", flush=True)
            ep = await wait_for_endpoint(model["spec"], model["served_name"], a.endpoint_wait)
            if ep and ep["urls"] != model["urls"]:
                print(f"[ab] endpoint moved: {model['urls']} -> {ep['urls']}", flush=True)
                for c in clients:
                    try:
                        await c.close()
                    except Exception:
                        pass
                model["urls"] = ep["urls"]
                clients[:] = make_clients(model["urls"], adapters, context_window)
                cap = sems[0].n if sems else a.concurrency
                sems[:] = [Limiter(cap) for _ in clients]  # same list object, so the concurrency watcher sees them
                cfg.setdefault("endpoint_moves", []).append({"time": time.time(), "urls": ep["urls"],
                                                             "endpoint_job_id": ep.get("endpoint_job_id")})
                cfg_path.write_text(json.dumps(cfg, indent=1))
        total = len(todo)
        counter["n"] = 0
        tasks = {asyncio.create_task(one(i, r, k)): (r, k) for i, (r, k) in enumerate(todo)}
        _, pending = await asyncio.wait(tasks, timeout=(deadline - time.time()) if deadline else None)
        if pending:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            async with lock:
                with items_path.open("a") as f:
                    for t in pending:
                        r, k = tasks[t]
                        if (r["_task_name"], k) in results and results[(r["_task_name"], k)]["status"] != "infra_error":
                            continue  # finished in the cancellation window
                        rec = cutoff_row(r, k)
                        results[(rec["item_id"], k)] = rec
                        f.write(json.dumps(rec) + "\n")
            print(f"[ab] cutoff: {len(pending)} unfinished item(s) scored 0", flush=True)
            break

    if deadline and time.time() >= deadline:
        # No tails: every item without a row (never started, or a cutoff row from an earlier attempt) is a cutoff.
        missing = [(r, k) for k in range(a.repeats) for r in rows if (r["_task_name"], k) not in results]
        for r, k in missing:
            prev = done.get((r["_task_name"], k))
            # an earlier infra_error stays infra (excluded, counts toward validity); anything else is a cutoff
            results[(r["_task_name"], k)] = prev if prev and prev["status"] == "infra_error" else cutoff_row(r, k)
        if missing:
            print(f"[ab] past the deadline: {len(missing)} item(s) without a result scored 0 (cutoff)", flush=True)

    final_rows = sorted(results.values(), key=lambda r: (r["item_id"], r["repeat"]))
    tmp = items_path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in final_rows))
    tmp.replace(items_path)
    summary = fold_summary(run_dir, summarize(final_rows, cfg))   # M3.4 trace judge: n_flagged
    summary["wall_s"] = round(time.time() - t_start, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    for c in clients:
        try:
            await c.close()
        except Exception:
            pass
    return 0 if summary["valid"] else 3


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="serving:<name> or url:<base_url>@<served_name>")
    p.add_argument("--summarize-only", action="store_true",
                   help="recompute summary.json of an existing run from its items.jsonl (e.g. after the trace judge "
                        "wrote flags.jsonl); no model calls")
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int, default=0, help="stratified subset of N tasks (round-robin over domains)")
    p.add_argument("--repeats", type=int, default=1, help="AA runs 1")
    p.add_argument("--domains", default="all", help="comma list of public domains, or 'all' (the six scored ones)")
    p.add_argument("--tasks", default=None, help="comma list of task names (e.g. sales.multi_hop_lookup)")
    p.add_argument("--max-turns", type=int, default=50, help="AA: 50")
    p.add_argument("--concurrency", type=int, default=32, help="concurrent tasks per endpoint URL")
    p.add_argument("--sampling", default=None, help="JSON merged over the endpoint's request overrides (null deletes a key)")
    p.add_argument("--reasoning-effort", default=None,
                   help="override the profile's effort (top level and chat_template_kwargs), e.g. medium")
    p.add_argument("--context-window", type=int, default=CONTEXT_CAP,
                   help="context cap in tokens (M3.0: 131072; 0 = the served max_model_len, only for the explicit "
                        "Qwen3.8 full-context calibration runs)")
    p.add_argument("--no-profile", action="store_true", help="ignore the endpoint's request profile (overrides+adapters)")
    p.add_argument("--adapters", default=None, help="comma list replacing the profile's adapters")
    p.add_argument("--aa-name", default=None, help="AA leaderboard model name for aa_reference")
    p.add_argument("--infra-retry-rounds", type=int, default=2)
    p.add_argument("--endpoint-wait", type=float, default=1800,
                   help="before each infra retry round, wait up to this many seconds for a serving: endpoint to be live again")
    p.add_argument("--client-retries", type=int, default=40, help="upstream per-call retry attempts (AA: 30)")
    p.add_argument("--out-root", default=str(RUNS_ROOT))
    p.add_argument("--subset-seed", type=int, default=None,
                   help="with --limit: a seeded random draw within each domain instead of the first tasks per domain")
    p.add_argument("--deadline", default=None,
                   help="cutoff (epoch seconds or local HH:MM): unfinished items are cancelled and score 0 (status cutoff)")
    p.add_argument("--concurrency-file", default=None,
                   help="a file holding an integer; the per-URL in-flight cap follows it (polled every 30 s)")
    p.add_argument("--serving-record", default=None,
                   help="registry name to snapshot into config.json when --model is a url: spec (e.g. a gate bypass)")
    p.add_argument("--extend-deadline", action="store_true",
                   help="on resume, replace the run's stored deadline with a later --deadline")
    p.add_argument("--note", default=None, help="free text recorded in config.json (e.g. why this endpoint)")
    p.add_argument("--force", action="store_true", help="reuse a run dir even if its config differs")
    return p.parse_args(argv)


def summarize_only(a: argparse.Namespace) -> int:
    run_dir = Path(a.out_root) / a.run_id
    cfg = json.loads((run_dir / "config.json").read_text())
    cfg.setdefault("run_dir", str(run_dir))
    rows = jsonl.read(run_dir / "items.jsonl", required=True)
    last = {(r["item_id"], r["repeat"]): r for r in rows}
    old = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    summary = fold_summary(run_dir, summarize(sorted(last.values(), key=lambda r: (r["item_id"], r["repeat"])), cfg))
    if "wall_s" in old:
        summary["wall_s"] = old["wall_s"]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0 if summary["valid"] else 3


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.summarize_only:
        return summarize_only(a)
    if not a.model:
        raise SystemExit("--model is required")
    return asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
