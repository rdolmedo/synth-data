#!/usr/bin/env python3
"""Conversation-sticky reverse proxy for vLLM endpoints (OpenAI chat/completions and
Anthropic /v1/messages).

Pins each conversation (keyed by its opening messages, identical on every turn of an
agent rollout) to one vLLM rank so all its turns reuse that rank's prefix cache.  A new
conversation goes to the rank with the lowest KV-cache usage (polled every few seconds);
conversations already past their first turns when the proxy (re)started keep the
legacy hash route, so a restart does not move them off their cache.  Other paths go to
the first backend.

  sticky_proxy.py --port 8001 --backends http://localhost:8101,http://localhost:8102,...

Keys: /chat/completions and /completions use the first two messages (system + first user
turn); /v1/messages (Claude Code et al.) uses the system prompt plus the first user message,
text only (cache_control markers move between turns and are ignored).
/v1/messages/count_tokens follows the same pin.  Responses, streamed (SSE) or not, are
passed through chunk by chunk.

With --cap-tokens N, a request whose rendered prompt has >= N tokens (counted with the
backend's /tokenize or /v1/messages/count_tokens) is not forwarded; the proxy answers with
a final assistant turn carrying CAP_MARK, so the episode ends there and the verifier scores
the state reached within N tokens of context:
  - chat request without tools (terminus-2): a terminus-2 JSON turn that marks the task
    complete (gen/export_sft.py strips these turns and flags the rollout `capped`);
  - chat request with tools (OpenCode, ...): plain text, finish_reason "stop";
  - /v1/messages (Claude Code): a text block, stop_reason "end_turn";
each as SSE when the request asked for a stream.  /v1/messages has its own threshold,
--cap-tokens-messages (default: --cap-tokens).

While the file /tmp/$USER/drain_<port> exists, /metrics reports a full KV cache, so
dispatchers choosing the least-loaded endpoint start no new rollouts here (in-flight
conversations keep being served) -- used to empty a group before restarting it.

Per-rollout Pi proxy options (gen/pi_cont16k.py; one proxy for synthesis and eval since #480, all off by default --
see --help). Eval (aa_evals pi_eval, eval_q35): --sampling-seed, --request-overrides (merged on the per-turn path),
--reasoning-effort, --context-window, --response-dir / --request-events / --dump-dir capture, --kv-gate[-file],
--gate-key, --conversation-id / --gate-client, --dp-rank. Synthesis: --no-load-poll (aliases --disable-load-polling,
--skip-single-backend-load-poll, --single-backend-no-poll; one backend only), --backend-headers JSON (e.g.
X-data-parallel-rank), --request-overrides with --request-overrides-mode update (shallow update of every
chat/completions request), --chat-template-kwargs JSON (merged), --mirror-reasoning-content, --wire-log FILE (every
served request and its raw response in one append-only file, see synth/wire.py). Both: --inline-reasoning-history,
--backend-force-close. Template and capture options apply on the per-turn cap path (--turn-max-tokens), as used by
pi-cont16k, which always passes --turn-continue 0 (no hidden turns).
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import sys
import time

import aiohttp
from aiohttp import web


def conv_key(body: bytes) -> str:
    try:
        msgs = json.loads(body).get("messages") or []
    except ValueError:
        return ""
    parts = []
    for m in msgs[:2]:
        c = m.get("content")
        parts.append(c if isinstance(c, str) else json.dumps(c, sort_keys=True))
    return "\x00".join(p[:4000] for p in parts)


def merge_request_overrides(payload: dict, overrides: dict) -> dict:
    """aa_evals M3.2: a model's serving-registry request settings (teacher [request].overrides, with the run's
    reasoning effort applied) on every sampled request. Top-level keys replace the client's; chat_template_kwargs
    merges key by key (the overrides win)."""
    for k, v in overrides.items():
        if k == 'chat_template_kwargs' and isinstance(v, dict):
            payload['chat_template_kwargs'] = {**(payload.get('chat_template_kwargs') or {}), **v}
        else:
            payload[k] = v
    return payload


def inline_reasoning_history(payload):
    """aa-serving's adapter (inference/sticky_proxy.py there, nemotron35-super's `inline_reasoning_history`): the
    pinned Nemotron template reads reasoning_content, which vLLM renames to reasoning; inline <think> tags survive
    both, so each earlier assistant turn keeps its reasoning in the prompt."""
    for message in payload.get('messages', []):
        if message.get('role') != 'assistant':
            continue
        reasoning = message.get('reasoning_content') or message.get('reasoning')
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        content = message.get('content') or ''
        prefix = '<think>\n' + reasoning + '</think>'
        if isinstance(content, str):
            message['content'] = prefix + content
        elif isinstance(content, list):
            message['content'] = [{'type': 'text', 'text': prefix}] + content
        else:
            raise ValueError('Unsupported assistant content for reasoning preservation')
        message.pop('reasoning_content', None)
        message.pop('reasoning', None)
    kwargs = payload.setdefault('chat_template_kwargs', {})
    kwargs['truncate_history_thinking'] = False
    return payload


# A 502 the per-rollout response-budget guard answers for itself (tb_agents/response_budget_proxy.py
# GUARD_ERROR_HEADER: its accounting failed and stays failed for the session) is final: holding and retrying it for
# --hold seconds only hid it from Pi and hung the session (11-15, #531). Upstream/server 502/503s keep the hold.
GUARD_ERROR_HEADER = "X-Response-Budget-Error"


def guard_error(r) -> bool:
    return bool(r.headers.get(GUARD_ERROR_HEADER))


CAP_MARK = "[harbor-gen context cap reached; episode ended by the generation harness]"
REPO_ROOT = str(Path(__file__).resolve().parents[1])


def append_line(path: str, line: str) -> None:
    """One whole JSONL line per write(2) on an O_APPEND descriptor; a short write continues, and a write that fails
    after part of the line went out ends the fragment with a newline before raising, so the next record is not glued
    onto it (same rule as tb_agents/jsonlog.py, #451; kept here so inference/ needs no tb_agents)."""
    data = (line + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o666)
    try:
        view, sent = memoryview(data), 0
        try:
            while sent < len(data):
                sent += os.write(fd, view[sent:])
        except OSError:
            if sent:
                try:
                    os.write(fd, b"\n")
                except OSError:
                    pass
            raise
    finally:
        os.close(fd)

TURN_CONTINUE = "Thinking exceeded the maximum thinking budget, please continue."
LENGTH_STOP = re.compile(rb'"(?:finish_reason|stop_reason)"\s*:\s*"(?:length|max_tokens)"')


def _sse_events(buf: bytes):
    # LF delimits SSE lines; Unicode separators (NEL, U+2028/9) may occur raw inside JSON strings (task-synthesis
    # 19d41f3d; str.splitlines would split an event).
    for line in buf.decode("utf-8", "replace").split("\n"):
        if line.startswith("data:"):
            d = line[5:].strip()
            if d and d != "[DONE]":
                try:
                    yield json.loads(d)
                except ValueError:
                    pass


def assistant_from_response(buf: bytes, stream: bool, anthropic: bool) -> dict:
    """The (length-truncated) assistant message of a response, as the client would send it back in history.
    Unfinished tool calls are dropped (a cut-off call cannot be executed)."""
    if anthropic:
        blocks = []
        if stream:
            for e in _sse_events(buf):
                t = e.get("type")
                if t == "content_block_start":
                    blocks.append(dict(e.get("content_block") or {}))
                elif t == "content_block_delta" and blocks:
                    d, b = e.get("delta") or {}, blocks[-1]
                    if d.get("type") == "thinking_delta":
                        b["thinking"] = (b.get("thinking") or "") + d.get("thinking", "")
                    elif d.get("type") == "signature_delta":
                        b["signature"] = (b.get("signature") or "") + d.get("signature", "")
                    elif d.get("type") == "text_delta":
                        b["text"] = (b.get("text") or "") + d.get("text", "")
        else:
            try:
                blocks = list(json.loads(buf).get("content") or [])
            except ValueError:
                blocks = []
        keep = [b for b in blocks if b.get("type") in ("thinking", "text")]
        for b in keep:
            if b.get("type") == "thinking":
                b.setdefault("signature", "")
        return {"role": "assistant", "content": keep or [{"type": "text", "text": ""}]}
    content, reasoning, rkey = "", "", "reasoning_content"
    if stream:
        for e in _sse_events(buf):
            for c in e.get("choices") or []:
                d = c.get("delta") or {}
                content += d.get("content") or ""
                for k in ("reasoning_content", "reasoning"):
                    if d.get(k):
                        reasoning += d[k]
                        rkey = k
    else:
        try:
            m = (json.loads(buf).get("choices") or [{}])[0].get("message") or {}
        except ValueError:
            m = {}
        content = m.get("content") or ""
        for k in ("reasoning_content", "reasoning"):
            if m.get(k):
                reasoning, rkey = m[k], k
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg[rkey] = reasoning
    return msg


def continue_payload(payload: dict, assistant: dict) -> dict:
    p = json.loads(json.dumps(payload))
    p.setdefault("messages", []).extend([assistant, {"role": "user", "content": TURN_CONTINUE}])
    return p


def clamp_max_tokens(payload: dict, cap: int, anthropic: bool) -> None:
    keys = ["max_tokens"] if anthropic else ["max_tokens", "max_completion_tokens"]
    present = [k for k in keys if payload.get(k) is not None]
    for k in present:
        payload[k] = min(int(payload[k]), cap)
    if not present:
        payload["max_tokens"] = cap




def cap_response(model: str, n_prompt: int) -> dict:
    content = json.dumps({"analysis": CAP_MARK, "plan": "", "commands": [], "task_complete": True})
    return {"id": f"chatcmpl-cap-{int(time.time() * 1000)}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": 1, "total_tokens": n_prompt + 1}}


def sse(data: dict, event: str | None = None) -> bytes:
    return ((f"event: {event}\n" if event else "") + f"data: {json.dumps(data)}\n\n").encode()


def cap_chat_stream(model: str, n_prompt: int, content: str) -> bytes:
    cid, t = f"chatcmpl-cap-{int(time.time() * 1000)}", int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": t, "model": model}
    out = sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": content},
                                    "logprobs": None, "finish_reason": None}]})
    out += sse({**base, "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}]})
    out += sse({**base, "choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": 1,
                                                 "total_tokens": n_prompt + 1}})
    return out + b"data: [DONE]\n\n"


def cap_message(model: str, n_prompt: int) -> dict:
    return {"id": f"msg_cap{int(time.time() * 1000)}", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": CAP_MARK}], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": n_prompt, "output_tokens": 1}}


def cap_message_stream(model: str, n_prompt: int) -> bytes:
    m = cap_message(model, n_prompt)
    start = {**m, "content": [], "stop_reason": None, "usage": {"input_tokens": n_prompt, "output_tokens": 0}}
    return (sse({"type": "message_start", "message": start}, "message_start")
            + sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                  "content_block_start")
            + sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": CAP_MARK}},
                  "content_block_delta")
            + sse({"type": "content_block_stop", "index": 0}, "content_block_stop")
            + sse({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                   "usage": {"input_tokens": n_prompt, "output_tokens": 1}}, "message_delta")
            + sse({"type": "message_stop"}, "message_stop"))


def msgs_of(body: bytes) -> list:
    try:
        return json.loads(body).get("messages") or []
    except ValueError:
        return []


def key_of(msgs: list, n: int) -> str:
    parts = []
    for m in msgs[:n]:
        c = m.get("content")
        parts.append(c if isinstance(c, str) else json.dumps(c, sort_keys=True))
    return hashlib.md5("\x00".join(p[:4000] for p in parts).encode()).hexdigest()


def _block_text(c) -> str:
    """Anthropic content (str or block list) -> stable text: cache_control and other
    per-turn markers are dropped, only the payload of each block is kept."""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return json.dumps(c, sort_keys=True)
    out = []
    for b in c:
        if not isinstance(b, dict):
            out.append(str(b))
        elif "text" in b:
            out.append(str(b["text"]))
        else:
            out.append(json.dumps({k: v for k, v in b.items() if k != "cache_control"}, sort_keys=True))
    return "\x01".join(out)


def anthropic_key(payload: dict) -> str:
    msgs = payload.get("messages") or []
    first = msgs[0].get("content") if msgs and isinstance(msgs[0], dict) else ""
    s = _block_text(payload.get("system") or "")[:8000] + "\x00" + _block_text(first)[:8000]
    return hashlib.md5(s.encode()).hexdigest()


def anthropic_chars(payload: dict) -> int:
    """Upper-bound-ish size of an Anthropic request in characters."""
    n = len(_block_text(payload.get("system") or ""))
    for m in payload.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    n += sum(len(b.get(f) or "") for f in ("text", "thinking") if isinstance(b.get(f), str))
                    if "input" in b:
                        n += len(json.dumps(b["input"]))
                    if "content" in b:
                        n += len(json.dumps(b["content"]))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--listen-fd", type=int, help="inherit an already bound listening socket")
    ap.add_argument("--backends", required=True)
    ap.add_argument("--hold", type=float, default=1800, help="seconds to wait out an unreachable backend")
    ap.add_argument("--no-load-poll", "--disable-load-polling", "--skip-single-backend-load-poll",
                    "--single-backend-no-poll", dest="no_load_poll", action="store_true",
                    help="no background routing-load polls (exactly one backend); explicit /metrics still works")
    ap.add_argument("--backend-headers", default="", help="JSON object of extra backend headers (e.g. X-data-parallel-rank)")
    ap.add_argument("--cap-tokens", type=int, default=0, help="end episodes whose prompt reaches this many tokens (0 = off)")
    ap.add_argument("--cap-tokens-messages", type=int, default=-1,
                    help="cap for /v1/messages (-1 = same as --cap-tokens, 0 = off)")
    ap.add_argument("--cap-log", default="")
    ap.add_argument("--pins-file", default="", help="persist conversation->backend pins here (survive restarts)")
    ap.add_argument("--pins-override", default="",
                    help="JSON {key: backend} re-read when it changes (every 10 s): moves conversations at their next "
                         "request without a restart (e.g. off a rank whose KV cache is thrashing)")
    ap.add_argument("--shutdown-timeout", type=float, default=600,
                    help="on SIGTERM, seconds in-flight requests may finish after the listener closes")
    ap.add_argument("--turn-max-tokens", type=int, default=0,
                    help="per-turn output cap (reasoning + answer) for chat/completions and /v1/messages; a response "
                         "cut at the cap is never returned while retries remain (0 = off)")
    ap.add_argument("--turn-retries", type=int, default=0, help="plain resamples after a capped response (default: none; 2 = 3 samples total)")
    ap.add_argument("--turn-continue", type=int, default=-1,
                    help="then continuation rounds: the last capped message stays in the conversation, followed by "
                         "the user message TURN_CONTINUE, until a response ends under the cap (-1 = no limit; a hard "
                         "bound of 100 rounds, the context window or the client going away ends it -- only then is a "
                         "capped response returned as is)")
    ap.add_argument("--turn-log", default="", help="JSONL: one line per capped-turn episode (attempts, outcome)")
    ap.add_argument("--req-log", default="", help="append one JSON line per routed request (key, backend, size)")
    ap.add_argument("--dump-dir", default="", help="debug: save the bodies of the first --dump-max routed requests")
    ap.add_argument("--dump-max", type=int, default=200)
    ap.add_argument("--context-window", type=int, default=0,
                    help="reserve output inside this window; does not terminate or compact conversations")
    ap.add_argument("--sampling-seed", type=int, help="per-rollout base seed; increment for each request")
    ap.add_argument("--temperature", type=float, default=None,
                    help="with --sampling-seed and no --request-overrides: the temperature every request gets "
                         "(default 1.0; eval_q35 multi-temperature runs, #167)")
    ap.add_argument("--top-p", type=float, default=None,
                    help="with --sampling-seed and no --request-overrides: top_p every request gets (default 0.95)")
    ap.add_argument("--response-dir", default="", help="durably save native backend response bytes, including interrupted streams")
    ap.add_argument("--request-events", default="", help="lossless request-message deltas and effective generation parameters")
    ap.add_argument("--kv-gate", default="",
                    help="HIGH,LOW: hold new generation requests while the backend's vLLM KV usage is >= HIGH, until it "
                         "is <= LOW (its /metrics, polled at most every 10 s; at most 20 min per request). Running "
                         "requests keep going, so KV never fills from many long contexts growing at once.")
    ap.add_argument("--reasoning-effort", default="max",
                    help="reasoning_effort set on every sampled request (K2-Horizon's vLLM parser accepts high/medium/low)")
    ap.add_argument("--request-overrides", default="",
                    help="JSON: the model's registry request settings (aa_evals M3.2), merged into every sampled "
                         "request after the seed; replaces the fixed T 1.0 / top_p 0.95 / --reasoning-effort / "
                         "clear_thinking defaults, which then apply only without this option")
    ap.add_argument("--request-overrides-mode", choices=("merge", "update"), default="merge",
                    help="merge (aa_evals): on the per-turn path, after the seed, chat_template_kwargs merged key by "
                         "key; update (synthesis, task-synthesis' proxy): a shallow dict update of every "
                         "chat/completions request before routing")
    ap.add_argument("--chat-template-kwargs", default="", help="JSON object merged into chat_template_kwargs of capped requests")
    ap.add_argument("--mirror-reasoning-content", action="store_true",
                    help="copy assistant reasoning_content into `reasoning` when absent (vLLM 0.31 input field); "
                         "--sampling-seed always does")
    ap.add_argument("--wire-log", default="", help="append every served request and its raw (possibly unfinished) "
                    "response to one JSONL file (synth.wire format; one inode per attempt)")
    ap.add_argument("--inline-reasoning-history", action="store_true",
                    help="registry adapter inline_reasoning_history (Nemotron Super): earlier reasoning inlined as <think>")
    ap.add_argument("--backend-force-close", action="store_true",
                    help="new backend connection per request (no keep-alive reuse races after long tool pauses)")
    ap.add_argument("--dp-rank", type=int, default=-1,
                    help="send every request to this vLLM data-parallel rank (X-data-parallel-rank); -1 = vLLM balances")
    ap.add_argument("--conversation-id", default="",
                    help="stable ID for a dedicated rollout proxy, sent as X-Conversation-Id to admission gates")
    ap.add_argument("--gate-client", default="", help="benchmark label sent as X-Gate-Client for fair admission")
    ap.add_argument("--kv-gate-file", default="",
                    help="read the KV usage for --kv-gate from this JSON {t, kv} (written by the serving job) instead "
                         "of polling the backend's /metrics; a reading older than 120 s leaves the gate open")
    ap.add_argument("--gate-key", default="",
                    help="this rollout's key for the eval's context-aware turn gate (eval_q35/gate_server.py, its "
                         "address in --kv-gate-file's gate_url): ask it before every turn, wait while it says no, "
                         "report every finished turn")
    a = ap.parse_args()
    if (a.temperature is not None or a.top_p is not None) and (a.sampling_seed is None or a.request_overrides):
        ap.error("--temperature/--top-p apply only with --sampling-seed and without --request-overrides")
    request_overrides = json.loads(a.request_overrides) if a.request_overrides else {}
    update_overrides = request_overrides if a.request_overrides_mode == "update" else {}
    if update_overrides:
        if a.sampling_seed is not None:
            ap.error("--request-overrides-mode update does not combine with --sampling-seed (the seed belongs in the "
                     "overrides)")
        request_overrides = {}   # applied in handle_request, not on the per-turn path
    template_kwargs = json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else {}
    backend_headers = json.loads(a.backend_headers) if a.backend_headers else {}
    wire_log = None
    if a.wire_log:
        if REPO_ROOT not in sys.path:   # this script runs from inference/
            sys.path.insert(0, REPO_ROOT)
        from synth.wire import WireLog
        wire_log = WireLog(a.wire_log)
    sampling_request_index = 0
    previous_wire_messages = []
    previous_wire_tools = None
    if a.turn_max_tokens < 0 or a.turn_retries < 0 or a.turn_continue < -1:
        ap.error("invalid turn cap/retry/continuation limit")
    cap_msgs = a.cap_tokens if a.cap_tokens_messages < 0 else a.cap_tokens_messages
    backends = [b.rstrip("/") for b in a.backends.split(",")]
    if a.no_load_poll and len(backends) != 1:
        ap.error("--no-load-poll (--disable-load-polling) requires exactly one backend")
    session: dict = {}
    pins: dict[str, str] = {}               # conversation key -> backend
    if a.pins_file and os.path.exists(a.pins_file):
        try:
            pins.update({k: v for k, v in json.load(open(a.pins_file)).items() if v in backends})
        except ValueError:
            pass
    load = {b: 0.0 for b in backends}       # last polled KV usage per backend
    drain_file = f"/tmp/{os.environ.get('USER', 'u')}/drain_{a.port}"
    pins_ver = [0]                          # bumped on every pin change, so moves get saved too

    async def watch_overrides():
        seen = None
        while True:
            try:
                st = os.stat(a.pins_override)
                if (st.st_mtime, st.st_size) != seen:
                    seen = (st.st_mtime, st.st_size)
                    ov = {k: v for k, v in json.load(open(a.pins_override)).items() if v in backends}
                    moved = sum(1 for k, v in ov.items() if pins.get(k) != v)
                    pins.update(ov)
                    pins_ver[0] += 1
                    print(f"[proxy] {time.strftime('%H:%M:%S')} pins override: {moved} moved ({len(ov)} listed)", flush=True)
            except (OSError, ValueError):
                pass
            await asyncio.sleep(10)

    async def poll_load():
        while True:
            for b in backends:
                try:
                    async with session["s"].get(b + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                        txt = await r.text()
                    m = re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\} ([0-9.e+-]+)", txt, re.M)
                    if m:
                        load[b] = float(m.group(1))
                except Exception:  # noqa: BLE001
                    load[b] = 2.0  # unreachable: avoid for new conversations
            await asyncio.sleep(5)

    def pick(key: str, n_msgs: int, legacy: str) -> str:
        if key in pins:
            return pins[key]
        if n_msgs >= 5:  # began before this proxy started: keep its old (hash) rank
            b = backends[int(hashlib.md5(legacy.encode()).hexdigest(), 16) % len(backends)]
        else:
            b = min(backends, key=lambda x: load[x])
            load[b] += 0.01  # spread a burst of new conversations before the next poll
        pins[key] = b
        pins_ver[0] += 1
        return b

    def route(body: bytes) -> str:
        msgs = msgs_of(body)
        return pick(key_of(msgs, 2), len(msgs), conv_key(body))

    async def save_pins():
        last = -1
        while True:
            await asyncio.sleep(30)
            if a.pins_file and pins_ver[0] != last:
                last = pins_ver[0]
                tmp = a.pins_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(pins, f)
                os.replace(tmp, a.pins_file)

    async def on_startup(app):
        session["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=7200),
                                             trust_env=False,
                                             connector=aiohttp.TCPConnector(limit=0, force_close=a.backend_force_close))
        if not a.no_load_poll:
            session["poll"] = asyncio.create_task(poll_load())
        session["save"] = asyncio.create_task(save_pins())
        if a.pins_override:
            session["override"] = asyncio.create_task(watch_overrides())

    async def on_cleanup(app):
        await session["s"].close()

    async def metrics(req: web.Request) -> web.Response:
        """Least-loaded backend's KV usage, so a client choosing between this proxy
        (several ranks) and single ranks sends it a proportional share of new work."""
        if os.path.exists(drain_file):
            return web.Response(text='vllm:kv_cache_usage_perc{engine="proxy"} 1.0\n')
        vals = []
        for b in backends:
            try:
                async with session["s"].get(b + "/metrics", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    txt = await r.text()
                m = re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\} ([0-9.e+-]+)", txt, re.M)
                if m:
                    vals.append(float(m.group(1)))
            except Exception:  # noqa: BLE001
                pass
        kv = min(vals) if vals else 1.0
        return web.Response(text=f'vllm:kv_cache_usage_perc{{engine="proxy"}} {kv}\n')

    capped: set[str] = set()   # conversations already past the cap

    def log_cap(k: str, n: int, n_msgs: int, path: str) -> None:
        if k not in capped and a.cap_log:
            append_line(a.cap_log, json.dumps({"t": time.time(), "port": a.port, "conv": k, "prompt_tokens": n,
                                               "n_messages": n_msgs, "path": path}))
        capped.add(k)

    last_seen = {"messages": None, "tools": None, "tokens": 0}
    reports: set = set()   # turn-gate reports in flight (referenced until done)

    def context_bound(payload: dict) -> int | None:
        """Upper bound on the rendered prompt when the request only appends to the previous one: the previous
        prompt + completion tokens, plus every UTF-8 byte of the new messages (a token is at least one byte)
        and 32 template tokens per message. None when no bound is known (then /tokenize decides)."""
        prev, msgs = last_seen["messages"], payload.get("messages") or []
        # Compressed media bytes do not bound processor-expanded vision tokens.
        # Keep this CLI standalone; use the same text-content rule as Pi's guard.
        for message in msgs:
            content = message.get('content')
            if content is not None and not isinstance(content, str):
                if not isinstance(content, list) or any(
                        not isinstance(block, dict) or block.get('type') not in ('text', 'input_text', 'output_text')
                        for block in content):
                    return None
        if prev is None or payload.get("tools") != last_seen["tools"] or len(msgs) <= len(prev) or msgs[:len(prev)] != prev:
            return None
        new = msgs[len(prev):]
        return last_seen["tokens"] + sum(len(json.dumps(m, ensure_ascii=False).encode()) + 32 for m in new) + 64

    async def n_tokens(b: str, payload: dict) -> int | None:
        req_t = {"model": payload.get("model"), "messages": payload.get("messages"), "add_generation_prompt": True}
        if payload.get("tools"):  # tool schemas are part of the rendered prompt
            req_t["tools"] = payload["tools"]
        if payload.get("chat_template_kwargs"):  # e.g. truncate_history_thinking changes the length
            req_t["chat_template_kwargs"] = payload["chat_template_kwargs"]
        try:
            async with session["s"].post(b + "/tokenize", json=req_t, timeout=aiohttp.ClientTimeout(total=120)) as r:
                if r.status == 200:
                    count = (await r.json()).get("count")
                    if type(count) is int and count >= 0:
                        return count
        except Exception:  # noqa: BLE001
            pass
        return None

    async def over_cap(b: str, body: bytes) -> int | None:
        """Prompt token count if this chat request is at/over the cap, else None."""
        if not a.cap_tokens:
            return None
        try:
            payload = json.loads(body)
        except ValueError:
            return None
        msgs = payload.get("messages") or []
        k2 = key_of(msgs, 2)
        chars = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in msgs
                    if isinstance(m.get("content") or "", str))
        if k2 not in capped and chars < 2 * a.cap_tokens:   # < ~2 chars/token is never reached
            return None
        n = await n_tokens(b, payload)
        if n is not None and n >= a.cap_tokens:
            log_cap(k2, n, len(msgs), "chat")
            return n
        return None

    async def over_cap_messages(b: str, payload: dict, k: str) -> int | None:
        if not cap_msgs:
            return None
        if k not in capped and anthropic_chars(payload) < 2 * cap_msgs:
            return None
        try:
            async with session["s"].post(b + "/v1/messages/count_tokens", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=120)) as r:
                n = int((await r.json()).get("input_tokens")) if r.status == 200 else None
        except Exception:  # noqa: BLE001
            n = None
        if n is not None and n >= cap_msgs:
            log_cap(k, n, len(payload.get("messages") or []), "messages")
            return n
        return None

    dumped = [0]

    def note(path: str, key: str, b: str, n_msgs: int, size: int, stream: bool, body: bytes = b"") -> None:
        if a.dump_dir and dumped[0] < a.dump_max:
            dumped[0] += 1
            os.makedirs(a.dump_dir, exist_ok=True)
            with open(os.path.join(a.dump_dir, f"{dumped[0]:04d}_{key[:8]}_{path.strip('/').replace('/', '_')}.json"), "wb") as f:
                f.write(body)
        if a.req_log:
            append_line(a.req_log, json.dumps({"t": round(time.time(), 2), "path": path, "key": key, "backend": b,
                                               "n_messages": n_msgs, "bytes": size, "stream": stream}))

    async def canned(req: web.Request, stream: bool, body: bytes | dict) -> web.StreamResponse:
        if not stream:
            return web.json_response(body)
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "Cache-Control": "no-cache"})
        await resp.prepare(req)
        await resp.write(body)
        await resp.write_eof()
        return resp

    async def capped_turn(req: web.Request, b: str, payload: dict, headers: dict, anthropic: bool) -> web.StreamResponse:
        """Per-turn output cap: generate with max_tokens <= --turn-max-tokens and never hand a length-truncated
        response to the agent while retries remain: plain resamples first, then continuation rounds in which the
        last capped message stays in the conversation followed by the user message TURN_CONTINUE.
        Responses are buffered; streaming clients get SSE keep-alive comments meanwhile, then the whole stream."""
        nonlocal sampling_request_index, previous_wire_messages, previous_wire_tools
        if a.sampling_seed is not None:
            payload['seed'] = (a.sampling_seed + sampling_request_index) % 2147483647
            if request_overrides:   # the model's own settings (registry); nothing Qwen- or GLM-specific
                merge_request_overrides(payload, request_overrides)
            else:
                payload['temperature'] = 1.0 if a.temperature is None else a.temperature
                payload['top_p'] = 0.95 if a.top_p is None else a.top_p
                payload['reasoning_effort'] = a.reasoning_effort
                payload['chat_template_kwargs'] = {**payload.get('chat_template_kwargs', {}), 'clear_thinking': False}
            # vLLM 0.31 accepts input thinking through `reasoning` and exposes
            # it to GLM's template as reasoning_content. Pi uses the older
            # reasoning_content HTTP field; preserve its truncated turns too.
            for message in payload.get('messages', []):
                if message.get('reasoning_content') is not None and 'reasoning' not in message:
                    message['reasoning'] = message['reasoning_content']
        elif request_overrides:
            merge_request_overrides(payload, request_overrides)
        if template_kwargs:   # --chat-template-kwargs (synthesis teachers' chat_template_kwargs_env)
            payload['chat_template_kwargs'] = {**payload.get('chat_template_kwargs', {}), **template_kwargs}
        if a.mirror_reasoning_content:
            # vLLM 0.31 accepts input thinking through `reasoning` and exposes it to the
            # template as reasoning_content; Pi sends reasoning_content. Keep both.
            for message in payload.get('messages', []):
                if message.get('reasoning_content') is not None and 'reasoning' not in message:
                    message['reasoning'] = message['reasoning_content']
        if a.inline_reasoning_history:   # after the mirror: the adapter drops both API reasoning fields
            inline_reasoning_history(payload)
        sampling_request_index += 1
        cap = a.turn_max_tokens
        bound = context_bound(payload) if a.context_window and not anthropic else None
        if bound is not None and bound + cap + 1024 <= a.context_window:
            pass   # provably fits: no /tokenize render on the (busy) API server
        elif a.context_window and not anthropic:
            count = await n_tokens(b, payload)
            if count is None:
                return web.json_response({'error': {'message': 'Unable to verify context headroom'}}, status=503)
            if count >= a.context_window:
                # Worded so Pi's overflow detection matches ("exceeds the context window", "maximum context length
                # is N tokens", as vLLM's own error): Pi then compacts and retries instead of ending the episode.
                return web.json_response({'error': {'message': f'Prompt exceeds the context window: maximum context '
                                                               f'length is {a.context_window} tokens, prompt has {count}'}},
                                         status=400)
            cap = min(cap, a.context_window - count)
        clamp_max_tokens(payload, cap, anthropic)
        if a.request_events:
            messages = payload.get('messages', [])
            common = 0
            for old, new in zip(previous_wire_messages, messages):
                if old != new:
                    break
                common += 1
            record = {'request_index': sampling_request_index, 'time': time.time(),
                      'messages_delta': {'copy_previous_prefix_messages': common,
                                         'append_messages': messages[common:]},
                      'parameters': {k:v for k,v in payload.items() if k not in ('messages','tools')}}
            if payload.get('tools') != previous_wire_tools:
                record['tools'] = payload.get('tools')
            os.makedirs(os.path.dirname(a.request_events), exist_ok=True)
            append_line(a.request_events, json.dumps(record, ensure_ascii=False))
            previous_wire_messages = messages
            previous_wire_tools = payload.get('tools')
        wire_prefix = None
        if a.response_dir:
            os.makedirs(a.response_dir, exist_ok=True)
            wire_prefix = os.path.join(a.response_dir, f'{sampling_request_index:06d}')
            with open(wire_prefix + '.meta.json', 'w') as f:
                json.dump({'started': time.time(), 'seed': payload.get('seed'),
                           'agent_request': bool(payload.get('tools')), 'stream': bool(payload.get('stream')),
                           'max_tokens': cap, 'context_window': a.context_window}, f)
        stream = bool(payload.get("stream"))
        n_plain = 1 + a.turn_retries
        total = n_plain + (a.turn_continue if a.turn_continue >= 0 else 100)
        chain = payload   # grows by (capped message, TURN_CONTINUE) per continuation round
        st = {"resp": None, "ka": None}
        t0 = time.time()

        async def keepalive():
            while True:
                await asyncio.sleep(15)
                await st["resp"].write(b": keepalive\n\n")

        async def fetch(data: bytes):
            deadline = time.monotonic() + a.hold
            while True:
                wire_id = wire_log.request(data, b, req.path_qs) if wire_log else None
                try:
                    async with session["s"].request("POST", b + req.path_qs, data=data, headers=headers) as r:
                        if r.status in (502, 503) and time.monotonic() < deadline and not guard_error(r):
                            raise aiohttp.ClientConnectionError(f"backend status {r.status}")
                        rh = {k: v for k, v in r.headers.items()
                              if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")}
                        if r.status != 200:
                            error_body = await r.read()
                            if wire_log:
                                wire_log.error(wire_id, r.status, error_body)
                            return r.status, rh, error_body
                        if stream and st["resp"] is None:
                            st["resp"] = web.StreamResponse(status=200, headers=rh)
                            await st["resp"].prepare(req)
                            st["ka"] = asyncio.create_task(keepalive())
                        chunks = []
                        wire = open(wire_prefix + '.response', 'ab', buffering=0) if wire_prefix else None
                        try:
                            async for chunk in r.content.iter_any():
                                if wire:
                                    wire.write(chunk)
                                if st["ka"] is not None and st["ka"].done():   # client went away: stop generating
                                    raise ConnectionResetError("client disconnected")
                                chunks.append(chunk)
                                if wire_log:
                                    wire_log.data(wire_id, chunk)
                        finally:
                            if wire:
                                wire.close()
                        if wire_log:
                            wire_log.end(wire_id, r.status)
                        return 200, rh, b"".join(chunks)
                except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as error:
                    if wire_log:
                        wire_log.error(wire_id, f'transport:{type(error).__name__}', str(error).encode())
                    if time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(10)

        def truncated(buf: bytes) -> bool:
            if stream:
                return bool(LENGTH_STOP.search(buf))
            try:
                j = json.loads(buf)
            except ValueError:
                return False
            if anthropic:
                return j.get("stop_reason") == "max_tokens"
            return any(c.get("finish_reason") == "length" for c in j.get("choices") or [])

        last = None
        n_cut = n_att = 0
        sample_usage = []
        try:
            for i in range(total):
                if i >= n_plain:   # continuation round: keep the capped message, ask the model to continue
                    chain = continue_payload(chain, assistant_from_response(last[1], stream, anthropic))
                p = chain
                if i and p.get("seed") is not None:   # a fixed client seed would repeat the same capped sample
                    p = {**p, "seed": int(p["seed"]) + i}
                n_att += 1
                status, rh, buf = await fetch(json.dumps(p).encode())
                if status != 200:
                    if last is None:   # nothing usable yet: pass the error through unchanged
                        if st["resp"] is None:
                            return web.Response(status=status, body=buf, headers=rh)
                        break
                    break              # a retry failed: fall back to the last (capped) response
                last = (rh, buf)
                try:
                    events = list(_sse_events(buf)) if stream else [json.loads(buf)]
                except ValueError:
                    events = []
                usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
                sample_usage.append(usage)
                if usage.get("prompt_tokens") is not None:
                    last_seen.update(messages=payload.get("messages") or [], tools=payload.get("tools"),
                                     tokens=int(usage["prompt_tokens"]) + int(usage.get("completion_tokens") or 0))
                if not truncated(buf):
                    break
                n_cut += 1
        except ConnectionResetError:
            if st["ka"] is not None:
                st["ka"].cancel()
            return st["resp"] or web.Response(status=499)
        finally:
            if st["ka"] is not None:
                st["ka"].cancel()
                await asyncio.gather(st["ka"], return_exceptions=True)
            if a.gate_key and sample_usage:
                reports.add(t := asyncio.create_task(report_turn(sample_usage[0])))
                t.add_done_callback(reports.discard)
            if n_cut and a.turn_log:
                append_line(a.turn_log, json.dumps({"t": round(t0, 1), "path": req.path, "backend": b,
                                                   "n_messages": len(payload.get("messages") or []),
                                                   "capped_attempts": n_cut, "attempts": n_att,
                                                   "continued": max(0, n_att - n_plain), "sample_usage": sample_usage,
                                                   "outcome": "ok" if last is not None and not truncated(last[1])
                                                   else "gave_up", "secs": round(time.time() - t0, 1)}))
        rh, buf = last if last is not None else ({}, b"")
        if st["resp"] is None:
            return web.Response(status=200, body=buf, headers=rh)
        st["ka"].cancel()
        await asyncio.gather(st["ka"], return_exceptions=True)
        try:
            await st["resp"].write(buf)
            await st["resp"].write_eof()
        except ConnectionResetError:
            pass
        return st["resp"]

    # Model clock: seconds with at least one generation request in flight through this proxy
    # (queueing, prefill, decode, retries).  GET /proxy/model-clock serves it to an agent that
    # charges its time limit for environment time only (tb_agents/harness_agents.py:PiCont16kEnvClock).
    clock = {"busy": 0.0, "depth": 0, "since": 0.0, "requests": 0}
    gate = {"open": True, "kv": 0.0, "at": -1e9, "held": 0, "held_sec": 0.0, "url": None}

    async def backend_kv(b: str) -> float:
        if time.monotonic() - gate["at"] >= 10:
            gate["at"] = time.monotonic()
            if a.kv_gate_file:
                try:
                    with open(a.kv_gate_file) as f:
                        j = json.load(f)
                    gate["kv"] = float(j["kv"]) if time.time() - float(j["t"]) < 120 else 0.0
                    gate["url"] = j.get("gate_url") or gate["url"]   # the turn gate's address (it does not move)
                except (OSError, ValueError, KeyError, TypeError):
                    gate["kv"] = 0.0
                return gate["kv"]
            try:
                async with session["s"].get(b + "/metrics", timeout=aiohttp.ClientTimeout(total=30)) as r:
                    m = re.search(r"^vllm:kv_cache_usage_perc\{[^}]*\} ([0-9.e+-]+)", await r.text(), re.M)
                if m:
                    gate["kv"] = float(m.group(1))
            except Exception:  # noqa: BLE001 -- keep the last reading
                pass
        return gate["kv"]

    async def wait_gate(b: str) -> None:
        high, low = (float(x) for x in a.kv_gate.split(","))
        t0 = time.monotonic()
        held = False
        while time.monotonic() - t0 < 1200:
            kv = await backend_kv(b)
            if gate["open"] and kv >= high:
                gate["open"] = False
            elif not gate["open"] and kv <= low:
                gate["open"] = True
                if held:   # spread the requests released together (they would refill KV at once)
                    await asyncio.sleep(random.uniform(0, 30))
            if gate["open"]:
                break
            held = True
            await asyncio.sleep(5)
        if held:
            gate["held"] += 1
            gate["held_sec"] += time.monotonic() - t0

    def need_estimate(payload: dict) -> int:
        """About the prompt tokens of this request: the previous prompt + completion, plus ~4 bytes per token of the
        messages appended since (the last turn's assistant message is already in that completion)."""
        prev, msgs = last_seen["messages"], payload.get("messages") or []
        size = lambda ms: sum(len(json.dumps(m, ensure_ascii=False).encode()) // 4 + 8 for m in ms)  # noqa: E731
        if prev is not None and last_seen["tokens"] and len(msgs) > len(prev) and msgs[:len(prev)] == prev:
            new = msgs[len(prev):]
            return last_seen["tokens"] + size(new[1:] if new[0].get("role") == "assistant" else new)
        return size(msgs)

    async def wait_turn(payload: dict) -> None:
        """eval_q35 turn gate: ask the serving job's gate server whether to send this turn, every 2 s until it says go.
        No gate address -> go; 10 min without an answer (gate gone) -> go."""
        await backend_kv(backends[0])
        need = need_estimate(payload)
        t0, failing = time.monotonic(), None
        while gate["url"]:
            try:
                async with session["s"].post(gate["url"] + "/turn", json={"key": a.gate_key, "need": need},
                                             timeout=aiohttp.ClientTimeout(total=10)) as r:
                    go = bool((await r.json()).get("go", True))
                failing = None
            except Exception:  # noqa: BLE001
                failing = failing or time.monotonic()
                go = time.monotonic() - failing > 600
            if go:
                break
            await asyncio.sleep(2)
            await backend_kv(backends[0])
        if time.monotonic() - t0 > 2:
            gate["held"] += 1
            gate["held_sec"] += time.monotonic() - t0

    async def report_turn(usage0: dict) -> None:
        """Tell the turn gate this rollout's turn finished: the first sample's prompt and prefix-cache hit, and the
        context the turn ends with (prompt + completion of its last sample)."""
        if not (a.gate_key and gate["url"]) or usage0.get("prompt_tokens") is None:
            return
        p = int(usage0["prompt_tokens"])
        try:
            async with session["s"].post(gate["url"] + "/done", timeout=aiohttp.ClientTimeout(total=10), json={
                    "key": a.gate_key, "prompt": p, "completion": max(0, int(last_seen["tokens"] or 0) - p),
                    "cached": (usage0.get("prompt_tokens_details") or {}).get("cached_tokens")}) as r:
                await r.read()
        except Exception:  # noqa: BLE001 -- the gate learns this context from the next turn's estimate
            pass

    def clock_busy() -> float:
        return clock["busy"] + (time.monotonic() - clock["since"] if clock["depth"] else 0.0)

    async def handle(req: web.Request) -> web.StreamResponse:
        if req.path == "/proxy/model-clock":
            return web.json_response({"model_sec": clock_busy(), "in_flight": clock["depth"],
                                      "requests": clock["requests"], "gate_holds": gate["held"],
                                      "gate_held_sec": round(gate["held_sec"], 1)})
        if req.method != "POST" or not (req.path.endswith("/chat/completions") or req.path.endswith("/completions")
                                        or req.path.endswith("/v1/messages")):
            return await handle_request(req)
        if clock["depth"] == 0:
            clock["since"] = time.monotonic()
        clock["depth"] += 1
        clock["requests"] += 1
        try:
            if a.gate_key and req.path.endswith("/chat/completions"):   # the eval's turn gate (model time too)
                try:
                    payload = json.loads(await req.read())
                except ValueError:
                    payload = {}
                await wait_turn(payload if isinstance(payload, dict) else {})
            if a.kv_gate:   # waiting for KV room is model time, like queueing inside vLLM
                await wait_gate(backends[0])
            return await handle_request(req)
        finally:
            clock["depth"] -= 1
            if clock["depth"] == 0:
                clock["busy"] += time.monotonic() - clock["since"]

    async def handle_request(req: web.Request) -> web.StreamResponse:
        if req.path == "/metrics":
            return await metrics(req)
        body = await req.read()
        if update_overrides and req.path.endswith('/chat/completions'):   # --request-overrides-mode update
            payload = json.loads(body)
            payload.update(update_overrides)
            body = json.dumps(payload, ensure_ascii=False).encode()
        if req.path.endswith("/messages") or req.path.endswith("/messages/count_tokens"):
            try:
                payload = json.loads(body)
            except ValueError:
                payload = {}
            msgs = payload.get("messages") or []
            k = anthropic_key(payload)
            b = pick(k, len(msgs), k)
            stream = bool(payload.get("stream"))
            note(req.path, k, b, len(msgs), len(body), stream, body)
            if req.path.endswith("/messages"):
                n = await over_cap_messages(b, payload, k)
                if n is not None:
                    model = payload.get("model") or ""
                    return await canned(req, stream, cap_message_stream(model, n) if stream else cap_message(model, n))
        elif req.path.endswith("/chat/completions") or req.path.endswith("/completions"):
            b = route(body)
            if a.req_log or a.dump_dir:
                msgs = msgs_of(body)
                try:
                    stream = bool(json.loads(body).get("stream"))
                except ValueError:
                    stream = False
                note(req.path, key_of(msgs, 2), b, len(msgs), len(body), stream, body)
            if req.path.endswith("/chat/completions"):
                n = await over_cap(b, body)
                if n is not None:
                    payload = json.loads(body)
                    model = payload.get("model") or ""
                    if not payload.get("tools"):   # terminus-2 (JSON protocol, no tool calling)
                        return web.json_response(cap_response(model, n))
                    if payload.get("stream"):
                        return await canned(req, True, cap_chat_stream(model, n, CAP_MARK))
                    r = cap_response(model, n)
                    r["choices"][0]["message"]["content"] = CAP_MARK
                    return web.json_response(r)
        else:
            b = backends[0]
        headers = {k: v for k, v in req.headers.items() if k.lower() not in ("host", "content-length")}
        if a.conversation_id:
            headers["X-Conversation-Id"] = a.conversation_id
        if a.gate_client:
            headers["X-Gate-Client"] = a.gate_client
        if a.dp_rank >= 0:  # vLLM data-parallel engines: keep this rollout on one rank (its prefix cache)
            headers["X-data-parallel-rank"] = str(a.dp_rank)
        headers.update(backend_headers)   # --backend-headers (synthesis: X-data-parallel-rank per attempt)
        if a.turn_max_tokens and req.method == "POST" and (req.path.endswith("/chat/completions")
                                                            or req.path.endswith("/v1/messages")):
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                return await capped_turn(req, b, payload, headers, req.path.endswith("/v1/messages"))
        # A backend restart (connection refused, or dropped before the response started) is
        # waited out for up to --hold seconds, so the agent's call just takes longer instead
        # of failing the rollout.  (Non-streaming completions send headers only when done.)
        deadline = time.monotonic() + a.hold
        resp = None
        while True:
            try:
                async with session["s"].request(req.method, b + req.path_qs, data=body, headers=headers) as r:
                    if r.status in (502, 503) and time.monotonic() < deadline and not guard_error(r):
                        raise aiohttp.ClientConnectionError(f"backend status {r.status}")
                    resp = web.StreamResponse(status=r.status, headers={k: v for k, v in r.headers.items()
                                                                        if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")})
                    await resp.prepare(req)
                    try:
                        async for chunk in r.content.iter_any():
                            await resp.write(chunk)
                        await resp.write_eof()
                    except ConnectionResetError:
                        # Client went away (an aborted request, or an SSE client closing right after
                        # its final event): leaving the `async with` closes the backend request,
                        # which aborts any remaining generation in vLLM.
                        pass
                    return resp
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError):
                if resp is not None or time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(10)

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_route("*", "/{tail:.*}", handle)
    listen = {"sock": socket.socket(fileno=a.listen_fd)} if a.listen_fd is not None else {
        "host": "0.0.0.0", "port": a.port}
    web.run_app(app, **listen, access_log=None, shutdown_timeout=a.shutdown_timeout)


if __name__ == "__main__":
    main()
