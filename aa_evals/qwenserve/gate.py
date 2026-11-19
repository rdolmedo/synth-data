"""Adaptive admission gate in front of one vLLM engine (#70): every eval client goes through it, sets a high in-flight
cap, and the gate decides what runs.

  python -m aa_evals.qwenserve.gate --upstream http://127.0.0.1:8000 --port 8001 [--log gate.jsonl]

Transparent reverse proxy: every path and method is forwarded as is, responses (SSE or not) are relayed chunk by
chunk with the upstream status. Only POST /v1/chat/completions and /v1/completions are gated; everything else
(/v1/models, /metrics, /health, /tokenize, ...) passes straight through.

Admission (the data-generation pipeline's kv_fit, synth.admission):
* A conversation's identity (`conversation_key`): the client's X-Conversation-Id header when it sends one; else, from
  its second turn on, its first two messages plus its first model turn (identical on every later turn of an agent, and
  usually different between concurrent repeats of a task, whose first two messages are the same). Clients with
  identical first answers must send distinct conversation headers; content alone cannot distinguish them. A first turn (no model turn
  yet) always starts a new conversation under a fresh key; when it ends, it waits for its second turn, which claims it
  (the oldest finished first turn with the same first two messages) and keeps its context, engine and live-turn
  priority. Until 2026-11-06 the key was the first two messages alone (inference/sticky_proxy.py): repeats 2-5 of a
  task entered as turns of repeat 1's live conversation and skipped admission (M3.3, #176).
  Requests with tools or more than two messages belong to multi-turn conversations; their context stays cached in
  vLLM between turns, so the gate tracks each one's context (block-rounded) while it is idle, until `--idle-ttl`.
* Working set = in-flight KV (vLLM's gauge x capacity) + idle live conversations' contexts. Every tick
  synth.admission.step decides how many NEW conversations may start (room) against target x capacity, hill-climbing
  the target on generated tok/s (`target_search`), with preemptions, the in-flight gauge, vLLM's waiting queue and a
  prefix-reuse collapse as brakes.
* An under-used engine never holds new conversations (2026-11-03, aa-qwen-tbsci: ~20 slow TB-Sci agents' idle
  contexts, 2.6M tokens, held room at 0 against a target the throughput search had driven to its 0.3 floor while
  generation was ~0; 62 new conversations waited on an idle engine). While vLLM's in-flight KV is under
  `underused_kv`, nothing waits inside vLLM and no preemption/KV brake is on, the target cannot sink below its start
  value and new conversations may start up to what the GPU and CPU KV tiers hold together (`tier_fill_max`); the
  decode-growth budget and hard brakes below still apply. An idle conversation is released after --idle-ttl
  seconds without a request (default 300).
* Footprints that track reality (#378; c30-gemma26: 112 "live" conversations held 38M idle tokens against ~4.6M real
  while KV sat at ~1%, and room stayed 0): a conversation whose client leaves mid-turn (a killed attempt) is dropped
  at once; a turn without usage records its prompt as estimated from its text plus what it streamed (not request
  bytes / 3: CritPt's ~780 kB bodies made 324k-token "contexts"), never above the engine's max_model_len; an idle
  conversation counts in the working set only for `idle_count_seconds` (120; c30's inter-turn gap p99 was 37-111 s)
  and stays live (engine, turn priority) until --idle-ttl; and an under-used engine admits at least
  `underused_room_min` new conversations per tick. gate_replay.py replays a gate log against the engine's metrics.
* A turn of a live conversation is admitted before its client's new conversations and does not wait for room (its
  prefix is cached; holding it would waste that KV and stall an agent). Everything is capped at --max-inflight
  (max_num_seqs).
* Fair between clients (2026-11-03: on one server, GDP.pdf got 0/100 in 10 min behind agent turns and Omniscience's
  short answers waited behind 546 queued HLE questions). A client is the `X-Gate-Client` header, else the source
  host. Each client has its own queues and the gate admits round-robin across clients, a live turn first within each.
  A single-turn request costs room in proportion to its expected KV (prompt estimate + the client's mean completion,
  learned from usage) instead of a whole agent's; one whose expected KV is under `short_tokens` takes the short lane,
  which skips room and the decode-growth budget (up to `short_max_inflight` at once) and stops only for the hard
  brakes (in-flight KV high, preemptions, stale metrics, vLLM's waiting queue).
* Decode growth (2026-11-03: HLE-medium single-turn requests grew 40-100k reasoning tokens after admission; KV 0.99,
  107 waiting inside vLLM, preemptions, gen 2.8k -> 1.9k tok/s). Every in-flight request reserves its client's
  expected output minus what it has streamed so far; the expected output is the p90 (`output_quantile`) of the
  client's recent completions together with its in-flight requests' output so far (a lower bound: the first answers
  to complete are the short ones). A new request starts only if its prompt + expected output fits the GPU budget:
  `inflight_kv_max` x the GPU KV, less vLLM's in-flight KV and every reservation (the CPU tier holds idle prefixes,
  not decode). A client with fewer than `output_learn_min` completions gets the `output_tokens_prior` and at most
  `probe_inflight` uncapped single-turn requests in flight. Hard brakes for new requests: vLLM's waiting queue
  non-empty on `waiting_ticks` consecutive ticks, and `preemption_hold_seconds` after any preemption, which also
  shrinks the GPU budget (recovering slowly).
* Backpressure (2026-11-03: 72 new conversations held with room 0 until Pi's 30-min timeout fired in the queue): when a
  client's new conversations stall (none of its requests admitted for a while), a new conversation that has waited
  `reject_after_seconds` with no admission for its client, or that arrives while its client already has
  `reject_queue_new` queued and nothing admitted for `reject_stall_seconds`, gets 503 with `Retry-After` (an
  OpenAI-style `server_overloaded` error), so the client backs off and retries instead of timing out. A slow but
  moving queue, e.g. the opening burst, is not rejected (a retry would only lose its place). Live turns never are.
* Client disconnects propagate (2026-11-03, qa1: after three client jobs were removed, ~700 queued and in-flight
  requests nobody read kept vLLM at KV 1.0 with preemptions): a queued request whose connection closed is dropped at
  the next tick or dispatch, and an in-flight one is cancelled within a second, which closes the upstream connection
  so vLLM aborts it (non-streamed requests included: nothing is written to the client until they end).
* Every response carries `X-Gate: 1` (clients can tell a gate from a bare engine without the registry).
* Queueing is never model time: nothing reaches vLLM before admission, the response carries `X-Gate-Queue-Ms`, and
  --log gets one line per request (enqueue/admit/end times, queue ms, class, status). A client that disconnects while
  queued is dropped from the queue.
GET /gate/status: the live state (target, room, queue lengths, working set, causes). GET /metrics: vLLM's metrics
followed by the gate's (`gate:queue_seconds` histogram by class, `gate:inflight`, `gate:queued`, `gate:target`,
`gate:room`, `gate:requests_total`), so queue p50/p95 come from the same scrape as vLLM's.
"""
import argparse
import asyncio
import collections
import hashlib
import itertools
import json
from pathlib import Path
import queue
import re
import sys
import threading
import time
import urllib.request

import aiohttp
from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inference'))
from synth import admission  # noqa: E402
from replay_bench import scrape  # noqa: E402

GATED = ('/v1/chat/completions', '/v1/completions')
HOP = {'host', 'content-length', 'transfer-encoding', 'content-encoding', 'connection'}
# Upstream connections: drop idle pooled connections before vLLM does (VLLM_HTTP_TIMEOUT_KEEP_ALIVE, 5 s; aiohttp keeps
# them 15 s by default), and resend a request whose pooled connection was reset before any response (2026-11-05:
# 26 of 10,199 Lightning replay requests came back 502 that way: ClientOSError 104, connection reset by peer).
UPSTREAM_KEEPALIVE_S = 3.0
UPSTREAM_RETRIES = 2
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Defaults for an eval server: start at 0.6 of the KV (mixed agentic + QA load), climb on throughput.
# With a CPU KV tier synth's defaults start the target at 1.0 and floor it there (Super's idle agents); eval agents
# are mostly in a model call, so the gate keeps the validated 0.6 start and 0.3 floor and lets only the ceiling grow
# with the tier (x-gate replay, 2026-11-03: 0.6 -> 0.53 held the knee at hit 0.94).
POLICY = {'target_search': 'throughput', 'kv_target_fraction': 0.6, 'kv_target_min': 0.3, 'kv_target_max': 0.95,
          'kv_target_fraction_offload': 0.6, 'kv_target_min_offload': 0.3,
          'agent_tokens_prior': 8000, 'new_agents_per_backend_per_period': 16, 'reuse_min_expected_tokens': 200_000,
          'max_waiting': 8, 'search_settle_seconds': 60, 'search_measure_seconds': 120, 'reuse_window_seconds': 120,
          # fairness between clients: single-turn requests whose expected KV is under short_tokens take the short lane
          'short_tokens': 16384, 'short_max_inflight': None,
          # a short-lane request that streams past short_tokens or runs past short_seconds is demoted out of the lane
          'short_seconds': 120,
          # decode growth: expected output per client, GPU budget for in-flight KV + reservations, hard brakes
          'output_tokens_prior': 32768, 'output_learn_min': 5, 'output_quantile': 0.9, 'probe_inflight': 16,
          'inflight_kv_max': 0.9, 'inflight_kv_min': 0.5, 'inflight_kv_recover_per_s': 0.001,
          'preemption_hold_seconds': 120, 'waiting_ticks': 2,
          # backpressure for new conversations
          'reject_after_seconds': 600, 'reject_queue_new': 64, 'reject_stall_seconds': 60, 'retry_after_seconds': 60,
          # an under-used engine never holds new conversations (see observe): at least underused_room_min per tick
          'underused_kv': 0.3, 'tier_fill_max': 0.9, 'underused_room_min': 2,
          # an idle conversation's context counts in the working set this long after its last request (c30: Pi's
          # inter-turn gap p99 37-111 s on five servers); it stays live (engine, turn priority) until --idle-ttl
          'idle_count_seconds': 120,
          # per-client KV share of the pool's GPU budget (see Gate.share_hold); weights by client name, default 1
          'client_share': False, 'client_weights': {}, 'share_wait_seconds': 10}
UNDERUSE_BLOCKERS = {'stale_metrics', 'unknown_kv_capacity', 'kv_inflight_high', 'preemption'}
BYTES_PER_TOKEN = 3.0   # request JSON bytes per prompt token (Pi bodies: ~2.7-3.3); replaced by usage when present
IMAGE_TOKENS = 1024     # per inline image (its base64 bytes say nothing about its tokens)
KV_BRAKES = {'stale_metrics', 'unknown_kv_capacity', 'kv_inflight_high', 'preemption'}
CLIENT_TTL = 3600.0


def conv_key(payload):
    parts = []
    for m in (payload.get('messages') or [])[:2]:
        c = m.get('content')
        parts.append(c if isinstance(c, str) else json.dumps(c, sort_keys=True))
    if not parts:
        parts = [str(payload.get('prompt'))[:4000]]
    return hashlib.md5('\x00'.join(p[:4000] for p in parts).encode()).hexdigest()


CONV_HEADERS = ('X-Conversation-Id', 'X-Session-Id')


def model_turn_sig(m):
    """The parts of a model turn a client resends unchanged: its text and tool calls (not its reasoning, which some
    clients drop from history)."""
    c = m.get('content')
    calls = [(tc.get('id'), (tc.get('function') or {}).get('name'), (tc.get('function') or {}).get('arguments'))
             for tc in m.get('tool_calls') or []]
    return json.dumps([c, calls], sort_keys=True, default=str)


def conversation_key(payload, headers=None):
    """(key, prefix, first): `prefix` = conv_key (the first two messages). With a conversation header the key is the
    header's value and prefix is None (it must not claim a headerless conversation); else, once the request carries a model turn, prefix + the first model turn's signature; a first turn
    returns key None and first True (the caller gives it a fresh key)."""
    msgs = payload.get('messages') or []
    first_index = next((i for i, m in enumerate(msgs) if m.get('role') == 'assistant'), len(msgs))
    # A user-only opening has no system message: its second message is the answer, not part of the stable prefix.
    prefix = conv_key({**payload, 'messages': msgs[:first_index]})
    for h in CONV_HEADERS:
        v = (headers or {}).get(h)
        if v:
            return 'h:' + hashlib.md5(v.encode()).hexdigest(), None, False
    first = msgs[first_index] if first_index < len(msgs) else None
    if first is None:
        return None, prefix, True
    return hashlib.md5((prefix + '\x00' + model_turn_sig(first)).encode()).hexdigest(), prefix, False


def multi_turn(payload):
    return bool(payload.get('tools')) or len(payload.get('messages') or []) > 2


def prompt_estimate(payload):
    """Prompt tokens, roughly, from the text (images at IMAGE_TOKENS each): a single-turn request's size, and an agent
    turn's context when its response carries no usage (tool-call arguments and resent reasoning included)."""
    chars, images = len(json.dumps(payload.get('tools') or '')), 0
    for m in payload.get('messages') or []:
        if m.get('tool_calls'):
            chars += min(len(json.dumps(m['tool_calls'])), 4_000_000)
        for f in ('reasoning_content', 'reasoning'):
            if isinstance(m.get(f), str):
                chars += len(m[f])
        c = m.get('content')
        for part in ([{'type': 'text', 'text': c}] if isinstance(c, str) else c or []):
            if not isinstance(part, dict):
                continue
            if part.get('type') == 'text':
                chars += len(part.get('text') or '')
            elif part.get('type') in ('image_url', 'input_image', 'image'):
                images += 1
            else:
                chars += min(len(json.dumps(part)), 1_000_000)
    if 'prompt' in payload:
        chars += len(str(payload['prompt']))
    return int(chars / BYTES_PER_TOKEN) + images * IMAGE_TOKENS


class Request:
    __slots__ = ('fut', 'key', 'multi', 'client', 'tokens', 'cap', 'stats', 't', 'http', 'engine', 'cls', 't_admit',
                 'first_of', 'est', 'payload')

    def __init__(self, fut, key, multi, client, tokens, cap, t=None, http=None):
        self.fut, self.key, self.multi, self.client, self.tokens, self.cap = fut, key, multi, client, tokens, cap
        self.t = time.time() if t is None else t
        self.http = http                         # the client's aiohttp request (None in unit tests)
        self.engine = None                       # the Engine it was admitted to
        self.cls, self.t_admit = None, None      # its admission class (a short one may be demoted to 'new')
        self.stats = {'status': None, 'bytes': 0, 'usage': None, 'events': 0}   # events ~ streamed tokens so far
        self.first_of = None                     # a first turn: its conversation's first-two-messages key
        self.est = tokens                        # prompt tokens estimated from the text (a context without usage)
        self.payload = None                      # an agent turn's request: estimated only if its response has no usage

    def gone(self):
        """The client closed its connection."""
        return self.http is not None and gone(self.http)


def gone(req):
    t = req.transport
    return t is None or t.is_closing()


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


class Client:
    def __init__(self, prior=0):
        self.cont, self.new = collections.deque(), collections.deque()
        self.inflight, self.last = 0, time.time()
        self.last_admit = self.last              # its last admission (backpressure acts only on a stalled client)
        self.running = set()                     # its admitted requests
        self.outs, self.out_n = collections.deque(maxlen=200), 0   # completion tokens of its finished requests
        self.out = prior                         # expected completion tokens (set every tick)

    def learn(self, completion):
        self.out_n += 1
        self.outs.append(completion)

    def estimate(self, policy):
        """p90 of recent completions and in-flight outputs so far; the prior until enough completions."""
        q = quantile(list(self.outs) + [r.stats['events'] for r in self.running], policy['output_quantile'])
        self.out = q if self.out_n >= policy['output_learn_min'] else max(q, policy['output_tokens_prior'])
        return self.out

    def learned(self, policy):
        return self.out_n >= policy['output_learn_min']


def cache_geometry(base):
    """(capacity tokens, block tokens, CPU tier tokens) from vLLM's cache_config_info gauge."""
    text = OPENER.open(base + '/metrics', timeout=20).read().decode()
    m = re.search(r'^vllm:cache_config_info\{([^}]*)\}', text, re.M)
    if not m:
        return None, None, 0
    labels = dict(re.findall(r'(\w+)="([^"]*)"', m[1]))
    block = int(labels.get('block_size') or 16)
    size = labels.get('kv_cache_size_tokens') or ''   # vLLM's logged capacity (net of hybrid state blocks)
    blocks = labels.get('num_gpu_blocks') or ''
    capacity = int(size) if size.isdigit() else (int(blocks) * block if blocks.isdigit() else None)
    cpu = labels.get('num_cpu_blocks') or ''
    return capacity or None, block, (int(cpu) * block if cpu.isdigit() else 0)


def max_model_len(base):
    """The served context (vLLM /v1/models max_model_len), or None."""
    data = json.loads(OPENER.open(base + '/v1/models', timeout=20).read().decode())
    lens = [m.get('max_model_len') for m in data.get('data') or [] if isinstance(m.get('max_model_len'), int)]
    return max(lens) if lens else None


OFFLOAD_BLOCKS = re.compile(r'SimpleCPUOffloadWorker \[CPU\]: \d+ tensors, (\d+) CPU blocks')   # as synth.serve


def log_offload_tokens(path, block):
    """CPU KV tier tokens from vLLM's log: its cache_config_info gauge leaves num_cpu_blocks unset (vLLM 0.32)."""
    try:
        with open(path, errors='replace') as f:
            blocks = [int(b) for b in OFFLOAD_BLOCKS.findall(f.read())]
    except OSError:
        return None
    return blocks[-1] * block if blocks and block else None


class Logger:
    """One JSON line per request, written by a thread: a slow file system never stalls the event loop."""
    def __init__(self, path):
        self.q = queue.Queue(maxsize=100_000)
        self.path = path
        if path:
            threading.Thread(target=self.run, daemon=True).start()

    def __call__(self, row):
        if self.path:
            try:
                self.q.put_nowait(row)
            except queue.Full:
                pass

    def run(self):
        while True:
            rows = [self.q.get()]
            while not self.q.empty() and len(rows) < 1000:
                rows.append(self.q.get_nowait())
            try:
                with open(self.path, 'a') as f:
                    f.write(''.join(json.dumps(r) + '\n' for r in rows))
            except OSError:
                pass


class Engine:
    """One vLLM engine behind the gate, with its own admission state: kv_fit room and target, decode-growth budget,
    brakes, in-flight and short-lane counts."""
    def __init__(self, i, upstream, policy, max_inflight, capacity=None, block=None, offload=None, vllm_log=None):
        self.i, self.upstream, self.vllm_log = i, upstream.rstrip('/'), vllm_log
        self.capacity, self.block, self.offload = capacity, block, offload
        self.max_len = None                 # the engine's max_model_len (/v1/models): bounds a conversation's context
        self.max_inflight = max_inflight
        self.short_max = policy['short_max_inflight'] or max(4, max_inflight // 8)
        self.inflight = self.short_inflight = 0
        self.running = set()                # requests admitted to this engine
        self.room = 0                       # new agents' worth of KV that may start now (single-turn: a fraction)
        self.causes = ['stale_metrics']
        self.brakes = []                    # the gate's own hard brakes: vllm_waiting, preemption_hold
        self.gpu_frac = policy['inflight_kv_max']
        self.gpu_free = 0.0                 # GPU KV tokens left for new requests' prompt + expected output
        self.reserved = 0
        self.waiting_ticks, self.hold_until = 0, 0.0
        self.b = {}                         # synth.admission per-backend state
        self.mean = policy['agent_tokens_prior']
        self.prev = None
        self.status = {'started': time.time()}

    def blocked(self):
        return bool(self.brakes or KV_BRAKES.intersection(self.causes))

    def open(self):
        return self.inflight < self.max_inflight


ENGINE_ATTRS = ('upstream', 'capacity', 'block', 'offload', 'max_len', 'vllm_log', 'b', 'mean', 'prev', 'room', 'causes',
                'brakes', 'gpu_frac', 'gpu_free', 'reserved', 'waiting_ticks', 'hold_until', 'short_inflight', 'short_max',
                'max_inflight', 'status')


class Gate:
    """The admission gate over one or more engines (a pool). Client queues, conversations, fairness and the per-client
    KV share are the pool's; room, budget and brakes are each engine's. Attributes of a single engine are also
    reachable on the gate (gate.room is engines[0].room)."""
    BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800, 3600)

    def __init__(self, upstream, *, policy=None, max_inflight=128, idle_ttl=300.0, tick=5.0, capacity=None,
                 block=None, offload=None, log=None, status_path=None, vllm_log=None):
        self.policy = {**POLICY, **(policy or {})}
        ups = [upstream] if isinstance(upstream, str) else list(upstream)
        logs = vllm_log if isinstance(vllm_log, (list, tuple)) else [vllm_log] * len(ups)
        self.engines = [Engine(i, u, self.policy, max_inflight, capacity, block, offload, logs[i])
                        for i, u in enumerate(ups)]
        self.idle_ttl, self.tick = idle_ttl, tick
        self.log = Logger(log)
        self.status_path = status_path
        self.convs = {}                     # key -> {'tokens', 'last', 'inflight', 'multi', 'engine'}
        self.firsts = {}                    # first-two-messages key -> deque of finished first turns' conv keys
        self.seq = itertools.count()        # fresh keys for first turns
        self.clients = {}                   # client id -> Client (its queues), in round-robin order
        self.turn = 0                       # round-robin position
        self.hold = frozenset()             # clients held by the per-client KV share (see share_hold)
        self.protect = frozenset()          # clients under their share it holds the others for
        self.share = {}                     # client -> (usage, share) tokens, last computed
        self.counts = collections.Counter()
        self.hist = {c: [0] * (len(self.BUCKETS) + 1) + [0.0] for c in ('cont', 'new', 'short')}  # buckets, +Inf, sum

    @property
    def inflight(self):
        return sum(e.inflight for e in self.engines)

    @inflight.setter
    def inflight(self, v):
        self.engines[0].inflight = v

    def queued(self, cls):
        return sum(len(getattr(c, cls)) for c in self.clients.values())

    def record_queue(self, cls, seconds):
        h = self.hist[cls]
        for i, le in enumerate(self.BUCKETS):
            if seconds <= le:
                h[i] += 1
        h[len(self.BUCKETS)] += 1
        h[-1] += seconds

    def metrics_text(self):
        out = ['# HELP gate:queue_seconds Time a request waited in the admission gate before reaching vLLM.',
               '# TYPE gate:queue_seconds histogram']
        for cls, h in self.hist.items():
            for i, le in enumerate(self.BUCKETS):
                out.append(f'gate:queue_seconds_bucket{{class="{cls}",le="{le}"}} {h[i]}')
            out.append(f'gate:queue_seconds_bucket{{class="{cls}",le="+Inf"}} {h[len(self.BUCKETS)]}')
            out.append(f'gate:queue_seconds_count{{class="{cls}"}} {h[len(self.BUCKETS)]}')
            out.append(f'gate:queue_seconds_sum{{class="{cls}"}} {h[-1]:.3f}')
        out += ['# TYPE gate:inflight gauge', f'gate:inflight {self.inflight}',
                '# TYPE gate:queued gauge', f'gate:queued{{class="cont"}} {self.queued("cont")}',
                f'gate:queued{{class="new"}} {self.queued("new")}',
                '# TYPE gate:short_inflight gauge', f'gate:short_inflight {sum(e.short_inflight for e in self.engines)}',
                '# TYPE gate:engines gauge', f'gate:engines {len(self.engines)}',
                '# TYPE gate:share_hold gauge', f'gate:share_hold {len(self.hold)}',
                '# TYPE gate:client_queued gauge']
        for name, c in self.clients.items():
            label = re.sub(r'["\\\n]', '_', name)[:100]
            use, share = self.share.get(name, (0, 0))
            out += [f'gate:client_queued{{client="{label}",class="cont"}} {len(c.cont)}',
                    f'gate:client_queued{{client="{label}",class="new"}} {len(c.new)}',
                    f'gate:client_inflight{{client="{label}"}} {c.inflight}',
                    f'gate:client_kv_tokens{{client="{label}"}} {int(use)}',
                    f'gate:client_share_tokens{{client="{label}"}} {int(share)}',
                    f'gate:client_held{{client="{label}"}} {int(name in self.hold)}']
        pooled = len(self.engines) > 1
        for metric, kind in (('reserved_tokens', 'gauge'), ('gpu_free_tokens', 'gauge'),
                             ('gpu_budget_fraction', 'gauge'), ('brake', 'gauge'), ('target', 'gauge'),
                             ('room', 'gauge'), ('engine_inflight', 'gauge')):
            if metric == 'engine_inflight' and not pooled:
                continue
            out.append(f'# TYPE gate:{metric} {kind}')
            for e in self.engines:
                lab = f'engine="{e.i}"' if pooled else ''
                if metric == 'brake':
                    sep = ',' if lab else ''
                    out += [f'gate:brake{{{lab}{sep}cause="{b}"}} {int(b in e.brakes)}'
                            for b in ('vllm_waiting', 'preemption_hold')]
                    continue
                v = {'reserved_tokens': int(e.reserved), 'gpu_free_tokens': int(e.gpu_free),
                     'gpu_budget_fraction': f'{e.gpu_frac:.4f}', 'target': e.status.get('target') or 0,
                     'room': e.room, 'engine_inflight': e.inflight}[metric]
                out.append(f'gate:{metric}{{{lab}}} {v}' if lab else f'gate:{metric} {v}')
        gone = sum(1 for c in self.convs.values() if c.get('gone'))
        out += ['# TYPE gate:live_conversations gauge', f'gate:live_conversations {len(self.convs) - gone}',
                '# TYPE gate:dropped_conversations gauge', f'gate:dropped_conversations {gone}',
                '# TYPE gate:requests_total counter']
        out += [f'gate:requests_total{{outcome="{k}"}} {v}' for k, v in sorted(self.counts.items())]
        return '\n'.join(out) + '\n'

    # -- admission -------------------------------------------------------------------------------------------------
    def expected_out(self, r):
        out = self.clients[r.client].out
        return min(out, r.cap) if r.cap else out

    def expected_tokens(self, r):
        """A request's expected KV growth: its prompt (single-turn; a live turn's context is cached) plus the client's
        expected completion, bounded by max_tokens."""
        return r.tokens + self.expected_out(r)

    def short(self, r):
        return not r.multi and self.expected_tokens(r) <= self.policy['short_tokens']

    def probing(self, r, c):
        """A client still being learned has at most probe_inflight uncapped single-turn requests in flight."""
        return (not r.multi and not c.learned(self.policy) and not (r.cap and r.cap <= self.policy['output_tokens_prior'])
                and sum(not x.multi for x in c.running) >= self.policy['probe_inflight'])

    def fits(self, r, c, e=None, short=False):
        """Hard brakes, the probe limit of a client still being learned, and (not for the short lane, whose few small
        requests are bounded by short_max_inflight) the engine's GPU budget for decode growth."""
        e = e or self.engines[0]
        if e.blocked() or self.probing(r, c):
            return False
        return short or self.expected_tokens(r) <= e.gpu_free or not e.inflight

    def home(self, r):
        """A live turn's engine: where its conversation's prefix is cached."""
        conv = self.convs.get(r.key)
        if conv and conv.get('engine') is not None and conv['engine'] < len(self.engines):
            return self.engines[conv['engine']]
        return max(self.engines, key=lambda e: (e.open(), e.gpu_free))

    def lane(self, c, held=False, protected=False):
        """What `c` may start now: (class, request, engine) or None. Requests whose client left are dropped first."""
        for q in (c.cont, c.new):
            while q and (q[0].fut.done() or q[0].gone()):
                q.popleft().fut.cancel()
        for r in itertools.islice(c.cont, 64):           # a live turn goes to its own engine
            if r.fut.done():
                continue
            e = self.home(r)
            if e.open():
                return 'cont', r, e
        if not c.new:
            return None
        r = c.new[0]
        if self.short(r):
            lanes = [e for e in self.engines if e.short_inflight < e.short_max and e.open()]
            if lanes:
                ok = [e for e in lanes if self.fits(r, c, e, short=True)]
                return ('short', r, max(ok, key=lambda e: e.gpu_free)) if ok else None
        if held:
            return None
        # a client the KV share protects needs only the decode budget, not kv_fit room (ab-long, 2026-11-03: room 0
        # behind idle agents' contexts kept protected HLE requests out as well, and both clients starved)
        ok = [e for e in self.engines if (e.room > 0 or protected) and e.open() and self.fits(r, c, e)]
        return ('new', r, max(ok, key=lambda e: e.gpu_free)) if ok else None

    def usage(self, c):
        """GPU KV a client holds or is expected to grow into: each in-flight request's context (a live turn's is its
        conversation's last known context) plus its expected output (or what it has streamed, if more)."""
        total = 0
        for r in c.running:
            ctx = (self.convs.get(r.key) or {}).get('tokens', 0) if r.multi else r.tokens
            total += ctx + max(self.expected_out(r), r.stats['events'])
        return total

    def queued_tokens(self, c, limit=256):
        """Expected KV of a client's queued requests (a live turn: its conversation's context plus expected output)."""
        total = 0
        for r in itertools.islice(itertools.chain(c.cont, c.new), limit):
            ctx = (self.convs.get(r.key) or {}).get('tokens', 0) if r.multi else r.tokens
            total += ctx + self.expected_out(r)
        return total

    def share_hold(self, now=None):
        """Per-client KV share (#110): the pool's GPU budget is split max-min fair by weight over the active clients'
        demand (in-flight KV + queued requests' expected KV): a client that needs less than an equal split keeps what
        it needs and the rest is split among the others.
        While a client under its share has waited `share_wait_seconds` with a new request that would still fit its
        share, clients at or over their share start no new non-short requests (their live turns and the short lane go
        on), so the KV they free goes to the waiting client instead of to their next new conversation. Work-conserving:
        nobody is held while every waiting client is at its share."""
        p = self.policy
        self.protect = frozenset()
        if not p['client_share']:
            self.share = {}
            return frozenset()
        now = time.time() if now is None else now
        active = {n: c for n, c in self.clients.items() if c.running or c.new or c.cont}
        budget = sum(e.gpu_frac * (e.capacity or 0) for e in self.engines)
        weights = p.get('client_weights') or {}
        w = {n: float(weights.get(n, 1.0)) for n in active}
        use = {n: self.usage(c) for n, c in active.items()}
        demand = {n: use[n] + self.queued_tokens(c) for n, c in active.items()}
        self.share = {n: (use[n], s) for n, s in maxmin(budget, demand, w).items()}
        if len(active) < 2 or not budget or sum(demand.values()) <= budget:
            return frozenset()                   # no contention: everyone's demand fits
        waiting = {n for n, c in active.items()
                   if c.new and not self.short(c.new[0]) and not self.probing(c.new[0], c)
                   and now - c.new[0].t >= p['share_wait_seconds']
                   and self.share[n][0] + self.expected_tokens(c.new[0]) <= self.share[n][1]}
        self.protect = frozenset(waiting)
        if not waiting:
            return frozenset()
        return frozenset(n for n, c in active.items()
                         if n not in waiting and c.new and self.share[n][0] >= self.share[n][1])

    def dispatch(self):
        self.hold = self.share_hold()
        while True:
            names = list(self.clients)
            for i in range(len(names)):
                name = names[(self.turn + i) % len(names)]
                c = self.clients[name]
                pick = self.lane(c, name in self.hold, name in self.protect)
                if pick:
                    self.turn = (self.turn + i + 1) % len(names)
                    break
            else:
                return
            cls, r, e = pick
            if cls == 'cont':
                c.cont.remove(r)
            else:
                c.new.popleft()
            if cls == 'new':
                e.room -= 1 if r.multi else min(1.0, self.expected_tokens(r) / max(e.mean, 1024))
            elif cls == 'short':
                e.short_inflight += 1
            e.gpu_free -= self.expected_tokens(r)
            e.inflight += 1
            e.running.add(r)
            r.engine, r.cls, r.t_admit = e, cls, time.time()
            c.inflight += 1
            c.running.add(r)
            c.last_admit = time.time()
            conv = self.convs.setdefault(r.key, {'tokens': 0, 'last': time.time(), 'inflight': 0, 'multi': r.multi})
            conv['inflight'] += 1
            conv['engine'] = e.i
            if conv.pop('gone', None):              # a resent turn of a conversation whose client had left
                self.counts['conversation_revived'] += 1
            r.fut.set_result(cls)

    def observe(self, snaps, now):
        """One admission tick from each engine's /metrics scrape (None when a scrape failed; a single snapshot for a
        single engine)."""
        snaps = snaps if isinstance(snaps, list) else [snaps]
        for k in [k for k, c in self.convs.items()
                  if not c['inflight'] and (not c['multi'] or now - c['last'] > self.idle_ttl)]:
            del self.convs[k]
        for p in list(self.firsts):         # first turns whose conversation expired unclaimed
            q = collections.deque(k for k in self.firsts[p] if k in self.convs)
            if q:
                self.firsts[p] = q
            else:
                del self.firsts[p]
        for k in [k for k, c in self.clients.items()
                  if not (c.cont or c.new or c.inflight) and now - c.last > CLIENT_TTL]:
            del self.clients[k]
        for c in self.clients.values():     # requests whose client left: dropped (their handler returns)
            for name in ('cont', 'new'):
                q = getattr(c, name)
                if any(r.gone() for r in q):
                    for r in q:
                        if r.gone():
                            r.fut.cancel()
                    setattr(c, name, collections.deque(r for r in q if not r.fut.done()))
        for c in self.clients.values():     # backpressure: the oldest new conversations get a 503 to retry later
            while c.new and (c.new[0].fut.done()
                             or now - max(c.new[0].t, c.last_admit) > self.policy['reject_after_seconds']):
                r = c.new.popleft()
                if not r.fut.done():
                    r.fut.set_result('reject')
        for c in self.clients.values():
            c.estimate(self.policy)
        self.hold = self.share_hold(now)
        for e, snap in zip(self.engines, snaps):
            self.observe_engine(e, snap, now)

    def observe_engine(self, e, snap, now):
        # 2026-11-03 (#110 ab-long): HLE's first completions are its short ones, so its learned p90 let uncapped HLE
        # requests into the short lane, where they grew to tens of thousands of tokens and held every short slot for
        # minutes (16 Omniscience-like requests queued, none in flight). A short-lane request that has streamed past
        # short_tokens or run past short_seconds leaves the lane: its slot frees, its reservation stays.
        for r in e.running:
            if r.cls == 'short' and (r.tokens + r.stats['events'] > self.policy['short_tokens']
                                     or now - (r.t_admit or now) > self.policy['short_seconds']):
                r.cls = 'new'
                e.short_inflight -= 1
                self.counts['short_demoted'] += 1
        # c30 (#378): conversations of finished or killed attempts stayed "live" until --idle-ttl, and their contexts
        # filled the working set (c30-gemma26: 112 live, 38M idle tokens against ~4.6M real, KV ~1%, room 0). Only a
        # conversation in flight or idle for less than idle_count_seconds counts (live and idle tokens); the others
        # keep their engine and turn priority until the TTL.
        convs = [c for c in self.convs.values() if c.get('engine', 0) == e.i]
        count_s = self.policy.get('idle_count_seconds')
        counted = [c for c in convs if c['inflight'] or (not c.get('gone') and (count_s is None
                                                                                 or now - c['last'] <= count_s))]
        live = len(counted)
        e.reserved = sum(max(0, self.expected_out(r) - r.stats['events']) for r in e.running)
        idle = sum(admission.footprint(c['tokens'], e.block or 16) for c in counted if not c['inflight'])
        if snap is None or not e.capacity:
            e.room, causes = 0, ['stale_metrics' if snap is None else 'unknown_kv_capacity']
            e.causes, e.gpu_free = causes, 0.0
            e.status.update(t=now, room=0, causes=causes)
            return
        preempted = e.prev is not None and snap.get('preempt', 0) > e.prev[1].get('preempt', 0)
        if preempted:
            e.hold_until = now + self.policy['preemption_hold_seconds']
            e.gpu_frac = max(self.policy['inflight_kv_min'], e.gpu_frac * 0.9)
        elif now >= e.hold_until and e.prev is not None:
            e.gpu_frac = min(self.policy['inflight_kv_max'],
                             e.gpu_frac + self.policy['inflight_kv_recover_per_s'] * (now - e.prev[0]))
        e.waiting_ticks = e.waiting_ticks + 1 if (snap.get('wait') or 0) > 0 else 0
        e.brakes = (['vllm_waiting'] if e.waiting_ticks >= self.policy['waiting_ticks'] else []) + \
                   (['preemption_hold'] if now < e.hold_until else [])
        e.gpu_free = e.gpu_frac * e.capacity - (snap.get('kv') or 0) * e.capacity - e.reserved
        if e.prev is None:
            e.prev = (now, snap)
            e.room = min(self.policy['new_agents_per_backend_per_period'],
                         int(self.policy['kv_target_fraction'] * e.capacity / e.mean))
            e.causes = []
            return
        t0, s0 = e.prev
        dt = max(now - t0, 1e-3)
        d = {k: snap.get(k, 0) - s0.get(k, 0) for k in ('gen', 'src_compute', 'src_gpu', 'src_cpu', 'preempt')}
        obs = {'kv': snap.get('kv'), 'wait': snap.get('wait'), 'preempted': d['preempt'] > 0, 'stale': False,
               'capacity': e.capacity, 'offload': e.offload or 0, 'own_idle_tokens': idle, 'own_agents': live,
               'other_idle_tokens': 0, 'other_agents': 0,
               'reuse_expected': d['src_compute'] + d['src_gpu'] + d['src_cpu'],
               'reuse_cached': d['src_gpu'] + d['src_cpu'], 'gen_tok_s': d['gen'] / dt}
        if live >= 4:
            e.mean = admission.update_mean(e.mean, admission.working_set(obs) / live, dt,
                                           self.policy['agent_tokens_halflife_seconds']
                                           if 'agent_tokens_halflife_seconds' in self.policy
                                           else admission.DEFAULTS['agent_tokens_halflife_seconds'])
        m = max(e.mean, self.policy['agent_tokens_prior'] * self.policy.get('agent_tokens_floor', 1.0), 1024)
        st = admission.step(e.b, self.policy, now, obs, m)
        room, e.causes = st['room'], st['causes']
        underused = ((snap.get('kv') or 0) < self.policy['underused_kv'] and not (snap.get('wait') or 0)
                     and not obs['preempted'] and not UNDERUSE_BLOCKERS.intersection(st['causes']))
        if underused:
            start = self.policy['kv_target_fraction_offload' if e.offload else 'kv_target_fraction']
            e.b['target'] = max(e.b.get('target', start), start)
            ceiling = self.policy['tier_fill_max'] * (e.capacity + (e.offload or 0))
            free = int((ceiling - admission.working_set(obs)) / m)
            # never 0 (#378): idle contexts the engine is not holding in flight must not stop an idle engine, up to
            # max_num_seqs live conversations (more could never run at once)
            floor = self.policy['underused_room_min'] if live < e.max_inflight else 0
            room = max(room, min(self.policy['new_agents_per_backend_per_period'], max(floor, free)))
        e.room = room
        e.prev = (now, snap)
        e.status.update(t=round(now, 1), gen_tok_s=round(obs['gen_tok_s']), kv=snap.get('kv'),
                        running=snap.get('run'), waiting=snap.get('wait'), inflight=e.inflight,
                        short_inflight=e.short_inflight, live_conversations=live,
                        stale_conversations=sum(1 for c in convs if not c['inflight'] and not c.get('gone')
                                                and count_s is not None and now - c['last'] > count_s),
                        dropped_conversations=sum(1 for c in convs if c.get('gone') and not c['inflight']),
                        idle_tokens=idle, mean_tokens=round(m), capacity=e.capacity, offload=e.offload,
                        reserved_tokens=int(e.reserved), gpu_free_tokens=int(e.gpu_free),
                        gpu_budget_fraction=round(e.gpu_frac, 4), brakes=e.brakes, underused=underused,
                        preemptions=snap.get('preempt'),
                        **{k: st[k] for k in ('target', 'working_set_tokens', 'reuse', 'causes', 'limit')},
                        room=room)

    def client_status(self):
        return {k: {'queued_cont': len(c.cont), 'queued_new': len(c.new), 'inflight': c.inflight,
                    'expected_out': round(c.out), 'completed': c.out_n,
                    'kv_tokens': int(self.share.get(k, (0, 0))[0]), 'share_tokens': int(self.share.get(k, (0, 0))[1]),
                    'held': k in self.hold, 'last_admit_age_s': round(time.time() - c.last_admit)}
                for k, c in self.clients.items()}

    def full_status(self):
        """/gate/status: a single engine's state at the top level (as before pooling), every engine's under
        `engines`, and the pool's queues and clients."""
        top = {**self.engines[0].status} if len(self.engines) == 1 else {
            't': max((e.status.get('t') or 0) for e in self.engines),
            'gen_tok_s': sum(e.status.get('gen_tok_s') or 0 for e in self.engines)}
        return {**top, 'engines': [{'upstream': e.upstream, **e.status} for e in self.engines],
                'inflight': self.inflight, 'short_inflight': sum(e.short_inflight for e in self.engines),
                'queued_cont': self.queued('cont'), 'queued_new': self.queued('new'),
                'live_conversations': sum(1 for c in self.convs.values() if not c.get('gone')),
                'dropped_conversations': sum(1 for c in self.convs.values() if c.get('gone')),
                'clients': self.client_status(), 'held': sorted(self.hold),
                'counts': dict(self.counts)}

    async def geometry(self, e):
        try:
            if not e.capacity:
                e.capacity, e.block, offload = await asyncio.to_thread(cache_geometry, e.upstream)
                e.offload = e.offload if e.offload is not None else offload
            if e.max_len is None and e.capacity:   # until it answers (0: no max_model_len given, no cap)
                e.max_len = await asyncio.to_thread(max_model_len, e.upstream) or 0
        except Exception:  # noqa: BLE001  (vLLM not up yet)
            pass
        if e.capacity and not e.offload and e.vllm_log:
            e.offload = await asyncio.to_thread(log_offload_tokens, e.vllm_log, e.block) or e.offload

    async def ticker(self):
        while True:
            await asyncio.gather(*[self.geometry(e) for e in self.engines])
            snaps = await asyncio.gather(*[asyncio.to_thread(scrape, e.upstream) for e in self.engines])
            self.observe(list(snaps), time.time())
            self.dispatch()
            if self.status_path:
                await asyncio.to_thread(self.write_status)
            await asyncio.sleep(self.tick)

    def write_status(self):
        try:
            tmp = Path(str(self.status_path) + '.tmp')
            tmp.write_text(json.dumps(self.full_status()))
            tmp.replace(self.status_path)
        except OSError:
            pass

    # -- proxy -----------------------------------------------------------------------------------------------------
    async def engine_metrics(self):
        """vLLM's /metrics of every engine; with several, each sample gets a `replica` label (vLLM's samples already
        carry an `engine` label of their own; HELP/TYPE once)."""
        async def get(e):
            try:
                async with self.session.get(e.upstream + '/metrics') as r:
                    return await r.text()
            except (aiohttp.ClientError, asyncio.TimeoutError):
                return ''
        texts = await asyncio.gather(*[get(e) for e in self.engines])
        if len(texts) == 1:
            return texts[0]
        return ''.join(label_engine(t, e.i, comments=e.i == 0) for t, e in zip(texts, self.engines))

    async def handle(self, req):
        if req.path == '/gate/status':
            return web.json_response(self.full_status(), headers={'X-Gate': '1'})
        body = await req.read()
        headers = {k: v for k, v in req.headers.items() if k.lower() not in HOP}
        if req.method == 'GET' and req.path == '/metrics':
            text = await self.engine_metrics()
            return web.Response(text=text + self.metrics_text(), content_type='text/plain', headers={'X-Gate': '1'})
        if req.method != 'POST' or req.path not in GATED:
            return await self.relay(req, body, headers, {})
        try:
            payload = json.loads(body)
        except ValueError:
            return await self.relay(req, body, headers, {})
        multi = multi_turn(payload)
        key, prefix, first = conversation_key(payload, req.headers)
        if first:
            key = f'{prefix}#{next(self.seq)}'
        elif prefix is not None and key not in self.convs:
            self.claim(prefix, key)
        name = req.headers.get('X-Gate-Client') or req.remote or '?'
        client = self.clients.get(name) or self.clients.setdefault(name, Client(self.policy['output_tokens_prior']))
        client.last = enq = time.time()
        cap = payload.get('max_completion_tokens') or payload.get('max_tokens')
        est = None if multi else prompt_estimate(payload)
        r = Request(asyncio.get_running_loop().create_future(), key, multi, name,
                    0 if multi else est, cap if isinstance(cap, int) and cap > 0 else None, http=req)
        r.first_of = prefix if first and multi else None
        r.est, r.payload = est, payload if multi else None
        fut = r.fut
        live = key in self.convs and self.convs[key]['multi']
        if live:
            self.convs[key]['last'] = enq          # a queued turn keeps its conversation (and engine) alive
        if not live and len(client.new) >= self.policy['reject_queue_new'] \
                and enq - client.last_admit > self.policy['reject_stall_seconds']:
            return self.reject(name, key, enq)
        (client.cont if live else client.new).append(r)
        self.dispatch()
        try:
            cls = await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled() and fut.result() != 'reject':   # admitted as the client left
                self.release(r, client, fut.result(), len(body))
            dropped = fut.cancelled() and r.gone()    # we dropped it: the client had left
            fut.cancel()
            self.counts['left_queued'] += 1
            if live:
                self.forget(key)                     # its attempt is gone (#378)
            self.log({'t_enq': round(enq, 3), 't_end': round(time.time(), 3), 'key': key[:12], 'client': name,
                      'status': 'left_queued'})
            if dropped:
                return web.Response(status=499)
            raise
        if cls == 'reject':
            return self.reject(name, key, enq)
        admitted = time.time()
        queue_ms = round((admitted - enq) * 1000)
        self.counts[cls] += 1
        self.record_queue(cls, admitted - enq)
        stats = r.stats
        task = asyncio.ensure_future(self.relay(req, body, headers, {'X-Gate-Queue-Ms': str(queue_ms)}, stats,
                                                r.engine.upstream))
        watch = asyncio.ensure_future(self.watch(req, task))
        try:
            return await task
        except asyncio.CancelledError:
            if not watch.done():
                # the client disconnected (aiohttp cancels the handler, which cancels the relay): no usage comes,
                # unless the relay had already finished
                if not (task.done() and not task.cancelled() and task.exception() is None):
                    stats['status'] = 'client_gone'
                raise
            stats['status'] = 'client_gone'          # the watcher cancelled the upstream request
            self.counts['client_gone'] += 1
            return web.Response(status=499)
        finally:
            watch.cancel()
            self.release(r, client, cls, len(body))
            self.log({'t_enq': round(enq, 3), 't_admit': round(admitted, 3), 't_end': round(time.time(), 3),
                      'queue_ms': queue_ms, 'class': cls, 'client': name, 'key': key[:12], 'multi': multi,
                      'engine': r.engine.i, 'status': stats['status'],
                      'req_bytes': len(body), 'resp_bytes': stats['bytes'],
                      'total_tokens': (stats['usage'] or {}).get('total_tokens'),
                      'completion_tokens': (stats['usage'] or {}).get('completion_tokens')})

    def claim(self, prefix, key):
        """A second turn (`key` not yet known): take over the oldest finished first turn with the same first two
        messages, which becomes this conversation (its context, engine and liveness). Any finished first turn of the task
        serves: they share the same prompt and differ only in their first answer."""
        q = self.firsts.get(prefix)
        while q:
            old = q.popleft()
            c = self.convs.get(old)
            if c is not None and not c['inflight'] and not c.get('gone'):
                self.convs[key] = self.convs.pop(old)
                break
        if q is not None and not q:
            del self.firsts[prefix]

    def forget(self, key):
        """An idle conversation whose client left: its context no longer counts (`gone`), but its key stays
        until the TTL, so a client that timed out and resends the turn gets it back as a live turn (and never claims
        another repeat's first turn)."""
        c = self.convs.get(key)
        if c is not None and not c['inflight'] and not c.get('gone'):
            c['gone'] = True                    # out of the working set (observe_engine); its context is kept
            self.counts['conversation_dropped'] += 1

    def reject(self, name, key, enq):
        self.counts['rejected'] += 1
        wait = self.policy['retry_after_seconds']
        self.log({'t_enq': round(enq, 3), 't_end': round(time.time(), 3), 'key': key[:12], 'client': name,
                  'status': 503, 'class': 'reject'})
        return web.json_response(
            {'error': {'message': f'gate: server at capacity for new conversations; retry after {wait} s',
                       'type': 'server_overloaded', 'code': 503}},
            status=503, headers={'Retry-After': str(wait), 'X-Gate': '1',
                                 'X-Gate-Queue-Ms': str(round((time.time() - enq) * 1000))})

    @staticmethod
    async def watch(req, task, every=1.0):
        """Cancel an in-flight relay when its client leaves: the upstream connection closes and vLLM aborts."""
        while not task.done():
            await asyncio.sleep(every)
            if gone(req) and not task.done():
                task.cancel()
                return

    def release(self, r, client, cls, body_bytes):
        """A request admitted as `cls` has ended: free its slot, record its context and completion, admit the next."""
        stats = r.stats
        e = r.engine or self.engines[0]
        client.running.discard(r)
        e.running.discard(r)
        c = self.convs.get(r.key)
        if c:
            c['inflight'] -= 1
            c['last'] = time.time()
            usage = stats['usage'] or {}
            # Without usage (a cut stream), the prompt as estimated from its text plus what streamed: c30 (#378) used
            # request bytes / 3, and CritPt's ~780 kB bodies (images, escaped JSON) made ~324k-token "contexts"
            tokens = usage.get('total_tokens')
            if not tokens:
                est = r.est if r.est is not None else prompt_estimate(r.payload) if r.payload is not None else None
                tokens = (est if est is not None else int(body_bytes / BYTES_PER_TOKEN)) + stats.get('events', 0)
            r.payload = None
            c['tokens'] = min(tokens, e.max_len) if e.max_len else tokens
            if stats['status'] in ('client_gone', 'client_reset') and c['multi'] and not c['inflight']:
                # its client left mid-turn: the attempt was killed or gave up, so its conversation is gone (c30:
                # killed workers' conversations held the gate for the TTL). A retry of the turn enters as new.
                self.forget(r.key)
            elif r.first_of and c['multi'] and not c['inflight'] and stats['status'] == 200:
                self.firsts.setdefault(r.first_of, collections.deque()).append(r.key)   # awaits its second turn
        e.inflight -= 1
        client.inflight -= 1
        client.last = time.time()
        if (r.cls or cls) == 'short':
            e.short_inflight -= 1
        completion = (stats['usage'] or {}).get('completion_tokens')
        if stats['status'] == 200 and isinstance(completion, int):
            client.learn(completion)
        self.dispatch()

    async def relay(self, req, body, headers, extra, stats=None, upstream=None):
        stats = stats if stats is not None else {}
        upstream = upstream or self.engines[0].upstream
        for attempt in range(UPSTREAM_RETRIES + 1):
            responded = False
            try:
                async with self.session.request(req.method, upstream + req.path_qs, data=body, headers=headers) as r:
                    responded = True
                    resp = web.StreamResponse(status=r.status, headers={**{k: v for k, v in r.headers.items()
                                                                          if k.lower() not in HOP}, **extra, 'X-Gate': '1'})
                    stats['status'] = r.status
                    await resp.prepare(req)
                    tail = b''
                    try:
                        async for chunk in r.content.iter_any():
                            await resp.write(chunk)
                            stats['bytes'] = stats.get('bytes', 0) + len(chunk)
                            stats['events'] = stats.get('events', 0) + chunk.count(b'\ndata: ') + chunk.startswith(b'data: ')
                            tail = (tail + chunk)[-4096:]
                        await resp.write_eof()
                    except ConnectionResetError:   # client left: leaving the context aborts the vLLM request
                        stats['status'] = 'client_reset'
                        return resp
                    stats['usage'] = last_usage(tail)
                    return resp
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                # A pooled connection the engine closed while idle (vLLM's HTTP keep-alive is 5 s) fails before any
                # response with a reset or a disconnect: the engine never took the request, so send it again on a
                # fresh connection. Anything after the response started cannot be replayed.
                if not responded and attempt < UPSTREAM_RETRIES and isinstance(
                        exc, (aiohttp.ClientOSError, aiohttp.ServerDisconnectedError)):
                    stats['upstream_retries'] = stats.get('upstream_retries', 0) + 1
                    continue
                stats['status'] = 'upstream_error'
                return web.json_response({'error': {'message': f'gate: upstream unavailable: {exc!r}'[:500],
                                                     'type': 'upstream_unavailable'}}, status=502,
                                         headers={**extra, 'X-Gate': '1'})

    def app(self):
        async def startup(app):
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
                                                 connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=UPSTREAM_KEEPALIVE_S),
                                                 trust_env=False)
            app['ticker'] = asyncio.create_task(self.ticker())

        async def cleanup(app):
            app['ticker'].cancel()
            await self.session.close()
        app = web.Application(client_max_size=1 << 31)
        app.on_startup.append(startup)
        app.on_cleanup.append(cleanup)
        app.router.add_route('*', '/{tail:.*}', self.handle)
        return app


def maxmin(budget, demand, weight):
    """Weighted max-min fair split of `budget` over `demand` (water-filling): {client: share}."""
    share, left, todo = {}, float(budget), dict(demand)
    while todo:
        wsum = sum(weight[n] for n in todo) or 1.0
        level = left / wsum                      # budget per unit weight at this round
        small = {n: d for n, d in todo.items() if d <= level * weight[n]}
        if not small:
            share.update({n: level * weight[n] for n in todo})
            break
        for n, d in small.items():
            share[n] = d
            left -= d
            del todo[n]
    return share


def _delegate(name):
    return property(lambda self: getattr(self.engines[0], name), lambda self, v: setattr(self.engines[0], name, v))


for _name in ENGINE_ATTRS:
    setattr(Gate, _name, _delegate(_name))

SAMPLE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{([^}]*)\})?(\s.*)$')


def label_engine(text, i, comments=True):
    """Prometheus text with a replica="i" label on every sample (HELP/TYPE lines kept only if `comments`)."""
    out = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            if comments and line:
                out.append(line)
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        labels = f'replica="{i}"' + (',' + m[3] if m[3] else '')
        out.append(f'{m[1]}{{{labels}}}{m[4]}')
    return '\n'.join(out) + '\n'


def last_usage(tail):
    """The usage object of a response's end (non-streamed JSON, or a stream's include_usage chunk), else None."""
    i = tail.rfind(b'"usage"')
    if i < 0:
        return None
    m = re.search(rb'"usage":\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', tail[i:])
    try:
        return json.loads(m[1]) if m else None
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--upstream', required=True,
                    help='vLLM base, e.g. http://127.0.0.1:8000 (no /v1); several, comma-separated, for a pool')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--max-inflight', type=int, default=128, help="vLLM's max_num_seqs (per engine)")
    ap.add_argument('--idle-ttl', type=float, default=300.0)
    ap.add_argument('--tick', type=float, default=5.0)
    ap.add_argument('--policy', default='{}', help='synth.admission policy overrides (JSON)')
    ap.add_argument('--log', default='', help='per-request JSON lines')
    ap.add_argument('--status', default='', help='status JSON rewritten every tick')
    ap.add_argument('--vllm-log', default='', help="the engines' logs, comma-separated (CPU KV tier size)")
    a = ap.parse_args()
    ups = [u for u in a.upstream.split(',') if u]
    logs = [x or None for x in a.vllm_log.split(',')] if a.vllm_log else [None] * len(ups)
    gate = Gate(ups, policy=json.loads(a.policy), max_inflight=a.max_inflight, idle_ttl=a.idle_ttl,
                tick=a.tick, log=a.log or None, status_path=a.status or None, vllm_log=logs)
    web.run_app(gate.app(), host=a.host, port=a.port, access_log=None, shutdown_timeout=30, handler_cancellation=True)


if __name__ == '__main__':
    main()
