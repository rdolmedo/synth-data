# Trace judge (M3.4, #177)

Reads the **model's** trace of each finished attempt and flags **infra failures**: things outside the model that
broke and could have changed the outcome. It never flags model mistakes or benchmark flaws, and it doesn't read
graders' own traces.

| Category | What |
|---|---|
| `sandbox_kill` | sandbox/container kill, OOM, memory cap killing the **agent** or the box; fork/disk/fd exhaustion the model didn't cause |
| `network` | proxy, network or model-endpoint outage (connection refused/reset, 5xx from serving, proxy 503) |
| `harness_error` | harness, chat-template or tool-call-parse error |
| `harness_cutoff` | truncation or cutoff by the harness or run (run cutoff, a token cap below the protocol, dropped inputs) |
| `silent_zero` | an empty or missing answer caused by infra |
| `grader_crash` | verifier/grader crash visible in the trace (a verifier **timeout** on the model's work is not one) |

Not flagged (model outcomes or protocol limits): wrong answers; the context window, per-turn output cap or agent
time limit running out; the memory cap killing a process the model started; the model killing its own agent
(`pkill -f` matching the agent's command line, `kill <pid>`); EOS inside reasoning (also in agentic runs: a last message
that is only unclosed reasoning, finish stop, even with tool-call tags drafted inside it); max tokens at the protocol cap.
The pipeline's own protocol-limit zeros are decisive model outcomes on every benchmark: `error_kind=input_too_long`
(e.g. a GDP.pdf document too long for the 128k cap with the 32k output floor), unless the pipeline marked the attempt infra.
The v5 judge distinguishes each request's remaining output room from the total context window. AutomationBench
records its configured dynamic budget policy and observed peak prompt+completion count; the exact per-request
budgets are unavailable. CritPt failure usage can include model and summary responses absent from saved history;
successful rows can omit summary usage. These aggregate totals never prove a per-request cap hit or automatically
clear a suspicious attempt. Raw earlier opinions remain available when a trace is reviewed with this context.
HLE traces expose the run's recorded image policy. An explicit `image_inputs.sent=false` means text-only
generation even for questions referring to figures; missing images then reflect the selected modality protocol.
Legacy runs without that field remain unspecified, and enabled-image runs receive no such explanation.
An exception returned *as a tool result* by the benchmark's own tools or simulated API (e.g. AutomationBench's
simulated Gmail raising on a list `to`) is benchmark output, not `harness_error`. A crash under the verifier whose first
non-library frame is the model's solution (`/app/...`, a C extension it built) is the solution failing, not
`grader_crash`; so are verifier failures where pytest ran the tests and every failure is an `AssertionError`, or an
`ImportError` raised in the solution's own code (TB 2.1's verifier venv lacks a module the solution imported; log noise
in "Captured log" sections, such as Selenium teardown retries, is dropped from the digest). LLM flags of these kinds
are kept in `flags.jsonl` with `demoted` set and confidence ≤ 0.3. So is a proxy 503 that hits only hosts the model
made up: the cluster proxy answers CONNECT to a host with no DNS record with 503. That needs other hosts to have
worked through the proxy in the same attempt, more hosts working than failing, each failing host never reached, and
no DNS record for any of them at judge time (NXDOMAIN or no address; Squid's `ERR_DNS_FAIL` and curl/urllib3 resolver
errors mark the failing host; SERVFAIL counts when public DNS agrees); without usable DNS the flag stays. When every
failing name is missing both locally and in public DNS, the 503 is explained by itself and no other host needs to have
worked (a single probe of a made-up host); a publicly resolvable host that failed once stays flagged. When the
remote server itself answered (httpx `Server/Client error '4xx/5xx …' for url`, urllib/requests `HTTPError`, an
anti-bot challenge page) the tunnel worked and it's the site's outcome, not network infra. A bare `503 Service
Unavailable` from Stirrup's `web_fetch` is httpx's `ProxyError` (the proxy refused CONNECT) and stays a proxy failure.
`tracejudge.n_remote_site_limited` counts attempts where a site refused the model's scraping (429/403/503 from the
site, challenge pages), with `remote_limited_hosts`; it's the same for every model (no `web_search` tool), not infra. A proxy
CONNECT refusal to one existing host, where other hosts worked through the proxy in the same tool result or batch of
calls (at least as many as were refused), is our egress being blocked or that upstream being down: demoted and counted
as `tracejudge.n_egress_blocked` with `egress_blocked_hosts` (DuckDuckGo blocks cluster IPs). Refusals with no
same-call success, or to most hosts of a call, stay flagged. DNS
verdicts come from the local resolver, then Google DNS-over-HTTPS through the proxy, and are cached in
`TRACEJUDGE_DNS_CACHE` (default `/data/aai/aa_evals/tracejudge/dns_cache.json`), so a re-judge reproduces them. Proxy failures to a site that was answering 429/403 in the same
attempt (a bulk scrape that the site throttles, then refuses) are the site's limit too, when at least 2 other hosts
worked. A single-turn request whose client retried (gateway 5xx, the serving gate's "server at capacity")
and then got a complete answer (`finish_reason` stop) is not a network failure either.

## How it works

1. **Adapter** (`layouts.py`) reads one attempt into a `Trace`: harness events, model turns, what the pipeline
   recorded. Layouts: `pi_eval` (TB 2.1, TB-Sci, TB-Sci-lite), `hle`, `omniscience`, `gdppdf`, `critpt`,
   `automationbench`, `gdpval` (native completed outcomes, before pairwise scoring), and `generic`
   (any `items.jsonl` with a `transcript` pointer).
2. **Rules** (`rules.py`): exit codes, harness events and error strings. Decisive infra (the pipeline's own infra
   label, a memcap kill of `pi`, a run cutoff, a missing reward file) is flagged without the LLM; decisive model
   outcomes (the model's own kill command right before the agent died, EOS in reasoning, max tokens at the cap) are
   not flagged and skip the LLM under the existing `fails`, `suspicious`, and `all` policies.
   `--llm every` sends every trajectory to the configured LLM, including decisive model outcomes.
3. **LLM** (`judge.py`) reads a ~7k-token digest (`digest.py`) of every attempt that failed or that the rules found
   suspicious (policy `fails`, the default). Long agentic traces are condensed, not chunked: harness events first,
   a one-line-per-step timeline with error lines kept whole, the last 8 steps in detail.

A flag counts when its confidence is ≥ 0.5.

## Usage

```bash
cd <a worktree of aa-evals>
# online: follow a running run, judge each attempt as it finishes; folds n_flagged into summary.json at the end
python -m aa_evals.tracejudge watch --run-dir /data/aai/aa_evals/runs/<bench>/<run_id> --judge serving:aa-judge-gemma4-31b
# a finished run (only attempts not judged yet; --fold updates summary.json)
python -m aa_evals.tracejudge judge --run-dir <dir> --judge serving:aa-judge-gemma4-31b --fold
# rules only (no GPU): --llm none
# require an actual LLM opinion for every trajectory (including decisive model outcomes)
python -m aa_evals.tracejudge judge --run-dir <dir> --llm every --judge serving:aa-judge-gemma4-31b
# one attempt's digest and rule hits, for debugging a flag
python -m aa_evals.tracejudge digest --run-dir <dir> --key <key>
```

The policy does not change which keys are already present in the output file. To add missing LLM opinions to
rules-only rows, select those keys with `--keys`, use `--rejudge`, and save a separate `--out` file for review.

Each LLM result retains its per-call finish reason, response ID, usage and requested output cap, including the
one parse re-ask and calls before an endpoint error. Missing usage remains unknown (`null`), including aggregate
token totals when any call has unknown usage. `--save-raw` additionally saves the returned model JSON, content
and reasoning in that key's local flags row. It is off by default and changes no model request. No request
headers, credentials, environment or client transport objects are recorded. The records describe each judge
`chat` invocation; lower-level transport retries are not reconstructed as additional model responses.

`watch` is light on CPU (a few threads untarring TB archives). Run it in tmux next to the benchmark's own driver.
It stops when the run's `summary.json` says complete and every finished attempt is judged (`--idle-exit-h` to stop
earlier). Restarting it resumes: keys already in `flags.jsonl` are skipped.

**The hook.** Every benchmark's summary code calls, right before writing `summary.json`:

```python
from aa_evals.tracejudge import fold_summary
summary = fold_summary(run_dir, summary)
```

It adds `n_flagged` (counted attempts the judge flags that the pipeline scored as model outcomes: suspected silent
infra failures to look at), `n_flagged_by_category`, and `tracejudge` (`n_judged`, `n_flagged_known` = flags that
agree with `n_infra`/`n_cutoff`, `n_llm_errors`, `flagged_keys`, the judge). With no `flags.jsonl` yet,
`n_flagged` is `None`. It never raises.

**`flags.jsonl`**: one row per judged physical attempt (a re-judge appends; the last row wins):
`key, item_id, repeat, bench, counted, known_infra, recorded{...}, prefilter, hits[...], flags[{category, evidence,
ref, confidence, source}], flagged, categories, llm{ok, reason, prompt_tokens, completion_tokens, latency_s}, judge,
code_sha`.

## Adding a benchmark (GDPval-AA, SciCode-Verified, ...)

The `generic` layout works with no code if the run follows the aa_evals contract:
- `items.jsonl` rows carry `item_id`, `repeat`, `status`, `score`, and `transcript`: a path (absolute or relative
  to the run dir) to a JSON file, or `<file>.jsonl#<key>` where the row's `key` (or `item_id/r<repeat>`) matches;
- the transcript holds OpenAI-style `messages` (agentic: assistant `tool_calls`, `tool` results) and/or a single
  response block (`generation` or `response`: `content`, `reasoning`, `finish_reason`, `usage`), plus
  `error`/`errors` and, optionally, `events`: a list of harness lines (strings or `{source, text}`) such as
  sandbox exit codes, retries, timeouts. Harness events are what the judge leans on most, so write them.

`watch` sees an attempt when its `items.jsonl` row appears; a benchmark that writes `items.jsonl` only at the end
gets judged at the end (`judge`), or adds an adapter (`layouts.py`, ~30 lines) over its per-attempt files.

SciCode's generic view additionally reads existing raw generations, chains and config (`scicode_context_version=1`).
It includes bounded exact dependency instructions, the official import-extraction protocol and its saved-code match,
recorded OR/per-environment grading outcomes, and definitions of traceback symbols present in earlier generated code.
A `test.py` frame can contain cumulative model code; a matching definition alone cannot rule out test-code shadowing.
Missing optional artifacts leave the ordinary generic trace usable and origins unresolved. This context never demotes
flags or changes generation/grading. An authorized rejudge with a separate output file can apply it to saved runs;
preserve previous opinions and their code SHA.

GDPval's native adapter reads `attempts/<task_id>__rN/outcome.json` and its transcript before `items.jsonl` exists.
It keeps the scored generic logical keys (`task_id/rN`), later item scores and explicit network policy, all history
conversations and original source paths. A retryable infra outcome is not a final logical result: the worker replaces
its transcript under the same key. Exhausted infra outcomes remain visible as known infra. Trace-watch completion
depends on the selected generation outcomes, independently of pending pairwise judgments; incomplete/unreadable
outcomes stay pending. Existing items-only exports retain the generic contract. No generation files or verdicts change.

## Judge model

**Gemma 4 31B-it FP8 (`gemma4-31b-it-fp8`, 1×H100), thinking on, prompt `tj-v4`.** Thinking off is ~8× cheaper
per call but flags 32 of 99 clean traces (blind), so it is not an option. Production endpoint (coordinator,
2026-11-05): the shared `serving:aa-judge-gemma4-31b`, the CLI default. No separate trace-judge server.

Results on the #66 labeled set (`/data/aai/aa_evals/tracejudge/eval_v2/`):

| Set | Mode | Recall on infra | False flags on clean + hard negatives |
|---|---|---|---|
| dev (158) | prod: rules + LLM | 51/53 (real 29/29, injected 22/24) | 0/99 |
| held-out (126, new seed, no overlap) | prod | 42/44 (real 20/20, injected 22/24) | 0/82 |
| dev | blind: LLM alone, pipeline labels and hints hidden | 47/53 | 3/99 |
| dev | blind, thinking off | 49/53 | 32/99 |

Other one-GPU candidates on the same sets and code (shared servers, 2026-11-05; Qwen3.8 at `--effort medium`):

| Judge | dev prod recall / false | held-out prod recall / false | dev blind recall / false | latency per call |
|---|---|---|---|---|
| **Gemma 4 31B FP8** (pick) | **51/53 / 0/99** | **42/44 / 0/82** | 47/53 / 3/99 | 37 s |
| Gemma 4 26B-A4B | 51/53 / 1/99 | 43/44 / 1/82 | 48/53 / 7/99 | 20 s |
| Qwen3.8-27B | 45/53 / 0/99 | 36/44 / 0/82 | 42/53 / 2/99 | 23-27 s |
| Super | 42/53 / 0/99 | 34/44 / 0/82 | 38/53 / 0/99 | 17-20 s |

All four flag every real #66 infra failure in prod mode. They differ on injected faults: Qwen3.8 misses every chat-template
garbage answer, and Super misses about half the injected faults (disk full, fork failure, proxy outage, tool parse).
The 26B matches the 31B's recall at about half the latency but adds false flags: it treats the per-turn output cap and
the context cap as harness cutoffs. It is the fallback if the 31B server is down. The 26B's held-out false flag is a
proxy 503 on `cobol-modernization`, labeled clean: `gnucobol.org` has no DNS record (now demoted, see above).

`tj-v3` (2026-11-05, four false flags from the M3.4 validation loops) on the 31B, prod: dev 51/53 and 1/99, held-out
42/44 and 1/82. The misses are 4 injected empty-response faults. The false flags are `torch-pipeline-parallelism`
(its image ships without python) and the proxy 503 on `cobol-modernization` (a host with no DNS record; demoted since). The
table above is `tj-v2`.

The prompt and rules were tuned on the dev set (v1 blind: 41/53 and 4/99); the held-out set measures them after.
The misses are injected faults: an agent that stops after an empty model turn (2 of 4) and a raw `<tool_call>` in the
model's text (1 of 4). On two full #66 finals (TB-Sci-lite r4 medium and TB 2.1 r4 medium, 404 attempts), the new
flags were the #145 memcap kill (real), a proxy 503 on `mteb-leaderboard` (later traced to `mteb-leaderboard.net`, a host the model made up),
and one false flag: `torch-pipeline-parallelism`, whose image ships without python (a task property).

**Cost.** The LLM reads ~45% of TB attempts (the failed or suspicious ones), with ~8-11k prompt and ~700 completion
tokens each. On one shared H100 at 24 in flight, that came to 132 s of wall time for TB 2.1's 267 attempts and 225 s
for lite's 137, or 0.5-1.6 GPU-s per attempt. Rules alone (`--llm none`) take 10-30 s of CPU per run.

## Labeled set

`labels.py` builds the set from the #66 runs and compares judges.

```bash
python -m aa_evals.tracejudge.labels build --out labels.jsonl --scan <rules-only scan of the TB runs>
python -m aa_evals.tracejudge.labels run --labels labels.jsonl --judge serving:<judge> --mode blind --out res.jsonl
python -m aa_evals.tracejudge eval --labels labels.jsonl --flags name=res.jsonl ...
```

Offline default-index package denials retain a mandatory source-review marker. A linked,
plain `pip install` proxy denial under an explicit offline policy keeps the verifier's
score and stops automatic infrastructure retries; it does not establish that the index
was public. This includes an explicit public primary `-i`: inherited secondary
index configuration is not proved absent by that flag. Curl/wget public-request
handling remains separate. Private/mixed/overridden commands, model/backend errors and
setup/grader failures retain their infrastructure checks.

Bounded static-directory/status-echo curl commands and proxy-variable unsetting
are recognized without accepting dynamic/private targets. A literal Python
`requests.get` diagnostic is recognized only through a restricted AST with public
constant URLs and bounded timeouts. It can establish that subsequent ordinary
assistant reflection refers to the tool's policy denial; assistant error fields
and backend stop reasons remain checked. Bounded `git ls-remote` and default-source
`apt-get install/download/update` denials also retain a mandatory source-review
marker. Git rewrite/configuration and apt source configuration require their own
immutable source proofs. Commands mixing source managers fail closed. Local pip
indexes, including a task's localhost PyPI server, never enter the public-denial
exception; evaluator-injected local-routing defects require separate repair.
A pip compound may end with exact static Python version introspection (`import
module; print(module.__version__)`) only; that entire compound remains pending
source review. Calls, extra statements, aliases, file reads and private probes
in that postlude fail closed.

`n_offline_proxy_review_pending` is a separate audit-completeness count discovered
directly from native completed outcomes, independent of missing/partial/malformed
trace rows or LLM flags. Native earliest-noninfra retry selection determines counted
reviews; separate physical marker counts preserve audit-only history. Unreadable
native outcome objects fail closed. Full/final adoption requires zero pending reviews. The raw outcome's
`offline_proxy_review_required` stays true even after review. A root-written host file
`attempts/<key>/offline-proxy-review.json` can resolve it only with schema_version1,
the exact physical key, verdict `expected_policy_denial`, and these bindings:

- `archive_sha256` and `attempt_sha256`: actual immutable trial archive and attempt.json.
- `marker_evidence_sha256`: SHA256 of `json.dumps(evidence, sort_keys=True, separators=(',', ':')).encode()`.
- `source_image`: absolute actual native image `path` and `sha256`.
- `instruction`: pinned `bundle_sha256`, `member` (`tasks/<task>/instruction.md`) and instruction `sha256`.
- `index_audit`: absolute path and SHA256 of the independent source-index audit JSON.

The source-index audit records schema_version1, key, the same verdict,
`public_default_index: true`, nonempty public `indexes`, factual `evidence`, and
`bindings` containing archive_sha256, attempt_sha256, marker_evidence_sha256,
source_image_sha256, instruction_sha256 and bundle_sha256. The reviewer must inspect
the actual image's pip source/configuration, environment/index overrides and linked
failure evidence. This host receipt is a manual attestation backed by immutable
artifacts; the code does not infer public defaults from an LLM verdict. Duplicate
fields/indexes, mismatched hashes, missing proofs and private/ambiguous indexes leave
the marker pending. Version1 receipts approve pip markers only. Git/apt markers
require schema_version2 in both the host receipt and source audit, using
`source_audit: {path, sha256}` instead of `index_audit`. The audit retains the same
generation/image/instruction bindings, verdict and overall factual evidence. Its
`sources` list must contain exactly one entry per required manager (`pip`, `git`,
or `apt`), with `public_effective_sources: true`, a nonempty unique public `urls`
list and nonempty manager-specific `evidence`. The reviewer must inspect actual
image/config/env and all prior source-modifying agent calls, including git
insteadOf/transport overrides and apt source/config changes. No v1 pip approval
can resolve a git/apt marker. Duplicate or private source entries remain pending.

An explicit offline agent's bounded literal probes of its recorded non-loopback
host model bridge and the known cluster proxy may instead produce
`agent_host_egress_unresolved`. This requires linked bash results with
ENETUNREACH for both identities, the exact actual trial-config bytes/bridge URL,
and a subsequent normal assistant response. Arbitrary private/task services,
localhost pip, unknown commands and model/backend error fields retain their
infrastructure checks. This marker is unresolved, not a public-target exception;
its pending count blocks score adoption and release.

Only a separate schema_version3 host receipt/audit with verdict
`expected_agent_host_denial` can resolve it. It uses `host_probe_audit: {path,
sha256}` and an exact archived `relay_config: {member, sha256}`. Both the marker
and actual archived relay config must agree on the bridge identity. The audit
must bind the existing generation/image/instruction hashes plus
`relay_config_sha256` and `host_probe_targets_sha256` (canonical JSON digest of
the sorted unique marker targets). It records exact `probe_targets`,
`known_bridge_url`, `known_proxy_url`, `model_requests_continued: true`,
`non_task_host_probes: true`, `required_task_service_failure: false`, and factual
source/wire evidence. Any accompanying pip/git/apt marker still requires its
own matching public source proof in `sources`. Public v1/v2 receipts cannot
approve host probes; a v3 receipt cannot approve a different private or task
service, stale relay configuration, or a localhost index. All proofs remain
manual root attestations; an LLM clean opinion cannot resolve a marker.
Summary folds revalidate current receipts rather than trusting
stale trace rows. Raw generations and outcomes are never rewritten by this reader.

A causal HarborPi startup traceback through `_write_custom_models_json` for a
failed `chmod 600 /tmp/harbor-pi-agent/models.json` is infrastructure when no
contradictory assistant/request/token activity exists. Missing logs alone never
establish this condition. A model-issued chmod after inference is unaffected;
the raw verifier reward is retained and the infrastructure score is unset.

An unrecognized linked bash result with a proxy denial may produce
`agent_tool_network_origin_not_proven` only when the actual relay config is bound
and a later paid primary response is recorded. This stops automatic generation
retries but leaves mandatory source review pending. Prose and source code in a
normal assistant message are not transport observations; explicit error fields
retain their infrastructure checks.

Only a schema_version4 host receipt with verdict `expected_agent_tool_denial`
can resolve this marker. It binds the original archive/attempt, marker digest,
source image and original instruction, plus exact archived `relay_config` and
`session` members/hashes. `tool_network_audit` points to a separate hash-bound
root audit. The reader checks the linked tool command/result and paid primary
continuation against that archived session. The audit explicitly attests review
of all effective sources/prior mutations, continued model requests, and absence
of primary transport, evaluator or required task-service failure or unresolved
source contradictions. Accompanying manager markers still need matching public
source proofs. Older receipts cannot clear this marker; unresolved host probes
remain subject to their separate review.

A typed evaluator `verifier/grading-pending.json` for unsupported Caffe input
adapters resolves generation with `score: null` and `grading_review_required`.
It never authorizes a replacement generation. Genuine backend/evaluator failures
retain priority. Final score adoption requires the score reader to report zero
pending grades; an unknown missing reward retains existing handling.
