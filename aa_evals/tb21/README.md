# aa_evals/tb21: Terminal-Bench 2.1 (issue #67)

TB 2.1 replaces TB 4.0 as the overnight TB eval (#66): AA has Qwen3.8 27B at 5.6 on TB 4.0 but 79.8 (Xhigh) and
65.2 (Medium) on TB 2.1.

## M3 protocol (M3.0 #173, M3.2 #175): one command per model

```
python -m aa_evals.tb21.run --model serving:<name> --run-id <id>                        # any M3.1 model
python -m aa_evals.tb21.run --model serving:aa-qwen-m32-tb --reasoning-effort medium --run-id m34-qwen-medium
python -m aa_evals.tb21.run --model url:http://host:port@served --request-preset <teacher> --run-id <id>
python -m aa_evals.tb21.run ... --context-window 262144     # Qwen3.8 calibration runs only (its native length)
python -m aa_evals.tb21.run --agent oracle --run-id oracle-v1 --repeats 1
python -m aa_evals.tb21.run --run-id ID --status | --score | --stop | --resubmit N
```

| | M3 default | Before M3 (rounds 1–4, #66) |
|---|---|---|
| Context | **131,072 (128k)**, episode ends at the window; `--context-window 262144` only for the Qwen3.8 calibration | 132,000 (r1–r3), 262,144 (r4) |
| Repeats | **5** (`--repeats`, alias `--attempts`); seeds `q35-tb21-v1/<task>/<k>`, so k = 0–2 are r3/r4's seeds | 3 |
| Request settings | **the model's own**: the serving registry's `[request]` block (`endpoints.json` → `request`: `overrides`, `adapters`, `pi_thinking`, `env`), or `--request-preset TEACHER` for `url:` models, plus `--request-overrides JSON`. `--reasoning-effort` replaces the effort at the top level and in `chat_template_kwargs`. The Pi proxy merges them into every sampled request (`--request-overrides`); there are no built-in sampling defaults | T 1.0 / top_p 0.95 / `reasoning_effort` / `clear_thinking: false` fixed in the proxy |
| Efforts (M3.0) | DeepSeek V4.1 Flash: max (its registry setting); Qwen3.8-27B: `--reasoning-effort medium` and `xhigh`; Super, Lightning, Qwen3.6, Gemma 4, Qwen3.5-4B: reasoning on, as their registry sets it | medium, xhigh |
| History adapters | registry `adapters`: `inline_reasoning_history` (Super) inlines earlier reasoning as `<think>`; `mirror_reasoning_content` is always on in the proxy | mirror only |
| DeepSeek context counting | registry `env` `PI_CONTEXT_RENDERER=deepseek_v41` + `PI_CONTEXT_TOKENIZER`: the context guard renders the prompt with the vendored V4.1 encoder and counts it in-process (vLLM's `/tokenize` crashes on V4.1 tool histories); ported from aa-serving | — |
| Time limit (M3.5, user 2026-11-05) | **7,200 s (2 h) of environment time per task**, the binding limit (model time excluded, so it does not depend on server load); wall-clock backstop 2 × 2 h = **4 h**; the same caps as TB-Sci-lite, uniform for every task (`agent_cap_sec` 7200 on the six-hour bundle: `PI_ENVCLOCK_MULT` 1/3, Harbor `--agent-timeout-multiplier` 0.667). Measured on the M3.4 Qwen3.8 medium run (6 h cap): 9 of 444 attempts ran past 2 h of wall time (5 of them passed; at most −1.1 pp under a 2 h *wall* cap, less under the env clock). The main effect is bounding slow servers' tails (DeepSeek) at 4 h | 21,600 s (6 h) env clock, 12 h backstop (r1–r4, M3.4) |
| Images / text-only models | Historical runs and M3.5 DeepSeek use text-only input. New runs freeze explicit `model_input` from actual serving launch capability; image-enabled models can send native `read` tool images. Unknown endpoints default to recorded text-only unless explicitly declared. See [input policy](../common/pi_eval/README.md#model-input-modalities). | text-only: image reads never sent `image_url` |
| Per-sandbox memory cap (#378, user 2026-11-13) | **1.5x the task's `memory_mb`** (`HARBOR_NS_MEMCAP=1.5`, harbor_ns `memcap.py`, as TB-Sci-lite): a 1 s watchdog kills the sandbox's largest process (Pi last) when its unreclaimable memory exceeds the cap, so one runaway task cannot get the whole worker held. A kill is logged in the attempt's `harbor.log` ("harbor_ns memcap: sandbox at … MB > limit …"). AA's harness enforces 1.0x (Docker `mem_limit`) | none (c30 and earlier) |
| Task instruction delivery (M3.5, #219) | **on stdin**: Harbor's Pi launch is rewritten (`tb_agents/harness_agents.py` `PiCached.exec_as_agent`) so the instruction is uploaded to `/tmp/harbor-pi-agent/instruction.md` and read as `pi … < instruction.md`; no process cmdline holds it. Pi trims piped stdin, so the instruction's final newline is dropped (otherwise byte-identical). Reason: with the instruction as Pi's last argument (M3.4 and earlier), a model that kills processes by command-line match on a word from the instruction (`pgrep -f win311 | xargs kill -9`) killed its own agent: `NonZeroAgentExitCodeError (exit -9)`, scored as a model outcome. Recount over every Qwen3.8 TB 2.1 / lite run (r3, r4, M3.4; ~2,300 attempts): 6 self-kills, about 0.26%. Five matched a word from the instruction, which the stdin launch fixes: `qemu-alpine-ssh` (r3 medium, exit -9), `mcmc-sampling-stan` (r3 xhigh, `pkill -f analysis.R`, exit -15), `rstan-to-pystan` (r4 medium, a PID from `pgrep -af pystan`, exit -15), `install-windows-3.11` (M3.4, exit -9), and TB-Sci-lite `survival-band-screening` (r4 medium, `pkill -f predict.py`, exit -15). One matched the launcher itself and is **not** fixed: `build-pmars` (M3.4, `pkill -f "bash -c"` also kills Harbor's `bash -c` that runs Pi). The rate may be higher for models that pattern-kill freely. (TB-Sci-lite r4 `apo-holo-docking` is a different artifact, the `HARBOR_NS_MEMCAP` watchdog killing Pi; see `aa_evals/tbsci_lite/README.md`.) AA's Terminus 2 drives tmux from outside the container, so this cannot happen there. `PI_INSTRUCTION_ARGV=1` restores the argv launch (to reproduce M3.4) | argv (the M3.4 validation runs, code 92fda39, still use it) |
| AA reference | `--aa-slug` defaults from the served name and effort (`pi_eval/run.py` `AA_SLUGS`; e.g. `qwen3-8-27b-medium`) | `--aa-slug` by hand |
| `summary.json` additions | `n_infra`, `n_cutoff`, `n_flagged` (trace judge, M3.4: `fold_summary` over `RUN/flags.jsonl`; `null` until it ran), `n_repeats`, `judge: null` (the verifier decides), `model`, `reasoning_effort`, `context_cap`, `code_commit`, `request_overrides`, `request_source`, `request_adapters`; per-attempt rows in `items.jsonl` | — |

## Method: our Pi protocol (user decision 2026-11-02 ~23:40)

The user chose our in-house protocol so the numbers compare with every past tb21p180 / tbmix eval. It is eval_q35's
`pi180` harness run through `aa_evals/common/pi_eval` (README there) against the serving registry:

| | Setting |
|---|---|
| Tasks | the 89 tasks of `/data/aai/tbenchtrain/eval_q35/bundle/tb21_6h.tar` (sha256 `363d5b93…`), the bundle of all past runs |
| Harness | Pi 1.2.0 pi-cont16k, bash180 bundle (`/data/aai/tbench/agent_cache_pi_cont_bash180_v1`) |
| Context | 131,072 tokens (M3; 132,000 in rounds 1–3); at the limit the episode ends and the verifier scores the state (no compaction) |
| Output | 16,384 tokens per request incl. reasoning, 0 resamples |
| Sampling | the model's registry request settings (M3, above); rounds 1–4: T 1.0, top_p 0.95 (proxy), `reasoning_effort` medium / xhigh |
| Time limit | 21,600 s of environment time per task (model time excluded), 2× wall-clock backstop; approved by the user via the coordinator (option a, 2026-11-02 ~23:50). **M3.5 on: 7,200 s, 4 h backstop** (M3 protocol table) |
| Seeds | namespace `q35-tb21-v1` (the past runs' seeds) |
| Attempts | 5 per task (M3.0; AA: 3 repeats; rounds 1–4: 3); pass@1 = mean over tasks of the mean attempt outcome |
| Sandbox | harbor_ns, **single-uid** (as every past tb21p180 run; see the oracle validation); offline verifier toolchain of eval_q35 |
| Infra | eval_q35's classification and up to 6 same-seed retries; infra never scores; > 2% infra makes a run invalid |
| Concurrency | round 1: ≤ 48 in flight on shared `aa-qwen-any`; round 3: dedicated `serving:aa-qwen-tb21` (1×B200, admission gate), 128 in flight; `RUN/control.json` changes it live |

**Deviations from AA's method** (pi180 protocol choices, kept by the user's decision to use our protocol):
- **128k context for every model** (M3.0, #173): AA runs each model at its native context. On Qwen3.8 the window cost
  8.2 pp at xhigh (r3 132k vs r4 262k, p = 0.006) and about 6 pp at medium; the calibration runs use 262k.
- **5 repeats** (M3.0) instead of AA's 3.
- **Image inputs** (above): historical runs omitted image attachments. New vision-enabled runs can attach
  native tool images; this differs from AA's terminal-only Terminus 2 and must be reported with the score.
- **16,384 output tokens per request** (reasoning included; a capped turn continues with the protocol's continuation
  message): AA's rule is the model's maximum output. Round 1 hit the cap 6 times in 15 attempts.
- **132,000-token context**, ending the episode at the limit (no compaction), vs Qwen3.8's 262,144. In round 3, 19%
  (medium) and 29% (xhigh) of attempts ended there, and 10–13% of those passed. **User decision (2026-11-03, ~12:30):
  the finals rerun at 262k** (`--context-window 262144`, round 4: `r4-262k-medium`, `r4-262k-xhigh`), pi180 otherwise
  unchanged. The summary reports `n_context_limit` (at the run's window), `n_past_132k` (attempts whose context passed
  132k, where round 3 would have ended them) and the pass rate of each.
- Harness, time limit and repeats: see "AA's method" below.

**Scoring** (`aa_evals/common/pi_eval/score.py`, overnight.md CUTOFF rules): `score` is pass@1 over all 89 × 3 attempts
with cutoffs (unresolved at the report deadline, `n_cutoff`) counted 0; `score_excl_oracle_failures` likewise without
the three tasks whose reference fails; `*_resolved_only` are secondary. Infrastructure errors are never scored;
`valid` = infra errors ≤ 2%. A verifier timeout scores 0 (AA's rule); a verifier that crashed or never ran is an
infrastructure retry.

## Dataset pinning

Harbor registry `terminal-bench/terminal-bench-2-1`, revision 6 (`latest`, published 2026-04-30,
content hash `sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a`), downloaded to
`/data/aai/aa_evals/data/tb21/tasks_rev6/terminal-bench-2-1`. Compared file by file with `eval_tb21.tar`, the source
of `tb21_6h.tar`: same 89 tasks and files except
- `.gitignore` files (only in eval_tb21.tar), and
- `sanitize-git-repo/tests/test_outputs.py`, whose fake secrets eval_tb21.tar writes as concatenated literals
  (`"AKIA" + "1234567890123456"`) and the registry as plain strings: same values, same tests.

`tb21_6h.tar` differs from `eval_tb21.tar` only in `[agent] timeout_sec` (21600; eval_q35/prepare_bundle.py checks it).
(`terminal-bench/terminal-bench@2.0` in the legacy registry is TB 2.0, not 2.1; the `terminal-bench/terminal-bench`
package has only 3.0.0 and 4.0.0.)

## AA's method (not used; the AA-faithful path)

From AA's methodology page (Terminal-Bench 2.1, a constituent through Intelligence Index v4.2):
- **Terminus 2** harness in an E2B sandbox, the full 89 tasks;
- **250 episodes** max;
- per-task agent timeout **max(7,200 s, the task's own timeout)**;
- **pass@1 averaged over 3 repeats**; a task passes only if every test passes;
- general AA rules: lab-recommended sampling (T 0.6 for reasoning models otherwise), 30 API retries;
- context handling not stated (Terminus 2's default is summarization on overflow).

Harbor 0.24 ships Terminus 2 (`-a terminus-2`, `max_turns`, JSON parser, summarization on by default; it needs
`model_info.max_input_tokens` for a hosted_vllm model). It is not wired into an aa_evals runner. The mini-swe-agent
runner (`aa_evals/common/harbor_mswea`) is AA's TB 4.0 / TB-Science harness, not its TB 2.1 one.

## Oracle validation (2026-11-03, `--agent oracle`, the same worker / sandbox / verifier path)

Runs under `/data/aai/aa_evals/runs_fallback/tb21/`. 0 infrastructure errors in every run.

| Run | Mode | Pass |
|---|---|---|
| `oracle-v1` | multi-uid | 82/89 |
| `oracle-v1-nomu` | single-uid | 86/89 |
| `oracle-fail-x3` | multi-uid, the 7 oracle-v1 failures × 3 | see below |
| `oracle-fail-nomu` | single-uid, the same 7 × 1 | see below |

| Task | multi-uid | single-uid | Reading |
|---|---|---|---|
| build-cython-ext | 0/4 | 0/2 | reference fails one test of pyknotid's own suite (`test_pyknotid_repository_tests`); models pass it (94/413 attempts of past eval_q35 runs) |
| mcmc-sampling-stan | 0/4 | 0/2 | reference's unpinned R install: RcppParallel now needs cmake, so rstan is missing; models pass it (65/403 past attempts; Qwen passed it in round 1) |
| protein-assembly | 0/4 | 0/2 | reference's dnachisel optimisation fails its translation constraint every time; **no model has passed it either** (0/224 past attempts): possibly unsolvable in our runtime, reported separately |
| custom-memory-heap-crash | 0/4 | 2/2 | multi-uid: `ulimit -c unlimited` gets EPERM (condor's hard core limit is 0) and `set -e` stops solve.sh. Fixed for later runs: worker jobs set `coresize = -1` (hard limit unlimited, soft 0) |
| mailman | 0/4 | 2/2 | multi-uid: postfix spool dirs lose their build-time owner (the nsfake chown is only an xattr, dropped on unpack); postsuper gets EACCES. A harbor_ns fix is proposed by aa-tbsci |
| fix-ocaml-gc | 3/4 | 2/2 | the verifier's `git clone` of github.com/sadiqj/ocaml failed once (transient) |
| largest-eigenval | 1/4 | 1/2 | timing test (`test_speedup`) misses by tens of µs on a busy worker; flaky, as in past runs |

**Decision: TB 2.1 runs single-uid** (`multiuid: False` in `run.py`), as every past tb21p180 run did: the reference
solutions pass 86/89 single-uid vs 82/89 multi-uid, and multi-uid fixes no TB 2.1 task. (Round 1 ran multi-uid
before this; none of its 15 tasks is affected.) The three tasks whose reference fails in both modes stay in the
benchmark (AA runs all 89): build-cython-ext and mcmc-sampling-stan are solved by models; protein-assembly was
never solved before round 3, but Qwen3.8 solves it since (r3 medium 1/3, r3 xhigh 1/3, r4 medium 1/3, r4 xhigh 2/3,
M3.4 medium 3/5), so it is solvable; the summary still reports `score_excl_oracle_failures`.

**Benchmark flaws seen in model runs (documented, not fixed; the same at AA):**
- `feal-differential-cryptanalysis`: the task image has numpy in the system `python3` (the agent's), but the
  verifier runs pytest in its own uv venv (`/root/.cache/uv/archive-v0/…/bin/python`) without numpy, and the
  instruction says nothing about dependencies; an `attack.py` that imports numpy fails at test time
  (`ModuleNotFoundError`, M3.4 a3). Scored as a model outcome; solutions without numpy pass (M3.4 3/5).

## Round 1 (Qwen3.8 27B, medium, `r1-medium`)

15 tasks × 1 attempt, stratified by category × hard/not-hard (largest remainder, seed 67: 5 hard, 10 not), on
`serving:aa-qwen-any` (1×B200, shared with the other benchmarks), at most 48 in flight.

Tasks: build-pmars, dna-assembly, feal-linear-cryptanalysis, financial-document-processor, gpt2-codegolf,
large-scale-text-editing, make-doom-for-mips, mcmc-sampling-stan, overfull-hbox, pypi-server, pytorch-model-recovery,
qemu-startup, query-optimize, raman-fitting, sanitize-git-repo.

Results and failure taxonomy: issue #67.
