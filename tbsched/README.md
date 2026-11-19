# tbsched: one campaign scheduler for synthesis and eval

Design: #451 (comment "Workstream 1 design"), approved 2026-11-18. It replaces, over time, the overlapping systems that
decide how many sessions run where: synth's kv_fit admission and worker scaling, the aa_evals dispatcher and
`pi_eval/adaptive.py`, eval_q35's adaptive limit and orchestrate passes, and the M5 ladder scripts (autoscale, split,
fence, rounds, cleanup). The qwenserve gate stays as it is (request level, server side). Stdlib only, so frozen copies
run it. Named `tbsched`, not `sched`, which would shadow Python's stdlib module.

| Module | What it does | Replaces |
|---|---|---|
| `work.py` | `pending(run)`: unclaimed + unclaimed retries + dead claims = claimable; live claims; drained | 3 synth and 4 M5 versions of "work left" |
| `control.py` | per-engine limits (`aimd`), single-turn cap, load-aware hit evidence, `/metrics` windows | pi_eval adaptive, dispatcher `adaptive()`/autobudget, eval_q35 AdaptiveLimit, M5 autoscale tick |
| `kv_fit.py` | synth's `admission.step`, verbatim (kv_fit, slope search, blind cap) | `synth/admission.py` once synth imports it |
| `place.py` | credits per (run, engine) (`allocate`, `withdraw`) and `route` at claim | dispatcher `decide()`, pi_eval `hashed_url`/`pin`, gen_flat hash, synth backend choice |
| `lifecycle.py` | workers (size, trim, recycle, stop) and engine groups (steal, end game, release) | M5 trim/recycle/steal/release_idle/split/fence, dispatcher growth/release, synth fleet scaling |
| `turngate.py`, `gate.py` | eval_q35's TurnGate verbatim + `CollapseGate` (#477), `GateServer`, fail-open client | `turn_gate_host.py`, eval_q35 turn gate / gate server |
| `state.py` | per-entry files, overrides, O_APPEND events, mkdir lease | in-memory and shared-JSON controller state |
| `loop.py` | the process: `python -m tbsched.loop CAMPAIGN [--once]` (shadow by default) | |
| `sim.py` | simulator of engines, sessions and policies for the tests | |

## Rules it encodes (each has a test)

- **Per-engine limits:** pressure on one engine never cuts its sibling.
- **When the hit rate counts:** the prefix-hit rate is evidence only on a loaded engine. It is not evidence on a mixed
  engine (single-turn or replay traffic), nor while a launch wave is still on its cold first turns. The client-side
  efficiency, which counts reusable prefixes only, is preferred when it is known.
- **After a thrash cut:** the limit learns a ceiling below the level that thrashed and re-probes it every 45 min.
- **The hit rate with a CPU KV tier:** it comes from `prompt_tokens_by_source` whenever vLLM exposes it, so prefixes
  restored from the tier count as hits. `prefix_cache_hits` alone counts GPU hits only: in the 11-18 replay it read
  0.00 at 99% KV while the tier served a third of all prompt tokens.
- **When a limit binds:** live sessions plus credit being spent (unspent credit up to the sessions started in the
  last 2 min). Credit is refilled at every scrape within a rolling per-minute burst, so short sessions do not leave
  slots empty between decisions.
- **`policy: "guard"`** (opt-in, for CPU-KV-tier engines): aimd's cuts on thrash evidence, but no GPU-KV-gauge or
  running caps. It raises only while each step adds at least 2% generated tok/s and two standard errors of the noise.
  A step that does not pay returns to the previous limit, which becomes a ceiling until a later probe (the hold
  doubles per failed probe). Zero throughput is no evidence. `control.params()` gives a policy its full settings.
- **Credits:**
  - credit goes where the room is, so routing by credit is least-loaded routing;
  - the hash, with the round's seed namespace in it, only breaks ties;
  - unspent credit counts as used, and it is withdrawn at once on a cut, a drain or a gate hold;
  - the burst is per engine group, and replays are capped per engine (24, 8 per tick).
- **Workers:**
  - sized to live sessions plus spendable credit, never to a stale limit;
  - a black-hole worker is removed once it runs nothing (it is listed as `recycle_wait` until then), and the next
    tick backfills it; a run with claimable work and no worker always gets one (someone must reap dead claims);
  - idle surplus workers are trimmed after a grace period;
  - a drained run gets `STOP_WORKERS` (an unreadable queue is unknown, never drained); tbsched clears its own
    `STOP_WORKERS` when work comes back, and never one that a person wrote.
- **Engine groups:**
  - a hungry group joins the largest backlog of the same model (10-min cooldown); there are no splits or fences.
    Opt-in per run (`"steal": true`): pi_eval workers use their own registry only, so a joined group would strand
    credit until a hook routes across registries;
  - in the end game, small backlogs are consolidated;
  - a group is released after 2 idle ticks (tail cut only where the pipeline allows it), and runs left without
    engines are stopped;
  - a group is never released while it is marked `"shared": true`, while its engines cannot be read (an engine
    unreadable for over 10 min with no live claim stops blocking it), while they serve more requests than our sessions
    explain (another client), or while they hold live claims we could not place. The reason is logged once per change
    (`group_busy`). A release that removed nothing (no job record, not our job) leaves the group active;
  - an unhooked run is never spread onto another group in `on` mode (its workers would not use it).
- **Watchdog:** a session idle between model requests for longer than the watchdog's gap ends as an infra retry.
  Time inside a request or at the gate is never idle.
- **Collapse gate:** it is open until the cache collapses on a loaded engine. It then holds the youngest sessions'
  turns (agents are never killed), steers itself by cache efficiency, and reopens after 5 clean minutes. Sessions
  leave it by `/end`, or by a `/sync` from whoever hosts it, and a client fails open after `max_wait`. Hosting
  and syncing it from the loop come with the gate hook PR; the loop does not start a gate yet.
- **State:** one writer per file, operator overrides are read-only, and events are written as whole lines. An
  epoch-numbered lease rules out overlapping schedulers: of two newcomers one wins, and a superseded holder exits
  before acting.
- **Mixed groups:** `"mixed": true` marks a group whose engines also serve traffic not listed as runs (e.g. the
  aa_evals dispatcher's single-turn lanes); there the hit rate is no thrash evidence.
- **Live view:** dead claims are work, not engine load; unpinned pi_eval claims count on their hashed engine; KV
  usage is the max over data-parallel ranks. Shadow credit is written with `mode: shadow`, which hooked workers must
  ignore.

## Simulator results (old rules vs tbsched, same load; `tests/test_sim.py`)

| Lesson | Old (11-15 rules) | tbsched |
|---|---|---|
| Over-admitted engine (#6 10-25 shape), 2.5 h | 70 min at hit < 0.5, 137 done | 18 min with the collapse gate, 304 done |
| TB-Sci-lite ×3 parallel rounds, 2 h | 324 done | 363 (aimd); 505 with the gate and ceiling 320 |
| CritPt + co-located replays, 1.5 h | 776 t2 done | 931, ≤ 24 replays per engine |
| CritPt near the knee, 2 h | 1,503 done | 1,507 (no regression) |
| Big + small backlog on 2 registries | 1.56 h makespan, nothing released | 0.86 h: one join, end-game drain, both released ≤ 3 min after the end |
| 5% hung sessions (GDPval-like) | 2.4–2.5 h makespan (2 h cap) | 0.6–0.7 h (watchdog) |
| Single-turn co-located on a light Pi engine | — | 4.4–4.8× generated tokens, no hit-based cut |

The simulator gives the policies the true session count of every engine. The loop's own view of claims (dead and
unpinned claims, heartbeats, condor) is covered by `tests/test_loop.py` instead. The simulator is a model calibrated to
the 11-15 Super measurements (CritPt-like hit about 0.9 at 100 sessions per
engine, below 0.6 at 240). The `synth.bench` replays on real engines (design, validation step 2) are the evidence for
any default change.

## Status: flag-gated, nothing uses it yet

- `mode = "shadow"` (the default) decides and records, under `CAMPAIGN/sched/`, but acts on nothing.
- `mode = "on"` acts only for runs marked `"hooked": true`, meaning their workers read tbsched credits. The pipeline
  hooks are separate PRs, and each cutover needs the user's OK:
  - pi_eval routing by credit (`run.py --sched-state`, `aa_evals/common/pi_eval/sched_router.py`);
  - synth's `admission.py` shim and controller calls;
  - eval_q35 importing the turn gate.
- `kv_fit.py` and `turngate.py` are verbatim copies. `tests/test_gate.py::Copies` fails if either side changes alone,
  until the pipelines import them from here.
- Defaults: aimd ceiling 160 sessions/engine (the measured healthy Super range); the collapse gate is opt-in (approved
  decision 4) until the replay shows it recovers an engine in minutes.

## Replays of recorded traffic

`python -m synth.bench --adapt-hours H --sched sched|hash|proto [--sched-aimd JSON] [--sched-gate] [--sched-stationary]
[--gap-scale X] ...` runs tbsched (or the 11-15 rules) as the admission client of a replay (`bench_client.py`).
`--sched-stationary` starts the initial fill mid-trajectory (a mix of context ages instead of one ageing cohort);
`--gap-scale` stretches the recorded tool gaps toward live agents'. The results are on #451.

## Run

```
python -m pytest -q tbsched                              # ~20 s, CPU
python -m tbsched.loop /data/aai/<campaign> --once   # one shadow tick (needs CAMPAIGN/sched/config.json)
```
