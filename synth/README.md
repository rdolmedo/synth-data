# synth: one Pi cont16k data-synthesis pipeline for every teacher

`synth/` replaces the eight per-teacher campaign stacks that were built in parallel on the
`generation/*` branches (merged into this branch for history; their findings are in
`docs/campaigns/`). Teachers differ only in `synth/teachers/<slug>.toml`; everything else is
shared: serving supervision, admission, scheduling, rollout workers, the wire log, token
accounting, scoring and finalization.

```
synth/campaigns/<name>.toml ──prepare──> <root>/<id>/ops/campaign.json (frozen), task bundles, protocol.json
                                         controller job (CPU) ── owns everything below
serving jobs (GPU)  synth.serve      N vLLM replicas/ranks per job, preflight, staging, endpoints.json
workers (CPU)       synth.worker     inbox -> gen/run_harbor.sh -> gen/pi_cont16k.py -> Harbor + Pi 1.2.0
                                     Pi -> sticky_proxy (overrides, --wire-log) -> context guard -> replica
exporter (child)    synth.export     native Pi graph -> segments, unique-token count, loss weights
```

## Running a campaign

```bash
cd <repo>; PY=/opt/aai/venvs/harbor/bin/python
$PY -m synth.prepare synth/campaigns/nemotron35-super-top8.toml --dry-run   # print the frozen config
$PY -m synth.prepare synth/campaigns/nemotron35-super-top8.toml             # freeze, build bundles, protocol
$PY -m synth.prepare synth/campaigns/nemotron35-super-top8.toml --submit    # + submit the controller job
```

The controller submits and replaces its own serving and worker jobs (receipts in
`ops/jobs/`, ledger `ops/owned-jobs.json`; it only ever removes receipted jobs). State and
progress: `ops/progress.json`, `ops/status.log`, `ops/admission.json`,
`ops/serving/job-*/endpoints.json`, `ops/metrics/`. Live knobs, re-read every 15 s:
`ops/admission-policy.json` (any key of `[admission]`, plus `worker_count`, `serving_jobs` (fewer engines:
drain, release, restore; jobs above it are never submitted, #234), `max_agents_per_backend`, `max_infrastructure_failures_per_task`), `ops/blocked-tasks.json`
(a list: no more attempts, live ones cancelled) and
`ops/serving/control.json` (teacher tunables such as `mtp`/`max_seqs`; bumping `generation`
restarts every engine in place). `ops/CANCELLED.json` blocks new submissions. A finished
campaign writes `ops/completion.json`.

`[serving.autoscale]` (synth/autoscale.py, #234): start with `start` engines and add or drain engines from demand
(active + dispatchable attempts vs engines x kv_fit limit, with hysteresis) up to `jobs` and the live cap; a drained
engine is released once its last attempt ends and is the first one back. CPU replay of a finished campaign under any
thresholds: `python -m synth.analysis.autoscale_replay --campaign <dir> [--grid]`.

Stages:

1. `smoke`: the configured smoke attempts must pass. The checks are:
   * reasoning saved;
   * tool results saved;
   * continuation, and a zero-loss terminal at the context limit;
   * local token count == vLLM `/tokenize`.

   A smoke lost to the infrastructure is retried up to 3 times, and only the latest attempt is judged. The
   capped smoke needs no reward, and at least one smoke must be scored.
2. `running`.
3. `finalizing`: release serving, stop workers, re-hash every export.
4. `complete`.

## Ending a campaign early, sharing serving

* **Tail rule.** A campaign's last rollouts are not worth holding its GPUs for. The rule fires when, for
  `hold_minutes` (20):
  * at most `idle_fraction` (10%) of the serving admission capacity is in flight;
  * at most `max_remaining_fraction` (3%) of the planned attempts are still to start.

  The controller then cuts the live attempts (kept as truncated data, `TailCut`), finalizes and releases
  serving. Override the defaults in `[tail]`, or live in `ops/tail-policy.json`. Evaluation reports pass@1
  both with unresolved attempts as failures and over resolved attempts only, with the unresolved reasons.
* **`ops/CLOSE_TRAIN.json`** (`{"reason": ...}`) ends training now with what it has; live attempts are kept
  as truncated data. `completion.json` records the close and lists the tasks below target.
* **`ops/serving/RETAIN.json`** keeps a campaign's serving jobs up past its completion. A successor with
  `[serving] adopt_ops = "<that campaign>/ops"` runs on them, provided the engine command, context and output
  limits are identical. Adoption is recorded in the source's `ops/serving/adopters/`, and RETAIN.json holds the
  jobs only while an adopter is attached or for an hour after it was written, so launch the successor within
  that hour. An adopter that is done detaches and never removes a job its source still runs; the last campaign
  to let go of the job removes it.
* **`ops/worker-resource-overrides.json`** sets the shape of workers submitted from then on. Keys are an index
  (`"7"`) or an inclusive range (`"0-39"`); values may set `slots`, `cpus`, `memory_gib` and `disk_gib`, and an
  exact index wins over a range. Running workers keep their shape. Dispatch fills workers by the fraction of their
  own slots in use, so a mixed fleet carries load in proportion to slots (worker-shape A/Bs within one campaign).

## Task streams (`synth/stream.py`)

A campaign normally freezes its task set at `synth.prepare` (one bundle, its list and sha256 in `ops/campaign.json`),
so the 330M task-synthesis proof of concept traced in batches (s1, s2, s2b, ...): Super idled while a batch
finished, and every batch had its own tail. With `[train.stream]` a running campaign takes new tasks as they are
generated:

```toml
[train]
mode = "per_task"                  # also tokens; tokens_budget and validation work as with a frozen list
trajectories_per_task = 1
[train.stream]
dir = "/data/aai/tbench/tasksynth/streams/<name>"
poll_sec = 60                      # default
# train.tasks + bundle_source are optional: a frozen first part of the list, then the stream's tasks
```

* **Producer.** `<dir>/chunks/NNNN.tar` (Harbor tasks, `tasks/<name>/...`) and `NNNN.json`, the commit record written
  last (the chunk's tasks in campaign order, the tar's sha256, the producer's source note). Chunks are immutable and
  task names unique across the stream; `publish` refuses a task whose `task.toml` campaigns could not admit. One
  producer at a time (`<dir>/publish.lock`; a dead holder on the same host is taken over, else
  `rm -r <dir>/publish.lock`). `python -m synth.stream publish|end|status`; for task synthesis,
  `python -m task_synthesis.pipeline.campaign stream --rounds ... --stream <dir> [--exclude <bundles>] --follow --end`
  publishes judge-passed round tasks while the rounds run, after decontamination and near-duplicate checks against
  every earlier chunk and the excluded bundles (rejections in `<dir>/producer/rejected.json`, final except
  `invalid_task`, which is checked again at each publish in case the task was repaired; ~1.5 min and ~5 GB of
  memory per publish at 100k tasks). `END.json` (`synth.stream end`) closes the stream; with `--follow --end` that is
  once every spec has a record, so `round rejudge` judge errors first.
* **Controller.** It admits committed chunks in index order (a missing index holds back later ones): checks the tar
  against its record, rewrites agent limits as `synth.prepare` does, and keeps the result as
  `ops/stream/train-NNNN.tar` with its audit `train-NNNN.json`; `ops/stream/chunks.json` lists every admitted chunk
  with its source and sha256 (so does the train manifest; `completion.json` has a summary). A task whose `task.toml`
  cannot be rewritten is left out of its chunk and listed under `invalid`. A chunk that cannot be admitted at all
  (e.g. its tar no longer matches its record) is retried every poll and holds back the later ones and `END.json`;
  `ops/stream/skip.json` (`{"<index>": "reason"}`) passes over it. The new tasks go to the end of the train list;
  workers extract a chunk the first time an attempt needs one of its tasks. A smoke task from the stream waits for
  its chunk, and fails the smoke once the stream has ended without it.
* **More work coming.** While the stream is open (no `END.json`, or chunks it counts not yet admitted) and a token
  budget, if any, still opens tasks, the campaign never reaches its per-task target, the tail rules do not fire,
  serving is not drained, and `tokens` mode waits for the next chunk instead of cycling. Under a `saved` token budget, slots that no new task can take go to
  `overflow_per_task` extra trajectories as before (`overflow_per_task = 0` leaves them idle). A `saved` budget still
  closes training the moment it is reached, and a `projected` budget that no longer opens tasks lets the opened tasks
  finish, open stream or not; `ops/CLOSE_TRAIN.json` works as usual. After the end the
  campaign behaves like a frozen one.
* `progress.json` shows `stream` (chunks, tasks, end, last admission, the last admission error). A campaign without
  `[train.stream]` is unchanged.

## Trajectory bank (`synth/bank.py`)

One growing corpus per teacher and dataset, `/data/aai/tbench/synth/bank/<teacher>/<dataset>/` (user,
2026-10-25: serving-recipe searches generate the production datasets, TMax and CalibForge at 8 per prompt with the
Pi harness, and keep everything). The bank indexes campaign exports, it does not copy them:

* `python -m synth.bank deposit /data/aai/tbench/synth/bank <campaign root>` adds the campaign's usable,
  finalized trajectories with its serving recipe (teacher tunables, admission, Pi bundle, commit), the verifier
  reward and `verifier_infra_failure` (proxy or download errors in the verifier output: the reward is not evidence).
  Re-depositing replaces the campaign's entry, so deposit running campaigns again and once more at their close.
* `<bank>/manifest.json` is a train manifest of the union: list it in `train.baseline_manifests` and a new campaign
  on that teacher and dataset only tops each task up to `trajectories_per_task`.
* `python -m synth.bank summary <bank>` prints totals and the members with their recipes.

## Packed attempts (`synth/archive.py`)

An attempt directory holds ~27-31 files. On 2026-10-25 18:25Z the campaigns filled the 41.9M-file /data quota
(`lfs quota -u user /data`, column files): every new file failed with `OSError(122)`. Attempts are therefore
packed into shards, by default:

* **Layout**, per campaign and kind: `<root>/<kind>/attempts/`
  * `attempts-NNNNN.jsonl.gz`: one line per attempt with what training and the bank read: the export record,
    reward, `verifier_infra_failure`, `truncated`, the worker result, the first wire request with a system prompt
    and tools (the rule of `sft_qwen35/prepare.py`), the native Pi sessions and the export JSON;
  * `raw-NNNNN.zip`: every file of the attempt directories, symlinks as symlink entries (never followed);
  * `index.json`: attempt id -> `[shard, line, reward, infra, truncated]`, plus the shards.
* **When**: the controller packs finished train attempts with final exports in shards of 1,000 while the campaign
  runs (at least 10 min old, checked every 5 min), and everything left (train, eval, smoke) after
  `completion.json`. `[archive]` overrides `incremental`, `pack_on_complete`, `shard_size`, `workers`,
  `min_age_sec` and `interval_sec`.
* **By hand**: `python -m synth.archive pack <roots...> [--kind train] [--workers N] [--delete] [--only ids.json]`
  (train of a running campaign only with `--only`), and `python -m synth.archive verify <root>`.
* **Reading**: `archive.read_attempts(root, kind, known=..., tasks=...)` (= `iter_attempts`) and
  `archive.read_raw(root, uid, relpath, kind)`. The exporter keeps packed attempts' index entries, bank records
  carry an `archive` pointer, and `sft_qwen35/prepare.py` (sft branches since ed30e3e) reads shards plus attempts
  not packed yet.
* **Consistent while a pack runs (#166, #236).** A pack writes the shard, then `index.json` (atomically), and only
  then deletes the directories. `read_attempts` reads the index and its shards, then the unpacked directories in
  chunks with the index re-read after each (`read_dirs`: an attempt packed during its read comes from its shard),
  then the index again (attempts packed before the listing). `read_raw` tries the directory before the index. The
  bank deposit and `traces.new_attempts` use the same read.
  * A shard line whose attempt was packed before its export was finalized has `"record": null` (1,526 on m1-hero).
    When the manifest marks it usable and finalized and the line holds its sessions and first request, the line
    gets the manifest record and `record_from_manifest: true` (as `prepare.py` takes it).
  * **Completeness:** every usable, finalized `manifest.json` record must come back with its record, first request
    and sessions; otherwise `archive.Incomplete` names the missing attempts (`check='warn'` logs them instead, as
    the trace judge's `--follow` rescan does). m1-hero: 115,680 of 115,680 (2026-11-05).
* **How a shard is made**: built in node-local scratch (`SYNTH_ARCHIVE_STAGE`, else `_CONDOR_SCRATCH_DIR`,
  `TMPDIR`, `/tmp`; zip and gzip write in small chunks, which Lustre took at ~0.4 MB/s), checked (member sizes,
  CRCs), copied to /data with `dd bs=64M` and checked by SHA-256. Directories are deleted, in a thread pool, only
  after the shard's index entry is written. A restarted pack adopts shards an interrupted run published and checks
  sizes against the zip before deleting. A failing shard is logged and the others continue. The packer gives the
  owner back read access first (an agent's sandbox left a directory without its execute bit) and fails a shard on
  FIFOs and other special files instead of blocking on them.
* **Result**: the 17 campaigns finished by 2026-10-25 went from 9.86M files / 1.63 TiB to 719 files / 0.23 TiB. Any further /data
  cleanup is handled by the user (issue #10).

## Package cache (`synth/package_cache.py`, issue #60)

Image builds and offline sandboxes fetch the same few hundred packages over and over (task-synthesis r6a-h: 6,190
package installs, 551 unique; 27% of builds run `apt-get update`, ~50 MB of Ubuntu indexes each). Each job that builds
images runs one package cache next to it, on node-local disk, with the cluster proxy as its upstream:

- **Round jobs** (`task_synthesis.pipeline.round`, on by default; `--package-cache off` disables it,
  `--package-cache-gib` caps the store, added to `request_disk`) and **campaign workers** (`harbor.package_cache`,
  default true; `harbor.package_cache_gib`, default 10).
- **Builds** (harbor_ns `RUN` steps): `http_proxy` (apt, other `http://` downloads) and `PIP_INDEX_URL` point at the
  cache while it is up; `https://` keeps going to the cluster proxy. The environment is computed per step, so a step
  started (or retried) while the cache is down uses the cluster proxy. A build context with a pip configuration file
  (`pip.conf`, `pip.ini`, `PIP_CONFIG_FILE`, `pip config set`) keeps pip on its own settings: `PIP_INDEX_URL` would
  override it. Command-line options (`--index-url`, `-i`, requirements-file lines) and an image `ENV PIP_INDEX_URL`
  take precedence over the cache's variable anyway, so those builds use the cache for everything else.
- **Offline sandboxes** (`harbor.package_proxy`): the per-sandbox registries-only proxy forwards allowed apt requests
  to the cache (cluster proxy if the cache is down) and logs the cache's verdict and the file's SHA-256 next to each
  request. `harbor.package_cache_pip = true` also serves pip (`PIP_INDEX_URL=http://127.0.0.1:3128/pypi/simple/`);
  off by default: the variable overrides an index a task's own `pip.conf` names (pip-configuration tasks exist) and
  pip prints "Looking in indexes", which ends up in the traces. S1: agents' proxy traffic was 90% apt.
- **Task authors' sandboxes** (round jobs; internet through slirp4netns, no route to the host's loopback): the same
  package proxy, in "any host" mode, at `127.0.0.1:3128` in the sandbox. `http_proxy` points there, so apt (all plain
  HTTP on port 80) goes through the round job's cache, the rest of plain HTTP to the cluster proxy, and
  `PIP_INDEX_URL=http://127.0.0.1:3128/pypi/simple/` serves pip from the cache; other `https://` keeps the cluster
  proxy. `--author-package-cache apt+pip` (default) | `apt` (pip direct: the author does not see "Looking in
  indexes") | `off`. If the proxy cannot start, the sandbox keeps the cluster proxy; if the cache goes down, its
  requests go to the cluster proxy (pip straight to PyPI). Each author's requests are in
  `work/<spec>/author-fetch.jsonl`. Why: in wy0, 69% of authors installed packages, apt/pip took 7% of their session
  time, and 18% of those authors hit the 600 s command timeout in apt/pip (84 of 101 timeouts in apt), through the
  cluster proxy. Author replay (`package_cache_bench author`, 300 wy0 authors, 1,050 install commands, each run
  three ways side by side): proxy requests 13,156 -> 1,929 (apt) / 1,845 (apt+pip), bytes 34.4 -> 22.7 / 9.5 GiB,
  exit codes and 100,495 installed versions identical except two races in the authors' own background installs.
- **pip's own cache:** pip keeps downloads in `~/.cache/pip` only for https and trusted hosts, so wherever pip uses
  the cache's index it also gets `PIP_TRUSTED_HOST=127.0.0.1`, and files carry upstream's `Cache-Control` plus a
  `Date`: images and sandboxes end up with the same pip cache as without the package cache (before 2026-11-03,
  cache-on images had an empty one).

What it guarantees:

- **Same packages as without it.** apt `/pool/` files and `by-hash` indexes are immutable (by-hash verified against
  the digest in the URL); `InRelease`/`Release` and PyPI project pages are revalidated upstream (conditional GET) once
  older than 300 s (`--fresh`), so a build sees what upstream lists, minus at most a mirror's lag. PyPI pages are
  pypi.org's own (JSON or HTML) with file links made relative; files are checked against the BLAKE2b-256 in their path.
- **No corrupt files.** Concurrent requests share one download; clients stream it while it arrives but its last
  64 KiB wait for the length and digest checks; a truncated or mismatching download is dropped and its clients see
  a cut connection (apt and pip retry).
- **Upstream down:** retries, then a breaker (fail fast 30-300 s). An index validated within 6 h (`--stale-max`) is
  served (logged `stale`); anything else is a 503, which apt/pip report like a proxy failure: harbor_ns rebuilds the
  image, and `check()` classifies a build lost that way as infrastructure (`infra`), never as the generator's error.
- **Upstream slow:** a revalidation with no answer after 10 s (`--revalidate-wait`) serves the stored index (within
  6 h, logged `stale`) and completes for later clients. The proxy sometimes holds a request for a minute; apt drops
  an index after ~60 s, pip after 15 s, and a stalled `apt-get update` then retries for many minutes (the replay
  lost one build to its 30-minute timeout that way). A cold miss has nothing to serve and waits, as without the cache.
- **Slow or broken downloads:** concurrent builds share one download of a file, so a large file crawling on its
  proxy connection (below 1 MB/s for 30 s, over 50 MB left; replay: a 906 MB torch wheel at 0.4 MB/s while new
  connections got 2-25 MB/s) or a connection that breaks mid-body continues on a new connection with a Range
  request (a 206 for exactly the rest, at most 3 times). Checked by the same incremental SHA-256 and digest.
- **Cache down:** its supervisor (`CacheProcess`, in the round/worker process) removes the registry file, restarts it
  with backoff on its old port, and builds use the cluster proxy meanwhile. The cache exits when its driver dies.
- **Disk:** LRU eviction above the cap; below 10 GiB free on the scratch filesystem (`--min-free-gib`) it stops storing
  and streams straight through. Never on Lustre (no file locks needed anyway; the store is one file per object).
- **Warm start:** at its end a round job merges its store into the seed, one tar in
  `/data/aai/tbench/tasksynth/pkgcache/seed/` (`current.json` names it; the previous one is kept, older ones
  deleted; mkdir lock; at most 12 GiB). A new cache serves at once and unpacks the seed in the background (10 GiB:
  15 s on a quiet node, over 3 min on a busy one); a file fetched before the restore reaches it keeps the fetched
  copy, and a cache restarted mid-restore resumes it.

Operations: it starts by itself with the round job or campaign worker (nothing to launch). Disable it with
`--package-cache off` (round) or `harbor.package_cache = false` (campaign config); harbor_ns used outside these
(no `HARBOR_NS_PACKAGE_CACHE` in the environment) never uses it.

```bash
PY=/opt/aai/venvs/harbor/bin/python
$PY -m synth.package_cache stats --registry $_CONDOR_SCRATCH_DIR/pkgcache.json   # counters (inside the job)
$PY -m synth.package_cache seed-info                                            # current seed
$PY -m synth.package_cache publish --root <store dir>                           # merge a store into the seed
# warm the seed from a set of tasks (cache off/on replay, publishes with --publish-production):
$PY -m synth.package_cache_bench sample --out <dir> && $PY -m synth.package_cache_bench submit-replay --out <dir>
```

Not covered: npm (`registry.npmjs.org`, 3-4% of tasks and authors), uv (0 of 2,000 sampled authors, 0.4% of tasks)
and other HTTPS hosts (download.pytorch.org: 1.5% of tasks, but most of the bytes left in a warm build replay; GitHub,
model downloads). They tunnel through the cluster proxy as before: caching HTTPS needs a pull-through endpoint per
registry (npm would work like PyPI: a registry URL in the environment plus tarball URL rewriting) or TLS
interception, and an index a task names on pip's command line (`--index-url https://download.pytorch.org/...`)
beats any variable.

Per-job files: `<scratch>/pkgcache.json` (registry: URL, port, pid), `<scratch>/pkgcache.err` (the cache's stdout:
counters every 10 min), `<scratch>/pkgcache/logs/requests.jsonl` (every request: tag, URL, verdict, bytes, ms,
SHA-256, package, version). Round jobs write `rounds/<round>/pkgcache-stats.json` (`.s<N>` for the other jobs of a sharded round) at the end; each attempt's
`attempt-<k>.check.json` has `fetch` (counts) and `fetches` (packages: URL, version, SHA-256) of its build.

Design choice: one cache per job, not one shared service. A job's own builds are where the repeats are (a round job
builds ~1,000+ images; the 551 unique packages are fetched once per job, or never with the seed), so a shared
service would add little hit rate but a network service, discovery, cross-node traffic and a single point of failure
whose outage sends every job back to the proxy at once (the burst that collapsed it at ~1.5-3k sandboxes). Per job,
the cache lives and dies with the job, needs no ports beyond loopback, and an outage touches ≤48 builds.
Alternatives considered: apt-cacher-ng (apt only, needs a binary and its own config; PyPI would need a second
service, e.g. devpi with SQLite on local disk, or proxpi), squid (cannot cache HTTPS without TLS interception, and
the image would have to trust its CA), devpi (heavier than needed: we only need a read-through mirror). One small
asyncio process covers apt and PyPI with one store, one log and the integrity and fallback rules above.

## Protocol and sandbox limits of new campaigns (2026-11-14, #378 train/test audit fixes 1 and 4)

`synth.prepare` freezes these into every **new** campaign's `ops/campaign.json` (and `ops/protocol.json`). A campaign
frozen before keeps what it recorded: no `finish_notice` means no notice, and a `[harbor]` without the keys below keeps
the old limits. Re-preparing such a campaign with this checkout is refused (`freeze`: different settings).

* **Pi finish notice, v4 sticky (default on).** User, 2026-11-13: SFT trace generation keeps the notice, to avoid a
  train/test gap. Ported from aa-evals (#382, #395): `tb_agents/pi_context_limit.py` is the same guard code and notice
  text as aa-evals' (`finish_plan`, `FINISH_WARN`/`FINISH_FINAL`, `apply_sticky`, `notice_entry`, ...).
  `protocol.finish_notice` is the record aa-evals' pi_eval freezes (`finish_notice_record(40000, 16384)`: warn at
  40,000 tokens left, a 16,384-token final turn, `policy` `submit-at-context-limit-v4-sticky-notice`, `text_sha256` of
  the two notice texts). The worker passes `PI_FINISH_WARN_AT`/`PI_FINISH_RESERVE`/`PI_FINISH_STICKY`. The guard writes
  the notice map to `pi-cont16k/finish-notices.jsonl` (created empty at startup). `synth/archive.py` ships its parsed
  lines as `finish_notices` in each archive line, and `sft_qwen35/prepare.py` (sft branch) joins them to the training
  messages. A line without the key had no notice, so older exports read unchanged. A v4 attempt whose map is lost
  ships an entry without the map keys, so the SFT splice drops it ("unmapped").
* **Context: 131,072** (was 132,000), the eval protocol. Today's reference for it is aa-evals' pi_eval and the c30
  launcher (`aa_evals/campaign/launch.py:111`). eval_q35 moves to v4 at 131,072 for new runs (eval PR pending,
  branch `eval-harness-from-data`). It is also the engines' `max_model_len`, so a new campaign cannot adopt serving
  jobs started at 132,000 (`[serving] adopt_ops`): `synth/controller.py` `adopt_serving` refuses a different
  `context_tokens`.
* **Smoke attempts with `context_window`** (the hand-written campaigns' capped smoke uses 32,768) are below the
  notice's `warn_at` (40,000 tokens left), so every turn of such a smoke gets a notice, and below 24,576 left a
  "final turn" one. The smoke checks still hold: they look at reasoning, continuations and the zero-loss context-limit
  terminal, not at the notices. The window is left as it is: those TOMLs record existing campaigns, and the production
  driver (`task_synthesis/pipeline/production.py`) runs one uncapped smoke without a window.
* **Per-benchmark overrides (set by the user).** In the campaign TOML: `[protocol] context_tokens = N` and
  `finish_notice = "v4" | "WARN_AT:RESERVE" | "off"` for the whole campaign; `[[protocol.overrides]]` with
  `tasks = ["glob", ...]` and `context_tokens` and/or `finish_notice` for a task source within it (first match wins;
  `context_tokens` at most the campaign's; `tasks` must be a list of strings, each override a table). Each attempt's effective values go to `raw/<uid>/attempt.json` `protocol`
  and to `pi-cont16k/protocol.json`. Differences from eval that come from these overrides are deliberate.
* **Sandbox limits** (`synth/sandbox_limits.py`). Generation stays a bit stricter than the eval:
  | | new campaigns | before | eval (eval_q35) |
  |---|---|---|---|
  | memory (`harbor.sandbox_memory`) | PSS guard at max(1.5 x `memory_mb`, 24 GiB), `pi` spared unless alone | 1.0 x `memory_mb`, 4 GiB undeclared, kills `pi` too | memcap max(2 x, 32 GiB), Pss_Anon + Pss_Shmem, spares `pi` |
  | stack (`harbor.stack`, or `harbor.stack_kb` to set a limit) | unlimited, as eval (`"unlimited (as eval)"`) | 8 MiB | unlimited |
  | disk (`harbor.diskcap`) | max(3 x `storage_mb`, 30 GiB): eval's default rule, since synth tasks are not TB tasks (`storage_mb` 10240 -> 30 GiB) | none | per family (#425) |
  | network | offline + package allowlist proxy (unchanged) | same | full internet |

  Stack: not 64 MiB, because glibc sizes a new thread's default stack from the soft limit (2 MiB when unlimited). A
  64 MiB limit would give 64 MiB threads, so tasks with `RLIMIT_AS` / `ulimit -v` caps would behave differently from
  the eval. A runaway recursion now grows until the PSS guard (stack pages are anonymous memory) or `synth.memory_guard`
  stops it, as in eval. Eval's memcap (`HARBOR_NS_MEMCAP`) is set to 0 for every attempt, so it never runs next to
  synth's own guard. A partial `[harbor.diskcap]` or `[harbor.sandbox_memory]` table is merged into the defaults:
  `{min_mb = 40960}` keeps factor 3, and `{factor = 2.0}` keeps the 24 GiB floor and the spared `pi`.

  For campaigns of TB tasks, `harbor.diskcap_family = "tb2.1" | "tblite"` takes eval's per-family and per-task caps
  (`DISKCAP_FAMILIES`, copied from eval_q35/common.py). The family's `one_per_worker` list is recorded but not
  enforced. The disk cap is harbor_ns/diskcap.py from the eval branch: RLIMIT_FSIZE on every process of the box plus
  a watchdog that kills the writer. Each attempt's peak and kills go to `raw/<uid>/diskcap.json`.
* **Hung-session watchdog** (`harbor.hung_watchdog`, #531; `tb_agents/hung.py`, the rule aa_evals pi_eval runs): new
  campaigns get it (idle 600 s, the most lenient suite of pi_eval's measured table); `false` turns it off;
  `{idle_sec, stall_sec}` sets the limits. Once a minute per attempt the worker checks Pi: a session doing nothing
  (no model request in flight, no tool call outstanding) for longer than `idle_sec` is stopped and recorded as
  `infrastructure_failure: HungSession` (a mid-run failure: counted toward `max_infrastructure_failures_per_task`).
  The agent's end (Harbor's `trajectory.json`) disarms it, so the verifier is never cut. The stall rule needs the eval
  profile's `wire-responses/` and the accounting rule a response-budget guard; synthesis runs neither by default.
  Frozen campaigns without the key run without it.

## Design rules (each from an incident in the per-teacher campaigns)

* **Separate jobs.** Serving, the controller and workers are independent HTCondor jobs, so a
  controller restart (code reload, node loss) never kills a rollout; a lost worker or replica
  only interrupts its own attempts, which are kept as truncated data.
* **Single publisher.** The controller records an attempt in `controller/state.json` before
  writing `ops/workers/<i>/inbox/<uid>.json`; completion is `raw/<uid>/worker-result.json`.
  Ownership uses mkdir leases (Lustre has no `flock`).
* **Pinned replicas, per-backend admission.** Each attempt stays on one replica/rank (prefix
  cache). `dp_internal` teachers get a per-attempt `X-data-parallel-rank`. Two admission modes:
  * `admission.mode = "kv_fit"` (`synth/admission.py`, new campaigns). It runs as many agents as maximizes each
    backend's generated tokens/s within its KV:
    * **Working set** (measured): the in-flight KV gauge × capacity, plus the idle agents' cached contexts. Each
      agent's context guard writes its context to `PI_CONTEXT_STATE`, and the worker heartbeat carries it. Serving
      publishes `kv_capacity_tokens` and `kv_block_tokens`.
    * **Ramp guard**: a backend admits while max(working set, agents × mean demand) ≤ target × capacity. The mean
      demand is a slow average floored by `agent_tokens_prior` (TMax: 20k).
    * **Target**: each backend hill-climbs gen tok/s from 0.4 (`target_search = "throughput"`). Cache reuse is
      only a collapse brake (< 0.5). It is vLLM `cached_tokens` against the previous turn's prompt, and needs the
      teacher tunable `prompt_tokens_details = 1`.
    * **Fast fill** (CPU-tier backends, opt-in `inflight_fill_below`, 0.55 for Super): a decision window filled to
      the target whose in-flight KV stayed below the threshold without a preemption grows the target x1.3 at once.
      Long-tool workloads leave the GPU batch short of requests at one GPU's worth of target (hero CalibForge: 35
      running of ~185 agents per B200, 3.6k gen tok/s for an hour; 6.0-6.3k at target 2.0).
    * **Other brakes**: in-flight KV ≥ 0.92 (resume at 0.85 after 30 s), preemptions, and more than 64 queued
      requests.
    * **Shared serving**: campaigns publish their idle working sets under `<serving ops>/serving/load/`.
    * **Workers**: `workers.max_count` lets the worker pool grow to the agents admission can run.
    * **Stalled supervisors**: if a supervisor's files are stale, the controller probes the engine's `/health` and
      `/metrics` itself. Attempts are interrupted only when an engine restarted or its job ended.

    vLLM's `kv_cache_usage` gauge counts only in-flight blocks: idle agents' cached prefixes count as free.
    Admitting on it ran the 2026-10-17/18 Super campaigns at 4-7% mean KV and 12-17 sequences per B200 (860 gen tok/s
    per GPU). kv_fit ran Super on TMax at ~4.2k per B200 in steady state; see `docs/campaigns/synth-20261021.md`,
    which also covers why the free KV is not idle headroom for Super (each request allocates and frees ~20k tokens
    of Mamba/MTP state).
  * The original limit controller (frozen campaigns without a mode):
    * pause on KV ≥ `emergency_kv_high`, a KV growth forecast, preemptions (limit × 0.8) or
      queueing;
    * resume below `resume_kv` for 90 s;
    * raise the limit by `limit_step` while KV < `increase_below_kv`.
* **Dispatch saves once per tick, and ticks are 15 s.** A tick's assignments are recorded first, then the
  state is saved once, then the inboxes are published. Saving the whole state per assignment (31 MB at TMax
  scale) once stretched the 15 s loop to 1-3 min. The loop sleeps only the rest of the period, and slow ticks
  log their phases to `controller/slow-ticks.jsonl`.
* **Inodes.** One wire log per attempt (`synth/wire.py`; the old per-request archives used ~4
  files per model call and exhausted the /data quota); attempts packed into shards (above); compile caches on
  node-local scratch; telemetry and heartbeats are best-effort and never fatal.
* **Warm engine caches** (`synth/engine_cache.py`). A cold Super engine spent ~14 of its 14-17 start-up minutes
  rebuilding FlashInfer JIT modules, the MoE autotune and torch.compile. Engines keep the checkpoint, CUDA_HOME
  and their caches under `/tmp/synth` (HTCondor mounts /tmp per job, so the paths that ninja lines and vLLM's
  compile hash embed are the same in every job). Before vLLM starts, a job extracts the snapshot of its engine
  configuration from `/data/aai/tbench/synth/shared/engine-cache/<key>.tar`; the first healthy engine
  without one publishes it (key: argv without port, compile env, venv distributions, GPU model). vLLM's cache root
  (FlashInfer MoE autotune results, torch.compile artifacts) is never shared and is emptied before every engine start:
  an engine that loads one an earlier start wrote, even its own job's, runs ~16% slower per step. Seeded Super start:
  ~4.5 min instead of ~14. `serving.engine_cache = false` turns the snapshots off.
* **Scale-down.** Once nothing is left to assign, idle serving jobs and workers are released,
  so six-hour tail tasks do not hold every GPU; they come back if work reappears.
* **Six-hour agent limit everywhere.** Bundles rewrite `[agent]` *and every `[steps.agent]`*
  timeout (multi-step tasks such as `ot-engine-debug` kept 1800 s step limits before);
  verifier and build limits stay original.
* **Bash default timeout.** The `bash180_v1` Pi bundle kills commands after 180 s unless the model passes a
  larger `timeout`. The tool description states this. Hung commands (interactive prompts,
  `server &` + `wait`) once held about half of all attempt-hours.
* **Sandbox devices.** harbor_ns bind-mounts /dev devices onto whiteout (`c 0 0`) nodes, so
  `find -type f` and `grep -r` see character devices, as in Docker. Over regular files they read
  /dev/urandom for hours.
* **Sandboxed memory.** `harbor.resource_guard` (default on) enables the harbor_ns PSS guard at
  each task's declared memory, so one runaway task cannot get a whole worker held. New campaigns use
  `harbor.sandbox_memory` (above); `synth.memory_guard` still keeps the whole job inside its memory.
* **Image builds.** harbor_ns build steps run on `harbor.build_cpus` (4) cores and are killed when their process
  tree passes `harbor.build_mem_gb` (24 GB) RSS. RTS: nvm compiling node from source took 90+ GB and whole
  48-slot workers.
* **Verifying attempts hold no KV.** `kv_fit` admission counts an attempt against its engine only until the agent
  finishes (`post_agent` in the worker heartbeat). On the RTS pilot 83% of live attempts were in a 4-5 min verifier.
* **Offline sandboxes** (`harbor.package_proxy`). Their `no_proxy` also lists `.internal`, `.lan`, `.corp`,
  `.home.arpa`, the loopback and RFC 1918 ranges and the names in the image's own `/etc/hosts` (except the registries
  the proxy serves): tasks name their
  services (`api.lab.internal`) in `/etc/hosts` from the agent's shell after the agent started, so the per-exec
  `/etc/hosts` names never reached its environment and the package proxy refused them (S1 traces). The proxy's 403
  for a local host says to connect directly.
* **Cluster proxy.** proxy.internal:8080 fails at ~1.5-3k concurrent sandboxes. `admission.proxy_probe` stops new attempts
  while apt/PyPI/astral.sh fail through it; `harbor.verifier_toolchain` runs pytest verifiers offline
  (`synth/verifier_toolchain.py`); `verifier_infra_retries` re-run only verifiers whose tests never started.
* **Worker scratch.** Workers evict harbor_ns image caches before every attempt, against thresholds
  relative to the filesystem size. Each attempt's selected task copy is deleted after it runs. On TMax,
  thousands of distinct images once filled a worker's scratch and failed 7,391 attempts in minutes. harbor_ns deletes
  a failed build's image directory (its log stays in `<shared>/logs`), eviction takes orphaned directories whose build
  lock is free, and every build appends its image size and the scratch left to `<HARBOR_NS_LOCAL>/logs/builds.jsonl`
  (stderr warns below 15% free): at 1,000 builds per task-synthesis round job, failed builds filled 200 GB.
* **Infrastructure failures do not use up attempts.** They have their own cap,
  `train.max_infrastructure_failures_per_task` (32; RTS ran 4), after which the task stops. `WorkerInterrupted` is one
  of them.

## Training modes

* `tokens`: cycle all tasks until `train.tokens_target` saved usable tokens (the 110M campaigns).
* `per_task`: give every task `train.trajectories_per_task` trajectories, counting the usable
  records of `train.baseline_manifests` (an existing corpus); fewest-first, live attempts count.
  A new trajectory counts once its attempt finished without an infrastructure interruption
  (interrupted ones stay in the corpus). `max_extra_attempts_per_task` bounds retries for tasks
  that keep failing to export.
* `per_task` + `train.tokens_budget`: a fixed random subset of a large task set. This is how the
  TMax-15K and CalibForge campaigns ran. Tasks open in the order of `train.tasks` (a pre-shuffled list).
  `train.budget_stop` decides how the budget ends a campaign:
  * `"saved"` (default since 2026-10-24; user: "keep running at max concurrency until we hit the desired token
    budget"): tasks keep opening while the saved tokens are under the budget, so every backend stays at its
    admission limit and serving is never drained for a tail. The moment the budget is reached, training closes:
    live attempts are cut (kept as truncated data), so the last opened tasks may end below their count. Saved
    tokens are estimated as exported tokens plus finished, not yet exported trajectories at the running mean
    (export passes lag by ~30 min at 35k trajectories).
    While nothing can open (validation behind, list exhausted) and the budget is open, free slots go to up to
    `train.overflow_per_task` (1) extra trajectories per opened task rather than idling (user: "9 rather than 8 is
    fine").
  * `"projected"`: tasks open only while the projected tokens are under the budget. Projected tokens are the
    estimated saved tokens plus, for each opened task, its trajectories still owed × the mean per trajectory.
    Every opened task gets its full count, and the tail rule cuts the end.

  The mean per trajectory is floored by `tokens_per_trajectory_prior` until 90% of started attempts have finished,
  because early finishers are short. Once training has stopped for good and nothing is in flight, serving is
  released before the final exports and verification, which need no GPU.

  With `train.validation` (a `{task: bool}` file from a no-agent sweep running ahead), a task opens only once
  it is validated, and invalid tasks are skipped. Opening looks ahead `validation_window` (256) positions past
  the first unresolved one. A valid task behind the furthest opened position still opens after the budget
  closes, so the subset is exactly "every valid task before position P".

Evaluation (`[eval]`, e.g. TB2.1 89 × 3) uses attempt-major scheduling, an admission share,
same-seed retries for explicit infrastructure failures only, and never selects by reward.

## Token accounting

`synth/export.py` reconstructs the native Pi session graph. Each native event is counted once;
branch/compaction boundaries reference context instead of copying it; the system prompt and tool
schemas (from the first served request in the wire log) are counted once per trajectory. Context
tokens count; loss weight is 1 only for complete assistant turns (`stop`/`toolUse`) that no shared
annotation zeroes (the context guard's synthetic terminal is 0). The serialization is the
teacher's (`[counting]`): the publisher chat template with fixed kwargs, or the DeepSeek V4/V4.1
encoder module, with the checkpoint's tokenizer.

## Teachers

| slug | serving | data (pi-cont16k-110m, 20261016T171932Z) | TB2.1 pass@1 |
|---|---|---|---|
| glm53-flash | 4×B200, DP4/EP4 external LB, MTP2 | 111.07M tokens, 1,677 trajectories | 76.03% |
| qwen38-flash-next | 4×B200, TP4+EP, MTP3, Engram offload | 111.48M, 1,534 | 73.41% |
| deepseek-v4-flash-0731 | 4×B200, DP4/EP4 internal, DSpark3 | 111.60M, 1,728 | 67.04% |
| deepseek-v41-flash | 4×B200, TP4, DSpark5, Engram offload | 110.46M, 1,536 (finalization blocked on 2 verifier files lost to EIO) | 78.65% |
| deepseek-v41-flash-1017 | 4×B200, DP4/EP4 internal, DSpark5, shared Engram | 123.03M, 1,710 (synth campaign 20261017) | 78.3% (82.6% of 253 resolved) |
| nemotron35-super | 1×B200/replica, MTP1 | 110.68M, 1,905 | 56.93% |
| nemotron35-bolt | 1×B200/replica, FP8-on-load, DFlash3 draft | 111.19M, 1,742 | 45.32% |
| qwen38-27b | 1×B200/replica, MTP3 | 104.88M, 1,518 (stopped by the /data inode quota) | 59.4% of 261/267 |
| k2-horizon | 1×B200/replica, ngram3, cached-MoVA plugin | 9.59M (cancelled: ~400–700 output tok/s per B200) | — |

`synth/tests/test_core.py` checks every spec against the vLLM command its campaign recorded
(`synth/tests/fixtures/recorded-commands.json`; Super and GLM against their launcher scripts).
Runtime caveats live in each spec's header (e.g. Bolt's and Qwen3.8-Flash-Next's runtimes overlay
the shared `venvs/vllm032`, which has since moved; GLM served from a node-local venv snapshot).

Campaigns run on `synth/` itself (DeepSeek-1017, the Super top-up to 8 per seed task, and Super on TMax-15K
and CalibForge) are in [`docs/campaigns/synth-20261017.md`](../docs/campaigns/synth-20261017.md). That doc
includes the token-matched 230M subsets.

## Adding a teacher

1. Fetch the checkpoint:

   ```bash
   $PY -m synth.fetch <repo_id> <revision> /data/aai/tbench/synth/shared/checkpoints/<slug>/<revision> --submit
   ```

   This runs a CPU job. It downloads to node-local /tmp and `dd`s each file to /data: the Hub client
   writes Lustre at a few MB/s, while `dd` runs at ~1 GB/s. Every file is verified against the Hub's hashes.
   `checkpoint-ready.json` is written last.
2. Copy the closest spec. It needs:
   * identity: `model`, `revision`, `checkpoint`, `served_name`;
   * `[runtime]`: venv, pinned `vllm_version`, CUDA_HOME mode, node-local cache env;
   * `[serving]`:
     * `mode` = `replicas` | `dp_external_lb` | `dp_internal`;
     * `gpus_per_replica`, `entrypoint`;
     * `args`, with `{model_dir}`/`{port}`/`{max_model_len}`/tunable placeholders;
     * optional args per tunable;
     * `stage_extra` and `kernel_wheels` if needed;
   * `[request]`: Pi client model, thinking level, sampling overrides, history-reasoning adapter;
   * `[counting]`: the serializer;
   * resource and admission defaults.
3. Tune serving before a large campaign:

   ```bash
   $PY -m synth.tune synth/campaigns/<slug>-tuning.toml --submit
   ```

   One GPU allocation runs the candidates of `[tuning]` back to back. A candidate is teacher overrides,
   tunables and concurrency levels. The load is `synth.bench`, a replay of a finished campaign's recorded Pi
   requests with their tool gaps (`gap_scale` stretches them; with `closed_loop = true` each agent's history
   carries its own generated answers, which anything reusing the previous answer needs). A candidate's `check`
   runs a client script against the healthy engine before its replay. Results go to `ops/tuning/results.jsonl`
   (closed loop: per-turn cache stats in `ops/tuning/turns-<candidate>.jsonl`). Candidates can be appended to
   `ops/tuning/plan.json` while the job runs. Candidates share one engine cache, and the job exits once
   `ops/tuning/DONE` exists.
4. Run a smoke campaign. The smoke gate cross-checks token counting against vLLM.

## External task sets

`python -m synth.datasets` turns a published task set into a campaign's inputs:

* `bundle`: the Harbor bundle;
* `tmax-base`, `register-base`: shared harbor_ns bases, published without nsfake xattrs;
* `calibforge`: Dockerfile builds instead of Docker Hub pulls;
* `decontaminate`: seeds/METHODOLOGY.md §6.1/6.3 features against held-out sets; flagged tasks are
  excluded;
* `order`: a seeded shuffle;
* `tmax-validation`, `validation-status`: no-agent validity sweeps written as `train.validation`. A `nop`
  run counts only if the tests ran. `--min-failures N` defers a false verdict to the Nth failed run;
  `--assume-valid FILE` opens tasks before their sweep decides. A task with no verdict holds the opening cursor,
  so `train.validation_window` must cover tasks that may never get one.

`python -m synth.prefix_bases` shares images between tasks whose Dockerfiles start with the same install-only
prefix: one base `local/<dataset>-prefix-<id>:1` per prefix; each task Dockerfile becomes `FROM` it plus its own lines.
With `harbor.persist_images = false` task images stay in worker scratch, and `workers.affinity` +
`train.image_groups` keep a base's tasks on one worker.

Examples: `synth/campaigns/nemotron35-super-tmax.toml`, `synth/campaigns/nemotron35-super-rts-1b.toml`
(record: `docs/campaigns/synth-20261026-rts.md`).

## Tests

```bash
PYTHONPATH=/opt/aai/venvs/testdeps:$PWD \
  /opt/aai/venvs/harbor/bin/python -m pytest -q synth/tests tb_agents inference/test_pi_cont16k.py \
  inference/test_sticky_proxy_consolidated.py inference/test_nemotron_reasoning_guard.py harbor_ns
```
The finish notice and sandbox limits: `synth/tests/test_traintest.py`, the finish-notice tests at the end of
`tb_agents/test_pi_context_limit.py` (aa-evals' own, plus the synth guard's options), `harbor_ns/test_memcap.py`,
`harbor_ns/test_diskcap.py` (from the eval branch) and `harbor_ns/tests/test_campaign_limits.py`.
(`venvs/testdeps` holds pytest only; the harbor venv has no pip.)
