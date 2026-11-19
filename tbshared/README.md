# tbshared: the shared layer of the cluster pipelines

One maintained copy of the code that the pipelines share:

- task synthesis (`task-synthesis`, `data-synthesis`, `glm-data-synthesis`);
- checkpoint evals (`eval`);
- AA evals and M5 (`aa-evals`, `aa-serving`);
- SFT (`sft`).

The goal is that an optimization is made once and every pipeline gets it. The user approved the layer on 2026-11-14: "let's share as much as possible (as long as it makes sense); so that we don't need to maintain and optimize multiple similar parallel branches". It came out of #378's port audit, recorded in `/data/aai/aa_evals/audits/port_matrix_20261114/matrix.md`.

## On the trunk `main` (#480)

`main` holds every pipeline in one tree, and its `tbshared/` is the canonical copy (`tbshared/TRUNK`, no `PIN`):

- **Changes land here,** through PRs into `main` with this directory's tests. Branch `shared` and the vendoring below
  serve the production branches until each one is cut over to `main`; `sync` refuses the trunk.
- **`test_pin.py` on the trunk** checks the `ALIASES` copies (e.g. `aa_evals/common/condor_cost.py`) against
  `tbshared/` itself, byte for byte, and that no `PIN` exists. `python -m tbshared.sync --check` does the same.
- **Pipeline config per package:** `aa_evals/pipeline.json` (aa-evals, with its worker shapes), `eval_q35/pipeline.json`
  (eval), `synth/` and `task_synthesis/pipeline.json` (task-synthesis), `task_synthesis/serve/pipeline.json`
  (aa-serving: names the serving launcher's numerics section; serve_teacher's fill records name its --pipeline).
  The file nearest above the calling module governs it (below).
- **Numerics guard:** every pipeline a package names is checked against its section (`test_numerics.py`,
  the differences `main` is known to have are listed in `numerics.TRUNK_KNOWN_GAPS`, each with its reason, and
  reported apart; anything else fails); `python -m tbshared.numerics check` with no pipeline checks them all, the
  same way.

## How it is laid out

Branch `shared` holds `tbshared/`. Every pipeline branch carries a **byte-identical copy** at the same path, `tbshared/`, so imports read the same everywhere (`from tbshared import condor_cost`). Each copy also has `tbshared/PIN`, the `shared` commit it came from.

| Rule | Why |
|---|---|
| Vendored copy, pinned by commit | Every pipeline freezes jobs by snapshotting its own tree (`git archive` in eval_q35, frozen worktrees elsewhere). A path on /data or a submodule would not be in that snapshot. **When a pipeline migrates, each of its snapshot mechanisms must include `tbshared/`.** One example is aa-evals' `harbor_mswea.run.CODE_DIRS`. |
| `tbshared/pipeline.json` (pipeline-local) | The pipeline's own config, read through `pipeline_config.py` (below). Like PIN and ALIASES it is in `sync.LOCAL`: `sync` keeps it across bumps, the pin check ignores it, and one committed to `shared` by mistake is never vendored. It is committed on the pipeline branch, so frozen snapshots of `tbshared/` carry it. |
| `tbshared/ALIASES` (pipeline-local) | Some pipelines snapshot only some directories, for example aa-evals' per-benchmark copies of `aa_evals/`. Such a pipeline lists in-tree copies (`DEST SOURCE`, e.g. `aa_evals/common/condor_cost.py condor_cost.py`). `sync` writes each copy byte-identical, and the pin check covers it. The old import paths keep working inside partial snapshots, and still cannot fork. An alias source must not import `tbshared` (checked). When a line is removed from ALIASES, delete its old copy in the same change. |
| `test_pin.py` in every pipeline | It fails when `tbshared/` or an alias differs from its PIN, or when PIN is missing. A local "quick fix" in one pipeline's copy then fails that pipeline's tests instead of forking. It is skipped only on `shared` itself and outside a git worktree (frozen snapshots). |
| Changes only on `shared` | Open a PR against `shared`, with this directory's tests. Then bump each pipeline with `python -m tbshared.sync` in a PR on that pipeline's branch, where its own tests must pass. A bump is a version change, not a port. `sync` refuses commits not merged into `shared` (`--allow-unmerged` while a `shared` PR is under review) and refuses to run in `shared`'s own worktree. |
| `python -m tbshared.drift [--fetch]` | Lists every pipeline's PIN against `shared`'s head, and the shared commits each one is missing. A pin that is not on `shared` is flagged. |

```
git fetch origin shared
python -m tbshared.sync                 # in a pipeline worktree: vendor origin/shared, write tbshared/PIN
python SHARED_WORKTREE/tbshared/sync.py origin/shared --root .   # first migration (no tbshared/ here yet)
python -m tbshared.sync --check         # what test_pin.py checks
python -m tbshared.drift --fetch        # who is behind
```

## What is shared, and what is not

**Shared (mechanism).** Everything here is stdlib-only, so a single file can run frozen (the watcher does):

- `condor_cost.py`, for CPU jobs:
  - submission on a bid ladder, starting on the lowest rung whose nodes have room for the job's shape now;
  - the per-user watcher job, which moves an idle job up a rung in place. It is a ladder job itself, bidding 36 on its own 36/56 ladder; while it waits, every `ensure_watcher` call does one escalation pass in its place;
  - `periodic_release` on memory holds (`auto_release=False` for jobs that cannot resume after a restart);
  - with it (1.5), the serving jobs' memory net (`holds.py`): a literal `request_memory` becomes `BASE + 25% of BASE x memory holds so far`, so a worker that really needs more does not loop hold, release, hold at the same size. The release bound is unchanged (`NumHolds <= 5`). The base is the shape: `parse_shape` (the value's leading size, as before 1.5), the ladder's fit check and `ledger.read_submit` read it. A submit file that sets its own `periodic_release` (e.g. `holds.submit_lines`) is left as written, so nothing grows twice. **A hold costs units once a job is filled:** at 16 GB per CPU, the next 25% is above what is free (4c/64G grows to 80G: 5 units, not 4). That is intended: a job that needs more memory pays for it.
  - shape arithmetic (`fits`, `units`: the charge of a request as given; `fill_shape`, `shape_caps`, `fill_plan`);
  - filling to the binding resource (1.5). The user's rule: "since mem is free for enough cpus, make sure to size the other one accordingly, since we will pay the same. according to the one that binds". `fill_shape(cpus, mem_gb, gpus=0, node=None)` gives GB = max(GB, 16 x CPUs) and CPUs = max(CPUs, ceil(GB / 16)), within `shape_caps`. Whole CPUs can add under 1 unit (2c/200G becomes 13c/208G: 12.5 units become 13). No node class here has 16 GB per CPU (B200 and A100 7.9, H100 10.5, the fat CPU nodes 15.7), so:
    - **GPU jobs:** CPUs and GB are at most the per-GPU share x GPUs (TotalCpus / TotalGPUs, TotalMemory / TotalGPUs) of every class the job may land on. That is the minimum over every `CUDACapability ==` its requirements name, OR'd ones included, or over every GPU node when none is named. A filled job then never strands GPUs. Without class info (condor_status down, a class with no node, another comparison on CUDACapability) the job is not filled. On 11-14, per GPU: B200 48 CPUs / 377 GB; H100 24 / 378 (8-GPU) and 24 / 251 (4-GPU); A100 16 / 250 (8-GPU), 32 / 251 or 236 (4-GPU), 42 / 335 (3-GPU); CUDA 7.5 4.5 / 94. Snowball 8xB200 at 1050 GB becomes 66c/1056G (cap 384c/3022G).
    - **CPU jobs:** CPUs at most what a CPU job can get on any node, and GB at most the most memory among nodes that give it the filled CPU count. A node with GPUs keeps min(16, 8 x GPUs) CPUs and min(64000, 32000 x GPUs) MB for GPU jobs, as in `fits`. On 11-14: 190c/100G could reach 3023 GB (the 192c CPU node), 256c/100G 2959 GB (the B200s), 400c none. The fill only goes where it is free: `fill_plan` plans the batch unfilled and filled, and the filled plan is **worse** if it starts any job on a higher rung, or leaves more jobs that fit on no node now. A worse plan gets RAM capped at the bottom rung's free GB per free CPU (tried as its own plan); if that is still worse, the batch stays unfilled.
    - **Fixed bids** (`submit(sub, bid)`) never move up, so the check runs as a one-rung ladder `(bid,)`: the fill is kept only if the room at floors <= bid stays at least min(n, the unfilled room). For example, 16c/128G to 256G halves the room.
    - **One place fills:** `fill_plan` rewrites the job text in `submit` and `submit_file`, and the bid plan reads that text's shape (`plan_bids` takes the shape as given). `fill=False` opts out per call. The watcher is not filled (1 CPU / 2 GB).
    - condor_status is read once per `STATUS_TTL` (300 s) and cached. A failure is cached for `STATUS_FAIL_TTL` (60 s), so one submit does not wait 3 x 120 s.
    - A request that is not a plain number (`$(NCPU)`, `4096 * 2`) is not filled, and keeps AUTO_RELEASE and the memory net as before.
  - **Fill decisions are data:** every batch appends one line to `fill.jsonl` next to its submit file (where the job's condor files go). Each line has `t`, `pipeline` (from the `tbshared/pipeline.json` above the module), `batch`, `submit_file`, `n`, `gpus`, `before`, `after`, `decision` (filled, capped, unfilled, unchanged or off), `reason`, and `plans` (unfilled / filled / capped / used). Records also have `nowhere` (jobs that fit no node now, per plan). `python -m tbshared.condor_cost fill-report DIR...` (unreadable files and malformed records count as bad lines; it never raises) counts batches and jobs per decision, the capped share and the pipelines, and gives the rung mix as submitted against the unfilled plan, so a campaign shows whether filling moved jobs off the cheap tier.
  - the bad-node list: `bad_nodes()`, read at every submission (#531 A9: a long-lived watcher kept the list it imported): `BAD_NODES` as this file says now, plus the live file `/data/aai/tbench/shared/bad-nodes.txt` (`$TBSHARED_BAD_NODES_FILE`; node names, spaces or commas, `#` comments), which reaches running processes started from frozen copies too. Never read `BAD_NODES` itself at submission.
- `reslog.py`: the one cgroup sampler of every pipeline (#445 item 2: tbshared's, aa-serving's serving sampler, eval_q35's and synth.telemetry's merged). It records memory, peak, rss/shmem/cache/dirty/writeback (`stat`, raw; `mem_stat`, normalized), `hard` (non-reclaimable: what to size RAM from), oom/failcnt, CPU, `cores`, run-queue wait, /dev/shm and scratch use, and with `--procs` the large processes, from the job's own cgroup. It exits within 1 s of its parent (`run OUT INTERVAL --parent PID`), or runs as a thread (`reslog.start`); `python -I tbshared/reslog.py summary FILE` gives start-up and serving peaks.
  - CPU contention (1.5). Condor jobs here see cgroup v1 (`cpu,cpuacct:/htcondor/<slot>/...`) with no CFS quota (`cpu_quota_us` -1, from `cpu.cfs_quota_us`, or `cpu.max` on v2): CPUs are limited by the cpuset (affinity mask) alone, so CFS throttle counters stay 0 and are not logged. The cost of a cpuset is run-queue wait. It is read per thread from `/proc/<tid>/schedstat`, for the threads of the job's cgroup and its descendants (v1 `tasks`, v2 `cgroup.threads`):
    - `cpu_runq_wait_ms` / `cpu_run_ms`: the summed deltas since the previous sample (schedstat fields 2 and 1), over tids read in both samples. wait / run is the share of extra time spent queueing.
    - **Both are lower bounds.** Threads that start or exit within the interval drop out, and so does a reused tid (a negative delta). They are null on the first sample.
    - Coverage: compare `cpu_run_ms` with the `cpu_sec` delta. Short-lived processes (tool commands) mostly drop out: on a live interactive job (gpu133, 11-14) the matched tids held 9.7 of the cgroup's 28.8 CPU-s in one minute. Wait includes wakeup latency even when cores are idle (about 0.6 s/min there at 0.5 of 8 cores busy), so compare it across load levels rather than reading it as absolute delay.
    - `cpu_runq_threads`: the job's tid count; `cpu_runq_sampled`: how many were read; `cpu_runq_matched`: how many were in both samples. Above 5,000 tids a deterministic subset is read: a fixed hash of the tid under a threshold, so two samples' subsets nest.
    - Unreadable files give null; a field never fails the sample.
- `fastio.py`: writes to /data that leave no dirty page cache in a job's memory cgroup:
  - `copy_direct` / `copy_tree_direct` (O_DIRECT);
  - `append_line` (fsync + drop);
  - `trim_torn_tail`.
- `holds.py`: held server jobs, under the policy decided on 2026-11-14. A held server must never block its run.
  - **Memory holds** are released by condor with 25% more memory per hold, at most twice (optim-servers' aa-serving net, moved here unchanged).
  - **Any other hold, or a memory hold past that bound:** the run removes the job, but only while it is still held, and submits a replacement that excludes the node (`remove_held`). After memory holds the replacement gets 1.5x the memory. There are at most 3 replacements per server.
  - **A user's own hold** (code 1) is left alone.
- `ledger.py`: credits from HTCondor user event logs (condor_history keeps only the newest ~10k records):
  - run segments, one per execute (001) up to its end event, or open;
  - per segment: host, CPU time, peak memory, holds, and the requested shape;
  - open or late-closed segments end at the job's queue state or last output write (`close_stale`);
  - floors per node from `condor_status` or saved snapshots, never guessed from host names;
  - `segment_cost` and `summarize`: job-h, GPU-h, unit-h, credits at floor and bid, use against request;
  - CLI: `python -m tbshared.ledger 'LOG_GLOB' --since ISO --until ISO --group node-kind,floor`.
- `pipeline_config.py` (1.5): reads the pipeline-local `tbshared/pipeline.json` (below).
- `numerics.py`, `numerics_record.json`, `test_numerics.py` (1.6): the numerics guard (below).
- `sync.py`, `drift.py`, `test_pin.py`: the tooling above.

**Planned next, in this order:**
1. Server release, and starting dependents only once a server is ready (in task-synthesis since #413; to share).
2. One serving harness that merges task-synthesis's and aa-serving's `serve_teacher.py` with eval_q35's `serve_job.sh` start-up checks.
3. One recipe per model, plus workload profiles.

In a recipe, the fields that affect numerics will be marked and guarded by a test: weights and quantization, KV dtype, kernels, chat template, parsers, sampling defaults. Changing any of them needs a new eval baseline. Until the recipes exist, the numerics guard below covers the specs as they are.

**Per pipeline (policy and protocol).** Each pipeline keeps:
- its own config for bids, ladders and steps, shapes and caps;
- its protocol: eval scoring and memcaps, the Pi harness and dispatch, synthesis logic, SFT recipes.

For example, aa-evals workers use the 26/36/56 ladder, and so do eval_q35 workers from the eval port PR on. Task-synthesis workers do not wait at 36, because waiting starved the servers. The shared code takes policy as arguments; its defaults are aa-evals'.

## Per-pipeline config: `pipeline.json`

A small, generic file. A pipeline branch commits it next to its vendored copy (`tbshared/pipeline.json`, never on
`shared`); on the trunk each pipeline package commits its own (`<package>/pipeline.json`). Lookup
(`condor_cost.find_pipeline_json`): from the calling module's file up to the repository root, the first
`<dir>/pipeline.json` or `<dir>/tbshared/pipeline.json`, so a package's own file wins and a vendored branch's single
file covers everything:

```json
{"pipeline": "aa-evals",
 "worker_shapes": {"tb21": {"slots": 32, "cpus": 10, "mem_gb": 128, "basis": "one line: where the numbers come from",
                            "cpu_model": {"p95_full_slot": 13.25, "source": "memrun/report.md"}}}}
```

- `pipeline` (required): the pipeline's branch name, e.g. for a numerics record.
- `worker_shapes` (optional): per key (a benchmark, a job kind), positive integer `slots`, `cpus`, `mem_gb`, and a one-line `basis`; optionally `cpu_model`, an object of non-negative numbers or strings (`source` a string). Unknown keys are errors.

```python
from tbshared import pipeline_config
pipeline_config.pipeline_name()            # the caller's; raises PipelineConfigError when missing or malformed
pipeline_config.worker_shape('tb21', default=None)   # a copy of the shape; `default` when the file or key is missing
pipeline_config.load(path=None, start=None)          # the whole validated dict (`start`: a file whose config to read)
pipeline_config.find(start=None)           # the governing file, or None
```

`fill.jsonl` records name the pipeline of the module that submitted (`condor_cost.pipeline_name()`), or the
`pipeline=` a caller passes to `fill_plan`.

A malformed file always raises, also from `worker_shape`, so a typo never falls back silently to a default. Explicit command-line values should still win over a shape read from here; that is the caller's policy.

## Numerics guard

`numerics.py` extracts, per serving spec, only the fields that can change outputs, and `test_numerics.py` compares them with `numerics_record.json`, which has one section per pipeline, recorded from the branch heads. The guard is behaviour-preserving: it reads specs and launch scripts and never changes what is served.

- **Numerics:**
  - weights: revision; declared and effective weight quantization; the checkpoint's identity (below);
  - KV and Mamba/state cache dtypes;
  - attention, kernel and compilation flags (`--block-size` included, since it changes kernel choice);
  - numerics environment (plugins on `PYTHONPATH`, `VLLM_*` switches, `GRUG_*`, script `export`s that reach vLLM);
  - for campaigns, the expanded Pi finish notice, as `synth/config.finish_notice_config` freezes it from the branch's `tb_agents` (as eval records it);
  - repository code the spec loads, by sha256 of the import closure: entry points, parser plugins, plugin dirs on `PYTHONPATH` with their `*.dist-info/entry_points.txt` registrations (e.g. snowball-eval's `grug_vllm`);
  - max model len, and the output cap and its kind (`max_new_tokens` is hard, `max_tokens` is a default, or client-side);
  - chat template (sha256), its default kwargs (thinking flags) and parsers;
  - server sampling (`--override-generation-config`) and client request overrides (`pi_thinking`, temperature, ...);
  - extra stop tokens, seed and image limits;
  - the eval harness protocol, including the frozen Pi finish notice;
  - the vLLM runtime: declared version, SIF, CUDA home, entrypoint, kernel wheels by name and sha256;
  - the GPU type the job requests.

  Recorded as `adjacent.*` (numerics-adjacent: the distribution is meant to stay the same, outputs do not): speculative decoding, and the cpu-supersede eviction switch. A vLLM flag `numerics.py` does not know is recorded as `args.<flag>`: an unknown flag counts as numerics until it is classified.
- **Not numerics:**
  - memory utilization, max seqs, batched tokens, prefix caching, TP/DP/EP layout;
  - CUDA-graph capture sizes, `--async-scheduling`;
  - the CPU KV tier, gates, resources and bids, timeouts, caches;
  - `--load-format`, `--model-loader-extra-config`, `--engram-config`, but only at known-safe values (`--load-format dummy` is recorded);
  - paths themselves: a moved venv, checkpoint, draft model, kernel wheel or prebuilt kernel dir keeps its record when its content identity is the same.
- **Sources:**
  - aa-serving and task-synthesis: `synth/teachers/*.toml` under `serve_teacher.py`'s `PROFILES` (one spec per `replica_tunables` arm), and every `synth/campaigns/*.toml` as `synth/serve.py` runs it (`campaign/<name>`, plus `campaign/<name>/candidate/<c>` for tuning candidates). Each goes through the branch's own `vllm_launch.replica_command` and `replica_env`.
  - eval: `eval_q35/submit.py init` runs with stubbed I/O for each family x serving variant x GPU set x harness. A small interpreter then follows `serve_job.sh`'s argument building and exports.
  - Quantization labels come from each branch's own runtime recorder (`synth/numerics.py`, `eval_q35/serve_numerics.py`). Those recorders write each *job's* effective numerics after every override (`numerics.<job>.json`); this record is the *spec*, checked before anything runs.
- **Two classes of fields.**
  - **Repository-derived** (everything without a `host.` prefix) is strict and always checked.
  - **Host-derived** (`host.*`) is read from the node, not the repository:
    - the checkpoint: the sha256 of its small files (`config.json`, `generation_config.json`, `tokenizer_config.json`, chat template, `preprocessor_config.json`, `model.safetensors.index.json`, `crc32.txt`); every `*.safetensors` / `*.bin` / `*.pt` shard as name:size; and one sha256 over a 192 KiB head/middle/tail sample of each shard. A directory holding only copied configs, another SFT step, or a re-quantized export all differ.
    - the venv: the dist-info names of vllm, torch, triton, transformers and flashinfer.
    - templates outside the repository;
    - prebuilt kernel dirs (paths, sizes, and content or samples);
    - draft models and tokenizer dirs.

    Host-derived fields are compared too. When the filesystem itself cannot be read on this node (no /data mounted), they are skipped with a warning in the test output, never failed, so a node without /data never fails an unrelated PR. A missing checkpoint or venv on a filesystem the node *can* read is a real change and fails. The checkpoint's directory name is repository-derived (`weights.checkpoint_name`), so a different model name fails everywhere. `record` refuses to write unreadable host-derived values unless given `--force`, and prints how many there are.

    The shard sample is a deliberate trade-off: three 64 KiB windows per shard (head, middle, tail), not a full hash, so the check stays fast on /data. It catches any re-export, re-quantization or new training step, since those rewrite every byte. It misses a same-size edit made in place outside those windows, e.g. a partial LoRA merge into a few tensors. `crc32.txt`, when an export writes one, covers that case.
- **What the extractor cannot follow fails too.** In serve_job.sh, the interpreter cannot follow loops, `case`, functions, tests it does not model, or values from the job's environment or unknown commands. Every variable such a statement assigns or exports becomes unknown, and is reported wherever it reaches the launch. A statement that touches the vLLM arguments, exports, or sources a file is reported directly. So is any command that may write into what the engine loads, the staged checkpoint (`$M`) or the model source and template (`sed -i`, `cp` / `mv` / `dd of=` / `tee` / a `>` or `>>` redirection into them, in any branch). The script's own staging copies are recorded this way, so only a new or changed write fails. `source` / `.` of a repository file (`$CODE/...`) is followed; any other sourced file is reported. A new or vanished warning fails like a field change. So do specs that newly fail to extract: a skip entry carries the file's sha256, so an edit to a file that does not parse is still seen.
- **Opting in:** a pipeline branch runs the check in its tests when its `tbshared/pipeline.json` names `aa-serving`, `task-synthesis` or `eval`. The check is skipped without pipeline.json, on `shared`, and outside a git worktree. Only git-tracked files count.
- **Changing numerics deliberately:** a failing guard lists every changed, added or removed field.
  - A numerics change needs a new eval baseline, and that is the user's call.
  - If the change is intended, update the record in the same change, or right before the pipeline PR merges, so the pipeline never sits red. On a branch from `shared`, run `python -m tbshared.numerics record PIPELINE REPO --ref <the pipeline PR's head commit>`. It rewrites only that pipeline's section and keeps its `notes`. Then add a note for the change, merge that `shared` PR, and bump the pipeline in its PR.
  - In the record, campaigns are stored as deltas against their teacher spec (`$base`).
- **Not covered yet:**
  - sft;
  - submit-time overrides (`serve_teacher --tunables/--protocol`; eval_q35 `--vllm-extra-args`, `--quantization`, `--chat-template`, `--serving` beyond the defaults). The per-job runtime records cover these.
  - the SIF, and plugin directories on /data on `PYTHONPATH` (e.g. `cpu_supersede_v2`, `eot_checkpoint_v4`), which are recorded by path only: replacing their content in place is not seen;
  - host-derived fields on a node that cannot read them (see above);
  - "vllm default" attention, which is recorded as the flag, not as the backend each GPU picks;
  - eval_q35's B200 -> 2x H100 move, which is recorded as written.
- `python -m tbshared.numerics diff` prints, for models served by more than one pipeline, the fields whose values differ between them (port_matrix_20261114/serving/report.md section 2, from the record). `python -m tbshared.numerics extract PIPELINE ROOT [--ref REF]` prints one extraction.

## Rules for CPU jobs, as encoded here

| Rule | Detail |
|---|---|
| Charges | units = max(CPUs, GB/16) + 24 per GPU, per hour, on what the job *requests*, at the node's running price. That price is at least the node floor (MinRunningPrice, per node): mostly CPU-only and A100 5-25, H100 35, B200 55, but some A100 and CPU-only nodes are 35. Up to 16 GB per CPU is free, so request both at the binding level (`fill_shape`). |
| Bids | A job matches only nodes whose floor is at most its bid. Bid as low as gets a node soon, and let the watcher raise the bid. Job `rank` cannot steer placement. |
| Memory holds | They come from rss + shmem + dirty pages. Bulk writes to /data go through `fastio`. |
| Samplers | A sampler next to a job's main process must not outlive it. Run the main process as a child, not `exec`, and stop and `wait` the sampler before exiting. |
| Logs and state | Condor logs and run state go on /data, never /home. |
