# harbor_mswea: the shared Harbor + mini-swe-agent runner

This is AA's Terminal-Bench harness (stock mini-swe-agent, bash only) on our cluster. Its users are TB 4.0 (`aa_evals/tb4`, #28) and TB-Science (`aa_evals/tbsci`, #29). The method, the deviations from AA and the infra-vs-model rules are documented in `aa_evals/tb4/README.md`.

```bash
python -m aa_evals.common.harbor_mswea.run --bench B --tasks DIR_OR_TAR --model serving:NAME --run-id ID \
    --step-limit N [--repeats 3] [--limit N] [--include a,b] [--exclude a,b] [--sampling JSON] \
    [--max-concurrent 32] [--workers N] [--env-class MOD:CLASS] [--aa-slug SLUG --aa-metric-key KEY] \
    [--agent mini|oracle|nop]
python -m aa_evals.common.harbor_mswea.run --bench B --run-id ID --status | --score | --resubmit N
```

**Inputs**
- `--tasks` is either a directory of Harbor task dirs or a tar containing `tasks/<slug>/`.
- With `serving:<name>`, the sampling defaults to that registry's lab-recommended `request.overrides`. temperature, top_p and max_tokens go to litellm; every other key (such as `reasoning_effort` or `chat_template_kwargs`) goes in `extra_body`.

**Disk cap (#405).** `--diskcap auto|FACTOR[:MIN_MB]|off`, default `auto` (a bench whose defaults name a
`diskcap_family` gets eval_q35's per-task caps, `DISKCAP_FAMILIES`; the others `3:30720`): every sandbox is capped at
max(storage_mb × 3, 30 GiB) per file (RLIMIT_FSIZE) and in total (a watchdog kills its top writer), so one runaway
writer cannot fill the worker's shared scratch. Config `"diskcap"` (null = off; a config without the key, a run before #405, = off; eval_q35's format, per-task `"tasks"` included, but no `"one_per_worker"` scheduling here); each outcome records `diskcap` (the
effective setting) and `diskcap_limit_mb`, `sandbox_disk_peak_mb`, `sandbox_largest_file_mb`, `diskcap_kills`; the
box summary is `diskcap.json` in the trial archive (`harbor_ns/diskcap.py`, `aa_evals/common/diskcap.py`). The cap
spares the agent by process name (`pi`); mini-swe-agent runs as `python3.12` like task code, so it is not spared, but
it writes little itself (its commands are its children), so it is killed only if it is the top writer.

**Run directory:** `/data/aai/aa_evals/runs/<bench>/<run_id>/`. It holds `config.json`, `queue/`, `trials/<key>.tar.gz`, `transcripts/<key>.json.gz`, `items.jsonl` and `summary.json`. The last worker writes the last two.

| File | What |
|---|---|
| `agent.py` | `MiniSweAgentAA`: Harbor 0.24's `MiniSweAgent`, installed from the pinned bundle |
| `build_bundle.sh` | the bundle: a relocatable CPython 3.12 + mini-swe-agent from PyPI under `/opt/aa-mswea`, built as a CPU condor job, written to `/data/aai/aa_evals/agent_cache/mini-swe-agent/<version>/linux-x64/` |
| `ns_env.py` | `NsSidecarEnvironment`: harbor_ns plus Harbor's per-service operations (sidecar artifacts), from aa-tbsci |
| `run.py` | CLI: freezes the code, writes the config and queue, submits workers, reports status |
| `worker.py`, `worker_job.sh` | CPU condor worker: per-trial core sets and memory admission, one Harbor trial per attempt in node scratch |
| `queue.py` | Lustre mkdir-claim queue; infra failures get up to 3 more tries |
| `classify.py` | infrastructure vs. model outcome; mini's exit status, steps, finish reasons, model and environment time |
| `score.py` | `items.jsonl` and `summary.json`: pass@1 over repeats, task-bootstrap ci95, infra counts, `aa_reference` |
| `endpoint.py` | the contract's model specs (`serving:` / `url:`) |

Control files:
- `RUN/control.json` `{"limit": N}` changes the in-flight cap live.
- Touching `RUN/STOP_WORKERS` stops new claims; workers exit when idle.
- Jobs run the frozen copy in `/data/aai/aa_evals/code/<sha>[-<diff hash>]`. A code change reaches a run only through a new run id.
