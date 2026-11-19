# synth-cpu-supersede (vLLM general plugin)

Changes only the eviction order of the CPU KV tier (`SimpleCPUOffloadConnector`); hashes, prefix lookups and
outputs are untouched.

Nemotron 3.5 Super keeps one Mamba state per prompt boundary (`prefix_cache_retention_interval` 0). Every agent
turn hands the CPU tier its prompt-boundary state, one block in each of the 5 Mamba KV groups (~93 MB), while its
attention blocks are stored once. The tier is an LRU block pool, and only an agent's latest state is ever reused,
so superseded states make up most of what it holds. On the 2026-10-24 hero engines (200 GB = 10,748 blocks,
~9 turns/s) the tier turned over in ~2.5 minutes; 2.5% of TMax turns (after 1.5-5 minute tool gaps) lost their
whole prefix on both tiers, 35% of the prefill computed (CalibForge: 41%).

When a request finishes normally after resuming from state S (its prefix hit) and has handed the connector a later
boundary state of its own, the plugin moves S's CPU blocks to the front of the CPU free queue. Only states that
were some request's own prompt-boundary state and were hit by exactly one request qualify (shared prefixes keep
stock LRU); an aborted turn's hit does not count. `SYNTH_CPU_SUPERSEDE_GPU=1` demotes the GPU copies too.

Enable: `PYTHONPATH=/data/aai/tbench/vllm_plugins/cpu_supersede_v2` and `SYNTH_CPU_SUPERSEDE=1` in the
engine's environment (serving tunable `extra_env`); without the switch it patches nothing. vLLM loads it through
the `vllm.general_plugins` entry point. Every 60 s the log shows `synth-cpu-supersede v<N> last 60s: ...` with
finished / superseded / demoted_cpu / shared / not_produced / no_newer_state counts.

Pinned to vLLM 0.32.0rc1.dev190+g3f1bcfd82 (`venvs/vllm032`). Deployed copies are immutable:
`/data/aai/tbench/vllm_plugins/cpu_supersede_v<N>/`; a change ships as the next version. v1 counted a CPU-tier
hit twice (the request comes back to `update_state_after_alloc` after its async load), so every state looked shared
and nothing was demoted; v2 counts each request once.
CPU checks (on a deployed copy): `SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<deployed dir> <vllm032>/bin/python
check_cpu_supersede.py`.

**vLLM 0.34 (package 3.0, v3).** The entry point is `synth_cpu_supersede._entry:register`, which reads the imported
vLLM's version: below 0.34 it runs v2 (`__init__.py`, unchanged), from 0.34 on v3 (`vllm034.py`), same behaviour and
switches. 0.34 has no native equivalent (retention picks which Mamba checkpoints exist, not their eviction order; the
CPU tier is still one LRU pool). v3 follows 0.34's internals: Mamba groups are `KVCacheGroup`s recognised with
`kv_cache_utils.is_mamba_group` (no `kv_cache_spec`), and `BlockHashToBlockMap` holds block ids (one per key in
`_cache`, duplicates in `_dup_ids`). v2 itself loads on 0.34 but raises inside its per-request try, so it would demote
nothing and log a warning on every superseded turn. With MTP and this connector 0.34 hashes with one lookahead token
instead of dropping the trailing block, so a resumed hit lands on the checkpoint, (prompt - 1) // 16 * 16 (the
one-unit tolerance still covers `disable_lookahead_block_hashes`). Deploy as `cpu_supersede_v3`; on 0.32 it logs
`v2` and behaves as `cpu_supersede_v2`.
- CPU checks on 0.34 (no GPU: the 0.34 Python sources import on CPU ahead of a torch venv, e.g. vllm033's):
  `SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<0.34 site-packages or a dir holding its vllm>:<deployed dir> <python> -B
  check_cpu_supersede_v034.py`.
- Static check of every name, attribute and signature both 0.34 paths use: `python3
  ../check_vllm034_static.py <vllm package dir>` (no vLLM import; re-run it before deploying on a later vLLM).

Status (2026-10-25): see `docs/campaigns/super-throughput-20261025.md` section 6.
