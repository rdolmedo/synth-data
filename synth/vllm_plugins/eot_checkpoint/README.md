# synth-eot-checkpoint (vLLM general plugin)

Lets the next turn's prefix-cache hit reach through the previous answer on Nemotron 3.5 Super. Pi resends each
answer in the next request; a Mamba layer resumes only from a stored state, and stock vLLM 0.32 stores states only at
the prompt boundary, so the whole re-rendered answer is recomputed every turn.

At a clean stop (the stop token was the last token of its step, and the async scheduler's already-scheduled next step
consumed it), the request's running Mamba state covers every token through the stop token and its attention tail
block holds that token's KV. With MTP the stop token is often an accepted draft followed by more accepted tokens, so
the plugin cuts each step's sampled tokens after the first stop token (distribution-preserving). This engine runs
vLLM's V2 model runner (`vllm/v1/worker/gpu`): the cut must act on its rejection sampler's `num_sampled` /
`num_rejected` counts (v3); v2 patched the V1 sampler and never ran. Before the request's blocks are freed the
plugin registers both under an exact-length key (prefix-chain hash of the last full 16-token unit + the remaining
tokens), queues them with the turn's finished CPU-tier stores, and probes the key on both tiers for requests whose
prompt has a stop token past their standard hit.
A hit resumes at that exact (unaligned) position; vLLM's partial-hit copy-on-write and Mamba chunk splitting handle
the rest. When a request resumes from turn i's end state (or its prompt-boundary state) and finishes, turn i's end
and boundary states move to the front of the CPU and GPU free queues (as synth-cpu-supersede does for boundary
states), unless another request also resumed from the boundary state; a turn that itself resumed from an end
state demotes its own boundary state as soon as its end state registers (v4), so each agent keeps one live state.

The hit only lands when the re-rendered answer matches the generated tokens through the stop token: 99.4% of TMax
turns in the closed-loop replay (Pi streams; a history built from a non-streaming response does not match, since
vLLM right-strips its reasoning and text).

Enable: `PYTHONPATH=/data/aai/tbench/vllm_plugins/cpu_supersede_v2:/data/aai/tbench/vllm_plugins/eot_checkpoint_v<N>`,
`SYNTH_EOT_CHECKPOINT=1` (and `SYNTH_CPU_SUPERSEDE=1`) in the engine's environment; without the switch it patches
nothing. Per-request opt-out for checks: `vllm_xargs: {"synth_eot_skip": 1}`. Every 60 s the log shows
`synth-eot-checkpoint v<N> last 60s: ...` (stops, registered, ineligible reasons, gpu_hits, cpu_hits, hit_tokens,
stored, superseded, demoted).

Pinned to vLLM 0.32.0rc1.dev190+g3f1bcfd82 (`venvs/vllm032`) with Mamba align mode, prefix-match unit 16, MTP and
SimpleCPUOffloadConnector. Deployed copies are immutable: `/data/aai/tbench/vllm_plugins/eot_checkpoint_v<N>/`.

Checks:
- CPU, real vLLM classes: `SYNTH_EOT_CHECKPOINT=1 PYTHONPATH=<deployed dir> <vllm032>/bin/python
  check_eot_checkpoint.py`
- GPU, next-token distributions (end-state hit vs fresh prefill, stock prefix hits as the noise floor):
  `eot_logprob_check.py` (run by a `synth.tune` candidate's `check`).

**vLLM 0.34 (package 5.0, v5).** The entry point is `synth_eot_checkpoint._entry:register`, which reads the imported
vLLM's version: below 0.34 it runs v4 (`__init__.py`, unchanged), from 0.34 on v5 (`vllm034.py`), same design and
switches. 0.34 has no native equivalent: `MambaManager.cache_blocks_on_free` (default retention interval 0) keeps only
the latest decode-crossed block-boundary state, which exists only when an answer crosses a multiple of the hybrid
block (thousands of tokens), still recomputes the answer past it, and stays on the GPU; v5 coexists with it (an end
state within MIN_GAIN of that boundary leaves the hit to it). v4 cannot start on 0.34: it imports the removed MRV1
sampler in `register`, which vLLM does not guard. v5's changes: only the V2 rejection sampler is cut (its counts and
their consumers are unchanged); with MTP and this connector 0.34 hashes each 16-token unit together with the next
token (`num_hash_lookahead_tokens` 1, no trailing-block drop), so the end key's parent is the last unit ending at least
one token before the end (v4's key raised for the 1/16 of answers ending on a unit edge) and the prompt boundary is
(prompt - 1) // 16 * 16; the GPU lookup takes `max_cache_hit_length`; the CPU lookup, now handed the block-aligned
part of the local hit, offers only end states past the full local hit; `Request.skip_reading_external_kv`,
`KVCacheGroup.prefix_cacheable` and the id-based `BlockHashToBlockMap`; nothing is registered under `--use-replayssm`
(Mamba states written back only on flush). Deploy as `eot_checkpoint_v5` with `cpu_supersede_v3` (v5 reads
synth-cpu-supersede v3's shared-state counts); on 0.32 they behave as v4 / v2.
- CPU checks on 0.34 (no GPU: the 0.34 Python sources import on CPU ahead of a torch venv, e.g. vllm033's):
  `SYNTH_EOT_CHECKPOINT=1 SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<0.34 vllm parent>:<cpu_supersede_v3>:<deployed dir>
  <python> -B check_eot_checkpoint_v034.py` (real 0.34 KVCacheManager with lookahead hashes and the real CPU-tier
  lookup, allocation and store-completion methods).
- Static check: `python3 ../check_vllm034_static.py <vllm package dir>`.
- GPU: `eot_logprob_check.py` unchanged in method; it now retries `/reset_prefix_cache` until it succeeds (it fails
  while CPU stores are in flight) and reports `reset_failed` for the eot-cpu rows.

Versions: v1 (317e7e2) and v2 (bff73a5) superseded; v3 (910ca73) measured +13.3% generated tok/s on one B200
(closed-loop TMax replay vs synth-cpu-supersede v2 alone; logprob check within the stock prefix-hit noise on both
tiers); v4 (839e061) adds the early boundary demotion. Status: `docs/campaigns/super-throughput-20261025.md`
section 9.

Open items for the next revision (independent review of the #478 follow-up; neither changes outputs on Pi traffic):
- An end-state hit returns `shared_prefix_boundary=0` (`vllm034.py` `_gpu_lookup`), so with
  `--enable-mamba-shared-prefix-checkpoint` a junction stock vLLM found past the end state is dropped (throughput only;
  fix: keep `boundary` when it is past `end`).
- `_bonus_stop` keeps a request id after a stop that was not its step's last token; a resumable (streaming-input)
  request re-enqueued later would treat its final stop as a bonus. Chat completions requests are not resumable.
- v5 runs only on the builds in `_entry.VALIDATED_V5`; add a build after its `eot_logprob_check.py` passes.
