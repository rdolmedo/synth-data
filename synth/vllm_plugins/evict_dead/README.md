# synth-evict-dead (vLLM general plugin)

Changes only the order in which freed KV blocks are evicted; hashes, prefix lookups and outputs are untouched.
Stock vLLM sends unhashed blocks to the front of the free queue and hashed ones to the back (LRU). When a
**finished** request frees its blocks, this plugin also sends to the front:

* attention / MTP-draft blocks whose hash covers tokens past the request's prompt (completion blocks, and the
  prompt-tail block once decode filled it: vLLM then replaces its partial prompt hash by a full hash over
  completion tokens). An agent's next turn re-renders the completion, so it never hits them. Requests that hit
  max_tokens are skipped (a client continuing a truncated answer may send it back verbatim);
* Mamba state snapshots the request got as a prefix hit and then superseded with a later hashed state of its own,
  unless another request also hit or still holds them, or they sit on the request's shared-prefix junction.

Preempted requests, connector paths and failed loads keep stock behaviour.

Enable: put this directory on the engine's `PYTHONPATH` and set `SYNTH_EVICT_DEAD=1` (without it the plugin
patches nothing). vLLM loads it through the `vllm.general_plugins` entry point in every process.

The vLLM log shows `synth-evict-dead v1 active` at load, then every 60 s the freed blocks by class:
`unhashed` (stock front), `hashed_kept` (stock LRU), `demoted_attn`, `demoted_mamba` (sent to the front at
free time) and `demoted_mamba_late` (already free, moved to the front when the request finished).

Pinned to vLLM 0.32.0rc1.dev190+g3f1bcfd82 (`venvs/vllm032`). Deployed copies are immutable:
`/data/aai/tbench/vllm_plugins/evict_dead_v1/`; a change ships as `evict_dead_v2`.
CPU checks: `check_evict_dead.py` (run with the vllm032 python in a CPU job).
