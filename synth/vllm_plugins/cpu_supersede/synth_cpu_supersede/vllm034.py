"""synth-cpu-supersede v3: the vLLM 0.34 path of synth-cpu-supersede (v2's logic on 0.34's internals).

Selected by `_entry.register` when the installed vLLM is 0.34 or later; vLLM 0.32 keeps running v2 (`__init__.py`,
unchanged). Behaviour is v2's: when a request finishes normally after resuming from Mamba prompt-boundary state S and
has handed the CPU tier a later boundary state of its own, S's CPU blocks move to the front of the CPU free queue
(and the GPU copies with SYNTH_CPU_SUPERSEDE_GPU=1). Hashes, lookups and outputs are unchanged.

vLLM 0.34 has no native equivalent: `prefix_cache_retention_interval` decides which Mamba checkpoints exist, not
their eviction order, and the CPU tier is still one LRU pool (`SimpleCPUOffloadScheduler`; its new eager-mode
`_pending_local_hits` refresh even moves a GPU hit's CPU copy to the MRU end at admission).

What changed from v2 (0.34 internals):
- KV cache groups are `KVCacheGroup`s without `kv_cache_spec`; a Mamba group is `kv_cache_utils.is_mamba_group(g)`
  (its manager class is `MambaManager`), as 0.34's own `SimpleCPUOffloadScheduler._estimate_lazy_target_blocks`.
- `BlockHashToBlockMap._cache` maps a key to one block id (an int, not a block or a dict of blocks); keys cached by
  several blocks also list every id in `_dup_ids`. v2's `_demote` raised on the int (caught per request, so v2 on
  0.34 demotes nothing and logs a warning each superseded turn).
- With MTP and a connector that supports lookahead block hashes (SimpleCPUOffloadConnector does), 0.34 hashes every
  16-token unit together with the next token (`num_hash_lookahead_tokens` 1) instead of dropping the trailing
  block, so a resumed request's hit length is the Mamba checkpoint position itself, (prompt - 1) // 16 * 16. The
  one-hash-unit tolerance of `_resumed_state` still covers the block-drop configuration
  (`disable_lookahead_block_hashes`).

Nothing is patched unless SYNTH_CPU_SUPERSEDE=1. Every 60 s the plugin logs its counts.
"""
from collections import OrderedDict
import os
import time

VERSION = 'v3'
LOG_SECONDS = 60
MAX_TRACKED = 1 << 20

_patched = False
_logger = None
_hit = {}             # request id -> (tokens resumed from cache (GPU + CPU) at its first allocation, produced state)
_offered = {}         # request id -> highest boundary state (tokens) handed to the connector
_produced = OrderedDict()   # state hash -> requests that hit it; states that were some request's own boundary
_window = dict.fromkeys(('finished', 'superseded', 'demoted_cpu', 'demoted_gpu', 'shared', 'not_produced',
                         'no_newer_state'), 0)
_totals = dict(_window)
_last_log = [time.monotonic()]


def stats():
    return dict(_totals)


def register():
    """Called by `_entry.register` (the vllm.general_plugins entry point) in every process, EngineCore included."""
    if os.environ.get('SYNTH_CPU_SUPERSEDE') != '1':
        return
    _patch()


def _patch():
    global _patched, _logger
    if _patched:
        return
    from vllm.logger import init_logger
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    _logger = init_logger('vllm.synth_cpu_supersede')  # vLLM's handlers live on the "vllm" logger
    SimpleCPUOffloadScheduler.update_state_after_alloc = _record_hit(SimpleCPUOffloadScheduler.update_state_after_alloc)
    SimpleCPUOffloadScheduler._prepare_eager_store_specs = _record_offers(
        SimpleCPUOffloadScheduler._prepare_eager_store_specs)
    SimpleCPUOffloadScheduler.request_finished_all_groups = _on_finished(
        SimpleCPUOffloadScheduler.request_finished_all_groups)
    _patched = True
    _logger.info('synth-cpu-supersede %s (vLLM 0.34 path) active (pid %d): superseded Mamba boundary states leave '
                 'the CPU tier first%s', VERSION, os.getpid(), ' (GPU copies too)' if _gpu_enabled() else '')


def _gpu_enabled():
    return os.environ.get('SYNTH_CPU_SUPERSEDE_GPU') == '1'


def _mamba_groups(manager):
    from vllm.v1.core.kv_cache_utils import is_mamba_group
    groups = getattr(manager, '_synth_mamba_groups', None)
    if groups is None:
        groups = manager._synth_mamba_groups = [
            g for g, group in enumerate(manager.cpu_kv_cache_config.kv_cache_groups)
            if is_mamba_group(group) and g in manager.prefix_cacheable_group_ids]
    return groups


def _state_hash(manager, request, tokens):
    """The prefix hash a Mamba state at `tokens` is cached under (hash-block aligned), or None."""
    if tokens <= 0 or tokens % manager.hash_block_size:
        return None
    index = tokens // manager.hash_block_size - 1
    hashes = request.block_hashes
    return hashes[index] if index < len(hashes) else None


def _resumed_state(manager, request, tokens):
    """The produced boundary state a request resumed from: at its hit length, which is the Mamba checkpoint position
    ((prompt - 1) // 16 * 16 with lookahead hashes; one hash unit less with the MTP block drop), or one hash block
    either side."""
    for delta in (0, -manager.hash_block_size, manager.hash_block_size):
        state = _state_hash(manager, request, tokens + delta)
        if state is not None and state in _produced:
            return state
    return None


def _record_hit(orig):
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        stats = getattr(request, 'prefill_stats', None)
        # Once per request: a CPU-tier hit comes back here after its async load (with num_external_tokens 0), and a
        # deferred load (Scheduler.deferred_loads) calls this a step later than its admission.
        if stats is not None and request.num_preemptions <= 0 and request.request_id not in _hit:
            tokens = stats.num_local_cached_tokens + stats.num_external_cached_tokens
            state = _resumed_state(self, request, tokens)
            if state is not None:
                _produced[state] += 1
            _hit[request.request_id] = (tokens, state)
        return orig(self, request, blocks, num_external_tokens)
    update_state_after_alloc.__wrapped__ = orig
    return update_state_after_alloc


def _record_offers(orig):
    def _prepare_eager_store_specs(self, scheduler_output):
        block_state = scheduler_output.kv_connector_block_state
        for req_id, entries in (block_state.boundary_state_offloads if block_state is not None else {}).items():
            top = max((tokens for _, _, tokens in entries), default=0)   # (group_id, block_id, boundary_tokens)
            if top > _offered.get(req_id, 0):
                _offered[req_id] = top
        return orig(self, scheduler_output)
    _prepare_eager_store_specs.__wrapped__ = orig
    return _prepare_eager_store_specs


def _on_finished(orig):
    from vllm.v1.request import RequestStatus
    normal = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)

    def request_finished_all_groups(self, request, block_ids):
        (hit, state), offered = _hit.pop(request.request_id, (0, None)), _offered.pop(request.request_id, 0)
        try:
            if request.status in normal:
                _count('finished')
                _supersede(self, request, hit, offered)
            elif state is not None and _produced.get(state):
                _produced[state] -= 1  # an aborted turn's retry is the same agent, not a second one
        except Exception as exc:  # eviction order is an optimisation; never fail a request over it
            _logger.warning('synth-cpu-supersede: skipped %s: %r', request.request_id, exc)
        _maybe_log()
        return orig(self, request, block_ids)
    request_finished_all_groups.__wrapped__ = orig
    return request_finished_all_groups


def _supersede(manager, request, hit, offered):
    if offered <= 0:
        _count('no_newer_state')
        return
    own = _state_hash(manager, request, offered)
    if own is not None:
        _produced[own] = _produced.pop(own, 0)
        while len(_produced) > MAX_TRACKED:
            _produced.popitem(last=False)
    if hit <= 0 or offered <= hit:
        _count('no_newer_state')
        return
    old = _resumed_state(manager, request, hit)
    if old is None or old == own:
        _count('not_produced')
        return
    if _produced[old] != 1:
        _count('shared')
        return
    del _produced[old]
    _count('superseded')
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    keys = [make_block_hash_with_group_id(old, g) for g in _mamba_groups(manager)]
    _count('demoted_cpu', _demote(manager.cpu_block_pool, keys))
    if _gpu_enabled() and manager._gpu_block_pool is not None:
        _count('demoted_gpu', _demote(manager._gpu_block_pool, keys))


def cached_blocks(pool, key):
    """Every block of `pool` cached under `key`. vLLM 0.34's BlockHashToBlockMap keeps one block id per key in
    `_cache` and, for keys several blocks are cached under, all their ids (insertion order) in `_dup_ids`."""
    table = pool.cached_block_hash_to_block
    ids = table._dup_ids.get(key)
    if ids is None:
        block_id = table._cache.get(key)
        ids = () if block_id is None else (block_id,)
    return [pool.blocks[i] for i in ids]


def _demote(pool, keys):
    """Move free blocks cached under `keys` to the front of the pool's free queue (evicted first)."""
    blocks = list({b.block_id: b for key in keys for b in cached_blocks(pool, key)
                   if b.ref_cnt == 0 and not b.is_null}.values())   # a block listed twice would be removed twice
    if blocks:
        queue = pool.free_block_queue
        for block in blocks:
            queue.remove(block)
        queue.prepend_n(blocks)
    return len(blocks)


def _count(key, n=1):
    if n:
        _window[key] += n
        _totals[key] += n


def _maybe_log():
    now = time.monotonic()
    if now - _last_log[0] < LOG_SECONDS or _logger is None:
        return
    window = ' '.join(f'{k}={v}' for k, v in _window.items())
    totals = ' '.join(f'{k}={v}' for k, v in _totals.items())
    _logger.info('synth-cpu-supersede %s last %.0fs: %s | totals: %s | tracked states %d', VERSION,
                 now - _last_log[0], window, totals, len(_produced))
    for k in _window:
        _window[k] = 0
    _last_log[0] = now
