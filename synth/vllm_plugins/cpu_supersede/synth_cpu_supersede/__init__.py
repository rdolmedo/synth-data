"""synth-cpu-supersede v2: a vLLM general plugin that changes only the eviction order of the CPU KV tier.

Nemotron 3.5 Super keeps one Mamba state per prompt boundary (prefix_cache_retention_interval 0): every agent turn
hands the SimpleCPUOffloadConnector its prompt-boundary state, one block in each of the 5 Mamba KV groups (~93 MB),
while its attention blocks are stored only once. The CPU tier is an LRU block pool, so these per-turn states are
most of what it holds, and the agent's previous states stay there until LRU reaches them. On the 2026-10-24 hero
engines (200 GB, 10,748 blocks, ~9 turns/s) that is a turnover of ~2.5 minutes: 2.5% of TMax turns, those after
~1.5-5 minute gaps, lost their whole prefix (cached = 0) on both tiers, 35% of all prefill computed (CalibForge 41%).

A turn's next request only resumes from the latest state. When a request finishes normally after resuming from
state S (its prefix hit) and has handed the connector a later boundary state of its own, S is superseded: this
plugin moves S's CPU blocks to the front of the CPU free queue (evicted first). Only states that were a request's
own prompt-boundary state and were hit by exactly one request qualify, so shared prefixes (system prompt, a task's
first turn shared by its attempts once more than one hit it) keep stock LRU. With SYNTH_CPU_SUPERSEDE_GPU=1 the GPU
copies are demoted the same way.

Hashes, lookups and outputs are unchanged; a misjudged block costs only a recomputation. Nothing is patched unless
SYNTH_CPU_SUPERSEDE=1. Every 60 s the plugin logs its counts.
"""
from collections import OrderedDict
import os
import time

VERSION = 'v2'
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
    """vllm.general_plugins entry point; vLLM calls it in every process (EngineCore included)."""
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
    _logger.info('synth-cpu-supersede %s active (pid %d): superseded Mamba boundary states leave the CPU tier '
                 'first%s', VERSION, os.getpid(), ' (GPU copies too)' if _gpu_enabled() else '')


def _gpu_enabled():
    return os.environ.get('SYNTH_CPU_SUPERSEDE_GPU') == '1'


def _mamba_groups(manager):
    from vllm.v1.kv_cache_interface import MambaSpec
    groups = getattr(manager, '_synth_mamba_groups', None)
    if groups is None:
        groups = manager._synth_mamba_groups = [
            g for g, spec in enumerate(manager.cpu_kv_cache_config.kv_cache_groups)
            if isinstance(spec.kv_cache_spec, MambaSpec) and g in manager.prefix_cacheable_group_ids]
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
    (floor(prompt / 16) * 16 - 16 with MTP), or one hash block either side."""
    for delta in (0, -manager.hash_block_size, manager.hash_block_size):
        state = _state_hash(manager, request, tokens + delta)
        if state is not None and state in _produced:
            return state
    return None


def _record_hit(orig):
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        stats = getattr(request, 'prefill_stats', None)
        # Once per request: a CPU-tier hit comes back here after its async load (v1 counted it twice, so every
        # state looked shared and none was demoted).
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
            top = max((tokens for _, _, tokens in entries), default=0)
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


def _demote(pool, keys):
    """Move free blocks cached under `keys` to the front of the pool's free queue (evicted first)."""
    blocks = []
    for key in keys:
        found = pool.cached_block_hash_to_block._cache.get(key)
        candidates = [found] if hasattr(found, 'block_id') else list(found.values()) if found else []
        blocks += [b for b in candidates if b.ref_cnt == 0 and not b.is_null]
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
