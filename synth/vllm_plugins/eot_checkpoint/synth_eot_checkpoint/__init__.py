"""synth-eot-checkpoint v4: keep each answer's final Mamba state so the next turn's cache hit reaches through it.

Pi resends the previous answer in the next request; on Nemotron 3.5 Super it re-renders token for token (99.4% of
turns) and is ~42% of the prefill an engine computes with synth-cpu-supersede (2026-10-25 closed-loop replay: see
docs/campaigns/super-throughput-20261025.md section 9). It is never a cache hit: a Mamba layer resumes only from a stored state, and states are stored at the prompt boundary (16-token
units, minus MTP's re-prefill unit), never inside or after the answer.

When a request stops on a stop token that was the last token of its step, the async scheduler has already scheduled
the next step, which consumes that token: afterwards the request's running Mamba state (its first state slot) covers
every token up to and including the stop token, and its attention tail block holds that token's KV. With MTP the
stop token is often an accepted draft followed by more accepted tokens, whose states the next step overwrites; so
the plugin cuts every step's sampled tokens after the first stop token. Tokens after a stop token are discarded
anyway, and a shorter accepted draft keeps the sampling distribution, so outputs are unchanged. (v2 cut only the V1
model runner's sampler; this engine runs the V2 model runner, whose rejection sampler returns per-request
num_sampled / num_rejected counts that everything downstream follows: v3 cuts those.) Before the
request's blocks are freed this plugin registers both under an exact-length key (the prefix-chain hash of the last
full 16-token unit plus the remaining tokens), hands them to the CPU tier with the turn's other finished stores, and
looks the key up on both tiers for later requests: at the positions right after a stop token in the new prompt. A
hit resumes at that exact position (the existing partial-hit copy-on-write handles the unaligned tail; the
scheduler's Mamba split then stops at the new prompt's checkpoint as usual).

States it supersedes leave the tiers first (as synth-cpu-supersede does for prompt-boundary states): when a request
that resumed from a turn's end state, or from that turn's prompt-boundary state, finishes normally, both of that
turn's states are moved to the front of the CPU (and GPU) free queues, unless another request also resumed from the
prompt-boundary state. A turn that itself resumed from an end state demotes its own prompt-boundary state as soon as
its end state is registered (v4): that state is then only the fallback for a re-render that diverges, and keeping
it would hold two live states per agent in the CPU tier instead of one (v2's replay: whole-prefix losses rose with
the extra stores). First turns keep it: sibling attempts share their prompts.

Nothing is patched unless SYNTH_EOT_CHECKPOINT=1. Pinned to vLLM 0.32.0rc1.dev190+g3f1bcfd82 with Mamba align mode,
prefix-match unit 16, MTP and SimpleCPUOffloadConnector. Every 60 s the plugin logs its counts.
"""
from array import array
from collections import OrderedDict
import hashlib
import os
import time

VERSION = 'v4'
SALT = b'synth-eot-v1'      # the key format is unchanged since v1
MIN_GAIN = 32            # an end-state hit must reach this many tokens past the standard hit
MAX_CANDIDATES = 4       # stop-token positions probed per lookup, latest first
EDGE = 8                 # skip stops this close to a scheduler-block edge (the state slots move there)
MAX_TRACKED = 1 << 18
# Stop tokens a step's sampled tokens are cut after (Nemotron 3.5 Super: </s> and <|im_end|>, its EOS ids).
STOP_IDS = tuple(int(x) for x in os.environ.get('SYNTH_EOT_STOP_IDS', '2,11').split(',') if x.strip())

_patched = False
_logger = None
_bonus_stop = set()      # request ids whose stop token was the last token of a step's output
_ends = OrderedDict()    # end-state key -> boundary-state hash of the same turn (or None)
_boundary_end = {}       # boundary-state hash -> end-state key of the same turn
_resumed = {}            # request id -> end-state key it resumed from
_hit_tokens = {}         # request id -> tokens its last lookup resumed from (GPU + CPU)
_cpu_end_hits = {}       # request id -> end position of a CPU-tier end-state hit awaiting allocation
_finish_stores = {}      # request id -> GPU block ids to store with the request's finished stores
_window = dict.fromkeys(('stops', 'registered', 'not_bonus', 'not_in_flight', 'edge', 'busy_block', 'gpu_hits',
                         'cpu_hits', 'hit_tokens', 'stored', 'superseded_end', 'superseded_boundary',
                         'demoted_cpu', 'demoted_gpu', 'shared', 'early_boundary'), 0)
_totals = dict(_window)
_last_log = [time.monotonic()]


def stats():
    return dict(_totals)


def register():
    """vllm.general_plugins entry point; vLLM calls it in every process (EngineCore included)."""
    if os.environ.get('SYNTH_EOT_CHECKPOINT') != '1':
        return
    _patch()


def _patch():
    global _patched, _logger
    if _patched:
        return
    from vllm.logger import init_logger
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    _logger = init_logger('vllm.synth_eot_checkpoint')
    RejectionSampler.forward = _cut_at_stop(RejectionSampler.forward)
    try:   # the V2 model runner (vllm/v1/worker/gpu) has its own rejection sampler
        from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as V2RejectionSampler
        V2RejectionSampler.__call__ = _cut_counts_at_stop(V2RejectionSampler.__call__)
    except ImportError:
        pass
    Scheduler._update_request_with_output = _record_stop(Scheduler._update_request_with_output)
    Scheduler._free_request = _register_end(Scheduler._free_request)
    KVCacheManager.get_computed_blocks = _gpu_lookup(KVCacheManager.get_computed_blocks)
    SimpleCPUOffloadScheduler._queue_finished_eager_store = _store_end(
        SimpleCPUOffloadScheduler._queue_finished_eager_store)
    SimpleCPUOffloadScheduler.get_num_new_matched_tokens = _cpu_lookup(
        SimpleCPUOffloadScheduler.get_num_new_matched_tokens)
    SimpleCPUOffloadScheduler.update_state_after_alloc = _cpu_alloc(SimpleCPUOffloadScheduler.update_state_after_alloc)
    _patched = True
    _logger.info('synth-eot-checkpoint %s active (pid %d): end-of-turn Mamba states are cached under exact-length '
                 'keys; sampled tokens are cut after stop tokens %s', VERSION, os.getpid(), STOP_IDS)


# ---------------------------------------------------------------------------------------------------------- keys

def end_key(tokens, block_hashes, end, hash_block_size):
    """Exact-length key of the prefix tokens[:end]: the chain hash of its last full unit plus the remaining tokens."""
    from vllm.v1.core import kv_cache_utils
    base = end // hash_block_size * hash_block_size
    parent = bytes(block_hashes[base // hash_block_size - 1]) if base else bytes(kv_cache_utils.NONE_HASH)
    return hashlib.sha256(SALT + parent + array('i', tokens[base:end]).tobytes()).digest()


def _skip(request):
    """Per-request opt-out (vllm_xargs {"synth_eot_skip": 1}): the GPU check's stock-path control."""
    extra = request.sampling_params.extra_args if request.sampling_params is not None else None
    return bool(extra and extra.get('synth_eot_skip'))


def _stop_ids(request):
    params = request.sampling_params
    return params.all_stop_token_ids if params is not None else ()


def _candidates(request, hash_block_size):
    """[(end, key)] for the last MAX_CANDIDATES positions right after a stop token in the prompt (leaving at least
    one token to compute), latest first. Computed once per request: the scheduler probes a waiting request on every
    step until it is scheduled."""
    cached = request.__dict__.get('_synth_eot_candidates')
    if cached is not None:
        return cached
    tokens, stop = request._all_token_ids, request.num_tokens - 1
    found = []
    for t in _stop_ids(request):
        i = 0
        while True:   # list.index scans at C speed; one call per occurrence
            try:
                i = tokens.index(t, i, stop)
            except ValueError:
                break
            found.append(i + 1)
            i += 1
    found = sorted(found, reverse=True)[:MAX_CANDIDATES]
    cached = [(end, end_key(tokens, request.block_hashes, end, hash_block_size)) for end in found]
    request._synth_eot_candidates = cached
    return cached


def _groups(coordinator):
    """(group id, manager, is_mamba) for every prefix-cacheable group, or None if a group type is unsupported."""
    cached = getattr(coordinator, '_synth_eot_groups', False)
    if cached is not False:
        return cached
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    out = []
    for gid, (manager, group) in enumerate(zip(coordinator.single_type_managers,
                                               coordinator.kv_cache_config.kv_cache_groups)):
        if not group.kv_cache_spec.prefix_cacheable:
            continue
        if type(manager) is MambaManager:
            out.append((gid, manager, True))
        elif type(manager) is FullAttentionManager:
            out.append((gid, manager, False))
        else:
            out = None
            break
    if out is not None and (not any(m for _, _, m in out) or len({m.block_size for _, m, _ in out}) != 1):
        out = None
    coordinator._synth_eot_groups = out
    return out


def _boundary_hash(manager, request):
    """Hash of the prompt-boundary Mamba state this request stored (its prompt checkpoint position), or None."""
    from vllm.v1.kv_cache_interface import get_mamba_prefill_checkpoint_position
    hb = manager.block_pool.hash_block_size
    pos = get_mamba_prefill_checkpoint_position(request.num_prompt_tokens, hb, manager.drop_eagle_checkpoint_block)
    index = pos // hb - 1
    return bytes(request.block_hashes[index]) if pos > 0 and index < len(request.block_hashes) else None


# ---------------------------------------------------------------------------------------------- registration

_stop_tensor = {}
_cut_seen = []


def cut_after_stop(ids):
    """In place: placeholder (-1) every sampled token after the first stop token of its row. Valid tokens stay a
    contiguous prefix, so the accepted counts, the next step's input token and the drafter all follow. No host sync."""
    import torch
    stops = _stop_tensor.get(ids.device)
    if stops is None:
        stops = _stop_tensor[ids.device] = torch.tensor(STOP_IDS, dtype=ids.dtype, device=ids.device)
    width = ids.shape[1]
    pos = torch.arange(width, device=ids.device)
    first = torch.where(torch.isin(ids, stops), pos, width).amin(dim=1, keepdim=True)
    ids.masked_fill_(pos > first, -1)
    return ids


def cut_counts_after_stop(ids, num_sampled, num_rejected):
    """V2 sampler output, in place: a row's tokens after its first stop token stop counting (num_sampled shrinks,
    num_rejected grows by the same amount; the dropped tokens become -1). No host sync."""
    import torch
    stops = _stop_tensor.get(ids.device)
    if stops is None:
        stops = _stop_tensor[ids.device] = torch.tensor(STOP_IDS, dtype=ids.dtype, device=ids.device)
    width = ids.shape[1]
    pos = torch.arange(width, device=ids.device)
    counted = pos < num_sampled[:, None]
    first = torch.where(torch.isin(ids, stops) & counted, pos, width).amin(dim=1)
    kept = torch.minimum(num_sampled, (first + 1).to(num_sampled.dtype))
    ids.masked_fill_(counted & (pos >= kept[:, None]), -1)
    num_rejected.add_(num_sampled - kept)
    num_sampled.copy_(kept)


def _cut_counts_at_stop(orig):
    def __call__(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        ids = out.sampled_token_ids
        if STOP_IDS and ids is not None and ids.dim() == 2 and out.num_sampled is not None:
            cut_counts_after_stop(ids, out.num_sampled, out.num_rejected)
            if not _cut_seen:
                _cut_seen.append(True)
                _logger.info('synth-eot-checkpoint %s: cutting the V2 rejection sampler output (%s, %s)', VERSION,
                             tuple(ids.shape), ids.dtype)
        return out
    __call__.__wrapped__ = orig
    return __call__


def _cut_at_stop(orig):
    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if STOP_IDS and out.sampled_token_ids is not None and out.sampled_token_ids.dim() == 2:
            cut_after_stop(out.sampled_token_ids)
        return out
    forward.__wrapped__ = orig
    return forward


def _record_stop(orig):
    def _update_request_with_output(self, request, new_token_ids, *args, **kwargs):
        n = len(new_token_ids)
        out, stopped = orig(self, request, new_token_ids, *args, **kwargs)
        if stopped and n and len(out) == n:
            _bonus_stop.add(request.request_id)
        return out, stopped
    _update_request_with_output.__wrapped__ = orig
    return _update_request_with_output


def _register_end(orig):
    from vllm.v1.request import RequestStatus

    def _free_request(self, request, *args, **kwargs):
        bonus = request.request_id in _bonus_stop
        _bonus_stop.discard(request.request_id)
        try:
            if request.status == RequestStatus.FINISHED_STOPPED:
                _count('stops')
                _register(self, request, bonus)
            _supersede(self, request)
        except Exception as exc:  # an optimisation; never fail a request over it
            _logger.warning('synth-eot-checkpoint: skipped %s: %r', request.request_id, exc)
        _resumed.pop(request.request_id, None)
        _hit_tokens.pop(request.request_id, None)
        _cpu_end_hits.pop(request.request_id, None)
        _maybe_log()
        try:
            return orig(self, request, *args, **kwargs)
        finally:
            _finish_stores.pop(request.request_id, None)   # no CPU connector (or it skipped the store)
    _free_request.__wrapped__ = orig
    return _free_request


def _register(scheduler, request, bonus):
    """Cache the request's end state (after its stop token) under its exact-length key."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    tokens = request._all_token_ids
    if not tokens or tokens[-1] not in _stop_ids(request):
        return
    if not bonus:
        return _count('not_bonus')
    in_flight = request.num_in_flight_tokens
    if in_flight <= 0:   # no step consumed the stop token: the state slot holds the state before it
        return _count('not_in_flight')
    coordinator = scheduler.kv_cache_manager.coordinator
    groups = _groups(coordinator)
    if not groups:
        return
    stop_pos = len(tokens) - 1
    end = stop_pos + 1
    bs = groups[0][1].block_size
    if not EDGE <= stop_pos % bs <= bs - in_flight - 1:
        return _count('edge')
    pool = coordinator.block_pool
    blocks = []
    for gid, manager, is_mamba in groups:
        req_blocks = manager.req_to_blocks.get(request.request_id) or []
        if stop_pos // bs >= len(req_blocks):
            return _count('busy_block')
        block = req_blocks[stop_pos // bs]
        if block.is_null or (is_mamba and (block.block_hash is not None or block.ref_cnt != 1)):
            return _count('busy_block')
        blocks.append((gid, block))
    key = end_key(tokens, request.block_hashes, end, pool.hash_block_size)
    for gid, block in blocks:
        pool._insert_block_hash(make_block_hash_with_group_id(key, gid), block,
                                num_tokens=end if block.block_hash is None else None)
    _finish_stores[request.request_id] = [block.block_id for _, block in blocks]
    boundary = _boundary_hash(next(m for _, m, is_mamba in groups if is_mamba), request)
    _ends[key] = boundary
    if boundary is not None:
        _boundary_end[boundary] = key
        if request.request_id in _resumed and not _shared(boundary):
            # Not a first turn (it resumed from its previous answer's end state): its boundary state is only the
            # fallback for a diverging re-render from now on.
            _demote_states(scheduler, [boundary])
            _count('early_boundary')
    while len(_ends) > MAX_TRACKED:
        old, old_boundary = _ends.popitem(last=False)
        if old_boundary is not None and _boundary_end.get(old_boundary) == old:
            del _boundary_end[old_boundary]
    _count('registered')


def _store_end(orig):
    from vllm.v1.simple_kv_offload.manager import TransferMeta

    def _queue_finished_eager_store(self, request, block_ids):
        orig(self, request, block_ids)
        ids = _finish_stores.pop(request.request_id, None)
        pool = self._gpu_block_pool
        if not ids or pool is None:
            return
        queued = {b for t in self._pending_finished_stores for b in t.gpu_block_ids}
        ids = [b for b in ids if b not in queued and b not in self._in_flight_store_gpu_blocks]
        if not ids or len(ids) >= self.cpu_block_pool.get_num_free_blocks():
            return
        cpu_blocks = self.cpu_block_pool.get_new_blocks(len(ids))
        self._pending_finished_stores.append(TransferMeta(
            ids, [b.block_id for b in cpu_blocks], [{} for _ in ids] if self.enable_kv_cache_events else None))
        self._in_flight_store_gpu_blocks.update(ids)
        pool.touch([pool.blocks[b] for b in ids])
        _count('stored', len(ids))
    _queue_finished_eager_store.__wrapped__ = orig
    return _queue_finished_eager_store


# --------------------------------------------------------------------------------------------------- lookups

def _find_end(pool, groups, request, start_tokens, min_end, null_block):
    """Longest end-state hit past min_end whose blocks are all cached in `pool`, as per-group block lists covering
    [start_tokens, end) (start_tokens is block aligned); or None."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    hb = pool.hash_block_size
    bs = groups[0][1].block_size
    lookup = pool.cached_block_hash_to_block
    for end, key in _candidates(request, hb):
        if end <= min_end:
            break
        tail_idx = (end - 1) // bs
        first = start_tokens // bs
        out = []
        for gid, _, is_mamba in groups:
            tail = lookup.get_one_block(make_block_hash_with_group_id(key, gid))
            if tail is None:
                break
            if is_mamba:
                out.append([null_block] * (tail_idx - first) + [tail])
                continue
            full = []
            for i in range(first, tail_idx):
                block = lookup.get_one_block(make_block_hash_with_group_id(request.block_hashes[(i + 1) * bs // hb - 1],
                                                                           gid))
                if block is None:
                    break
                full.append(block)
            if len(full) != tail_idx - first:
                break
            out.append(full + [tail])
        else:
            return end, key, out
    return None


def _gpu_lookup(orig):
    def get_computed_blocks(self, request):
        blocks, num_computed, boundary = orig(self, request)
        _hit_tokens[request.request_id] = num_computed
        _resumed.pop(request.request_id, None)
        try:
            if not _ends or not self.prefix_cache_lookup_enabled(request) or _skip(request):
                return blocks, num_computed, boundary
            groups = _groups(self.coordinator)
            if not groups:
                return blocks, num_computed, boundary
            found = _find_end(self.block_pool, groups, request, 0, num_computed + MIN_GAIN, self.block_pool.null_block)
            if found is None:
                return blocks, num_computed, boundary
            end, key, per_group = found
            full = [[] for _ in self.kv_cache_config.kv_cache_groups]
            for (gid, _, _), group_blocks in zip(groups, per_group):
                full[gid] = group_blocks
            _resumed[request.request_id] = key
            _hit_tokens[request.request_id] = end
            _count('gpu_hits')
            _count('hit_tokens', end - num_computed)
            return self.create_kv_cache_blocks(tuple(full)), end, 0
        except Exception as exc:
            _logger.warning('synth-eot-checkpoint: GPU lookup skipped for %s: %r', request.request_id, exc)
            return blocks, num_computed, boundary
    get_computed_blocks.__wrapped__ = orig
    return get_computed_blocks


def _cpu_lookup(orig):
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        _cpu_end_hits.pop(request.request_id, None)
        hit, is_async = orig(self, request, num_computed_tokens)
        if hit:
            _hit_tokens[request.request_id] = num_computed_tokens + hit
        try:
            if (hit is None or not _ends or request.skip_reading_prefix_cache or num_computed_tokens % self.block_size
                    or _skip(request)):
                return hit, is_async
            groups = _groups(self.cpu_coordinator)
            if not groups:
                return hit, is_async
            found = _find_end(self.cpu_block_pool, groups, request, num_computed_tokens,
                              num_computed_tokens + (hit or 0) + MIN_GAIN, self.cpu_block_pool.null_block)
            if found is None:
                return hit, is_async
            end, key, per_group = found
            if stale := self._pending_cpu_hits.pop(request.request_id, None):
                self._free_pending_cpu_hit(stale)
            full = [[] for _ in self.cpu_kv_cache_config.kv_cache_groups]
            for (gid, _, _), group_blocks in zip(groups, per_group):
                full[gid] = group_blocks
            self.cpu_block_pool.touch([b for g in full for b in g if not b.is_null])
            self._pending_cpu_hits[request.request_id] = (tuple(full), end - num_computed_tokens, num_computed_tokens)
            _cpu_end_hits[request.request_id] = end
            _resumed[request.request_id] = key
            _hit_tokens[request.request_id] = end
            _count('cpu_hits')
            _count('hit_tokens', end - num_computed_tokens - (hit or 0))
            return end - num_computed_tokens, True
        except Exception as exc:
            _logger.warning('synth-eot-checkpoint: CPU lookup skipped for %s: %r', request.request_id, exc)
            return hit, is_async
    get_num_new_matched_tokens.__wrapped__ = orig
    return get_num_new_matched_tokens


def _cpu_alloc(orig):
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        end = _cpu_end_hits.pop(request.request_id, None)
        if end is None or num_external_tokens == 0 or num_external_tokens % self.hash_block_size == 0:
            return orig(self, request, blocks, num_external_tokens)
        # An end-state hit is not hash-unit aligned; the stock method only asserts that alignment.
        saved = self.hash_block_size
        self.hash_block_size = 1
        try:
            return orig(self, request, blocks, num_external_tokens)
        finally:
            self.hash_block_size = saved
    update_state_after_alloc.__wrapped__ = orig
    return update_state_after_alloc


# ------------------------------------------------------------------------------------------------ supersede

def _supersede(scheduler, request):
    """When a request that resumed from turn i's end state or prompt-boundary state finishes normally, turn i's end
    state is superseded, and so is its boundary state when the request resumed from the end state (unless another
    request also resumed from it). Both leave the CPU and GPU tiers first. (synth-cpu-supersede demotes a boundary
    state the request itself resumed from.)"""
    from vllm.v1.request import RequestStatus
    if request.status not in (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED):
        return
    key = _resumed.get(request.request_id)
    boundary = None
    if key is not None:
        if key not in _ends:
            return
        boundary = _ends.pop(key)
        if boundary is not None:
            if _boundary_end.get(boundary) == key:
                del _boundary_end[boundary]
            if _shared(boundary):
                _count('shared')
                boundary = None
            else:
                _count('superseded_boundary')
        _count('superseded_end')
    else:
        tokens = _hit_tokens.get(request.request_id, 0)
        hb = scheduler.kv_cache_manager.block_pool.hash_block_size
        if tokens <= 0 or tokens % hb or tokens // hb > len(request.block_hashes):
            return
        key = _boundary_end.pop(bytes(request.block_hashes[tokens // hb - 1]), None)
        if key is None or key not in _ends:
            return
        del _ends[key]
        _count('superseded_end')
    _demote_states(scheduler, [key] if boundary is None else [key, boundary])


def _demote_states(scheduler, state_hashes):
    """Demote the Mamba blocks cached under these state hashes on the CPU and GPU tiers."""
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    mamba = [gid for gid, _, is_mamba in _groups(scheduler.kv_cache_manager.coordinator) or [] if is_mamba]
    keys = [make_block_hash_with_group_id(h, gid) for h in state_hashes for gid in mamba]
    cpu = getattr(getattr(scheduler, 'connector', None), 'scheduler_manager', None)
    if cpu is not None and getattr(cpu, 'cpu_block_pool', None) is not None:
        _count('demoted_cpu', _demote(cpu.cpu_block_pool, keys))
    _count('demoted_gpu', _demote(scheduler.kv_cache_manager.block_pool, keys))


def _shared(boundary):
    """Whether some request resumed from the boundary state (synth-cpu-supersede's count, if it is loaded)."""
    import sys
    supersede = sys.modules.get('synth_cpu_supersede')
    produced = getattr(supersede, '_produced', None)
    if not produced:
        return False
    from vllm.v1.core.kv_cache_utils import BlockHash
    return produced.get(BlockHash(boundary), 0) > 0


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
    if now - _last_log[0] < 60 or _logger is None:
        return
    window = ' '.join(f'{k}={v}' for k, v in _window.items())
    totals = ' '.join(f'{k}={v}' for k, v in _totals.items())
    _logger.info('synth-eot-checkpoint %s last %.0fs: %s | totals: %s | tracked %d', VERSION, now - _last_log[0],
                 window, totals, len(_ends))
    for k in _window:
        _window[k] = 0
    _last_log[0] = now
