"""synth-evict-dead v1: a vLLM general plugin that changes only the eviction order of freed KV blocks.

vLLM's BlockPool.free_blocks puts unhashed blocks at the front of the free queue (reused first) and hashed
blocks at the back (LRU). An agent's next turn extends its previous prompt, so two kinds of hashed blocks
freed by a finished request can never serve that next turn, yet stock LRU keeps them ahead of the idle
agents' reusable prefixes. This plugin sends them to the front instead:

  * attention / MTP-draft blocks whose hash covers tokens past the request's prompt (pure completion
    blocks, and the prompt-tail block once decode filled it and its partial prompt hash was promoted to
    a full hash over completion tokens): an extension of this prompt hits them only if it reproduces the
    sampled completion token for token, which a re-rendered chat history does not. Length-capped requests
    are skipped: a client continuing a truncated answer may send it back verbatim;
  * Mamba state snapshots this request got as a prefix hit and then superseded with a later hashed state
    of its own (align mode keeps one state per reusable boundary). They are skipped when another request
    also hit them, when another request still holds them, when they sit on the request's shared-prefix
    junction, or when they were re-keyed or evicted meanwhile.

Hashes, lookups and outputs are unchanged; a misclassified block costs only a recomputation. Nothing is
patched unless SYNTH_EVICT_DEAD=1. Every 60 s the plugin logs how many freed blocks went to each class.
"""
import os
import time

VERSION = 'v1'
LOG_SECONDS = 60

_patched = False
_ctx = None        # the Request whose blocks KVCacheManager.free / pop_blocks_for_free is releasing
_marks = {}        # id(block) -> stats key; the block goes to the queue front when its last reference is freed
_hits = {}         # (id(manager), request_id) -> ([(block, position, hash)], hit block ids, highest hit position)
_created = {}      # (id(manager), request_id) -> highest position of a Mamba state the request hashed itself
_hit_count = {}    # id(block) -> (hash, number of requests that hit the block under that hash)
_window = dict.fromkeys(('unhashed', 'hashed_kept', 'demoted_attn', 'demoted_mamba', 'demoted_mamba_late'), 0)
_totals = dict(_window)
_last_log = [time.monotonic()]
_logger = None


def stats():
    return dict(_totals)


def register():
    """vllm.general_plugins entry point; vLLM calls it in every process (EngineCore included)."""
    if os.environ.get('SYNTH_EVICT_DEAD') != '1':
        return
    _patch()


def _patch():
    global _patched, _logger
    if _patched:
        return
    from vllm.logger import init_logger
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SingleTypeKVCacheManager
    from vllm.v1.request import RequestStatus

    _logger = init_logger('vllm.synth_evict_dead')  # vLLM's handlers live on the "vllm" logger
    KVCacheManager.free = _with_request(KVCacheManager.free)
    KVCacheManager.pop_blocks_for_free = _with_request(KVCacheManager.pop_blocks_for_free)
    SingleTypeKVCacheManager.pop_blocks_for_free = _classify_on_pop(SingleTypeKVCacheManager.pop_blocks_for_free,
                                                                    MambaManager, RequestStatus.FINISHED_LENGTH_CAPPED)
    SingleTypeKVCacheManager.add_local_computed_blocks = _record_hits(
        SingleTypeKVCacheManager.add_local_computed_blocks, MambaManager)
    MambaManager.cache_blocks = _record_created(MambaManager.cache_blocks)
    BlockPool.free_blocks = _free_blocks(BlockPool.free_blocks)
    _patched = True
    _logger.info('synth-evict-dead %s active (pid %d): completion-covering attention blocks and superseded '
                 'Mamba hit states are evicted first', VERSION, os.getpid())


def _with_request(orig):
    def wrapper(self, request):
        global _ctx
        previous, _ctx = _ctx, request
        try:
            return orig(self, request)
        finally:
            _ctx = previous
    wrapper.__wrapped__ = orig
    return wrapper


def _classify_on_pop(orig, mamba_cls, length_capped):
    def pop_blocks_for_free(self, request_id):
        blocks = orig(self, request_id)
        key = (id(self), request_id)
        entry, created = _hits.pop(key, None), _created.pop(key, None)
        request = _ctx
        if (request is None or request.request_id != request_id or not request.is_finished()
                or not self.enable_caching):
            return blocks  # preemption, connector paths and failed loads keep stock LRU
        if isinstance(self, mamba_cls):
            if entry and created is not None:
                _demote_superseded_states(self, request, blocks, entry[0], created)
        elif request.status != length_capped:
            prompt = request.num_prompt_tokens
            for block in blocks:
                if block.is_null or block.ref_cnt != 1 or block.block_hash is None:
                    continue
                covered = block.block_hash_num_tokens
                if covered is not None and covered > prompt:
                    _marks[id(block)] = 'demoted_attn'
        return blocks
    pop_blocks_for_free.__wrapped__ = orig
    return pop_blocks_for_free


def _demote_superseded_states(manager, request, blocks, hits, created):
    held = {id(b) for b in blocks}
    late = []
    for block, position, block_hash in hits:
        if position is None or position >= created or position == request.shared_prefix_boundary:
            continue
        if block.is_null or block.block_hash != block_hash:
            continue  # re-keyed or evicted since the hit
        count = _hit_count.get(id(block))
        if count is None or count[0] != block_hash or count[1] > 1:
            continue  # another request reused it: a shared prefix, keep LRU
        if id(block) in held:
            if block.ref_cnt == 1:
                _marks[id(block)] = 'demoted_mamba'
        elif block.ref_cnt == 0:
            late.append(block)  # released mid-request (retired state or CoW source), already queued LRU
    if late:
        queue = manager.block_pool.free_block_queue
        for block in late:
            queue.remove(block)
        queue.prepend_n(late)
        _count('demoted_mamba_late', len(late))


def _record_hits(orig, mamba_cls):
    def add_local_computed_blocks(self, request_id, new_computed_blocks, *args, **kwargs):
        orig(self, request_id, new_computed_blocks, *args, **kwargs)
        if not isinstance(self, mamba_cls) or not self.enable_caching:
            return
        hits = []
        for block in self.req_to_blocks.get(request_id, ()):  # empty before the call: only hits and nulls
            if block.is_null or block.block_hash is None:
                continue
            block_hash = block.block_hash
            hits.append((block, block.block_hash_num_tokens, block_hash))
            count = _hit_count.get(id(block))
            _hit_count[id(block)] = (block_hash, count[1] + 1) if count and count[0] == block_hash else (block_hash, 1)
        if hits:
            _hits[(id(self), request_id)] = (hits, frozenset(id(b) for b, _, _ in hits),
                                             max((p for _, p, _ in hits if p is not None), default=-1))
    add_local_computed_blocks.__wrapped__ = orig
    return add_local_computed_blocks


def _note_created(manager, request_id, tail=6):
    """Raise the request's highest self-hashed Mamba position. Align mode hashes states near the
    computed frontier and nulls them a step or two later, so the last few entries suffice. Only
    requests that got a Mamba hit need it, and only until their own state passes every hit."""
    key = (id(manager), request_id)
    entry = _hits.get(key)
    if entry is None:
        return
    _, hit_ids, top = entry
    best = _created.get(key, -1)
    if best > top:
        return
    for block in manager.req_to_blocks.get(request_id, ())[-tail:]:
        if block.is_null or block.block_hash is None or id(block) in hit_ids:
            continue
        covered = block.block_hash_num_tokens
        if covered is not None and covered > best:
            best = covered
    if best >= 0:
        _created[key] = best


def _record_created(orig):
    def cache_blocks(self, request, *args, **kwargs):
        orig(self, request, *args, **kwargs)
        if self.enable_caching:
            _note_created(self, request.request_id)
    cache_blocks.__wrapped__ = orig
    return cache_blocks


def _free_blocks(orig):
    def free_blocks(self, ordered_blocks):
        blocks = ordered_blocks if isinstance(ordered_blocks, list) else list(ordered_blocks)
        dead = []
        if _marks:
            rest = []
            for block in blocks:
                if id(block) in _marks and (block.pool is None or block.pool is self):
                    dead.append((block, _marks.pop(id(block))))
                else:
                    rest.append(block)
        else:
            rest = blocks
        unhashed = kept = 0
        for block in rest:
            if (block.pool is None or block.pool is self) and block.ref_cnt == 1 and not block.is_null:
                if block.block_hash is None or not self.enable_caching:
                    unhashed += 1
                else:
                    kept += 1
        orig(self, rest)
        front = []
        for block, key in dead:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                front.append(block)
                _count(key, 1)
        if front:
            self.free_block_queue.prepend_n(front)
        _count('unhashed', unhashed)
        _count('hashed_kept', kept)
        _maybe_log()
    free_blocks.__wrapped__ = orig
    return free_blocks


def _count(key, n):
    if n:
        _window[key] += n
        _totals[key] += n


def _maybe_log():
    now = time.monotonic()
    if now - _last_log[0] < LOG_SECONDS or _logger is None:
        return
    window = ' '.join(f'{k}={v}' for k, v in _window.items())
    totals = ' '.join(f'{k}={v}' for k, v in _totals.items())
    _logger.info('synth-evict-dead %s freed blocks, last %.0fs: %s | totals: %s', VERSION, now - _last_log[0],
                 window, totals)
    for k in _window:
        _window[k] = 0
    _last_log[0] = now
