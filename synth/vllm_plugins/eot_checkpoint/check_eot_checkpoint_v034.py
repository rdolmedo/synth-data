"""CPU checks for synth-eot-checkpoint's vLLM 0.34 path (v5) against the real vLLM 0.34 classes.

    SYNTH_EOT_CHECKPOINT=1 SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<cpu_supersede dir>:<deployed dir> <python with vLLM 0.34> \
        check_eot_checkpoint_v034.py

No GPU is needed: the vLLM 0.34 Python sources import on CPU (a python-only tree works ahead of a torch venv on
PYTHONPATH, e.g. `PYTHONPATH=<dir holding the 0.34 vllm package>:... venvs/vllm033/bin/python -B`). The plugin is
loaded the way EngineCore loads it (vllm.plugins.load_general_plugins -> `_entry.register`). The GPU side is a real
0.34 KVCacheManager configured as Nemotron 3.5 Super serves on 0.34 (one attention + two Mamba align-mode groups,
hash unit 16, block 64, MTP with lookahead block hashes: num_hash_lookahead_tokens 1 and no trailing-block drop);
the CPU tier is a second real coordinator and block pool, on which the real SimpleCPUOffloadScheduler lookup,
allocation and store-completion methods run through the plugin's wrappers. The scheduler around them is a stand-in
with the attributes the plugin reads. Same scenarios as check_eot_checkpoint.py (vLLM 0.32) plus the 0.34 cases:
unit-aligned ends under lookahead hashes, the block-aligned local prefix the scheduler hands the connector, the
max_cache_hit_length bound, ReplaySSM. Exits non-zero on the first failed check.
"""
import importlib.util
import inspect
import os
import sys
from types import SimpleNamespace

import torch

# Test the copy PYTHONPATH names (a deployed version), not the source tree next to this script.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]

HB, BS, LOOKAHEAD = 16, 64, 1
EOS = 11


def check(cond, message):
    if not cond:
        print('FAIL:', message)
        sys.exit(1)
    print('ok:', message)


def main():
    assert os.environ.get('SYNTH_EOT_CHECKPOINT') == '1', 'run with SYNTH_EOT_CHECKPOINT=1'
    import vllm
    from vllm.plugins import load_general_plugins
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, make_block_hash_with_group_id
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.kv_cache_interface import (AttentionSpec, KVCacheConfig, KVCacheGroup, MambaSpec,
                                            get_mamba_prefill_checkpoint_position)
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as V2RejectionSampler

    print('vLLM', vllm.__version__, 'from', os.path.dirname(vllm.__file__))
    check(importlib.util.find_spec('vllm.v1.sample.rejection_sampler') is None,
          'vllm.v1.sample.rejection_sampler is gone (v4 imports it in register, which would fail engine start)')
    targets = {(Scheduler, '_update_request_with_output'), (Scheduler, '_free_request'),
               (KVCacheManager, 'get_computed_blocks'), (SimpleCPUOffloadScheduler, '_queue_finished_eager_store'),
               (SimpleCPUOffloadScheduler, 'get_num_new_matched_tokens'),
               (SimpleCPUOffloadScheduler, 'update_state_after_alloc'), (V2RejectionSampler, '__call__')}
    real = {(c, n): inspect.signature(getattr(c, n)) for c, n in targets}
    load_general_plugins()
    import synth_eot_checkpoint.vllm034 as plugin
    print('checking', plugin.__file__, plugin.VERSION)
    check(plugin._patched and plugin.VERSION == 'v5', 'the entry point selected the 0.34 path (v5)')
    supersede = sys.modules.get('synth_cpu_supersede.vllm034')
    print('synth-cpu-supersede 0.34 path loaded:', supersede is not None and supersede._patched)
    for (cls, name), sig in real.items():
        method = getattr(cls, name)
        check(hasattr(method, '__wrapped__') and inspect.signature(method.__wrapped__) == sig,
              f'{cls.__name__}.{name} wrapped with the installed signature')
    check(list(inspect.signature(KVCacheManager.get_computed_blocks).parameters) ==
          list(inspect.signature(KVCacheManager.get_computed_blocks.__wrapped__).parameters),
          'the GPU lookup wrapper spells the 0.34 parameters (request, max_cache_hit_length)')

    # V2 model runner: per-request counts are cut after the first counted stop token.
    ids = torch.tensor([[5, EOS, 7, 8], [5, 6, 7, 8], [EOS, 7, 9, 9], [5, 6, 7, EOS], [5, 2, EOS, 9], [EOS, 3, 4, 4]],
                       dtype=torch.int32)
    num_sampled = torch.tensor([4, 4, 2, 4, 3, 0], dtype=torch.int32)
    num_rejected = torch.tensor([0, 0, 2, 0, 1, 1], dtype=torch.int32)
    plugin.cut_counts_after_stop(ids, num_sampled, num_rejected)
    check(num_sampled.tolist() == [2, 4, 1, 4, 2, 0] and num_rejected.tolist() == [2, 0, 3, 0, 2, 1]
          and ids.tolist() == [[5, EOS, -1, -1], [5, 6, 7, 8], [EOS, -1, 9, 9], [5, 6, 7, EOS], [5, 2, -1, 9],
                               [EOS, 3, 4, 4]],
          'V2 sampler counts stop after the first counted stop token (num_rejected grows by the cut)')

    init_none_hash(sha256)
    attn = AttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float16)
    mamba = MambaSpec(block_size=BS, shapes=((4,),), dtypes=(torch.float16,), page_size_padded=attn.page_size_bytes)
    align = MambaManager.config(mamba_cache_mode='align', num_speculative_blocks=3)

    def manager(num_blocks):
        cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], prefix_cache_retention_interval=0,
                            kv_cache_groups=[KVCacheGroup({'a': attn}, FullAttentionManager.config(), BS),
                                             KVCacheGroup({'m0': mamba}, align, BS),
                                             KVCacheGroup({'m1': mamba}, align, BS)])
        return KVCacheManager(cfg, max_model_len=8192, scheduler_block_size=BS, hash_block_size=HB, use_eagle=False,
                              num_hash_lookahead_tokens=LOOKAHEAD, num_prefill_lookahead=LOOKAHEAD)

    gpu, cpu_side = manager(400), manager(400)
    check(gpu.block_pool.num_hash_lookahead_tokens == LOOKAHEAD, 'GPU pool hashes with one lookahead token')
    hasher = get_request_block_hasher(HB, sha256, num_hash_lookahead_tokens=LOOKAHEAD)
    params = SamplingParams(max_tokens=200, stop_token_ids=[EOS])
    params._all_stop_token_ids = {EOS, 2}
    blocks_of = lambda m, rid: [list(g) for g in m.coordinator.get_blocks(rid)]

    def run_turn(rid, prompt, answer, in_flight=4):
        """Prefill + decode one turn on the GPU manager; returns the request after its last step's output."""
        r = Request(rid, list(prompt), params, None, block_hasher=hasher)
        found, n, _ = gpu.get_computed_blocks(r)
        gpu.allocate_slots(r, len(prompt) - n, num_new_computed_tokens=n, new_computed_blocks=found)
        r.num_computed_tokens = len(prompt)
        for t in answer:
            r.append_output_token_ids(t)
            gpu.allocate_slots(r, 1, num_lookahead_tokens=3)
            r.num_computed_tokens += 1
        if in_flight:   # the in-flight step was scheduled from the stop token: num_computed - in_flight = stop
            r.num_computed_tokens += in_flight - 1
        r.num_in_flight_tokens = in_flight
        r.status = RequestStatus.FINISHED_STOPPED
        return r

    no_replayssm = SimpleNamespace(use_replayssm=False)
    sched = SimpleNamespace(kv_cache_manager=gpu, connector=None, max_model_len=8192, cache_config=no_replayssm)

    # Stop detection: the stop token is the step's last output (bonus) only when nothing was trimmed after it.
    probe = Request('probe', [1, 2, 3], params, None, block_hasher=hasher)
    probe.status = RequestStatus.RUNNING
    Scheduler._update_request_with_output(sched, probe, [5, EOS])
    check('probe' in plugin._bonus_stop, 'a stop token ending the step output marks a bonus stop')
    plugin._bonus_stop.clear()
    probe2 = Request('probe2', [1, 2, 3], params, None, block_hasher=hasher)
    probe2.status = RequestStatus.RUNNING
    out, stopped = Scheduler._update_request_with_output(sched, probe2, [EOS, 7, 8])
    check(stopped and out == [EOS] and 'probe2' not in plugin._bonus_stop,
          'a stop token followed by accepted drafts is not a bonus stop (its state slot is overwritten)')

    # Turn 1: prompt 200, answer of 42 tokens ending in EOS (inside block 3, so 0.34's native decode-resume state
    # does not exist; see the end of this script); the in-flight step consumed EOS.
    prompt1 = list(range(1000, 1200))
    answer1 = [2000 + i for i in range(41)] + [EOS]
    a = run_turn('a', prompt1, answer1)
    end = a.num_tokens
    check(len(a.block_hashes) == (end - LOOKAHEAD) // HB, 'request hashes stop one lookahead token short of the end')
    plugin._register(sched, a, bonus=True)
    key = plugin.end_key(a._all_token_ids, a.block_hashes, end, HB, LOOKAHEAD)
    tail = [g[(end - 1) // BS] for g in blocks_of(gpu, 'a')]
    check(all(gpu.block_pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, gid)) is b
              for gid, b in enumerate(tail)), 'end state registered on the attention tail and both Mamba state blocks')
    check(plugin._finish_stores.get('a') == [b.block_id for b in tail], 'end blocks queued for the CPU store')
    boundary_pos = get_mamba_prefill_checkpoint_position(len(prompt1), HB, False, num_hash_lookahead_tokens=LOOKAHEAD)
    check(plugin._ends.get(key) == bytes(a.block_hashes[boundary_pos // HB - 1]) and boundary_pos == 192,
          'end state tracked with its turn\'s boundary state ((prompt - 1) // 16 * 16 with lookahead hashes)')

    # Ineligible stops are counted, not registered.
    before = dict(plugin.stats())
    b0 = run_turn('b0', list(range(3000, 3200)), [2000 + i for i in range(9)] + [EOS], in_flight=0)
    plugin._register(sched, b0, bonus=True)
    check(plugin.stats()['not_in_flight'] == before['not_in_flight'] + 1 and 'b0' not in plugin._finish_stores,
          'a stop whose token no in-flight step consumed is not registered')
    plugin._register(sched, b0, bonus=False)
    check(plugin.stats()['not_bonus'] == before['not_bonus'] + 1, 'a non-bonus stop is not registered')
    rs = SimpleNamespace(kv_cache_manager=gpu, connector=None, cache_config=SimpleNamespace(use_replayssm=True))
    b0.num_in_flight_tokens = 4
    plugin._register(rs, b0, bonus=True)
    check(plugin.stats()['replayssm'] == before['replayssm'] + 1 and 'b0' not in plugin._finish_stores,
          'nothing is registered under ReplaySSM (the running state block is written back only on flush)')
    gpu.free(b0)

    # Store: the plugin appends the end blocks to the finished stores (the stock flush ran first).
    calls = []

    class CpuStand:
        """The SimpleCPUOffloadScheduler attributes the wrapped methods read (real CPU coordinator and pools)."""
        hash_block_size = HB
        block_size = BS
        _lazy_mode = False
        enable_kv_cache_events = False

        def __init__(self):
            self.cpu_coordinator = cpu_side.coordinator
            self.cpu_block_pool = cpu_side.block_pool
            self._gpu_block_pool = gpu.block_pool
            self.cpu_kv_cache_config = cpu_side.kv_cache_config
            self.prefix_cacheable_group_ids = (0, 1, 2)
            self.group_block_sizes = [BS, BS, BS]
            self._pending_cpu_hits, self._pending_local_hits = {}, {}
            self._reqs_to_store, self._reqs_to_load = {}, {}
            self._pending_finished_stores, self._in_flight_store_gpu_blocks = [], set()

        def _queue_finished_eager_store(self, request, block_ids):
            calls.append(('flush', request.request_id))

    for name in ('get_num_new_matched_tokens', 'update_state_after_alloc', '_free_pending_cpu_hit',
                 '_process_store_completion'):
        setattr(CpuStand, name, getattr(SimpleCPUOffloadScheduler, name))
    CpuStand._queue_finished_eager_store = plugin._store_end(CpuStand._queue_finished_eager_store)
    cpu = CpuStand()

    def store(request):
        """Queue the request's end blocks as the connector does at finish, then complete the transfer with the real
        _process_store_completion (registers every GPU hash on the CPU copy, releases both refs)."""
        cpu._queue_finished_eager_store(request, None)
        transfer = cpu._pending_finished_stores.pop()
        cpu._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids, transfer.block_meta)
        cpu._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)
        return transfer

    cpu._queue_finished_eager_store(a, None)
    check(calls == [('flush', 'a')] and cpu._pending_finished_stores
          and cpu._pending_finished_stores[-1].gpu_block_ids == [b.block_id for b in tail]
          and all(b.ref_cnt == 2 for b in tail), 'end blocks appended to the finished stores and pinned')
    transfer = cpu._pending_finished_stores.pop()
    cpu._process_store_completion(transfer.gpu_block_ids, transfer.cpu_block_ids, transfer.block_meta)
    cpu._in_flight_store_gpu_blocks.difference_update(transfer.gpu_block_ids)
    check(all(cpu_side.block_pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, g))
              is not None for g in range(3)), 'the real store completion registers the end key on the CPU copies')

    def mirror_full_attention(request, n):
        """Full attention blocks reach the CPU tier through eager stores."""
        for i, blk in enumerate(blocks_of(gpu, request.request_id)[0][:n]):
            cb = cpu_side.block_pool.get_new_blocks(1)[0]
            cpu_side.block_pool._insert_block_hash(blk.block_hash, cb, num_tokens=(i + 1) * BS)
            cpu_side.block_pool.free_blocks([cb])

    check(all(b.block_hash is not None for b in blocks_of(gpu, 'a')[0][:(end - 1) // BS]),
          'turn 1\'s full attention blocks are cached under lookahead hashes')
    mirror_full_attention(a, (end - 1) // BS)
    gpu.free(a)

    # Turn 2 (GPU tier): the re-rendered answer + tool output; the hit lands right after EOS.
    suffix = [1010, 10, 3263, 1010, 16, 1010] + list(range(5000, 5100))
    b = Request('b', list(a._all_token_ids) + suffix, params, None, block_hasher=hasher)
    check(plugin.end_key(b._all_token_ids, b.block_hashes, end, HB, LOOKAHEAD) == key,
          'turn 2 computes the same key from its own (longer) hash list')
    found, n, boundary = gpu.get_computed_blocks(b)
    check(n == end and plugin._resumed.get('b') == key, f'GPU lookup resumes at the end state ({n} == {end})')
    gpu.allocate_slots(b, b.num_tokens - n, num_new_computed_tokens=n, new_computed_blocks=found)
    cows = {(s.block_id, d.block_id) for m in gpu.coordinator.single_type_managers for s, d in m._pending_cow_copies}
    check({s for s, _ in cows} == {blk.block_id for blk in tail},
          'the hit copies the attention tail and both Mamba states before writing (copy-on-write)')

    # The scheduler hands the connector the block-aligned local prefix; the same end state on the CPU tier is not
    # offered again (the scheduler would keep the local hit and the pin would leak into a warning).
    aligned = n - n % BS
    hit, is_async = cpu.get_num_new_matched_tokens(b, aligned)
    check(hit == 0 and not is_async and 'b' not in cpu._pending_cpu_hits and plugin._resumed.get('b') == key,
          f'CPU lookup from the aligned prefix ({aligned}) does not re-offer the local end-state hit ({end})')

    # Bounds: max_cache_hit_length below the end, a diverged re-render, a request that skips cache reads.
    c0 = Request('c0', list(a._all_token_ids) + suffix, params, None, block_hasher=hasher)
    _, n_c0, _ = gpu.get_computed_blocks(c0, end - 1)
    check(n_c0 < end and 'c0' not in plugin._resumed, 'max_cache_hit_length below the end disables the end-state hit')
    c_tokens = list(a._all_token_ids)
    c_tokens[len(prompt1) + 5] += 1
    c = Request('c', c_tokens + suffix, params, None, block_hasher=hasher)
    _, n_c, _ = gpu.get_computed_blocks(c)
    check(n_c < len(prompt1) and 'c' not in plugin._resumed, 'a re-render that diverges inside the answer misses')
    skip = SamplingParams(max_tokens=200, stop_token_ids=[EOS], skip_reading_prefix_cache=True)
    skip._all_stop_token_ids = {EOS, 2}
    s = Request('s', list(a._all_token_ids) + suffix, skip, None, block_hasher=hasher)
    _, n_s, _ = gpu.get_computed_blocks(s)
    hit_s, _ = cpu.get_num_new_matched_tokens(s, 0)
    check(n_s == 0 and hit_s == 0 and 's' not in plugin._resumed and s.skip_reading_external_kv,
          'a request that skips prefix-cache reads gets no end-state hit on either tier')

    # Turn 2 (CPU tier): GPU copies evicted; the CPU lookup returns the end state and its full attention blocks.
    for blk in tail:
        gpu.block_pool._maybe_evict_cached_block(blk)
    d = Request('d', list(a._all_token_ids) + suffix, params, None, block_hasher=hasher)
    found_d, n_d, _ = gpu.get_computed_blocks(d)
    aligned_d = n_d - n_d % BS
    hit, is_async = cpu.get_num_new_matched_tokens(d, aligned_d)
    check(hit == end - aligned_d and is_async and plugin._resumed.get('d') == key,
          f'CPU lookup returns the end state past the local hit {n_d} ({aligned_d} + {hit} == {end})')
    pending = cpu._pending_cpu_hits['d'][0]
    check([len(g) for g in pending] == [(end - 1) // BS + 1 - aligned_d // BS] * 3
          and pending[1][-1].block_hash is not None,
          'CPU hit holds the full attention blocks + tail and the Mamba state at the tail index')
    # The scheduler drops the local partial tail (the external hit is longer), allocates the request's GPU blocks
    # for the external tokens, then the connector plans the load.
    found_d = gpu.truncate_computed_blocks(found_d, aligned_d)
    dg = gpu.allocate_slots(d, 0, num_new_computed_tokens=aligned_d, new_computed_blocks=found_d,
                            num_external_computed_tokens=hit, delay_cache_blocks=True)
    check(dg is not None, 'GPU blocks allocated for the unaligned external hit')
    cpu.update_state_after_alloc(d, gpu.get_blocks('d'), hit)
    load = cpu._reqs_to_load['d'].transfer_meta
    want_cpu = [blk.block_id for g in pending for blk in g if not blk.is_null]
    check(load.cpu_block_ids == want_cpu and len(load.gpu_block_ids) == len(want_cpu) and cpu.hash_block_size == HB,
          'the load copies every non-null CPU block (alignment assert bypassed for this hit, then restored)')
    gpu_ids = gpu.get_block_ids('d')
    mamba_targets = load.gpu_block_ids[-2:]
    check(all(gpu_ids[g][(end - 1) // BS] == mamba_targets[i] for i, g in enumerate((1, 2))),
          'Mamba states load into the request\'s state block at the end position')
    cpu_side.block_pool.free_blocks([cpu_side.block_pool.blocks[i] for i in load.cpu_block_ids])

    # Supersede: turn 2 finished after resuming from turn 1's end state -> turn 1's end + boundary states demoted.
    boundary_hash = plugin._ends[key]
    b.status = RequestStatus.FINISHED_STOPPED
    pool = cpu_side.block_pool
    end_cpu = [pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, g)) for g in (1, 2)]
    sched_cpu = SimpleNamespace(kv_cache_manager=gpu, connector=SimpleNamespace(scheduler_manager=cpu),
                                cache_config=no_replayssm)
    plugin._supersede(sched_cpu, b)
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    check(key not in plugin._ends and head.block_id in {blk.block_id for blk in end_cpu if blk is not None},
          'turn 1 end state demoted to the front of the CPU free queue')
    check(boundary_hash is None or boundary_hash not in plugin._boundary_end, 'turn 1 boundary state untracked')

    # Lookahead hashes: an end on a 16-token unit boundary. v4's key would read the hash of the unit ending at the
    # end, which does not exist yet (it covers the next token); v5 keys off the previous unit.
    e = run_turn('e', list(range(60000, 60200)), [4000 + i for i in range(39)] + [EOS])
    e_end = e.num_tokens
    check(e_end % HB == 0 and len(e.block_hashes) == e_end // HB - 1,
          f'unit-aligned end {e_end}: the hash of its last unit does not exist at registration')
    reg = plugin.stats()['registered']
    plugin._register(sched, e, bonus=True)
    check(plugin.stats()['registered'] == reg + 1, 'the unit-aligned end state registers')
    gpu.free(e)
    e2 = Request('e2', list(e._all_token_ids) + suffix, params, None, block_hasher=hasher)
    _, n_e2, _ = gpu.get_computed_blocks(e2)
    check(n_e2 == e_end and 'e2' in plugin._resumed, f'and the next turn resumes exactly there ({n_e2})')

    # A local stock hit that is not block aligned (prompt 230: its boundary state is at 224, a partial-tail hit)
    # still lets a longer CPU end state through.
    g1 = run_turn('g1', list(range(70000, 70230)), [5000 + i for i in range(39)] + [EOS])
    plugin._register(sched, g1, bonus=True)
    g_tail = [grp[(g1.num_tokens - 1) // BS] for grp in blocks_of(gpu, 'g1')]
    store(g1)
    mirror_full_attention(g1, (g1.num_tokens - 1) // BS)
    gpu.free(g1)
    for blk in g_tail:
        gpu.block_pool._maybe_evict_cached_block(blk)
    g2 = Request('g2', list(g1._all_token_ids) + suffix, params, None, block_hasher=hasher)
    plugin._hit_tokens['g2'] = 224          # as the GPU lookup records a prompt-boundary hit at 224
    hit_g, _ = cpu.get_num_new_matched_tokens(g2, 192)
    check(hit_g == g1.num_tokens - 192 and hit_g > 224 - 192,
          'an unaligned local hit (224) is superseded by the CPU end state (the scheduler loads from 192)')
    cpu._free_pending_cpu_hit(cpu._pending_cpu_hits.pop('g2'))

    # v4: a turn that resumed from an end state demotes its own boundary state as soon as its end state registers.
    f = run_turn('f', list(range(80000, 80200)), [3000 + i for i in range(19)] + [EOS])
    f_boundary = plugin._boundary_hash(gpu.coordinator.single_type_managers[1], f)
    staged = cpu_side.block_pool.get_new_blocks(2)
    for g, blk in zip((1, 2), staged):
        cpu_side.block_pool._insert_block_hash(make_block_hash_with_group_id(f_boundary, g), blk, num_tokens=192)
    cpu_side.block_pool.free_blocks(staged)
    plugin._resumed['f'] = b'previous end state'
    before = plugin.stats()['early_boundary']
    plugin._register(sched_cpu, f, bonus=True)
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    check(plugin.stats()['early_boundary'] == before + 1 and head.block_id in {b.block_id for b in staged},
          'a later turn demotes its own boundary state when its end state registers')
    h_req = run_turn('h', list(range(90000, 90200)), [3000 + i for i in range(19)] + [EOS])
    plugin._register(sched_cpu, h_req, bonus=True)
    check(plugin.stats()['early_boundary'] == before + 1, 'a first turn (no end-state resume) keeps its boundary state')
    # 0.34's native decode-resume state (MambaManager.cache_blocks_on_free, retention interval 0): an answer that
    # crosses a block boundary leaves the state at that boundary cached on the GPU. The end state still wins when it
    # reaches MIN_GAIN tokens past it; closer than that, the native state serves the hit (a short recompute).
    for rid, n_answer, want in (('n1', 40, 256), ('n2', 70, 300)):
        t1 = run_turn(rid, list(range(110000 + 1000 * n_answer, 110000 + 1000 * n_answer + 230)),
                      [6000 + i for i in range(n_answer - 1)] + [EOS])
        plugin._register(sched, t1, bonus=True)
        gpu.free(t1)
        t2 = Request(rid + 'b', list(t1._all_token_ids) + suffix, params, None, block_hasher=hasher)
        _, n_t2, _ = gpu.get_computed_blocks(t2)
        check(n_t2 == want and ((rid + 'b') in plugin._resumed) == (want == t1.num_tokens),
              f'answer {230}..{t1.num_tokens} crossing 256: next turn resumes at {n_t2} '
              f'({"end state" if want == t1.num_tokens else "native decode-resume state, end within MIN_GAIN"})')
    print('all checks passed; stats', plugin.stats())


if __name__ == '__main__':
    main()
