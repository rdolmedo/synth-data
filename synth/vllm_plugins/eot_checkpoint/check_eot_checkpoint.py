"""CPU checks for synth-eot-checkpoint against the real vLLM classes of venvs/vllm032.

    SYNTH_EOT_CHECKPOINT=1 PYTHONPATH=<deployed dir> <vllm032>/bin/python check_eot_checkpoint.py

The plugin is loaded the way EngineCore loads it (vllm.plugins.load_general_plugins). The GPU side is a real
KVCacheManager (one attention + two Mamba align-mode groups, MTP, hash unit 16, block 64); the CPU tier is a second
real coordinator and block pool, on which the real SimpleCPUOffloadScheduler lookup and allocation methods run through
the plugin's wrappers. The scheduler around them is a stand-in with the attributes the plugin reads. Exits non-zero on
the first failed check.
"""
import inspect
import os
import sys
from types import SimpleNamespace

import torch

# Test the copy PYTHONPATH names (a deployed version), not the source tree next to this script.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]

HB, BS = 16, 64
EOS = 11


def check(cond, message):
    if not cond:
        print('FAIL:', message)
        sys.exit(1)
    print('ok:', message)


def main():
    assert os.environ.get('SYNTH_EOT_CHECKPOINT') == '1', 'run with SYNTH_EOT_CHECKPOINT=1'
    from vllm.plugins import load_general_plugins
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, make_block_hash_with_group_id
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as V2RejectionSampler
    from vllm.v1.simple_kv_offload import manager as offload
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    targets = {(Scheduler, '_update_request_with_output'), (Scheduler, '_free_request'),
               (KVCacheManager, 'get_computed_blocks'), (SimpleCPUOffloadScheduler, '_queue_finished_eager_store'),
               (SimpleCPUOffloadScheduler, 'get_num_new_matched_tokens'),
               (SimpleCPUOffloadScheduler, 'update_state_after_alloc'), (RejectionSampler, 'forward'),
               (V2RejectionSampler, '__call__')}
    real = {(c, n): inspect.signature(getattr(c, n)) for c, n in targets}
    load_general_plugins()
    import synth_eot_checkpoint as plugin
    print('checking', plugin.__file__, plugin.VERSION)
    for (cls, name), sig in real.items():
        method = getattr(cls, name)
        check(hasattr(method, '__wrapped__') and inspect.signature(method.__wrapped__) == sig,
              f'{cls.__name__}.{name} wrapped with the installed signature')

    # Sampled tokens are cut after the first stop token (rows: accepted drafts + bonus, -1 = rejected).
    ids = torch.tensor([[5, EOS, 7, 8], [5, 6, 7, 8], [EOS, 7, -1, -1], [5, 6, 7, EOS], [5, 2, EOS, 9]],
                       dtype=torch.int32)
    plugin.cut_after_stop(ids)
    check(ids.tolist() == [[5, EOS, -1, -1], [5, 6, 7, 8], [EOS, -1, -1, -1], [5, 6, 7, EOS], [5, 2, -1, -1]],
          'sampled tokens after the first stop token become placeholders (valid prefix stays contiguous)')

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
    attn = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float16)
    mamba = MambaSpec(block_size=BS, shapes=((4,),), dtypes=(torch.float16,), mamba_cache_mode='align',
                      num_speculative_blocks=3, page_size_padded=attn.page_size_bytes)

    def manager(num_blocks):
        cfg = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], prefix_cache_retention_interval=0,
                            kv_cache_groups=[KVCacheGroupSpec(['a'], attn), KVCacheGroupSpec(['m0'], mamba),
                                             KVCacheGroupSpec(['m1'], mamba)])
        return KVCacheManager(cfg, max_model_len=8192, scheduler_block_size=BS, hash_block_size=HB, use_eagle=True)

    gpu, cpu_side = manager(400), manager(400)
    hasher = get_request_block_hasher(HB, sha256)
    params = SamplingParams(max_tokens=200, stop_token_ids=[EOS])
    params._all_stop_token_ids = {EOS, 2}
    blocks_of = lambda m, rid: [[b for b in g] for g in m.coordinator.get_blocks(rid)]

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
        r.num_in_flight_tokens = in_flight
        r.status = RequestStatus.FINISHED_STOPPED
        return r

    sched = SimpleNamespace(kv_cache_manager=gpu, connector=None, max_model_len=8192)

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

    # Turn 1: prompt 230, answer of 40 tokens ending in EOS; the in-flight step consumed EOS.
    prompt1 = list(range(1000, 1230))
    answer1 = [2000 + i for i in range(39)] + [EOS]
    a = run_turn('a', prompt1, answer1)
    end = a.num_tokens
    plugin._register(sched, a, bonus=True)
    key = plugin.end_key(a._all_token_ids, a.block_hashes, end, HB)
    tail = [g[(end - 1) // BS] for g in blocks_of(gpu, 'a')]
    check(all(gpu.block_pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, gid)) is b
              for gid, b in enumerate(tail)), 'end state registered on the attention tail and both Mamba state blocks')
    check(plugin._finish_stores.get('a') == [b.block_id for b in tail], 'end blocks queued for the CPU store')
    check(key in plugin._ends, 'end state tracked with its turn\'s boundary state')

    # Ineligible stops are counted, not registered.
    before = dict(plugin.stats())
    b0 = run_turn('b0', list(range(3000, 3230)), [2000 + i for i in range(9)] + [EOS], in_flight=0)
    plugin._register(sched, b0, bonus=True)
    check(plugin.stats()['not_in_flight'] == before['not_in_flight'] + 1 and 'b0' not in plugin._finish_stores,
          'a stop whose token no in-flight step consumed is not registered')
    plugin._register(sched, b0, bonus=False)
    check(plugin.stats()['not_bonus'] == before['not_bonus'] + 1, 'a non-bonus stop is not registered')
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
            self._pending_cpu_hits, self._reqs_to_store, self._reqs_to_load = {}, {}, {}
            self._pending_finished_stores, self._in_flight_store_gpu_blocks = [], set()

        def _queue_finished_eager_store(self, request, block_ids):
            calls.append(('flush', request.request_id))

    for name in ('get_num_new_matched_tokens', 'update_state_after_alloc', '_free_pending_cpu_hit'):
        setattr(CpuStand, name, getattr(SimpleCPUOffloadScheduler, name))
    CpuStand._queue_finished_eager_store = plugin._store_end(CpuStand._queue_finished_eager_store)
    cpu = CpuStand()
    cpu._queue_finished_eager_store(a, None)
    check(calls == [('flush', 'a')] and cpu._pending_finished_stores
          and cpu._pending_finished_stores[-1].gpu_block_ids == [b.block_id for b in tail]
          and all(b.ref_cnt == 2 for b in tail), 'end blocks appended to the finished stores and pinned')
    # Store completion as the connector runs it: copy every hash of each GPU block onto its CPU block.
    transfer = cpu._pending_finished_stores.pop()
    for gid_block, cid in zip(transfer.gpu_block_ids, transfer.cpu_block_ids):
        gb, cb = gpu.block_pool.blocks[gid_block], cpu_side.block_pool.blocks[cid]
        cpu_side.block_pool._insert_block_hash(gb.block_hash, cb, num_tokens=gb.block_hash_num_tokens)
        for h in gpu.block_pool.cached_block_hashes_by_block.get(gid_block, ()):
            cpu_side.block_pool._insert_block_hash(h, cb, num_tokens=None)
    cpu_side.block_pool.free_blocks([cpu_side.block_pool.blocks[c] for c in transfer.cpu_block_ids])
    gpu.block_pool.free_blocks([gpu.block_pool.blocks[g] for g in transfer.gpu_block_ids])
    # Full attention blocks reach the CPU tier through eager stores.
    for i, blk in enumerate(blocks_of(gpu, 'a')[0][:(end - 1) // BS]):
        cb = cpu_side.block_pool.get_new_blocks(1)[0]
        cpu_side.block_pool._insert_block_hash(blk.block_hash, cb, num_tokens=(i + 1) * BS)
        cpu_side.block_pool.free_blocks([cb])
    gpu.free(a)

    # Turn 2 (GPU tier): the re-rendered answer + tool output; the hit lands right after EOS.
    suffix = [1010, 10, 3263, 1010, 16, 1010] + list(range(5000, 5100))
    b = Request('b', list(a._all_token_ids) + suffix, params, None, block_hasher=hasher)
    found, n, boundary = gpu.get_computed_blocks(b)
    check(n == end and plugin._resumed.get('b') == key, f'GPU lookup resumes at the end state ({n} == {end})')
    gpu.allocate_slots(b, b.num_tokens - n, num_new_computed_tokens=n, new_computed_blocks=found)
    cows = {(s.block_id, d.block_id) for m in gpu.coordinator.single_type_managers for s, d in m._pending_cow_copies}
    check({s for s, _ in cows} == {blk.block_id for blk in tail},
          'the hit copies the attention tail and both Mamba states before writing (copy-on-write)')

    # A diverged re-render (answer token changed) does not hit the end state.
    c_tokens = list(a._all_token_ids)
    c_tokens[len(prompt1) + 5] += 1
    c = Request('c', c_tokens + suffix, params, None, block_hasher=hasher)
    _, n_c, _ = gpu.get_computed_blocks(c)
    check(n_c < len(prompt1) and 'c' not in plugin._resumed, 'a re-render that diverges inside the answer misses')

    # Turn 2 (CPU tier): GPU copies evicted; the CPU lookup returns the end state and its full attention blocks.
    for blk in tail:
        gpu.block_pool._maybe_evict_cached_block(blk)
    d = Request('d', list(a._all_token_ids) + suffix, params, None, block_hasher=hasher)
    _, n_d, _ = gpu.get_computed_blocks(d)
    hit, is_async = cpu.get_num_new_matched_tokens(d, 0)
    check(n_d == 0 and hit == end and is_async, f'CPU lookup returns the end state ({hit} == {end})')
    pending = cpu._pending_cpu_hits['d'][0]
    check([len(g) for g in pending] == [(end - 1) // BS + 1] * 3 and pending[1][-1].block_hash is not None,
          'CPU hit holds the full attention blocks + tail and the Mamba state at the tail index')
    # The scheduler allocates the request's GPU blocks for the external tokens, then the connector plans the load.
    dg = gpu.allocate_slots(d, 0, num_external_computed_tokens=hit, delay_cache_blocks=True)
    check(dg is not None, 'GPU blocks allocated for the unaligned external hit')
    gpu_ids = [[blk.block_id for blk in g] for g in gpu.coordinator.get_blocks('d')]
    cpu.update_state_after_alloc(d, SimpleNamespace(get_block_ids=lambda: gpu_ids), hit)
    load = cpu._reqs_to_load['d'].transfer_meta
    want_cpu = [blk.block_id for g in pending for blk in g if not blk.is_null]
    check(load.cpu_block_ids == want_cpu and len(load.gpu_block_ids) == len(want_cpu),
          'the load copies every non-null CPU block (the alignment assert was bypassed for this hit)')
    mamba_targets = load.gpu_block_ids[-2:]
    check(all(gpu_ids[g][(end - 1) // BS] == mamba_targets[i] for i, g in enumerate((1, 2))),
          'Mamba states load into the request\'s state block at the end position')

    # The load completes: the connector releases its CPU block refs.
    cpu_side.block_pool.free_blocks([cpu_side.block_pool.blocks[i] for i in load.cpu_block_ids])

    # Supersede: turn 2 finished after resuming from turn 1's end state -> turn 1's end + boundary states demoted.
    boundary_hash = plugin._ends[key]
    b.status = RequestStatus.FINISHED_STOPPED
    pool = cpu_side.block_pool
    end_cpu = [pool.cached_block_hash_to_block.get_one_block(make_block_hash_with_group_id(key, g)) for g in (1, 2)]
    sched_cpu = SimpleNamespace(kv_cache_manager=gpu, connector=SimpleNamespace(scheduler_manager=cpu))
    plugin._supersede(sched_cpu, b)
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    check(key not in plugin._ends and head.block_id in {blk.block_id for blk in end_cpu if blk is not None},
          'turn 1 end state demoted to the front of the CPU free queue')
    check(boundary_hash is None or boundary_hash not in plugin._boundary_end, 'turn 1 boundary state untracked')

    # v4: a turn that resumed from an end state demotes its own boundary state as soon as its end state registers.
    f = run_turn('f', list(range(80000, 80230)), [3000 + i for i in range(19)] + [EOS])
    f_boundary = plugin._boundary_hash(gpu.coordinator.single_type_managers[1], f)
    staged = cpu_side.block_pool.get_new_blocks(2)
    for g, blk in zip((1, 2), staged):
        cpu_side.block_pool._insert_block_hash(make_block_hash_with_group_id(f_boundary, g), blk, num_tokens=208)
    cpu_side.block_pool.free_blocks(staged)
    plugin._resumed['f'] = b'previous end state'
    before = plugin.stats()['early_boundary']
    plugin._register(sched_cpu, f, bonus=True)
    head = pool.free_block_queue.fake_free_list_head.next_free_block
    check(plugin.stats()['early_boundary'] == before + 1 and head.block_id in {b.block_id for b in staged},
          'a later turn demotes its own boundary state when its end state registers')
    g_req = run_turn('g', list(range(90000, 90230)), [3000 + i for i in range(19)] + [EOS])
    plugin._register(sched_cpu, g_req, bonus=True)
    check(plugin.stats()['early_boundary'] == before + 1, 'a first turn (no end-state resume) keeps its boundary state')
    print('all checks passed; stats', plugin.stats())


if __name__ == '__main__':
    main()
