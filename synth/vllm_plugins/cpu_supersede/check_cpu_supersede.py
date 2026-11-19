"""CPU checks for synth-cpu-supersede against the real vLLM classes of venvs/vllm032.

    SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<this dir> <vllm032>/bin/python check_cpu_supersede.py

The plugin is loaded the way EngineCore loads it (vllm.plugins.load_general_plugins). The CPU and GPU pools are real
BlockPools; the scheduler around them is a stand-in with the attributes the plugin reads, on which the real
SimpleCPUOffloadScheduler methods' wrappers are applied. Exits non-zero on the first failed check.
"""
import inspect
import os
import sys

import torch

# Test the copy PYTHONPATH names (a deployed version), not the source tree next to this script.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]

HB = 16          # hash block (prefix-match unit)
BS = 64          # scheduler / Mamba block, a multiple of HB as on the engine (4,336 there)
MAMBA_GROUPS = (1, 2)


def check(cond, message):
    if not cond:
        print('FAIL:', message)
        sys.exit(1)
    print('ok:', message)


def main():
    assert os.environ.get('SYNTH_CPU_SUPERSEDE') == '1', 'run with SYNTH_CPU_SUPERSEDE=1'
    from vllm.plugins import load_general_plugins
    from vllm.utils.hashing import sha256
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash, make_block_hash_with_group_id
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.request import RequestStatus
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    real = {name: inspect.signature(getattr(SimpleCPUOffloadScheduler, name)) for name in
            ('update_state_after_alloc', '_prepare_eager_store_specs', 'request_finished_all_groups')}
    load_general_plugins()
    import synth_cpu_supersede as plugin
    print('checking', plugin.__file__, plugin.VERSION)
    for name, sig in real.items():
        method = getattr(SimpleCPUOffloadScheduler, name)
        check(hasattr(method, '__wrapped__') and inspect.signature(method.__wrapped__) == sig,
              f'{name} wrapped with the installed signature {sig}')

    init_none_hash(sha256)

    class Group:
        def __init__(self, spec):
            self.kv_cache_spec = spec

    class Config:
        kv_cache_groups = [Group(FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float16))] + [
            Group(MambaSpec(shapes=((4,),), dtypes=(torch.float16,), block_size=BS)) for _ in MAMBA_GROUPS]

    calls = []

    class Stand:
        """The attributes of SimpleCPUOffloadScheduler the plugin reads; originals record their calls."""
        def __init__(self):
            self.hash_block_size = HB
            self.cpu_kv_cache_config = Config()
            self.prefix_cacheable_group_ids = [0, *MAMBA_GROUPS]
            self.cpu_block_pool = BlockPool(64, True, HB)
            self._gpu_block_pool = BlockPool(64, True, HB)

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            calls.append(('alloc', request.request_id))

        def _prepare_eager_store_specs(self, scheduler_output):
            calls.append(('store',))
            return [], [], [], None

        def request_finished_all_groups(self, request, block_ids):
            calls.append(('finished', request.request_id))
            return False, None

    Stand.update_state_after_alloc = plugin._record_hit(Stand.update_state_after_alloc)
    Stand._prepare_eager_store_specs = plugin._record_offers(Stand._prepare_eager_store_specs)
    Stand.request_finished_all_groups = plugin._on_finished(Stand.request_finished_all_groups)
    m = Stand()

    def hashes(tokens):
        out, parent = [], None
        for i in range(0, len(tokens) - len(tokens) % HB, HB):
            parent = hash_block_tokens(sha256, parent, tokens[i:i + HB])
            out.append(parent)
        return out

    class Stats:
        def __init__(self, local, external):
            self.num_local_cached_tokens, self.num_external_cached_tokens = local, external

    class Req:
        def __init__(self, rid, tokens, hit, external=0):
            self.request_id, self.block_hashes = rid, hashes(tokens)
            self.prefill_stats, self.num_preemptions = Stats(hit - external, external), 0
            self.status = RequestStatus.RUNNING

    class Offers:
        def __init__(self, offers):
            self.boundary_state_offloads = offers

    class Out:
        def __init__(self, offers):
            self.kv_connector_block_state = Offers(offers)

    def cache_state(pool, request, tokens):
        """A boundary state block per Mamba group, cached under the prefix hash at `tokens` and free (LRU tail)."""
        blocks = pool.get_new_blocks(len(MAMBA_GROUPS))
        for g, block in zip(MAMBA_GROUPS, blocks):
            pool._insert_block_hash(make_block_hash_with_group_id(request.block_hashes[tokens // HB - 1], g), block,
                                    num_tokens=tokens)
        pool.free_blocks(blocks)
        return [b.block_id for b in blocks]

    def front(pool, n):
        ids, block = [], pool.free_block_queue.fake_free_list_head.next_free_block
        while block is not pool.free_block_queue.fake_free_list_tail and len(ids) < n:
            ids.append(block.block_id)
            block = block.next_free_block
        return ids

    def turn(request, offered, status=RequestStatus.FINISHED_STOPPED):
        m.update_state_after_alloc(request, None, request.prefill_stats.num_external_cached_tokens)
        if request.prefill_stats.num_external_cached_tokens:
            m.update_state_after_alloc(request, None, 0)  # rescheduled after its async CPU load (counted once)
        m._prepare_eager_store_specs(Out({request.request_id: [(g, 0, offered) for g in MAMBA_GROUPS]}))
        request.status = status
        m.request_finished_all_groups(request, ([],) * 3)

    system = list(range(10_000, 10_000 + 5 * HB))
    agent = lambda k, n: system + list(range(100_000 * k, 100_000 * k + n))  # agent k's history, n tokens long

    # Turn 1 of agent A: resumes at the shared system prompt (never produced), produces state S1 at 160.
    a1 = Req('A1', agent(1, 100), hit=5 * HB)
    turn(a1, 160)
    s1 = cache_state(m.cpu_block_pool, a1, 160)
    filler = cache_state(m.cpu_block_pool, Req('F', agent(9, 400), 0), 320)
    check(front(m.cpu_block_pool, 2) != s1, 'S1 sits behind older free blocks (stock LRU)')
    check(plugin.stats()['not_produced'] == 1 and plugin.stats()['demoted_cpu'] == 0,
          'a turn resuming from the shared system prompt demotes nothing')
    # Turn 2 resumes from S1 (CPU hit) and produces S2 at 320: S1 is superseded and leaves the CPU tier first.
    a2 = Req('A2', agent(1, 340), hit=160, external=160)
    turn(a2, 320)
    check(front(m.cpu_block_pool, 2) == s1, 'S1 moved to the front of the CPU free queue')
    check(plugin.stats()['superseded'] == 1 and plugin.stats()['demoted_cpu'] == 2, 'one state, two Mamba blocks')
    check(m._gpu_block_pool.get_num_free_blocks() == 63, 'GPU pool untouched without SYNTH_CPU_SUPERSEDE_GPU')
    check(calls[-1] == ('finished', 'A2') and ('alloc', 'A2') in calls, 'the original methods still run')

    # A state hit by two requests (a task's first turn shared by two attempts) stays in LRU order.
    b1 = Req('B1', agent(2, 100), hit=0)
    turn(b1, 160)
    sb = cache_state(m.cpu_block_pool, b1, 160)
    other = Req('B1x', agent(2, 120), hit=160)  # a second attempt resumed from the same state
    m.update_state_after_alloc(other, None, 0)
    b2 = Req('B2', agent(2, 340), hit=160)
    turn(b2, 320)
    check(plugin.stats()['shared'] == 1 and front(m.cpu_block_pool, 2) != sb, 'a state two requests hit is kept')
    other.status = RequestStatus.FINISHED_STOPPED
    m.request_finished_all_groups(other, ([],) * 3)

    # Aborted and preempted requests keep stock behaviour; hit offset by the MTP block drop is still matched.
    c1 = Req('C1', agent(3, 100), hit=0)
    turn(c1, 160)
    sc = cache_state(m.cpu_block_pool, c1, 160)
    c2 = Req('C2', agent(3, 340), hit=160)
    turn(c2, 320, status=RequestStatus.FINISHED_ABORTED)
    check(front(m.cpu_block_pool, 2) != sc, 'an aborted turn demotes nothing')
    c3 = Req('C3', agent(3, 340), hit=144)  # one hash block short of the state
    turn(c3, 320)
    check(front(m.cpu_block_pool, 2) == sc, 'a hit one hash block off the state still supersedes it')
    check(not plugin._hit and not plugin._offered, 'per-request records are released at finish')

    # GPU copies follow with SYNTH_CPU_SUPERSEDE_GPU=1.
    os.environ['SYNTH_CPU_SUPERSEDE_GPU'] = '1'
    d1 = Req('D1', agent(4, 100), hit=0)
    turn(d1, 160)
    sd_cpu, sd_gpu = cache_state(m.cpu_block_pool, d1, 160), cache_state(m._gpu_block_pool, d1, 160)
    cache_state(m._gpu_block_pool, Req('G', agent(8, 400), 0), 320)
    turn(Req('D2', agent(4, 340), hit=160), 320)
    check(front(m.cpu_block_pool, 2) == sd_cpu and front(m._gpu_block_pool, 2) == sd_gpu,
          'GPU copies are demoted too when enabled')
    print('stats', plugin.stats())


if __name__ == '__main__':
    main()
