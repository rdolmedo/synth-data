"""CPU checks for synth-cpu-supersede's vLLM 0.34 path (v3) against the real vLLM 0.34 classes.

    SYNTH_CPU_SUPERSEDE=1 PYTHONPATH=<deployed dir> <python with vLLM 0.34> check_cpu_supersede_v034.py

No GPU is needed: the vLLM 0.34 Python sources import on CPU (a python-only tree works ahead of a torch venv on
PYTHONPATH, e.g. `PYTHONPATH=<dir holding the 0.34 vllm package>:<deployed dir> venvs/vllm033/bin/python -B`). The
plugin is loaded the way EngineCore loads it (vllm.plugins.load_general_plugins -> `_entry.register`). The CPU and
GPU pools are real 0.34 BlockPools and the KV cache groups real 0.34 KVCacheGroups; the scheduler around them is a
stand-in with the attributes the plugin reads, on which the real SimpleCPUOffloadScheduler methods' wrappers are
applied. Same scenarios as check_cpu_supersede.py (vLLM 0.32), plus 0.34's duplicate-key table (`_dup_ids`) and a
record of why v2 itself cannot run on 0.34. Exits non-zero on the first failed check.
"""
import inspect
import os
import sys

import torch

# Test the copy PYTHONPATH names (a deployed version), not the source tree next to this script.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or '.') != os.path.dirname(os.path.abspath(__file__))]

HB = 16          # hash block (prefix-match unit)
BS = 64          # scheduler / Mamba block, a multiple of HB (thousands of tokens on the engine)
MAMBA_GROUPS = (1, 2)


def check(cond, message):
    if not cond:
        print('FAIL:', message)
        sys.exit(1)
    print('ok:', message)


def main():
    assert os.environ.get('SYNTH_CPU_SUPERSEDE') == '1', 'run with SYNTH_CPU_SUPERSEDE=1'
    import vllm
    from vllm.plugins import load_general_plugins
    from vllm.utils.hashing import sha256
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash, make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, MambaSpec
    from vllm.v1.request import RequestStatus
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler

    print('vLLM', vllm.__version__, 'from', os.path.dirname(vllm.__file__))
    check(tuple(int(x) for x in vllm.__version__.split('.')[:2]) >= (0, 34), 'vLLM is 0.34 or later')
    real = {name: inspect.signature(getattr(SimpleCPUOffloadScheduler, name)) for name in
            ('update_state_after_alloc', '_prepare_eager_store_specs', 'request_finished_all_groups')}
    load_general_plugins()
    import synth_cpu_supersede.vllm034 as plugin
    print('checking', plugin.__file__, plugin.VERSION)
    check(plugin._patched and plugin.VERSION == 'v3', 'the entry point selected the 0.34 path (v3)')
    for name, sig in real.items():
        method = getattr(SimpleCPUOffloadScheduler, name)
        check(hasattr(method, '__wrapped__') and inspect.signature(method.__wrapped__) == sig,
              f'{name} wrapped with the installed signature {sig}')

    init_none_hash(sha256)
    attn = AttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float16)
    mamba = MambaSpec(block_size=BS, shapes=((4,),), dtypes=(torch.float16,), page_size_padded=attn.page_size_bytes)
    align = MambaManager.config(mamba_cache_mode='align', num_speculative_blocks=3)

    class Config:
        kv_cache_groups = [KVCacheGroup({'a': attn}, FullAttentionManager.config(), BS)] + [
            KVCacheGroup({f'm{g}': mamba}, align, BS) for g in MAMBA_GROUPS]

    calls = []

    class Stand:
        """The attributes of SimpleCPUOffloadScheduler the plugin reads; originals record their calls."""
        def __init__(self):
            self.hash_block_size = HB
            self.cpu_kv_cache_config = Config()
            self.prefix_cacheable_group_ids = (0, *MAMBA_GROUPS)
            self.cpu_block_pool = BlockPool(64, True, HB, num_hash_lookahead_tokens=1)
            self._gpu_block_pool = BlockPool(64, True, HB, num_hash_lookahead_tokens=1)

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
    check(plugin._mamba_groups(m) == list(MAMBA_GROUPS), 'Mamba groups found through is_mamba_group (KVCacheGroup)')

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

    def cache_state(pool, request, tokens, copies=1):
        """A boundary state block per Mamba group (`copies` blocks per key), cached under the prefix hash at
        `tokens` and free (LRU tail)."""
        blocks = pool.get_new_blocks(len(MAMBA_GROUPS) * copies)
        for i, block in enumerate(blocks):
            g = MAMBA_GROUPS[i % len(MAMBA_GROUPS)]
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
    cache_state(m.cpu_block_pool, Req('F', agent(9, 400), 0), 320)
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

    # Aborted and preempted requests keep stock behaviour; a hit one hash unit off (MTP block drop) still matches.
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

    # 0.34 table layout: a key cached by several blocks keeps one id in _cache and all of them in _dup_ids.
    e1 = Req('E1', agent(5, 100), hit=0)
    turn(e1, 160)
    se = cache_state(m.cpu_block_pool, e1, 160, copies=2)
    key = make_block_hash_with_group_id(e1.block_hashes[160 // HB - 1], MAMBA_GROUPS[0])
    table = m.cpu_block_pool.cached_block_hash_to_block
    check(isinstance(table._cache[key], int) and len(table._dup_ids[key]) == 2,
          'BlockHashToBlockMap holds block ids (_cache) and duplicate ids (_dup_ids)')
    check(sorted(b.block_id for b in plugin.cached_blocks(m.cpu_block_pool, key)) == sorted(se[0::2])
          and plugin.cached_blocks(m.cpu_block_pool, b'missing') == [], 'cached_blocks reads both tables')
    turn(Req('E2', agent(5, 340), hit=160), 320)
    check(sorted(front(m.cpu_block_pool, 4)) == sorted(se), 'every duplicate copy of a superseded state is demoted')

    # GPU copies follow with SYNTH_CPU_SUPERSEDE_GPU=1.
    os.environ['SYNTH_CPU_SUPERSEDE_GPU'] = '1'
    d1 = Req('D1', agent(4, 100), hit=0)
    turn(d1, 160)
    sd_cpu, sd_gpu = cache_state(m.cpu_block_pool, d1, 160), cache_state(m._gpu_block_pool, d1, 160)
    cache_state(m._gpu_block_pool, Req('G', agent(8, 400), 0), 320)
    turn(Req('D2', agent(4, 340), hit=160), 320)
    check(front(m.cpu_block_pool, 2) == sd_cpu and front(m._gpu_block_pool, 2) == sd_gpu,
          'GPU copies are demoted too when enabled')
    os.environ.pop('SYNTH_CPU_SUPERSEDE_GPU')

    # Why v2 cannot run on 0.34 unchanged: its helpers raise on the 0.34 structures (each superseded turn would be
    # skipped with a warning, so nothing is demoted).
    import synth_cpu_supersede as v2
    v = Req('V', agent(7, 100), hit=0)
    cache_state(m.cpu_block_pool, v, 160)
    try:
        v2._mamba_groups(Stand())
        v2_groups = 'no error'
    except AttributeError as exc:
        v2_groups = repr(exc)
    try:
        v2._demote(m.cpu_block_pool, [make_block_hash_with_group_id(v.block_hashes[160 // HB - 1], MAMBA_GROUPS[0])])
        v2_demote = 'no error'
    except AttributeError as exc:
        v2_demote = repr(exc)
    check('kv_cache_spec' in v2_groups and "'int' object" in v2_demote,
          f'v2 on 0.34 raises ({v2_groups}; {v2_demote})')
    print('stats', plugin.stats())


if __name__ == '__main__':
    main()
