"""CPU checks for synth-evict-dead against the real vLLM classes of venvs/vllm032.

    <vllm032>/bin/python check_evict_dead.py

Each scenario runs in a fresh interpreter (cwd outside this directory). The plugin is importable only when this
directory is on PYTHONPATH, exactly as on the serving node, and is loaded the way EngineCore does it
(vllm.plugins.load_general_plugins). Exits non-zero on the first failed check.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.abspath(__file__)
BS = 16


# ----------------------------------------------------------------------------- scenarios (child process)

def _load_plugins():
    from vllm.plugins import load_general_plugins
    load_general_plugins()


def _manager(num_blocks=20):
    import torch
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    init_none_hash(sha256)
    spec = FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float16)
    config = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[],
                           kv_cache_groups=[KVCacheGroupSpec(['layers.0.attn'], spec)])
    manager = KVCacheManager(config, max_model_len=4096, scheduler_block_size=BS, hash_block_size=BS,
                             enable_caching=True)
    return manager, get_request_block_hasher(BS, sha256)


def _request(rid, tokens, hasher):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    return Request(rid, list(tokens), SamplingParams(max_tokens=10000), None, block_hasher=hasher)


def _prefill(manager, request):
    blocks, hit, _ = manager.get_computed_blocks(request)
    out = manager.allocate_slots(request, request.num_tokens - hit, num_new_computed_tokens=hit,
                                 new_computed_blocks=blocks)
    assert out is not None, 'allocation failed'
    request.num_computed_tokens = request.num_tokens
    return hit


def _decode(manager, request, tokens):
    for token in tokens:
        request.append_output_token_ids(token)
        assert manager.allocate_slots(request, 1) is not None, 'decode allocation failed'
        request.num_computed_tokens = request.num_tokens


def _finish(manager, request, status='FINISHED_STOPPED'):
    from vllm.v1.request import RequestStatus
    request.status = getattr(RequestStatus, status)
    manager.free(request)


def _queue(pool):
    queue, ids = pool.free_block_queue, []
    block = queue.fake_free_list_head.next_free_block
    while block is not queue.fake_free_list_tail:
        ids.append(block.block_id)
        block = block.next_free_block
    return ids


def _table(manager, rid):
    return [b.block_id for b in manager.coordinator.single_type_managers[0].req_to_blocks[rid]]


def scenario_agents():
    """Two agents finish a 40-token-prompt / 40-token-completion turn, a third request churns 14 of the 19 blocks,
    then agent A sends its next turn (its prompt + a re-rendered completion that differs + new tokens)."""
    if os.environ.get('LOAD_PLUGINS') == '1':
        _load_plugins()
    manager, hasher = _manager()
    pool, trace = manager.block_pool, {}
    a1 = _request('A1', range(1000, 1040), hasher)
    _prefill(manager, a1)
    _decode(manager, a1, range(2000, 2040))
    trace['a1_table'] = _table(manager, 'A1')
    _finish(manager, a1)
    trace['queue_after_a1'] = _queue(pool)
    b1 = _request('B1', range(3000, 3040), hasher)
    _prefill(manager, b1)
    _decode(manager, b1, range(4000, 4040))
    trace['b1_table'] = _table(manager, 'B1')
    _finish(manager, b1)
    trace['queue_after_b1'] = _queue(pool)
    churn = _request('C', range(5000, 5000 + 14 * BS), hasher)
    trace['churn_hit'] = _prefill(manager, churn)
    trace['queue_after_churn'] = _queue(pool)
    a2 = _request('A2', [*range(1000, 1040), *range(6000, 6030), *range(7000, 7010)], hasher)
    blocks, hit, _ = manager.get_computed_blocks(a2)
    trace['a2_hit'] = hit
    trace['a2_hit_blocks'] = [b.block_id for b in blocks.blocks[0]]
    try:
        import synth_evict_dead
        trace['stats'] = synth_evict_dead.stats()
    except ImportError:
        trace['stats'] = None
    from vllm.v1.core.block_pool import BlockPool
    trace['free_blocks_patched'] = hasattr(BlockPool.free_blocks, '__wrapped__')
    return trace


def scenario_length_capped():
    """A request that stopped at max_tokens keeps stock LRU (a client may continue its answer verbatim)."""
    _load_plugins()
    import synth_evict_dead
    manager, hasher = _manager()
    d = _request('D', range(1000, 1040), hasher)
    _prefill(manager, d)
    _decode(manager, d, range(2000, 2040))
    table = _table(manager, 'D')
    _finish(manager, d, 'FINISHED_LENGTH_CAPPED')
    return {'table': table, 'queue': _queue(manager.block_pool), 'stats': synth_evict_dead.stats()}


def scenario_mamba():
    """Class B on a real BlockPool + MambaManager (align mode): a request hits state S at 32 tokens, hashes its
    own state T at 64, releases S mid-request, then finishes. Variants check every keep condition."""
    _load_plugins()
    import torch
    import synth_evict_dead as plugin
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager
    from vllm.v1.kv_cache_interface import MambaSpec

    class Finished:
        def __init__(self, rid, prompt, junction=0, finished=True):
            self.request_id, self.num_prompt_tokens = rid, prompt
            self.shared_prefix_boundary, self._finished = junction, finished

        def is_finished(self):
            return self._finished

    def setup():
        pool = BlockPool(num_gpu_blocks=12, enable_caching=True, hash_block_size=BS)
        spec = MambaSpec(block_size=BS, shapes=((4,),), dtypes=(torch.float16,), mamba_cache_mode='align')
        manager = MambaManager(spec, pool, enable_caching=True, kv_cache_group_id=1, scheduler_block_size=BS)
        return pool, manager

    def hashed(pool, tag, tokens):
        block = pool.get_new_blocks(1)[0]
        pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(tag * 32), 1), block, num_tokens=tokens)
        return block

    def hit(manager, pool, rid, state):
        manager.add_local_computed_blocks(rid, [pool.null_block, state], 2 * BS, 0)

    def finish(manager, request):
        plugin._ctx = request
        try:
            manager.free(request.request_id)
        finally:
            plugin._ctx = None

    def run(variant):
        pool, manager = setup()
        s = hashed(pool, b'S', 2 * BS)
        pool.free_blocks([s])  # cached and free, LRU tail
        hit(manager, pool, 'R', s)
        if variant == 'shared':
            hit(manager, pool, 'R2', s)
        if variant != 'nothing_later':
            t = hashed(pool, b'T', 4 * BS)
            manager.req_to_blocks['R'].append(t)
            plugin._note_created(manager, 'R')
        if variant != 'held':
            manager.req_to_blocks['R'][1] = pool.null_block  # retire S mid-request (align mode)
            pool.free_blocks([s])
        if variant == 'shared':
            finish(manager, Finished('R2', 2 * BS))
        junction = 2 * BS if variant == 'junction' else 0
        finish(manager, Finished('R', 70, junction=junction, finished=variant != 'preempted'))
        queue = _queue(pool)
        return {'s': s.block_id, 'front': queue[0], 's_rank': queue.index(s.block_id), 'len': len(queue)}

    return {v: run(v) for v in ('superseded', 'held', 'shared', 'nothing_later', 'junction', 'preempted')} | {
        'stats': plugin.stats()}


def scenario_entry_point():
    from importlib.metadata import entry_points
    from vllm.logger import _configure_vllm_root_logger
    names = [e.name for e in entry_points(group='vllm.general_plugins')]
    _configure_vllm_root_logger()  # vLLM processes configure logging first (run_engine_core), then load plugins
    _load_plugins()
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import MambaManager, SingleTypeKVCacheManager
    import inspect
    from vllm.v1.engine.core import EngineCore
    init = inspect.getsource(EngineCore.__init__)
    return {'entry_points': names,
            'patched': {name: hasattr(fn, '__wrapped__') for name, fn in [
                ('BlockPool.free_blocks', BlockPool.free_blocks), ('KVCacheManager.free', KVCacheManager.free),
                ('KVCacheManager.pop_blocks_for_free', KVCacheManager.pop_blocks_for_free),
                ('SingleTypeKVCacheManager.pop_blocks_for_free', SingleTypeKVCacheManager.pop_blocks_for_free),
                ('SingleTypeKVCacheManager.add_local_computed_blocks',
                 SingleTypeKVCacheManager.add_local_computed_blocks),
                ('MambaManager.cache_blocks', MambaManager.cache_blocks)]},
            'engine_core_loads_plugins_first': init.find('load_general_plugins()') != -1 and (
                init.find('load_general_plugins()') < init.find('self.scheduler')
                if 'self.scheduler' in init else True)}


SCENARIOS = {'agents': scenario_agents, 'length_capped': scenario_length_capped, 'mamba': scenario_mamba,
             'entry_point': scenario_entry_point}


# ----------------------------------------------------------------------------- driver (parent process)

def child(name, plugin_on_path, enabled, load_plugins=True):
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'SYNTH_EVICT_DEAD', 'VLLM_PLUGINS')}
    if plugin_on_path:
        env['PYTHONPATH'] = HERE
    if enabled:
        env['SYNTH_EVICT_DEAD'] = '1'
    env['LOAD_PLUGINS'] = '1' if load_plugins else '0'
    code = ('import importlib.util, json, sys\n'
            f'spec = importlib.util.spec_from_file_location("check_evict_dead", {SCRIPT!r})\n'
            'm = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n'
            f'print("RESULT " + json.dumps(m.SCENARIOS[{name!r}]()))\n')
    with tempfile.TemporaryDirectory() as cwd:
        p = subprocess.run([sys.executable, '-c', code], cwd=cwd, env=env, capture_output=True, text=True)
    lines = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
    if p.returncode or not lines:
        print(p.stdout[-4000:], p.stderr[-8000:], sep='\n')
        raise SystemExit(f'scenario {name} (path={plugin_on_path}, enabled={enabled}) failed')
    return json.loads(lines[-1][len('RESULT '):]), p.stdout + p.stderr


def check(cond, label):
    print(('PASS ' if cond else 'FAIL ') + label, flush=True)
    if not cond:
        raise SystemExit(1)


def main():
    import vllm
    print('vllm', vllm.__version__, flush=True)

    # (d) registration through the entry point, from PYTHONPATH, in a fresh interpreter
    on, log_on = child('entry_point', True, True)
    off, _ = child('entry_point', True, False)
    check('synth_evict_dead' in on['entry_points'], '(d) vllm.general_plugins entry point discovered via PYTHONPATH')
    check(all(on['patched'].values()), f'(d) enabled: every hook installed {on["patched"]}')
    check('synth-evict-dead v1 active' in log_on, '(d) enabled: load line in the vLLM log')
    check(not any(off['patched'].values()), '(d) SYNTH_EVICT_DEAD unset: nothing patched')
    check(on['engine_core_loads_plugins_first'], '(d) EngineCore.__init__ loads general plugins before the scheduler')

    # (c) stock equivalence: no plugin on the path vs plugin on the path but disabled
    stock, _ = child('agents', False, False, load_plugins=True)
    disabled, _ = child('agents', True, False, load_plugins=True)
    stock.pop('stats'), disabled.pop('stats')
    check(stock == disabled, '(c) SYNTH_EVICT_DEAD unset: block tables, free-queue order and hits identical to stock')
    check(not disabled['free_blocks_patched'], '(c) SYNTH_EVICT_DEAD unset: BlockPool.free_blocks is the stock function')

    # (a) + (b) class A on a real KVCacheManager
    patched, log = child('agents', True, True)
    a1, b1 = patched['a1_table'], stock['a1_table']
    check(a1 == b1, '(a) same block tables with and without the plugin')
    # A1: prompt 40 tokens -> blocks 0,1 hold prompt only; block 2 (tokens 32-47) was filled by decode and promoted to
    # a full hash over completion tokens; blocks 3,4 are completion. Freed in reverse order.
    junk, keep = a1[2:], a1[:2]
    q = patched['queue_after_a1']
    check(q[:3] == list(reversed(junk)), f'(a) completion-covering blocks {junk} go to the queue front: {q[:3]}')
    check(q[-2:] == list(reversed(keep)), f'(a) prompt blocks {keep} keep LRU order at the tail: {q[-2:]}')
    sq = stock['queue_after_a1']
    check(sq[-5:] == list(reversed(a1)), f'(a) stock appends all five hashed blocks at the tail: {sq[-5:]}')
    check(stock['a2_hit'] == 0, f'(b) stock: churn evicted agent A\'s prompt, next-turn hit {stock["a2_hit"]} tokens')
    check(patched['a2_hit'] == 2 * BS and patched['a2_hit_blocks'] == keep,
          f'(b) plugin: agent A\'s next turn still hits its prompt prefix ({patched["a2_hit"]} tokens, blocks '
          f'{patched["a2_hit_blocks"]}) while junk was evicted first')
    s = patched['stats']
    check(s['demoted_attn'] == 6 and s['hashed_kept'] == 4,
          f'(b) counters: 3 junk blocks per agent demoted, 2 prompt blocks per agent kept: {s}')

    capped, _ = child('length_capped', True, True)
    check(capped['queue'][-5:] == list(reversed(capped['table'])) and capped['stats']['demoted_attn'] == 0,
          f'(A) length-capped request keeps stock LRU: tail {capped["queue"][-5:]}, stats {capped["stats"]}')

    # class B (Mamba superseded hit states)
    m, _ = child('mamba', True, True)
    check(m['superseded']['front'] == m['superseded']['s'],
          f'(B) superseded hit state released mid-request is moved to the front: {m["superseded"]}')
    check(m['held']['front'] == m['held']['s'], f'(B) superseded hit state still held at finish goes to the front: '
                                                f'{m["held"]}')
    for variant in ('shared', 'nothing_later', 'junction', 'preempted'):
        check(m[variant]['s_rank'] != 0, f'(B) {variant}: hit state keeps its LRU place {m[variant]}')
    check(m['stats']['demoted_mamba_late'] == 1 and m['stats']['demoted_mamba'] == 1,
          f'(B) counters {m["stats"]}')
    print('ALL CHECKS PASSED', flush=True)


if __name__ == '__main__':
    main()
