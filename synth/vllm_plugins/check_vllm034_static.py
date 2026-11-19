"""Static check (no vLLM import, no GPU) that the vLLM 0.34 paths of synth-cpu-supersede (v3) and
synth-eot-checkpoint (v5) only rely on names, attributes and signatures that exist in a vLLM source tree.

    python3 check_vllm034_static.py <path to the vllm package directory of the target build>

It parses the target's sources with `ast` and checks:
1. every `from vllm... import X` in the plugins' vllm034.py modules (module exists, X defined at its top level);
2. the curated REQUIRED table below: wrapped methods with their exact parameter names, called functions with the
   parameters the plugins pass, and the attributes they read (class fields, properties, `self.x = ...` in any
   method of the class or its vLLM base classes);
3. BEHAVIOUR: source facts the port depends on (e.g. block ids in BlockHashToBlockMap, lookahead block hashes, the
   block-aligned local prefix handed to the connector), and ABSENT: what v2/v4 used that 0.34 no longer has.
Exits non-zero if anything is missing. Re-run it against any later vLLM before deploying the plugins there.
"""
import ast
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
PLUGIN_MODULES = [os.path.join(HERE, 'cpu_supersede', 'synth_cpu_supersede', 'vllm034.py'),
                  os.path.join(HERE, 'eot_checkpoint', 'synth_eot_checkpoint', 'vllm034.py')]

M_POOL = 'vllm.v1.core.block_pool'
M_UTILS = 'vllm.v1.core.kv_cache_utils'
M_KVM = 'vllm.v1.core.kv_cache_manager'
M_COORD = 'vllm.v1.core.kv_cache_coordinator'
M_STM = 'vllm.v1.core.single_type_kv_cache_manager'
M_SCHED = 'vllm.v1.core.sched.scheduler'
M_IFACE = 'vllm.v1.kv_cache_interface'
M_REQ = 'vllm.v1.request'
M_OFF = 'vllm.v1.simple_kv_offload.manager'
M_RS = 'vllm.v1.worker.gpu.spec_decode.rejection_sampler'

# (module, qualname, kind, detail). kind: 'wrap' = exact parameter names (the plugin replaces the method and calls
# the original with these), 'call' = these parameters exist in this order, 'attrs' = attributes of the class.
REQUIRED = [
    # --- methods the plugins wrap (both plugins wrap update_state_after_alloc)
    (M_OFF, 'SimpleCPUOffloadScheduler.update_state_after_alloc', 'wrap',
     ['self', 'request', 'blocks', 'num_external_tokens']),
    (M_OFF, 'SimpleCPUOffloadScheduler._prepare_eager_store_specs', 'wrap', ['self', 'scheduler_output']),
    (M_OFF, 'SimpleCPUOffloadScheduler.request_finished_all_groups', 'wrap', ['self', 'request', 'block_ids']),
    (M_OFF, 'SimpleCPUOffloadScheduler._queue_finished_eager_store', 'wrap', ['self', 'request', 'block_ids']),
    (M_OFF, 'SimpleCPUOffloadScheduler.get_num_new_matched_tokens', 'wrap',
     ['self', 'request', 'num_computed_tokens']),
    (M_SCHED, 'Scheduler._update_request_with_output', 'wrap', ['self', 'request', 'new_token_ids', 'is_stale']),
    (M_SCHED, 'Scheduler._free_request', 'wrap', ['self', 'request', 'delay_free_blocks']),
    (M_KVM, 'KVCacheManager.get_computed_blocks', 'wrap', ['self', 'request', 'max_cache_hit_length']),
    (M_RS, 'RejectionSampler.__call__', 'wrap', ['self', 'logits', 'input_batch', 'draft_logits']),
    # --- functions and methods the plugins call
    (M_OFF, 'SimpleCPUOffloadScheduler._free_pending_cpu_hit', 'call', ['self', 'pending']),
    (M_OFF, 'TransferMeta', 'attrs', ['gpu_block_ids', 'cpu_block_ids', 'block_meta']),
    (M_UTILS, 'make_block_hash_with_group_id', 'call', ['block_hash', 'group_id']),
    (M_UTILS, 'is_mamba_group', 'call', ['group']),
    (M_IFACE, 'get_mamba_prefill_checkpoint_position', 'call',
     ['num_tokens', 'hash_block_size', 'drop_eagle_block', 'num_hash_lookahead_tokens']),
    (M_POOL, 'BlockPool._insert_block_hash', 'call', ['self', 'block_hash_with_group_id', 'block', 'num_tokens']),
    (M_POOL, 'BlockPool.touch', 'call', ['self', 'blocks']),
    (M_POOL, 'BlockPool.get_new_blocks', 'call', ['self', 'num_blocks']),
    (M_POOL, 'BlockPool.get_num_free_blocks', 'call', ['self']),
    (M_POOL, 'BlockHashToBlockMap.get_one_block', 'call', ['self', 'key']),
    (M_UTILS, 'FreeKVCacheBlockQueue.remove', 'call', ['self', 'block']),
    (M_UTILS, 'FreeKVCacheBlockQueue.prepend_n', 'call', ['self', 'blocks']),
    (M_KVM, 'KVCacheManager.prefix_cache_lookup_enabled', 'call', ['self', 'request']),
    (M_KVM, 'KVCacheManager.create_kv_cache_blocks', 'call', ['self', 'blocks']),
    (M_REQ, 'Request.get_max_cache_hit_length', 'call', ['self', 'bound']),
    ('vllm.logger', 'init_logger', 'call', ['name']),
    # --- attributes read
    (M_OFF, 'SimpleCPUOffloadScheduler', 'attrs',
     ['cpu_kv_cache_config', 'prefix_cacheable_group_ids', 'hash_block_size', 'block_size', 'cpu_block_pool',
      '_gpu_block_pool', 'cpu_coordinator', '_pending_cpu_hits', '_pending_finished_stores',
      '_in_flight_store_gpu_blocks', 'enable_kv_cache_events']),
    ('vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector', 'SimpleCPUOffloadConnector',
     'attrs', ['scheduler_manager']),
    (M_SCHED, 'Scheduler', 'attrs', ['kv_cache_manager', 'connector', 'cache_config']),
    ('vllm.config.cache', 'CacheConfig', 'attrs', ['use_replayssm']),
    (M_KVM, 'KVCacheManager', 'attrs', ['coordinator', 'block_pool', 'kv_cache_config']),
    (M_COORD, 'KVCacheCoordinator', 'attrs', ['single_type_managers', 'kv_cache_config', 'block_pool']),
    (M_IFACE, 'KVCacheConfig', 'attrs', ['kv_cache_groups']),
    (M_IFACE, 'KVCacheGroup', 'attrs', ['prefix_cacheable', 'manager_cls']),
    (M_STM, 'FullAttentionManager', 'attrs', ['req_to_blocks', 'block_size', 'block_pool']),
    (M_STM, 'MambaManager', 'attrs',
     ['req_to_blocks', 'block_size', 'block_pool', 'mamba_cache_mode', 'drop_eagle_checkpoint_block']),
    (M_POOL, 'BlockPool', 'attrs',
     ['blocks', 'free_block_queue', 'cached_block_hash_to_block', 'null_block', 'hash_block_size',
      'num_hash_lookahead_tokens']),
    (M_POOL, 'BlockHashToBlockMap', 'attrs', ['_cache', '_dup_ids']),
    (M_UTILS, 'KVCacheBlock', 'attrs', ['block_id', 'ref_cnt', 'block_hash', 'is_null']),
    (M_UTILS, 'FreeKVCacheBlockQueue', 'attrs', ['fake_free_list_head', 'fake_free_list_tail']),
    (M_REQ, 'Request', 'attrs',
     ['request_id', 'status', 'block_hashes', '_all_token_ids', 'num_tokens', 'num_prompt_tokens', 'sampling_params',
      'num_in_flight_tokens', 'num_preemptions', 'prefill_stats', 'skip_reading_external_kv']),
    (M_REQ, 'RequestStatus', 'attrs', ['FINISHED_STOPPED', 'FINISHED_LENGTH_CAPPED']),
    ('vllm.sampling_params', 'SamplingParams', 'attrs', ['all_stop_token_ids', 'extra_args']),
    ('vllm.v1.metrics.stats', 'PrefillStats', 'attrs', ['num_local_cached_tokens', 'num_external_cached_tokens']),
    ('vllm.v1.core.sched.output', 'SchedulerOutput', 'attrs', ['kv_connector_block_state']),
    ('vllm.v1.core.sched.output', 'KVConnectorBlockState', 'attrs', ['boundary_state_offloads']),
    ('vllm.v1.worker.gpu.sample.output', 'SamplerOutput', 'attrs', ['sampled_token_ids', 'num_sampled',
                                                                    'num_rejected']),
    (M_UTILS, 'NONE_HASH', 'name', None),
    (M_UTILS, 'BlockHash', 'name', None),
    ('vllm.plugins', 'DEFAULT_PLUGINS_GROUP', 'name', None),
    ('vllm.plugins', 'load_general_plugins', 'call', []),
]

# (description, module, qualname or None for the whole file, regex that must match the source)
BEHAVIOUR = [
    ('BlockHashToBlockMap keeps block ids, one per key in _cache and every duplicate in _dup_ids', M_POOL,
     'BlockHashToBlockMap.__init__', r'self\._cache: dict\[BlockHashWithGroupId, int\][\s\S]*self\._dup_ids'),
    ('lookahead hashes: a unit hash exists once its lookahead tokens do (end_key picks its parent unit by this)',
     M_UTILS, 'get_request_block_hasher', r'num_tokens = request\.num_tokens - num_hash_lookahead_tokens'),
    ('lookahead hashes clear the EAGLE/MTP trailing-block drop', M_SCHED, 'Scheduler.__init__',
     r'if self\.num_hash_lookahead_tokens:\s*(#.*\s*)*self\.use_eagle_block_drop = False'),
    ('the connector gets the block-aligned local prefix (CPU end states must beat the full local hit)', M_SCHED,
     None, r'get_num_new_matched_tokens\(\s*request,\s*block_aligned_local\s*\)'),
    ('the scheduler keeps the local hit unless the external one is strictly longer than its partial tail',
     M_SCHED, None, r'if partial_tail and ext_tokens > partial_tail:'),
    ('update_state_after_alloc reads hash_block_size only in its alignment assert (bypassed for end-state loads)',
     M_OFF, 'SimpleCPUOffloadScheduler.update_state_after_alloc', lambda text: only_in_assert(text, 'hash_block_size')),
    ('_free_request runs the connector finish hook (end blocks queued) before freeing the blocks', M_SCHED,
     'Scheduler._free_request', r'self\._connector_finished\(request\)[\s\S]*self\._free_blocks\(request\)'),
    ('the finish hook reaches request_finished_all_groups (SupportsHMA)', M_SCHED, 'Scheduler._connector_finished',
     r'request_finished_all_groups\(\s*request, block_ids\s*\)'),
    ('request_finished_all_groups queues the finished eager stores', M_OFF,
     'SimpleCPUOffloadScheduler.request_finished_all_groups', r'self\._queue_finished_eager_store\(request, block_ids\)'),
    ('stop handling trims in place and returns (new_token_ids, stopped)', M_SCHED,
     'Scheduler._update_request_with_output', r'del new_token_ids\[num_new:\][\s\S]*return new_token_ids, stopped'),
    ('AsyncScheduler does not override the wrapped Scheduler methods', 'vllm.v1.core.sched.async_scheduler', None,
     r'\A(?![\s\S]*def (_update_request_with_output|_free_request)\b)'),
    ('the V2 rejection sampler returns num_sampled / num_rejected in SamplerOutput', M_RS,
     'RejectionSampler.__call__', r'SamplerOutput\([\s\S]*num_sampled=num_sampled[\s\S]*num_rejected=num_rejected'),
    ('the V2 runner samples drafts through rejection_sampler and hands on its counts',
     'vllm.v1.worker.gpu.model_runner', None,
     r'self\.rejection_sampler\([\s\S]*return sampler_output, sampler_output\.num_sampled, sampler_output\.num_rejected'),
    ('Mamba accepted counts follow num_sampled', 'vllm.v1.worker.gpu.model_states.mamba_hybrid', None,
     r'tl\.store\(num_accepted_ptr \+ req_state_idx, tl\.maximum\(num_sampled, 1\)\)'),
    ('the running Mamba state block of a step is column (seq_len - 1) // block (the end state block)',
     'vllm.v1.worker.mamba_utils', 'get_aligned_state_indices_multi_group_kernel',
     r'first_state_slot = tl\.maximum\(\(seq_lens - 1\) // CACHE_BLOCK_SIZE, 0\)'),
    ('Mamba checkpoints follow the attention replay boundary (no own block drop)', M_STM, 'MambaManager.__init__',
     r'self\.drop_eagle_checkpoint_block = False'),
    ('native decode-resume state is cached at free, after the connector finish hook (GPU only)', M_COORD,
     'KVCacheCoordinator.free', r'manager\.cache_blocks_on_free\(request\)'),
    ('the store completion copies every GPU hash onto the CPU block', M_OFF,
     'SimpleCPUOffloadScheduler._process_store_completion', r'cached_block_hashes_by_block\.get\('),
    ('general plugins: entry point functions are called unguarded (a raising register stops the engine)',
     'vllm.plugins', 'load_general_plugins', r'for func in plugins\.values\(\):\s*func\(\)'),
]

# What v2/v4 used that the target no longer has (each is why a 0.34 path exists).
ABSENT = [
    ('module', 'vllm.v1.sample.rejection_sampler', None),            # v4 patched the MRV1 sampler
    ('attr', M_IFACE, ('KVCacheGroup', 'kv_cache_spec')),             # v2/v4 read group.kv_cache_spec
    ('attr', M_REQ, ('Request', 'skip_reading_prefix_cache')),       # v4's CPU lookup read it
    ('attr', M_REQ, ('Request', '__slots__')),                       # v5 caches candidates in request.__dict__
    ('attr', M_COORD, ('KVCacheCoordinator', '__slots__')),          # and the group layout on the coordinator
]


class Source:
    def __init__(self, root):
        self.root = root
        self.trees = {}
        self.texts = {}

    def path(self, module):
        rel = module.split('.')[1:]
        base = os.path.join(self.root, *rel)
        for candidate in (base + '.py', os.path.join(base, '__init__.py')):
            if os.path.isfile(candidate):
                return candidate
        return None

    def tree(self, module):
        if module not in self.trees:
            path = self.path(module)
            self.texts[module] = open(path).read() if path else None
            self.trees[module] = ast.parse(self.texts[module]) if path else None
        return self.trees[module]

    def text(self, module, node=None):
        self.tree(module)
        text = self.texts[module]
        return text if node is None else ast.get_source_segment(text, node)

    def top(self, module, name):
        tree = self.tree(module)
        for node in tree.body if tree else ():
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        return None

    def top_names(self, module):
        names = set()
        for node in self.tree(module).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split('.')[0] for a in node.names)
            elif isinstance(node, ast.If):   # TYPE_CHECKING / version branches
                for sub in node.body + node.orelse:
                    if isinstance(sub, (ast.ClassDef, ast.FunctionDef)):
                        names.add(sub.name)
                    elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                        names.update((a.asname or a.name).split('.')[0] for a in sub.names)
        return names

    def resolve(self, module, qualname):
        """The ast node of `Class.method` (searching vLLM base classes) or of a top-level def."""
        parts = qualname.split('.')
        node = self.top(module, parts[0])
        if len(parts) == 1:
            return node
        return self.method(module, node, parts[1]) if isinstance(node, ast.ClassDef) else None

    def bases(self, module, cls):
        """(module, ClassDef) of each base class defined in the vLLM tree."""
        out = []
        imports = {}
        for node in self.tree(module).body:
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                for alias in node.names:
                    imports[alias.asname or alias.name] = node.module
        for base in cls.bases:
            if not isinstance(base, ast.Name):
                continue
            if self.top(module, base.id) is not None:
                out.append((module, self.top(module, base.id)))
            elif imports.get(base.id, '').startswith('vllm') and self.tree(imports[base.id]) is not None:
                target = self.top(imports[base.id], base.id)
                if target is not None:
                    out.append((imports[base.id], target))
        return out

    def method(self, module, cls, name):
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                return node
        for base_module, base in self.bases(module, cls):
            found = self.method(base_module, base, name)
            if found is not None:
                return found
        return None

    def attrs(self, module, cls):
        names = set()
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
                for sub in ast.walk(node):
                    targets = sub.targets if isinstance(sub, ast.Assign) else (
                        [sub.target] if isinstance(sub, (ast.AnnAssign, ast.AugAssign)) else [])
                    for t in targets:
                        for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
                            if (isinstance(el, ast.Attribute) and isinstance(el.value, ast.Name)
                                    and el.value.id == 'self'):
                                names.add(el.attr)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
        for base_module, base in self.bases(module, cls):
            names |= self.attrs(base_module, base)
        return names


def only_in_assert(text, attr):
    """Whether every `self.<attr>` in the function source sits inside an assert statement (and there is one)."""
    tree = ast.parse(textwrap.dedent(text))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == attr
            and isinstance(n.value, ast.Name) and n.value.id == 'self']
    in_asserts = {id(n) for a in ast.walk(tree) if isinstance(a, ast.Assert) for n in ast.walk(a)}
    return bool(uses) and all(id(n) in in_asserts for n in uses)


def params(fn):
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if a.vararg:
        names.append('*' + a.vararg.arg)
    names += [x.arg for x in a.kwonlyargs]
    if a.kwarg:
        names.append('**' + a.kwarg.arg)
    return names


def main():
    if len(sys.argv) != 2 or not os.path.isfile(os.path.join(sys.argv[1], 'v1', 'request.py')):
        sys.exit(__doc__)
    src = Source(os.path.abspath(sys.argv[1]))
    version = re.search(r"__version__ = version = '([^']+)'", open(os.path.join(src.root, '_version.py')).read())
    print('vLLM sources:', src.root, version[1] if version else '(no _version.py)')
    failures = []

    def result(ok, message):
        print(('ok:   ' if ok else 'FAIL: ') + message)
        if not ok:
            failures.append(message)

    # 1. imports of the plugin modules
    for path in PLUGIN_MODULES:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] == 'vllm':
                if src.tree(node.module) is None:
                    result(False, f'{os.path.basename(os.path.dirname(path))}: module {node.module} missing')
                    continue
                for alias in node.names:   # a top-level name of the module, or one of its submodules
                    result(alias.name in src.top_names(node.module) or src.path(f'{node.module}.{alias.name}'),
                           f'{os.path.basename(os.path.dirname(path))}: from {node.module} import {alias.name}')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] == 'vllm':
                        result(src.tree(alias.name) is not None, f'import {alias.name}')

    # 2. curated names, attributes, signatures
    for module, qualname, kind, detail in REQUIRED:
        if src.tree(module) is None:
            result(False, f'{module} missing')
            continue
        if kind == 'name':
            result(qualname in src.top_names(module), f'{module}.{qualname} defined')
            continue
        if kind == 'attrs':
            cls = src.top(module, qualname)
            if not isinstance(cls, ast.ClassDef):
                result(False, f'class {module}.{qualname} missing')
                continue
            have = src.attrs(module, cls)
            missing = [a for a in detail if a not in have]
            result(not missing, f'{qualname} has {", ".join(detail)}' + (f' (missing {missing})' if missing else ''))
            continue
        fn = src.resolve(module, qualname)
        if fn is None:
            result(False, f'{module}.{qualname} missing')
            continue
        got = params(fn)
        if kind == 'wrap':
            result(got == detail, f'{qualname}{tuple(got)} == wrapped signature {tuple(detail)}')
        else:
            it = iter(got)
            result(all(p in it for p in detail), f'{qualname}{tuple(got)} takes {tuple(detail)}')

    # 3. behaviour the port depends on, and what v2/v4 relied on that is gone
    for description, module, qualname, pattern in BEHAVIOUR:
        if src.tree(module) is None:
            result(False, f'{description}: {module} missing')
            continue
        node = src.resolve(module, qualname) if qualname else None
        if qualname and node is None:
            result(False, f'{description}: {module}.{qualname} missing')
            continue
        text = src.text(module, node)
        result(pattern(text) if callable(pattern) else re.search(pattern, text) is not None, description)
    for kind, module, detail in ABSENT:
        if kind == 'module':
            result(src.path(module) is None, f'absent: module {module}')
        else:
            cls = src.top(module, detail[0])
            result(cls is not None and detail[1] not in src.attrs(module, cls),
                   f'absent: {detail[0]}.{detail[1]}')

    # Information: lookahead hashes are on for SimpleCPUOffloadConnector unless it opts out.
    connector = 'vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector'
    overrides = 'supports_lookahead_block_hashes' in (src.text(connector) or '')
    print(f'info: SimpleCPUOffloadConnector {"overrides" if overrides else "inherits (True)"} '
          'supports_lookahead_block_hashes: with MTP the engine hashes with num_hash_lookahead_tokens = 1')
    print(f'{len(failures)} failure(s)' if failures else 'all static checks passed')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
